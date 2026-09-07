//! `DiscordApprovalBroker` — long-lived serenity client + post-only approval
//! card surface. All approval state persists in sqlite (handled by
//! `ApprovalActionHandler`); this broker holds no per-action memory.

use std::sync::Arc;

use async_trait::async_trait;
use serenity::all::{ChannelId, GatewayIntents, MessageId, UserId};
use tokio::sync::Notify;
use tracing::{error, info, warn};

use augmentagent_store::{Email, Store};

use crate::event_handler::Handler;
use crate::layout::{approval_message, flag_notice_message, scheduled_notice_message};
use crate::loops::LoopCommandParser;
use crate::{ApprovalActionHandler, ApprovalBroker, ApprovalError, JournalOps, QueryHandler};

#[derive(Clone)]
pub struct DiscordConfig {
    pub bot_token: String,
    /// Channel where approval cards get posted.
    pub channel_id: u64,
    /// Channel where wiki queries are accepted. `None` disables query replies
    /// in server channels (DMs still work if `allowed_user_id` is set).
    pub query_channel_id: Option<u64>,
    /// User ID allowed to send wiki queries OR click approval buttons. `None`
    /// = no allowlist (any user in the channel can drive).
    pub allowed_user_id: Option<u64>,
    /// Plugs wiki querying into Discord message handling. `None` disables it.
    pub query_handler: Option<Arc<dyn QueryHandler>>,
    /// Handles Approve / Revise / Skip clicks. `None` silently ignores.
    pub action_handler: Option<Arc<dyn ApprovalActionHandler>>,
    /// Store handle so the event handler can persist (draft, feedback, revised)
    /// triples to `draft_revisions` after a Revise (#37). `None` disables
    /// persistence — the broker still works, the data just isn't captured.
    pub store: Option<Arc<Store>>,
    /// LLM-backed parser for `/loop` create-text. `None` disables the create
    /// path (list / stop / help still work). The CLI wires this with a Claude
    /// reasoner; tests can omit it.
    pub loop_parser: Option<Arc<dyn LoopCommandParser>>,
    /// Wiki root for outbound answer attachments (#440). `ATTACH:` markers in
    /// wiki-ask answers resolve against (and are confined to) this directory;
    /// `None` refuses every marker with a visible note in the reply.
    pub wiki_root: Option<std::path::PathBuf>,
    /// Bridge into the cli's ShadowNote journaling flow (#428). `None`
    /// leaves `!journal` answering with a not-configured notice.
    pub journal_ops: Option<Arc<dyn JournalOps>>,
}

pub(crate) struct BrokerState {
    ready: Arc<Notify>,
    ready_flag: std::sync::atomic::AtomicBool,
    pub(crate) query_channel_id: Option<ChannelId>,
    pub(crate) allowed_user_id: Option<UserId>,
    pub(crate) query_handler: Option<Arc<dyn QueryHandler>>,
    pub(crate) action_handler: Option<Arc<dyn ApprovalActionHandler>>,
    pub(crate) approval_channel_id: ChannelId,
    /// Store handle for persisting Revise triples to `draft_revisions` (#37).
    pub(crate) store: Option<Arc<Store>>,
    /// LLM parser for `/loop` create text. `None` disables the create path.
    pub(crate) loop_parser: Option<Arc<dyn LoopCommandParser>>,
    /// Wiki root that `ATTACH:` answer markers are confined to (#440).
    pub(crate) wiki_root: Option<std::path::PathBuf>,
    /// ShadowNote journaling bridge for `!journal` (#428).
    pub(crate) journal_ops: Option<Arc<dyn JournalOps>>,
    /// Populated once, from the first `Ready` event. Used to distinguish the
    /// bot's own messages from the user's when building conversation context.
    pub(crate) bot_user_id: std::sync::OnceLock<UserId>,
}

impl BrokerState {
    #[allow(clippy::too_many_arguments)]
    fn new(
        approval_channel_id: ChannelId,
        query_channel_id: Option<ChannelId>,
        allowed_user_id: Option<UserId>,
        query_handler: Option<Arc<dyn QueryHandler>>,
        action_handler: Option<Arc<dyn ApprovalActionHandler>>,
        store: Option<Arc<Store>>,
        loop_parser: Option<Arc<dyn LoopCommandParser>>,
        wiki_root: Option<std::path::PathBuf>,
        journal_ops: Option<Arc<dyn JournalOps>>,
    ) -> Self {
        Self {
            ready: Arc::new(Notify::new()),
            ready_flag: std::sync::atomic::AtomicBool::new(false),
            query_channel_id,
            allowed_user_id,
            query_handler,
            action_handler,
            approval_channel_id,
            store,
            loop_parser,
            wiki_root,
            journal_ops,
            bot_user_id: std::sync::OnceLock::new(),
        }
    }

    pub(crate) fn mark_ready(&self) {
        self.ready_flag
            .store(true, std::sync::atomic::Ordering::SeqCst);
        self.ready.notify_waiters();
    }

    async fn await_ready(&self) {
        if self.ready_flag.load(std::sync::atomic::Ordering::SeqCst) {
            return;
        }
        self.ready.notified().await;
    }
}

pub struct DiscordApprovalBroker {
    http: Arc<serenity::http::Http>,
    channel_id: ChannelId,
}

impl DiscordApprovalBroker {
    /// Start the serenity client in a background task. Blocks the current task
    /// until the gateway is `Ready`, after which `post_approval` calls may be
    /// issued.
    pub async fn start(config: DiscordConfig) -> Result<Self, ApprovalError> {
        let approval_channel = ChannelId::new(config.channel_id);
        let state = Arc::new(BrokerState::new(
            approval_channel,
            config.query_channel_id.map(ChannelId::new),
            config.allowed_user_id.map(UserId::new),
            config.query_handler.clone(),
            config.action_handler.clone(),
            config.store.clone(),
            config.loop_parser.clone(),
            config.wiki_root.clone(),
            config.journal_ops.clone(),
        ));

        if state.allowed_user_id.is_none() {
            warn!(
                "DISCORD_ALLOWED_USER_ID is not set: the Discord bot has NO owner allowlist and is \
                 running fail-closed (#303) — all query DMs, button clicks, and modal submits will \
                 be refused until it is configured. Set DISCORD_ALLOWED_USER_ID to your Discord \
                 user ID to authorize the owner."
            );
        }

        let mut intents = GatewayIntents::GUILDS | GatewayIntents::GUILD_MESSAGES;
        if config.query_handler.is_some() {
            intents |= GatewayIntents::MESSAGE_CONTENT | GatewayIntents::DIRECT_MESSAGES;
        }

        let handler = Handler {
            state: Arc::clone(&state),
        };

        let mut client = serenity::Client::builder(&config.bot_token, intents)
            .event_handler(handler)
            .await?;

        let http = Arc::clone(&client.http);

        tokio::spawn(async move {
            if let Err(e) = client.start().await {
                error!("discord client exited: {e}");
            }
        });

        state.await_ready().await;
        info!(
            query_enabled = config.query_handler.is_some(),
            action_enabled = config.action_handler.is_some(),
            "discord approval broker online"
        );

        // `state` is moved into the Handler and dropped from the broker
        // after client.start. That's intentional: the broker is post-only.
        drop(state);

        Ok(Self {
            http,
            channel_id: approval_channel,
        })
    }
}

#[async_trait]
impl ApprovalBroker for DiscordApprovalBroker {
    async fn post_approval(
        &self,
        action_id: &str,
        email: &Email,
        draft: &str,
    ) -> Result<(), ApprovalError> {
        // First post for this action — redraft count is always 0 here. Revise
        // / quick-refine re-renders go through the event handler, which reads
        // the persisted count.
        let message = approval_message(action_id, email, draft, 0);
        self.channel_id
            .send_message(&*self.http, message)
            .await
            .map_err(|e| ApprovalError::Discord(e.to_string()))?;
        Ok(())
    }

    async fn post_flag_notice(
        &self,
        email: &Email,
        reason: &str,
    ) -> Result<(), ApprovalError> {
        let message = flag_notice_message(email, reason);
        self.channel_id
            .send_message(&*self.http, message)
            .await
            .map_err(|e| ApprovalError::Discord(e.to_string()))?;
        Ok(())
    }

    async fn post_scheduled_notice(
        &self,
        action_id: &str,
        email: &Email,
        sends_at_local: &str,
        sends_at_ms: i64,
        to_display: &str,
    ) -> Result<Option<(u64, u64)>, ApprovalError> {
        // Same approval channel as the cards — the notice IS the card's
        // replacement in the carousel (#501). The returned ids are persisted
        // by the caller so the engine can retire the notice at fire time.
        let message = scheduled_notice_message(
            action_id,
            email,
            sends_at_local,
            sends_at_ms,
            to_display,
        );
        let sent = self
            .channel_id
            .send_message(&*self.http, message)
            .await
            .map_err(|e| ApprovalError::Discord(e.to_string()))?;
        Ok(Some((sent.channel_id.get(), sent.id.get())))
    }

    async fn post_approval_card(
        &self,
        action_id: &str,
        email: &Email,
        draft: &str,
        redraft_count: u32,
    ) -> Result<Option<(u64, u64)>, ApprovalError> {
        // Back-to-queue repost (#501): honor the persisted redraft count so
        // a refined-to-cap card doesn't get its quick-refine row back, and
        // return the ids so the caller can roll the post back on a lost CAS.
        let message = approval_message(action_id, email, draft, i64::from(redraft_count));
        let sent = self
            .channel_id
            .send_message(&*self.http, message)
            .await
            .map_err(|e| ApprovalError::Discord(e.to_string()))?;
        Ok(Some((sent.channel_id.get(), sent.id.get())))
    }

    async fn delete_message(
        &self,
        channel_id: u64,
        message_id: u64,
    ) -> Result<(), ApprovalError> {
        ChannelId::new(channel_id)
            .delete_message(&*self.http, MessageId::new(message_id))
            .await
            .map_err(|e| ApprovalError::Discord(e.to_string()))
    }
}
