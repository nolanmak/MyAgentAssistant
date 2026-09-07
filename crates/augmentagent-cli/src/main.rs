//! `augmentagent` binary.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use augmentagent_approval_discord::{
    approval_message, ApprovalActionHandler, ApprovalActionOutcome,
    ApprovalBroker, DiscordApprovalBroker, DiscordConfig, LoopPoster,
    LoopRunner, LoopScheduler, NoopBroker, QueryHandler,
};
use augmentagent_channel_core::reasoner::{ask_opts, digest_opts, draft_opts};
use augmentagent_channel_core::{build_reasoner, FallbackReasoner, Reasoner};
use augmentagent_channel_email::gmail::{
    normalize_subject, Attachment, ComposioClient, GmailApi, ThreadSubject,
};
use augmentagent_channel_email::sigextract::{
    detect_signature_block, is_human_sender, signature_patch, strip_quoted_reply,
    SignatureExtractor,
};
use augmentagent_channel_email::{GmailChannel, GmailChannelConfig, OutboundObserver};
use augmentagent_channel_linkedin::{
    build_normshares_body, default_auth_path, is_linkedin_email, ConnectionRequestEngagement,
    FriendFeedEngagement, InvitationsTrigger, LinkedInApi, LinkedInAuth,
    LinkedInChannel, LinkedInChannelConfig, LinkedInFeedEngagement, LinkedInFriendFeedSource,
    OwnPostCommentEngagement, OwnPostsCommentTrigger, PostDraft, VoyagerClient,
    Visibility, ACCOUNT_PREFIX, DEFAULT_FEED_POLL_SECS, DEFAULT_FRIEND_FEED_POLL_SECS,
    DEFAULT_INVITATION_POLL_SECS, DEFAULT_MAX_ENGAGEMENTS_PER_DAY,
    DEFAULT_MAX_FRIEND_POSTS_PER_TICK, DEFAULT_MAX_REPLIES_PER_DAY,
    DEFAULT_OWN_POST_POLL_SECS, DEFAULT_POLL_SECS,
};
use augmentagent_channel_twitter::{
    default_auth_path as twitter_default_auth_path, validate_session as twitter_validate_session,
    CreateTweetClient, TwitterApi, TwitterAuth, TwitterClient, TwitterDmSource, TwitterFeedTrigger,
    ValidateOptions as TwitterValidateOptions,
};
use augmentagent_channel_linkedin::connections::{
    ConnectionSyncer, SyncMode, VoyagerConnectionsClient,
};
use augmentagent_channel_contacts::{
    CardDavSource, ContactsSource, ContactsSyncer, GooglePeopleSource,
};
use augmentagent_store::{ActionStatus, Store, TriageResult};
use async_trait::async_trait;

mod channel_router;
mod code_mode;
mod doc_cmd;
mod doctor;
mod env_cfg;
mod gmail_attach;
mod installers;
mod logs;
mod loop_cmd;
mod loops;
mod research;
mod self_improve;
mod service;
mod setup;
mod status;

#[derive(Parser)]
#[command(name = "augmentagent", version, about = "AugmentAgent Rust daemon")]
struct Cli {
    /// Path to sqlite db. Defaults to `AUGMENTAGENT_DB` env or `./data.db`.
    #[arg(long)]
    db: Option<PathBuf>,

    /// Path to skill dir. Defaults to `./skills/email-triage`.
    #[arg(long, default_value = "skills/email-triage")]
    skill_dir: PathBuf,

    /// Wiki root directory. When set, enables the three-call pipeline
    /// (triage → draft with wiki read → async ingest with wiki write).
    #[arg(long)]
    wiki_dir: Option<PathBuf>,

    /// Path to the wiki maintenance schema (committed to git).
    /// Defaults to `./schema/wiki-skill.md` when `--wiki-dir` is set.
    #[arg(long)]
    wiki_schema: Option<PathBuf>,

    /// Claude model override for drafting (`claude --model …`).
    #[arg(long)]
    model: Option<String>,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run one poll cycle and exit.
    PollOnce {
        /// Dry-run (default): writes `dry_run` actions, no drafts, no sends.
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        dry_run: bool,
    },
    /// Run the poll loop as a daemon.
    Serve {
        #[arg(long, default_value_t = 120)]
        interval_secs: u64,
        /// Dry-run (default true). Flip with `--dry-run false` after Phase 2 cutover.
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        dry_run: bool,
        /// Run without the Gmail/email channel (multi-tenant non-email agent).
        /// Default false ⇒ byte-identical to the single-tenant prod path.
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        no_email: bool,
    },
    /// List active gmail accounts from the shared db.
    AccountsList,
    /// Resolve + persist each connected Gmail's real address via Composio
    /// `GMAIL_GET_PROFILE`, so the dashboard + invoice entity picker show
    /// who's who instead of opaque IDs. Safe to re-run.
    AccountsBackfillEmails,
    /// #103 — self-improvement loop: pick an `agent-fixable` GitHub issue,
    /// fix it on an isolated worktree/branch (never main), run the
    /// verification gate, and open a DRAFT PR. Never auto-merges.
    ///
    /// #117 — `--multi-repo true` switches to the allowlisted multi-repo
    /// path: clone each enabled `agent_repos` entry into an isolated
    /// workspace, run its per-repo gate, and queue a *prompted* draft PR
    /// (Discord + dashboard approval) instead of opening one directly.
    /// `--approve/--reject <run-id>` resolves a queued gate row; the next
    /// `--approve-open true` pass opens the draft PR for approved rows.
    SelfImprove {
        /// Dry-run (default true): run the gate but stop before opening a PR.
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        dry_run: bool,
        /// #117: run the allowlisted multi-repo prompted-PR path instead of
        /// the single-repo issue-pickup path.
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        multi_repo: bool,
        /// #117: approve a queued gate row by id (flips it to `approved`).
        #[arg(long)]
        approve: Option<String>,
        /// #117: reject a queued gate row by id.
        #[arg(long)]
        reject: Option<String>,
        /// #117: open draft PRs for every already-`approved` gate row.
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        approve_open: bool,
    },
    /// #117 — manage the multi-repo agent-coding allowlist from the CLI
    /// (parity with the dashboard /repos admin view).
    Repos {
        #[command(subcommand)]
        op: ReposOp,
    },
    /// Wiki maintenance.
    Wiki {
        #[command(subcommand)]
        op: WikiOp,
    },
    /// FlyOnTheWall meeting transcripts (#915).
    Transcripts {
        #[command(subcommand)]
        op: TranscriptsOp,
    },
    /// Compose a morning digest of recent inbox activity.
    Digest {
        /// Window size in hours. Defaults to 24.
        #[arg(long, default_value_t = 24)]
        since: u32,
        /// Also post to DISCORD_CHANNEL_ID (uses DISCORD_BOT_TOKEN). Otherwise stdout only.
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        post_discord: bool,
    },
    /// Daily automated research: pull recent arXiv AI/agent papers + the
    /// latest from the leapmodel repo, compare them against our agent
    /// process via a swappable LLM driver (RESEARCH_LLM_CMD), file GitHub
    /// issues for the top gaps, and post a digest to Discord.
    Research {
        /// Look-back window in hours for arXiv submissions / leapmodel commits.
        #[arg(long, default_value_t = 24)]
        since_hours: u32,
        /// Also post the digest to DISCORD_CHANNEL_ID. Otherwise stdout only.
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        post_discord: bool,
        /// Dry-run (default true): print the issues that would be filed and
        /// the digest, but create no GitHub issues.
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        dry_run: bool,
        /// Cap on GitHub issues created this run (flag overrides
        /// RESEARCH_MAX_ISSUES env, which overrides the default of 3).
        #[arg(long)]
        max_issues: Option<u32>,
    },
    /// Gmail inbox tooling for Claude to invoke via Bash when the wiki
    /// can't answer a question.
    Gmail {
        #[command(subcommand)]
        op: GmailOp,
    },
    /// Document text extraction (#939): pdftotext/pandoc first, then Mistral
    /// OCR for scanned (image-only) PDFs when MISTRAL_API_KEY is set. Same
    /// pipeline the Discord attachment handler and `gmail get-attachment` use.
    Doc {
        #[command(subcommand)]
        op: DocOp,
    },
    /// Resume ingestion — one-shot seed of the wiki from the user's CV.
    Resume {
        #[command(subcommand)]
        op: ResumeOp,
    },
    /// LinkedIn DM channel: harvest cookies, poll the inbox, search threads.
    Linkedin {
        #[command(subcommand)]
        op: LinkedinOp,
    },
    /// X / Twitter channel: harvest session, post tweets/replies, poll the
    /// close-friend feed + DM inbox.
    Twitter {
        #[command(subcommand)]
        op: TwitterOp,
    },
    /// Discord channel: user-token REST client. Reads personal DMs + watched
    /// guild channels, routes through subscriptions (priority/digest/store_only).
    Discord {
        #[command(subcommand)]
        op: DiscordOp,
    },
    /// Slack channel: Composio-managed OAuth client. Reads watched DMs +
    /// channels via subscriptions (priority/digest/store_only).
    Slack {
        #[command(subcommand)]
        op: SlackOp,
    },
    /// Telegram Bot API channel (#74). Long-poll getUpdates, dispatch through
    /// channel_subscriptions. All ops are stubs in foundation/swarm-v1; impls
    /// land in the telegram-bot feature PR.
    TelegramBot {
        #[command(subcommand)]
        op: TelegramBotOp,
    },
    /// WhatsApp channel via whatsmeow Go sidecar (#74). All ops are stubs in
    /// foundation/swarm-v1; impls land in the whatsapp feature PR.
    Whatsapp {
        #[command(subcommand)]
        op: WhatsappOp,
    },
    /// Google Calendar -> wiki Meeting log ingestion (archived AugmentAgent#82). All ops are
    /// stubs in foundation/swarm-v1; impls land in the calendar feature PR.
    Calendar {
        #[command(subcommand)]
        op: CalendarOp,
    },
    /// ShadowNote journal → wiki ingest (#427). Opt-in: inert unless the
    /// SHADOWNOTE_* config is present in keyring/env (see epic #425).
    Journal {
        #[command(subcommand)]
        op: JournalOp,
    },
    /// iMessage history bundle → KB (#882). Opt-in: inert unless
    /// AUGMENTAGENT_IMESSAGE_REPO_DIR points at the bundle repo.
    Imessage {
        #[command(subcommand)]
        op: ImessageOp,
    },
    /// Person-page maintenance (identity-merge groundwork).
    Person {
        #[command(subcommand)]
        op: PersonOp,
    },
    /// Voice memo capture: drop-folder watcher + Whisper transcription ->
    /// wiki ingest. All ops stubs in foundation/swarm-v1.
    Voice {
        #[command(subcommand)]
        op: VoiceOp,
    },
    /// GitHub channel: notification + review-request triage. All ops stubs
    /// in foundation/swarm-v1.
    Github {
        #[command(subcommand)]
        op: GithubOp,
    },
    /// Deftform channel: operator-facing onboarding (#160). The channel
    /// itself is still inert (spike scaffold #116) until both
    /// `AUGMENTAGENT_DEFT_ENABLED=1` and a workspace token are persisted.
    Deft {
        #[command(subcommand)]
        op: DeftOp,
    },
    /// Reddit DM/inbox channel OAuth bootstrap (#48).
    Reddit {
        #[command(subcommand)]
        op: RedditOp,
    },
    /// SocialAPI.ai-managed social accounts (#245): list/disable the local
    /// account registry and drive the proxied OAuth connect flow.
    Socialapi {
        #[command(subcommand)]
        op: SocialapiOp,
    },
    /// Meetup.com group events → Discord digest (multi-tenant; no email).
    Meetup {
        #[command(subcommand)]
        op: MeetupOp,
    },
    /// Google Drive (via Composio) change feed → Discord (multi-tenant).
    Gdrive {
        #[command(subcommand)]
        op: GdriveOp,
    },
    /// Contacts → phone+address identity index (#62). Google People (via
    /// the Composio Google grant) and/or generic CardDAV.
    Contacts {
        #[command(subcommand)]
        op: ContactsOp,
    },
    /// Cross-platform compose-once content adapter (#53). One source draft
    /// fans out into per-platform variants. All ops stubs in
    /// foundation/swarm-v1.
    Compose {
        #[command(subcommand)]
        op: ComposeOp,
    },
    /// Proactive CRM scanner (#81): stale-contact / stale-commitment /
    /// event-reminder rules over wiki + sqlite. All ops stubs in
    /// foundation/swarm-v1.
    Proactive {
        #[command(subcommand)]
        op: ProactiveOp,
    },
    /// Headless-browser client (CDP-driven Chromium) for channels that fall
    /// back to DOM automation. All ops stubs in foundation/swarm-v1.
    Browser {
        #[command(subcommand)]
        op: BrowserOp,
    },
    /// Render a vertical (1080x1920) branded short-card mp4 from JSON props
    /// via the Remotion renderer sidecar (Phase 0 — see docs/REMOTION.md).
    /// Manually triggerable; no scheduler/governor/posting wiring yet.
    Render {
        /// inputProps as a JSON string, or `@path` to read JSON from a file.
        /// Shape: {title, body, accent?, durationSec}.
        #[arg(long)]
        props: String,
        /// Output mp4 path.
        #[arg(long)]
        out: PathBuf,
        /// Video codec.
        #[arg(long, default_value = "h264")]
        codec: String,
    },
    /// Per-recipient tone-mirroring (#73): backfill sent history, refresh
    /// per-scope voice profiles, and sweep stale rows on a schedule.
    Tone {
        #[command(subcommand)]
        op: ToneOp,
    },
    /// #64 — mine email signature blocks for role/title/company/phone and
    /// fill-blanks them into the wiki. Idempotent (safe to re-run); pulls
    /// emails first seen on/after `--since`. Dry-run JSON by default.
    BackfillSignatures {
        /// Lower bound (`YYYY-MM-DD`). Default: 180 days ago.
        #[arg(long)]
        since: Option<String>,
        /// Max emails to scan.
        #[arg(long, default_value_t = 2000)]
        limit: i64,
        /// Min per-field confidence to auto-fill the wiki; lower-confidence
        /// fields go to the daily Discord digest instead.
        #[arg(long, default_value_t = 0.7)]
        min_confidence: f64,
        /// Write wiki pages (default: dry-run JSON only).
        #[arg(long)]
        apply: bool,
    },
    /// Draft-quality eval tooling backed by `draft_revisions` (#37).
    Drafts {
        #[command(subcommand)]
        op: DraftsOp,
    },
    /// Draft approval-queue hygiene (#99): inspect + bulk-clear the pending
    /// backlog. Destructive ops are audit-logged to stdout + tracing.
    Approvals {
        #[command(subcommand)]
        op: ApprovalsOp,
    },
    /// RateGovernor (#83) audit + dump tooling. Reads `rate_events`,
    /// `rate_halts`, and `rate_warmup`. Channels write to these tables
    /// when they adopt the governor (sibling/feature PRs).
    Ratelimit {
        #[command(subcommand)]
        op: RatelimitOp,
    },
    /// #58 — engagement-automation scheduled posts. Queue / list / cancel an
    /// outbound post; the serve-tick fire loop previews it at T-30min and
    /// publishes it at T-0 via the per-platform poster.
    SchedulePost {
        #[command(subcommand)]
        op: SchedulePostOp,
    },
    /// #58.2/.3 — populate the durable inputs the engagement pollers consume:
    /// register one of your own posts to watch for comments, or add/remove a
    /// friend on the engagement watchlist.
    Engagement {
        #[command(subcommand)]
        op: EngagementOp,
    },

    // === setup+maintenance subcommands (alphabetical) ===
    /// Issue #2 — cross-channel router. `augmentagent channel <name> <op>` is
    /// a thin alias for the per-channel `augmentagent <name> <op>` form so
    /// the /setup skill (and the dashboard) can speak one shape for every
    /// channel. Pass-through trailing args (e.g. `--json`, `--dry-run`,
    /// `--account work@example.com`) are forwarded verbatim.
    Channel {
        /// Channel to dispatch to (e.g. `gmail`, `slack`, `telegram-bot`).
        #[arg(value_enum)]
        name: channel_router::ChannelName,
        /// Op to run. `arm` / `disarm` land in issue #7.
        #[arg(value_enum)]
        op: channel_router::ChannelOp,
        /// Pass-through flags forwarded verbatim to the underlying
        /// per-channel command (e.g. `--json`, `--dry-run false`).
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// #50 — Code-Mode debug surface. Wraps the Rust dispatcher + Deno
    /// sidecar so I6's acceptance test (and humans poking the pipeline) can
    /// drive a fixture program end-to-end without the polling daemon or the
    /// LLM. The runtime + manifest live in `augmentagent-channel-core`.
    CodeMode {
        #[command(subcommand)]
        op: code_mode::CodeModeOp,
    },
    /// #11 — read-only diagnostic checks. Composes the `status` aggregator
    /// (#1) with additional probes (sqlite integrity, keyring reachability,
    /// tool binaries on `$PATH`, build freshness, `.env` presence). Emits
    /// severity-tagged findings; exit 0 unless any check is `error`. `--fix`
    /// lands as a follow-up issue — doctor stays strictly read-only.
    Doctor {
        /// Force JSON (`--json`) or human table. Default: auto — JSON when
        /// stdout is piped, table on a tty.
        #[arg(long, num_args = 0..=1, default_missing_value = "true")]
        json: Option<bool>,
        /// Add slower probes (Composio whoami ping; Cerebras model-catalog
        /// check; per-channel validate summaries sourced from `status`).
        #[arg(long, default_value_t = false)]
        deep: bool,
    },
    /// #655/#667 — one live round-trip through the provider fallback chain.
    /// Builds the production reasoner (AUGMENTAGENT_REASONER_CHAIN +
    /// eligibility checks), sends a trivial text-only prompt, and prints the
    /// configured chain, active cooldown latches, and the response. This is
    /// the reproducible e2e probe for failover changes: point CLAUDE_CLI at
    /// a stub that refuses on quota and watch the call get served by the
    /// next provider.
    ReasonerSelftest {
        /// Prompt to send (default asks for a one-word reply).
        #[arg(long, default_value = "Reply with exactly one word: PONG")]
        prompt: String,
    },
    /// Issue #12 — read/write the sqlite `config` table so the `/setup`
    /// skill never has to parse or rewrite `.env`. Reads merge config over
    /// `process.env` (config wins) — same precedence as the dashboard.
    /// Secrets are masked in `list`; `get` prints raw values.
    Env {
        /// Op to run.
        #[command(subcommand)]
        op: env_cfg::EnvOp,
        /// Emit JSON. Applies to `list` and `get`; `set`/`unset` always
        /// emit JSON receipts.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// #6 — install a component by shelling out to the matching
    /// `scripts/install-*.sh` (or the systemd unit-template copy for
    /// `browser-sidecar`). Idempotent — re-running is safe.
    ///
    /// Each component subcommand accepts `--rebuild` (cargo + npm build
    /// before the install script) and `--json` (suppress live stream, emit
    /// a single JSON summary).
    Install {
        /// Which component to install.
        #[command(subcommand)]
        component: installers::InstallComponent,
    },
    /// Tail or dump the daemon's systemd-journal logs (wraps
    /// `journalctl --user -u <unit>`). Linux-only.
    ///
    /// `--unit` accepts short aliases: `daemon` → `augmentagent.service`,
    /// `dashboard` → `augmentagent-dashboard.service`, and any bare name
    /// `X` expands to `augmentagent-X.service`. Names already containing
    /// a `.` (e.g. `custom.service`) pass through unchanged.
    Logs {
        /// Unit to tail. Short aliases (`daemon`, `dashboard`, `web`, …)
        /// are expanded — see the command help for the full mapping.
        #[arg(long, default_value = "augmentagent.service")]
        unit: String,
        /// Stream new entries as they arrive (`journalctl -f`).
        #[arg(long, short = 'f', default_value_t = false)]
        follow: bool,
        /// How many recent entries to show (`journalctl -n <lines>`).
        #[arg(long, default_value_t = 200)]
        lines: u32,
        /// Only show entries on/after this time. Passed straight to
        /// `journalctl --since` (e.g. `"2026-05-20"`, `"1 hour ago"`).
        #[arg(long)]
        since: Option<String>,
        /// Emit one JSON object per line (`journalctl -o json`).
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// #212 — list + stop user-scheduled loops in the sqlite `user_loops`
    /// table. This is the surface for the Discord `/loop` scheduler (#104),
    /// the common case when a user asks the bot "kill the hello world
    /// loop": the row is in sqlite, the daemon ticks it, killing claude
    /// PIDs does nothing. Distinct from `loops` (plural, #175) which is
    /// OS-level process control for orphan Claude Code sessions.
    Loop {
        #[command(subcommand)]
        op: loop_cmd::LoopOp,
    },
    /// #175 — list + signal `claude` CLI processes on this host. Addresses
    /// orphan Claude Code sessions whose `/loop` skill kept firing after the
    /// session closed (the cross-session-state half of #174). `loops list`
    /// shows every claude PID; `loops stop <PID>` sends SIGTERM (or SIGKILL
    /// with `--force`). For Discord-scheduled `/loop` rows, use `loop`
    /// (singular) instead — those run inside the daemon, no claude PID
    /// involved.
    Loops {
        #[command(subcommand)]
        op: loops::LoopsOp,
    },
    /// Thin wrapper over `systemctl --user` for the augmentagent unit family.
    /// Lets the `/setup` skill (and humans) say `service restart --unit
    /// dashboard` instead of memorising unit names. Linux-only by design.
    Service {
        #[command(subcommand)]
        op: service::ServiceOp,
        /// Unit alias: `daemon` (default) | `dashboard` | `updater` | `digest`
        /// | `tone-refresh` | `browser-sidecar` | `tenant:<name>` | `all`, or
        /// a full systemd unit name (e.g. `augmentagent-digest.timer`).
        #[arg(long, default_value = "daemon")]
        unit: String,
        /// Emit machine-readable JSON (status op only).
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Operator-onboarding helpers. Issue #8 lands `setup harvest <channel>`,
    /// a cookie-harvest field-schema emitter that the `/setup` skill uses to
    /// drive Discord/Twitter/LinkedIn/Instagram credential capture via
    /// `AskUserQuestion` instead of an interactive `read` loop. Future ops
    /// (Oauth from #10) slot in alphabetically under `setup`.
    Setup {
        #[command(subcommand)]
        op: setup::SetupOp,
    },
    /// #1 — one-document health aggregator: daemon, dashboard, updater,
    /// core keys, per-channel configured/armed, queue depth. Source of truth
    /// for the `/setup` skill and ongoing maintenance.
    ///
    /// Exit codes: 0 ok, 10 degraded/needs-setup, 20 daemon-down,
    /// 30 dashboard-down, 40 config-invalid.
    Status {
        /// Force JSON (`--json true`) or human table (`--json false`).
        /// Default: auto — JSON when stdout is piped, table on a tty.
        #[arg(long, num_args = 0..=1, default_missing_value = "true")]
        json: Option<bool>,
        /// Narrow `channels` to just one entry (e.g. `--channel gmail`).
        #[arg(long)]
        channel: Option<String>,
        /// Placeholder for a future probe-cache; currently a no-op so the
        /// `/setup` skill can adopt the flag from day one.
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        refresh: bool,
    },
    /// #6 — uninstall a component by shelling out to the matching
    /// `scripts/uninstall-*.sh` (or undoing the systemd unit-file copy for
    /// `browser-sidecar`). Idempotent — safe on a never-installed system.
    ///
    /// Each component subcommand accepts `--json` (suppress live stream,
    /// emit a single JSON summary).
    Uninstall {
        /// Which component to uninstall.
        #[command(subcommand)]
        component: installers::UninstallComponent,
    },
    // === end setup+maintenance subcommands ===
}

#[derive(Subcommand)]
enum EngagementOp {
    /// #58.2 — register one of your own posts. The own-post comment poller
    /// then diffs incoming comments against `seen_comments` until
    /// `posted_at + --days` (default 7) and surfaces approval-gated replies.
    WatchPost {
        /// `linkedin` (twitter/instagram once their pollers land).
        #[arg(long, default_value = "linkedin")]
        platform: String,
        /// The post's stable id (LinkedIn `urn:li:activity:…`).
        #[arg(long)]
        external_id: String,
        /// How many days to keep polling this post. Default 7.
        #[arg(long, default_value_t = 7)]
        days: i64,
    },
    /// #58.3 — add (or refresh) a friend on the engagement watchlist.
    WatchFriend {
        #[arg(long, default_value = "linkedin")]
        platform: String,
        /// Platform handle (LinkedIn member urn `urn:li:fsd_profile:…`).
        #[arg(long)]
        handle: String,
        /// Optional `wiki/people/<slug>.md` to ground the draft prompt.
        #[arg(long)]
        wiki_slug: Option<String>,
        /// `high` (every post) | `medium` (weekly digest) | `low`
        /// (milestones only). Default `medium`.
        #[arg(long, default_value = "medium")]
        engagement: String,
    },
    /// List pending connection requests queued for triage.
    Invites,
}

#[derive(Subcommand)]
enum SchedulePostOp {
    /// Queue an outbound post for `--platform` at `--at` (RFC3339 / unix
    /// seconds). Status starts `queued`; serve drives it through the
    /// preview → posted lifecycle.
    Add {
        /// `linkedin` | `twitter` (or `x`) | `socialapi` |
        /// `socialapi:<sub-platform>`. Case-insensitive. Instagram is NOT
        /// schedulable — its posting path is the composer, not the scheduler.
        #[arg(long)]
        platform: String,
        /// Post body.
        #[arg(long)]
        body: String,
        /// Fire time: RFC3339 (`2026-05-20T15:00:00Z`) or unix seconds.
        #[arg(long)]
        at: String,
    },
    /// List not-yet-terminal scheduled posts (the queue).
    List,
    /// Cancel a queued / previewed post by id.
    Cancel {
        #[arg(long)]
        id: String,
    },
}

/// #117 — multi-repo agent-coding allowlist management (CLI parity with the
/// dashboard /repos admin view). Default-deny: a repo is untouchable until
/// it is `add`ed here (or via the dashboard).
#[derive(Subcommand)]
enum ReposOp {
    /// List allowlisted repos (`--all true` also shows revoked ones).
    List {
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        all: bool,
    },
    /// Allowlist (or re-grant / update) a repo.
    Add {
        /// `owner/name` GitHub full-name.
        #[arg(long)]
        full_name: String,
        /// Branch PRs target. Defaults to `main`.
        #[arg(long, default_value = "main")]
        base_branch: String,
        /// Per-repo verification-gate command (run with `bash -lc`).
        /// Empty ⇒ gate skipped (still prompt-gated, so safe).
        #[arg(long, default_value = "")]
        build_cmd: String,
        /// Comma-separated extra blast-radius path fragments for this repo.
        #[arg(long, default_value = "")]
        blast_radius_extra: String,
        /// Max changed lines accepted in one agent diff for this repo.
        #[arg(long, default_value_t = 600)]
        max_diff_lines: i64,
    },
    /// Revoke a repo (soft-disable + auto-reject its in-flight gate rows).
    Remove {
        #[arg(long)]
        full_name: String,
    },
    /// Show recent agent-PR run history (optionally for one repo).
    History {
        #[arg(long)]
        full_name: Option<String>,
        #[arg(long, default_value_t = 30)]
        limit: i64,
    },
}

#[derive(Subcommand)]
enum ToneOp {
    /// One-shot Composio backfill of `in:sent` history into `tone_examples`.
    /// Rows survive the cleaning + filter pipeline before insert.
    Backfill {
        /// Account entity_id to back-fill against. Defaults to all active
        /// gmail accounts in the store when omitted.
        #[arg(long)]
        account: Option<String>,
        /// Cap on messages pulled per account (Composio paginates 20/page).
        #[arg(long, default_value_t = 500)]
        limit: u32,
        /// Optional `after:YYYY/MM/DD` clause for the Gmail query. None = all-time.
        #[arg(long)]
        since: Option<String>,
    },
    /// Re-summarize one tone profile via Haiku and persist the result.
    Refresh {
        /// Scope to refresh. Accepts `global`, `domain:<domain>`, or
        /// `recipient:<bare_email>`.
        #[arg(long)]
        scope: String,
        /// Account entity_id the profile is keyed under.
        #[arg(long)]
        account: String,
    },
    /// Walk every tone profile and re-summarize any whose
    /// `sample_count - sample_count_at_refresh >= threshold`. Run from the
    /// systemd nightly timer (see `systemd/augmentagent-tone-refresh.*`).
    RefreshStale {
        #[arg(long, default_value_t = 5)]
        threshold: i64,
        /// Hard wallclock budget. Default 5min — matches the systemd timer's
        /// expectation. Bail with a warn log if exceeded; the next run picks
        /// up the leftovers because the staleness predicate is idempotent.
        #[arg(long, default_value_t = 300)]
        budget_secs: u64,
    },
}

#[derive(Subcommand)]
enum DraftsOp {
    /// Cluster recent Revise feedback by overlapping keywords. Surfaces
    /// recurring complaints ("shorter", "less formal", "fix tone") so the
    /// user can decide whether to bake the fix into the drafter prompt.
    FeedbackClusters {
        /// Look back this many days. Default 30.
        #[arg(long, default_value_t = 30u32)]
        since_days: u32,
        /// How many top patterns to print. Default 5.
        #[arg(long, default_value_t = 5usize)]
        top: usize,
    },
}

#[derive(Subcommand)]
enum ApprovalsOp {
    /// List the oldest pending drafts (action id, sender, subject, age).
    /// Read-only; safe to run anytime.
    List {
        /// Cap the number of rows printed. Default 50.
        #[arg(long, default_value_t = 50i64)]
        limit: i64,
    },
    /// Bulk-resolve every pending draft to `approved` (queue-hygiene escape
    /// hatch for "I've handled these out of band"). Does NOT send the Gmail
    /// drafts — it only clears the backlog so new triage isn't downgraded by
    /// backpressure. Requires `--yes` to actually mutate. Audit-logged.
    ApproveAll {
        /// Confirm the destructive op. Without it this is a dry-run preview.
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        yes: bool,
    },
    /// Expire pending drafts older than N days to `timed_out`. Requires
    /// `--yes` to mutate; otherwise prints what *would* be swept. Audit-logged.
    DiscardOlder {
        /// Age threshold in days. Pending rows older than this are expired.
        #[arg(long, default_value_t = 7i64)]
        days: i64,
        /// Confirm the destructive op. Without it this is a dry-run preview.
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        yes: bool,
    },
}

// ---------------------------------------------------------------------------
// Wave-A Cmd subcommand stubs (foundation/swarm-v1).
// Each *Op enum mirrors SlackOp's shape (Login / List / Subscribe / Unsub /
// Subscriptions / PollOnce where applicable). Match arms call unimplemented!
// pointing at the relevant issue so the feature PRs know exactly which arm
// to fill.
// ---------------------------------------------------------------------------

#[derive(Subcommand)]
enum TelegramBotOp {
    /// Persist bot token to keyring (issue #74).
    Login {
        #[arg(long)]
        token: String,
    },
    /// List connected bots from telegram_bots.
    Bots {
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        json: bool,
    },
    /// Disconnect a bot.
    RemoveBot { bot_username: String },
    /// List chats the bot has seen so far.
    ListChats {
        #[arg(long)]
        bot_username: Option<String>,
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        json: bool,
    },
    /// Add or update a subscription for a chat the bot can see.
    Subscribe {
        chat_id: String,
        #[arg(long, value_parser = ["priority", "digest", "store_only"])]
        mode: String,
        #[arg(long)]
        name: Option<String>,
        #[arg(long)]
        bot_username: Option<String>,
    },
    Subscriptions {
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        json: bool,
    },
    Unsubscribe {
        id: String,
    },
    PollOnce {
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        dry_run: bool,
    },
}

#[derive(Subcommand)]
enum WhatsappOp {
    /// Pair a new linked device. Spawns sidecar, prints QR, blocks until
    /// paired or timeout. Persists session to keyring + whatsapp_devices.
    Login {
        #[arg(long)]
        phone: String,
        #[arg(long, default_value_t = 60)]
        timeout_secs: u64,
    },
    Status {
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        json: bool,
    },
    Devices {
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        json: bool,
    },
    Unlink {
        phone: String,
    },
    ListChats {
        #[arg(long, default_value_t = 50)]
        limit: u32,
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        json: bool,
    },
    Subscribe {
        chat_jid: String,
        #[arg(long, value_parser = ["priority", "digest", "store_only"])]
        mode: String,
        #[arg(long)]
        name: Option<String>,
    },
    Subscriptions {
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        json: bool,
    },
    Unsubscribe {
        id: String,
    },
    /// Opt a chat into outbound sends (whatsapp_outbound_allowlist).
    AllowOutbound {
        chat_jid: String,
    },
    DenyOutbound {
        chat_jid: String,
    },
    /// Opt a chat into inbound triage (whatsapp_inbound_allowlist).
    AllowInbound {
        chat_jid: String,
    },
    DenyInbound {
        chat_jid: String,
    },
    PollOnce {
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        dry_run: bool,
    },
}

#[derive(Subcommand)]
enum CalendarOp {
    /// One-shot historical event ingest into the wiki Meeting log.
    /// Phase 2 — Phase 1 ships PollOnce only.
    Backfill {
        #[arg(long, default_value_t = 365)]
        days: u32,
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        dry_run: bool,
    },
    /// Run one Calendar poll cycle and exit.
    PollOnce {
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        dry_run: bool,
    },
    /// Inspect Calendar "subscriptions" — Phase 1 reuses gmail accounts as
    /// the Calendar entity list, so this prints the same accounts the
    /// Calendar poll iterates.
    Subscriptions {
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        json: bool,
    },
    /// #398 — propose an event and post a Discord approval card. The event
    /// is NOT created (and no invites go out) until the operator clicks
    /// Approve on the card; there is no unattended write path.
    CreateEvent {
        /// Event title
        #[arg(long)]
        summary: String,
        /// RFC3339 start with offset, e.g. 2026-07-10T15:00:00-04:00
        #[arg(long)]
        start: String,
        /// Duration in minutes (5-480)
        #[arg(long, default_value_t = 30)]
        duration_min: i64,
        /// Comma-separated attendee emails; each gets an invite on Approve
        #[arg(long, default_value = "")]
        attendees: String,
        /// Optional event description
        #[arg(long)]
        description: Option<String>,
        /// Attach a Google Meet room
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        meet: bool,
        /// Account email to create from (default: first active account)
        #[arg(long)]
        account: Option<String>,
        /// Post the Discord approval card. Without it this prints a
        /// preview and exits — nothing is written anywhere.
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        post: bool,
    },
    /// #399 — read-only schedule lookup across active accounts. Query
    /// mode's calendar tool; applies NO engagement filter (solo events,
    /// focus blocks, and all-day entries all show).
    ListEvents {
        /// RFC3339 window start (default: now)
        #[arg(long)]
        from: Option<String>,
        /// RFC3339 window end (default: --from + --days)
        #[arg(long)]
        to: Option<String>,
        /// Window length in days when --to is absent (clamped to 1-60)
        #[arg(long, default_value_t = 7)]
        days: i64,
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        json: bool,
    },
}

#[derive(Subcommand)]
enum JournalOp {
    /// Run one ShadowNote sync pass and exit. Default --dry-run true:
    /// decrypt + count without firing wiki ingest or advancing the sync
    /// watermark — this is the verify-gate exercise path.
    PollOnce {
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        dry_run: bool,
    },
    /// #900 — deliberately import a full-journal (base-sync) page set that
    /// a normal poll refuses. Capped per run and resumable: re-run, or let
    /// the daemon's ticks continue from the persisted cursor, until
    /// `watermark_ms` is reported.
    Backfill {
        #[arg(long, default_value_t = 200)]
        max_entries: usize,
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        dry_run: bool,
    },
    /// #900 — accept the journal as-is: set the sync watermark to now and
    /// drop any in-progress cursor, so only entries changed from now on
    /// are ingested.
    SkipToNow,
    /// #900 — print the persisted watermark and in-progress cursor.
    Status,
}

#[derive(Subcommand)]
enum PersonOp {
    /// Fold an auto-created contact stub (`*_at_contact.md`) into a canonical
    /// person page: move its identities + `## Source` lines (fill-blanks via
    /// `merge_person_page`), delete the stub, and repoint `identity_phone`
    /// rows so future syncs resolve to the surviving page. Dry-run JSON
    /// report by default; `--apply` writes.
    Merge {
        /// Slug of the stub to fold in (must end in `_at_contact`).
        from: String,
        /// Slug of the surviving canonical page.
        into: String,
        /// Write the target page, delete the stub, repoint the phone index.
        #[arg(long)]
        apply: bool,
    },
    /// #927 — raise a Discord approval card for every high-confidence
    /// duplicate: a stub whose phone / iMessage handle / email already sits on
    /// exactly one canonical page. Writes nothing; each merge runs only when
    /// the owner clicks Approve. Re-running skips pairs already decided.
    ProposeMerges,
}

#[derive(Subcommand)]
enum ImessageOp {
    /// Backfill conversation history into person pages (#885). Dry-run
    /// JSON report by default; pass `--apply` to write fill-blanks pages,
    /// stamp `updated:` with last-message dates, and index phones.
    Sync {
        /// Write wiki pages + phone index (default: dry-run only).
        #[arg(long)]
        apply: bool,
    },
    /// Run one incremental poll pass and exit (#886): persist new bundle
    /// entries as `emails` rows and advance the per-conversation cursor.
    /// Does not fire wiki ingest — that's the daemon loop's job.
    PollOnce,
}

#[derive(Subcommand)]
enum VoiceOp {
    /// Persist the capture-bot token into the keyring slot
    /// `augmentagent/telegram-capture`.
    Login {
        #[arg(long)]
        token: String,
    },
    /// Run one long-poll batch against the capture bot and exit.
    PollOnce {
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        dry_run: bool,
    },
    /// Run the voice-capture listener as a daemon (used by the
    /// augmentagent-telegram-capture systemd unit).
    Serve {
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        dry_run: bool,
    },
}

#[derive(Subcommand)]
enum RedditOp {
    /// Print the Reddit consent URL for the dashboard OAuth bootstrap.
    AuthUrl {
        #[arg(long)]
        client_id: String,
        #[arg(long)]
        redirect_uri: String,
        #[arg(long, default_value = "augmentagent")]
        state: String,
    },
    /// Exchange an authorization code for a permanent refresh token and
    /// persist it to the keyring.
    Exchange {
        #[arg(long)]
        client_id: String,
        #[arg(long)]
        code: String,
        #[arg(long)]
        redirect_uri: String,
    },
}

#[derive(Subcommand)]
enum SocialapiOp {
    /// List every SocialAPI.ai account in the local registry (active first,
    /// then inactive).
    List {
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        json: bool,
    },
    /// Disable an account by id (sets `active = 0`), removing it from the
    /// polling/posting loop without deleting its row.
    Disable {
        /// The SocialAPI.ai-managed account id (see `socialapi list`).
        account_id: String,
    },
    /// Drive the proxied OAuth connect flow — same as
    /// `setup oauth --provider socialapi`. Opens / prints the dashboard
    /// start URL and polls the rollup for a newly-appeared account. The
    /// dashboard `/oauth/socialapi/start` route itself lands in #247.
    Connect {
        /// Maximum seconds to wait for a new account before giving up.
        #[arg(long, default_value_t = 300)]
        timeout_secs: u64,
        /// Whether to attempt `xdg-open` on the start URL (headless: false).
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        open_browser: bool,
    },
    /// Draft a DM reply into a Discord approval card (#571). Mirrors
    /// `gmail compose --post`: without `--post` it prints the draft, with it
    /// the card is raised and Approve sends via the existing
    /// `approve_socialapi` path.
    Dm {
        /// Conversation to reply into. This is the send target.
        #[arg(long)]
        conversation_id: String,
        /// SocialAPI.ai account that owns the conversation. Required by the
        /// inbox API on send; omit only if you know the API can infer it.
        #[arg(long)]
        account_id: Option<String>,
        /// Who you're replying to, for the card title.
        #[arg(long)]
        with: Option<String>,
        /// Underlying network ("instagram", "x", ...) for the card title.
        #[arg(long)]
        platform: Option<String>,
        /// Draft text. Use `--body-file -` to read from stdin instead.
        #[arg(long)]
        body: Option<String>,
        #[arg(long)]
        body_file: Option<String>,
        /// The message you're replying to, so Revise can redraft against it.
        #[arg(long)]
        in_reply_to: Option<String>,
        /// Post the Discord approval card. Without it this prints the draft.
        #[arg(long, default_value_t = false, num_args = 0..=1, default_missing_value = "true", action = clap::ArgAction::Set)]
        post: bool,
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        json: bool,
    },
    /// Draft a reply to a comment on one of your posts into an approval card
    /// (#571). Public surface — the card is the gate.
    Comment {
        /// Post the comment sits under. This is the send target.
        #[arg(long)]
        post_id: String,
        /// Comment being replied to; threads the reply under it.
        #[arg(long)]
        comment_id: String,
        /// SocialAPI.ai account that owns the post.
        #[arg(long)]
        account_id: Option<String>,
        /// Comment author, for the card title.
        #[arg(long)]
        author: Option<String>,
        /// Underlying network ("instagram", "x", ...) for the card title.
        #[arg(long)]
        platform: Option<String>,
        #[arg(long)]
        body: Option<String>,
        #[arg(long)]
        body_file: Option<String>,
        /// The comment text you're replying to, so Revise can redraft.
        #[arg(long)]
        in_reply_to: Option<String>,
        #[arg(long, default_value_t = false, num_args = 0..=1, default_missing_value = "true", action = clap::ArgAction::Set)]
        post: bool,
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        json: bool,
    },
}

#[derive(Subcommand)]
enum GithubOp {
    /// Persist a personal-access token (or gh-cli token) to keyring.
    Login {
        #[arg(long)]
        token: String,
        #[arg(long)]
        login: String,
    },
    Subscribe {
        repo: String,
        #[arg(long, value_parser = ["priority", "digest", "store_only"])]
        mode: String,
    },
    Subscriptions {
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        json: bool,
    },
    Unsubscribe {
        id: String,
    },
    PollOnce {
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        dry_run: bool,
    },
}

#[derive(Subcommand)]
enum DeftOp {
    /// Persist a Deftform workspace API token to the Linux keyring.
    ///
    /// Verifies the token via `GET /workspace` (whoami) **before** writing
    /// anything. The server-confirmed `workspace_id` becomes the keychain
    /// account slot (`augmentagent/deft/<workspace_id>`). Mirrors
    /// `github login` / `voice login`.
    ///
    /// Issue your workspace token at <https://deftform.com/settings/api>.
    /// If `--token` is omitted the CLI reads one token from stdin so the
    /// secret never lands in shell history.
    Login {
        /// Workspace access token (Bearer). Optional — omit to read from
        /// stdin (one line, no echo of the value back).
        #[arg(long)]
        token: Option<String>,
        /// Override the API base. Defaults to the documented prod URL
        /// (`https://deftform.com/api/v1`). Used by tests against a mock
        /// server; no operator should need this in normal use.
        #[arg(long)]
        base_url: Option<String>,
    },
    /// Report whether a Deftform token is in the keyring + optionally
    /// reach out to `GET /workspace` to confirm it's still valid.
    Status {
        /// Workspace id the token was stored under. Linux Secret Service
        /// can't enumerate slots, so the caller must name the workspace.
        /// Falls back to the `AUGMENTAGENT_DEFT_WORKSPACE_ID` env var.
        #[arg(long)]
        workspace_id: Option<String>,
        /// Skip the live `whoami()` reachability probe — keychain-only check.
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        offline: bool,
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        json: bool,
    },
    /// Delete the stored token. Idempotent — no error if nothing is stored.
    Logout {
        /// Workspace id slot to remove. Falls back to
        /// `AUGMENTAGENT_DEFT_WORKSPACE_ID`.
        #[arg(long)]
        workspace_id: Option<String>,
    },
}

#[derive(Subcommand)]
enum MeetupOp {
    /// Watch a Meetup group's upcoming events (channel_id = group urlname).
    Subscribe {
        /// Group url-name slug, e.g. `code-coffee-philly`.
        urlname: String,
        #[arg(long, value_parser = ["digest", "store_only"], default_value = "digest")]
        mode: String,
    },
    Subscriptions {
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        json: bool,
    },
    Unsubscribe {
        id: String,
    },
    PollOnce {
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        dry_run: bool,
    },
    /// On-demand: list a group's upcoming events. No subscription or db
    /// needed — fetches live via the Meetup GraphQL client. This is the
    /// read-only surface query mode reaches through the `augmentagent
    /// meetup events` Bash allowlist entry (#319).
    Events {
        /// Group url-name slug, e.g. `code-coffee-philly`.
        urlname: String,
        #[arg(long, default_value_t = 5)]
        limit: usize,
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        json: bool,
    },
}

#[derive(Subcommand)]
enum GdriveOp {
    /// List connected Drive accounts (entity → email) in this db.
    Accounts {
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        json: bool,
    },
    PollOnce {
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        dry_run: bool,
    },
}

#[derive(Subcommand)]
enum ContactsOp {
    /// Sync contacts → wiki + phone index. Dry-run JSON by default; pass
    /// `--apply` to write fill-blanks pages + index phones.
    Sync {
        /// Backend: `google` (Composio People) or `carddav` (env-configured).
        #[arg(long, default_value = "google")]
        backend: String,
        /// Composio entity id for the `google` backend (the connected
        /// Google account). Ignored for `carddav`.
        #[arg(long, default_value = "default")]
        entity_id: String,
        /// Write wiki pages + phone index (default: dry-run only).
        #[arg(long)]
        apply: bool,
    },
}

#[derive(Subcommand)]
enum ComposeOp {
    /// Fan a single source draft out into per-platform variants. Each
    /// variant is approval-gated independently.
    FanOut {
        /// Path to a markdown/text file containing the source draft.
        #[arg(long)]
        source: PathBuf,
        /// CSV of target platforms. Defaults to "twitter,linkedin,instagram".
        /// Include the token `socialapi` to also fan the draft out across every
        /// connected SocialAPI.ai account (one adapted variant per account):
        /// a single cross-post approval card, then on approve one queued
        /// `scheduled_posts` row per account that #240's publisher sends.
        #[arg(long, default_value = "twitter,linkedin,instagram")]
        platforms: String,
        /// When the queued SocialAPI.ai cross-post rows should fire — RFC3339
        /// or unix seconds. Defaults to now (immediate). Only affects the
        /// `socialapi` cross-post rows.
        #[arg(long)]
        at: Option<String>,
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        dry_run: bool,
    },
}

#[derive(Subcommand)]
enum ProactiveOp {
    /// Run all enabled scans once and print/dispatch the resulting signals.
    ScanOnce {
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        dry_run: bool,
        /// Bypass the config `proactive_enabled` opt-in gate (manual test).
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        force: bool,
    },
    /// List recent ProactiveSignals from sqlite.
    Signals {
        #[arg(long, default_value_t = 25)]
        limit: u32,
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        json: bool,
    },
    /// Snooze a signal by id.
    Snooze {
        id: String,
        #[arg(long, default_value_t = 7)]
        days: u32,
    },
    /// Dismiss a signal by id.
    Dismiss {
        id: String,
    },
}

#[derive(Subcommand)]
enum BrowserOp {
    /// Start the browser sidecar stack (Xvfb + Chromium + Python sidecar)
    /// via systemd. Thin wrapper over `systemctl --user start` for the
    /// three units in `systemd/`. Idempotent.
    Start,
    /// Stop the browser sidecar stack via systemd.
    Stop,
    /// Import cookies from the local Chrome profile into the managed jar.
    /// Stub — wire when the cookie-jar story lands (out of scope for #75 v0).
    ImportCookies {
        #[arg(long)]
        profile: Option<String>,
    },
    /// Probe the sidecar (`ping`) and print connection info.
    Status {
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        json: bool,
    },
    /// Run the §10 acceptance test: navigate twitter.com, screenshot, and
    /// check for a logged-in DOM marker. Pass criterion = the spike works.
    AcceptanceTest {
        /// Where to save the screenshot.
        #[arg(long, default_value = "/tmp/twitter-acceptance.png")]
        out: PathBuf,
    },
}

#[derive(Subcommand)]
enum SlackOp {
    /// Validate + persist Slack auth JSON to Keychain. Keyed by team_id so
    /// multiple workspaces can coexist.
    Login {
        #[arg(long)]
        auth_json: PathBuf,
    },
    /// Persist a Slack auth bundle handed off from the dashboard OAuth
    /// callback. Takes only the Composio handles — team_id/team_name/user_id
    /// are derived server-side via SLACK_FETCH_TEAM_INFO + an auth-test call.
    /// This mirrors Orchid's pattern: trust ACTIVE status, no channel-list
    /// probe at OAuth time. Also upserts the row in `slack_workspaces`.
    PersistAuth {
        #[arg(long)] entity_id: String,
        #[arg(long)] connection_id: String,
        #[arg(long)] composio_api_key: String,
    },
    /// List connected Slack workspaces (from `slack_workspaces`).
    Workspaces {
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        json: bool,
    },
    /// Disconnect a workspace: hard-deletes the Keychain slot AND the
    /// `slack_workspaces` row. Subscriptions on that workspace get soft-
    /// deactivated. Reconnect via OAuth to start fresh.
    RemoveWorkspace { team_id: String },
    /// Nuclear reset for Slack state — drops every workspace row, every
    /// Slack subscription (hard delete), and every Keychain slot under
    /// `augmentagent/slack/*`. Use when local state is hopelessly out of
    /// sync with what Composio has on its side.
    Reset {
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        confirm: bool,
    },
    /// List conversations the user can see.
    ListConversations {
        /// Slack workspace `team_id`. Required when multiple workspaces are
        /// configured; defaults to the sole workspace when only one exists.
        #[arg(long)]
        team_id: Option<String>,
        /// Slack-style CSV of types to include.
        #[arg(long, default_value = "public_channel,private_channel,im,mpim")]
        types: String,
        #[arg(long, default_value_t = 50)]
        limit: u32,
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        json: bool,
    },
    /// Add or update a subscription in the shared channel_subscriptions table.
    Subscribe {
        channel_id: String,
        #[arg(long, value_parser = ["priority", "digest", "store_only"])]
        mode: String,
        #[arg(long)]
        name: Option<String>,
        /// Slack workspace `team_id` the channel belongs to. Required when
        /// multiple workspaces are configured.
        #[arg(long)]
        team_id: Option<String>,
    },
    /// List active subscriptions (platform='slack').
    Subscriptions {
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        json: bool,
    },
    /// Soft-remove a subscription by id.
    Unsubscribe { id: String },
    /// Run one poll cycle and exit.
    PollOnce {
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        dry_run: bool,
    },
}

#[derive(Subcommand)]
enum DiscordOp {
    /// Validate + persist harvested Discord creds JSON to Keychain.
    ///
    /// Creds JSON must contain `user_id`, `token`, `super_properties_b64`, and
    /// `user_agent`. Use `scripts/discord-harvest.sh` to produce it.
    Login {
        #[arg(long)]
        creds_json: PathBuf,
    },
    /// Report whether Discord auth is loaded (used by dashboard status panel).
    Status {
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        json: bool,
    },
    /// List DM channels (id + display name).
    ListDms {
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        json: bool,
    },
    /// List guilds (id + name).
    ListGuilds {
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        json: bool,
    },
    /// List text channels in a guild.
    ListGuildChannels {
        guild_id: String,
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        json: bool,
    },
    /// Add or update a subscription in the shared channel_subscriptions table.
    Subscribe {
        channel_id: String,
        #[arg(long, value_parser = ["priority", "digest", "store_only"])]
        mode: String,
        #[arg(long)]
        name: Option<String>,
    },
    /// List active subscriptions (platform='discord').
    Subscriptions {
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        json: bool,
    },
    /// Soft-remove a subscription by id.
    Unsubscribe { id: String },
    /// Run one poll cycle and exit.
    PollOnce {
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        dry_run: bool,
    },
}

#[derive(Subcommand)]
enum DocOp {
    /// Extract text from a PDF / DOCX / DOC. Writes `<file>.txt` beside the
    /// input unless `--out` is given (`--out -` prints the text to stdout after
    /// the receipt line) and reports whether the OCR stage ran.
    Extract {
        /// Path to the document.
        file: PathBuf,
        /// Output path for the text; `-` prints to stdout.
        #[arg(long)]
        out: Option<String>,
        /// Force the kind (pdf|docx|doc) when the filename doesn't tell.
        #[arg(long)]
        kind: Option<String>,
        /// Skip the OCR stage even when MISTRAL_API_KEY is configured.
        #[arg(long, default_value_t = false)]
        no_ocr: bool,
        /// Print a JSON receipt instead of the human line.
        #[arg(long, default_value_t = false, num_args = 0..=1, default_missing_value = "true", action = clap::ArgAction::Set)]
        json: bool,
    },
}

#[derive(Subcommand)]
enum GmailOp {
    /// Search all connected Gmail accounts with a Gmail query string
    /// (e.g. `from:jeremy@acme.com`, `subject:deadline after:2026/04/01`).
    /// Prints a short listing (from / subject / date / messageId / threadId)
    /// by default.
    Search {
        /// Gmail search query. Supports all operators `from:`, `to:`,
        /// `subject:`, `has:`, `after:`, `before:`, etc.
        #[arg(long)]
        query: String,
        /// Max results per account.
        #[arg(long, default_value_t = 20)]
        limit: u32,
        /// Also include the email body in the output.
        #[arg(long, default_value_t = false, num_args = 0..=1, default_missing_value = "true", action = clap::ArgAction::Set)]
        full: bool,
        /// Restrict the search to a single account instead of fanning out over
        /// all connected ones (#482). Matches the account's entity id (exact)
        /// or email (case-insensitive substring), so either a short handle or
        /// the full address selects the same account. Lets you route around a
        /// throttled account.
        #[arg(long)]
        account: Option<String>,
    },
    /// List a message's attachments (#937): index, filename, MIME type, and
    /// the attachmentId that `get-attachment` needs. Message ids come from
    /// `gmail search`.
    ListAttachments {
        /// Email address or Composio entity_id. Required when more than one
        /// account is connected (message ids are per-mailbox).
        #[arg(long)]
        account: Option<String>,
        /// Gmail messageId (from `gmail search`).
        #[arg(long)]
        message_id: String,
        #[arg(long, default_value_t = false, num_args = 0..=1, default_missing_value = "true", action = clap::ArgAction::Set)]
        json: bool,
    },
    /// Download one attachment (#937) to `/tmp/aa-doc-<id>-<idx>.<ext>` — a
    /// path the ask agent's Read tool may open — and, for PDF/DOCX/DOC,
    /// extract its text to the sibling `.txt`, running Mistral OCR when a PDF
    /// has no text layer and MISTRAL_API_KEY is set (#939).
    GetAttachment {
        /// Email address or Composio entity_id. Required when more than one
        /// account is connected.
        #[arg(long)]
        account: Option<String>,
        /// Gmail messageId (from `gmail search`).
        #[arg(long)]
        message_id: String,
        /// Select by attachmentId (from `list-attachments`)…
        #[arg(long)]
        attachment_id: Option<String>,
        /// …or by filename, case-insensitive, as printed by `gmail search`.
        #[arg(long)]
        name: Option<String>,
        /// Override the download path (the default is what the ask agent can Read).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Extract text for document kinds (default true).
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        extract: bool,
        #[arg(long, default_value_t = false, num_args = 0..=1, default_missing_value = "true", action = clap::ArgAction::Set)]
        json: bool,
    },
    /// List active Gmail accounts (so the chat agent can pick `--account`).
    Accounts {
        #[arg(long, default_value_t = false, num_args = 0..=1, default_missing_value = "true", action = clap::ArgAction::Set)]
        json: bool,
    },
    /// Create a new draft in Gmail. Returns the draft id (and a Gmail URL
    /// to open it in the web UI). Use `--thread-id` for a reply draft.
    ///
    /// When `--post` is set, also surface a Discord approval card (Approve /
    /// Revise / Skip) wired into the existing event handler, so the user can
    /// send straight from chat instead of copy/pasting the draft into Gmail
    /// (#352). Works for replies AND brand-new emails (#412): pass
    /// `--thread-id` + `--reply-to-message-id` together for a reply card, or
    /// neither for a new-email card (the action is keyed on a synthetic
    /// `compose:<draft_id>` id and Revise redrafts against the draft body).
    Compose {
        /// Email address (e.g. `me@example.com`) or Composio entity_id of the
        /// sending account. Required when more than one account is connected.
        #[arg(long)]
        account: Option<String>,
        /// Recipient address(es). Repeat the flag or pass a comma-separated
        /// list for multiple To recipients (#439): `--to a@x.com --to b@y.com`
        /// or `--to 'a@x.com, b@y.com'`. Display names are stripped.
        #[arg(long, required = true)]
        to: Vec<String>,
        /// Cc recipient(s). Repeatable / comma-separated like `--to` (#439).
        #[arg(long)]
        cc: Vec<String>,
        /// Bcc recipient(s). Repeatable / comma-separated like `--to` (#439).
        #[arg(long)]
        bcc: Vec<String>,
        #[arg(long)]
        subject: String,
        /// Body text. Use `--body-file -` to read from stdin instead.
        #[arg(long)]
        body: Option<String>,
        /// Path to a file containing the body. Use `-` for stdin. Mutually
        /// exclusive with `--body`.
        #[arg(long)]
        body_file: Option<String>,
        /// Thread to attach the draft to (makes it a reply). Accepts a Gmail
        /// threadId or a messageId from `gmail search` — a messageId is
        /// resolved to its thread automatically (#381). Must belong to the
        /// mailbox picked by `--account`.
        #[arg(long)]
        thread_id: Option<String>,
        #[arg(long, default_value_t = false, num_args = 0..=1, default_missing_value = "true", action = clap::ArgAction::Set)]
        json: bool,
        /// Also post a Discord approval card for this draft (#352, #412).
        /// For a reply card pass `--thread-id` + `--reply-to-message-id`;
        /// for a new-email card pass neither.
        #[arg(long, default_value_t = false, num_args = 0..=1, default_missing_value = "true", action = clap::ArgAction::Set)]
        post: bool,
        /// Allow a second independent draft/card even when a pending approval
        /// card already exists for this recipient + subject. Without this,
        /// compose --post replaces the pending follow-up card (#596).
        #[arg(long, default_value_t = false, num_args = 0..=1, default_missing_value = "true", action = clap::ArgAction::Set)]
        allow_duplicate: bool,
        /// Attach a local file to the draft (#417). One file; uploaded via
        /// Composio's attachment store before the draft is created.
        #[arg(long)]
        attach: Option<PathBuf>,
        /// Gmail messageId of the inbound message we're replying to. Used as
        /// the action row's messageId so the Approve / Revise / Skip handlers
        /// resolve the same way they do for auto-triage replies. Pass it
        /// (with `--thread-id`) for reply cards; omit for new-email cards.
        #[arg(long)]
        reply_to_message_id: Option<String>,
        /// Original sender address (shown on the approval card's `From`
        /// field). Defaults to `--to` when omitted, which is correct for a
        /// straight reply.
        #[arg(long)]
        reply_to_from: Option<String>,
        /// Original subject (used as the approval card's title). Defaults to
        /// the outgoing `--subject` with any leading `Re:` stripped.
        #[arg(long)]
        reply_to_subject: Option<String>,
        /// Original message body, used as the card's context block above the
        /// draft. Provide via `--reply-to-body-file -` for stdin. Empty when
        /// omitted (card still renders, just without inline context).
        #[arg(long)]
        reply_to_body: Option<String>,
        #[arg(long)]
        reply_to_body_file: Option<String>,
        /// #502 — propose a future send time for this draft. Requires
        /// `--post`: the card shows "[sends: <local time>]" and Approve ARMS
        /// the schedule instead of sending immediately (the daemon fires it
        /// when due). Accepts owner-local `YYYY-MM-DD HH:MM` (preferred —
        /// never hand-compute a future UTC offset across a DST boundary),
        /// RFC3339, `tomorrow 9am`, weekday forms, or `in Nm/Nh/Nd`.
        /// Must be between 2 minutes and 60 days out.
        #[arg(long)]
        send_at: Option<String>,
    },
    /// Replace an existing draft's content. Composio has no update-in-place
    /// tool (#382), so this creates a replacement draft and deletes the old
    /// one — it prints a NEW draft id; the old id stops working.
    UpdateDraft {
        #[arg(long)]
        account: Option<String>,
        #[arg(long)]
        draft_id: String,
        /// Recipient address(es); repeatable / comma-separated (#439).
        #[arg(long, required = true)]
        to: Vec<String>,
        /// Cc recipient(s) for the REPLACEMENT draft; repeatable /
        /// comma-separated. Like --attach, cc/bcc on the old draft are NOT
        /// carried over — re-pass them here (#439).
        #[arg(long)]
        cc: Vec<String>,
        /// Bcc recipient(s) for the replacement draft (#439).
        #[arg(long)]
        bcc: Vec<String>,
        #[arg(long)]
        subject: String,
        #[arg(long)]
        body: Option<String>,
        #[arg(long)]
        body_file: Option<String>,
        /// Thread to attach the replacement draft to. Accepts a threadId or a
        /// messageId (auto-resolved). When omitted, the old draft's thread is
        /// detected and preserved automatically where possible.
        #[arg(long)]
        thread_id: Option<String>,
        /// Attach a local file to the REPLACEMENT draft (#417). Note: the
        /// replacement carries only what you pass here — an attachment on
        /// the old draft is NOT carried over (Composio can't read it back).
        #[arg(long)]
        attach: Option<PathBuf>,
    },
    /// Send an existing draft.
    Send {
        #[arg(long)]
        account: Option<String>,
        #[arg(long)]
        draft_id: String,
    },
    /// Delete an unsent draft.
    DeleteDraft {
        #[arg(long)]
        account: Option<String>,
        #[arg(long)]
        draft_id: String,
    },
    /// Compose AND send in one shot. Use only when the user has explicitly
    /// confirmed the recipient/subject/body — there's no approval card.
    SendNow {
        #[arg(long)]
        account: Option<String>,
        /// Recipient address(es); repeatable / comma-separated (#439).
        #[arg(long, required = true)]
        to: Vec<String>,
        /// Cc recipient(s); repeatable / comma-separated (#439).
        #[arg(long)]
        cc: Vec<String>,
        /// Bcc recipient(s); repeatable / comma-separated (#439).
        #[arg(long)]
        bcc: Vec<String>,
        #[arg(long)]
        subject: String,
        #[arg(long)]
        body: Option<String>,
        #[arg(long)]
        body_file: Option<String>,
        #[arg(long)]
        thread_id: Option<String>,
        /// Attach a local file (#417).
        #[arg(long)]
        attach: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum LinkedinOp {
    /// Validate + persist harvested cookies from a JSON file.
    ///
    /// The JSON must contain `member_urn` and a `cookies` object with at
    /// least `li_at` and `JSESSIONID`. See docs/LINKEDIN.md for how to
    /// extract these from Chrome devtools.
    Login {
        /// Path to the cookies JSON file.
        #[arg(long)]
        cookies_json: PathBuf,
    },
    /// Run one LinkedIn poll cycle and exit. Respects `--dry-run`.
    PollOnce {
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        dry_run: bool,
    },
    /// Quick read-only check: list recent threads + print peer + snippet.
    /// Good smoke test after `login` to confirm cookies work.
    Recent,
    /// Sync 1st-degree connections into the wiki as dormant contacts (#61).
    ///
    /// Default is a **dry run**: prints a JSON [`SyncReport`] and writes
    /// nothing. Pass `--apply` to write fill-blanks-only wiki pages. Mode
    /// (full vs delta) is decided from the persisted cursor unless
    /// `--full` forces a full walk.
    ConnectionsSync {
        /// Write the merged pages (default: dry-run JSON only).
        #[arg(long)]
        apply: bool,
        /// Force a full sync regardless of the persisted cursor.
        #[arg(long)]
        full: bool,
    },
    /// Publish a feed post via Voyager `normShares` (#51/#77): text plus any
    /// number of images. Manual/test path — the daemon posts through the
    /// approval pipeline, not this command.
    Post {
        /// Post body (≤3000 chars; ~140 visible before the "see more" fold).
        #[arg(long)]
        text: String,
        /// Image to attach. Repeat for a multi-image post; order is preserved
        /// as display order. Capped by AUGMENTAGENT_LINKEDIN_MAX_IMAGES.
        #[arg(long = "image")]
        images: Vec<PathBuf>,
        /// Audience: `public` (default) or `connections`.
        #[arg(long, default_value = "public")]
        visibility: String,
        /// Build + print the request body, don't send.
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        dry_run: bool,
    },
    /// List recent LinkedIn DM threads with their conversation urns.
    ///
    /// Read-only. Exists because `linkedin dm --conversation-urn` needs an id
    /// the agent otherwise has no way to obtain: a DM that arrived only as a
    /// notification stub was never ingested, so it has no `thread_id` in the
    /// store to look up. Gmail has `gmail search` for exactly this; LinkedIn
    /// had nothing.
    RecentDms {
        /// Cap the number of threads listed.
        #[arg(long, default_value_t = 20)]
        limit: usize,
        #[arg(long, default_value_t = false, num_args = 0..=1, default_missing_value = "true", action = clap::ArgAction::Set)]
        json: bool,
    },
    /// Draft a DM reply into a Discord approval card (#572). Mirrors
    /// `gmail compose --post`: without `--post` it prints the draft, with it
    /// the card is raised and Approve sends via `approve_linkedin`.
    Dm {
        /// Conversation urn to reply into. This is the send target.
        #[arg(long)]
        conversation_urn: String,
        /// Who you're replying to, for the card title.
        #[arg(long)]
        with: Option<String>,
        /// Draft text. Use `--body-file -` to read from stdin instead.
        #[arg(long)]
        body: Option<String>,
        #[arg(long)]
        body_file: Option<String>,
        /// The message you're replying to, so Revise can redraft against it.
        #[arg(long)]
        in_reply_to: Option<String>,
        #[arg(long, default_value_t = false, num_args = 0..=1, default_missing_value = "true", action = clap::ArgAction::Set)]
        post: bool,
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        json: bool,
    },
    /// Draft a comment on a post into a Discord approval card (#572).
    /// Public surface — the card is the gate.
    Comment {
        /// Post urn to comment on. Approve calls `post_comment` with it.
        #[arg(long)]
        post_urn: String,
        /// Post author, for the card title.
        #[arg(long)]
        author: Option<String>,
        #[arg(long)]
        body: Option<String>,
        #[arg(long)]
        body_file: Option<String>,
        /// The post text you're replying to, so Revise can redraft.
        #[arg(long)]
        in_reply_to: Option<String>,
        #[arg(long, default_value_t = false, num_args = 0..=1, default_missing_value = "true", action = clap::ArgAction::Set)]
        post: bool,
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        json: bool,
    },
}

#[derive(Subcommand)]
enum TwitterOp {
    /// Validate + persist a harvested X session bundle from a JSON file.
    ///
    /// The JSON must contain `user_id`, `screen_name`, and a `cookies`
    /// object with at least `auth_token` and `ct0`. See
    /// docs/twitter-protocol.md + scripts/twitter-harvest.sh.
    Login {
        /// Path to the session JSON file.
        #[arg(long)]
        session_json: PathBuf,
    },
    /// Post a tweet (or reply with `--reply-to <id>`). Respects `--dry-run`
    /// and the hard 15/day quota. Media is deferred (phase 2).
    Post {
        #[arg(long)]
        text: String,
        /// When set, post as a reply to this tweet id.
        #[arg(long)]
        reply_to: Option<String>,
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        dry_run: bool,
    },
    /// Run one close-friend feed poll cycle and print the WorkItems found.
    /// Read-only — no replies are posted (those go via Discord approval).
    PollOnce,
    /// #14 operator validation harness. Given an already-harvested session
    /// (keychain / legacy file — run `twitter login` first), exercise every
    /// documented endpoint and print a pass/fail grid + per-probe response-
    /// shape fingerprint mapping to the `REQUIRES LIVE OPERATOR VALIDATION`
    /// flags in docs/twitter-protocol.md.
    ///
    /// **Mock-only by default**: without `--allow-live` (and no
    /// `AUGMENTAGENT_TWITTER_BASE_URL` capture-proxy override) the harness
    /// makes NO live x.com call — read probes are skipped and the report is
    /// flagged mock-only. A live sign-off REQUIRES `--allow-live` on a real
    /// session. Even when live it is read-only unless `--allow-write` is set.
    Validate {
        /// Emit the report as JSON (an attachable validation artifact)
        /// instead of the human table.
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        json: bool,
        /// Permit live x.com calls. OFF by default (mock-only build) — the
        /// harness never reaches x.com without this (or a capture-proxy
        /// base-url override).
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        allow_live: bool,
        /// Permit the live write probes (CreateTweet / DM send). OFF by
        /// default — the harness never posts public content without this.
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        allow_write: bool,
        /// Throwaway tweet id to reply to for a live CreateTweet probe
        /// (requires `--allow-write`).
        #[arg(long)]
        probe_reply_to: Option<String>,
        /// Conversation id for a live DM-send probe (requires
        /// `--allow-write`).
        #[arg(long)]
        probe_conversation_id: Option<String>,
    },
}

#[derive(Subcommand)]
enum ResumeOp {
    /// Parse a resume file and seed the wiki with an `about/me.md` and
    /// stub `people/<slug>.md` pages for every named contact.
    Ingest {
        /// Path to the resume. Supported: .txt, .md, .pdf (requires `pdftotext`).
        #[arg(long)]
        file: PathBuf,
    },
}

/// FlyOnTheWall transcript-repo operations (#915).
#[derive(Subcommand, Debug)]
enum TranscriptsOp {
    /// Fast-forward the transcript clone, then ingest any new meetings.
    ///
    /// Read-only against that repo: FlyOnTheWall owns it and re-pushes retitled
    /// meetings to the same path, so this side never commits, merges or pushes.
    /// Dedup is the `emails` table (`fotw:<id>`), so a rescan is free.
    Sync {
        /// Scan and report without pulling or ingesting. Defaults to true —
        /// pass `--dry-run false` for a live run (PR #922: same safe default
        /// as every other channel's bare command).
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        dry_run: bool,
        /// Ingest what is already on disk; skip the git pull.
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        no_pull: bool,
        /// Ingest only meetings whose export recorded disclosure (§11).
        #[arg(long, default_value_t = false)]
        require_disclosed: bool,
    },
}

#[cfg(test)]
mod transcripts_cli_args_tests {
    use super::*;

    fn sync_args(argv: &[&str]) -> (bool, bool, bool) {
        let cli = Cli::try_parse_from(argv).expect("argv must parse");
        match cli.cmd {
            Cmd::Transcripts {
                op:
                    TranscriptsOp::Sync {
                        dry_run,
                        no_pull,
                        require_disclosed,
                    },
            } => (dry_run, no_pull, require_disclosed),
            _ => panic!("expected transcripts sync"),
        }
    }

    /// PR #922 review — every other channel's bare command is a dry run
    /// (`default_value_t = true, ArgAction::Set`); a bare `transcripts sync`
    /// doing a live pull + ingest breaks that convention exactly where an
    /// operator is most likely to type it.
    #[test]
    fn bare_transcripts_sync_is_a_dry_run() {
        let (dry_run, no_pull, require_disclosed) =
            sync_args(&["augmentagent", "transcripts", "sync"]);
        assert!(dry_run, "bare `transcripts sync` must default to a dry run");
        assert!(!no_pull, "the pull itself stays on by default");
        assert!(!require_disclosed);
    }

    /// A live run is an explicit opt-in, same shape as the other channels.
    #[test]
    fn a_live_run_is_an_explicit_opt_in() {
        let (dry_run, ..) =
            sync_args(&["augmentagent", "transcripts", "sync", "--dry-run", "false"]);
        assert!(!dry_run);
        let (_, no_pull, _) =
            sync_args(&["augmentagent", "transcripts", "sync", "--no-pull", "true"]);
        assert!(no_pull);
    }
}

#[derive(Subcommand)]
enum WikiOp {
    /// Health-check the wiki: contradictions, orphans, stale claims, missing cross-refs.
    Lint {
        /// Write the report to this path. Default: stdout.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Ask the wiki a question. Spawns Opus with read-only access and prints the answer.
    Ask {
        /// The question. Wrap in quotes if multi-word.
        question: String,
        /// Also post the answer to the Discord approval channel via a
        /// one-shot HTTP client, honoring `ATTACH:` file markers (#440).
        /// Needs DISCORD_BOT_TOKEN + DISCORD_CHANNEL_ID in the env.
        #[arg(long, default_value_t = false)]
        post: bool,
    },
    /// Backfill v2 schema fields onto cold person pages via Haiku. See #78.
    ///
    /// Re-runnable; per-page idempotent via the `migrated:` marker. Writes
    /// per-batch git commits authored as Nolan Makatche.
    Migrate {
        /// Schema version target. Only `v2` is supported today.
        #[arg(long, default_value = "v2")]
        to: String,
        /// Don't write to disk or commit; print what would change.
        #[arg(long)]
        dry_run: bool,
        /// Bounded parallel Haiku calls. Default 4 (well under 50 RPM).
        #[arg(long, default_value_t = 4)]
        concurrency: usize,
        /// Only process the first N eligible pages. Useful for sample runs.
        #[arg(long)]
        limit: Option<usize>,
        /// Git branch label recorded in the run summary (the CLI does NOT
        /// switch branches — operator must check out the desired branch
        /// first). Default `migration/wiki-v2`.
        #[arg(long, default_value = "migration/wiki-v2")]
        branch: String,
        /// Run even if the daemon systemd unit is active. Default refuses
        /// to avoid races against live ingest writes.
        #[arg(long)]
        force: bool,
    },
    /// Two-way sync the wiki with its private GitHub mirror (epic #474).
    ///
    /// Commits local page changes, pulls owner edits made on GitHub
    /// (owner-wins on same-line conflicts), then pushes. Auth is the
    /// ambient `gh` credential — no token in `.env`. The wiki dir must
    /// already be a git repo with a private `origin` (bootstrap runs once
    /// out of band; see KB sync #1).
    Sync {
        /// Print the plan (local changes, ahead/behind) without mutating.
        #[arg(long)]
        dry_run: bool,
        /// Push-only: skip the pull/rebase step. Used for the very first
        /// push when `origin/main` doesn't exist yet.
        #[arg(long)]
        no_pull: bool,
    },
    /// Inspect or rebuild `index.md` (#642). The index is a derived
    /// catalog of every page under people/, threads/, projects/ —
    /// generated, never hand-kept.
    Index {
        /// Regenerate index.md from the pages on disk (atomic write).
        /// Without this flag, print coverage of the current index
        /// (entries vs. pages on disk) and change nothing.
        #[arg(long)]
        rebuild: bool,
    },
}

#[derive(Subcommand)]
enum RatelimitOp {
    /// Dump rate_events for one account in `[since, until]`. The artifact
    /// you'd attach to a LinkedIn / X appeal. Defaults to the last 7 days
    /// when `--since` is omitted.
    Audit {
        /// Account id (LinkedIn URN, X user id, IG handle). Required.
        #[arg(long)]
        account: String,
        /// Optional platform filter (`instagram`, `linkedin`, `twitter`).
        #[arg(long)]
        platform: Option<String>,
        /// ISO 8601 start. Default: now − 7 days.
        #[arg(long)]
        since: Option<String>,
        /// ISO 8601 end. Default: now.
        #[arg(long)]
        until: Option<String>,
        /// Emit JSON instead of a human table.
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        json: bool,
    },
    /// Print the active circuit-breaker halts (if any). Always JSON.
    Halts,
    /// Print the static cap-matrix (#83 §3) as JSON. Useful for verifying
    /// what the daemon thinks the caps are without grepping source.
    Caps,
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();
    // Send tracing to stderr so JSON-mode subcommands (consumed by the
    // dashboard via shell-out) don't get their stdout polluted with log
    // lines. Production systemd captures both streams to log files; in dev
    // you still see logs alongside data in the terminal.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    let db_path = cli
        .db
        .clone()
        .or_else(|| std::env::var("AUGMENTAGENT_DB").ok().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("data.db"));
    info!(db = %db_path.display(), "opening store");
    let store = Arc::new(Store::open(&db_path).context("open store")?);

    // Wiki page freshness (#642): the post-ingest index rebuild stamps every
    // entry with the age of its cited evidence. Installed here, once, so the
    // daemon, poll-once, and every wiki subcommand resolve identically.
    {
        let store = Arc::clone(&store);
        augmentagent_channel_core::ingest::set_first_seen_resolver(Arc::new(move |id: &str| {
            store.email_first_seen_at(id).ok().flatten()
        }));
    }

    match cli.cmd {
        Cmd::AccountsList => {
            let accounts = store.get_active_gmail_accounts()?;
            if accounts.is_empty() {
                println!("(no active gmail accounts)");
            } else {
                for a in accounts {
                    let email = if a.email.is_empty() {
                        "(unknown — run accounts-backfill-emails)".to_string()
                    } else {
                        a.email.clone()
                    };
                    println!(
                        "{}\tentity={}\temail={}\tactive={}",
                        a.id, a.entity_id, email, a.active
                    );
                }
            }
            Ok(())
        }
        Cmd::SelfImprove {
            dry_run,
            multi_repo,
            ref approve,
            ref reject,
            approve_open,
        } => {
            // Gate resolution + PR-opening are pure store/gh ops (no repo
            // pass). Handle them first so they short-circuit.
            if let Some(run_id) = approve {
                match store.approve_agent_pr_run(run_id)? {
                    Some(r) => println!(
                        "approved run {} ({} #{}); run `self-improve --approve-open true` to open the draft PR",
                        r.id, r.repo_full_name, r.issue_number
                    ),
                    None => println!("run {run_id}: not pending (already resolved or unknown)"),
                }
                return Ok(());
            }
            if let Some(run_id) = reject {
                match store.reject_agent_pr_run(run_id)? {
                    Some(r) => println!("rejected run {} ({})", r.id, r.repo_full_name),
                    None => println!("run {run_id}: not pending (already resolved or unknown)"),
                }
                return Ok(());
            }
            if approve_open {
                let msg = self_improve::open_approved_runs(&store).await?;
                println!("{msg}");
                return Ok(());
            }
            if multi_repo {
                let deploy_root = std::env::current_dir().context("current_dir")?;
                let (broker, _) =
                    build_broker(&cli, Arc::clone(&store), dry_run).await?;
                let msg = self_improve::run_multi_repo_once(
                    &store,
                    broker.as_ref(),
                    &deploy_root,
                    dry_run,
                )
                .await?;
                println!("{msg}");
                return Ok(());
            }
            let repo_root = std::env::current_dir().context("current_dir")?;
            let msg = self_improve::run_once(&repo_root, dry_run).await?;
            println!("{msg}");
            Ok(())
        }
        Cmd::Repos { op } => {
            match op {
                ReposOp::List { all } => {
                    let repos = store.list_agent_repos(!all)?;
                    if repos.is_empty() {
                        println!("(no allowlisted repos — default-deny)");
                    } else {
                        for r in repos {
                            println!(
                                "{}\tbase={}\tbuild={:?}\tcap={}\tenabled={}",
                                r.full_name,
                                r.base_branch,
                                r.build_cmd,
                                r.max_diff_lines,
                                r.enabled
                            );
                        }
                    }
                    Ok(())
                }
                ReposOp::Add {
                    full_name,
                    base_branch,
                    build_cmd,
                    blast_radius_extra,
                    max_diff_lines,
                } => {
                    let r = store.upsert_agent_repo(
                        &full_name,
                        &base_branch,
                        &build_cmd,
                        &blast_radius_extra,
                        max_diff_lines,
                    )?;
                    println!("allowlisted {} (base {})", r.full_name, r.base_branch);
                    Ok(())
                }
                ReposOp::Remove { full_name } => {
                    let cancelled = store.revoke_agent_repo(&full_name)?;
                    println!(
                        "revoked {full_name} ({cancelled} in-flight gate row(s) auto-rejected)"
                    );
                    Ok(())
                }
                ReposOp::History { full_name, limit } => {
                    let rows =
                        store.list_agent_pr_runs(full_name.as_deref(), limit)?;
                    if rows.is_empty() {
                        println!("(no agent-PR runs)");
                    } else {
                        for r in rows {
                            println!(
                                "{}\t#{}\t{}\t{}\t{}",
                                r.repo_full_name,
                                r.issue_number,
                                r.status,
                                r.pr_url.unwrap_or_else(|| "-".into()),
                                r.id
                            );
                        }
                    }
                    Ok(())
                }
            }
        }
        Cmd::AccountsBackfillEmails => {
            let lines = backfill_gmail_emails(&store, false).await?;
            if lines.is_empty() {
                println!("(no active gmail accounts)");
            } else {
                println!("entity\temail\tid");
                for l in lines {
                    println!("{l}");
                }
            }
            Ok(())
        }
        Cmd::PollOnce { dry_run } => {
            let (broker, _) = build_broker(&cli, Arc::clone(&store), dry_run).await?;
            let ch = build_channel(&cli, store, broker, dry_run, 120)?;
            let out = ch.poll_once().await?;
            println!("{out:#?}");
            Ok(())
        }
        Cmd::Serve {
            interval_secs,
            dry_run,
            no_email,
        } => {
            let (broker, approver) = build_broker(&cli, Arc::clone(&store), dry_run).await?;
            // Default (no_email=false) keeps the exact prod path: build + `?`
            // propagate + unconditional spawn. `--no-email true` makes a
            // tenant agent that runs Discord/GitHub/Meetup/Drive only.
            let gmail_ch = if no_email {
                info!("--no-email set: Gmail channel disabled (multi-tenant mode)");
                None
            } else {
                Some(build_channel(
                    &cli,
                    Arc::clone(&store),
                    Arc::clone(&broker),
                    dry_run,
                    interval_secs,
                )?)
            };
            // LinkedIn is optional — builds only if cookies exist; an absent
            // or invalid auth file downgrades the daemon to Gmail-only with
            // a warning, no crash.
            let linkedin_ch =
                match build_linkedin_channel(&cli, Arc::clone(&store), Arc::clone(&broker), dry_run)
                {
                    Ok(ch) => Some(ch),
                    Err(e) => {
                        warn!("linkedin channel disabled: {e:#}");
                        None
                    }
                };
            // LinkedIn friend-post engagement (#13). Independent 6h cadence;
            // self-disables when LinkedIn auth is absent (same gate as the
            // DM channel) or when no wiki is configured.
            let linkedin_feed = match build_linkedin_feed_engagement(
                &cli,
                Arc::clone(&store),
                Arc::clone(&broker),
                dry_run,
            ) {
                Ok(ch) => Some(ch),
                Err(e) => {
                    warn!("linkedin feed engagement disabled: {e:#}");
                    None
                }
            };
            // #58.2/.3/.4 — the three remaining engagement sub-features. Each
            // shares the DM channel's LinkedIn auth gate (self-disables with a
            // warning when auth is absent) and is inert until its durable
            // table is populated (own_posts / friend_watchlist / pending
            // invitations) — same proven-safe always-spawn-empty-is-free
            // pattern as the scheduled-post engine. All outbound stays
            // approval-gated + RateGovernor-capped.
            let own_post_engagement = match build_own_post_comment_engagement(
                &cli,
                Arc::clone(&store),
                Arc::clone(&broker),
                dry_run,
            ) {
                Ok(e) => Some(e),
                Err(e) => {
                    warn!("linkedin own-post comment engagement disabled: {e:#}");
                    None
                }
            };
            let friend_feed_engagement = match build_friend_feed_engagement(
                &cli,
                Arc::clone(&store),
                Arc::clone(&broker),
                dry_run,
            ) {
                Ok(e) => Some(e),
                Err(e) => {
                    warn!("linkedin friend-feed engagement disabled: {e:#}");
                    None
                }
            };
            let connection_triage = match build_connection_request_engagement(
                &cli,
                Arc::clone(&store),
                Arc::clone(&broker),
                dry_run,
            ) {
                Ok(e) => Some(e),
                Err(e) => {
                    warn!("linkedin connection-request triage disabled: {e:#}");
                    None
                }
            };
            // #243 — SocialAPI.ai own-post comment engagement. Gates on a
            // SocialAPI.ai key (env/keyring); self-disables with a warning when
            // absent. Inert until `own_posts` has platform="socialapi" rows —
            // same always-spawn-empty-is-free pattern as the LinkedIn engages.
            let socialapi_own_post = match build_socialapi_own_post_engagement(
                &cli,
                Arc::clone(&store),
                Arc::clone(&broker),
                dry_run,
            ) {
                Ok(e) => Some(e),
                Err(e) => {
                    warn!("socialapi own-post comment engagement disabled: {e:#}");
                    None
                }
            };
            // #242 — SocialAPI.ai inbound DM channel. Same key gate as the
            // own-post engagement; inert (polls an empty inbox) until accounts
            // are registered. Stops at the approval card — the reply send is
            // #244.
            let socialapi_dm = match build_socialapi_dm_channel(
                &cli,
                Arc::clone(&store),
                Arc::clone(&broker),
                dry_run,
            ) {
                Ok(c) => Some(Arc::new(c)),
                Err(e) => {
                    warn!("socialapi dm channel disabled: {e:#}");
                    None
                }
            };
            // Discord is optional too — builds only if creds are in Keychain.
            let discord_ch = match build_discord_channel(
                &cli,
                Arc::clone(&store),
                Arc::clone(&broker),
                dry_run,
            ) {
                Ok(ch) => Some(ch),
                Err(e) => {
                    warn!("discord channel disabled: {e:#}");
                    None
                }
            };
            let slack_ch = match build_slack_channel(
                &cli,
                Arc::clone(&store),
                Arc::clone(&broker),
                dry_run,
            ) {
                Ok(ch) => Some(ch),
                Err(e) => {
                    warn!("slack channel disabled: {e:#}");
                    None
                }
            };
            let github_ch = match build_github_channel(
                &cli,
                Arc::clone(&store),
                Arc::clone(&broker),
                dry_run,
            ) {
                Ok(ch) => Some(ch),
                Err(e) => {
                    warn!("github channel disabled: {e:#}");
                    None
                }
            };
            // Deft (#116) spike scaffold: linked but INERT. We deliberately
            // do NOT build/spawn a DeftChannel here — the doc's live-validation
            // TODOs (real submission/webhook JSON, confirmed product, token)
            // must clear first. This line just surfaces the arming-gate state
            // in logs and keeps the dep genuinely used. See
            // docs/deft-protocol.md §6/§7.
            if augmentagent_channel_deft::deft_enabled() {
                warn!(
                    "AUGMENTAGENT_DEFT_ENABLED is set but the deft channel is a \
                     spike scaffold and is intentionally not spawned (see \
                     docs/deft-protocol.md §7 go/no-go)"
                );
            }

            // Meetup self-gates on having ≥1 subscription, exactly like
            // github gates on a PAT — prod's db has none ⇒ never spawned.
            let meetup_ch = match build_meetup_channel(
                &cli,
                Arc::clone(&store),
                Arc::clone(&broker),
                dry_run,
            ) {
                Ok(ch) => Some(ch),
                Err(e) => {
                    warn!("meetup channel disabled: {e:#}");
                    None
                }
            };
            // Drive self-gates on having ≥1 connected account + a Composio
            // key — prod has neither ⇒ never spawned.
            let gdrive_ch = match build_gdrive_channel(
                Arc::clone(&store),
                Arc::clone(&broker),
                dry_run,
            ) {
                Ok(ch) => Some(ch),
                Err(e) => {
                    warn!("gdrive channel disabled: {e:#}");
                    None
                }
            };
            let shutdown = CancellationToken::new();
            let s2 = shutdown.clone();
            tokio::spawn(async move {
                if tokio::signal::ctrl_c().await.is_ok() {
                    info!("SIGINT received");
                    s2.cancel();
                }
            });
            // Collect the enabled channels' runners + optional digest scheduler.
            let mut tasks: Vec<tokio::task::JoinHandle<anyhow::Result<()>>> = Vec::new();

            // Voice-capture listener (#80): long-poll the capture bot. Inert
            // unless a token is in the keyring AND the chat allowlist is
            // non-empty — so prod (neither configured) never spawns it. The
            // dedicated systemd unit is the primary path; this in-process
            // spawn keeps single-host setups simple.
            if let Some(vl) = build_voice_listener(&cli, Arc::clone(&store), dry_run) {
                let sd = shutdown.clone();
                tasks.push(tokio::spawn(async move { vl.run(sd).await }));
            }
            // Proactive CRM runner (#81): 30-min loop over wiki people pages.
            // Only when a wiki is configured (no wiki ⇒ nothing to scan).
            // Dispatch is gated on !dry_run like the other outbound surfaces.
            if let Some(wiki_root) = cli.wiki_dir.clone() {
                let suppression = std::sync::Arc::new(
                    augmentagent_proactive::TableSuppression::new(Arc::clone(&store)),
                );
                let runner = augmentagent_proactive::runner::ProactiveRunner::new(
                    Arc::clone(&store),
                    Arc::clone(&broker),
                    wiki_root,
                    augmentagent_proactive::rules::default_scans(),
                )
                .with_suppression(suppression);
                let sd = shutdown.clone();
                tasks.push(tokio::spawn(async move { runner.run(sd).await }));
            } else {
                info!("proactive runner disabled: --wiki-dir not set");
            }
            // #58 — scheduled-post fire loop. 1-min serve tick: T-30min
            // preview card → T-0 publish via the per-platform poster, every
            // step gated by the merged RateGovernor (#83). Inert until a
            // post is queued (`augmentagent schedule-post add …`), so this
            // is safe to always spawn — empty queue == zero-cost tick. The
            // engine itself honours dry_run (mutes cards + real publish).
            {
                let governor: Arc<dyn augmentagent_channel_core::RateGovernor> =
                    Arc::new(
                        augmentagent_channel_core::SqliteGovernor::with_system_clock(
                            Arc::clone(&store),
                        ),
                    );
                let publisher: Arc<dyn augmentagent_channel_core::PostPublisher> =
                    Arc::new(MultiPlatformPublisher {
                        store: Arc::clone(&store),
                        repo_root: std::env::current_dir()
                            .unwrap_or_else(|_| PathBuf::from(".")),
                        dry_run,
                    });
                let engine = augmentagent_channel_core::ScheduledPostEngine::new(
                    Arc::clone(&store),
                    Arc::clone(&broker),
                    governor,
                    publisher,
                    dry_run,
                );
                let sd = shutdown.clone();
                tasks.push(tokio::spawn(async move { engine.run(sd).await }));
            }
            // #500 — scheduled email send fire loop. Fires `scheduled`
            // actions when `scheduledAtMs` comes due: CAS claim → Composio
            // send_draft → the same #449 bookkeeping order as Approve.
            // Always spawned when Composio is configured — an empty schedule
            // is a zero-cost tick, and the engine honours dry_run (logs
            // would-fire lines, sends nothing, leaves rows armed). Also owns
            // flagging rows stuck mid-send from a previous crash.
            match std::env::var("COMPOSIO_API_KEY") {
                Ok(api_key) => {
                    let engine = augmentagent_channel_email::ScheduledSendEngine::new(
                        Arc::clone(&store),
                        Arc::new(ComposioClient::new(api_key)),
                        Arc::clone(&broker),
                        dry_run,
                    )
                    .with_tick(Duration::from_secs(
                        scheduled_send_interval_secs_from_env(),
                    ));
                    let sd = shutdown.clone();
                    tasks.push(tokio::spawn(async move { engine.run(sd).await }));
                }
                Err(_) => warn!(
                    "COMPOSIO_API_KEY not set; scheduled email sends will not fire"
                ),
            }
            // #25: Gmail + LinkedIn now run through the generic
            // `ChannelRunner` (`run_arc`) instead of bespoke poll loops.
            // Behavior is unchanged — `run_arc` drives the same per-message
            // `process_email` pipeline via a `WorkItemHandler`; Gmail keeps
            // its independent retry ticker, LinkedIn keeps its 4h±10min
            // jittered cadence.
            if let Some(gmail_ch) = gmail_ch {
                let sd = shutdown.clone();
                let gmail_arc = Arc::new(gmail_ch);
                tasks.push(tokio::spawn(async move { gmail_arc.run_arc(sd).await }));
            }
            if let Some(li) = linkedin_ch {
                let sd = shutdown.clone();
                let li_arc = Arc::new(li);
                tasks.push(tokio::spawn(async move { li_arc.run_arc(sd).await }));
            }
            if let Some(lf) = linkedin_feed {
                let sd = shutdown.clone();
                tasks.push(tokio::spawn(async move { lf.run(sd).await }));
            }
            if let Some(op) = own_post_engagement {
                let sd = shutdown.clone();
                tasks.push(tokio::spawn(async move { op.run(sd).await }));
            }
            if let Some(ff) = friend_feed_engagement {
                let sd = shutdown.clone();
                tasks.push(tokio::spawn(async move { ff.run(sd).await }));
            }
            if let Some(ct) = connection_triage {
                let sd = shutdown.clone();
                tasks.push(tokio::spawn(async move { ct.run(sd).await }));
            }
            if let Some(sa) = socialapi_own_post {
                let sd = shutdown.clone();
                tasks.push(tokio::spawn(async move { sa.run(sd).await }));
            }
            if let Some(sdm) = socialapi_dm {
                let sd = shutdown.clone();
                tasks.push(tokio::spawn(async move { sdm.run(sd).await }));
            }
            if let Some(dc) = discord_ch {
                let sd = shutdown.clone();
                tasks.push(tokio::spawn(async move { dc.run(sd).await }));
                // Digest scheduler rides alongside the Discord channel when
                // Discord is enabled. Skips cleanly when no Digest-mode subs.
                let digest = augmentagent_channel_discord_dm::digest::DigestScheduler::new(
                    Arc::clone(&store),
                    build_reasoner(),
                    Arc::clone(&broker),
                    cli.wiki_dir.clone(),
                );
                let sd = shutdown.clone();
                tasks.push(tokio::spawn(async move { digest.run(sd).await }));
            }
            if let Some(sc) = slack_ch {
                let sd = shutdown.clone();
                tasks.push(tokio::spawn(async move { sc.run(sd).await }));
                // Slack workspace digest (#8). Rides alongside the Slack
                // channel exactly like the Discord digest does — the shared
                // scheduler is pinned to platform="slack" and skips cleanly
                // when there are no Digest-mode Slack subscriptions.
                let slack_digest = augmentagent_channel_slack::slack_digest_scheduler(
                    Arc::clone(&store),
                    build_reasoner(),
                    Arc::clone(&broker),
                    cli.wiki_dir.clone(),
                );
                let sd = shutdown.clone();
                tasks.push(tokio::spawn(async move { slack_digest.run(sd).await }));
            }
            if let Some(gh) = github_ch {
                let sd = shutdown.clone();
                tasks.push(tokio::spawn(async move { gh.run(sd).await }));
            }
            if let Some(mc) = meetup_ch {
                let sd = shutdown.clone();
                tasks.push(tokio::spawn(async move { mc.run(sd).await }));
            }
            if let Some(gd) = gdrive_ch {
                let sd = shutdown.clone();
                tasks.push(tokio::spawn(async move { gd.run(sd).await }));
            }
            // #630 — auto-PR loop: poll for open `agent-fixable` issues and
            // run the #103 self-improve pipeline (draft PR, never merge)
            // unattended. Opt-in via AUGMENTAGENT_AUTOPR=1 — every engaged
            // run spawns the claude CLI on the owner's subscription (#448),
            // so it must never start billing as a side effect of a deploy.
            // Serve's --dry-run flows through: the gate runs but no PR opens.
            match self_improve::AutoPrLoop::from_env(
                std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
                dry_run,
            ) {
                Some(ap) => {
                    let sd = shutdown.clone();
                    tasks.push(tokio::spawn(async move { ap.run(sd).await }));
                }
                None => info!("auto-PR loop disabled: AUGMENTAGENT_AUTOPR not set"),
            }
            // #427 — ShadowNote journal → wiki ingest. Self-gates on the
            // SHADOWNOTE_* keyring/env config (plus AWS creds via the SDK
            // default chain); an unconfigured box logs and moves on, same
            // as the reddit/github gates.
            match augmentagent_channel_journal::JournalRuntime::from_env().await {
                Ok(Some(runtime)) => {
                    let jc = augmentagent_channel_journal::JournalChannel::new(
                        Arc::clone(&store),
                        Arc::clone(&runtime.client),
                        runtime.dek.clone(),
                        build_reasoner(),
                        augmentagent_channel_journal::JournalChannelConfig {
                            owner_id: runtime.config.owner_id.clone(),
                            dry_run,
                            wiki_root: cli.wiki_dir.clone(),
                            wiki_schema_path: cli
                                .wiki_dir
                                .as_ref()
                                .map(|_| PathBuf::from("schema/wiki-skill.md")),
                            poll_interval: augmentagent_channel_journal::DEFAULT_POLL_INTERVAL,
                            max_entries_per_poll:
                                augmentagent_channel_journal::DEFAULT_MAX_ENTRIES_PER_POLL,
                            base_sync_threshold:
                                augmentagent_channel_journal::DEFAULT_BASE_SYNC_THRESHOLD,
                            allow_base_sync: false,
                            max_pages_per_poll:
                                augmentagent_channel_journal::DEFAULT_MAX_PAGES_PER_POLL,
                        },
                    );
                    let sd = shutdown.clone();
                    tasks.push(tokio::spawn(async move { jc.run(sd).await }));
                }
                Ok(None) => {
                    info!("shadownote journal channel disabled: SHADOWNOTE_* config not present");
                }
                Err(e) => {
                    warn!("shadownote journal channel disabled: {e:#}");
                }
            }
            // #886 — iMessage bundle → emails + wiki ingest. Self-gates on
            // AUGMENTAGENT_IMESSAGE_REPO_DIR; an unconfigured box logs and
            // moves on, same as the journal gate above.
            match augmentagent_channel_imessage::ImessageConfig::load() {
                Some(imcfg) => {
                    let store_c = Arc::clone(&store);
                    let reasoner = build_reasoner();
                    let wiki_root = cli.wiki_dir.clone();
                    let wiki_schema = wiki_root
                        .as_ref()
                        .and_then(|_| std::fs::read_to_string("schema/wiki-skill.md").ok());
                    let sd = shutdown.clone();
                    tasks.push(tokio::spawn(async move {
                        imessage_poll_loop(imcfg, store_c, reasoner, wiki_root, wiki_schema, sd)
                            .await
                    }));
                }
                None => {
                    info!("imessage channel disabled: AUGMENTAGENT_IMESSAGE_REPO_DIR not set");
                }
            }
            // #48 — Reddit channel. Self-gates on having completed the
            // dashboard OAuth bootstrap (refresh token in keyring); prod
            // without it never spawns this, exactly like github/meetup gate.
            match augmentagent_channel_reddit::RedditChannel::from_keychain() {
                Ok(rc) => {
                    let rc = Arc::new(rc);
                    let sd = shutdown.clone();
                    tasks.push(tokio::spawn(async move { rc.run(sd).await }));
                }
                Err(e) => {
                    info!("reddit channel disabled: {e:#}");
                }
            }
            // Nudge scheduler — surfaces pending approval cards one at a time
            // (serial queue). Cross-channel: any pending action (gmail /
            // linkedin / discord / slack) is eligible. The approver holds a
            // Weak ref back to the scheduler so resolve handlers can advance
            // the queue instantly on approve/skip without waiting for the
            // next tick. Skipped under dry-run (NoopBroker) — bumping
            // counters with no visible card is pointless.
            if !dry_run {
                let nudge = Arc::new(augmentagent_approval_discord::NudgeScheduler::new(
                    Arc::clone(&store),
                    Arc::clone(&broker),
                ));
                if let Some(ref approver) = approver {
                    approver
                        .nudge
                        .set(Arc::downgrade(&nudge))
                        .ok();
                }
                let sd = shutdown.clone();
                let nudge_for_task = Arc::clone(&nudge);
                tasks.push(tokio::spawn(async move { nudge_for_task.run(sd).await }));

                // #104 — /loop scheduled-task scheduler. Runs each due loop's
                // stored prompt through the wiki-ask reasoner and posts the
                // result back to the originating Discord channel/DM. Requires
                // a bot token (for the post-back HTTP client) and a wiki dir
                // (the reasoner toolbelt is scoped to it); skips with a log
                // otherwise. Gated on !dry_run alongside the other schedulers.
                match (
                    std::env::var("DISCORD_BOT_TOKEN").ok(),
                    cli.wiki_dir.clone(),
                ) {
                    (Some(token), Some(wiki_root)) => {
                        let repo_root = std::env::current_dir()
                            .unwrap_or_else(|_| PathBuf::from("."));
                        let runner = Arc::new(LoopReasonerRunner {
                            reasoner: build_reasoner(),
                            wiki_root,
                            repo_root,
                        });
                        let poster = Arc::new(DiscordLoopPoster {
                            http: Arc::new(serenity::http::Http::new(&token)),
                        });
                        let loops = Arc::new(LoopScheduler::new(
                            Arc::clone(&store),
                            runner,
                            poster,
                        ));
                        let sd = shutdown.clone();
                        tasks.push(tokio::spawn(async move { loops.run(sd).await }));
                    }
                    _ => {
                        info!(
                            "/loop scheduler disabled (needs DISCORD_BOT_TOKEN                              + --wiki-dir)"
                        );
                    }
                }
                // Approval auto-expire sweep (#99 / #220): periodically
                // expire pending approvals older than
                // AUGMENTAGENT_APPROVAL_AUTO_EXPIRE_DAYS (default 7d,
                // `0` = disabled, legacy alias
                // AUGMENTAGENT_STALE_DRAFT_DAYS) to `timed_out`, so an
                // abandoned backlog can't sit forever blocking new triage
                // via backpressure (#99). Tick interval is
                // AUGMENTAGENT_APPROVAL_AUTO_EXPIRE_INTERVAL_SECS
                // (default 3600 = 1h). Per-id audit lines (#220) land in
                // stdout for grep-ability. NOTE: the Discord card for
                // each expired id is NOT edited/deleted in this pass —
                // that hygiene pass is a deliberate follow-on; we ship
                // the sweep first.
                let sweep_store = Arc::clone(&store);
                let sd = shutdown.clone();
                tasks.push(tokio::spawn(async move {
                    run_stale_draft_sweep(sweep_store, sd).await
                }));

                // #219/#449 — outbound observer: when the user replies via
                // Gmail web/mobile, the pending approval card on that thread is
                // stale; this task supersedes those drafts on a 5-min tick.
                //
                // DEFAULT-ON as of #449. It shipped default-off because
                // `classify_outbound` could not recognize the daemon's own
                // sends (it compared SENT-folder ids against `actions.messageId`,
                // which holds *inbound* ids), so turning it on would have
                // recorded every agent reply as a manual user reply. That is
                // fixed: sends are now logged to `self_sent_messages` from the
                // id Gmail returns, so the observer can safely run for real.
                //
                // Leaving it off is what made the approval queue accumulate
                // cards for mail the user had already answered — the carousel
                // staleness in #449. Opt out with
                // AUGMENTAGENT_OUTBOUND_OBSERVER=0. Self-disables on
                // Gmail-less ("--no-email") tenants.
                let outbound_enabled = std::env::var("AUGMENTAGENT_OUTBOUND_OBSERVER")
                    .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false")))
                    .unwrap_or(true);
                if outbound_enabled && !no_email {
                    let obs_store = Arc::clone(&store);
                    let sd = shutdown.clone();
                    tasks.push(tokio::spawn(async move {
                        run_outbound_observer(obs_store, sd).await
                    }));
                } else if outbound_enabled && no_email {
                    info!(
                        "outbound observer enabled but --no-email is set; \
                         observer is Gmail-only and will NOT be started"
                    );
                } else {
                    info!(
                        "outbound observer disabled via AUGMENTAGENT_OUTBOUND_OBSERVER=0; \
                         approval cards will NOT be retired when you reply from Gmail \
                         web/mobile"
                    );
                }

                // #449 — staleness reconciliation. The user should never have
                // to tell us a card is stale; the daemon should already know.
                // Runs once at startup (to clear whatever accumulated while the
                // observer was off) and then on a slow tick.
                let reconcile_store = Arc::clone(&store);
                let sd = shutdown.clone();
                tasks.push(tokio::spawn(async move {
                    run_stale_approval_reconcile(reconcile_store, sd).await
                }));
            }

            // Self-healing: backfill any connected-Gmail addresses Composio
            // never surfaced on the connection, so the dashboard + invoice
            // entity picker show real emails. Detached + best-effort — a
            // flaky lookup must never take the daemon down, so this is
            // deliberately not awaited in `tasks`. Skipped for non-email
            // tenants (no Gmail accounts to backfill; avoids a needless
            // Composio call + warn log).
            if !no_email {
                let store_bf = Arc::clone(&store);
                tokio::spawn(async move {
                    match backfill_gmail_emails(&store_bf, true).await {
                        Ok(lines) if !lines.is_empty() => {
                            info!(updated = lines.len(), "gmail email backfill: {lines:?}");
                        }
                        Ok(_) => {}
                        Err(e) => warn!("gmail email backfill skipped: {e:#}"),
                    }
                });
            }
            for handle in tasks {
                handle.await??;
            }
            Ok(())
        }
        Cmd::Transcripts { ref op } => match op {
            TranscriptsOp::Sync {
                dry_run,
                no_pull,
                require_disclosed,
            } => {
                run_transcripts_sync(
                    &cli,
                    Arc::clone(&store),
                    *dry_run,
                    *no_pull,
                    *require_disclosed,
                )
                .await
            }
        },
        Cmd::Wiki { ref op } => match op {
            WikiOp::Lint { out } => run_wiki_lint(&cli, Arc::clone(&store), out.clone()).await,
            WikiOp::Ask { question, post } => run_wiki_ask(&cli, question.clone(), *post).await,
            WikiOp::Migrate {
                to,
                dry_run,
                concurrency,
                limit,
                branch,
                force,
            } => {
                run_wiki_migrate(
                    &cli,
                    to.clone(),
                    *dry_run,
                    *concurrency,
                    *limit,
                    branch.clone(),
                    *force,
                )
                .await
            }
            WikiOp::Sync { dry_run, no_pull } => run_wiki_sync(&cli, *dry_run, *no_pull).await,
            WikiOp::Index { rebuild } => run_wiki_index(&cli, Arc::clone(&store), *rebuild),
        },
        Cmd::Digest {
            since,
            post_discord,
        } => run_digest(&cli, store, since, post_discord).await,
        Cmd::Research {
            since_hours,
            post_discord,
            dry_run,
            max_issues,
        } => research::run_research(store, since_hours, post_discord, dry_run, max_issues).await,
        Cmd::Gmail { ref op } => match op {
            GmailOp::Search { query, limit, full, account } => {
                run_gmail_search(store, query.clone(), *limit, *full, account.clone()).await
            }
            GmailOp::Accounts { json } => run_gmail_accounts(store, *json).await,
            GmailOp::ListAttachments { account, message_id, json } => {
                gmail_attach::run_gmail_list_attachments(
                    store,
                    account.clone(),
                    message_id.clone(),
                    *json,
                )
                .await
            }
            GmailOp::GetAttachment {
                account, message_id, attachment_id, name, out, extract, json,
            } => {
                gmail_attach::run_gmail_get_attachment(
                    store,
                    account.clone(),
                    message_id.clone(),
                    attachment_id.clone(),
                    name.clone(),
                    out.clone(),
                    *extract,
                    *json,
                )
                .await
            }
            GmailOp::Compose {
                account, to, cc, bcc, subject, body, body_file, thread_id, json,
                post, allow_duplicate, attach, reply_to_message_id, reply_to_from,
                reply_to_subject, reply_to_body, reply_to_body_file, send_at,
            } => {
                run_gmail_compose(
                    store,
                    account.clone(),
                    to.clone(),
                    cc.clone(),
                    bcc.clone(),
                    subject.clone(),
                    body.clone(),
                    body_file.clone(),
                    thread_id.clone(),
                    *json,
                    *post,
                    *allow_duplicate,
                    attach.clone(),
                    reply_to_message_id.clone(),
                    reply_to_from.clone(),
                    reply_to_subject.clone(),
                    reply_to_body.clone(),
                    reply_to_body_file.clone(),
                    send_at.clone(),
                )
                .await
            }
            GmailOp::UpdateDraft {
                account, draft_id, to, cc, bcc, subject, body, body_file, thread_id, attach,
            } => {
                run_gmail_update_draft(
                    store,
                    account.clone(),
                    draft_id.clone(),
                    to.clone(),
                    cc.clone(),
                    bcc.clone(),
                    subject.clone(),
                    body.clone(),
                    body_file.clone(),
                    thread_id.clone(),
                    attach.clone(),
                )
                .await
            }
            GmailOp::Send { account, draft_id } => {
                run_gmail_send_draft(store, account.clone(), draft_id.clone()).await
            }
            GmailOp::DeleteDraft { account, draft_id } => {
                run_gmail_delete_draft(store, account.clone(), draft_id.clone()).await
            }
            GmailOp::SendNow {
                account, to, cc, bcc, subject, body, body_file, thread_id, attach,
            } => {
                run_gmail_send_now(
                    store,
                    account.clone(),
                    to.clone(),
                    cc.clone(),
                    bcc.clone(),
                    subject.clone(),
                    body.clone(),
                    body_file.clone(),
                    thread_id.clone(),
                    attach.clone(),
                )
                .await
            }
        },
        Cmd::Doc { ref op } => match op {
            DocOp::Extract { file, out, kind, no_ocr, json } => {
                doc_cmd::run_doc_extract(file.clone(), out.clone(), kind.clone(), *no_ocr, *json)
                    .await
            }
        },
        Cmd::Resume { ref op } => match op {
            ResumeOp::Ingest { file } => run_resume_ingest(&cli, file.clone()).await,
        },
        Cmd::Linkedin { ref op } => match op {
            LinkedinOp::Login { cookies_json } => run_linkedin_login(cookies_json.clone()).await,
            LinkedinOp::PollOnce { dry_run } => {
                let (broker, _) = build_broker(&cli, Arc::clone(&store), *dry_run).await?;
                let ch = build_linkedin_channel(&cli, store, broker, *dry_run)?;
                let out = ch.poll_once().await?;
                println!("{out:#?}");
                Ok(())
            }
            LinkedinOp::Recent => run_linkedin_recent().await,
            LinkedinOp::ConnectionsSync { apply, full } => {
                let (broker, _) = build_broker(&cli, Arc::clone(&store), !apply).await?;
                run_linkedin_connections_sync(&cli, store, broker, *apply, *full).await
            }
            LinkedinOp::RecentDms { limit, json } => {
                let repo_root =
                    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
                run_linkedin_recent_dms(repo_root, *limit, *json).await
            }
            LinkedinOp::Dm {
                conversation_urn,
                with,
                body,
                body_file,
                in_reply_to,
                post,
                json,
            } => {
                run_linkedin_dm(
                    Arc::clone(&store),
                    conversation_urn.clone(),
                    with.clone(),
                    body.clone(),
                    body_file.clone(),
                    in_reply_to.clone(),
                    *post,
                    *json,
                )
                .await
            }
            LinkedinOp::Comment {
                post_urn,
                author,
                body,
                body_file,
                in_reply_to,
                post,
                json,
            } => {
                run_linkedin_comment(
                    Arc::clone(&store),
                    post_urn.clone(),
                    author.clone(),
                    body.clone(),
                    body_file.clone(),
                    in_reply_to.clone(),
                    *post,
                    *json,
                )
                .await
            }
            LinkedinOp::Post {
                text,
                images,
                visibility,
                dry_run,
            } => {
                run_linkedin_post(
                    Arc::clone(&store),
                    text.clone(),
                    images.clone(),
                    visibility.clone(),
                    *dry_run,
                )
                .await
            }
        },
        Cmd::Twitter { ref op } => match op {
            TwitterOp::Login { session_json } => {
                run_twitter_login(session_json.clone()).await
            }
            TwitterOp::Post {
                text,
                reply_to,
                dry_run,
            } => {
                run_twitter_post(store, text.clone(), reply_to.clone(), *dry_run).await
            }
            TwitterOp::PollOnce => run_twitter_poll_once(&cli).await,
            TwitterOp::Validate {
                json,
                allow_live,
                allow_write,
                probe_reply_to,
                probe_conversation_id,
            } => {
                run_twitter_validate(
                    *json,
                    *allow_live,
                    *allow_write,
                    probe_reply_to.clone(),
                    probe_conversation_id.clone(),
                )
                .await
            }
        },
        Cmd::Slack { ref op } => match op {
            SlackOp::Login { auth_json } => run_slack_login(store, auth_json.clone()).await,
            SlackOp::PersistAuth {
                entity_id,
                connection_id,
                composio_api_key,
            } => run_slack_persist_auth(
                store,
                entity_id.clone(),
                connection_id.clone(),
                composio_api_key.clone(),
            )
            .await,
            SlackOp::Workspaces { json } => run_slack_workspaces(store, *json),
            SlackOp::RemoveWorkspace { team_id } => {
                run_slack_remove_workspace(store, team_id.clone())
            }
            SlackOp::Reset { confirm } => run_slack_reset(store, *confirm),
            SlackOp::ListConversations { team_id, types, limit, json } => {
                run_slack_list_conversations(store, team_id.clone(), types.clone(), *limit, *json).await
            }
            SlackOp::Subscribe { channel_id, mode, name, team_id } => {
                run_slack_subscribe(store, channel_id.clone(), mode.clone(), name.clone(), team_id.clone())
            }
            SlackOp::Subscriptions { json } => run_slack_subscriptions(store, *json),
            SlackOp::Unsubscribe { id } => run_slack_unsubscribe(store, id.clone()),
            SlackOp::PollOnce { dry_run } => {
                let (broker, _) = build_broker(&cli, Arc::clone(&store), *dry_run).await?;
                let ch = build_slack_channel(&cli, store, broker, *dry_run)?;
                let out = ch.poll_once().await?;
                println!("{out:#?}");
                Ok(())
            }
        },
        Cmd::Discord { ref op } => match op {
            DiscordOp::Login { creds_json } => run_discord_login(creds_json.clone()).await,
            DiscordOp::Status { json } => run_discord_status(*json).await,
            DiscordOp::ListDms { json } => run_discord_list_dms(*json).await,
            DiscordOp::ListGuilds { json } => run_discord_list_guilds(*json).await,
            DiscordOp::ListGuildChannels { guild_id, json } => {
                run_discord_list_guild_channels(guild_id.clone(), *json).await
            }
            DiscordOp::Subscribe { channel_id, mode, name } => {
                run_discord_subscribe(store, channel_id.clone(), mode.clone(), name.clone())
            }
            DiscordOp::Subscriptions { json } => run_discord_subscriptions(store, *json),
            DiscordOp::Unsubscribe { id } => run_discord_unsubscribe(store, id.clone()),
            DiscordOp::PollOnce { dry_run } => {
                let (broker, _) = build_broker(&cli, Arc::clone(&store), *dry_run).await?;
                let ch = build_discord_channel(&cli, store, broker, *dry_run)?;
                let out = ch.poll_once().await?;
                println!("{out:#?}");
                Ok(())
            }
        },
        // ----- wave-A foundation stubs --------------------------------
        // Each arm calls unimplemented! pointing at the relevant issue so
        // feature PRs know exactly which arm to fill.
        Cmd::TelegramBot { ref op } => match op {
            TelegramBotOp::Login { token } => {
                run_telegram_bot_login(store, token.clone()).await
            }
            TelegramBotOp::Bots { json } => run_telegram_bot_bots(store, *json),
            TelegramBotOp::RemoveBot { bot_username } => {
                run_telegram_bot_remove(store, bot_username.clone())
            }
            TelegramBotOp::ListChats { bot_username, json } => {
                run_telegram_bot_list_chats(store, bot_username.clone(), *json)
            }
            TelegramBotOp::Subscribe {
                chat_id,
                mode,
                name,
                bot_username,
            } => run_telegram_bot_subscribe(
                store,
                chat_id.clone(),
                mode.clone(),
                name.clone(),
                bot_username.clone(),
            ),
            TelegramBotOp::Subscriptions { json } => {
                run_telegram_bot_subscriptions(store, *json)
            }
            TelegramBotOp::Unsubscribe { id } => {
                run_telegram_bot_unsubscribe(store, id.clone())
            }
            TelegramBotOp::PollOnce { dry_run } => {
                let (broker, _) = build_broker(&cli, Arc::clone(&store), *dry_run).await?;
                let ch = build_telegram_bot_channel(&cli, store, broker, *dry_run)?;
                let out = ch.poll_once().await?;
                println!("{out:#?}");
                Ok(())
            }
        },
        Cmd::Whatsapp { op } => match op {
            WhatsappOp::Login { .. }
            | WhatsappOp::Status { .. }
            | WhatsappOp::Devices { .. }
            | WhatsappOp::Unlink { .. }
            | WhatsappOp::ListChats { .. }
            | WhatsappOp::Subscribe { .. }
            | WhatsappOp::Subscriptions { .. }
            | WhatsappOp::Unsubscribe { .. }
            | WhatsappOp::AllowOutbound { .. }
            | WhatsappOp::DenyOutbound { .. }
            | WhatsappOp::AllowInbound { .. }
            | WhatsappOp::DenyInbound { .. }
            | WhatsappOp::PollOnce { .. } => {
                unimplemented!("see issue #74 (whatsapp feature PR)")
            }
        },
        Cmd::Journal { op } => match op {
            JournalOp::PollOnce { dry_run } => {
                run_journal_poll_once(cli.wiki_dir.clone(), store, dry_run, None, false).await?;
                Ok(())
            }
            JournalOp::Backfill {
                max_entries,
                dry_run,
            } => {
                run_journal_poll_once(
                    cli.wiki_dir.clone(),
                    store,
                    dry_run,
                    Some(max_entries),
                    true,
                )
                .await?;
                Ok(())
            }
            JournalOp::SkipToNow => {
                run_journal_skip_to_now(store).await?;
                Ok(())
            }
            JournalOp::Status => {
                run_journal_status(store).await?;
                Ok(())
            }
        },
        Cmd::Person { ref op } => match op {
            PersonOp::Merge { from, into, apply } => {
                run_person_merge(&cli, store, from, into, *apply)?;
                Ok(())
            }
            PersonOp::ProposeMerges => propose_high_confidence_merges(&cli, &store).await,
        },
        Cmd::Imessage { ref op } => match op {
            ImessageOp::Sync { apply } => {
                run_imessage_sync(&cli, store, *apply)?;
                Ok(())
            }
            ImessageOp::PollOnce => {
                run_imessage_poll_once(store)?;
                Ok(())
            }
        },
        Cmd::Calendar { op } => match op {
            CalendarOp::Backfill { .. } => {
                anyhow::bail!(
                    "calendar backfill is Phase 2 — see issue #400 (scope ported from archived AugmentAgent#82 §12)"
                )
            }
            CalendarOp::PollOnce { dry_run } => {
                run_calendar_poll_once(cli.wiki_dir.clone(), store, dry_run).await?;
                Ok(())
            }
            CalendarOp::Subscriptions { json } => {
                run_calendar_subscriptions(store, json)?;
                Ok(())
            }
            CalendarOp::ListEvents {
                from,
                to,
                days,
                json,
            } => {
                run_calendar_list_events(store, from, to, days, json).await?;
                Ok(())
            }
            CalendarOp::CreateEvent {
                summary,
                start,
                duration_min,
                attendees,
                description,
                meet,
                account,
                post,
            } => {
                run_calendar_create_event(
                    store,
                    summary,
                    start,
                    duration_min,
                    attendees,
                    description,
                    meet,
                    account,
                    post,
                )
                .await?;
                Ok(())
            }
        },
        Cmd::Voice { ref op } => match op {
            VoiceOp::Login { token } => run_voice_login(token.clone()),
            VoiceOp::PollOnce { dry_run } => {
                run_voice_poll_once(&cli, Arc::clone(&store), *dry_run).await
            }
            VoiceOp::Serve { dry_run } => {
                run_voice_serve(&cli, Arc::clone(&store), *dry_run).await
            }
        },
        Cmd::Reddit { ref op } => match op {
            RedditOp::AuthUrl {
                client_id,
                redirect_uri,
                state,
            } => {
                println!(
                    "{}",
                    augmentagent_channel_reddit::authorize_url(
                        client_id,
                        redirect_uri,
                        state
                    )
                );
                Ok(())
            }
            RedditOp::Exchange {
                client_id,
                code,
                redirect_uri,
            } => {
                let creds = augmentagent_channel_reddit::exchange_code(
                    client_id,
                    code,
                    redirect_uri,
                )
                .await
                .context("reddit code exchange")?;
                augmentagent_channel_reddit::RedditAuth::save(&creds)
                    .context("persist reddit creds")?;
                println!("{{\"ok\":true}}");
                Ok(())
            }
        },
        Cmd::Socialapi { ref op } => match op {
            SocialapiOp::List { json } => {
                let accounts = store
                    .list_socialapi_accounts()
                    .context("list socialapi accounts")?;
                if *json {
                    let arr: Vec<serde_json::Value> = accounts
                        .iter()
                        .map(|a| {
                            serde_json::json!({
                                "id": a.id,
                                "brand_id": a.brand_id,
                                "platform": a.platform,
                                "display_name": a.display_name,
                                "account_handle": a.account_handle,
                                "active": a.active,
                            })
                        })
                        .collect();
                    println!("{}", serde_json::to_string_pretty(&arr)?);
                } else if accounts.is_empty() {
                    println!("(no socialapi accounts)");
                } else {
                    for a in &accounts {
                        println!(
                            "{}\tplatform={}\thandle={}\tactive={}",
                            a.id,
                            a.platform,
                            a.account_handle.as_deref().unwrap_or("-"),
                            a.active
                        );
                    }
                }
                Ok(())
            }
            SocialapiOp::Disable { account_id } => {
                let now_ms = chrono::Utc::now().timestamp_millis();
                let n = store
                    .set_socialapi_account_active(account_id, false, now_ms)
                    .context("disable socialapi account")?;
                if n == 0 {
                    println!(
                        "{}",
                        serde_json::json!({"ok": false, "error": "unknown account_id", "account_id": account_id})
                    );
                } else {
                    println!(
                        "{}",
                        serde_json::json!({"ok": true, "account_id": account_id, "active": false})
                    );
                }
                Ok(())
            }
            SocialapiOp::Connect {
                timeout_secs,
                open_browser,
            } => {
                // Same path as `setup oauth --provider socialapi`: drive the
                // shared OAuth runner against the dashboard's proxied flow.
                let args = setup::oauth::OauthArgs {
                    provider: setup::oauth::OauthProvider::Socialapi,
                    timeout_secs: *timeout_secs,
                    open_browser: *open_browser,
                    json: true,
                };
                setup::oauth::run(&args).await
            }
            SocialapiOp::Dm {
                conversation_id,
                account_id,
                with,
                platform,
                body,
                body_file,
                in_reply_to,
                post,
                json,
            } => {
                run_socialapi_dm(
                    Arc::clone(&store),
                    conversation_id.clone(),
                    account_id.clone(),
                    with.clone(),
                    platform.clone(),
                    body.clone(),
                    body_file.clone(),
                    in_reply_to.clone(),
                    *post,
                    *json,
                )
                .await
            }
            SocialapiOp::Comment {
                post_id,
                comment_id,
                account_id,
                author,
                platform,
                body,
                body_file,
                in_reply_to,
                post,
                json,
            } => {
                run_socialapi_comment(
                    Arc::clone(&store),
                    post_id.clone(),
                    comment_id.clone(),
                    account_id.clone(),
                    author.clone(),
                    platform.clone(),
                    body.clone(),
                    body_file.clone(),
                    in_reply_to.clone(),
                    *post,
                    *json,
                )
                .await
            }
        },
        Cmd::Github { ref op } => match op {
            GithubOp::Login { token, login } => {
                run_github_login(token.clone(), login.clone()).await
            }
            GithubOp::Subscribe { repo, mode } => {
                run_github_subscribe(store, repo.clone(), mode.clone())
            }
            GithubOp::Subscriptions { json } => run_github_subscriptions(store, *json),
            GithubOp::Unsubscribe { id } => run_github_unsubscribe(store, id.clone()),
            GithubOp::PollOnce { dry_run } => {
                let (broker, _) = build_broker(&cli, Arc::clone(&store), *dry_run).await?;
                let ch = build_github_channel(&cli, store, broker, *dry_run)?;
                let out = ch.poll_once().await?;
                println!("{out:#?}");
                Ok(())
            }
        },
        Cmd::Deft { ref op } => match op {
            DeftOp::Login { token, base_url } => {
                run_deft_login(
                    token.clone(),
                    base_url.clone(),
                    &mut StdinTokenReader,
                )
                .await
            }
            DeftOp::Status {
                workspace_id,
                offline,
                json,
            } => run_deft_status(workspace_id.clone(), *offline, *json).await,
            DeftOp::Logout { workspace_id } => run_deft_logout(workspace_id.clone()),
        },
        Cmd::Meetup { ref op } => match op {
            MeetupOp::Subscribe { urlname, mode } => {
                run_meetup_subscribe(store, urlname.clone(), mode.clone())
            }
            MeetupOp::Subscriptions { json } => run_meetup_subscriptions(store, *json),
            MeetupOp::Unsubscribe { id } => run_meetup_unsubscribe(store, id.clone()),
            MeetupOp::PollOnce { dry_run } => {
                let (broker, _) = build_broker(&cli, Arc::clone(&store), *dry_run).await?;
                let ch = build_meetup_channel(&cli, store, broker, *dry_run)?;
                let out = ch.poll_once().await?;
                println!("{out:#?}");
                Ok(())
            }
            MeetupOp::Events {
                urlname,
                limit,
                json,
            } => run_meetup_events(urlname.clone(), *limit, *json).await,
        },
        Cmd::Gdrive { ref op } => match op {
            GdriveOp::Accounts { json } => run_gdrive_accounts(store, *json),
            GdriveOp::PollOnce { dry_run } => {
                let (broker, _) = build_broker(&cli, Arc::clone(&store), *dry_run).await?;
                let ch = build_gdrive_channel(store, broker, *dry_run)?;
                let out = ch.poll_once().await?;
                println!("{out:#?}");
                Ok(())
            }
        },
        Cmd::Contacts { ref op } => match op {
            ContactsOp::Sync {
                backend,
                entity_id,
                apply,
            } => {
                let (broker, _) = build_broker(&cli, Arc::clone(&store), !apply).await?;
                run_contacts_sync(&cli, store, broker, backend, entity_id, *apply)
                    .await
            }
        },
        Cmd::Compose { ref op } => match op {
            ComposeOp::FanOut {
                source,
                platforms,
                at,
                dry_run,
            } => {
                run_compose_fan_out(
                    &cli,
                    Arc::clone(&store),
                    source.clone(),
                    platforms.clone(),
                    at.clone(),
                    *dry_run,
                )
                .await
            }
        },
        Cmd::Proactive { ref op } => match op {
            ProactiveOp::ScanOnce { dry_run, force } => {
                run_proactive_scan_once(
                    &cli,
                    Arc::clone(&store),
                    *dry_run,
                    *force,
                )
                .await
            }
            ProactiveOp::Signals { limit, json } => {
                run_proactive_signals(Arc::clone(&store), *limit, *json)
            }
            ProactiveOp::Snooze { id, days } => {
                run_proactive_snooze(Arc::clone(&store), id.clone(), *days)
            }
            ProactiveOp::Dismiss { id } => {
                run_proactive_dismiss(Arc::clone(&store), id.clone())
            }
        },
        Cmd::Browser { op } => match op {
            BrowserOp::Start => run_browser_start().await,
            BrowserOp::Stop => run_browser_stop().await,
            BrowserOp::Status { json } => run_browser_status(json).await,
            BrowserOp::AcceptanceTest { out } => run_browser_acceptance(out).await,
            BrowserOp::ImportCookies { .. } => {
                // Out of scope for #75 v0 — cookie jar lands later (depends
                // on the persistent profile having been logged-in via VNC).
                unimplemented!("cookie import deferred — see follow-up to #75")
            }
        },
        Cmd::Render { props, out, codec } => run_render(props, out, codec).await,
        Cmd::Tone { op } => match op {
            ToneOp::Backfill { account, limit, since } => {
                run_tone_backfill(store, account, limit, since).await
            }
            ToneOp::Refresh { scope, account } => {
                run_tone_refresh(store, scope, account).await
            }
            ToneOp::RefreshStale { threshold, budget_secs } => {
                run_tone_refresh_stale(store, threshold, budget_secs).await
            }
        },
        Cmd::BackfillSignatures {
            ref since,
            limit,
            min_confidence,
            apply,
        } => {
            let (broker, _) = build_broker(&cli, Arc::clone(&store), !apply).await?;
            run_backfill_signatures(
                &cli,
                store,
                broker,
                since.clone(),
                limit,
                min_confidence,
                apply,
            )
            .await
        }
        Cmd::Drafts { op } => match op {
            DraftsOp::FeedbackClusters { since_days, top } => {
                run_drafts_feedback_clusters(store, since_days, top)
            }
        },
        Cmd::Approvals { op } => match op {
            ApprovalsOp::List { limit } => run_approvals_list(store, limit),
            ApprovalsOp::ApproveAll { yes } => run_approvals_approve_all(store, yes),
            ApprovalsOp::DiscardOlder { days, yes } => {
                run_approvals_discard_older(store, days, yes)
            }
        },
        Cmd::Ratelimit { op } => match op {
            RatelimitOp::Audit {
                account,
                platform,
                since,
                until,
                json,
            } => run_ratelimit_audit(store, account, platform, since, until, json),
            RatelimitOp::Halts => run_ratelimit_halts(store),
            RatelimitOp::Caps => run_ratelimit_caps(),
        },
        Cmd::SchedulePost { ref op } => run_schedule_post(store, op).await,
        Cmd::Engagement { ref op } => run_engagement(store, op).await,

        // === setup+maintenance subcommands (alphabetical) ===
        Cmd::Channel { name, op, args } => channel_router::dispatch(name, op, args).await,
        Cmd::CodeMode { op } => code_mode::run(store, op).await,
        Cmd::Doctor { json, deep } => {
            let code = doctor::run(store, json, deep).await?;
            std::process::exit(code);
        }
        Cmd::Env { ref op, json } => env_cfg::run_env(op, json),
        Cmd::ReasonerSelftest { ref prompt } => run_reasoner_selftest(prompt).await,
        Cmd::Install { component } => installers::run_install(component).await,
        Cmd::Logs {
            unit,
            follow,
            lines,
            since,
            json,
        } => logs::run_logs(unit, follow, lines, since, json).await,
        Cmd::Loop { op } => loop_cmd::run(store, op).await,
        Cmd::Loops { op } => loops::run(op).await,
        Cmd::Service { op, ref unit, json } => service::run_service(op, unit, json).await,
        Cmd::Setup { ref op } => setup::run_setup(op).await,
        Cmd::Status {
            json,
            channel,
            refresh,
        } => {
            let code = status::run(store, json, channel, refresh).await?;
            // Drop straight to the process exit code so degraded/down states
            // are scriptable from the `/setup` skill. The store is dropped
            // here cleanly via RAII on the `Arc` going out of scope at
            // `std::process::exit`.
            std::process::exit(code);
        }
        Cmd::Uninstall { component } => installers::run_uninstall(component).await,
        // === end setup+maintenance subcommands ===
    }
}

/// `approvals list` — read-only dump of the oldest pending drafts (#99).
fn run_approvals_list(store: Arc<Store>, limit: i64) -> Result<()> {
    let limit = limit.max(1);
    let rows = store.oldest_pending_actions(limit)?;
    let total = store.pending_reply_count()?;
    if rows.is_empty() {
        println!("No pending drafts. Backlog is clear.");
        return Ok(());
    }
    println!(
        "{} pending draft(s) (showing {} oldest first):\n",
        total,
        rows.len()
    );
    for (id, from, subject, age_ms) in &rows {
        println!(
            "  {id}  [{}]  {from} — {}",
            humanize_age(*age_ms),
            truncate(subject, 80)
        );
    }
    if total > rows.len() as i64 {
        println!("\n  (+{} more — raise --limit to see them)", total - rows.len() as i64);
    }
    Ok(())
}

/// `approvals approve-all` — bulk-resolve every pending draft to `approved`
/// (#99). Queue-hygiene only: does NOT send the Gmail drafts. `--yes`-gated;
/// audit-logged to stdout + tracing.
fn run_approvals_approve_all(store: Arc<Store>, yes: bool) -> Result<()> {
    let pending = store.oldest_pending_actions(i64::MAX)?;
    if pending.is_empty() {
        println!("No pending drafts to clear.");
        return Ok(());
    }
    if !yes {
        println!(
            "DRY RUN — would resolve {} pending draft(s) to 'approved' \
             (no Gmail send). Re-run with --yes true to apply:\n",
            pending.len()
        );
        for (id, from, subject, age_ms) in &pending {
            println!("  {id}  [{}]  {from} — {}", humanize_age(*age_ms), truncate(subject, 80));
        }
        return Ok(());
    }
    let mut cleared = 0usize;
    for (id, from, subject, _age) in &pending {
        if store.mark_pending_approved(id)? {
            cleared += 1;
            info!(action_id = %id, %from, "approvals approve-all: resolved pending draft");
            println!("[audit] approve-all resolved {id}  {from} — {}", truncate(subject, 80));
        }
    }
    info!(cleared, requested = pending.len(), "approvals approve-all complete");
    println!("\nCleared {cleared} pending draft(s). Gmail drafts were NOT sent.");
    Ok(())
}

/// `approvals discard-older <days>` — expire pending drafts older than N days
/// to `timed_out` (#99). `--yes`-gated; audit-logged.
fn run_approvals_discard_older(store: Arc<Store>, days: i64, yes: bool) -> Result<()> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let days = days.max(0);
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);
    let cutoff_ms = now_ms - days * 24 * 60 * 60 * 1000;
    // Preview: enumerate the rows that would be swept. A row's creation time
    // is `now_ms - age_ms`; it's stale when that is at/before the cutoff.
    let stale: Vec<_> = store
        .oldest_pending_actions(i64::MAX)?
        .into_iter()
        .filter(|(_, _, _, age_ms)| now_ms - age_ms <= cutoff_ms)
        .collect();
    if stale.is_empty() {
        println!("No pending drafts older than {days}d.");
        return Ok(());
    }
    if !yes {
        println!(
            "DRY RUN — would expire {} pending draft(s) older than {days}d to \
             'timed_out'. Re-run with --yes true to apply:\n",
            stale.len()
        );
        for (id, from, subject, age_ms) in &stale {
            println!("  {id}  [{}]  {from} — {}", humanize_age(*age_ms), truncate(subject, 80));
        }
        return Ok(());
    }
    let swept_ids = store.expire_pending_older_than(cutoff_ms)?;
    let swept = swept_ids.len();
    info!(swept, days, "approvals discard-older complete");
    for (id, from, subject, age_ms) in &stale {
        println!(
            "[audit] discard-older expired {id}  [{}]  {from} — {}",
            humanize_age(*age_ms),
            truncate(subject, 80)
        );
    }
    println!("\nExpired {swept} pending draft(s) older than {days}d.");
    Ok(())
}

/// Lightweight recurring-feedback surfacing for `draft_revisions` (#37 Phase 2
/// scaffolding). Tokenizes feedback strings into lowercase 1-grams + 2-grams,
/// drops a small stop-list, ranks by document frequency, and prints the top
/// `top` patterns with one example feedback per cluster.
///
/// Embedding-based clustering (HDBSCAN over Voyage / local embeddings) is the
/// long-term plan; this is a deliberately dumb v0 that ships value while we
/// gather enough rows to justify the heavier stack.
fn run_drafts_feedback_clusters(
    store: Arc<Store>,
    since_days: u32,
    top: usize,
) -> Result<()> {
    use std::collections::HashMap;
    let since_ms = i64::from(since_days) * 24 * 60 * 60 * 1000;
    let rows = store.list_recent_feedback(since_ms)?;
    if rows.is_empty() {
        println!("(no feedback in the last {since_days} days)");
        return Ok(());
    }
    let stop: std::collections::HashSet<&str> = [
        "the", "a", "an", "is", "it", "to", "of", "and", "or", "in", "on", "for",
        "be", "this", "that", "but", "not", "with", "as", "at", "by", "i", "me",
        "my", "we", "our", "you", "your", "should", "would", "could", "make",
        "please", "less", "more",
    ]
    .into_iter()
    .collect();
    let mut counts: HashMap<String, (usize, String)> = HashMap::new();
    for row in &rows {
        let Some(fb) = row.feedback_text.as_deref() else { continue };
        let lower = fb.to_lowercase();
        let words: Vec<&str> = lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty() && w.len() > 2 && !stop.contains(w))
            .collect();
        // Count unique terms per row so a single very long feedback doesn't
        // dominate.
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for w in &words {
            seen.insert((*w).to_string());
        }
        for pair in words.windows(2) {
            seen.insert(format!("{} {}", pair[0], pair[1]));
        }
        for term in seen {
            let entry = counts.entry(term).or_insert((0, fb.to_string()));
            entry.0 += 1;
        }
    }
    let mut ranked: Vec<(String, usize, String)> = counts
        .into_iter()
        .filter(|(_, (n, _))| *n >= 2)
        .map(|(t, (n, ex))| (t, n, ex))
        .collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    println!(
        "Top {} feedback patterns over last {since_days}d ({} total revisions):",
        top.min(ranked.len()),
        rows.len()
    );
    for (term, count, example) in ranked.into_iter().take(top) {
        println!("  {count:>3}x  \"{term}\"  e.g. {}", truncate(&example, 80));
    }
    Ok(())
}

fn parse_iso_or_now_offset(s: Option<String>, default_offset_ms: i64) -> Result<i64> {
    match s {
        Some(raw) => {
            // Accept full RFC3339 ("2026-05-14T00:00:00Z") or just a date
            // ("2026-05-14"); the latter is interpreted as UTC midnight.
            if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(&raw) {
                Ok(dt.timestamp_millis())
            } else if let Ok(d) = chrono::NaiveDate::parse_from_str(&raw, "%Y-%m-%d") {
                let dt = d.and_hms_opt(0, 0, 0).unwrap().and_utc();
                Ok(dt.timestamp_millis())
            } else {
                anyhow::bail!("could not parse {raw:?} as ISO date or RFC3339 timestamp");
            }
        }
        None => Ok(chrono::Utc::now().timestamp_millis() + default_offset_ms),
    }
}

fn run_ratelimit_audit(
    store: Arc<Store>,
    account: String,
    platform: Option<String>,
    since: Option<String>,
    until: Option<String>,
    json: bool,
) -> Result<()> {
    let since_ms = parse_iso_or_now_offset(since, -7 * 24 * 3600 * 1000)?;
    let until_ms = parse_iso_or_now_offset(until, 0)?;
    let rows = store
        .rate_audit_query(&account, platform.as_deref(), since_ms, until_ms)
        .context("rate_audit_query")?;
    if json {
        println!("{}", serde_json::to_string_pretty(&rows)?);
    } else if rows.is_empty() {
        println!(
            "no rate_events for account={account} platform={:?} in [{since_ms}, {until_ms}]",
            platform
        );
    } else {
        println!("{} rate_events:\n", rows.len());
        println!(
            "  {:<24}  {:<10}  {:<18}  {:<10}  {:<14}  {}",
            "occurred_at_ms", "platform", "action", "status", "target", "cause"
        );
        for r in &rows {
            let target = r.target_id.as_deref().unwrap_or("-");
            println!(
                "  {:<24}  {:<10}  {:<18}  {:<10}  {:<14}  {}",
                r.occurred_at_ms, r.platform, r.action_kind, r.status, target, r.cause
            );
        }
    }
    Ok(())
}

fn run_ratelimit_halts(store: Arc<Store>) -> Result<()> {
    use augmentagent_channel_core::governor::Platform;
    let mut active = Vec::new();
    for p in [
        Platform::Instagram,
        Platform::LinkedIn,
        Platform::Twitter,
        Platform::TikTok,
        Platform::Bluesky,
    ] {
        if let Some(h) = store
            .rate_halt_state(p.as_str())
            .context("rate_halt_state")?
        {
            active.push(h);
        }
    }
    println!("{}", serde_json::to_string_pretty(&active)?);
    Ok(())
}

fn run_ratelimit_caps() -> Result<()> {
    use augmentagent_channel_core::governor::RATE_TABLE;
    let rows: Vec<_> = RATE_TABLE
        .iter()
        .map(|r| {
            serde_json::json!({
                "platform": r.platform.as_str(),
                "action": r.action.as_str(),
                "day": r.day,
                "hour": r.hour,
                "burst_5m": r.burst_5m,
                "min_gap_secs": r.min_gap.as_secs(),
                "source_url": r.source_url,
            })
        })
        .collect();
    println!("{}", serde_json::to_string_pretty(&rows)?);
    Ok(())
}

/// Match an account against a `--account` filter (#482): exact entity-id match,
/// or a case-insensitive substring of the email — so `nolanmak7` and the full
/// address both select the same account.
fn account_matches(account: &augmentagent_store::Account, filter: &str) -> bool {
    let f = filter.trim();
    if f.is_empty() {
        return false;
    }
    account.entity_id == f
        || account
            .email
            .to_ascii_lowercase()
            .contains(&f.to_ascii_lowercase())
}

async fn run_gmail_search(
    store: Arc<Store>,
    query: String,
    limit: u32,
    full: bool,
    account_filter: Option<String>,
) -> Result<()> {
    let api_key = std::env::var("COMPOSIO_API_KEY").context("COMPOSIO_API_KEY env var required")?;
    let gmail = ComposioClient::new(api_key).with_rate_limit_store(Arc::clone(&store));
    let mut accounts = store.get_active_gmail_accounts()?;
    if accounts.is_empty() {
        println!("(no active gmail accounts)");
        return Ok(());
    }

    // #482: optionally scope to a single account so the user can route around a
    // throttled one, since search otherwise fans out over every account.
    if let Some(filter) = account_filter.as_deref() {
        accounts.retain(|a| account_matches(a, filter));
        if accounts.is_empty() {
            println!("(no active gmail account matches --account {filter:?})");
            return Ok(());
        }
    }

    // #482: emit a per-account status line for EVERY account on stdout. A
    // failure on one account must never silently blank it out — the user has
    // to be able to tell "0 results" apart from "skipped: <error>", and the
    // healthy accounts' results always come through regardless.
    let mut total = 0usize;
    for account in &accounts {
        match gmail.fetch_with_query(&account.entity_id, &query, limit).await {
            Err(e) => {
                // Also mirror to stderr so it shows up in logs, but stdout is
                // what the user sees — never a silent skip.
                eprintln!("account {} search failed: {e}", account.entity_id);
                println!(
                    "## account {} ({}) — SKIPPED: {e}",
                    account.entity_id, account.email
                );
            }
            Ok(emails) if emails.is_empty() => {
                println!(
                    "## account {} ({}) — 0 results",
                    account.entity_id, account.email
                );
            }
            Ok(emails) => {
                total += emails.len();
                println!(
                    "## account {} ({}) — {} results",
                    account.entity_id,
                    account.email,
                    emails.len()
                );
                for (i, email) in emails.iter().enumerate() {
                    // threadId is what compose --thread-id / send-now --thread-id
                    // actually need; printing only messageId forced callers to
                    // guess (#381).
                    println!(
                        "[{:>2}] from: {}\n     subject: {}\n     date: {}\n     messageId: {}\n     threadId: {}",
                        i + 1,
                        email.from,
                        email.subject,
                        email.date,
                        email.message_id,
                        email.thread_id.as_deref().unwrap_or("-")
                    );
                    // #629 — surface the full recipient set so reply-all /
                    // compose callers can enumerate everyone on the chain
                    // instead of only addresses that appear as senders.
                    if !email.to.is_empty() {
                        println!("     to: {}", email.to);
                    }
                    if !email.cc.is_empty() {
                        println!("     cc: {}", email.cc);
                    }
                    // #811 — attachment metadata was invisible here, so an
                    // attachment-only email looked like an empty message.
                    if !email.attachments.is_empty() {
                        println!("     attachments: {}", email.attachments.join(", "));
                    }
                    if full {
                        println!("     body:\n{}\n", indent_body(&email.body, 7));
                    }
                }
                println!();
            }
        }
    }
    if total == 0 {
        println!("(no results across {} account(s))", accounts.len());
    }
    Ok(())
}

#[cfg(test)]
mod gmail_search_account_filter_tests {
    use augmentagent_store::Account;

    fn acct(entity_id: &str, email: &str) -> Account {
        Account {
            id: entity_id.to_string(),
            connection_id: None,
            entity_id: entity_id.to_string(),
            email: email.to_string(),
            active: true,
        }
    }

    #[test]
    fn account_matches_by_entity_id_and_email_substring() {
        let a = acct("augmentagent-123", "nolanmak7@gmail.com"); // pii-ok: synthetic
        assert!(super::account_matches(&a, "augmentagent-123")); // exact entity id
        assert!(super::account_matches(&a, "NOLANMAK7@gmail.com")); // pii-ok: full email, case-insensitive
        assert!(super::account_matches(&a, "nolanmak7")); // pii-ok: email substring (the ergonomic form)
        assert!(!super::account_matches(&a, "someone-else")); // pii-ok: non-match
        assert!(!super::account_matches(&a, "")); // empty never matches
    }
}

fn indent_body(body: &str, cols: usize) -> String {
    let pad = " ".repeat(cols);
    body.lines()
        .map(|l| format!("{pad}{l}"))
        .collect::<Vec<_>>()
        .join("\n")
}

// ---------------------------------------------------------------------------
// Browser sidecar CLI handlers (issue #75 v0).
//
// `start`/`stop`/`status` are thin systemd wrappers; `acceptance-test` does
// the §10 round-trip through the sidecar. `import-cookies` is unimplemented
// — deferred to a follow-up issue.
// ---------------------------------------------------------------------------

const BROWSER_UNITS: &[&str] = &[
    "augmentagent-xvfb.service",
    "augmentagent-chromium.service",
    "augmentagent-browser-sidecar.service",
];

fn run_systemctl(args: &[&str]) -> Result<std::process::Output> {
    use std::process::Command;
    let out = Command::new("systemctl")
        .arg("--user")
        .args(args)
        .output()
        .with_context(|| format!("failed to spawn systemctl --user {}", args.join(" ")))?;
    Ok(out)
}

async fn run_browser_start() -> Result<()> {
    for unit in BROWSER_UNITS {
        let out = run_systemctl(&["start", unit])?;
        if !out.status.success() {
            eprintln!(
                "systemctl start {unit} failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            return Err(anyhow::anyhow!("systemctl start {unit} failed"));
        }
        println!("started {unit}");
    }
    println!(
        "\nall three units up. socket: {:?}",
        augmentagent_browser_client::default_socket_path()
    );
    println!("if this is a fresh profile: complete one-time login per sidecars/browser/README.md");
    Ok(())
}

async fn run_browser_stop() -> Result<()> {
    // Stop in reverse order so dependents go down first.
    for unit in BROWSER_UNITS.iter().rev() {
        let out = run_systemctl(&["stop", unit])?;
        if !out.status.success() {
            eprintln!(
                "systemctl stop {unit} failed (continuing): {}",
                String::from_utf8_lossy(&out.stderr)
            );
        } else {
            println!("stopped {unit}");
        }
    }
    Ok(())
}

async fn run_browser_status(json: bool) -> Result<()> {
    use augmentagent_browser_client::{default_socket_path, BrowserClient};

    let mut units_state = Vec::new();
    for unit in BROWSER_UNITS {
        let out = run_systemctl(&["is-active", unit])?;
        let state = String::from_utf8_lossy(&out.stdout).trim().to_string();
        units_state.push((*unit, state));
    }

    let sock = default_socket_path();
    let sock_exists = sock.exists();
    let mut ping_ok = false;
    let mut ping_err: Option<String> = None;
    if sock_exists {
        match BrowserClient::connect(&sock).await {
            Ok(client) => match client.ping().await {
                Ok(()) => ping_ok = true,
                Err(e) => ping_err = Some(e.to_string()),
            },
            Err(e) => ping_err = Some(e.to_string()),
        }
    } else {
        ping_err = Some(format!("socket not present at {:?}", sock));
    }

    if json {
        let body = serde_json::json!({
            "units": units_state.iter().map(|(u, s)| {
                serde_json::json!({"unit": u, "state": s})
            }).collect::<Vec<_>>(),
            "socket": sock.to_string_lossy(),
            "socket_exists": sock_exists,
            "ping_ok": ping_ok,
            "ping_error": ping_err,
        });
        println!("{}", serde_json::to_string_pretty(&body)?);
    } else {
        for (u, s) in &units_state {
            println!("{u}: {s}");
        }
        println!("socket: {} (exists={})", sock.display(), sock_exists);
        if ping_ok {
            println!("ping: OK");
        } else if let Some(e) = ping_err {
            println!("ping: FAIL — {e}");
        }
    }
    Ok(())
}

async fn run_browser_acceptance(out_path: PathBuf) -> Result<()> {
    use augmentagent_browser_client::{default_socket_path, BrowserClient};

    let sock = default_socket_path();
    println!("connecting to sidecar at {}", sock.display());
    let client = BrowserClient::connect(&sock).await.with_context(|| {
        format!(
            "connect failed — is augmentagent-browser-sidecar.service running? socket: {}",
            sock.display()
        )
    })?;

    println!("ping...");
    client.ping().await.context("ping failed")?;

    println!("navigate https://twitter.com");
    if let Err(e) = client.navigate("https://twitter.com").await {
        if e.is_auth_required() {
            println!(
                "FAIL — AuthRequired: complete one-time login per sidecars/browser/README.md"
            );
            return Err(anyhow::anyhow!("auth required"));
        }
        return Err(e).context("navigate failed");
    }

    println!("screenshot -> {}", out_path.display());
    let _bytes = client
        .screenshot(&out_path)
        .await
        .context("screenshot failed")?;

    println!("evaluate logged-in DOM marker");
    let v = client
        .evaluate(
            "!!document.querySelector(\"[data-testid='SideNav_AccountSwitcher_Button']\")",
        )
        .await
        .context("evaluate failed")?;
    let logged_in = v.as_bool().unwrap_or(false);
    if logged_in {
        println!("PASS — screenshot at {}", out_path.display());
        Ok(())
    } else {
        println!(
            "FAIL — logged-out DOM. Complete one-time login per sidecars/browser/README.md\n\
             screenshot saved to {} for inspection.",
            out_path.display()
        );
        Err(anyhow::anyhow!("not logged in"))
    }
}

// ---------------------------------------------------------------------------
// Renderer sidecar CLI handler (Remotion Phase 0 — see docs/REMOTION.md).
//
// Manually triggerable: connect to the renderer sidecar, render the
// ShortCard composition from JSON props, print the output path + bytes.
// No scheduler / governor / posting wiring (later phases).
// ---------------------------------------------------------------------------

async fn run_render(props: String, out: PathBuf, codec: String) -> Result<()> {
    use augmentagent_renderer_client::{default_socket_path, RendererClient};

    // `--props` accepts an inline JSON string or `@path` to a JSON file.
    let raw = if let Some(path) = props.strip_prefix('@') {
        std::fs::read_to_string(path)
            .with_context(|| format!("read props file {path}"))?
    } else {
        props
    };
    let props_json: serde_json::Value =
        serde_json::from_str(raw.trim()).context("--props is not valid JSON")?;

    let sock = default_socket_path();
    println!("connecting to renderer sidecar at {}", sock.display());
    let client = RendererClient::connect(&sock).await.with_context(|| {
        format!(
            "connect failed — is augmentagent-renderer.service running? socket: {}",
            sock.display()
        )
    })?;

    println!("ping...");
    client.ping().await.context("ping failed")?;

    println!(
        "render -> {} (codec={codec})\nprops: {}",
        out.display(),
        serde_json::to_string(&props_json).unwrap_or_default()
    );
    let result = client
        .render_with(
            props_json,
            &out,
            &codec,
            augmentagent_renderer_client::DEFAULT_RENDER_TIMEOUT_MS,
        )
        .await
        .context("render failed")?;

    println!(
        "OK — {} ({} bytes, {} ms server-side)",
        result.path, result.bytes, result.duration_ms
    );
    Ok(())
}

/// Resolve the user-supplied `--account` flag (email address OR Composio
/// entity_id) to a concrete entity_id. If `selector` is None and there's
/// exactly one active Gmail account, return that account's entity_id.
/// Otherwise error with a helpful message listing options.
fn resolve_gmail_entity_id(
    store: &Store,
    selector: Option<String>,
) -> Result<(String, String)> {
    let accounts = store.get_active_gmail_accounts()?;
    if accounts.is_empty() {
        anyhow::bail!("no active gmail accounts; connect one first");
    }
    if let Some(s) = selector {
        // email match (case-insensitive) takes priority over entity_id match.
        let lower = s.to_ascii_lowercase();
        if let Some(a) = accounts
            .iter()
            .find(|a| a.email.to_ascii_lowercase() == lower)
        {
            return Ok((a.entity_id.clone(), a.email.clone()));
        }
        if let Some(a) = accounts.iter().find(|a| a.entity_id == s) {
            return Ok((a.entity_id.clone(), a.email.clone()));
        }
        let known: Vec<String> = accounts
            .iter()
            .map(|a| format!("{} ({})", a.email, a.entity_id))
            .collect();
        anyhow::bail!(
            "no active gmail account matches '{s}'. Known accounts:\n  - {}",
            known.join("\n  - ")
        );
    }
    if accounts.len() == 1 {
        let a = &accounts[0];
        return Ok((a.entity_id.clone(), a.email.clone()));
    }
    let known: Vec<String> = accounts.iter().map(|a| a.email.clone()).collect();
    anyhow::bail!(
        "--account required (multiple gmail accounts active): {}",
        known.join(", ")
    );
}

fn read_body(body: Option<String>, body_file: Option<String>) -> Result<String> {
    match (body, body_file) {
        (Some(_), Some(_)) => anyhow::bail!("pass --body OR --body-file, not both"),
        // Inline --body values get backslash escapes interpreted (#418):
        // callers (humans and the wiki-ask agent alike) write
        // `--body "line1\n\nline2"`, and in shell double quotes that \n is a
        // literal backslash+n which Gmail then DISPLAYS as text — the "ugly
        // formatting" bug. No real email body wants a visible backslash-n.
        // File/stdin bodies are passed through verbatim.
        (Some(b), None) => Ok(unescape_body(&b)),
        (None, Some(p)) => {
            if p == "-" {
                let mut buf = String::new();
                std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)
                    .context("read body from stdin")?;
                Ok(buf)
            } else {
                std::fs::read_to_string(&p).with_context(|| format!("read body file {p}"))
            }
        }
        (None, None) => anyhow::bail!("either --body or --body-file is required"),
    }
}

/// Interpret the escape sequences `\n`, `\t`, and `\r\n`→`\n` in an inline
/// `--body` string (#418). `\\` escapes a backslash so a deliberate literal
/// `\n` remains expressible as `\\n`. Anything else after a backslash is
/// passed through untouched.
fn unescape_body(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.peek() {
            Some('n') => {
                chars.next();
                out.push('\n');
            }
            Some('t') => {
                chars.next();
                out.push('\t');
            }
            // \r\n (escaped CRLF) collapses to one newline; a lone \r too.
            Some('r') => {
                chars.next();
                if chars.peek() == Some(&'\\') {
                    let mut ahead = chars.clone();
                    ahead.next();
                    if ahead.peek() == Some(&'n') {
                        chars.next();
                        chars.next();
                    }
                }
                out.push('\n');
            }
            Some('\\') => {
                chars.next();
                out.push('\\');
            }
            _ => out.push('\\'),
        }
    }
    out
}

async fn run_gmail_accounts(store: Arc<Store>, json: bool) -> Result<()> {
    let accounts = store.get_active_gmail_accounts()?;
    if json {
        let rows: Vec<_> = accounts
            .iter()
            .map(|a| {
                serde_json::json!({
                    "email": a.email,
                    "entity_id": a.entity_id,
                    "active": a.active,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&rows)?);
        return Ok(());
    }
    if accounts.is_empty() {
        println!("(no active gmail accounts)");
        return Ok(());
    }
    for a in &accounts {
        println!("{}\t{}", a.email, a.entity_id);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
/// #417 — Upload `--attach` (when given) to Composio's attachment store,
/// printing what was attached so the operator can SEE it happened. Fails
/// loudly on a missing/unreadable file before any draft is created.
async fn upload_attach_if_given(
    gmail: &ComposioClient,
    attach: Option<PathBuf>,
) -> Result<Option<Attachment>> {
    let Some(path) = attach else { return Ok(None) };
    let meta = std::fs::metadata(&path)
        .with_context(|| format!("attachment not found: {}", path.display()))?;
    let att = gmail
        .upload_attachment("GMAIL_CREATE_EMAIL_DRAFT", &path)
        .await
        .context("attachment upload failed")?;
    println!(
        "attached: {} ({} KB, {})",
        att.name,
        meta.len().div_ceil(1024),
        att.mimetype
    );
    Ok(Some(att))
}

/// Canonicalize a user-supplied `--thread-id` before any draft is created
/// (#381). Accepts a Gmail threadId or a messageId (resolved to its thread);
/// fails with an actionable message when the id isn't in `account_email`'s
/// mailbox — the raw Gmail 404 ("Requested entity was not found") gave the
/// caller nothing to act on.
async fn resolve_compose_thread_id(
    gmail: &ComposioClient,
    entity_id: &str,
    account_email: &str,
    thread_id: Option<String>,
) -> Result<Option<String>> {
    let Some(t) = thread_id.filter(|t| !t.is_empty()) else {
        return Ok(None);
    };
    let resolved = gmail.resolve_thread_id(entity_id, &t).await.with_context(|| {
        format!(
            "--thread-id {t} was not found in account {account_email}. Pass the threadId \
             printed by `gmail search` (a messageId also works), and make sure --account \
             is the account that actually contains the thread — ids from one mailbox \
             don't exist in another."
        )
    })?;
    Ok(Some(resolved))
}

/// #439 — flatten repeated `--to`/`--cc`/`--bcc` values (each possibly a
/// comma-separated list, display names allowed) into bare addresses, failing
/// fast on anything that doesn't look like an address rather than letting
/// Composio reject the whole draft after the attachment/thread work is done.
fn normalize_recipients(flag: &str, values: &[String]) -> Result<Vec<String>> {
    let list: Vec<String> = values
        .iter()
        .flat_map(|v| augmentagent_channel_email::gmail::split_recipients(v))
        .collect();
    for addr in &list {
        anyhow::ensure!(
            addr.contains('@') && !addr.contains(char::is_whitespace),
            "{flag} value {addr:?} doesn't look like an email address"
        );
    }
    Ok(list)
}

/// Remove recipient metadata that is rendered on approval cards but must
/// never become part of the Gmail message body. Older cards could feed their
/// display-only `CC:`/`[cc: ...]` line back through the revise prompt, after
/// which the model occasionally returned it as email text. The #652
/// `[subject: …]` marker is display-only in the same way — but ONLY in its
/// bracketed form: a bare `Subject: …` line is real prose (#650 handles the
/// leading-header case separately).
fn strip_approval_envelope_markers(body: &str) -> String {
    body.lines()
        .filter(|line| {
            let trimmed = line.trim();
            let bracketed = trimmed
                .strip_prefix('[')
                .and_then(|s| s.strip_suffix(']'));
            if let Some(inner) = bracketed {
                if let Some((name, _)) = inner.split_once(':') {
                    if name.trim().eq_ignore_ascii_case("subject") {
                        return false;
                    }
                }
            }
            let marker = bracketed.unwrap_or(trimmed);
            let Some((name, value)) = marker.split_once(':') else {
                return true;
            };
            let name = name.trim().to_ascii_lowercase();
            let value = value.trim();
            let is_recipient_list = !value.is_empty()
                && value.split(',').all(|recipient| {
                    let recipient = recipient.trim();
                    recipient.contains('@') && !recipient.contains(char::is_whitespace)
                });
            !(matches!(name.as_str(), "to" | "cc" | "bcc") && is_recipient_list)
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim_end_matches('\n')
        .to_string()
}

/// #650 — split a leading `Subject: …` line off a body that is about to become
/// a Gmail message. The subject belongs in the header; a body that opens with
/// one ships that line to the recipient as visible text. Returns the remainder
/// (a verbatim slice, so CRLF endings and the trailing newline survive) plus
/// the dropped value, or the body unchanged when the first non-blank line is
/// not a subject header — a `Subject:` inside quoted or forwarded text further
/// down is real message content.
fn strip_leading_subject_line(body: &str) -> (String, Option<String>) {
    const HEADER: &str = "subject:";
    let mut offset = 0;
    for line in body.split_inclusive('\n') {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            offset += line.len();
            continue;
        }
        let Some((header, value)) = trimmed.get(..HEADER.len()).zip(trimmed.get(HEADER.len()..))
        else {
            return (body.to_string(), None);
        };
        if !header.eq_ignore_ascii_case(HEADER) {
            return (body.to_string(), None);
        }
        offset += line.len();
        for after in body[offset..].split_inclusive('\n') {
            if !after.trim().is_empty() {
                break;
            }
            offset += after.len();
        }
        return (body[offset..].to_string(), Some(value.trim().to_string()));
    }
    (body.to_string(), None)
}

/// Do two subjects say the same thing? Reply/forward prefixes and case are
/// noise here: a reply's header is `Re: venue question` while the body echoes
/// back the bare `venue question`, and both name one subject.
fn subjects_agree(a: &str, b: &str) -> bool {
    const PREFIXES: [&str; 3] = ["re:", "fwd:", "fw:"];
    fn core(s: &str) -> String {
        let mut s = s.trim();
        loop {
            let lower = s.to_ascii_lowercase();
            match PREFIXES.iter().find(|p| lower.starts_with(**p)) {
                Some(p) => s = s[p.len()..].trim_start(),
                None => return lower,
            }
        }
    }
    core(a) == core(b)
}

/// #650 — reconcile a body that opens with its own `Subject:` header against
/// the subject the Gmail write is actually going to send. Returns the body to
/// send plus the dropped line's value, or the reason the write must not
/// happen. Refusing here (rather than at the API) keeps #412 discipline: no
/// draft exists yet, so there is no orphan to clean up.
///
/// Dropping the line is only safe when it repeats the outgoing subject. A
/// DIFFERENT one is a subject the author meant to set, and no body can supply
/// a header, so silently dropping it would send a message under a subject
/// nobody asked for. Say so instead and let the caller put it where it
/// belongs — which for a threaded write means starting a new thread, since
/// [`ensure_subject_matches_thread`] (#651) holds a reply to its thread's subject.
fn body_without_leaked_subject(
    body: &str,
    outgoing_subject: &str,
) -> Result<(String, Option<String>), String> {
    let (rest, Some(dropped)) = strip_leading_subject_line(body) else {
        return Ok((body.to_string(), None));
    };
    if !subjects_agree(&dropped, outgoing_subject) {
        return Err(format!(
            "the body opens with \"Subject: {dropped}\" but the message goes out as \
             \"{outgoing_subject}\" — a Subject line in the body is delivered as \
             visible text, never as the header"
        ));
    }
    if rest.trim().is_empty() {
        return Err(
            "the body is nothing but its own \"Subject:\" line — the subject is \
             already on the message, the body needs the message text"
                .into(),
        );
    }
    Ok((rest, Some(dropped)))
}

/// CLI adapter for [`body_without_leaked_subject`]: a body-level `Subject:`
/// naming something other than `--subject` is a usage error, and one that
/// merely repeats `--subject` is dropped with a note.
fn body_for_gmail_write(body: String, subject: &str) -> Result<String> {
    match body_without_leaked_subject(&body, subject) {
        Ok((clean, dropped)) => {
            if let Some(dropped) = dropped {
                eprintln!(
                    "note: dropped leading \"Subject: {dropped}\" line from the body; \
                     the subject header is --subject ({subject})"
                );
            }
            Ok(clean)
        }
        Err(e) => anyhow::bail!("{e}; pass the intended subject as --subject"),
    }
}

#[derive(Debug, PartialEq, Eq)]
enum ComposePendingDisposition {
    Create,
    Replace {
        action_id: String,
        draft_id: Option<String>,
    },
}

/// Decide what an actionable compose should do when an equivalent pending
/// card already exists. A normal follow-up replaces the pending card's
/// action; `--allow-duplicate` remains the explicit escape hatch for users
/// who truly want two separate emails under the same subject.
fn compose_pending_disposition(
    post: bool,
    allow_duplicate: bool,
    pending: Option<(&str, Option<&str>)>,
) -> ComposePendingDisposition {
    if post && !allow_duplicate {
        if let Some((action_id, draft_id)) = pending {
            return ComposePendingDisposition::Replace {
                action_id: action_id.to_string(),
                draft_id: draft_id.map(str::to_string),
            };
        }
    }
    ComposePendingDisposition::Create
}

/// Subject to recreate a revised Gmail draft with. `thread_id` is what makes a
/// card a reply — the same field the recreated draft is threaded off — so the
/// `Re:` prefix follows it: new-email compose cards (#675) keep the subject the
/// user passed to `--subject` verbatim.
fn revise_subject(original: &str, thread_id: Option<&str>) -> String {
    if thread_id.is_none() || original.to_ascii_lowercase().starts_with("re:") {
        return original.to_string();
    }
    format!("Re: {original}")
}

/// #652 — the subject a recreated draft carries after a Revise. A subject
/// asked for in THIS revise wins and is used verbatim (no forced `Re:` — the
/// user asked for those words); otherwise an override from an earlier round
/// still stands, so "make it shorter" can't silently undo it; otherwise the
/// subject derived from the inbound mail, which is the pre-#652 behavior.
fn revised_subject(
    asked_for: Option<&str>,
    persisted: Option<&str>,
    original: &str,
    thread_id: Option<&str>,
) -> String {
    asked_for
        .or(persisted)
        .map(str::to_string)
        .unwrap_or_else(|| revise_subject(original, thread_id))
}

/// Refuse a threaded compose whose `--subject` disagrees with the thread
/// (#651). Gmail sends a reply under the thread's ORIGINAL subject and drops
/// the one passed alongside `thread_id`, so such a message goes out under a
/// header that has nothing to do with its content — as three recipients saw.
/// Only that original counts as a match: a subject introduced later in the
/// thread, or a `Fwd:` of it, is still not what recipients would see.
/// Returns the refusal text, or `None` when there is nothing to compare (no
/// `--subject`, subject-less thread).
fn thread_subject_conflict(subject: &str, thread: &ThreadSubject) -> Option<String> {
    let wanted = normalize_subject(subject);
    if wanted.is_empty() {
        return None;
    }
    match thread {
        // Every message on the thread came back untitled: there is no header
        // to contradict, and refusing would block replying to it at all.
        ThreadSubject::Missing => None,
        // Unknowable ⇒ unverifiable. Sending anyway is the gamble that put
        // the wrong header in three inboxes, so refuse and name the way out.
        ThreadSubject::Undetermined => Some(format!(
            "could not tell which subject this thread was started under: it carries more than one \
             and the provider returned no timestamps to order them by. A reply on --thread-id is \
             sent under the thread's original subject, so {subject:?} cannot be verified. Drop \
             --thread-id to start a new thread under {subject:?}, or send the reply from Gmail."
        )),
        ThreadSubject::Original(actual) => {
            let sent_as = normalize_subject(actual);
            if sent_as.is_empty() || sent_as == wanted {
                return None;
            }
            Some(format!(
                "--subject {subject:?} does not match the thread's subject {actual:?}. A reply on \
                 --thread-id is sent under the thread's subject, so this message would go out \
                 with a header that doesn't match its content. Either pass --subject \
                 \"Re: {actual}\" to reply in-thread, or drop --thread-id to start a new thread \
                 under {subject:?}."
            ))
        }
    }
}

/// Fail-closed guard for [`thread_subject_conflict`], run before any draft is
/// created — and before any attachment upload — so a refusal never strands
/// orphan work (#412 discipline). Costs one extra thread fetch per threaded
/// compose; a provider failure refuses rather than sends, because an
/// unverified subject is exactly what #651 put in front of recipients.
async fn ensure_subject_matches_thread(
    gmail: &ComposioClient,
    entity_id: &str,
    thread_id: &str,
    subject: &str,
) -> Result<()> {
    let thread = gmail
        .fetch_thread_subject(entity_id, thread_id)
        .await
        .context(
            "could not verify the thread's subject before drafting; retry, or drop --thread-id \
             to start a new thread",
        )?;
    if let Some(conflict) = thread_subject_conflict(subject, &thread) {
        anyhow::bail!(conflict);
    }
    Ok(())
}

/// #652 × #651 — which thread the recreated draft joins. Gmail sends a
/// threaded message under the THREAD's subject and drops any other, so a
/// subject the user asked for during Revise can only take effect on a new
/// thread. Keep the thread while the outgoing subject still names the
/// inbound one (a `Re:`/`Fwd:` prefix is fine — that is every ordinary
/// reply); start a new thread the moment it says something else. Never
/// accept a subject change and then let Gmail discard it.
fn thread_for_revised_subject<'a>(
    outgoing_subject: &str,
    inbound_subject: &str,
    thread_id: Option<&'a str>,
) -> Option<&'a str> {
    let thread_id = thread_id?;
    // An untitled inbound is a subject too: a Revise that gives it one is a
    // change Gmail would drop on-thread just the same.
    if normalize_subject(outgoing_subject) == normalize_subject(inbound_subject) {
        Some(thread_id)
    } else {
        None
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_gmail_compose(
    store: Arc<Store>,
    account: Option<String>,
    to: Vec<String>,
    cc: Vec<String>,
    bcc: Vec<String>,
    subject: String,
    body: Option<String>,
    body_file: Option<String>,
    thread_id: Option<String>,
    json: bool,
    post: bool,
    allow_duplicate: bool,
    attach: Option<PathBuf>,
    reply_to_message_id: Option<String>,
    reply_to_from: Option<String>,
    reply_to_subject: Option<String>,
    reply_to_body: Option<String>,
    reply_to_body_file: Option<String>,
    send_at: Option<String>,
) -> Result<()> {
    // #502 — resolve and bound the proposed send time BEFORE any Gmail
    // write, so a bad value can't strand an orphan draft (#412 discipline).
    // Requires --post: a proposal without a card would have no Approve to
    // arm it and no surface to show it.
    let send_at_ms = match send_at.as_deref().filter(|s| !s.trim().is_empty()) {
        Some(raw) => {
            anyhow::ensure!(
                post,
                "--send-at requires --post: the proposed time is carried by \
                 the approval card and armed on Approve"
            );
            let now = chrono::Local::now();
            let at_ms = augmentagent_channel_core::timeparse::parse_send_at(raw, now)
                .map_err(|e| anyhow::anyhow!("--send-at: {e}"))?;
            augmentagent_channel_core::timeparse::validate_send_at(
                at_ms,
                now.timestamp_millis(),
            )
            .map_err(|e| anyhow::anyhow!("--send-at: {e}"))?;
            Some(at_ms)
        }
        None => None,
    };
    // Normalize recipients up front (#439): `to` becomes one canonical
    // comma-joined string of bare addresses. Everything downstream — the
    // duplicate guard, the approval card's From field (which the Revise
    // redraft round-trips back into create_draft), the JSON output — carries
    // the full list, and the Composio client re-splits it into
    // recipient_email + extra_recipients at the wire.
    let to = normalize_recipients("--to", &to)?;
    anyhow::ensure!(!to.is_empty(), "--to requires at least one email address");
    let to = to.join(", ");
    let cc = normalize_recipients("--cc", &cc)?;
    let bcc = normalize_recipients("--bcc", &bcc)?;
    let body_str = body_for_gmail_write(read_body(body, body_file)?, &subject)?;
    // Validate the --post flag pairing BEFORE any Gmail write, so a usage
    // error can't strand an orphan draft in the mailbox (#412).
    if post
        && thread_id.as_deref().map_or(false, |t| !t.is_empty())
            != reply_to_message_id.as_deref().map_or(false, |m| !m.is_empty())
    {
        anyhow::bail!(
            "--thread-id and --reply-to-message-id go together: pass both (reply card) \
             or neither (new-email card)"
        );
    }
    let (entity_id, email) = resolve_gmail_entity_id(&store, account)?;
    // #419 / #596 — preserve one active approval identity while allowing a
    // conversational follow-up to replace its body/envelope. The old action
    // is superseded only after the replacement Gmail draft and Discord card
    // both succeed, so a failed follow-up leaves the original actionable.
    let pending = if post && !allow_duplicate {
        store
            .find_pending_action_for_recipient(&entity_id, &to, &subject)
            .context("duplicate-guard lookup")?
    } else {
        None
    };
    // #500 — an ARMED scheduled send is not replaceable: the Replace arm's
    // supersede is pending-only, so "replacing" it would leave the old
    // schedule armed alongside the new card and both would eventually send.
    // Refuse and point at the escape hatches instead.
    if let Some((action_id, _, status)) = pending.as_ref() {
        if status == "scheduled" {
            anyhow::bail!(
                "a send to this recipient/subject is already SCHEDULED \
                 (action {action_id}). Cancel or send it from its Discord \
                 notice first, or pass --allow-duplicate to compose a \
                 separate email."
            );
        }
    }
    let pending_replacement = match compose_pending_disposition(
        post,
        allow_duplicate,
        pending
            .as_ref()
            .map(|(action_id, draft_id, _status)| (action_id.as_str(), draft_id.as_deref())),
    ) {
        ComposePendingDisposition::Replace {
            action_id,
            draft_id,
        } => Some((action_id, draft_id)),
        ComposePendingDisposition::Create => None,
    };
    let api_key = std::env::var("COMPOSIO_API_KEY").context("COMPOSIO_API_KEY env var required")?;
    let gmail = ComposioClient::new(api_key);
    let thread_id = resolve_compose_thread_id(&gmail, &entity_id, &email, thread_id).await?;
    if let Some(t) = thread_id.as_deref() {
        ensure_subject_matches_thread(&gmail, &entity_id, t, &subject).await?;
    }
    let attachment = upload_attach_if_given(&gmail, attach).await?;
    let draft_id = gmail
        .create_draft_with_attachment(
            &entity_id,
            &to,
            &subject,
            &body_str,
            thread_id.as_deref(),
            attachment.as_ref(),
            &cc,
            &bcc,
        )
        .await
        .context("create_draft via Composio failed")?;
    if post {
        post_reply_approval_card(
            &store,
            &entity_id,
            &email,
            &draft_id,
            &to,
            &cc,
            &bcc,
            &subject,
            &body_str,
            attachment.as_ref().map(|a| a.name.as_str()),
            thread_id.as_deref(),
            reply_to_message_id.as_deref(),
            reply_to_from.as_deref(),
            reply_to_subject.as_deref(),
            reply_to_body.as_deref(),
            reply_to_body_file.as_deref(),
            send_at_ms,
        )
        .await?;
        if let Some((old_action_id, old_draft_id)) = pending_replacement {
            match store.mark_pending_superseded_by_ids(
                std::slice::from_ref(&old_action_id),
                "superseded by follow-up compose",
            ) {
                Ok(1) => {
                    if let Some(old_draft_id) = old_draft_id {
                        if let Err(e) = gmail.delete_draft(&entity_id, &old_draft_id).await {
                            tracing::warn!(
                                action_id = %old_action_id,
                                draft_id = %old_draft_id,
                                "follow-up compose: delete superseded draft failed: {e}"
                            );
                        }
                    }
                    println!(
                        "approval card {old_action_id} superseded by the replacement card"
                    );
                }
                Ok(_) => tracing::warn!(
                    action_id = %old_action_id,
                    "follow-up compose: pending action was already resolved"
                ),
                Err(e) => tracing::warn!(
                    action_id = %old_action_id,
                    "follow-up compose: failed to supersede old action: {e}"
                ),
            }
        }
        if json {
            println!(
                "{}",
                serde_json::json!({
                    "draft_id": draft_id,
                    "account": email,
                    "entity_id": entity_id,
                    "to": to,
                    "cc": cc,
                    "bcc": bcc,
                    "subject": subject,
                    "thread_id": thread_id,
                    "attachment": attachment.as_ref().map(|a| a.name.clone()),
                    "approval_card_posted": true,
                })
            );
        } else {
            println!("approval card posted to Discord for draft {draft_id}");
        }
        return Ok(());
    }
    if json {
        println!(
            "{}",
            serde_json::json!({
                "draft_id": draft_id,
                "account": email,
                "entity_id": entity_id,
                "to": to,
                "cc": cc,
                "bcc": bcc,
                "subject": subject,
                "thread_id": thread_id,
                "attachment": attachment.as_ref().map(|a| a.name.clone()),
                "open_in_gmail": format!("https://mail.google.com/mail/u/0/#drafts?compose={draft_id}"),
            })
        );
    } else {
        println!("draft created: id={draft_id}");
        println!("account: {email}");
        println!("to:      {to}");
        if !cc.is_empty() {
            println!("cc:      {}", cc.join(", "));
        }
        if !bcc.is_empty() {
            println!("bcc:     {}", bcc.join(", "));
        }
        println!("subject: {subject}");
        if let Some(a) = &attachment {
            println!("attachment: {} ({})", a.name, a.mimetype);
        }
        println!("open in gmail: https://mail.google.com/mail/u/0/#drafts?compose={draft_id}");
    }
    Ok(())
}

/// #352 / #412 — Post a Discord approval card for a draft the wiki-ask agent
/// just composed in a chat session. Writes the same `actions`-table shape as
/// the auto-triage path so the daemon's Approve/Revise/Skip handlers don't
/// need to distinguish where the card came from. Upserts an `emails` row when
/// one isn't already present so the Revise redraft path has the original body
/// to work against.
///
/// Two shapes (#412):
/// - **Reply**: `--thread-id` + `--reply-to-message-id` present — the card is
///   anchored to the real inbound message, exactly the #352 flow.
/// - **New email**: both absent — there is no inbound, so the action is keyed
///   on a synthetic `compose:<draft_id>` message id (the same convention the
///   compose fan-out uses), the card's From field shows the recipient, and
///   Revise redrafts against the draft body alone. Approve → send_draft,
///   Skip → delete_draft, identical to replies.
#[allow(clippy::too_many_arguments)]
async fn post_reply_approval_card(
    store: &Store,
    entity_id: &str,
    account_email: &str,
    draft_id: &str,
    to: &str,
    cc: &[String],
    bcc: &[String],
    out_subject: &str,
    draft_body: &str,
    attachment_name: Option<&str>,
    thread_id: Option<&str>,
    reply_to_message_id: Option<&str>,
    reply_to_from: Option<&str>,
    reply_to_subject: Option<&str>,
    reply_to_body: Option<&str>,
    reply_to_body_file: Option<&str>,
    send_at_ms: Option<i64>,
) -> Result<()> {
    use augmentagent_store::Email as StoreEmail;

    let thread = thread_id.filter(|t| !t.is_empty());
    let reply_msg_id = reply_to_message_id.filter(|s| !s.is_empty());
    // A reply anchor only makes sense as a pair: a message id without its
    // thread (or vice versa) would post a card whose Approve sends an
    // unthreaded reply while claiming otherwise. run_gmail_compose validates
    // this BEFORE creating the draft; re-check defensively for other callers.
    anyhow::ensure!(
        thread.is_some() == reply_msg_id.is_some(),
        "--thread-id and --reply-to-message-id go together: pass both (reply card) \
         or neither (new-email card)"
    );
    // New-email cards (#412): key the action on a synthetic message id so the
    // approve/skip/revise handlers — which join actions→emails on messageId —
    // find the upserted row below. `compose:` prefix matches the fan-out
    // convention and can never collide with a real Gmail hex id.
    let synthetic_msg_id = format!("compose:{draft_id}");
    let msg_id: &str = reply_msg_id.unwrap_or(&synthetic_msg_id);
    let token = std::env::var("DISCORD_BOT_TOKEN")
        .context("DISCORD_BOT_TOKEN required for --post (set it in the daemon's .env)")?;
    let cid: u64 = std::env::var("DISCORD_CHANNEL_ID")
        .context("DISCORD_CHANNEL_ID required for --post")?
        .parse()
        .context("DISCORD_CHANNEL_ID must be numeric")?;

    let original_body = match (reply_to_body, reply_to_body_file) {
        (Some(_), Some(_)) => {
            anyhow::bail!("--reply-to-body and --reply-to-body-file are mutually exclusive");
        }
        (Some(b), None) => b.to_string(),
        (None, Some(p)) => read_body(None, Some(p.to_string()))?,
        (None, None) => String::new(),
    };
    let original_from = reply_to_from.unwrap_or(to).to_string();
    let stripped = out_subject
        .strip_prefix("Re:")
        .or_else(|| out_subject.strip_prefix("re:"))
        .or_else(|| out_subject.strip_prefix("RE:"))
        .unwrap_or(out_subject)
        .trim_start()
        .to_string();
    let original_subject = reply_to_subject
        .map(str::to_string)
        .unwrap_or_else(|| if stripped.is_empty() { out_subject.to_string() } else { stripped });

    // Upsert an inbound row when the daemon hasn't already ingested this
    // message. The Revise handler reads `emails.body` to redraft against the
    // original; without a row, revise has nothing to redraft. When the row
    // already exists (the auto-triage path saw it first), upsert refreshes
    // its bytes — same shape the email channel writes. For a new-email card
    // the row is purely synthetic (empty body, no thread): it exists so the
    // actions→emails join resolves and Approve can find the entity id.
    let inbound = StoreEmail {
        attachments: Vec::new(),
        message_id: msg_id.to_string(),
        to: String::new(),
        cc: String::new(),
        thread_id: thread.map(str::to_string),
        from: original_from.clone(),
        subject: original_subject.clone(),
        body: original_body.clone(),
        date: String::new(),
        account_entity_id: Some(entity_id.to_string()),
        platform: "gmail".into(),
        kind: "dm".into(),
    };
    store
        .upsert_email(&inbound)
        .context("upsert inbound email for action linkage")?;

    let action_id = store
        .log_action(
            msg_id,
            thread,
            &original_from,
            &original_subject,
            Some(&original_body),
            Some(draft_body),
            ActionStatus::Pending,
        )
        .context("log action row")?;
    store
        .set_action_draft_id(&action_id, draft_id)
        .context("set action draft id")?;
    // #473 — record the envelope the draft was actually created with, so the
    // Revise redraft re-sends to the SAME To/cc/bcc instead of falling back
    // to the card's From (the original sender on reply cards — which dropped
    // an overridden To and every cc/bcc). Best-effort: a failure leaves the
    // pre-#473 from-based revise behavior, never blocks the card.
    if let Err(e) = store.set_action_envelope(
        &action_id,
        Some(to),
        Some(&cc.join(", ")),
        Some(&bcc.join(", ")),
    ) {
        tracing::warn!(action_id, "set_action_envelope after compose card failed: {e}");
    }
    // #502 — persist the --send-at proposal on the still-PENDING row. The
    // owner still approves; run_approve sees the future proposal and ARMS
    // the schedule instead of sending. Back-to-queue and unschedule clear
    // it, so a reposted card's Approve sends immediately again.
    if let Some(at_ms) = send_at_ms {
        if let Err(e) = store.set_action_scheduled_at(&action_id, Some(at_ms)) {
            tracing::warn!(
                action_id,
                "set_action_scheduled_at after compose card failed: {e}"
            );
        }
    }

    let http = serenity::http::Http::new(&token);
    let channel = serenity::all::ChannelId::new(cid);
    // The card must SHOW the attachment (#417 — "can't see they've been
    // attached") and any cc/bcc (#439 — "render all recipients"). Display
    // only: the actions row keeps the clean body so the Revise redraft
    // prompt isn't polluted with the marker lines.
    let mut markers = String::new();
    // #473 — when the envelope To differs from the card's From display (a
    // reply card whose routing was overridden, e.g. the intro pattern:
    // reply-to Josh, To Omer), surface it: the whole bug was body and
    // envelope disagreeing with nothing on the card to show it.
    if !original_from.eq_ignore_ascii_case(to) {
        markers.push_str(&format!("\n[to: {to}]"));
    }
    if !cc.is_empty() {
        markers.push_str(&format!("\n[cc: {}]", cc.join(", ")));
    }
    if !bcc.is_empty() {
        markers.push_str(&format!("\n[bcc: {}]", bcc.join(", ")));
    }
    if let Some(name) = attachment_name {
        markers.push_str(&format!("\n[attachment: {name}]"));
    }
    // #502 — surface the proposed send time on the card: Approve on this
    // card arms a schedule, and the owner must see that before clicking.
    if let Some(at_ms) = send_at_ms {
        markers.push_str(&format!("\n[sends: {}]", format_local_send_time(at_ms)));
    }
    let card_body = if markers.is_empty() {
        draft_body.to_string()
    } else {
        format!("{draft_body}\n{markers}")
    };
    let card = approval_message(&action_id, &inbound, &card_body, 0);
    channel
        .send_message(&http, card)
        .await
        .context("send approval card to Discord")?;

    // Mark this action as the ACTIVE nudge (count 0 → 1). Without this the
    // daemon's serial-queue scheduler sees a pending row with nudgeCount=0
    // and promotes it — posting a duplicate card for the draft the user is
    // already looking at (#412; latent since #352).
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0i64);
    if let Err(e) =
        store.record_nudge(&action_id, now_ms + augmentagent_store::NUDGE_INTERVAL_MS)
    {
        tracing::warn!(action_id, "record_nudge after compose card failed: {e}");
    }

    let _ = account_email;
    Ok(())
}


/// Raise a Discord approval card for an operator-initiated social draft
/// (#571 / #572).
///
/// This is the social counterpart of [`post_reply_approval_card`] and follows
/// the same contract deliberately, because the approve/revise/skip handlers
/// and the nudge scheduler all key off the same rows:
///
///   1. upsert an `emails` row so the actions→emails join resolves and the
///      Revise handler has something to redraft against;
///   2. log a `Pending` action carrying the draft body;
///   3. post the card;
///   4. claim the nudge slot so the serial-queue scheduler doesn't promote
///      the same row and post a duplicate card.
///
/// `platform`, `kind`, `thread_id` and `message_id` are what the approve
/// handlers dispatch on — see `approve_socialapi` / `approve_linkedin`. Get
/// them wrong and Approve either fails or sends to the wrong target, so each
/// caller documents its own mapping.
#[allow(clippy::too_many_arguments)]
async fn post_social_approval_card(
    store: &Store,
    platform: &str,
    kind: &str,
    message_id: &str,
    thread_id: &str,
    account_entity_id: Option<&str>,
    counterparty: &str,
    subject: &str,
    context_body: &str,
    draft_body: &str,
) -> Result<String> {
    use augmentagent_store::Email as StoreEmail;

    anyhow::ensure!(
        !draft_body.trim().is_empty(),
        "refusing to card an empty draft body"
    );
    anyhow::ensure!(
        !thread_id.trim().is_empty(),
        "a send target is required: Approve routes on the email's thread id"
    );

    let token = std::env::var("DISCORD_BOT_TOKEN")
        .context("DISCORD_BOT_TOKEN required for --post (set it in the daemon's .env)")?;
    let cid: u64 = std::env::var("DISCORD_CHANNEL_ID")
        .context("DISCORD_CHANNEL_ID required for --post")?
        .parse()
        .context("DISCORD_CHANNEL_ID must be numeric")?;

    let inbound = StoreEmail {
        attachments: Vec::new(),
        message_id: message_id.to_string(),
        to: String::new(),
        cc: String::new(),
        thread_id: Some(thread_id.to_string()),
        from: counterparty.to_string(),
        subject: subject.to_string(),
        body: context_body.to_string(),
        date: String::new(),
        account_entity_id: account_entity_id.map(str::to_string),
        platform: platform.to_string(),
        kind: kind.to_string(),
    };
    store
        .upsert_email(&inbound)
        .context("upsert inbound row for action linkage")?;

    let action_id = store
        .log_action(
            message_id,
            Some(thread_id),
            counterparty,
            subject,
            Some(context_body),
            Some(draft_body),
            ActionStatus::Pending,
        )
        .context("log action row")?;

    let http = serenity::http::Http::new(&token);
    let channel = serenity::all::ChannelId::new(cid);
    let card = approval_message(&action_id, &inbound, draft_body, 0);
    if let Err(e) = channel.send_message(&http, card).await {
        // The rows are already written — `approval_message` needs the action
        // id, so the action has to exist before the card can be built. A
        // failed send would therefore strand a `pending` action that nobody
        // can see, which the nudge scheduler later promotes into a surprise
        // card for a draft the operator never asked about. Mark it Error so
        // it drops out of the pending queue and shows up as a failure
        // instead.
        if let Err(e2) = store.update_action_status(
            &action_id,
            ActionStatus::Error,
            None,
            Some(&format!("approval card send failed: {e}")),
        ) {
            tracing::warn!(action_id, "could not mark orphaned action as errored: {e2}");
        }
        return Err(anyhow::Error::new(e).context("send approval card to Discord"));
    }

    // Claim the nudge slot (count 0 → 1). Without this the daemon's
    // serial-queue scheduler sees a pending row with nudgeCount=0, promotes
    // it, and posts a second card for the draft already on screen — the #412
    // bug, which is latent for every new card-raising path.
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0i64);
    if let Err(e) =
        store.record_nudge(&action_id, now_ms + augmentagent_store::NUDGE_INTERVAL_MS)
    {
        tracing::warn!(action_id, "record_nudge after social card failed: {e}");
    }
    Ok(action_id)
}


/// `socialapi dm` (#571) — draft a DM reply and optionally card it.
///
/// The email row this writes is what `approve_socialapi` dispatches on:
/// `platform = "socialapi"`, `kind = "dm"`, and `thread_id` = the
/// conversation id it will `send_dm` to. `account_entity_id` carries the
/// owning account, which the send request needs.
#[allow(clippy::too_many_arguments)]
async fn run_socialapi_dm(
    store: Arc<Store>,
    conversation_id: String,
    account_id: Option<String>,
    with: Option<String>,
    platform: Option<String>,
    body: Option<String>,
    body_file: Option<String>,
    in_reply_to: Option<String>,
    post: bool,
    json: bool,
) -> Result<()> {
    let draft = read_body(body, body_file)?;
    anyhow::ensure!(!draft.trim().is_empty(), "--body (or --body-file) is required");
    let counterparty = with.unwrap_or_else(|| "them".to_string());
    let subject = match augmentagent_channel_socialapi::platform_label(
        platform.as_deref().unwrap_or(""),
    ) {
        Some(p) => format!("[{p} DM from {counterparty}]"),
        None => format!("[DM from {counterparty}]"),
    };
    if !post {
        if json {
            println!(
                "{}",
                serde_json::json!({
                    "conversation_id": conversation_id,
                    "subject": subject,
                    "draft": draft,
                    "approval_card_posted": false,
                })
            );
        } else {
            println!("{subject}\n\n{draft}");
        }
        return Ok(());
    }
    // Synthetic id: this is an operator-initiated draft, not an ingested
    // message, so there is no platform message id to key on. `compose:`
    // mirrors the gmail convention and can never collide with a real one.
    let message_id = format!("compose:socialapi:dm:{conversation_id}");
    let action_id = post_social_approval_card(
        &store,
        augmentagent_channel_socialapi::PLATFORM,
        augmentagent_channel_core::trigger::kind::DM,
        &message_id,
        &conversation_id,
        account_id.as_deref(),
        &counterparty,
        &subject,
        in_reply_to.as_deref().unwrap_or(""),
        &draft,
    )
    .await?;
    if json {
        println!(
            "{}",
            serde_json::json!({
                "action_id": action_id,
                "conversation_id": conversation_id,
                "subject": subject,
                "approval_card_posted": true,
            })
        );
    } else {
        println!("approval card posted to Discord (action {action_id})");
    }
    Ok(())
}

/// `socialapi comment` (#571) — draft a reply to a comment on your own post.
///
/// `approve_socialapi` reads `thread_id` as the post to reply under and
/// `message_id` as the parent comment id, so the reply threads correctly.
#[allow(clippy::too_many_arguments)]
async fn run_socialapi_comment(
    store: Arc<Store>,
    post_id: String,
    comment_id: String,
    account_id: Option<String>,
    author: Option<String>,
    platform: Option<String>,
    body: Option<String>,
    body_file: Option<String>,
    in_reply_to: Option<String>,
    post: bool,
    json: bool,
) -> Result<()> {
    let draft = read_body(body, body_file)?;
    anyhow::ensure!(!draft.trim().is_empty(), "--body (or --body-file) is required");
    let author = author.unwrap_or_else(|| "them".to_string());
    let subject = match augmentagent_channel_socialapi::platform_label(
        platform.as_deref().unwrap_or(""),
    ) {
        Some(p) => format!("[{p} comment on your post by {author}]"),
        None => format!("[Comment on your post by {author}]"),
    };
    if !post {
        if json {
            println!(
                "{}",
                serde_json::json!({
                    "post_id": post_id,
                    "comment_id": comment_id,
                    "subject": subject,
                    "draft": draft,
                    "approval_card_posted": false,
                })
            );
        } else {
            println!("{subject}\n\n{draft}");
        }
        return Ok(());
    }
    // NOT synthetic: approve_socialapi passes `email.message_id` as the
    // parent `comment_id` on the reply, so it must be the real platform id.
    let action_id = post_social_approval_card(
        &store,
        augmentagent_channel_socialapi::PLATFORM,
        augmentagent_channel_core::trigger::kind::OWN_POST_COMMENT,
        &comment_id,
        &post_id,
        account_id.as_deref(),
        &author,
        &subject,
        in_reply_to.as_deref().unwrap_or(""),
        &draft,
    )
    .await?;
    if json {
        println!(
            "{}",
            serde_json::json!({
                "action_id": action_id,
                "post_id": post_id,
                "comment_id": comment_id,
                "approval_card_posted": true,
            })
        );
    } else {
        println!("approval card posted to Discord (action {action_id})");
    }
    Ok(())
}


/// `linkedin dm` (#572) — draft a DM reply and optionally card it.
///
/// `approve_linkedin` treats any kind that is NOT `post_engagement` as a DM
/// and calls `send_message(email.thread_id)`, so the conversation urn goes on
/// `thread_id` and the kind is `dm`.
#[allow(clippy::too_many_arguments)]
async fn run_linkedin_dm(
    store: Arc<Store>,
    conversation_urn: String,
    with: Option<String>,
    body: Option<String>,
    body_file: Option<String>,
    in_reply_to: Option<String>,
    post: bool,
    json: bool,
) -> Result<()> {
    let draft = read_body(body, body_file)?;
    anyhow::ensure!(!draft.trim().is_empty(), "--body (or --body-file) is required");
    let counterparty = with.unwrap_or_else(|| "them".to_string());
    let subject = format!("[LinkedIn DM from {counterparty}]");
    if !post {
        if json {
            println!(
                "{}",
                serde_json::json!({
                    "conversation_urn": conversation_urn,
                    "subject": subject,
                    "draft": draft,
                    "approval_card_posted": false,
                })
            );
        } else {
            println!("{subject}\n\n{draft}");
        }
        return Ok(());
    }
    let message_id = format!("compose:linkedin:dm:{conversation_urn}");
    let action_id = post_social_approval_card(
        &store,
        "linkedin",
        augmentagent_channel_core::trigger::kind::DM,
        &message_id,
        &conversation_urn,
        None,
        &counterparty,
        &subject,
        in_reply_to.as_deref().unwrap_or(""),
        &draft,
    )
    .await?;
    if json {
        println!(
            "{}",
            serde_json::json!({
                "action_id": action_id,
                "conversation_urn": conversation_urn,
                "approval_card_posted": true,
            })
        );
    } else {
        println!("approval card posted to Discord (action {action_id})");
    }
    Ok(())
}

/// `linkedin comment` (#572) — draft a comment on a post and optionally card
/// it.
///
/// `approve_linkedin` dispatches `kind == "post_engagement"` to
/// `post_comment(email.message_id)`, so the POST URN goes on `message_id`,
/// not `thread_id`. `thread_id` still carries it so the card has a stable
/// send target for the shared helper's invariant.
#[allow(clippy::too_many_arguments)]
async fn run_linkedin_comment(
    store: Arc<Store>,
    post_urn: String,
    author: Option<String>,
    body: Option<String>,
    body_file: Option<String>,
    in_reply_to: Option<String>,
    post: bool,
    json: bool,
) -> Result<()> {
    let draft = read_body(body, body_file)?;
    anyhow::ensure!(!draft.trim().is_empty(), "--body (or --body-file) is required");
    let author = author.unwrap_or_else(|| "them".to_string());
    let subject = format!("[LinkedIn comment on a post by {author}]");
    if !post {
        if json {
            println!(
                "{}",
                serde_json::json!({
                    "post_urn": post_urn,
                    "subject": subject,
                    "draft": draft,
                    "approval_card_posted": false,
                })
            );
        } else {
            println!("{subject}\n\n{draft}");
        }
        return Ok(());
    }
    let action_id = post_social_approval_card(
        &store,
        "linkedin",
        "post_engagement",
        &post_urn,
        &post_urn,
        None,
        &author,
        &subject,
        in_reply_to.as_deref().unwrap_or(""),
        &draft,
    )
    .await?;
    if json {
        println!(
            "{}",
            serde_json::json!({
                "action_id": action_id,
                "post_urn": post_urn,
                "approval_card_posted": true,
            })
        );
    } else {
        println!("approval card posted to Discord (action {action_id})");
    }
    Ok(())
}


/// `linkedin recent-dms` — list recent DM threads and their conversation urns.
///
/// Read-only: one Voyager GET, no writes, nothing sent. Its whole purpose is
/// to hand the agent the `--conversation-urn` that `linkedin dm` requires.
async fn run_linkedin_recent_dms(
    repo_root: PathBuf,
    limit: usize,
    json: bool,
) -> Result<()> {
    let auth = LinkedInAuth::load_with_migration(&repo_root)
        .context("load linkedin auth (run `linkedin login`)")?;
    let my_urn = auth.member_urn.clone();
    let voyager = VoyagerClient::new(auth);
    let dms = voyager
        .fetch_recent_dms()
        .await
        .context("fetch recent linkedin dms")?;
    // Drop our own outbound messages: you don't reply to yourself, and they
    // would only pad the list the agent has to disambiguate.
    let rows: Vec<_> = dms
        .into_iter()
        .filter(|d| !d.is_outbound(&my_urn))
        .take(limit)
        .collect();
    if json {
        let out: Vec<_> = rows
            .iter()
            .map(|d| {
                serde_json::json!({
                    "conversation_urn": d.conversation_urn,
                    "message_urn": d.message_urn,
                    "peer_name": d.peer_name,
                    "text": d.text,
                    "delivered_at_ms": d.delivered_at_ms,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }
    if rows.is_empty() {
        println!("(no recent inbound LinkedIn DMs)");
        return Ok(());
    }
    for d in &rows {
        let preview: String = d.text.chars().take(70).collect();
        println!("{}\n  from: {}\n  {}\n", d.conversation_urn, d.peer_name, preview);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn run_gmail_update_draft(
    store: Arc<Store>,
    account: Option<String>,
    draft_id: String,
    to: Vec<String>,
    cc: Vec<String>,
    bcc: Vec<String>,
    subject: String,
    body: Option<String>,
    body_file: Option<String>,
    thread_id: Option<String>,
    attach: Option<PathBuf>,
) -> Result<()> {
    let to = normalize_recipients("--to", &to)?;
    anyhow::ensure!(!to.is_empty(), "--to requires at least one email address");
    let to = to.join(", ");
    let cc = normalize_recipients("--cc", &cc)?;
    let bcc = normalize_recipients("--bcc", &bcc)?;
    let body_str = body_for_gmail_write(read_body(body, body_file)?, &subject)?;
    let (entity_id, email) = resolve_gmail_entity_id(&store, account)?;
    // #500 — refuse while a send of exactly this draft is in flight: update
    // is create-replacement + DELETE-old, which would yank the draft out
    // from under the Composio send. The window is short (bounded by
    // SEND_DRAFT_TIMEOUT); retry in a couple of minutes.
    if store.draft_id_in_flight(&draft_id).unwrap_or(false) {
        anyhow::bail!(
            "draft {draft_id} is being sent right now (action in 'sending'); \
             wait for the send to finish before updating it"
        );
    }
    let api_key = std::env::var("COMPOSIO_API_KEY").context("COMPOSIO_API_KEY env var required")?;
    let gmail = ComposioClient::new(api_key);
    // Update = create replacement + delete old (#382). Keep the reply
    // threaded: use the explicit --thread-id when given, otherwise detect
    // the old draft's thread. Detection is best-effort — a lookup failure
    // downgrades to an unthreaded draft rather than blocking the update.
    let thread_id = match thread_id.filter(|t| !t.is_empty()) {
        Some(t) => resolve_compose_thread_id(&gmail, &entity_id, &email, Some(t)).await?,
        None => match gmail.get_draft_thread_id(&entity_id, &draft_id).await {
            Ok(t) => t,
            Err(e) => {
                eprintln!(
                    "warning: could not determine draft {draft_id}'s thread ({e}); \
                     the replacement draft will start a new thread"
                );
                None
            }
        },
    };
    if let Some(t) = thread_id.as_deref() {
        ensure_subject_matches_thread(&gmail, &entity_id, t, &subject).await?;
    }
    let attachment = upload_attach_if_given(&gmail, attach).await?;
    let new_id = gmail
        .update_draft_with_attachment(
            &entity_id,
            &draft_id,
            &to,
            &subject,
            &body_str,
            thread_id.as_deref(),
            attachment.as_ref(),
            &cc,
            &bcc,
        )
        .await
        .context("update_draft (create replacement + delete old) via Composio failed")?;
    // #419 card-sync — any pending approval card pointing at the replaced
    // draft would otherwise Approve a deleted id. Repoint it at the new
    // draft and refresh the stored body so Revise sees the current text.
    match store.find_pending_action_ids_by_draft_id(&draft_id) {
        Ok(ids) => {
            for action_id in ids {
                if let Err(e) = store.set_action_draft_id(&action_id, &new_id) {
                    eprintln!("warning: card {action_id} not repointed to new draft: {e}");
                    continue;
                }
                if let Err(e) = store.set_action_draft_body(&action_id, &body_str) {
                    eprintln!("warning: card {action_id} body not refreshed: {e}");
                }
                // #473 — the replacement draft's envelope is now the card's
                // envelope; keep the Revise carry-through in step with it.
                if let Err(e) = store.set_action_envelope(
                    &action_id,
                    Some(&to),
                    Some(&cc.join(", ")),
                    Some(&bcc.join(", ")),
                ) {
                    eprintln!("warning: card {action_id} envelope not refreshed: {e}");
                }
                println!("approval card {action_id} now follows the new draft");
            }
        }
        Err(e) => eprintln!("warning: card-sync lookup failed: {e}"),
    }
    println!("draft updated: new id={new_id} (replaces {draft_id}) account={email}");
    if let Some(t) = thread_id {
        println!("thread:  {t}");
    }
    println!("open in gmail: https://mail.google.com/mail/u/0/#drafts?compose={new_id}");
    Ok(())
}

/// #449 — remember that WE sent this message, so the OutboundObserver's SENT
/// scan skips it instead of recording it as a reply the user wrote by hand.
///
/// Every daemon send path must funnel through here. Missing a call site is not
/// cosmetic: the observer would treat that send as the user replying, supersede
/// the live drafts on the thread, and make the already-replied guard silently
/// suppress every future draft on it.
///
/// Best-effort by construction — the mail has already gone out by the time we
/// are called, so a bookkeeping failure must never surface as a send failure.
/// It is logged loudly instead.
/// #502 — is a --send-at proposal still worth arming at Approve time?
/// A proposal at or inside the minimum lead expired naturally: approving it
/// means "send it", and run_schedule's validator would reject it anyway —
/// falling through to the immediate send is both the intuitive outcome and
/// the only non-dead-end one.
fn send_at_proposal_is_live(at_ms: i64, now_ms: i64) -> bool {
    at_ms > now_ms + augmentagent_channel_core::timeparse::MIN_LEAD_MS
}

fn record_self_send(
    store: &Store,
    sent_message_id: Option<&str>,
    thread_id: Option<&str>,
    entity_id: Option<&str>,
    action_id: Option<&str>,
) {
    // #500 — single implementation shared with the ScheduledSendEngine, so
    // the empty-id tolerance and the loud-failure logging can't drift apart
    // between the two send paths.
    augmentagent_channel_email::record_self_send(
        store,
        sent_message_id,
        thread_id,
        entity_id,
        action_id,
    );
}

async fn run_gmail_send_draft(
    store: Arc<Store>,
    account: Option<String>,
    draft_id: String,
) -> Result<()> {
    let (entity_id, email) = resolve_gmail_entity_id(&store, account)?;
    let api_key = std::env::var("COMPOSIO_API_KEY").context("COMPOSIO_API_KEY env var required")?;
    let gmail = ComposioClient::new(api_key);
    let sent_id = gmail
        .send_draft(&entity_id, &draft_id)
        .await
        .context("send_draft via Composio failed")?;
    record_self_send(&store, sent_id.as_deref(), None, Some(&entity_id), None);
    println!("sent: draft={draft_id} account={email}");
    Ok(())
}

async fn run_gmail_delete_draft(
    store: Arc<Store>,
    account: Option<String>,
    draft_id: String,
) -> Result<()> {
    let (entity_id, email) = resolve_gmail_entity_id(&store, account)?;
    let api_key = std::env::var("COMPOSIO_API_KEY").context("COMPOSIO_API_KEY env var required")?;
    let gmail = ComposioClient::new(api_key);
    gmail
        .delete_draft(&entity_id, &draft_id)
        .await
        .context("delete_draft via Composio failed")?;
    println!("deleted: draft={draft_id} account={email}");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn run_gmail_send_now(
    store: Arc<Store>,
    account: Option<String>,
    to: Vec<String>,
    cc: Vec<String>,
    bcc: Vec<String>,
    subject: String,
    body: Option<String>,
    body_file: Option<String>,
    thread_id: Option<String>,
    attach: Option<PathBuf>,
) -> Result<()> {
    let to = normalize_recipients("--to", &to)?;
    anyhow::ensure!(!to.is_empty(), "--to requires at least one email address");
    let to = to.join(", ");
    let cc = normalize_recipients("--cc", &cc)?;
    let bcc = normalize_recipients("--bcc", &bcc)?;
    let body_str = body_for_gmail_write(read_body(body, body_file)?, &subject)?;
    let (entity_id, email) = resolve_gmail_entity_id(&store, account)?;
    let api_key = std::env::var("COMPOSIO_API_KEY").context("COMPOSIO_API_KEY env var required")?;
    let gmail = ComposioClient::new(api_key);
    let thread_id = resolve_compose_thread_id(&gmail, &entity_id, &email, thread_id).await?;
    if let Some(t) = thread_id.as_deref() {
        ensure_subject_matches_thread(&gmail, &entity_id, t, &subject).await?;
    }
    let attachment = upload_attach_if_given(&gmail, attach).await?;
    let draft_id = gmail
        .create_draft_with_attachment(
            &entity_id,
            &to,
            &subject,
            &body_str,
            thread_id.as_deref(),
            attachment.as_ref(),
            &cc,
            &bcc,
        )
        .await
        .context("create_draft (send-now) failed")?;
    let sent_id = gmail
        .send_draft(&entity_id, &draft_id)
        .await
        .context("send_draft (send-now) failed")?;
    record_self_send(
        &store,
        sent_id.as_deref(),
        thread_id.as_deref(),
        Some(&entity_id),
        None,
    );
    print!("sent: account={email} to={to}");
    if !cc.is_empty() {
        print!(" cc={}", cc.join(", "));
    }
    if !bcc.is_empty() {
        print!(" bcc={}", bcc.join(", "));
    }
    println!(" subject=\"{subject}\" draft_id={draft_id}");
    Ok(())
}

/// Resolve each active connected Gmail's address via Composio
/// `GMAIL_GET_PROFILE` and persist it to `gmail_accounts.email`. The OAuth
/// connect flow never captured it, so without this the dashboard + invoice
/// entity picker can only show opaque IDs.
///
/// `only_missing` limits work to rows whose email is still blank (the
/// self-healing startup pass). Best-effort: a flaky/expired account is logged
/// in the returned mapping and skipped, never aborting the whole sweep.
async fn backfill_gmail_emails(store: &Store, only_missing: bool) -> Result<Vec<String>> {
    let api_key =
        std::env::var("COMPOSIO_API_KEY").context("COMPOSIO_API_KEY env var required")?;
    let gmail = ComposioClient::new(api_key);
    let accounts = store.get_active_gmail_accounts()?;
    let mut lines = Vec::new();
    for a in accounts {
        if only_missing && !a.email.is_empty() {
            continue;
        }
        match gmail.get_profile_email(&a.entity_id).await {
            Ok(email) => {
                store.update_gmail_account_email(&a.id, &email)?;
                lines.push(format!("{}\t{}\t{}", a.entity_id, email, a.id));
            }
            Err(e) => {
                lines.push(format!("{}\t<lookup failed: {e}>\t{}", a.entity_id, a.id));
            }
        }
    }
    Ok(lines)
}

/// #219 — periodically observe outbound (SENT) Gmail and supersede stale
/// pending drafts on threads the user replied to out-of-band.
///
/// DEFAULT-ON as of #449 (opt out with `AUGMENTAGENT_OUTBOUND_OBSERVER=0`).
/// The caller already checks the env var before spawning this — this function
/// assumes it should run. Tick interval defaults to 5 min, override via
/// `AUGMENTAGENT_OUTBOUND_OBSERVER_INTERVAL_SECS`. Failure of a single
/// `poll_once` is logged and swallowed; the daemon must never crash on a
/// transient Composio hiccup. Wiki-ingest of the emitted events is
/// intentionally deferred to a follow-on (see `outbound.rs` doc).
async fn run_outbound_observer(
    store: Arc<Store>,
    shutdown: CancellationToken,
) -> Result<()> {
    let api_key = match std::env::var("COMPOSIO_API_KEY") {
        Ok(v) if !v.is_empty() => v,
        _ => {
            warn!(
                "outbound observer enabled but COMPOSIO_API_KEY is unset — \
                 observer NOT started; approval cards will not be retired when \
                 you reply from Gmail web/mobile"
            );
            return Ok(());
        }
    };
    let interval_secs: u64 = std::env::var("AUGMENTAGENT_OUTBOUND_OBSERVER_INTERVAL_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|n| *n >= 30)
        .unwrap_or(300);
    let gmail = Arc::new(ComposioClient::new(api_key).with_rate_limit_store(Arc::clone(&store)));
    let observer = OutboundObserver::new(store, gmail, 200);
    let mut ticker = tokio::time::interval(Duration::from_secs(interval_secs));
    info!(interval_secs, "outbound observer started");
    loop {
        tokio::select! {
            _ = shutdown.cancelled() => {
                info!("outbound observer: shutdown signal received");
                return Ok(());
            }
            _ = ticker.tick() => {
                match observer.poll_once().await {
                    Ok(events) if events.is_empty() => {}
                    Ok(events) => info!(
                        new_outbound = events.len(),
                        "outbound observer: classified user-authored sends"
                    ),
                    Err(e) => warn!("outbound observer: poll_once failed: {e:#}"),
                }
            }
        }
    }
}

/// Configured cutoff (in days) for the periodic auto-expire sweep (#220).
///
/// Resolution order:
/// 1. `AUGMENTAGENT_APPROVAL_AUTO_EXPIRE_DAYS` (the #220 canonical name).
/// 2. `AUGMENTAGENT_STALE_DRAFT_DAYS` (the #99 legacy name — kept so
///    existing deploys don't silently flip back to defaults on upgrade).
/// 3. Default `7` (one week — matches the user's verbatim request).
///
/// Returns `None` when the effective value is `0`, the user-facing
/// "disabled" toggle (e.g. on vacation, where we *want* the backlog to
/// pile up rather than auto-expire genuinely-important drafts).
fn auto_expire_days_from_env() -> Option<i64> {
    let raw = std::env::var("AUGMENTAGENT_APPROVAL_AUTO_EXPIRE_DAYS")
        .or_else(|_| std::env::var("AUGMENTAGENT_STALE_DRAFT_DAYS"))
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .filter(|n| *n >= 0)
        .unwrap_or(7);
    if raw == 0 {
        None
    } else {
        Some(raw)
    }
}

/// Configured tick interval (in seconds) for the periodic auto-expire
/// sweep (#220). Env-tunable via
/// `AUGMENTAGENT_APPROVAL_AUTO_EXPIRE_INTERVAL_SECS`; defaults to one
/// hour. Clamped to a minimum of 1s so a fat-fingered `0` can't busy-loop
/// the sweeper.
fn auto_expire_interval_secs_from_env() -> u64 {
    std::env::var("AUGMENTAGENT_APPROVAL_AUTO_EXPIRE_INTERVAL_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|n| *n >= 1)
        .unwrap_or(60 * 60)
}

/// One tick of the auto-expire sweep, extracted from the async loop so
/// unit tests can drive it deterministically against a fresh store
/// without spinning a tokio runtime. Returns the ids that were
/// transitioned this tick (empty when `days == None`, i.e. the sweep is
/// disabled — the test contract from #220).
fn sweep_stale_drafts_tick(store: &Store, days: Option<i64>) -> Result<Vec<String>> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let Some(days) = days else {
        return Ok(Vec::new());
    };
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);
    let cutoff_ms = now_ms - days * 24 * 60 * 60 * 1000;
    Ok(store.expire_pending_older_than(cutoff_ms)?)
}

/// Periodic stale-draft auto-expire (#99 / #220). Expires pending
/// approvals older than `AUGMENTAGENT_APPROVAL_AUTO_EXPIRE_DAYS` (default
/// 7, `0` = disabled, legacy alias `AUGMENTAGENT_STALE_DRAFT_DAYS`) to
/// `timed_out`, so an abandoned backlog can't permanently wedge new
/// triage behind the cap-at-25 backpressure (#99). Tick interval is
/// `AUGMENTAGENT_APPROVAL_AUTO_EXPIRE_INTERVAL_SECS` (default 3600).
///
/// Best-effort: a failed sweep logs and retries next tick; it never
/// takes the daemon down. Per #220, emits one `info!` line per expired
/// id so the audit trail is grep-able in the stdout log.
///
/// When the cutoff is `0` (disabled) the function logs the disabled
/// state once and returns immediately — no idle tokio task burning a
/// timer slot. This is the vacation-mode toggle.
/// #449 — how often the staleness reconciliation re-checks the queue. The
/// observer supersedes threads the instant it sees a user reply, so this loop
/// is the slow backstop for the rules the observer can't see (bulk senders) and
/// for anything that accumulated while the observer was off. 30 minutes.
const STALE_RECONCILE_INTERVAL_SECS: u64 = 1800;

/// #500 — scheduled-send engine cadence. Values <= 0 or unparsable fall back
/// to the 60s default.
fn scheduled_send_interval_secs_from_env() -> u64 {
    std::env::var("AUGMENTAGENT_SCHEDULED_SEND_INTERVAL_SECS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or_else(|| {
            augmentagent_channel_email::scheduled::DEFAULT_TICK.as_secs()
        })
}


/// #449 — retire approval cards that no longer deserve the user's attention,
/// without being asked to.
///
/// Two rules, both decidable from state we already hold:
///
/// 1. **The user already answered the thread.** `outbound_thread_log` (fed by
///    the OutboundObserver) knows every reply the user sent from Gmail
///    web/mobile. A pending card on such a thread is asking the user to answer
///    mail they have already answered.
/// 2. **The sender is a bulk/marketing address.** These should never have been
///    drafted at all (#451). Under the tightened `is_human_sender` rules they
///    no longer will be — but ~100 of them are already sitting in the queue,
///    and they are what pushed it past the old backpressure cap and starved
///    real threads of drafts (#450).
///
/// Superseding (not deleting) keeps the audit trail: the row stays, with the
/// reason in `errorMessage`, and simply stops being served by the carousel.
async fn run_stale_approval_reconcile(
    store: Arc<Store>,
    shutdown: CancellationToken,
) -> Result<()> {
    let mut ticker = tokio::time::interval(Duration::from_secs(STALE_RECONCILE_INTERVAL_SECS));
    info!(
        interval_secs = STALE_RECONCILE_INTERVAL_SECS,
        "stale approval reconciliation started"
    );
    loop {
        tokio::select! {
            _ = shutdown.cancelled() => {
                info!("stale approval reconciliation: shutdown signal received");
                return Ok(());
            }
            // Fires immediately on the first tick, which is what clears the
            // backlog left behind by the observer having been off.
            _ = ticker.tick() => {
                match reconcile_stale_approvals_tick(&store) {
                    Ok(0) => {}
                    Ok(n) => info!(
                        retired = n,
                        "stale approval reconciliation: retired stale cards"
                    ),
                    Err(e) => warn!("stale approval reconciliation failed: {e:#}"),
                }
            }
        }
    }
}

/// One reconciliation pass. Returns how many cards were retired. Split out from
/// the loop so it is directly unit-testable.
fn reconcile_stale_approvals_tick(store: &Store) -> Result<usize> {
    let pending = store.pending_actions_for_reconcile()?;
    let scheduled = store.scheduled_actions_for_reconcile()?;
    if pending.is_empty() && scheduled.is_empty() {
        return Ok(0);
    }

    let mut bulk_ids: Vec<String> = Vec::new();
    let mut empty_ids: Vec<String> = Vec::new();
    let mut answered_threads: Vec<String> = Vec::new();

    for row in &pending {
        // Rule 3 (#484) — the card has no draft to approve. This happens when a
        // draft was published with an empty body (historically the retry path
        // did this on a triage failure, fixed in #454; kept as a rule because
        // ANY path that leaves an empty draft produces a permanently-stale card
        // — there is nothing to approve, so it never drains and just sits in the
        // carousel until the 7-day expire). Retire it regardless of sender.
        if row.draft_empty {
            info!(
                action_id = %row.id,
                from = %row.from_email,
                subject = %row.subject,
                "stale approval: retiring card with no draft to approve"
            );
            empty_ids.push(row.id.clone());
            continue;
        }
        // Rule 2 — bulk/marketing sender. Cheap, purely local, so check first.
        if !is_human_sender(&row.from_email, &row.body) {
            info!(
                action_id = %row.id,
                from = %row.from_email,
                subject = %row.subject,
                "stale approval: retiring card from bulk/automated sender"
            );
            bulk_ids.push(row.id.clone());
            continue;
        }
        // Rule 1 — the user already replied on this thread. `i64::MIN` as the
        // "after" bound asks the broad question ("any user reply on this thread
        // at all?"), which is the right one for a card that is still sitting
        // unanswered in the queue: if the user has spoken on this thread since
        // we raised it, the draft we are holding is stale by definition.
        let Some(tid) = row.thread_id.as_deref() else {
            continue;
        };
        match store.thread_has_user_reply_after(tid, i64::MIN) {
            Ok(true) => {
                info!(
                    action_id = %row.id,
                    thread = %tid,
                    from = %row.from_email,
                    "stale approval: retiring card, user already replied on thread"
                );
                answered_threads.push(tid.to_string());
            }
            Ok(false) => {}
            Err(e) => warn!(
                action_id = %row.id,
                thread = %tid,
                "stale approval: thread_has_user_reply_after failed: {e:#}"
            ),
        }
    }

    // #500 — scheduled sends get Rule 1 ONLY (user replied on the thread),
    // bounded to replies AFTER the schedule was ARMED, and retired through
    // the per-row CAS — never through the thread-wide answered_threads flip
    // below. Both properties are load-bearing: the pending pass's Rule 1 is
    // deliberately UNbounded ("any reply ever"), so sharing its thread list
    // would cancel armed schedules over replies that predate them; and Rule 2
    // must never run on scheduled rows (`fromEmail` on compose cards holds
    // the RECIPIENT — the bulk-sender heuristic would cancel a scheduled
    // send to any newsletter-looking address). This durable pass is the
    // backstop for the engine's fire-time guard: a transient failure there
    // would otherwise let the send fire over the owner's manual reply.
    let mut scheduled_retired = 0usize;
    for (action_id, tid, armed_at_ms) in &scheduled {
        match store.thread_has_user_reply_after(tid, *armed_at_ms) {
            Ok(true) => {
                if store
                    .mark_scheduled_superseded(
                        action_id,
                        "superseded: you replied on this thread after scheduling",
                        "reconcile",
                    )
                    .unwrap_or(false)
                {
                    info!(
                        action_id = %action_id,
                        thread = %tid,
                        "stale approval: cancelled scheduled send, user \
                         replied after arming"
                    );
                    scheduled_retired += 1;
                }
            }
            Ok(false) => {}
            Err(e) => warn!(
                action_id = %action_id,
                thread = %tid,
                "stale approval: thread_has_user_reply_after failed: {e:#}"
            ),
        }
    }

    let mut retired = scheduled_retired;
    retired += store.mark_pending_superseded_by_ids(
        &bulk_ids,
        "superseded: bulk/automated sender, no reply needed",
    )?;
    retired += store.mark_pending_superseded_by_ids(
        &empty_ids,
        "superseded: no draft to approve (empty draft body)",
    )?;
    answered_threads.sort();
    answered_threads.dedup();
    for tid in &answered_threads {
        let ids = store.mark_pending_drafts_superseded_by_thread(
            tid,
            "superseded: you already replied on this thread",
        )?;
        retired += ids.len();
    }
    Ok(retired)
}

async fn run_stale_draft_sweep(
    store: Arc<Store>,
    shutdown: CancellationToken,
) -> Result<()> {
    let days = auto_expire_days_from_env();
    let Some(days) = days else {
        info!(
            "approval auto-expire sweep disabled \
             (AUGMENTAGENT_APPROVAL_AUTO_EXPIRE_DAYS=0)"
        );
        return Ok(());
    };
    let interval_secs = auto_expire_interval_secs_from_env();
    let mut ticker = tokio::time::interval(Duration::from_secs(interval_secs));
    info!(
        auto_expire_days = days,
        interval_secs,
        "approval auto-expire sweep started"
    );
    loop {
        tokio::select! {
            _ = shutdown.cancelled() => {
                info!("approval auto-expire sweep: shutdown signal received");
                return Ok(());
            }
            _ = ticker.tick() => {
                match sweep_stale_drafts_tick(&store, Some(days)) {
                    Ok(ids) if ids.is_empty() => {}
                    Ok(ids) => {
                        // Per-id audit line so the operator can `grep`
                        // the stdout log for "auto-expired <id>" and
                        // see exactly which drafts went away.
                        for id in &ids {
                            info!(
                                action_id = %id,
                                auto_expire_days = days,
                                "approval auto-expire: expired stale pending draft"
                            );
                        }
                        info!(
                            swept = ids.len(),
                            auto_expire_days = days,
                            "approval auto-expire sweep: summary"
                        );
                    }
                    Err(e) => warn!("approval auto-expire sweep failed: {e:#}"),
                }
            }
        }
    }
}

async fn run_digest(
    cli: &Cli,
    store: Arc<Store>,
    since_hours: u32,
    post_discord: bool,
) -> Result<()> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);
    let window_ms = (since_hours as i64) * 60 * 60 * 1000;
    let since_ms = now_ms - window_ms;

    // Gather the raw stats we hand Claude as user-message context.
    let counts = store.action_counts_since(since_ms)?;
    let recent = store.recent_emails_since(since_ms, 40)?;
    let pending = store.pending_reply_count()?;
    // #100: explicit, *exhaustive* enumeration of the two sets that are
    // action items — flagged (in window) and pending (all-time backlog).
    // These are independent of the 40-row recency sample above, which
    // silently dropped flagged/pending items once volume exceeded 40.
    let flagged = store.flagged_actions_since(since_ms)?;
    let pending_rows = store.pending_actions()?;
    // Hard overflow cap so a pathological backlog can't blow the Discord
    // message limit. Backpressure (#99) keeps `pending` well under this in
    // practice; the digest prompt is told to enumerate every listed row.
    const DIGEST_LIST_CAP: usize = 25;

    let mut ctx = String::new();
    ctx.push_str(&format!(
        "Time window: last {since_hours} hour(s)\n\n## Action counts by status\n"
    ));
    if counts.is_empty() {
        ctx.push_str("(no actions in window)\n");
    } else {
        for (status, n) in &counts {
            ctx.push_str(&format!("- {status}: {n}\n"));
        }
    }
    ctx.push_str(&format!("\n## Pending replies (awaiting approval)\n- {pending}\n"));

    // ## Flagged items (all) — exhaustive, not a recency sample.
    ctx.push_str(&format!(
        "\n## Flagged items (all, last {since_hours}h) — EXHAUSTIVE\n"
    ));
    if flagged.is_empty() {
        ctx.push_str("(none flagged in window)\n");
    } else {
        let total = flagged.len();
        for (from, subject, reason) in flagged.iter().take(DIGEST_LIST_CAP) {
            ctx.push_str(&format!(
                "- {from} — {} — reason: {}\n",
                truncate(subject, 120),
                truncate(reason, 160)
            ));
        }
        if total > DIGEST_LIST_CAP {
            ctx.push_str(&format!("- (+{} more)\n", total - DIGEST_LIST_CAP));
        }
    }

    // ## Pending approvals (all) — entire backlog, oldest first.
    ctx.push_str("\n## Pending approvals (all, oldest first) — EXHAUSTIVE\n");
    if pending_rows.is_empty() {
        ctx.push_str("(no drafts awaiting approval)\n");
    } else {
        let total = pending_rows.len();
        for (from, subject, age_ms) in pending_rows.iter().take(DIGEST_LIST_CAP) {
            ctx.push_str(&format!(
                "- {from} — {} — waiting {}\n",
                truncate(subject, 120),
                humanize_age(*age_ms)
            ));
        }
        if total > DIGEST_LIST_CAP {
            ctx.push_str(&format!("- (+{} more)\n", total - DIGEST_LIST_CAP));
        }
    }

    ctx.push_str("\n## Recent emails (from / subject / triage)\n");
    if recent.is_empty() {
        ctx.push_str("(no emails in window)\n");
    } else {
        for (from, subject, triage) in &recent {
            let t = triage.as_deref().unwrap_or("(unprocessed)");
            ctx.push_str(&format!(
                "- [{t}] {from} — {}\n",
                truncate(subject, 120)
            ));
        }
    }

    // Compose the digest via Claude.
    let reasoner = build_reasoner();
    let opts = digest_opts(cli.wiki_dir.clone());
    info!(window_hours = since_hours, post_discord, "composing digest");
    let digest = reasoner.call(&opts, &ctx).await?;

    println!("{digest}");

    if post_discord {
        post_digest_to_discord(&digest)
            .await
            .context("post_digest_to_discord")?;
        info!("digest posted to Discord");
    }
    Ok(())
}

/// Coarse "how long has this been waiting" string for the digest pending
/// list. Millisecond input; rounds down to the largest sensible unit.
fn humanize_age(ms: i64) -> String {
    let secs = ms.max(0) / 1000;
    let mins = secs / 60;
    let hours = mins / 60;
    let days = hours / 24;
    if days >= 1 {
        format!("{days}d")
    } else if hours >= 1 {
        format!("{hours}h")
    } else if mins >= 1 {
        format!("{mins}m")
    } else {
        "<1m".to_string()
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let mut end = max.saturating_sub(3);
        while !s.is_char_boundary(end) && end > 0 {
            end -= 1;
        }
        format!("{}...", &s[..end])
    }
}

/// Post the digest text to DISCORD_CHANNEL_ID using a bare serenity::Http
/// client (no gateway, no state). Works as a one-shot from a cron-like job.
/// Splits on paragraph boundaries for Discord's 2000-char limit.
pub(crate) async fn post_digest_to_discord(digest: &str) -> Result<()> {
    use serenity::all::{ChannelId, CreateMessage};
    use serenity::http::Http;

    let token = std::env::var("DISCORD_BOT_TOKEN").context("DISCORD_BOT_TOKEN env var required")?;
    let channel_id: u64 = std::env::var("DISCORD_CHANNEL_ID")
        .context("DISCORD_CHANNEL_ID env var required")?
        .parse()
        .context("DISCORD_CHANNEL_ID must be numeric")?;

    let http = Http::new(&token);
    let channel = ChannelId::new(channel_id);

    for chunk in augmentagent_approval_discord::chunk_for_discord(digest) {
        channel
            .send_message(&http, CreateMessage::new().content(chunk))
            .await
            .context("discord send_message")?;
    }
    Ok(())
}

async fn run_resume_ingest(cli: &Cli, file: PathBuf) -> Result<()> {
    let wiki_root = cli
        .wiki_dir
        .clone()
        .context("--wiki-dir is required for resume ingest")?;
    if !wiki_root.is_dir() {
        anyhow::bail!(
            "wiki dir {} does not exist — run `augmentagent wiki lint` once or create it first",
            wiki_root.display()
        );
    }

    let text = extract_resume_text(&file)?;
    if text.trim().is_empty() {
        anyhow::bail!("resume at {} produced empty text", file.display());
    }

    let opts = augmentagent_channel_core::reasoner::resume_opts(wiki_root.clone());
    let user_msg = format!(
        "Seed the wiki from this resume. Today's date: {today}. Follow the procedure in your system prompt exactly.\n\n<resume>\n{text}\n</resume>\n",
        today = chrono::Local::now().format("%Y-%m-%d"),
        text = text,
    );

    info!(wiki = %wiki_root.display(), file = %file.display(), "running resume ingest");
    let reasoner = build_reasoner();
    let report = reasoner.call(&opts, &user_msg).await?;
    println!("{report}");
    Ok(())
}

fn extract_resume_text(path: &std::path::Path) -> Result<String> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "txt" | "md" => std::fs::read_to_string(path)
            .with_context(|| format!("read resume at {}", path.display())),
        "pdf" => {
            // Shell out to `pdftotext` (poppler-utils). Avoids a PDF crate
            // dependency; pdftotext is already installed on most Linuxes and
            // on macOS via brew.
            use std::process::Command;
            let output = Command::new("pdftotext")
                .arg(path)
                .arg("-") // stdout
                .output()
                .with_context(|| {
                    "pdftotext missing — install via `apt install poppler-utils` (Ubuntu) or `brew install poppler` (macOS)"
                })?;
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                anyhow::bail!("pdftotext failed: {stderr}");
            }
            Ok(String::from_utf8_lossy(&output.stdout).into_owned())
        }
        _ => anyhow::bail!(
            "unsupported resume extension '{}' — use .txt, .md, or .pdf",
            ext
        ),
    }
}

/// #236: a current-time stamp prepended to every query-mode user turn. Query
/// mode previously only knew the date, so time-relative reasoning ("is it too
/// late to email?", flight urgency) was unreliable. This lives on the
/// (non-cached) user turn, so it never invalidates the cached system prompt.
fn now_awareness_line() -> String {
    format_now_awareness_line(chrono::Local::now())
}

fn format_now_awareness_line<Tz>(now: chrono::DateTime<Tz>) -> String
where
    Tz: chrono::TimeZone,
    Tz::Offset: std::fmt::Display,
{
    format!(
        "Current local time: {}. Use this for any time-relative reasoning.\n\n",
        now.format("%A %Y-%m-%d %H:%M:%S %:z")
    )
}

#[cfg(test)]
mod now_awareness_tests {
    #[test]
    fn now_line_carries_datetime_and_offset() {
        use chrono::{FixedOffset, TimeZone};
        let dt = FixedOffset::west_opt(4 * 3600)
            .unwrap()
            .with_ymd_and_hms(2026, 7, 19, 11, 35, 0)
            .unwrap();
        let line = super::format_now_awareness_line(dt);
        assert!(line.starts_with("Current local time: "));
        assert!(line.contains("2026-07-19"));
        assert!(line.contains("11:35:00"));
        assert!(line.contains("-04:00"));
        assert!(line.ends_with("\n\n"));
    }
}

async fn run_wiki_ask(cli: &Cli, question: String, post: bool) -> Result<()> {
    let wiki_root = cli
        .wiki_dir
        .clone()
        .context("--wiki-dir is required for wiki ask")?;

    let reasoner = build_reasoner();
    let repo_root = std::env::current_dir().context("current_dir")?;
    let opts = augmentagent_channel_core::reasoner::ask_opts(wiki_root.clone(), repo_root);
    info!(wiki = %wiki_root.display(), "wiki ask");
    // #389 — same owner-rules preamble the Discord query path injects, so
    // CLI asks behave identically (and the injection is testable headless).
    let question = match owner_rules_block(&wiki_root) {
        Some(rules) => format!(
            "<owner_rules>\nStanding rules from the owner (wiki/about/me.md). These are \
             HIGHEST PRIORITY: when they conflict with any default behavior in your \
             instructions, the owner's rules win. Apply them on the first attempt, \
             without being asked.\n\n{rules}</owner_rules>\n\n{question}"
        ),
        None => question,
    };
    // #236 — prepend the current time so the agent can reason about "now".
    let question = format!("{}{question}", now_awareness_line());
    // #446 — `call_transcript`, not `call`: the ask prompt puts the deliverable
    // first and the wiki-filing receipt last, and `call` keeps only the final
    // text block, so the receipt would overwrite the answer.
    let answer = reasoner.call_transcript(&opts, &question).await?;
    println!("{answer}");

    // #440 — `--post` pushes the answer through the same ATTACH-marker
    // pipeline the Discord query path uses, via a one-shot HTTP client (no
    // gateway). This is the CLI end-to-end for outbound file delivery: a
    // real message, with real attachments, lands in the approval channel.
    if post {
        use augmentagent_approval_discord::attachments::prepare_answer_delivery;
        use serenity::builder::CreateMessage;
        use serenity::model::id::ChannelId;

        let token =
            std::env::var("DISCORD_BOT_TOKEN").context("DISCORD_BOT_TOKEN required for --post")?;
        let channel: u64 = std::env::var("DISCORD_CHANNEL_ID")
            .context("DISCORD_CHANNEL_ID required for --post")?
            .parse()
            .context("DISCORD_CHANNEL_ID must be numeric")?;
        let http = serenity::http::Http::new(&token);

        let (text, attachments) = prepare_answer_delivery(&answer, Some(&wiki_root)).await;
        let n_files = attachments.len();
        let chunks = augmentagent_approval_discord::chunk_for_discord(&text);
        let total = chunks.len();
        for (idx, chunk) in chunks.into_iter().enumerate() {
            let mut builder = CreateMessage::new().content(chunk);
            if idx == 0 && !attachments.is_empty() {
                builder = builder.add_files(attachments.iter().cloned());
            }
            ChannelId::new(channel)
                .send_message(&http, builder)
                .await
                .with_context(|| format!("post answer chunk {}/{total} to discord", idx + 1))?;
            if idx + 1 < total {
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            }
        }
        println!(
            "posted to discord channel {channel}: {total} message(s), {n_files} attachment(s)"
        );
    }
    Ok(())
}

async fn run_wiki_lint(cli: &Cli, store: Arc<Store>, out: Option<PathBuf>) -> Result<()> {
    let wiki_root = cli
        .wiki_dir
        .clone()
        .context("--wiki-dir is required for wiki lint")?;
    let schema_path = cli
        .wiki_schema
        .clone()
        .unwrap_or_else(|| PathBuf::from("schema/wiki-skill.md"));
    let schema = std::fs::read_to_string(&schema_path)
        .with_context(|| format!("read schema at {}", schema_path.display()))?;

    // Computed freshness (#642) — mechanical, produced before the model
    // pass so the report carries it even if the reasoner has nothing to say.
    let freshness_section =
        wiki_freshness_section(&wiki_root, &|id| store.email_first_seen_at(id).ok().flatten());

    let reasoner = build_reasoner();
    let opts = augmentagent_channel_core::reasoner::lint_opts(schema, wiki_root.clone());
    let user_msg = format!(
        "Run the lint workflow from your system prompt against the wiki at `{}`. Produce a markdown report listing findings by category (contradictions, orphans, stale, missing pages, broken links). Use relative paths. End with a short summary line.\n",
        wiki_root.display()
    );

    info!(wiki = %wiki_root.display(), "running wiki lint");
    let report = reasoner.call(&opts, &user_msg).await?;
    let report = format!("{report}\n\n{freshness_section}");

    match out {
        Some(path) => {
            std::fs::write(&path, &report)
                .with_context(|| format!("write lint report to {}", path.display()))?;
            println!("wiki lint report written to {}", path.display());
        }
        None => {
            println!("{report}");
        }
    }
    Ok(())
}

/// Pages with computed freshness, worst-first buckets, rendered as a
/// markdown section appended to the lint report (#642). Mechanical — no
/// model involvement, so its numbers are exact and reproducible.
fn wiki_freshness_section(
    wiki_root: &std::path::Path,
    resolve: &dyn Fn(&str) -> Option<i64>,
) -> String {
    use augmentagent_wiki::freshness::{self, PageStatus};

    const STALE_DAYS: i64 = 90; // matches the schema's lint guidance
    const LIST_CAP: usize = 20;

    let today = freshness::today_utc();
    let mut fresh = 0usize;
    let mut stale: Vec<(String, freshness::Date)> = Vec::new();
    let mut unknown: Vec<String> = Vec::new();
    let mut deprecated = 0usize;
    let mut expired: Vec<(String, freshness::Date)> = Vec::new();

    for dir in ["people", "threads", "projects"] {
        let Ok(rd) = std::fs::read_dir(wiki_root.join(dir)) else {
            continue;
        };
        for ent in rd.flatten() {
            let path = ent.path();
            if !path.is_file() || path.extension().and_then(|e| e.to_str()) != Some("md") {
                continue;
            }
            let Ok(page) = std::fs::read_to_string(&path) else {
                continue;
            };
            let rel = format!("{dir}/{}", ent.file_name().to_string_lossy());
            let f = freshness::compute(&page, resolve);
            if f.status == PageStatus::Deprecated {
                deprecated += 1;
                continue;
            }
            if f.past_stale_after(today) {
                expired.push((rel.clone(), f.stale_after.expect("past implies present")));
            }
            match (f.as_of, f.age_days(today)) {
                (Some(date), Some(age)) if age > STALE_DAYS => stale.push((rel, date)),
                (Some(_), _) => fresh += 1,
                (None, _) => unknown.push(rel),
            }
        }
    }

    stale.sort_by_key(|(_, d)| *d); // oldest evidence first
    unknown.sort();
    expired.sort();

    let total = fresh + stale.len() + unknown.len() + deprecated;
    let mut s = String::from("## Freshness (computed)\n\n");
    s.push_str(
        "Fact age = newest `emails.firstSeenAt` across each page's cited messageIds \
         (`sources:` + inline `m:` cites) plus owner `verified:` dates. Pages whose \
         evidence doesn't resolve are UNKNOWN — that is never \"fresh\" (G1).\n\n",
    );
    s.push_str(&format!(
        "- {total} pages: {fresh} fresh (evidence ≤{STALE_DAYS}d) · {} stale (>{STALE_DAYS}d) · {} unknown evidence · {deprecated} deprecated\n",
        stale.len(),
        unknown.len(),
    ));
    let fmt_capped = |items: &[String]| -> String {
        let mut line = items[..items.len().min(LIST_CAP)].join(", ");
        if items.len() > LIST_CAP {
            line.push_str(&format!(", … and {} more", items.len() - LIST_CAP));
        }
        line
    };
    if !expired.is_empty() {
        let rows: Vec<String> = expired
            .iter()
            .map(|(rel, d)| format!("{rel} (stale_after {d})"))
            .collect();
        s.push_str(&format!("- past explicit `stale_after:`: {}\n", fmt_capped(&rows)));
    }
    if !stale.is_empty() {
        let rows: Vec<String> = stale
            .iter()
            .map(|(rel, d)| format!("{rel} ({d})"))
            .collect();
        s.push_str(&format!("- oldest evidence first: {}\n", fmt_capped(&rows)));
    }
    if !unknown.is_empty() {
        s.push_str(&format!("- unknown evidence: {}\n", fmt_capped(&unknown)));
    }
    s
}

#[cfg(test)]
mod wiki_freshness_section_tests {
    use super::wiki_freshness_section;

    #[test]
    fn buckets_and_caps_render() {
        let td = tempfile::TempDir::new().unwrap();
        let root = td.path();
        std::fs::create_dir_all(root.join("people")).unwrap();
        let mk = |name: &str, fm_extra: &str, source: &str| {
            std::fs::write(
                root.join("people").join(name),
                format!("---\nkind: person\n{fm_extra}sources: [{source}]\n---\n\nx\n"),
            )
            .unwrap();
        };
        mk("fresh.md", "", "id-fresh");
        mk("stale.md", "", "id-stale");
        mk("unknown.md", "", "id-ghost");
        mk("dead.md", "status: deprecated\n", "id-fresh");
        mk("expired.md", "stale_after: 2020-01-01\n", "id-fresh");

        let now_ms = chrono::Utc::now().timestamp_millis();
        let resolve = move |id: &str| match id {
            "id-fresh" => Some(now_ms),
            "id-stale" => Some(now_ms - 200 * 86_400_000),
            _ => None,
        };
        let s = wiki_freshness_section(root, &resolve);
        assert!(
            s.contains("5 pages: 2 fresh (evidence ≤90d) · 1 stale (>90d) · 1 unknown evidence · 1 deprecated"),
            "unexpected section:\n{s}"
        );
        assert!(s.contains("people/stale.md ("), "stale page must be listed: {s}");
        assert!(s.contains("- unknown evidence: people/unknown.md"));
        assert!(s.contains("people/expired.md (stale_after 2020-01-01)"));
    }
}

/// `wiki index` — inspect or rebuild the derived index.md (#642).
fn run_wiki_index(cli: &Cli, store: Arc<Store>, rebuild: bool) -> Result<()> {
    let wiki_root = cli
        .wiki_dir
        .clone()
        .context("--wiki-dir is required for wiki index")?;

    if rebuild {
        let stats =
            augmentagent_wiki::rebuild_index(&wiki_root, &|id: &str| {
                store.email_first_seen_at(id).ok().flatten()
            })?;
        let skipped = if stats.unreadable > 0 {
            format!(", {} unreadable pages skipped", stats.unreadable)
        } else {
            String::new()
        };
        println!(
            "index.md rebuilt: {} people, {} threads, {} projects — {} pages{skipped}",
            stats.people,
            stats.threads,
            stats.projects,
            stats.total(),
        );
        return Ok(());
    }

    // Coverage report: pages on disk vs. entries in the current index.
    let index = std::fs::read_to_string(wiki_root.join("index.md")).unwrap_or_default();
    let indexed: std::collections::BTreeSet<String> = index
        .lines()
        .filter_map(|l| {
            let rest = l.strip_prefix("- [")?;
            let end = rest.find("](")?;
            Some(rest[..end].to_string())
        })
        .collect();

    let mut missing_total = 0usize;
    for dir in ["people", "threads", "projects"] {
        let mut on_disk = 0usize;
        let mut missing = 0usize;
        if let Ok(rd) = std::fs::read_dir(wiki_root.join(dir)) {
            for ent in rd.flatten() {
                let p = ent.path();
                if p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("md") {
                    on_disk += 1;
                    let rel = format!("{dir}/{}", ent.file_name().to_string_lossy());
                    if !indexed.contains(&rel) {
                        missing += 1;
                    }
                }
            }
        }
        missing_total += missing;
        println!("{dir}: {on_disk} pages on disk, {missing} missing from index");
    }
    let dangling = indexed
        .iter()
        .filter(|rel| !wiki_root.join(rel.as_str()).is_file())
        .count();
    println!(
        "index entries: {} total, {dangling} pointing at missing files; {missing_total} pages not indexed. Run `wiki index --rebuild` to regenerate.",
        indexed.len(),
    );
    Ok(())
}

/// Per-page outcome for stderr progress + final summary. Local to the
/// migration runner — not surfaced anywhere else.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Failed payload is for future structured-error reporting.
enum MigrateOutcome {
    Migrated { dropped: usize },
    Skipped(&'static str),
    Failed(String),
}

/// Build the model wrapper prompt from the schema doc + the citation rule.
/// Centralised so tests on the migrate module are not coupled to the prose.
fn migration_system_prompt(schema_body: &str) -> String {
    format!(
        "{schema_body}\n\n## Migration task\n\nYou are running a one-shot v2 migration over a single person page. Read its existing content carefully and infer ONLY the v2 fields (`affiliations`, `events`, `introduced_by`) that are EXPLICITLY supported by the page body. Do NOT invent. Do NOT write `cadence`, `trust`, `topics`, or `strength` — those fields are user/derived.\n\nFor every `events` and `affiliations` entry, include `source_message_id: <id>` citing a messageId from the page's existing `sources:` list. Entries without a citation will be dropped.\n\nIf you cannot infer any v2 field with confidence, return an empty YAML mapping. That's the right answer for thin pages.\n\nOutput ONLY a YAML mapping (or a fenced ```yaml block). No prose. No explanation."
    )
}

/// Build the per-page user prompt: full page contents.
fn migration_user_prompt(slug: &str, page: &str) -> String {
    format!("Page: people/{slug}.md\n\n{page}")
}

#[allow(clippy::too_many_arguments)]
async fn run_wiki_migrate(
    cli: &Cli,
    to: String,
    dry_run: bool,
    concurrency: usize,
    limit: Option<usize>,
    branch: String,
    force: bool,
) -> Result<()> {
    use augmentagent_channel_core::reasoner::wiki_migrate_opts;
    use augmentagent_wiki::migrate::{
        apply_patch, classify, parse_patch, parse_sources, render_patch_lines,
        split_frontmatter, validate_citations, MigrationDecision,
    };
    use augmentagent_wiki::with_page_lock;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::Semaphore;

    if to != "v2" {
        anyhow::bail!("--to must be `v2` (only v2 is supported today, got {to:?})");
    }

    let wiki_root = cli
        .wiki_dir
        .clone()
        .context("--wiki-dir is required for wiki migrate")?;
    let schema_path = cli
        .wiki_schema
        .clone()
        .unwrap_or_else(|| PathBuf::from("schema/wiki-skill.md"));
    let schema = std::fs::read_to_string(&schema_path)
        .with_context(|| format!("read schema at {}", schema_path.display()))?;

    // §7 pre-flight: refuse to run while the daemon could be writing pages.
    if !force {
        match tokio::process::Command::new("systemctl")
            .args(["--user", "is-active", "augmentagent.service"])
            .output()
            .await
        {
            Ok(out) => {
                let s = String::from_utf8_lossy(&out.stdout);
                if s.trim() == "active" {
                    anyhow::bail!(
                        "augmentagent.service is active — pause it first to avoid racing live ingest writes:\n  systemctl --user stop augmentagent.service\nThen re-run, and resume after merge:\n  systemctl --user start augmentagent.service\nOr override with --force (NOT RECOMMENDED for the live wiki)."
                    );
                }
            }
            Err(e) => {
                tracing::warn!("systemctl pre-flight check failed: {e}; continuing");
            }
        }
    }

    let layout = augmentagent_wiki::WikiLayout::new(wiki_root.clone());
    let people_dir = layout.people_dir();
    if !people_dir.is_dir() {
        anyhow::bail!("people dir missing: {}", people_dir.display());
    }

    let mut all_paths: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(&people_dir)
        .with_context(|| format!("read people dir {}", people_dir.display()))?
    {
        let e = entry?;
        let p = e.path();
        if p.extension().and_then(|s| s.to_str()) == Some("md") {
            all_paths.push(p);
        }
    }
    all_paths.sort();
    let total = all_paths.len();

    // Pre-classify (no model spend) and partition.
    let mut eligible: Vec<PathBuf> = Vec::new();
    let mut skipped_v2: usize = 0;
    let mut skipped_migrated: usize = 0;
    let mut skipped_garbage: usize = 0;
    for p in &all_paths {
        let body = match std::fs::read_to_string(p) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("read {} failed: {e}", p.display());
                continue;
            }
        };
        match classify(&body) {
            MigrationDecision::Eligible => eligible.push(p.clone()),
            MigrationDecision::AlreadyV2 => skipped_v2 += 1,
            MigrationDecision::AlreadyMigrated => skipped_migrated += 1,
            MigrationDecision::NoFrontmatter => {
                skipped_garbage += 1;
                eprintln!("[skip:no-fm] {}", relpath(p, &wiki_root));
            }
        }
    }

    if let Some(n) = limit {
        eligible.truncate(n);
    }

    eprintln!(
        "wiki migrate: total={total} eligible={} already_v2={} already_migrated={} no_fm={} concurrency={} branch={} dry_run={}",
        eligible.len(),
        skipped_v2,
        skipped_migrated,
        skipped_garbage,
        concurrency,
        branch,
        dry_run,
    );

    if eligible.is_empty() {
        eprintln!("nothing to migrate");
        return Ok(());
    }

    let today_iso = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let system_prompt = migration_system_prompt(&schema);
    let opts = std::sync::Arc::new(wiki_migrate_opts(system_prompt, wiki_root.clone()));
    let reasoner = build_reasoner();
    let sem = std::sync::Arc::new(Semaphore::new(concurrency.max(1)));

    let migrated = std::sync::Arc::new(AtomicUsize::new(0));
    let dropped_total = std::sync::Arc::new(AtomicUsize::new(0));
    let failed = std::sync::Arc::new(AtomicUsize::new(0));

    // Per-task: returns (path, outcome) so the orchestrator can stage the
    // exact set of pages it wrote without re-walking the directory.
    let mut set: tokio::task::JoinSet<(PathBuf, MigrateOutcome)> = tokio::task::JoinSet::new();
    for path in eligible.clone() {
        let opts = std::sync::Arc::clone(&opts);
        let reasoner = std::sync::Arc::clone(&reasoner);
        let sem = std::sync::Arc::clone(&sem);
        let migrated = std::sync::Arc::clone(&migrated);
        let dropped_total = std::sync::Arc::clone(&dropped_total);
        let failed = std::sync::Arc::clone(&failed);
        let today_iso = today_iso.clone();
        let wiki_root_for_log = wiki_root.clone();
        let path_for_task = path.clone();

        set.spawn(async move {
            let _permit = match sem.acquire_owned().await {
                Ok(p) => p,
                Err(_) => {
                    failed.fetch_add(1, Ordering::SeqCst);
                    return (
                        path_for_task,
                        MigrateOutcome::Failed("semaphore closed".into()),
                    );
                }
            };
            let slug = path_for_task
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
            let display = relpath(&path_for_task, &wiki_root_for_log);

            let result: anyhow::Result<MigrateOutcome> =
                with_page_lock(&path_for_task, || async {
                    let body = tokio::fs::read_to_string(&path_for_task).await?;
                    // Re-check classification under the lock — another task
                    // could have flipped state since the pre-scan.
                    match classify(&body) {
                        MigrationDecision::AlreadyMigrated => {
                            return Ok(MigrateOutcome::Skipped("already-migrated"));
                        }
                        MigrationDecision::AlreadyV2 => {
                            return Ok(MigrateOutcome::Skipped("already-v2"));
                        }
                        MigrationDecision::NoFrontmatter => {
                            return Ok(MigrateOutcome::Skipped("no-frontmatter"));
                        }
                        MigrationDecision::Eligible => {}
                    }
                    let user = migration_user_prompt(&slug, &body);
                    let raw = reasoner.call(&opts, &user).await?;
                    let patch = parse_patch(&raw)?;
                    let (fm, _) = split_frontmatter(&body)
                        .ok_or_else(|| anyhow::anyhow!("frontmatter vanished mid-flight"))?;
                    let allowed = parse_sources(fm);
                    let filt = validate_citations(patch, &allowed);
                    let rendered = render_patch_lines(&filt.filtered, &today_iso)?;
                    let next = apply_patch(&body, &rendered)?;
                    if !dry_run {
                        tokio::fs::write(&path_for_task, next.as_bytes()).await?;
                    }
                    Ok(MigrateOutcome::Migrated {
                        dropped: filt.dropped,
                    })
                })
                .await;

            let outcome = match result {
                Ok(o @ MigrateOutcome::Migrated { dropped }) => {
                    migrated.fetch_add(1, Ordering::SeqCst);
                    dropped_total.fetch_add(dropped, Ordering::SeqCst);
                    eprintln!("[migrated] {display} dropped={dropped}");
                    o
                }
                Ok(o @ MigrateOutcome::Skipped(reason)) => {
                    eprintln!("[skip:{reason}] {display}");
                    o
                }
                Ok(o) => o,
                Err(e) => {
                    failed.fetch_add(1, Ordering::SeqCst);
                    eprintln!("[fail] {display}: {e:#}");
                    MigrateOutcome::Failed(format!("{e:#}"))
                }
            };
            (path_for_task, outcome)
        });
    }

    // Drain results, batching commits of *successfully migrated* paths
    // every 25. JoinSet preserves no order — we batch by completion order.
    let mut pending_batch: Vec<PathBuf> = Vec::new();
    let mut batch_counter: usize = 0;
    while let Some(joined) = set.join_next().await {
        let (path, outcome) = joined.context("migrate task join")?;
        if matches!(outcome, MigrateOutcome::Migrated { .. }) {
            pending_batch.push(path);
            if !dry_run && pending_batch.len() >= 25 {
                let pending = std::mem::take(&mut pending_batch);
                batch_counter += 1;
                git_commit_batch(&wiki_root, &pending, batch_counter).await?;
            }
        }
    }
    if !dry_run && !pending_batch.is_empty() {
        batch_counter += 1;
        git_commit_batch(&wiki_root, &pending_batch, batch_counter).await?;
    }

    let migrated_n = migrated.load(Ordering::SeqCst);
    let dropped_n = dropped_total.load(Ordering::SeqCst);
    let failed_n = failed.load(Ordering::SeqCst);
    // §7 cost estimate: ~$0.009 per migrated page (Haiku, ~4k in / 1k out).
    let cost_est = migrated_n as f64 * 0.009;

    eprintln!("---");
    eprintln!("wiki migrate summary");
    eprintln!("  total pages          : {total}");
    eprintln!("  migrated             : {migrated_n}");
    eprintln!("  skipped (already v2) : {skipped_v2}");
    eprintln!("  skipped (marker)     : {skipped_migrated}");
    eprintln!("  skipped (no fm)      : {skipped_garbage}");
    eprintln!("  failed               : {failed_n}");
    eprintln!("  dropped uncited      : {dropped_n}");
    eprintln!("  est. Haiku cost      : ${:.2}", cost_est);
    eprintln!("  commits              : {batch_counter}");
    if dry_run {
        eprintln!("  (dry run — no writes, no commits)");
    }
    Ok(())
}

/// Render a wiki path relative to `wiki_root` for human-readable logs.
fn relpath(p: &std::path::Path, wiki_root: &std::path::Path) -> String {
    p.strip_prefix(wiki_root)
        .unwrap_or(p)
        .display()
        .to_string()
}

/// Stage and commit a batch of migrated pages. Authored as Nolan Makatche
/// per project convention; per-command `-c user.name/email` to avoid
/// requiring global git config.
async fn git_commit_batch(
    wiki_root: &std::path::Path,
    paths: &[PathBuf],
    batch_no: usize,
) -> Result<()> {
    if paths.is_empty() {
        return Ok(());
    }
    let mut add = tokio::process::Command::new("git");
    add.arg("-C").arg(wiki_root).arg("add").arg("--");
    for p in paths {
        let rel = p.strip_prefix(wiki_root).unwrap_or(p);
        add.arg(rel);
    }
    let st = add.status().await.context("git add")?;
    if !st.success() {
        anyhow::bail!("git add failed: {st:?}");
    }

    let msg = format!("wiki: migrate batch {batch_no} to v2");
    // Commit identity comes from env (no hardcoded personal data); neutral
    // fallback so a public checkout is clean. Override via .env.
    let git_author_name = std::env::var("AUGMENTAGENT_GIT_AUTHOR_NAME")
        .unwrap_or_else(|_| "AugmentAgent".to_string());
    let git_author_email = std::env::var("AUGMENTAGENT_GIT_AUTHOR_EMAIL")
        .unwrap_or_else(|_| "augmentagent@localhost".to_string());
    let st = tokio::process::Command::new("git")
        .arg("-C")
        .arg(wiki_root)
        .arg("-c")
        .arg(format!("user.name={git_author_name}"))
        .arg("-c")
        .arg(format!("user.email={git_author_email}"))
        .arg("commit")
        .arg("-m")
        .arg(&msg)
        .status()
        .await
        .context("git commit")?;
    if !st.success() {
        // Tolerate "nothing to commit" so the migration doesn't abort.
        eprintln!("git commit batch {batch_no} returned {st:?}; continuing");
    } else {
        eprintln!("[commit] batch {batch_no}: {} pages", paths.len());
    }
    Ok(())
}

/// Files that must NEVER reach the private mirror: the daemon's SQLite
/// store (+ WAL/SHM sidecars), secrets, and per-page lock files. The
/// wiki-local `.gitignore` already excludes these; the sync guard is
/// belt-and-suspenders against an accidental `git add -f` or a botched
/// `.gitignore` edit (KB sync #4).
const WIKI_SYNC_FORBIDDEN: &[&str] = &["data.db", "data.db-wal", "data.db-shm", ".env"];

fn wiki_sync_check_forbidden(list: &str, what: &str) -> Result<()> {
    for f in list.lines() {
        let f = f.trim();
        if f.is_empty() {
            continue;
        }
        let base = f.rsplit('/').next().unwrap_or(f);
        if WIKI_SYNC_FORBIDDEN.contains(&base) || base.ends_with(".lock") {
            anyhow::bail!(
                "wiki sync refused: sensitive/non-content file `{f}` is {what} in the wiki repo. \
                 Fix wiki/.gitignore before syncing — this must never reach the mirror."
            );
        }
    }
    Ok(())
}

/// Run `git -C <dir> <args>`; return whether it exited 0.
async fn git_run(dir: &std::path::Path, args: &[&str]) -> Result<bool> {
    let st = tokio::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .status()
        .await
        .with_context(|| format!("spawn git {}", args.join(" ")))?;
    Ok(st.success())
}

/// Run `git -C <dir> <args>` and capture stdout; bail on non-zero.
async fn git_capture(dir: &std::path::Path, args: &[&str]) -> Result<String> {
    let out = tokio::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .await
        .with_context(|| format!("spawn git {}", args.join(" ")))?;
    if !out.status.success() {
        anyhow::bail!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// The env-driven git identity every wiki-repo write commits with (G3 of
/// #642: one identity, or GitHub contribution attribution breaks).
fn wiki_git_identity() -> (String, String) {
    let name = std::env::var("AUGMENTAGENT_GIT_AUTHOR_NAME")
        .unwrap_or_else(|_| "AugmentAgent".to_string());
    let email = std::env::var("AUGMENTAGENT_GIT_AUTHOR_EMAIL")
        .unwrap_or_else(|_| "augmentagent@localhost".to_string());
    (name, email)
}

/// `git_run` with the wiki identity injected via `-c user.name/-c user.email`.
/// Required for any git verb that CREATES commits — `pull --rebase` replays
/// local commits and needs a committer identity. Global git config is
/// intentionally unset on the daemon host, so a bare rebase dies with
/// "unable to auto-detect email address" the first time local and remote
/// main diverge — turning every owner-side GitHub edit into a kb-conflict
/// branch instead of the documented owner-wins resolution.
async fn git_run_as_committer(dir: &std::path::Path, args: &[&str]) -> Result<bool> {
    let (name, email) = wiki_git_identity();
    let st = tokio::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .arg("-c")
        .arg(format!("user.name={name}"))
        .arg("-c")
        .arg(format!("user.email={email}"))
        .args(args)
        .status()
        .await
        .with_context(|| format!("spawn git {}", args.join(" ")))?;
    Ok(st.success())
}

/// Push current HEAD to `origin/main`. If the remote moved under us
/// (someone edited on GitHub between our fetch and push), re-pull
/// owner-wins once and retry.
async fn wiki_sync_push(dir: &std::path::Path) -> Result<()> {
    for attempt in 0..2 {
        if git_run(dir, &["push", "origin", "HEAD:main"]).await? {
            println!("[wiki sync] pushed to origin/main");
            return Ok(());
        }
        if attempt == 0 {
            eprintln!("[wiki sync] push rejected (remote moved); re-pulling owner-wins and retrying");
            let _ = git_run(dir, &["fetch", "origin", "main"]).await;
            let _ =
                git_run_as_committer(dir, &["pull", "--rebase", "-X", "ours", "origin", "main"])
                    .await;
        }
    }
    anyhow::bail!("wiki sync: push to origin/main failed after one retry");
}

/// #478: validate an owner-facing wiki page that a GitHub-side edit could have
/// broken, returning a human-readable problem string if the daemon can no
/// longer parse it the way it relies on. Pure (no I/O) so it is unit-testable.
///
/// - `about/me.md`: `owner_rules_block` injects the "Writing style preferences"
///   / "Agent behavior rules" sections into every query turn (#389). An edit
///   that renames or empties both sections silently disables that injection.
/// - `people/*.md`: a valid person page has a YAML frontmatter block; an edit
///   that breaks the `---` delimiters makes `split_frontmatter` return `None`.
fn wiki_sync_validate_page(rel_path: &str, content: &str) -> Option<String> {
    if rel_path == "about/me.md" {
        let has_rules = ["Writing style preferences", "Agent behavior rules"]
            .iter()
            .any(|s| {
                extract_md_section(content, s)
                    .map(|b| !b.trim().is_empty())
                    .unwrap_or(false)
            });
        if !has_rules {
            return Some(format!(
                "{rel_path}: neither owner-rules section (\"Writing style preferences\" / \
                 \"Agent behavior rules\") parses — owner-rules injection is now disabled"
            ));
        }
    } else if rel_path.starts_with("people/") && rel_path.ends_with(".md") {
        if augmentagent_wiki::migrate::split_frontmatter(content).is_none() {
            return Some(format!("{rel_path}: YAML frontmatter block no longer parses"));
        }
    }
    None
}

#[cfg(test)]
mod wiki_sync_validation_tests {
    #[test]
    fn me_md_missing_owner_rules_sections_is_flagged() {
        let ok = "# Me\n\n## Writing style preferences\n- concise\n\n## Agent behavior rules\n- x\n";
        assert!(super::wiki_sync_validate_page("about/me.md", ok).is_none());
        // An external edit that dropped both rule sections -> flagged.
        let broken = "# Me\n\nJust a bio, no rule sections.\n";
        assert!(super::wiki_sync_validate_page("about/me.md", broken).is_some());
    }

    #[test]
    fn person_page_broken_frontmatter_is_flagged() {
        let ok = "---\nname: Dana\n---\n\n## Tone\nwarm\n"; // pii-ok: synthetic
        assert!(super::wiki_sync_validate_page("people/dana.md", ok).is_none()); // pii-ok: synthetic
        // Frontmatter delimiters broken by an edit -> flagged.
        let broken = "name: Dana\n\nno frontmatter delimiters\n"; // pii-ok: synthetic
        assert!(super::wiki_sync_validate_page("people/dana.md", broken).is_some()); // pii-ok: synthetic
    }

    #[test]
    fn unrelated_paths_are_never_flagged() {
        assert!(super::wiki_sync_validate_page("threads/x.md", "anything").is_none());
        assert!(super::wiki_sync_validate_page("index.md", "").is_none());
    }
}

/// `wiki sync` — reconcile the local knowledge base with its private
/// GitHub mirror. Two-way: commit local page changes, pull owner edits
/// (owner-wins), push. Reuses the ambient `gh` credential. See epic #474.
/// `AUGMENTAGENT_MY_EMAIL` — a comma-separated list of the operator's own
/// addresses (personal gmail, work domain, export account). Every one of them
/// is "me" for meeting-roster self-exclusion (#922 follow-up).
fn parse_my_emails(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

#[cfg(test)]
mod my_email_parse_tests {
    use super::parse_my_emails;

    /// #922 follow-up — AUGMENTAGENT_MY_EMAIL is a comma-separated list of
    /// the operator's own addresses. Whitespace and empty entries are noise,
    /// not addresses.
    #[test]
    fn a_comma_separated_list_parses_to_clean_addresses() {
        assert_eq!(
            parse_my_emails(" a@example.com, B2@example.com ,,"),
            vec!["a@example.com".to_string(), "B2@example.com".to_string()]
        );
        assert_eq!(
            parse_my_emails("solo@example.com"),
            vec!["solo@example.com".to_string()]
        );
        assert!(parse_my_emails("").is_empty());
        assert!(parse_my_emails(" , ").is_empty());
    }
}

/// The `emails` table as transcript dedup: `record` returns true for a first
/// sighting (and remembers it), false for a re-push of the same meeting.
struct TranscriptSeenLog(Arc<Store>);
impl augmentagent_channel_fotw::runner::SeenLog for TranscriptSeenLog {
    fn record(&self, email: &augmentagent_store::Email) -> Result<bool> {
        Ok(self.0.upsert_email(email)?)
    }
}

/// A dry run must not write the dedup row, or the real run that follows
/// would find every meeting already seen and ingest nothing — but it must
/// still *read* the table, or it reports every already-ingested meeting as
/// new and prints its body under "would ingest" (PR #922 review).
struct TranscriptDryLog(Arc<Store>);
impl augmentagent_channel_fotw::runner::SeenLog for TranscriptDryLog {
    fn record(&self, email: &augmentagent_store::Email) -> Result<bool> {
        Ok(self.0.email_first_seen_at(&email.message_id)?.is_none())
    }
}

#[cfg(test)]
mod transcript_seen_log_tests {
    use super::*;
    use augmentagent_channel_fotw::runner::SeenLog;

    fn test_email(id: &str) -> augmentagent_store::Email {
        augmentagent_store::Email {
            message_id: id.into(),
            thread_id: None,
            from: "sender".into(),
            subject: "subject".into(),
            body: "body".into(),
            date: "2026-09-01T00:00:00Z".into(),
            to: String::new(),
            cc: String::new(),
            attachments: Vec::new(),
            account_entity_id: None,
            platform: "fotw".into(),
            kind: "email".into(),
        }
    }

    /// PR #922 review — a dry run must report what a live run would do:
    /// read-only duplicate detection, still writing nothing. It used to call
    /// every already-ingested meeting "new" and print its full body under
    /// "would ingest".
    #[test]
    fn a_dry_run_sees_duplicates_without_writing() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let store = Arc::new(Store::open(dir.path().join("t.db")).expect("open store"));
        let live = TranscriptSeenLog(Arc::clone(&store));
        let dry = TranscriptDryLog(Arc::clone(&store));

        assert!(
            dry.record(&test_email("fotw:m1")).unwrap(),
            "an unseen meeting reads as new"
        );
        assert_eq!(
            store.email_first_seen_at("fotw:m1").unwrap(),
            None,
            "a dry run writes no dedup row"
        );

        assert!(live.record(&test_email("fotw:m1")).unwrap());
        assert!(
            !dry.record(&test_email("fotw:m1")).unwrap(),
            "a recorded meeting reads as a duplicate in a dry run"
        );
    }
}

/// `augmentagent transcripts sync` (#915).
///
/// Fast-forward the FlyOnTheWall transcript clone, then hand any meeting the
/// `emails` table has not seen to the wiki ingest funnel. Read-only against
/// that repo throughout: FlyOnTheWall owns it.
///
/// The calendar roster comes from rows the gcal channel already wrote — see
/// `augmentagent_channel_fotw::calendar` — so this needs no Google credentials
/// of its own, and degrades to name-based linking when a recording has no
/// matching invite.
async fn run_transcripts_sync(
    cli: &Cli,
    store: Arc<Store>,
    dry_run: bool,
    no_pull: bool,
    require_disclosed: bool,
) -> Result<()> {
    use augmentagent_channel_fotw::calendar::{rosters_from_rows, GcalRow};
    use augmentagent_channel_fotw::distill::RosterMember;
    use augmentagent_channel_fotw::match_event::{match_event, EventWindow, Match};
    use augmentagent_channel_fotw::runner::{scan, ScanOpts, SeenLog};

    let repo = std::env::var("AUGMENTAGENT_TRANSCRIPTS_DIR")
        .map(PathBuf::from)
        .context("set AUGMENTAGENT_TRANSCRIPTS_DIR to the transcript repo clone")?;
    let wiki_root = cli
        .wiki_dir
        .clone()
        .context("--wiki-dir is required for transcripts sync")?;
    let my_emails = parse_my_emails(&std::env::var("AUGMENTAGENT_MY_EMAIL").unwrap_or_default());

    if !no_pull && !dry_run {
        match augmentagent_channel_fotw::sync::sync(&repo) {
            Ok(0) => println!("[transcripts] already up to date"),
            Ok(n) => println!("[transcripts] pulled {n} commit(s)"),
            Err(e) => anyhow::bail!("transcripts sync: {e}"),
        }
    }

    // The roster index, built once per run: a few hundred rows, consulted per
    // meeting by the closure below.
    let events = rosters_from_rows(
        &store
            .gcal_attendee_rows()
            .unwrap_or_default()
            .into_iter()
            .map(|(message_id, from, received_at)| GcalRow {
                message_id,
                from,
                received_at,
            })
            .collect::<Vec<_>>(),
    );
    let windows: Vec<EventWindow> = events.iter().map(|e| e.window.clone()).collect();
    let resolve = |doc: &augmentagent_channel_fotw::parse::MeetingDoc| {
        let m = match_event(doc.started_at_ms, doc.duration_ms, &windows);
        // Only a single match names anyone; ambiguity is reported, never resolved.
        let roster: Vec<RosterMember> = match &m {
            Match::Single(w) => events
                .iter()
                .find(|e| e.window.event_id == w.event_id)
                .map(|e| e.roster.clone())
                .unwrap_or_default(),
            _ => Vec::new(),
        };
        (m, roster)
    };

    let log: Box<dyn SeenLog> = if dry_run {
        Box::new(TranscriptDryLog(Arc::clone(&store)))
    } else {
        Box::new(TranscriptSeenLog(Arc::clone(&store)))
    };

    let meetings_dir = repo.join("meetings");
    let opts = ScanOpts {
        dir: &meetings_dir,
        require_disclosed,
        my_emails: &my_emails,
    };
    let (report, emails) = scan(&opts, log.as_ref(), &resolve)?;

    println!(
        "[transcripts] {} new, {} duplicate, {} skipped, {} unreadable",
        report.ingested.len(),
        report.duplicates,
        report.skipped.len(),
        report.unreadable.len()
    );
    for (id, why) in &report.skipped {
        println!("[transcripts] skipped {id}: {why:?}");
    }
    if dry_run {
        for e in &emails {
            println!("\n--- would ingest {} ---\n{}", e.message_id, e.body);
        }
        return Ok(());
    }

    let spawned = emails.len();
    let reasoner = build_reasoner();
    let schema = resolve_wiki_schema(cli)
        .context("wiki schema not found; transcripts ingest needs schema/wiki-skill.md")?;
    for email in emails {
        let id = email.message_id.clone();
        augmentagent_channel_core::ingest::spawn_ingest(
            Arc::clone(&reasoner),
            wiki_root.clone(),
            // The relevance gate rides on the schema for this platform only: a
            // recorded meeting is not a message addressed to anyone, and a
            // personal capture should leave no trace.
            format!(
                "{schema}\n{}",
                augmentagent_channel_fotw::distill::RELEVANCE_GATE
            ),
            email,
            augmentagent_channel_core::decision::DecisionKind::Meeting,
            Some("meeting transcript".to_string()),
            None,
            augmentagent_channel_core::ingest::IngestTrigger::Meeting,
        );
        println!("[transcripts] ingesting {id}");
    }
    if spawned == 0 {
        return Ok(());
    }
    // PR #922 — this is a one-shot process: returning while jobs are queued
    // would drop the tokio runtime and kill them mid-ingest, after the dedup
    // row already marked each meeting seen — never to be retried. Hold the
    // process open until the pool drains. The pool itself (#899) keeps
    // concurrency at its worker cap (2), so this cannot re-create the
    // 2026-08-31 `claude -p` burst; it only makes the process live as long
    // as its own jobs.
    let pool = augmentagent_channel_core::ingest::IngestPool::global();
    println!("[transcripts] waiting for {spawned} ingest job(s)…");
    let drained = pool
        .wait_idle(std::time::Duration::from_secs(15 * 60))
        .await;
    let stats = pool.stats();
    if !drained {
        anyhow::bail!(
            "transcripts ingest did not drain in 15m ({} queued, {} in flight) — \
             the recorded meetings will read as duplicates on the next run; \
             delete their fotw:<id> rows from `emails` to retry",
            stats.queued,
            stats.in_flight
        );
    }
    println!(
        "[transcripts] ingest finished ({} completed, {} dropped)",
        stats.completed, stats.dropped
    );
    Ok(())
}

async fn run_wiki_sync(cli: &Cli, dry_run: bool, no_pull: bool) -> Result<()> {
    let wiki_root = cli
        .wiki_dir
        .clone()
        .context("--wiki-dir is required for wiki sync")?;
    let wiki_root = wiki_root
        .canonicalize()
        .with_context(|| format!("canonicalize wiki dir {}", wiki_root.display()))?;

    // The wiki must be its own git repo. Bootstrap (git init + private
    // remote) runs once, out of band — see KB sync #1.
    if !wiki_root.join(".git").exists() {
        anyhow::bail!(
            "{} is not a git repo — bootstrap it first (git init + private origin). See epic #474 / KB sync #1.",
            wiki_root.display()
        );
    }

    // Guard: nothing sensitive/non-content may already be tracked.
    let tracked = git_capture(&wiki_root, &["ls-files"]).await?;
    wiki_sync_check_forbidden(&tracked, "tracked")?;

    let porcelain = git_capture(&wiki_root, &["status", "--porcelain"]).await?;
    let dirty: Vec<&str> = porcelain
        .lines()
        .filter(|l| !l.trim().is_empty())
        .collect();

    if dry_run {
        println!("[wiki sync] dry-run against {}", wiki_root.display());
        println!("  local changes: {}", dirty.len());
        for l in &dirty {
            println!("    {l}");
        }
        let _ = git_run(&wiki_root, &["fetch", "origin", "main"]).await;
        if let Ok(counts) = git_capture(
            &wiki_root,
            &["rev-list", "--left-right", "--count", "origin/main...HEAD"],
        )
        .await
        {
            println!("  behind/ahead vs origin/main: {}", counts.trim());
        }
        return Ok(());
    }

    // 1. Stage + commit local changes.
    let (git_author_name, git_author_email) = wiki_git_identity();

    if !dirty.is_empty() {
        if !git_run(&wiki_root, &["add", "-A"]).await? {
            anyhow::bail!("git add -A failed");
        }
        // Re-check the STAGED set before we commit anything.
        let staged = git_capture(&wiki_root, &["diff", "--cached", "--name-only"]).await?;
        wiki_sync_check_forbidden(&staged, "staged")?;

        let msg = format!("wiki: sync {} change(s)", dirty.len());
        let committed = git_run(
            &wiki_root,
            &[
                "-c",
                &format!("user.name={git_author_name}"),
                "-c",
                &format!("user.email={git_author_email}"),
                "commit",
                "-m",
                &msg,
            ],
        )
        .await?;
        if committed {
            println!("[wiki sync] committed {} local change(s)", dirty.len());
        }
    } else {
        println!("[wiki sync] no local changes");
    }

    if no_pull {
        // First-push / push-only mode (origin/main may not exist yet).
        return wiki_sync_push(&wiki_root).await;
    }

    // 2. Pull owner edits (owner-wins). `git pull --rebase` replays our
    // local commits onto origin/main; in rebase terms the upstream
    // (owner's GitHub state) is "ours", so `-X ours` makes OWNER edits win
    // same-line conflicts — the confirmed policy. Non-conflicting edits on
    // both sides still merge cleanly.
    if !git_run(&wiki_root, &["fetch", "origin", "main"]).await? {
        // No remote branch yet or offline — nothing to reconcile; push.
        return wiki_sync_push(&wiki_root).await;
    }
    let pre_pull_head = git_capture(&wiki_root, &["rev-parse", "HEAD"]).await?;
    let pre_pull_head = pre_pull_head.trim().to_string();
    let pulled = git_run_as_committer(
        &wiki_root,
        &["pull", "--rebase", "-X", "ours", "origin", "main"],
    )
    .await?;
    if !pulled {
        // Rebase couldn't auto-resolve (e.g. delete/modify). Preserve the
        // daemon's work on a backup branch, abort, and fail loudly. NO
        // data is lost — the owner reconciles manually (KB sync #4).
        let _ = git_run(&wiki_root, &["rebase", "--abort"]).await;
        let short: String = pre_pull_head.chars().take(8).collect();
        let backup = format!("kb-conflict-{short}");
        let _ = git_run(&wiki_root, &["branch", "-f", &backup, &pre_pull_head]).await;
        let _ = git_run(&wiki_root, &["push", "origin", &format!("{backup}:{backup}")]).await;
        anyhow::bail!(
            "wiki sync: unresolved rebase conflict. Daemon commits preserved on branch `{backup}` \
             (pushed to origin). Owner must reconcile manually. NO data lost."
        );
    }

    // #478: validate owner-facing structure after the pull, so a GitHub-side
    // edit that breaks me.md's owner-rules sections or a person-page's
    // frontmatter is caught and warned about (in wiki-sync.log) rather than
    // silently degrading. Only inspects files the pull actually changed.
    let post_pull_head = git_capture(&wiki_root, &["rev-parse", "HEAD"]).await?;
    if post_pull_head.trim() != pre_pull_head {
        let changed = git_capture(
            &wiki_root,
            &["diff", "--name-only", &format!("{pre_pull_head}..HEAD")],
        )
        .await
        .unwrap_or_default();
        let mut problems: Vec<String> = Vec::new();
        for rel in changed.lines().map(str::trim).filter(|l| !l.is_empty()) {
            if let Ok(content) = std::fs::read_to_string(wiki_root.join(rel)) {
                if let Some(p) = wiki_sync_validate_page(rel, &content) {
                    problems.push(p);
                }
            }
        }
        if !problems.is_empty() {
            warn!(
                "wiki sync: pulled owner edits, but {} page(s) no longer validate — \
                 fix them on GitHub:\n  - {}",
                problems.len(),
                problems.join("\n  - ")
            );
        }
    }

    // 3. Push.
    wiki_sync_push(&wiki_root).await
}

/// Adapter: bridges the Discord broker's `QueryHandler` trait to our
/// #655/#667 — `reasoner-selftest`: one live text-only round-trip through
/// the production provider chain. Prints the chain, any active cooldown
/// latches, and the answer. Exit non-zero when the whole chain fails, so a
/// timer/doctor wrapper can alert on it.
async fn run_reasoner_selftest(prompt: &str) -> Result<()> {
    use augmentagent_channel_core::{CooldownLatch, ReasonerOpts};

    let reasoner = build_reasoner();
    println!("chain: {}", reasoner.provider_names().join(" → "));
    let print_cooldowns = |label: &str| {
        let latches = CooldownLatch::system().active();
        if latches.is_empty() {
            println!("{label}: none active");
        } else {
            for (provider, entry) in &latches {
                println!("{label}: {provider} latched until {} ({})", entry.until, entry.reason);
            }
        }
    };
    print_cooldowns("cooldowns");

    // Text-only opts (no tools) so every configured provider is eligible —
    // this is the widest possible probe of the chain. Quality tier keeps the
    // probe on the same models the important presets use.
    let opts = ReasonerOpts {
        system_prompt: "You are a diagnostic probe. Follow the user's instruction exactly, \
                        with no preamble."
            .into(),
        model: Some(
            std::env::var("AUGMENTAGENT_OPUS_MODEL").unwrap_or_else(|_| "claude-opus-4-8".into()),
        ),
        allowed_tools: vec![],
        add_dirs: vec![],
        permission_mode: "default".into(),
        cwd: None,
        env: vec![],
        settings_json: None,
        restrict_env: false,
        audit_logger: None,
        audit_notifier: None,
        session_id: None,
    };
    let result = reasoner.call(&opts, prompt).await;
    // The latches the call itself took are the observable half of a failover
    // — without this line a fault-injection run (#666) can see WHICH
    // provider answered but not that the failed one was actually latched.
    print_cooldowns("cooldowns (after call)");
    match result {
        Ok(text) => {
            println!("response: {text}");
            Ok(())
        }
        Err(e) => {
            eprintln!("selftest FAILED: {e:#}");
            std::process::exit(1);
        }
    }
}

/// `FallbackReasoner` + `ask_opts`. Lives in the CLI to avoid a circular
/// dep between the discord crate and the channel-email crate.
struct WikiQuerier {
    reasoner: Arc<FallbackReasoner>,
    wiki_root: PathBuf,
    repo_root: PathBuf,
}

/// #389 — Owner rules travel with EVERY query-mode prompt, injected at
/// request time rather than left for the model to (maybe) read.
///
/// The system prompt used to only *instruct* the model to read `about/me.md`
/// "before drafting anything" — loading was conditional on the model first
/// classifying the turn as drafting, so corrections the user filed there
/// were dead letters exactly when the turn was misclassified (observed
/// 2026-07-09: both Bo Motlagh sessions read me.md first and still ignored
/// the deliverable-placement rule; a wiki file can't outrank prompt
/// structure). Injecting the sections as a highest-priority preamble removes
/// the classification step entirely, and — unlike the compile-embedded
/// schema — picks up new corrections on the very next turn, no rebuild.
///
/// Returns `None` when me.md is missing or has none of the wanted sections
/// (fresh installs) — the prompt then behaves exactly as before.
fn owner_rules_block(wiki_root: &std::path::Path) -> Option<String> {
    // "Writing style preferences" = how drafts read; "Agent behavior rules"
    // = how turns are conducted (deliverable placement, routing). The split
    // is documented in schema/wiki-ask.md's durable-facts pass.
    const SECTIONS: [&str; 2] = ["Writing style preferences", "Agent behavior rules"];
    // Generous cap: me.md rule sections are a handful of bullets today;
    // truncation is a guard against unbounded growth, not an expectation.
    const MAX_BLOCK_CHARS: usize = 4000;

    let me = std::fs::read_to_string(wiki_root.join("about").join("me.md")).ok()?;
    let mut block = String::new();
    for sec in SECTIONS {
        if let Some(body) = extract_md_section(&me, sec) {
            let body = body.trim();
            if !body.is_empty() {
                block.push_str("### ");
                block.push_str(sec);
                block.push('\n');
                block.push_str(body);
                block.push_str("\n\n");
            }
        }
    }
    if block.trim().is_empty() {
        return None;
    }
    if block.len() > MAX_BLOCK_CHARS {
        let mut end = MAX_BLOCK_CHARS;
        while end > 0 && !block.is_char_boundary(end) {
            end -= 1;
        }
        block.truncate(end);
        block.push_str("\n[truncated — read wiki/about/me.md for the rest]\n");
    }
    Some(block)
}

/// Return the body of the `## <heading>` section of a markdown doc: the text
/// after the heading line up to (not including) the next `## ` heading or
/// EOF. Exact heading match after the `## ` prefix (trimmed).
fn extract_md_section<'a>(md: &'a str, heading: &str) -> Option<&'a str> {
    let mut start: Option<usize> = None;
    for line in md.lines() {
        let idx = line.as_ptr() as usize - md.as_ptr() as usize;
        if let Some(h) = line.strip_prefix("## ") {
            match start {
                None if h.trim() == heading => start = Some(idx + line.len()),
                Some(s) => return Some(&md[s..idx]),
                None => {}
            }
        }
    }
    start.map(|s| &md[s..])
}

#[async_trait]
impl QueryHandler for WikiQuerier {
    async fn answer(
        &self,
        ctx: &augmentagent_approval_discord::AuditCtx,
        question: &str,
    ) -> anyhow::Result<String> {
        let mut opts = ask_opts(self.wiki_root.clone(), self.repo_root.clone());
        // #132 / #201 — Stamp this request's session id onto every audit
        // record produced by the spawn, and (if we have the bits from the
        // Discord side) plug in a per-request notifier so high-risk tool
        // calls ping back to the originating channel.
        opts.session_id = Some(ctx.session_id.clone());
        if let (Some(http), Some(channel_id)) = (ctx.http.clone(), ctx.channel_id) {
            opts.audit_notifier = Some(std::sync::Arc::new(DiscordAuditNotifier {
                http,
                channel_id,
            }));
        }
        // #389 — prepend the owner's standing rules as a highest-priority
        // block so they apply on the first attempt, every turn.
        let prompt = match owner_rules_block(&self.wiki_root) {
            Some(rules) => format!(
                "<owner_rules>\nStanding rules from the owner (wiki/about/me.md). These are \
                 HIGHEST PRIORITY: when they conflict with any default behavior in your \
                 instructions, the owner's rules win. Apply them on the first attempt, \
                 without being asked.\n\n{rules}</owner_rules>\n\n{question}"
            ),
            None => question.to_string(),
        };
        // #236 — prepend the current time so the agent can reason about "now".
        let prompt = format!("{}{prompt}", now_awareness_line());
        // #446 — see `wiki ask`: the Discord reply must carry every text block
        // the model emitted, not just the trailing wiki-filing receipt.
        self.reasoner.call_transcript(&opts, &prompt).await
    }
}

/// Bridge: turns the raw serenity bits in `AuditCtx` into a channel-core
/// [`AuditNotifier`] impl. Lives in the CLI crate because it's the only
/// crate that depends on BOTH the discord crate (for `serenity` + `AuditCtx`)
/// and channel-core (for the trait). Sees `&AuditRecord` directly so it can
/// reuse [`augmentagent_channel_core::format_notice`] verbatim.
#[derive(Debug, Clone)]
struct DiscordAuditNotifier {
    http: std::sync::Arc<serenity::http::Http>,
    channel_id: serenity::model::id::ChannelId,
}

#[async_trait]
impl augmentagent_channel_core::AuditNotifier for DiscordAuditNotifier {
    async fn notify(
        &self,
        _session_id: &str,
        record: &augmentagent_channel_core::AuditRecord,
    ) {
        let body = augmentagent_channel_core::format_notice(record);
        let builder = serenity::builder::CreateMessage::new().content(body);
        if let Err(e) = self.channel_id.send_message(&*self.http, builder).await {
            tracing::warn!("tool-audit notify failed: {e}");
        }
    }
}

/// `/loop` runner (#104): fires a stored loop prompt through the exact same
/// `claude` reasoner + `ask_opts` toolbelt the wiki-ask path uses, so
/// `/loop 1h what's new in my inbox` behaves identically to asking the bot.
struct LoopReasonerRunner {
    reasoner: Arc<FallbackReasoner>,
    wiki_root: PathBuf,
    repo_root: PathBuf,
}

#[async_trait]
impl LoopRunner for LoopReasonerRunner {
    async fn run_prompt(&self, prompt: &str) -> anyhow::Result<String> {
        let opts = ask_opts(self.wiki_root.clone(), self.repo_root.clone());
        // #389 — loops fire through the same query toolbelt, so they carry
        // the same owner-rules preamble as interactive asks.
        let prompt = match owner_rules_block(&self.wiki_root) {
            Some(rules) => format!(
                "<owner_rules>\nStanding rules from the owner (wiki/about/me.md). These are \
                 HIGHEST PRIORITY: when they conflict with any default behavior in your \
                 instructions, the owner's rules win.\n\n{rules}</owner_rules>\n\n{prompt}"
            ),
            None => prompt.to_string(),
        };
        // #236 — prepend the current time so loop-driven queries know "now".
        let prompt = format!("{}{prompt}", now_awareness_line());
        // #446 — loops render their output to Discord too; same reasoning.
        self.reasoner.call_transcript(&opts, &prompt).await
    }
}

/// `loop` command parser: asks Haiku to extract {interval, prompt, duration?}
/// from arbitrary phrasing. See `loop_parse_opts` for the system prompt.
struct LoopReasonerParser {
    reasoner: Arc<FallbackReasoner>,
}

#[async_trait]
impl augmentagent_approval_discord::LoopCommandParser for LoopReasonerParser {
    async fn parse(
        &self,
        raw: &str,
    ) -> std::result::Result<augmentagent_approval_discord::ParsedLoop, String> {
        use augmentagent_channel_core::Reasoner;
        let opts = augmentagent_channel_core::reasoner::loop_parse_opts();
        let answer = match self.reasoner.call(&opts, raw).await {
            Ok(a) => a,
            Err(e) => {
                tracing::warn!("loop parser claude call failed: {e:#}");
                return Err(format!("couldn't reach claude to parse loop: {e}"));
            }
        };
        parse_loop_json(&answer)
    }
}

/// Strip code fences if Claude added them, then extract the first JSON object
/// and shape it into a `ParsedLoop` or a user-facing error.
fn parse_loop_json(raw: &str) -> std::result::Result<augmentagent_approval_discord::ParsedLoop, String> {
    let text = raw.trim();
    // Tolerate ```json … ``` or ``` … ``` fences.
    let stripped = text
        .strip_prefix("```json")
        .or_else(|| text.strip_prefix("```"))
        .and_then(|s| s.rsplit_once("```").map(|(body, _)| body))
        .unwrap_or(text)
        .trim();
    // Extract the first {...} object so prose around the JSON is tolerated.
    let json_blob = match (stripped.find('{'), stripped.rfind('}')) {
        (Some(a), Some(b)) if b > a => &stripped[a..=b],
        _ => return Err(format!("loop parser returned no JSON: {raw}")),
    };
    let parsed: serde_json::Value = match serde_json::from_str(json_blob) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!("loop parser json decode failed: {e}; raw={raw}");
            return Err("couldn't parse loop spec — try `loop 5m do thing`".to_string());
        }
    };
    if let Some(err) = parsed.get("error").and_then(|v| v.as_str()) {
        return Err(err.to_string());
    }
    let interval = parsed
        .get("interval_secs")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| "loop parser omitted interval".to_string())?;
    let prompt = parsed
        .get("prompt")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "loop parser omitted prompt".to_string())?;
    let duration_secs = parsed
        .get("duration_secs")
        .and_then(|v| if v.is_null() { None } else { v.as_i64() });
    // #231 — cron-style scheduling. The LLM parser emits `cron_expr` +
    // `tz` (both strings) when the user asked for day-of-week / specific
    // time. Either-both-or-neither; we don't enforce that here — the
    // scheduler/store boundary validates before persisting.
    let cron_expr = parsed
        .get("cron_expr")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let tz = parsed
        .get("tz")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    Ok(augmentagent_approval_discord::ParsedLoop {
        interval_secs: interval,
        prompt,
        duration_secs,
        cron_expr,
        tz,
    })
}

#[cfg(test)]
mod unescape_body_tests {
    use super::unescape_body;

    #[test]
    fn newline_and_tab_escapes_interpret() {
        assert_eq!(unescape_body("a\\n\\nb\\tc"), "a\n\nb\tc");
    }

    #[test]
    fn the_john_repro_renders_paragraphs() {
        let body = "Hey John,\\n\\nGreat catching up on the phone today.\\n\\nBest,\\nNolan";
        let out = unescape_body(body);
        assert!(out.contains("Hey John,\n\nGreat"));
        assert!(!out.contains('\\'), "no backslashes may survive: {out}");
    }

    #[test]
    fn plain_text_and_real_newlines_untouched() {
        assert_eq!(unescape_body("line1\nline2"), "line1\nline2");
        assert_eq!(unescape_body("no escapes here"), "no escapes here");
    }

    #[test]
    fn double_backslash_yields_literal_backslash_n() {
        assert_eq!(unescape_body("show \\\\n literally"), "show \\n literally");
    }

    #[test]
    fn crlf_escape_collapses_to_newline() {
        assert_eq!(unescape_body("a\\r\\nb"), "a\nb");
    }

    #[test]
    fn unknown_escape_passes_through() {
        assert_eq!(unescape_body("path\\qthing"), "path\\qthing");
    }
}

#[cfg(test)]
mod approval_body_tests {
    use super::{
        body_without_leaked_subject, compose_pending_disposition, revise_subject, revised_subject,
        strip_approval_envelope_markers, strip_leading_subject_line, subjects_agree,
        thread_for_revised_subject, thread_subject_conflict, ComposePendingDisposition,
        ThreadSubject,
    };

    #[test]
    fn pending_follow_up_replaces_the_existing_card() {
        assert_eq!(
            compose_pending_disposition(true, false, Some(("action-1", Some("draft-1")))),
            ComposePendingDisposition::Replace {
                action_id: "action-1".into(),
                draft_id: Some("draft-1".into()),
            }
        );
    }

    #[test]
    fn explicit_duplicate_override_still_creates_another_card() {
        assert_eq!(
            compose_pending_disposition(true, true, Some(("action-1", Some("draft-1")))),
            ComposePendingDisposition::Create
        );
    }

    #[test]
    fn removes_display_only_recipient_lines() {
        let body = "Hi Bo,\n\nThanks.\n\nCC: ccrimi75@gmail.com";
        assert_eq!(strip_approval_envelope_markers(body), "Hi Bo,\n\nThanks.");
        assert_eq!(
            strip_approval_envelope_markers("Hi\n[cc: ccrimi75@gmail.com]"),
            "Hi"
        );
        assert_eq!(
            strip_approval_envelope_markers("Hi\nCC: a@example.com, b@example.com"),
            "Hi"
        );
    }

    #[test]
    fn keeps_normal_prose_and_non_email_colons() {
        let body = "Hi,\n\nCC: means carbon copy in this sentence.\nTo: the team";
        assert_eq!(strip_approval_envelope_markers(body), body);
        // #652 (codex on #855): the reposted card's `[subject: …]` marker is
        // display-only and must never reach a sent body; a bare `Subject:`
        // line is prose and stays.
        assert_eq!(
            strip_approval_envelope_markers("[subject: Invoice for July]\nHi Alice,\n[to: a@example.com]"),
            "Hi Alice,"
        );
        assert_eq!(
            strip_approval_envelope_markers("Subject: not a marker\nHi"),
            "Subject: not a marker\nHi"
        );
    }

    #[test]
    fn leading_subject_line_is_dropped() {
        assert_eq!(
            strip_leading_subject_line("Subject: Hosting a PTE event\n\nHey Casey,\n\nThanks."),
            (
                "Hey Casey,\n\nThanks.".to_string(),
                Some("Hosting a PTE event".to_string())
            )
        );
    }

    #[test]
    fn subject_match_is_case_and_whitespace_insensitive() {
        assert_eq!(
            strip_leading_subject_line("\n  SUBJECT:  venue question \nHi"),
            ("Hi".to_string(), Some("venue question".to_string()))
        );
    }

    #[test]
    fn mid_body_subject_line_survives() {
        let body = "Hi,\n\n> Subject: old thread\nbye";
        assert_eq!(strip_leading_subject_line(body), (body.to_string(), None));
    }

    #[test]
    fn remainder_is_preserved_verbatim_including_crlf() {
        assert_eq!(
            strip_leading_subject_line("Subject: x\r\n\r\nHi\r\nBye\r\n"),
            ("Hi\r\nBye\r\n".to_string(), Some("x".to_string()))
        );
    }

    #[test]
    fn body_without_subject_is_untouched() {
        let body = "Hey Casey,\n\nThanks.";
        assert_eq!(strip_leading_subject_line(body), (body.to_string(), None));
    }

    /// #650 verbatim: the reply body the agent handed to `gmail compose`
    /// opened with the subject it meant for the header, so the recipient saw
    /// a literal "Subject:" line above the greeting.
    #[test]
    fn the_reported_subject_line_never_reaches_the_recipient() {
        assert_eq!(
            body_without_leaked_subject(
                "Subject: Hosting a PTE event\n\nHey Casey,\n\nWe'd love to host it.",
                "Re: Hosting a PTE event"
            ),
            Ok((
                "Hey Casey,\n\nWe'd love to host it.".to_string(),
                Some("Hosting a PTE event".to_string())
            ))
        );
    }

    #[test]
    fn a_body_subject_that_disagrees_with_the_header_is_refused() {
        let err = body_without_leaked_subject(
            "Subject: Hosting a PTE event\n\nHey Casey,",
            "Re: Thursday's venue",
        )
        .unwrap_err();
        assert!(err.contains("Hosting a PTE event"), "{err}");
        assert!(err.contains("Thursday's venue"), "{err}");
    }

    #[test]
    fn a_subject_only_body_is_refused_before_any_gmail_write() {
        assert!(
            body_without_leaked_subject("Subject: Hosting a PTE event\n", "Hosting a PTE event")
                .is_err()
        );
    }

    /// #650/#651 discipline (codex on #857): every Gmail write path checks
    /// the body's own `Subject:` line and the thread's subject BEFORE it
    /// uploads an attachment or creates a draft, so a refusal never strands
    /// an orphan upload. Pinned structurally on the three CLI paths.
    #[test]
    fn refusals_run_before_any_attachment_upload_on_every_write_path() {
        let src = include_str!("main.rs");
        for f in ["async fn run_gmail_compose(", "async fn run_gmail_update_draft(", "async fn run_gmail_send_now("] {
            let start = src.find(f).expect(f);
            let body = &src[start..start + src[start..].find("\n}\n").expect("fn end")];
            let body_gate = body.find("body_for_gmail_write(").expect("body subject gate");
            let thread_gate = body.find("ensure_subject_matches_thread(").expect("thread subject gate (#651)");
            let upload = body.find("upload_attach_if_given(").expect("attachment upload");
            let draft = body.find("create_draft_with_attachment(").or_else(|| body.find("update_draft_with_attachment(")).expect("draft write");
            assert!(body_gate < upload && thread_gate < upload, "{f}: a refusal must precede the upload");
            assert!(upload < draft, "{f}: upload precedes the draft write");
        }
    }

    #[test]
    fn reply_and_forward_prefixes_do_not_make_two_subjects_disagree() {
        assert!(subjects_agree("Hosting a PTE event", "RE: hosting a PTE event"));
        assert!(subjects_agree("Fwd: Re: venue", "venue"));
        assert!(!subjects_agree("venue", "venue tomorrow"));
    }

    #[test]
    fn revising_a_new_email_card_never_prefixes_re() {
        assert_eq!(
            revise_subject("Ship Systems x EFS", None),
            "Ship Systems x EFS"
        );
    }

    #[test]
    fn revising_a_reply_card_prefixes_re_at_most_once() {
        assert_eq!(
            revise_subject("Ship Systems x EFS", Some("t1")),
            "Re: Ship Systems x EFS"
        );
        assert_eq!(revise_subject("Re: hi", Some("t1")), "Re: hi");
        assert_eq!(revise_subject("RE: hi", Some("t1")), "RE: hi");
    }

    // ---- #651: a --subject that disagrees with the thread it replies on ----

    fn thread() -> ThreadSubject {
        ThreadSubject::Original("Ground Floor as a venue?".into())
    }

    #[test]
    fn conflicting_subject_on_a_thread_is_refused_naming_both_subjects() {
        let msg = thread_subject_conflict("Hosting an event at Tactic?", &thread())
            .expect("a subject unrelated to the thread must be refused");
        assert!(
            msg.contains("Hosting an event at Tactic?"),
            "refusal must quote the given subject: {msg}"
        );
        assert!(
            msg.contains("Ground Floor as a venue?"),
            "refusal must quote the thread's subject: {msg}"
        );
        assert!(
            msg.contains("--thread-id"),
            "refusal must name both ways out: {msg}"
        );
    }

    #[test]
    fn subject_matching_the_thread_is_allowed_ignoring_prefix_and_case() {
        assert_eq!(
            thread_subject_conflict("Re: Ground Floor as a venue?", &thread()),
            None
        );
        assert_eq!(
            thread_subject_conflict("ground floor as a venue?", &thread()),
            None
        );
    }

    // A subject someone introduced later in the thread is NOT a pass: Gmail
    // still sends under the original, so allowing it re-opens #651.
    #[test]
    fn subject_renamed_mid_thread_is_refused_against_the_original() {
        let msg = thread_subject_conflict(
            "Renamed mid-thread",
            &ThreadSubject::Original("Original".into()),
        )
        .expect("only the thread's original subject may be replied under");
        assert!(
            msg.contains("Original"),
            "refusal must quote the subject Gmail will send under: {msg}"
        );
    }

    // A forward is a different subject line: Gmail would still send it under
    // the thread's own subject, so "Fwd: X" on a thread titled "X" is the
    // same header/intent mismatch as any other rename.
    #[test]
    fn forwarding_subject_on_a_thread_is_refused() {
        assert!(
            thread_subject_conflict("Fwd: Original", &ThreadSubject::Original("Original".into()))
                .is_some(),
            "a forward would go out under the plain thread subject"
        );
        // Replying to a thread that IS a forward still matches.
        assert_eq!(
            thread_subject_conflict(
                "Re: Fwd: Original",
                &ThreadSubject::Original("Fwd: Original".into())
            ),
            None
        );
    }

    // Can't tell what Gmail will send under ⇒ can't verify ⇒ refuse.
    #[test]
    fn an_unknowable_thread_subject_is_refused() {
        let msg =
            thread_subject_conflict("Hosting an event at Tactic?", &ThreadSubject::Undetermined)
                .expect("an unverifiable thread subject must not be drafted against");
        assert!(
            msg.contains("--thread-id"),
            "refusal must name the way out: {msg}"
        );
    }

    #[test]
    fn nothing_to_compare_never_refuses() {
        assert_eq!(
            thread_subject_conflict("Anything", &ThreadSubject::Missing),
            None
        );
        assert_eq!(
            thread_subject_conflict("Anything", &ThreadSubject::Original("  ".into())),
            None
        );
        assert_eq!(thread_subject_conflict("   ", &thread()), None);
        // No subject stated ⇒ no intent to contradict, even unverifiably.
        assert_eq!(
            thread_subject_conflict("", &ThreadSubject::Undetermined),
            None
        );
    }

    #[test]
    fn a_user_written_re_subject_survives_on_a_new_email_card() {
        assert_eq!(revise_subject("Re: hi", None), "Re: hi");
    }

    #[test]
    fn a_subject_asked_for_during_revise_reaches_the_draft_verbatim() {
        // #652 — the redraft's `Subject:` line beats both the persisted
        // override and the derived reply subject, with no forced `Re:`: the
        // user asked for those exact words.
        assert_eq!(
            revised_subject(
                Some("Invoice for July"),
                Some("Older override"),
                "Ship Systems x EFS",
                Some("t1"),
            ),
            "Invoice for July"
        );
    }

    #[test]
    fn a_later_revise_keeps_the_subject_an_earlier_one_set() {
        // "make it shorter" must not silently undo the subject change.
        assert_eq!(
            revised_subject(None, Some("Invoice for July"), "Ship Systems x EFS", Some("t1")),
            "Invoice for July"
        );
    }

    // #652 × #651 (codex on #855): Gmail sends a threaded reply under the
    // thread's subject, so a revised subject only takes effect off-thread.
    #[test]
    fn a_new_subject_leaves_the_thread_but_a_reply_subject_keeps_it() {
        // The reported case: "Invoice for July" asked for on a reply in the
        // "Ship Systems x EFS" thread ⇒ new thread, never a silently-dropped header.
        assert_eq!(
            thread_for_revised_subject("Invoice for July", "Ship Systems x EFS", Some("t1")),
            None
        );
        // Ordinary replies (derived or user-written `Re:`) stay threaded.
        assert_eq!(
            thread_for_revised_subject("Re: Ship Systems x EFS", "Ship Systems x EFS", Some("t1")),
            Some("t1")
        );
        assert_eq!(
            thread_for_revised_subject("re: RE: ship systems x efs", "Re: Ship Systems x EFS", Some("t1")),
            Some("t1")
        );
        // A forward is a different message to recipients (#651's rule), so it
        // leaves the thread too.
        assert_eq!(
            thread_for_revised_subject("Fwd: Ship Systems x EFS", "Ship Systems x EFS", Some("t1")),
            None
        );
        // No thread to begin with, or an untitled inbound: nothing to leave.
        assert_eq!(thread_for_revised_subject("Invoice for July", "Ship Systems x EFS", None), None);
        // An untitled inbound thread: a new subject leaves it, staying untitled keeps it.
        assert_eq!(thread_for_revised_subject("Invoice for July", "", Some("t1")), None);
        assert_eq!(thread_for_revised_subject("Re: ", "", Some("t1")), Some("t1"));
    }

    // Structural (codex on #855): the subject override is written only
    // inside the success arm of the draft-id write, after the CAS.
    #[test]
    fn revise_persists_the_subject_only_with_its_draft_id() {
        let src = include_str!("main.rs");
        let start = src.find("match self.store.refresh_pending_draft(action_id, &redraft)").expect("CAS");
        let body = &src[start..start + 4000];
        let id_write = body.find("match self.store.set_action_draft_id(action_id, &new_draft_id)").expect("gated id write");
        let ok_arm = body[id_write..].find("Ok(()) =>").expect("success arm") + id_write;
        let subject_write = body[id_write..].find("set_action_subject(action_id, Some(&subject))").expect("subject write") + id_write;
        let err_arm = body[id_write..].find("Err(e) => tracing::warn!").expect("failure arm") + id_write;
        assert!(ok_arm < subject_write && subject_write < err_arm, "subject write must sit in the Ok arm");
    }

    #[test]
    fn an_untouched_subject_still_derives_from_the_inbound_mail() {
        assert_eq!(
            revised_subject(None, None, "Ship Systems x EFS", Some("t1")),
            "Re: Ship Systems x EFS"
        );
    }
}

#[cfg(test)]
mod owner_rules_tests {
    use super::{extract_md_section, owner_rules_block};

    const ME_MD: &str = "# About Me\n\n## Identity\n\nNolan.\n\n## Writing style preferences\n\n- No em-dashes. (user said, 2026-05-04)\n- Deliverable is the text itself. (user said, 2026-07-08)\n\n## Agent behavior rules\n\n- Email asks end with an approval card.\n\n## Routing preferences\n\n- VIPs flagged.\n";

    #[test]
    fn extracts_section_body_up_to_next_heading() {
        let body = extract_md_section(ME_MD, "Writing style preferences").unwrap();
        assert!(body.contains("No em-dashes"));
        assert!(body.contains("Deliverable is the text itself"));
        assert!(!body.contains("approval card"), "must stop at next ## heading");
    }

    #[test]
    fn extracts_last_section_to_eof() {
        let body = extract_md_section(ME_MD, "Routing preferences").unwrap();
        assert!(body.contains("VIPs flagged"));
    }

    #[test]
    fn missing_section_returns_none() {
        assert!(extract_md_section(ME_MD, "Nonexistent").is_none());
    }

    #[test]
    fn block_includes_style_and_behavior_sections() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("about")).unwrap();
        std::fs::write(tmp.path().join("about").join("me.md"), ME_MD).unwrap();
        let block = owner_rules_block(tmp.path()).expect("block should build");
        assert!(block.contains("### Writing style preferences"));
        assert!(block.contains("### Agent behavior rules"));
        assert!(block.contains("approval card"));
        // Non-rule sections must NOT leak into every prompt.
        assert!(!block.contains("VIPs flagged"));
        assert!(!block.contains("Nolan."));
    }

    #[test]
    fn missing_me_md_yields_none() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(owner_rules_block(tmp.path()).is_none());
    }

    #[test]
    fn oversized_block_truncates_with_marker() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("about")).unwrap();
        let big = format!(
            "## Writing style preferences\n\n{}\n",
            "- rule with some padding text to inflate the size\n".repeat(200)
        );
        std::fs::write(tmp.path().join("about").join("me.md"), big).unwrap();
        let block = owner_rules_block(tmp.path()).unwrap();
        assert!(block.len() < 4200, "cap not applied: {} chars", block.len());
        assert!(block.contains("[truncated"), "missing truncation marker");
    }
}

#[cfg(test)]
mod loop_parser_tests {
    use super::parse_loop_json;

    #[test]
    fn parses_strict_json() {
        let raw = r#"{"interval_secs": 300, "prompt": "say hi", "duration_secs": 900}"#;
        let p = parse_loop_json(raw).unwrap();
        assert_eq!(p.interval_secs, 300);
        assert_eq!(p.prompt, "say hi");
        assert_eq!(p.duration_secs, Some(900));
    }

    #[test]
    fn parses_with_null_duration() {
        let raw = r#"{"interval_secs": 60, "prompt": "ping", "duration_secs": null}"#;
        let p = parse_loop_json(raw).unwrap();
        assert_eq!(p.duration_secs, None);
    }

    #[test]
    fn strips_code_fences() {
        let raw = "```json\n{\"interval_secs\": 30, \"prompt\": \"x\", \"duration_secs\": null}\n```";
        let p = parse_loop_json(raw).unwrap();
        assert_eq!(p.interval_secs, 30);
        assert_eq!(p.prompt, "x");
    }

    #[test]
    fn surfaces_error_field_verbatim() {
        let raw = r#"{"error": "couldn't find an interval"}"#;
        let err = parse_loop_json(raw).unwrap_err();
        assert_eq!(err, "couldn't find an interval");
    }

    #[test]
    fn tolerates_prose_around_json() {
        let raw = "Sure! {\"interval_secs\": 300, \"prompt\": \"do thing\", \"duration_secs\": null} hope this helps.";
        let p = parse_loop_json(raw).unwrap();
        assert_eq!(p.interval_secs, 300);
    }

    #[test]
    fn rejects_empty_prompt() {
        let raw = r#"{"interval_secs": 300, "prompt": "   ", "duration_secs": null}"#;
        assert!(parse_loop_json(raw).is_err());
    }

    #[test]
    fn rejects_missing_interval() {
        let raw = r#"{"prompt": "x", "duration_secs": null}"#;
        assert!(parse_loop_json(raw).is_err());
    }
}

/// Posts a loop's result back to the originating Discord channel/DM using a
/// bare serenity HTTP client (no gateway) — same approach as the digest
/// poster. `channel_ref` is the stringified channel id captured at creation.
struct DiscordLoopPoster {
    http: Arc<serenity::http::Http>,
}

#[async_trait]
impl LoopPoster for DiscordLoopPoster {
    async fn post_to(&self, channel_ref: &str, body: &str) -> anyhow::Result<()> {
        use serenity::all::{ChannelId, CreateMessage};
        let cid: u64 = channel_ref
            .parse()
            .with_context(|| format!("loop channel_ref not a u64: {channel_ref}"))?;
        let channel = ChannelId::new(cid);
        for chunk in augmentagent_approval_discord::chunk_for_discord(body) {
            channel
                .send_message(&*self.http, CreateMessage::new().content(chunk))
                .await
                .context("discord send_message (loop result)")?;
        }
        Ok(())
    }
}

/// #428 — bridge implementing the discord crate's `JournalOps` trait:
/// `!journal` text → paragraph HTML → KMS envelope encrypt → AppSync
/// `createEntry` (shows up in the ShadowNote app), plus an immediate wiki
/// ingest so wiki-ask sees the entry before the next poller pass. Every
/// failure reply carries the entry text back so nothing is silently lost.
struct CliJournalOps {
    runtime: augmentagent_channel_journal::JournalRuntime,
    reasoner: Arc<FallbackReasoner>,
    wiki_root: Option<PathBuf>,
    wiki_schema: Option<String>,
}

impl CliJournalOps {
    async fn save_html(&self, title: Option<String>, html: &str) -> Result<String, String> {
        use augmentagent_channel_core::decision::DecisionKind;
        use augmentagent_channel_journal as journal;
        use augmentagent_channel_journal::JournalApi as _;

        let text_for_replies = journal::html_to_text(html);
        let Some(kms_arn) = self.runtime.config.kms_key_arn.as_deref() else {
            return Err(format!(
                "SHADOWNOTE_KMS_KEY_ARN isn't set, so I can't encrypt — entry NOT saved. \
                 Your text (safe to retry):\n{text_for_replies}"
            ));
        };
        let content = journal::encrypt_entry_content(html, kms_arn, self.runtime.dek.as_ref())
            .await
            .map_err(|e| {
                format!(
                    "encrypt failed ({e}) — entry NOT saved. Your text (safe to retry):\n{text_for_replies}"
                )
            })?;
        let created_at = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
        let entry = self
            .runtime
            .client
            .create_entry(journal::NewEntry {
                created_at,
                content,
                // Both title sources (composer model, `!journal done <title>`)
                // converge here; neither may carry markup into the app.
                title: title.as_deref().and_then(journal::compose::sanitize_title),
                topic: Some("Journal".into()),
            })
            .await
            .map_err(|e| {
                format!(
                    "ShadowNote save failed ({e}) — entry NOT saved. Your text (safe to retry):\n{text_for_replies}"
                )
            })?;
        // Immediate wiki ingest — the syncEntries poller would pick it up
        // within a cycle anyway; this just closes the gap to "ask about it
        // right now".
        if let (Some(root), Some(schema)) = (&self.wiki_root, &self.wiki_schema) {
            augmentagent_channel_core::ingest::spawn_ingest(
                Arc::clone(&self.reasoner),
                root.clone(),
                schema.clone(),
                journal::channel::synthetic_journal_email(&entry, &text_for_replies),
                DecisionKind::Capture,
                Some("discord journaling session".to_string()),
                None,
                augmentagent_channel_core::ingest::IngestTrigger::Journal,
            );
        }
        Ok(format!(
            "📓 Saved to ShadowNote — “{}” ({})",
            entry.title.as_deref().unwrap_or("Journal entry"),
            entry.created_at
        ))
    }
}

#[async_trait]
impl augmentagent_approval_discord::JournalOps for CliJournalOps {
    async fn save_text(&self, title: Option<String>, text: &str) -> Result<String, String> {
        let html = augmentagent_channel_journal::compose::text_to_paragraphs(text);
        if html.is_empty() {
            return Err("that message was empty after trimming — nothing saved".to_string());
        }
        self.save_html(title, &html).await
    }

    async fn compose_and_save(
        &self,
        history: &str,
        title_override: Option<String>,
    ) -> Result<String, String> {
        use augmentagent_channel_journal::compose;

        let raw = self
            .reasoner
            .call(&compose::compose_opts(), &compose::compose_user_message(history))
            .await
            .map_err(|e| {
                format!(
                    "compose failed ({e:#}) — nothing saved. You can save directly with \
                     `!journal <text>`."
                )
            })?;
        let (composed_title, html) = compose::parse_composed_entry(&raw);
        if html.is_empty() {
            return Err(
                "the composer returned an empty entry — nothing saved. Try `!journal <text>`."
                    .to_string(),
            );
        }
        self.save_html(title_override.or(composed_title), &html).await
    }
}

/// Executes Approve / Revise / Skip clicks against sqlite + Composio +
/// reasoner. Backed entirely by the persistent action row — no in-memory
/// state — so cards remain valid across daemon restarts and indefinitely.
///
/// Routes each click to Gmail or LinkedIn based on the email's
/// `account_entity_id` prefix (`linkedin:` = LinkedIn, else Gmail).
struct ReplyApprover {
    store: Arc<Store>,
    gmail: Arc<ComposioClient>,
    /// Composio Google Calendar client. Executes calendar-event proposals
    /// on Approve (#398). Same API key as `gmail`.
    calendar: Arc<augmentagent_channel_calendar::ComposioCalendarClient>,
    /// Optional voyager client. `None` = LinkedIn disabled for this run
    /// (cookies not configured). Any LinkedIn-tagged action hitting this
    /// approver with a None client surfaces as `Failed`.
    linkedin: Option<Arc<VoyagerClient>>,
    /// Optional Discord client. `None` = Discord disabled for this run
    /// (auth not loaded). Any discord-tagged action hits `Failed`.
    discord: Option<Arc<augmentagent_channel_discord_dm::DiscordClient>>,
    /// Per-workspace Slack clients keyed by Slack `team_id`. Empty map =
    /// Slack disabled for this run (no workspaces loaded). Slack-tagged
    /// actions whose `team_id` isn't in the map surface as `Failed`.
    slack: std::collections::HashMap<String, Arc<augmentagent_channel_slack::SlackClient>>,
    /// Per-bot Telegram clients keyed by numeric `bot_id`. Empty map =
    /// Telegram disabled for this run. Telegram-tagged actions whose
    /// `bot_id` isn't in the map surface as `Failed`.
    telegram: std::collections::HashMap<i64, Arc<augmentagent_channel_telegram_bot::TelegramBotClient>>,
    /// Optional GitHub REST client. `None` = no PAT in keyring; any
    /// github-tagged action hits `Failed`.
    github: Option<Arc<augmentagent_channel_github::GithubClient>>,
    /// Optional SocialAPI.ai client. `None` = no `SOCIALAPI_API_KEY` /
    /// keyring entry this run; any socialapi-tagged action hits `Failed`.
    /// Drives the comment-reply / DM-reply send on Approve (#244).
    socialapi: Option<Arc<augmentagent_channel_socialapi::SocialApiClient>>,
    reasoner: Arc<FallbackReasoner>,
    draft_skill: String,
    wiki_root: Option<PathBuf>,
    /// Set after construction (in serve) to allow approve/skip handlers to
    /// trigger the next queue card immediately on terminal outcome. Held as
    /// `Weak` to break the Approver ↔ Scheduler ↔ Broker reference cycle.
    /// Empty in dry-run / one-shot poll commands.
    nudge: std::sync::OnceLock<std::sync::Weak<augmentagent_approval_discord::NudgeScheduler>>,
    /// #501 — broker handle for the schedule verbs: `schedule` posts the
    /// scheduled notice, Send Now / Cancel delete it, Back to queue reposts
    /// the approval card. Same set-after-construction `Weak` pattern as
    /// `nudge` (the broker's event handler holds this approver strongly, so
    /// a strong back-reference would cycle). Empty in dry-run / one-shot
    /// commands — every use is best-effort.
    broker: std::sync::OnceLock<std::sync::Weak<dyn ApprovalBroker>>,
}

impl ReplyApprover {
    fn handle_load(
        &self,
        action_id: &str,
    ) -> Option<augmentagent_store::ActionWithEmail> {
        match self.store.get_action_with_email(action_id) {
            Ok(a) => a,
            Err(e) => {
                tracing::warn!(action_id, "approver: store lookup failed: {e}");
                None
            }
        }
    }

    /// Surface the next queue item if the user just resolved one. Best-effort:
    /// if the scheduler is gone (Weak upgrade fails) or the post fails, the
    /// next 60s scheduler tick will catch up. Called only after Approved or
    /// Skipped outcomes — not on Revised (revise keeps the card active).
    async fn trigger_next_nudge(&self) {
        let Some(weak) = self.nudge.get() else { return };
        let Some(scheduler) = weak.upgrade() else { return };
        if let Err(e) = scheduler.post_next_if_idle().await {
            tracing::warn!("trigger_next_nudge failed: {e:#}");
        }
    }

    /// #501 — upgraded broker handle, if the serve arm wired one. `None`
    /// (dry-run, one-shot commands, broker torn down) skips notice work —
    /// the schedule state machine never depends on Discord succeeding.
    fn broker_handle(&self) -> Option<Arc<dyn ApprovalBroker>> {
        self.broker.get().and_then(std::sync::Weak::upgrade)
    }

    /// #501 — delete a previously snapshotted scheduled-notice pointer pair.
    /// Cancel and Back-to-queue clear the pointers inside their CAS UPDATE,
    /// so those callers snapshot `action_notice` BEFORE the CAS and pass the
    /// pair here. Debug-level on failure: the Discord-side event handler
    /// also deletes the interaction's own message, so "already gone" is the
    /// common case, not a problem.
    async fn delete_notice_pair(&self, action_id: &str, notice: Option<(String, String)>) {
        let Some((chan, msg)) = notice else { return };
        let Some(broker) = self.broker_handle() else { return };
        if let (Ok(c), Ok(m)) = (chan.parse::<u64>(), msg.parse::<u64>()) {
            if let Err(e) = broker.delete_message(c, m).await {
                tracing::debug!(
                    action_id,
                    "notice delete failed (may already be gone): {e}"
                );
            }
        }
    }

    /// #501 — delete the persisted scheduled-notice message and clear its
    /// pointers. For rows whose CAS does NOT clear pointers itself (the
    /// Send Now claim retains them). The verb-aware startup sweep is the
    /// durable backstop when this best-effort pass misses.
    async fn delete_stored_notice(&self, action_id: &str) {
        let notice = self.store.action_notice(action_id).ok().flatten();
        if notice.is_some() {
            self.delete_notice_pair(action_id, notice).await;
            let _ = self.store.clear_action_notice(action_id);
        }
    }

    async fn approve_linkedin(
        &self,
        action_id: &str,
        action: augmentagent_store::ActionWithEmail,
    ) -> ApprovalActionOutcome {
        let Some(linkedin) = self.linkedin.as_ref() else {
            return ApprovalActionOutcome::Failed {
                message: "LinkedIn is not configured (no cookies); run `linkedin login`".into(),
            };
        };
        let Some(body) = action.action.draft_body.as_deref() else {
            return ApprovalActionOutcome::Failed {
                message: "no draft body on action; cannot send".into(),
            };
        };
        // #785 — the assumes marker is an approval-card carrier: it annotates
        // what the draft is presuming and must never reach the recipient. The
        // Gmail rail scrubs it when the email channel builds the Gmail draft;
        // every other platform sends `actions.draftBody` verbatim, so each of
        // those send paths scrubs here. No marker ⇒ unchanged bytes.
        let owned = augmentagent_approval_discord::strip_assumes_for_send(body);
        let body = owned.as_str();

        // Two LinkedIn dispatch shapes share this handler, distinguished by
        // the email row's `kind`:
        //  - `post_engagement` (#13): message_id IS the post urn; the
        //    approved draft is a supportive comment → `post_comment`.
        //  - `connection_request` (#549): thread_id is the invitation urn →
        //    `act_on_invitation`. Handled first, because it is the one kind
        //    whose thread_id is NOT a conversation.
        //  - everything else (DM reply): thread_id is the conversationUrn →
        //    `send_message`. Anything new whose thread_id is not a
        //    conversation MUST get its own branch above rather than falling
        //    here — that is exactly how #549 happened.
        if action.email.kind == "connection_request" {
            let Some(invitation_urn) = action.email.thread_id.as_deref() else {
                return ApprovalActionOutcome::Failed {
                    message: "no invitation urn on email; cannot accept".into(),
                };
            };
            return match linkedin.act_on_invitation(invitation_urn, true).await {
                Ok(()) => {
                    // The draft body is deliberately NOT sent. An invitation
                    // urn is not a conversation urn — there is no thread to
                    // send into until LinkedIn mints one on accept. A welcome
                    // note is a follow-up action, not part of accepting.
                    let _ = self.store.update_action_status(
                        action_id,
                        ActionStatus::Sent,
                        Some(body),
                        None,
                    );
                    let _ = self
                        .store
                        .mark_email_processed(&action.email.message_id, TriageResult::Reply);
                    let _ = self.store.log_linkedin_action(
                        &uuid::Uuid::new_v4().to_string(),
                        "connection_request",
                        Some(invitation_urn),
                        "ok",
                        chrono::Utc::now().timestamp_millis(),
                        None,
                    );
                    tracing::info!(action_id, "linkedin invitation accepted via approval handler");
                    ApprovalActionOutcome::Approved
                }
                Err(e) => {
                    let msg = format!("linkedin act_on_invitation: {e}");
                    let _ = self.store.update_action_status(
                        action_id,
                        ActionStatus::Error,
                        None,
                        Some(&msg),
                    );
                    ApprovalActionOutcome::Failed { message: msg }
                }
            };
        }
        if action.email.kind == "post_engagement" {
            let post_urn = action.email.message_id.as_str();
            match linkedin.post_comment(post_urn, body).await {
                Ok(_) => {
                    let _ = self.store.update_action_status(
                        action_id,
                        ActionStatus::Sent,
                        Some(body),
                        None,
                    );
                    let _ = self
                        .store
                        .mark_email_processed(post_urn, TriageResult::Reply);
                    // Durable cap accounting (#13): record the successful
                    // engagement so the feed trigger's daily cap survives
                    // restarts and we never double-comment the same post.
                    let _ = self.store.log_linkedin_action(
                        &uuid::Uuid::new_v4().to_string(),
                        "post_engagement",
                        Some(post_urn),
                        "ok",
                        chrono::Utc::now().timestamp_millis(),
                        None,
                    );
                    tracing::info!(action_id, "linkedin comment posted via approval handler");
                    ApprovalActionOutcome::Approved
                }
                Err(e) => {
                    let msg = format!("linkedin post_comment: {e}");
                    let _ = self.store.update_action_status(
                        action_id,
                        ActionStatus::Error,
                        None,
                        Some(&msg),
                    );
                    ApprovalActionOutcome::Failed { message: msg }
                }
            }
        } else {
            let Some(conv_urn) = action.email.thread_id.as_deref() else {
                return ApprovalActionOutcome::Failed {
                    message: "no conversationUrn on email; cannot send".into(),
                };
            };
            match linkedin.send_message(conv_urn, body).await {
                Ok(_) => {
                    let _ = self.store.update_action_status(
                        action_id,
                        ActionStatus::Sent,
                        Some(body),
                        None,
                    );
                    let _ = self
                        .store
                        .mark_email_processed(&action.email.message_id, TriageResult::Reply);
                    tracing::info!(action_id, "linkedin reply sent via approval handler");
                    ApprovalActionOutcome::Approved
                }
                Err(e) => {
                    let msg = format!("linkedin send_message: {e}");
                    let _ = self.store.update_action_status(
                        action_id,
                        ActionStatus::Error,
                        None,
                        Some(&msg),
                    );
                    ApprovalActionOutcome::Failed { message: msg }
                }
            }
        }
    }

    async fn revise_linkedin(
        &self,
        action_id: &str,
        feedback: &str,
        action: augmentagent_store::ActionWithEmail,
    ) -> ApprovalActionOutcome {
        // LinkedIn has no server-side draft to swap — we just regenerate
        // text, update the action row, and re-post the card.
        let previous_draft = action.action.draft_body.clone().unwrap_or_default();
        let opts = draft_opts(self.draft_skill.clone(), self.wiki_root.clone());
        let prompt = augmentagent_channel_core::prompt::redraft_message(
            &action.email,
            &previous_draft,
            feedback,
        );
        let redraft = match self.reasoner.call(&opts, &prompt).await {
            Ok(s) => strip_approval_envelope_markers(s.trim()),
            Err(e) => {
                return ApprovalActionOutcome::Failed {
                    message: format!("redraft call failed: {e}"),
                };
            }
        };
        let _ = self.store.update_action_status(
            action_id,
            ActionStatus::Pending,
            Some(&redraft),
            None,
        );
        let _ = self.store.reset_nudge_schedule(action_id);
        tracing::info!(action_id, "linkedin revise: new draft persisted");
        ApprovalActionOutcome::Revised {
            email: action.email,
            draft: redraft,
        }
    }

    fn skip_linkedin(
        &self,
        action_id: &str,
        action: augmentagent_store::ActionWithEmail,
    ) -> ApprovalActionOutcome {
        // Nothing to delete server-side — LinkedIn has no draft concept.
        let _ = self.store.update_action_status(
            action_id,
            ActionStatus::Rejected,
            None,
            Some("skipped by approver"),
        );
        let _ = self
            .store
            .mark_email_processed(&action.email.message_id, TriageResult::Reply);
        ApprovalActionOutcome::Skipped
    }

    async fn approve_discord(
        &self,
        action_id: &str,
        action: augmentagent_store::ActionWithEmail,
    ) -> ApprovalActionOutcome {
        let Some(discord) = self.discord.as_ref() else {
            return ApprovalActionOutcome::Failed {
                message: "Discord is not configured; run `augmentagent discord login`".into(),
            };
        };
        let Some(channel_id) = action.email.thread_id.as_deref() else {
            return ApprovalActionOutcome::Failed {
                message: "no channel id on email; cannot send".into(),
            };
        };
        let Some(body) = action.action.draft_body.as_deref() else {
            return ApprovalActionOutcome::Failed {
                message: "no draft body on action; cannot send".into(),
            };
        };
        // #785 — scrub the card-only assumes marker before it ships.
        let owned = augmentagent_approval_discord::strip_assumes_for_send(body);
        let body = owned.as_str();
        match discord.send_message(channel_id, body).await {
            Ok(_) => {
                let _ = self.store.update_action_status(
                    action_id,
                    ActionStatus::Sent,
                    Some(body),
                    None,
                );
                let _ = self
                    .store
                    .mark_email_processed(&action.email.message_id, TriageResult::Reply);
                tracing::info!(action_id, "discord reply sent via approval handler");
                ApprovalActionOutcome::Approved
            }
            Err(e) => {
                let msg = format!("discord send_message: {e}");
                let _ = self.store.update_action_status(
                    action_id,
                    ActionStatus::Error,
                    None,
                    Some(&msg),
                );
                ApprovalActionOutcome::Failed { message: msg }
            }
        }
    }

    async fn revise_discord(
        &self,
        action_id: &str,
        feedback: &str,
        action: augmentagent_store::ActionWithEmail,
    ) -> ApprovalActionOutcome {
        let previous_draft = action.action.draft_body.clone().unwrap_or_default();
        let opts = draft_opts(self.draft_skill.clone(), self.wiki_root.clone());
        let prompt = augmentagent_channel_core::prompt::redraft_message(
            &action.email,
            &previous_draft,
            feedback,
        );
        let redraft = match self.reasoner.call(&opts, &prompt).await {
            Ok(s) => s.trim().to_string(),
            Err(e) => {
                return ApprovalActionOutcome::Failed {
                    message: format!("redraft call failed: {e}"),
                };
            }
        };
        let _ = self.store.update_action_status(
            action_id,
            ActionStatus::Pending,
            Some(&redraft),
            None,
        );
        let _ = self.store.reset_nudge_schedule(action_id);
        tracing::info!(action_id, "discord revise: new draft persisted");
        ApprovalActionOutcome::Revised {
            email: action.email,
            draft: redraft,
        }
    }

    fn skip_discord(
        &self,
        action_id: &str,
        action: augmentagent_store::ActionWithEmail,
    ) -> ApprovalActionOutcome {
        let _ = self.store.update_action_status(
            action_id,
            ActionStatus::Rejected,
            None,
            Some("skipped by approver"),
        );
        let _ = self
            .store
            .mark_email_processed(&action.email.message_id, TriageResult::Reply);
        ApprovalActionOutcome::Skipped
    }

    /// Approve→send for SocialAPI.ai (#244). Routes by the action's `kind`:
    ///   - `own_post_comment` → `reply_comment(post_id, …)` where the parent
    ///     post id rides on `email.thread_id` (see own_posts.rs `into_email`).
    ///   - `dm` → `send_dm(conversation_id, …)` where the conversation id
    ///     rides on `email.thread_id` (see inbound.rs `into_email`).
    /// On success marks the action `sent` (recording the returned external id
    /// in the audit cause) and marks the source message processed; failures
    /// surface as `Failed` and stamp the action `error`, mirroring the other
    /// channels. The human approval gate has already passed by the time this
    /// runs.
    async fn approve_socialapi(
        &self,
        action_id: &str,
        action: augmentagent_store::ActionWithEmail,
    ) -> ApprovalActionOutcome {
        let Some(client) = self.socialapi.as_ref() else {
            return ApprovalActionOutcome::Failed {
                message: "SocialAPI.ai is not configured (no SOCIALAPI_API_KEY); \
                          run `setup oauth --provider socialapi`"
                    .into(),
            };
        };
        let Some(body) = action.action.draft_body.as_deref() else {
            return ApprovalActionOutcome::Failed {
                message: "no draft body on action; cannot send".into(),
            };
        };
        // #785 — scrub the card-only assumes marker before it ships.
        let owned = augmentagent_approval_discord::strip_assumes_for_send(body);
        let body = owned.as_str();
        // Both comment-reply and DM-reply carry their send target on
        // `email.thread_id` (parent post id / conversation id respectively).
        let Some(target) = action.email.thread_id.as_deref() else {
            return ApprovalActionOutcome::Failed {
                message: "no thread id on email; cannot send socialapi reply".into(),
            };
        };
        let kind = action.email.kind.as_str();
        let send_result: anyhow::Result<String> = match kind {
            k if k == augmentagent_channel_core::trigger::kind::OWN_POST_COMMENT => {
                // The owning account rides on the email row (#543); legacy
                // rows carry the bare platform string there instead.
                let account_id = action
                    .email
                    .account_entity_id
                    .clone()
                    .filter(|a| !a.is_empty() && a != augmentagent_channel_socialapi::PLATFORM);
                let req = augmentagent_channel_socialapi::CommentReplyRequest {
                    text: body.to_string(),
                    // Thread the reply under the comment being answered — the
                    // email's message_id IS the platform comment id.
                    comment_id: Some(action.email.message_id.clone()),
                    private: None,
                    account_id,
                };
                client.reply_comment(target, &req).await
            }
            k if k == augmentagent_channel_core::trigger::kind::DM => {
                // The sending account rides on the email row, set by the DM
                // source from the conversation. Comment rows stamp the bare
                // platform string there instead, hence the filter.
                let account_id = action
                    .email
                    .account_entity_id
                    .clone()
                    .filter(|a| !a.is_empty() && a != augmentagent_channel_socialapi::PLATFORM);
                let req = augmentagent_channel_socialapi::DmSendRequest {
                    text: body.to_string(),
                    account_id,
                    attachment_url: None,
                };
                client.send_dm(target, &req).await
            }
            other => {
                return ApprovalActionOutcome::Failed {
                    message: format!("unsupported socialapi action kind `{other}`"),
                };
            }
        };

        match send_result {
            Ok(external_id) => {
                let _ = self.store.update_action_status(
                    action_id,
                    ActionStatus::Sent,
                    Some(body),
                    Some(&format!("socialapi {kind} -> {external_id}")),
                );
                let _ = self
                    .store
                    .mark_email_processed(&action.email.message_id, TriageResult::Reply);
                tracing::info!(
                    action_id,
                    kind,
                    external_id = %external_id,
                    "socialapi reply sent via approval handler"
                );
                ApprovalActionOutcome::Approved
            }
            Err(e) => {
                let msg = format!("socialapi {kind} send: {e}");
                let _ = self.store.update_action_status(
                    action_id,
                    ActionStatus::Error,
                    None,
                    Some(&msg),
                );
                ApprovalActionOutcome::Failed { message: msg }
            }
        }
    }

    fn skip_socialapi(
        &self,
        action_id: &str,
        action: augmentagent_store::ActionWithEmail,
    ) -> ApprovalActionOutcome {
        // Nothing to delete server-side — SocialAPI.ai has no draft concept.
        let _ = self.store.update_action_status(
            action_id,
            ActionStatus::Rejected,
            None,
            Some("skipped by approver"),
        );
        let _ = self
            .store
            .mark_email_processed(&action.email.message_id, TriageResult::Reply);
        ApprovalActionOutcome::Skipped
    }

    async fn revise_socialapi(
        &self,
        action_id: &str,
        feedback: &str,
        action: augmentagent_store::ActionWithEmail,
    ) -> ApprovalActionOutcome {
        // SocialAPI.ai has no server-side draft — we just regenerate the reply
        // text, update the action row in place, and let the broker re-post the
        // card. Mirrors `revise_discord` / `revise_linkedin`.
        let previous_draft = action.action.draft_body.clone().unwrap_or_default();
        let opts = draft_opts(self.draft_skill.clone(), self.wiki_root.clone());
        let prompt = augmentagent_channel_core::prompt::redraft_message(
            &action.email,
            &previous_draft,
            feedback,
        );
        let redraft = match self.reasoner.call(&opts, &prompt).await {
            Ok(s) => s.trim().to_string(),
            Err(e) => {
                return ApprovalActionOutcome::Failed {
                    message: format!("redraft call failed: {e}"),
                };
            }
        };
        let _ = self.store.update_action_status(
            action_id,
            ActionStatus::Pending,
            Some(&redraft),
            None,
        );
        let _ = self.store.reset_nudge_schedule(action_id);
        tracing::info!(action_id, "socialapi revise: new draft persisted");
        ApprovalActionOutcome::Revised {
            email: action.email,
            draft: redraft,
        }
    }

    async fn approve_telegram(
        &self,
        action_id: &str,
        action: augmentagent_store::ActionWithEmail,
    ) -> ApprovalActionOutcome {
        let Some(client) = self.resolve_telegram_client(&action.email) else {
            return ApprovalActionOutcome::Failed {
                message: "Telegram bot not available; reconnect via `augmentagent telegram-bot login`".into(),
            };
        };
        // `thread_id` carries the chat_id (set by `message_to_email`).
        let Some(chat_id) = action
            .email
            .thread_id
            .as_deref()
            .and_then(|s| s.parse::<i64>().ok())
        else {
            return ApprovalActionOutcome::Failed {
                message: "no chat_id on email; cannot send".into(),
            };
        };
        let Some(body) = action.action.draft_body.as_deref() else {
            return ApprovalActionOutcome::Failed {
                message: "no draft body on action; cannot send".into(),
            };
        };
        // #785 — scrub the card-only assumes marker before it ships.
        let owned = augmentagent_approval_discord::strip_assumes_for_send(body);
        let body = owned.as_str();
        // `message_id` shape is "tg:<chat>:<msg_id>" — use it as the
        // reply_to target so the bot's response is threaded under the
        // original message in Telegram's UI.
        let reply_to: Option<i64> = action
            .email
            .message_id
            .strip_prefix("tg:")
            .and_then(|s| s.rsplit_once(':'))
            .and_then(|(_chat, mid)| mid.parse::<i64>().ok());
        match client.send_message(chat_id, body, reply_to).await {
            Ok(sent) => {
                let _ = self.store.update_action_status(
                    action_id,
                    ActionStatus::Sent,
                    Some(body),
                    None,
                );
                let _ = self
                    .store
                    .mark_email_processed(&action.email.message_id, TriageResult::Reply);
                tracing::info!(
                    action_id,
                    sent_message_id = sent.message_id,
                    "telegram reply sent via approval handler"
                );
                ApprovalActionOutcome::Approved
            }
            Err(e) => {
                let msg = format!("telegram send_message: {e}");
                let _ = self.store.update_action_status(
                    action_id,
                    ActionStatus::Error,
                    None,
                    Some(&msg),
                );
                ApprovalActionOutcome::Failed { message: msg }
            }
        }
    }

    async fn revise_telegram(
        &self,
        action_id: &str,
        feedback: &str,
        action: augmentagent_store::ActionWithEmail,
    ) -> ApprovalActionOutcome {
        // Telegram has no server-side draft — just regenerate locally and
        // bounce the action row back to Pending so the broker re-renders.
        let previous_draft = action.action.draft_body.clone().unwrap_or_default();
        let opts = draft_opts(self.draft_skill.clone(), self.wiki_root.clone());
        let prompt = augmentagent_channel_core::prompt::redraft_message(
            &action.email,
            &previous_draft,
            feedback,
        );
        let redraft = match self.reasoner.call(&opts, &prompt).await {
            Ok(s) => s.trim().to_string(),
            Err(e) => {
                return ApprovalActionOutcome::Failed {
                    message: format!("redraft call failed: {e}"),
                };
            }
        };
        let _ = self.store.update_action_status(
            action_id,
            ActionStatus::Pending,
            Some(&redraft),
            None,
        );
        tracing::info!(action_id, "telegram revise: new draft persisted");
        ApprovalActionOutcome::Revised {
            email: action.email,
            draft: redraft,
        }
    }

    fn skip_telegram(
        &self,
        action_id: &str,
        action: augmentagent_store::ActionWithEmail,
    ) -> ApprovalActionOutcome {
        let _ = self.store.update_action_status(
            action_id,
            ActionStatus::Rejected,
            None,
            Some("skipped by approver"),
        );
        let _ = self
            .store
            .mark_email_processed(&action.email.message_id, TriageResult::Reply);
        ApprovalActionOutcome::Skipped
    }

    /// Resolve the right `TelegramBotClient` for this action.
    /// 1. Parse `bot_id` out of `email.account_entity_id`
    ///    (`telegram:bot:<bot_id>`).
    /// 2. If only one bot is loaded, use it (back-compat for rows lacking
    ///    a `bot:` tag).
    fn resolve_telegram_client(
        &self,
        email: &augmentagent_store::Email,
    ) -> Option<Arc<augmentagent_channel_telegram_bot::TelegramBotClient>> {
        let bot_id = email
            .account_entity_id
            .as_deref()
            .and_then(augmentagent_channel_telegram_bot::extract_bot_id);
        if let Some(bid) = bot_id {
            if let Some(c) = self.telegram.get(&bid) {
                return Some(Arc::clone(c));
            }
            return None;
        }
        if self.telegram.len() == 1 {
            return self.telegram.values().next().cloned();
        }
        None
    }

    /// Resolve the right SlackClient for this action. Priority:
    /// 1. Parse `team_id` out of `email.account_entity_id` ("slack:team:TXX").
    /// 2. If only one workspace is loaded, use it (back-compat for legacy rows).
    fn resolve_slack_client(
        &self,
        email: &augmentagent_store::Email,
    ) -> Option<Arc<augmentagent_channel_slack::SlackClient>> {
        let team_id = email
            .account_entity_id
            .as_deref()
            .and_then(|s| s.strip_prefix("slack:team:"))
            .map(str::to_string);
        if let Some(tid) = team_id {
            if let Some(c) = self.slack.get(&tid) {
                return Some(Arc::clone(c));
            }
            return None;
        }
        if self.slack.len() == 1 {
            return self.slack.values().next().cloned();
        }
        None
    }

    async fn approve_slack(
        &self,
        action_id: &str,
        action: augmentagent_store::ActionWithEmail,
    ) -> ApprovalActionOutcome {
        let Some(slack) = self.resolve_slack_client(&action.email) else {
            return ApprovalActionOutcome::Failed {
                message: "Slack workspace not available; reconnect in dashboard or `augmentagent slack login`".into(),
            };
        };
        let Some(channel_id) = action.email.thread_id.as_deref() else {
            return ApprovalActionOutcome::Failed {
                message: "no channel id on email; cannot send".into(),
            };
        };
        let Some(body) = action.action.draft_body.as_deref() else {
            return ApprovalActionOutcome::Failed {
                message: "no draft body on action; cannot send".into(),
            };
        };
        // #785 — scrub the card-only assumes marker before it ships.
        let owned = augmentagent_approval_discord::strip_assumes_for_send(body);
        let body = owned.as_str();
        match slack.send_message(channel_id, body).await {
            Ok(ts) => {
                let _ = self.store.update_action_status(
                    action_id,
                    ActionStatus::Sent,
                    Some(body),
                    None,
                );
                let _ = self
                    .store
                    .mark_email_processed(&action.email.message_id, TriageResult::Reply);
                tracing::info!(action_id, ts, "slack reply sent via approval handler");
                ApprovalActionOutcome::Approved
            }
            Err(e) => {
                let msg = format!("slack send_message: {e}");
                let _ = self.store.update_action_status(
                    action_id,
                    ActionStatus::Error,
                    None,
                    Some(&msg),
                );
                ApprovalActionOutcome::Failed { message: msg }
            }
        }
    }

    async fn revise_slack(
        &self,
        action_id: &str,
        feedback: &str,
        action: augmentagent_store::ActionWithEmail,
    ) -> ApprovalActionOutcome {
        let previous_draft = action.action.draft_body.clone().unwrap_or_default();
        let opts = draft_opts(self.draft_skill.clone(), self.wiki_root.clone());
        let prompt = augmentagent_channel_core::prompt::redraft_message(
            &action.email,
            &previous_draft,
            feedback,
        );
        let redraft = match self.reasoner.call(&opts, &prompt).await {
            Ok(s) => s.trim().to_string(),
            Err(e) => {
                return ApprovalActionOutcome::Failed {
                    message: format!("redraft call failed: {e}"),
                };
            }
        };
        let _ = self.store.update_action_status(
            action_id,
            ActionStatus::Pending,
            Some(&redraft),
            None,
        );
        tracing::info!(action_id, "slack revise: new draft persisted");
        ApprovalActionOutcome::Revised {
            email: action.email,
            draft: redraft,
        }
    }

    fn skip_slack(
        &self,
        action_id: &str,
        action: augmentagent_store::ActionWithEmail,
    ) -> ApprovalActionOutcome {
        let _ = self.store.update_action_status(
            action_id,
            ActionStatus::Rejected,
            None,
            Some("skipped by approver"),
        );
        let _ = self
            .store
            .mark_email_processed(&action.email.message_id, TriageResult::Reply);
        ApprovalActionOutcome::Skipped
    }

    async fn approve_github(
        &self,
        action_id: &str,
        action: augmentagent_store::ActionWithEmail,
    ) -> ApprovalActionOutcome {
        use augmentagent_channel_github::api::GithubApi;
        let Some(github) = self.github.as_ref() else {
            return ApprovalActionOutcome::Failed {
                message: "GitHub PAT not loaded; run `augmentagent github login`".into(),
            };
        };
        let Some(locator) = augmentagent_channel_github::outbound_target(&action.email) else {
            return ApprovalActionOutcome::Failed {
                message: "no <owner>/<repo>#<n> on email; cannot post comment".into(),
            };
        };
        let Some(body) = action.action.draft_body.as_deref() else {
            return ApprovalActionOutcome::Failed {
                message: "no draft body on action; cannot send".into(),
            };
        };
        // #785 — scrub the card-only assumes marker before it ships.
        let owned = augmentagent_approval_discord::strip_assumes_for_send(body);
        let body = owned.as_str();
        match github
            .post_issue_comment(&locator.owner, &locator.repo, locator.number, body)
            .await
        {
            Ok(comment_id) => {
                let _ = self.store.update_action_status(
                    action_id,
                    ActionStatus::Sent,
                    Some(body),
                    None,
                );
                let _ = self
                    .store
                    .mark_email_processed(&action.email.message_id, TriageResult::Reply);
                tracing::info!(
                    action_id,
                    comment_id,
                    "github comment posted via approval handler"
                );
                ApprovalActionOutcome::Approved
            }
            Err(e) => {
                let msg = format!("github post_issue_comment: {e}");
                let _ = self.store.update_action_status(
                    action_id,
                    ActionStatus::Error,
                    None,
                    Some(&msg),
                );
                ApprovalActionOutcome::Failed { message: msg }
            }
        }
    }

    async fn revise_github(
        &self,
        action_id: &str,
        feedback: &str,
        action: augmentagent_store::ActionWithEmail,
    ) -> ApprovalActionOutcome {
        let previous_draft = action.action.draft_body.clone().unwrap_or_default();
        let opts = draft_opts(self.draft_skill.clone(), self.wiki_root.clone());
        let prompt = augmentagent_channel_core::prompt::redraft_message(
            &action.email,
            &previous_draft,
            feedback,
        );
        let redraft = match self.reasoner.call(&opts, &prompt).await {
            Ok(s) => s.trim().to_string(),
            Err(e) => {
                return ApprovalActionOutcome::Failed {
                    message: format!("redraft call failed: {e}"),
                };
            }
        };
        let _ = self.store.update_action_status(
            action_id,
            ActionStatus::Pending,
            Some(&redraft),
            None,
        );
        let _ = self.store.reset_nudge_schedule(action_id);
        tracing::info!(action_id, "github revise: new draft persisted");
        ApprovalActionOutcome::Revised {
            email: action.email,
            draft: redraft,
        }
    }

    fn skip_github(
        &self,
        action_id: &str,
        action: augmentagent_store::ActionWithEmail,
    ) -> ApprovalActionOutcome {
        // Best-effort: nothing to delete server-side. (Marking the
        // notification thread read happens at the channel layer on dispatch;
        // here we just close out the action row.)
        let _ = self.store.update_action_status(
            action_id,
            ActionStatus::Rejected,
            None,
            Some("skipped by approver"),
        );
        let _ = self
            .store
            .mark_email_processed(&action.email.message_id, TriageResult::Reply);
        ApprovalActionOutcome::Skipped
    }
}

#[async_trait]
impl ApprovalActionHandler for ReplyApprover {
    async fn approve(&self, action_id: &str) -> ApprovalActionOutcome {
        let outcome = self.run_approve(action_id).await;
        // Scheduled: Approve on a --send-at proposal armed the timer (#502)
        // — the card leaves the queue either way, so advance the carousel.
        if matches!(
            outcome,
            ApprovalActionOutcome::Approved | ApprovalActionOutcome::Scheduled { .. }
        ) {
            self.trigger_next_nudge().await;
        }
        outcome
    }

    async fn skip(&self, action_id: &str) -> ApprovalActionOutcome {
        let outcome = self.run_skip(action_id).await;
        if matches!(outcome, ApprovalActionOutcome::Skipped) {
            self.trigger_next_nudge().await;
        }
        outcome
    }

    async fn revise(&self, action_id: &str, feedback: &str) -> ApprovalActionOutcome {
        // Revise does NOT advance the queue — the card stays active until the
        // user finally approves or skips. The instant-new-draft response is
        // handled by the broker's event handler from the Revised outcome.
        self.run_revise(action_id, feedback).await
    }

    async fn is_resolved(&self, action_id: &str) -> bool {
        match self.handle_load(action_id) {
            Some(a) => a.action.status != "pending",
            None => false,
        }
    }

    async fn schedule(&self, action_id: &str, at_ms: i64) -> ApprovalActionOutcome {
        let outcome = self.run_schedule(action_id, at_ms).await;
        if matches!(outcome, ApprovalActionOutcome::Scheduled { .. }) {
            // The scheduled card left the queue — surface the next one, the
            // same instant-advance approve/skip get.
            self.trigger_next_nudge().await;
        }
        outcome
    }

    async fn send_now(&self, action_id: &str) -> ApprovalActionOutcome {
        // No nudge trigger: the row left the carousel at schedule time.
        self.run_send_now(action_id).await
    }

    async fn cancel_schedule(&self, action_id: &str) -> ApprovalActionOutcome {
        self.run_cancel_schedule(action_id).await
    }

    async fn back_to_queue(&self, action_id: &str) -> ApprovalActionOutcome {
        // No nudge trigger: the repost + record_nudge inside makes this row
        // the active card again.
        self.run_back_to_queue(action_id).await
    }

    async fn is_schedule_live(&self, action_id: &str) -> bool {
        match self.handle_load(action_id) {
            Some(a) => matches!(a.action.status.as_str(), "scheduled" | "sending"),
            None => false,
        }
    }
}

impl ReplyApprover {
    async fn run_approve(&self, action_id: &str) -> ApprovalActionOutcome {
        let Some(action) = self.handle_load(action_id) else {
            return ApprovalActionOutcome::NotFound;
        };
        if action.action.status != "pending" {
            return ApprovalActionOutcome::AlreadyResolved {
                status: action.action.status,
            };
        }
        // #927 — kind first: an identity-merge card carries platform "wiki"
        // and no draft, so every arm below would either miss it or fail it.
        if action.email.kind == IDENTITY_MERGE_KIND {
            let wiki = self.wiki_root.as_deref();
            return Self::approve_identity_merge(&self.store, wiki, action_id, action);
        }
        if action.email.platform == "discord" {
            return self.approve_discord(action_id, action).await;
        }
        if action.email.platform == "slack" {
            return self.approve_slack(action_id, action).await;
        }
        if action.email.platform == "telegram" {
            return self.approve_telegram(action_id, action).await;
        }
        if action.email.platform == "github" {
            return self.approve_github(action_id, action).await;
        }
        if action.email.platform == augmentagent_channel_socialapi::PLATFORM {
            return self.approve_socialapi(action_id, action).await;
        }
        if action.email.platform == "gcal" {
            return self.approve_gcal(action_id, action).await;
        }
        if is_linkedin_email(&action.email) {
            return self.approve_linkedin(action_id, action).await;
        }
        let Some(draft_id) = action.draft_id.as_deref() else {
            return ApprovalActionOutcome::Failed {
                message: "no draftId on action; cannot send".into(),
            };
        };
        let Some(entity_id) = action.email.account_entity_id.as_deref() else {
            return ApprovalActionOutcome::Failed {
                message: "no accountEntityId on email; cannot send".into(),
            };
        };

        // #502 — a card carrying a --send-at proposal: Approve ARMS the
        // schedule instead of sending now (run_schedule owns the CAS, the
        // notice, and the carousel advance). A proposal already past — or
        // inside the minimum lead — expired naturally: fall through to the
        // immediate send, which is what approving a stale proposal means.
        // Back-to-queue/unschedule NULL the proposal, so a reposted card's
        // Approve lands here with None and sends immediately (the #501
        // review's re-arm loop is impossible by construction).
        if let Ok(Some(at_ms)) = self.store.action_scheduled_at(action_id) {
            if send_at_proposal_is_live(at_ms, chrono::Utc::now().timestamp_millis()) {
                return self.run_schedule(action_id, at_ms).await;
            }
            tracing::info!(
                action_id,
                at_ms,
                "approve: --send-at proposal already due; sending immediately"
            );
        }

        // #500 — claim the row (`pending → sending`) before the multi-second
        // Composio round-trip. The status read at the top of this fn is not a
        // gate: a Schedule pick, a second Approve click, or the dashboard can
        // race the send window, and the old unconditional Sent/Error flips
        // would stomp whatever they wrote. Losing the claim means someone
        // else resolved the row first — report it, run no side effects.
        match self.store.claim_action_for_send(
            action_id,
            ActionStatus::Pending,
            "discord",
        ) {
            Ok(true) => {}
            Ok(false) => {
                let status = self
                    .handle_load(action_id)
                    .map(|a| a.action.status)
                    .unwrap_or_else(|| "resolved".into());
                return ApprovalActionOutcome::AlreadyResolved { status };
            }
            Err(e) => {
                return ApprovalActionOutcome::Failed {
                    message: format!("claim for send failed: {e}"),
                };
            }
        }

        self.send_claimed_gmail(action_id, &action, draft_id, entity_id, false)
            .await
    }

    /// #500/#501 — the post-claim Gmail send tail, shared by Approve
    /// (`pending → sending`) and Send Now (`scheduled → sending`). The
    /// caller owns the claim; this owns the bounded Composio round-trip and
    /// every bookkeeping step in the #449-mandated order (self-send record
    /// BEFORE the status flip). A timeout is an UNKNOWN outcome (the send
    /// may have landed) and is retry-EXEMPT for both callers. A plain send
    /// error follows `retry_exempt_on_error`: Approve passes `false` (stays
    /// eligible for the generic retry tick, exactly as before #500); Send
    /// Now passes `true` — schedule-born rows must NEVER enter
    /// `list_retryable_replies`, whose re-dispatch through `dispatch_reply`
    /// would repost a card for a send the owner already resolved (#501
    /// review).
    async fn send_claimed_gmail(
        &self,
        action_id: &str,
        action: &augmentagent_store::ActionWithEmail,
        draft_id: &str,
        entity_id: &str,
        retry_exempt_on_error: bool,
    ) -> ApprovalActionOutcome {
        // Hard wall-clock bound (#500): ComposioClient has no request
        // timeout, and a hung send that outlives the engine's stuck-claim
        // grace would be flipped to error under us — then complete anyway,
        // recording a delivered email as failed.
        let sent_id = match tokio::time::timeout(
            augmentagent_channel_email::SEND_DRAFT_TIMEOUT,
            self.gmail.send_draft(entity_id, draft_id),
        )
        .await
        {
            Ok(Ok(id)) => id,
            Ok(Err(e)) => {
                let msg = format!("send_draft: {e}");
                // Conditional (`WHERE status = 'sending'`); the retry-cap
                // stamp is the caller's policy — see the doc comment.
                let stamp = retry_exempt_on_error
                    .then_some(augmentagent_channel_email::RETRY_EXEMPT_RETRY_COUNT);
                let _ = self.store.finish_send_error(action_id, &msg, stamp, "discord");
                return ApprovalActionOutcome::Failed { message: msg };
            }
            Err(_elapsed) => {
                let msg = format!(
                    "send_draft timed out after {}s — the message may or may \
                     not have been delivered; check the thread in Gmail \
                     before resending",
                    augmentagent_channel_email::SEND_DRAFT_TIMEOUT.as_secs()
                );
                let _ = self.store.finish_send_error(
                    action_id,
                    &msg,
                    Some(augmentagent_channel_email::RETRY_EXEMPT_RETRY_COUNT),
                    "discord",
                );
                return ApprovalActionOutcome::Failed { message: msg };
            }
        };
        // #449 — the approve button is the daemon's primary send path. Record
        // the id BEFORE flipping status, so an observer tick racing this send
        // can never catch the message in SENT without knowing it was ours.
        record_self_send(
            &self.store,
            sent_id.as_deref(),
            action.email.thread_id.as_deref(),
            Some(entity_id),
            Some(action_id),
        );
        let _ = self.store.finish_send_sent(action_id, "discord");
        let _ = self
            .store
            .mark_email_processed(&action.email.message_id, TriageResult::Reply);
        // Tone-mirroring v1 (#73): the post-edit body the user actually
        // approved is gold for the voice profile. Best-effort — failures
        // here must NOT change the user-visible Approved outcome.
        match self.store.record_user_edit_as_tone_example(action_id) {
            Ok(Some(id)) => {
                tracing::debug!(action_id, tone_example_id = %id, "captured user_edit tone example")
            }
            Ok(None) => {}
            Err(e) => tracing::warn!(action_id, "record_user_edit_as_tone_example failed: {e}"),
        }
        tracing::info!(action_id, "reply sent via approval handler");
        ApprovalActionOutcome::Approved
    }

    /// #398 — Approve on a calendar-event card: parse the machine payload
    /// (EventDraft JSON) from the emails row and execute the create. The
    /// event exists — and invites go out — only after this succeeds.
    async fn approve_gcal(
        &self,
        action_id: &str,
        action: augmentagent_store::ActionWithEmail,
    ) -> ApprovalActionOutcome {
        use augmentagent_channel_calendar::{CalendarApi, CalendarError, EventDraft};

        let Some(entity_id) = action.email.account_entity_id.as_deref() else {
            return ApprovalActionOutcome::Failed {
                message: "no accountEntityId on gcal action; cannot create".into(),
            };
        };
        let draft: EventDraft = match serde_json::from_str(&action.email.body) {
            Ok(d) => d,
            Err(e) => {
                let msg = format!("gcal proposal payload parse failed: {e}");
                let _ = self.store.update_action_status(
                    action_id,
                    ActionStatus::Error,
                    None,
                    Some(&msg),
                );
                return ApprovalActionOutcome::Failed { message: msg };
            }
        };
        match self.calendar.create_event(entity_id, "primary", &draft).await {
            Ok(created) => {
                let link = created
                    .html_link
                    .clone()
                    .unwrap_or_else(|| "(no link returned)".into());
                let final_body = format!(
                    "{}\ncreated: {link}",
                    action.action.draft_body.clone().unwrap_or_default()
                );
                let _ = self.store.update_action_status(
                    action_id,
                    ActionStatus::Sent,
                    Some(&final_body),
                    None,
                );
                let _ = self
                    .store
                    .mark_email_processed(&action.email.message_id, TriageResult::Reply);
                tracing::info!(
                    action_id,
                    event_id = ?created.id,
                    link,
                    "gcal event created"
                );
                ApprovalActionOutcome::Approved
            }
            Err(CalendarError::Forbidden { message }) => {
                let msg = format!(
                    "calendar write scope missing — re-consent the Google connection \
                     with calendar.events, then re-propose the event: {message}"
                );
                let _ = self.store.update_action_status(
                    action_id,
                    ActionStatus::Error,
                    None,
                    Some(&msg),
                );
                ApprovalActionOutcome::Failed { message: msg }
            }
            Err(e) => {
                let msg = format!("create_event: {e}");
                let _ = self.store.update_action_status(
                    action_id,
                    ActionStatus::Error,
                    None,
                    Some(&msg),
                );
                ApprovalActionOutcome::Failed { message: msg }
            }
        }
    }

    /// #927 — Approve on an identity-merge card: claim the row, parse the
    /// payload off `originalBody` (never the rendered card), and run the same
    /// executor `person merge --apply` does, guards and all. Takes its two
    /// dependencies rather than `&self` so a click can be exercised in a test.
    fn approve_identity_merge(
        store: &Store,
        wiki_root: Option<&Path>,
        action_id: &str,
        action: augmentagent_store::ActionWithEmail,
    ) -> ApprovalActionOutcome {
        // CAS `pending → sending` before ANY write — the wiki's or this row's.
        // A double-click must not run the merge twice (the second would fail on
        // the deleted stub), and a racing Skip must not reject one that ran.
        match store.claim_action_for_send(action_id, ActionStatus::Pending, "discord") {
            Ok(true) => {}
            Ok(false) => {
                let status = store
                    .get_action_with_email(action_id)
                    .ok()
                    .flatten()
                    .map(|a| a.action.status)
                    .unwrap_or_else(|| "resolved".into());
                return ApprovalActionOutcome::AlreadyResolved { status };
            }
            Err(e) => {
                return ApprovalActionOutcome::Failed {
                    message: format!("identity merge: claim failed: {e}"),
                };
            }
        }
        let raw = action.action.original_body.clone().unwrap_or_default();
        let payload: IdentityMergePayload = match serde_json::from_str(&raw) {
            Ok(p) => p,
            Err(e) => {
                let msg = format!("identity-merge payload parse failed: {e}");
                let _ =
                    store.update_action_status(action_id, ActionStatus::Error, None, Some(&msg));
                return ApprovalActionOutcome::Failed { message: msg };
            }
        };
        match execute_person_merge(
            wiki_root,
            store,
            &payload.stub_slug,
            &payload.candidate_slug,
            true,
        ) {
            Ok(report) => {
                let filled = report.moved.join(", ");
                let summary = format!(
                    "merged {} into {} (filled: {}; phone rows repointed: {})",
                    report.from,
                    report.into,
                    if filled.is_empty() { "nothing" } else { &filled },
                    report.phone_rows_repointed,
                );
                let _ = store.update_action_status(action_id, ActionStatus::Sent, Some(&summary), None);
                let _ = store.mark_email_processed(&action.email.message_id, TriageResult::Reply);
                tracing::info!(action_id, summary, "identity merge applied via approval handler");
                ApprovalActionOutcome::Approved
            }
            Err(e) => {
                // Recoverable by hand: re-create the stub with `imessage sync`
                // and re-scan — `error` is deliberately not "settled".
                let msg = format!("identity merge: {e:#}");
                let _ =
                    store.update_action_status(action_id, ActionStatus::Error, None, Some(&msg));
                ApprovalActionOutcome::Failed { message: msg }
            }
        }
    }

    async fn run_skip(&self, action_id: &str) -> ApprovalActionOutcome {
        let Some(action) = self.handle_load(action_id) else {
            return ApprovalActionOutcome::NotFound;
        };
        if action.action.status != "pending" {
            return ApprovalActionOutcome::AlreadyResolved {
                status: action.action.status,
            };
        }
        // #927 — a merge card needs no arm of its own: no draft to delete and
        // platform `wiki`, so it matches none of the arms below and reaches the
        // #500 tail, whose CAS off `pending` writes the `rejected` that both
        // loses to a claimed Approve and cools the pair off for the scan.
        if action.email.platform == "discord" {
            return self.skip_discord(action_id, action);
        }
        if action.email.platform == "slack" {
            return self.skip_slack(action_id, action);
        }
        if action.email.platform == "telegram" {
            return self.skip_telegram(action_id, action);
        }
        if action.email.platform == "github" {
            return self.skip_github(action_id, action);
        }
        if action.email.platform == augmentagent_channel_socialapi::PLATFORM {
            return self.skip_socialapi(action_id, action);
        }
        if is_linkedin_email(&action.email) {
            return self.skip_linkedin(action_id, action);
        }
        // #500 — resolve via CAS BEFORE the slow Gmail round-trip. The old
        // order (delete draft, then unconditional Rejected flip) left a
        // multi-second window where an Approve could claim the row
        // (`pending → sending`) mid-delete and this unconditional write would
        // stomp the claim — recording a delivered email as rejected, with
        // its draft also deleted under the in-flight send. Losing the CAS
        // means another surface resolved the row first: run no side effects.
        match self.store.try_resolve_action(
            action_id,
            ActionStatus::Rejected,
            "discord",
            Some("skipped by approver"),
        ) {
            Ok(true) => {}
            Ok(false) => {
                let status = self
                    .handle_load(action_id)
                    .map(|a| a.action.status)
                    .unwrap_or_else(|| "resolved".into());
                return ApprovalActionOutcome::AlreadyResolved { status };
            }
            Err(e) => {
                return ApprovalActionOutcome::Failed {
                    message: format!("skip: resolve failed: {e}"),
                };
            }
        }
        // Best-effort cleanup of the unsent Gmail draft — the row is already
        // Rejected, so a failed delete just leaves an orphan draft.
        if let (Some(draft_id), Some(entity_id)) = (
            action.draft_id.as_deref(),
            action.email.account_entity_id.as_deref(),
        ) {
            if let Err(e) = self.gmail.delete_draft(entity_id, draft_id).await {
                tracing::warn!(action_id, draft_id, "skip: delete_draft failed: {e}");
            }
        }
        let _ = self
            .store
            .mark_email_processed(&action.email.message_id, TriageResult::Reply);
        ApprovalActionOutcome::Skipped
    }

    async fn run_revise(&self, action_id: &str, feedback: &str) -> ApprovalActionOutcome {
        let Some(action) = self.handle_load(action_id) else {
            return ApprovalActionOutcome::NotFound;
        };
        if action.action.status != "pending" {
            return ApprovalActionOutcome::AlreadyResolved {
                status: action.action.status,
            };
        }
        if action.email.kind == IDENTITY_MERGE_KIND {
            return ApprovalActionOutcome::Failed {
                message: "not applicable to identity merges".into(),
            };
        }
        if action.email.platform == "gcal" {
            // v1 (#398): no LLM re-draft for calendar proposals. The card
            // stays pending — the operator can Skip and re-ask query mode
            // with the changes.
            return ApprovalActionOutcome::Failed {
                message: "Revise isn't supported for calendar-event cards yet — \
                          Skip this card and ask again with the changes."
                    .into(),
            };
        }
        if action.email.platform == "discord" {
            return self.revise_discord(action_id, feedback, action).await;
        }
        if action.email.platform == "slack" {
            return self.revise_slack(action_id, feedback, action).await;
        }
        if action.email.platform == "telegram" {
            return self.revise_telegram(action_id, feedback, action).await;
        }
        if action.email.platform == "github" {
            return self.revise_github(action_id, feedback, action).await;
        }
        if action.email.platform == augmentagent_channel_socialapi::PLATFORM {
            return self.revise_socialapi(action_id, feedback, action).await;
        }
        if is_linkedin_email(&action.email) {
            return self.revise_linkedin(action_id, feedback, action).await;
        }
        let Some(entity_id) = action.email.account_entity_id.as_deref() else {
            return ApprovalActionOutcome::Failed {
                message: "no accountEntityId on email; cannot revise".into(),
            };
        };
        // Strip the #35 needs-input and #785 assumes markers so the redraft
        // model sees the clean reply text, not the card-only carriers. No
        // markers ⇒ the draft is returned unchanged (pre-#35 behavior).
        let previous_draft = augmentagent_approval_discord::split_needs_input(
            &action.action.draft_body.clone().unwrap_or_default(),
        )
        .0;
        let previous_draft = augmentagent_approval_discord::split_assumes(&previous_draft).0;

        // 1. Generate revised draft via reasoner.
        let opts = draft_opts(self.draft_skill.clone(), self.wiki_root.clone());
        let prompt =
            augmentagent_channel_core::prompt::redraft_message(&action.email, &previous_draft, feedback);
        let redraft = match self.reasoner.call(&opts, &prompt).await {
            Ok(s) => s.trim().to_string(),
            Err(e) => {
                return ApprovalActionOutcome::Failed {
                    message: format!("redraft call failed: {e}"),
                };
            }
        };
        // #652 — a requested subject change comes back as a leading
        // `Subject:` line. Split it off before anything persists or sends the
        // text, so the header never lands in the body (#650) and the next
        // round's previous_draft stays clean.
        let (new_subject, redraft) =
            augmentagent_channel_core::prompt::split_redraft_subject(&redraft);

        // 2. Create a fresh Gmail draft with the revised body.
        // #473 — recreate the draft with the envelope the card was composed
        // with, when one was recorded. Pre-#473 behavior (To = emails.from,
        // no cc/bcc) silently dropped an overridden To and every cc/bcc on
        // reply cards: the intro pattern ("moving you to BCC") lost its BCC
        // — and its actual recipient — the moment the user hit Revise.
        let envelope = self
            .store
            .get_action_envelope(action_id)
            .unwrap_or_else(|e| {
                tracing::warn!(action_id, "revise: envelope lookup failed: {e}");
                None
            });
        // The thread id is kept whichever subject wins — this is still the
        // same reply, just with the header the user asked for.
        let subject = revised_subject(
            new_subject.as_deref(),
            envelope.as_ref().and_then(|env| env.subject.as_deref()),
            &action.email.subject,
            action.email.thread_id.as_deref(),
        );
        // #650 — a leading `Subject:` line the split above did not claim (an
        // empty one, or one the model echoed after the split point) must not
        // ship as body text: drop it when it repeats the outgoing subject,
        // fail the revise when it names a different one. The card stays
        // pending and Revise can run again.
        let redraft = match body_without_leaked_subject(&redraft, &subject) {
            Ok((clean, dropped)) => {
                if dropped.is_some() {
                    tracing::warn!(action_id, "revise: dropped leading Subject: line from redraft");
                }
                clean
            }
            Err(message) => {
                tracing::warn!(action_id, "revise: rejected redraft: {message}");
                return ApprovalActionOutcome::Failed {
                    message: format!("redraft carries its own subject line: {message}"),
                };
            }
        };
        let to = envelope
            .as_ref()
            .and_then(|env| env.to.clone())
            .unwrap_or_else(|| action.email.from.clone());
        let split = |v: &Option<String>| -> Vec<String> {
            v.as_deref()
                .map(augmentagent_channel_email::gmail::split_recipients)
                .unwrap_or_default()
        };
        let cc = envelope.as_ref().map(|env| split(&env.cc)).unwrap_or_default();
        let bcc = envelope.as_ref().map(|env| split(&env.bcc)).unwrap_or_default();
        // The redraft runs the same system prompt as the first draft, so it
        // may carry its own #785 assumes fence. Gmail gets the scrubbed body;
        // the persisted one keeps the fence so the reposted card can render
        // the warning against the NEW draft's assumptions.
        let gmail_redraft = augmentagent_approval_discord::strip_assumes_for_send(&redraft);
        // A subject that no longer names the inbound one cannot ride the
        // thread (Gmail would send under the thread's subject, #651): it
        // starts a new thread instead, which is what the user asked for.
        let draft_thread_id = thread_for_revised_subject(
            &subject,
            &action.email.subject,
            action.email.thread_id.as_deref(),
        );
        if action.email.thread_id.is_some() && draft_thread_id.is_none() {
            tracing::info!(
                action_id,
                subject = %subject,
                "revise: new subject starts a new thread (Gmail keeps a thread's subject)"
            );
        }
        let new_draft_id = match self
            .gmail
            .create_draft_with_attachment(
                entity_id,
                &to,
                &subject,
                &gmail_redraft,
                draft_thread_id,
                None,
                &cc,
                &bcc,
            )
            .await
        {
            Ok(id) => id,
            Err(e) => {
                return ApprovalActionOutcome::Failed {
                    message: format!("create_draft: {e}"),
                };
            }
        };

        // 3. Update sqlite: new draft body + new draft id — but only while
        // the row is STILL pending (#500). The reasoner + Gmail round-trips
        // above take seconds; a Schedule pick / Approve / supersede landing
        // meanwhile must not be stomped back to pending by the old
        // unconditional write. Loser cleans up the draft it just created.
        // The OLD draft is deleted only AFTER this CAS wins (#501 review):
        // deleting it first left a SchedulePick that landed mid-redraft
        // armed against an already-deleted draft — the engine would then
        // fire into a retry-exempt error. The brief window where both
        // drafts exist in Gmail is harmless.
        match self.store.refresh_pending_draft(action_id, &redraft) {
            Ok(true) => {}
            Ok(false) | Err(_) => {
                if let Err(e) = self.gmail.delete_draft(entity_id, &new_draft_id).await {
                    tracing::warn!(
                        action_id,
                        new_draft_id,
                        "revise: cleanup of orphan redraft failed: {e}"
                    );
                }
                let status = self
                    .handle_load(action_id)
                    .map(|a| a.action.status)
                    .unwrap_or_else(|| "resolved".into());
                return ApprovalActionOutcome::AlreadyResolved { status };
            }
        }
        // The subject override is only meaningful next to the draft id it
        // belongs to (#652): it is written after the CAS above won AND the
        // draft id landed, so a failed id write can never leave the row
        // carrying a subject for a draft it does not point at.
        match self.store.set_action_draft_id(action_id, &new_draft_id) {
            Ok(()) => {
                if new_subject.is_some() {
                    if let Err(e) = self.store.set_action_subject(action_id, Some(&subject)) {
                        tracing::warn!(action_id, "revise: could not persist the new subject: {e}");
                    }
                }
            }
            Err(e) => tracing::warn!(
                action_id,
                new_draft_id,
                "revise: could not record the new draft id; subject override not persisted: {e}"
            ),
        }
        let _ = self.store.reset_nudge_schedule(action_id);

        // 4. Delete the now-stale old draft best-effort — the row already
        // points at the replacement, so nothing can send the old one.
        if let Some(old) = action.draft_id.as_deref() {
            if let Err(e) = self.gmail.delete_draft(entity_id, old).await {
                tracing::warn!(action_id, old_draft = old, "revise: delete old draft failed: {e}");
            }
        }

        if new_subject.is_some() {
            tracing::info!(action_id, subject = %subject, "revise: applied a new subject");
        }
        tracing::info!(action_id, new_draft_id, "revise: new draft posted");
        ApprovalActionOutcome::Revised {
            email: action.email,
            draft: redraft,
        }
    }

    /// #501 — arm a scheduled send from the card's Schedule control. All
    /// state changes ride the #500 CAS primitives; the notice post is
    /// best-effort on top.
    ///
    /// TODO(#501): publish a status_bus `StatusChanged` here (and on
    /// cancel/fire) for dashboard parity once the bus has production
    /// publishers — today NO surface publishes it (approve/skip included),
    /// so there is no existing pattern to extend.
    async fn run_schedule(&self, action_id: &str, at_ms: i64) -> ApprovalActionOutcome {
        // Central time guard: every entry point (select token, custom modal,
        // later --send-at) funnels through this ONE validation at the CAS
        // layer, so no parser can forget it.
        let now_ms = chrono::Utc::now().timestamp_millis();
        if let Err(message) =
            augmentagent_channel_core::timeparse::validate_send_at(at_ms, now_ms)
        {
            return ApprovalActionOutcome::Failed { message };
        }
        let Some(action) = self.handle_load(action_id) else {
            return ApprovalActionOutcome::NotFound;
        };
        // The scheduled pipeline is a deferred Gmail send_draft (#499), but
        // the Schedule select rides on EVERY card — gate non-email cards out
        // BEFORE arming, or the engine would claim the row at fire time and
        // flip it straight to a retry-exempt error. Same dispatch ladder as
        // run_approve.
        let non_gmail = matches!(
            action.email.platform.as_str(),
            "discord" | "slack" | "telegram" | "github" | "gcal"
        ) || action.email.platform == augmentagent_channel_socialapi::PLATFORM
            || is_linkedin_email(&action.email);
        if non_gmail {
            return ApprovalActionOutcome::Failed {
                message: "scheduling is only supported for email drafts".into(),
            };
        }
        if action.draft_id.is_none() {
            return ApprovalActionOutcome::Failed {
                message: "no draftId on action; cannot schedule".into(),
            };
        }
        // CAS `pending → scheduled`: a double-pick's loser (or a racing
        // Approve / supersede) reports the fresh status and runs no side
        // effects — never a second schedule.
        match self.store.schedule_action(action_id, at_ms, "discord") {
            Ok(true) => {}
            Ok(false) => {
                let status = self
                    .handle_load(action_id)
                    .map(|a| a.action.status)
                    .unwrap_or_else(|| "resolved".into());
                return ApprovalActionOutcome::AlreadyResolved { status };
            }
            Err(e) => {
                return ApprovalActionOutcome::Failed {
                    message: format!("schedule failed: {e}"),
                };
            }
        }
        let local = format_local_send_time(at_ms);
        // The notice's "To" line is the ACTUAL send target: the #473
        // envelope To when one was recorded (compose/intro cards with
        // overridden routing), else the card's From (#501 review).
        let to_display = self
            .store
            .get_action_envelope(action_id)
            .unwrap_or_else(|e| {
                tracing::warn!(action_id, "schedule: envelope lookup failed: {e}");
                None
            })
            .and_then(|env| env.to)
            .unwrap_or_else(|| action.email.from.clone());
        // Post the scheduled notice (Send Now | Back to queue | Cancel) and
        // persist its pointers so the engine can retire it at fire time.
        // Best-effort: the schedule is armed either way — the engine cleanup
        // and the verb-aware startup sweep tolerate a missing notice.
        match self.broker_handle() {
            Some(broker) => {
                match broker
                    .post_scheduled_notice(action_id, &action.email, &local, at_ms, &to_display)
                    .await
                {
                    Ok(Some((channel_id, message_id))) => {
                        if let Err(e) = self.store.set_action_notice(
                            action_id,
                            &channel_id.to_string(),
                            &message_id.to_string(),
                        ) {
                            tracing::warn!(
                                action_id,
                                "schedule: persist notice pointers failed: {e}"
                            );
                        }
                    }
                    Ok(None) => {}
                    Err(e) => {
                        tracing::warn!(action_id, "schedule: post notice failed: {e}")
                    }
                }
            }
            None => tracing::debug!(
                action_id,
                "schedule: no broker handle; skipping scheduled notice"
            ),
        }
        tracing::info!(action_id, at_ms, local = %local, "schedule armed via approval handler");
        ApprovalActionOutcome::Scheduled { at_ms, local }
    }

    /// #501 — Send Now from the scheduled notice: a direct
    /// `scheduled → sending` CAS into the shared send tail. NEVER routes
    /// through `pending` — that would re-enter the nudge queue and, after
    /// #502, re-arm the proposal.
    async fn run_send_now(&self, action_id: &str) -> ApprovalActionOutcome {
        match self.store.claim_action_for_send(
            action_id,
            ActionStatus::Scheduled,
            "discord",
        ) {
            Ok(true) => {}
            Ok(false) => {
                let status = self
                    .handle_load(action_id)
                    .map(|a| a.action.status)
                    .unwrap_or_else(|| "resolved".into());
                return ApprovalActionOutcome::AlreadyResolved { status };
            }
            Err(e) => {
                return ApprovalActionOutcome::Failed {
                    message: format!("claim for send failed: {e}"),
                };
            }
        }
        // Load the row FRESH, after the claim (engine pattern): an
        // update-draft repoint can land while the row sits scheduled, and a
        // pre-claim snapshot's draftId would point at a Gmail draft that no
        // longer exists.
        let Some(action) = self.handle_load(action_id) else {
            let _ = self.store.finish_send_error(
                action_id,
                "send now: action row disappeared after claim",
                Some(augmentagent_channel_email::RETRY_EXEMPT_RETRY_COUNT),
                "discord",
            );
            self.delete_stored_notice(action_id).await;
            return ApprovalActionOutcome::NotFound;
        };
        let (Some(draft_id), Some(entity_id)) = (
            action.draft_id.clone(),
            action.email.account_entity_id.clone(),
        ) else {
            // Deterministic dead end for a claimed schedule-born row — same
            // retry-exempt stamp as the engine's fire path: the generic
            // retry tick re-dispatches through dispatch_reply and must never
            // pick these up.
            let msg = "send now: action has no draftId/accountEntityId; cannot send";
            let _ = self.store.finish_send_error(
                action_id,
                msg,
                Some(augmentagent_channel_email::RETRY_EXEMPT_RETRY_COUNT),
                "discord",
            );
            self.delete_stored_notice(action_id).await;
            return ApprovalActionOutcome::Failed {
                message: msg.into(),
            };
        };
        let outcome = self
            .send_claimed_gmail(action_id, &action, &draft_id, &entity_id, true)
            .await;
        // Sent OR dead, the schedule is over — retire the notice either way
        // (#501 review): the ephemeral told the owner about a failure, and a
        // stale "Sends <t>" notice must not keep advertising a row that will
        // never fire. The event handler also deletes the interaction's own
        // message on success; both are best-effort, whichever runs second is
        // a no-op.
        self.delete_stored_notice(action_id).await;
        outcome
    }

    /// #501 — Cancel from the scheduled notice: `scheduled → rejected` with
    /// the unsent Gmail draft deleted, the run_skip convention for a draft
    /// the owner decided against.
    async fn run_cancel_schedule(&self, action_id: &str) -> ApprovalActionOutcome {
        let Some(action) = self.handle_load(action_id) else {
            return ApprovalActionOutcome::NotFound;
        };
        // Snapshot the notice pointers FIRST — the cancel CAS clears them in
        // the same UPDATE (so a failed delete can't be retried against a
        // resolved row forever).
        let notice = self.store.action_notice(action_id).ok().flatten();
        match self.store.cancel_scheduled_action(
            action_id,
            "schedule cancelled by approver",
            "discord",
        ) {
            Ok(true) => {}
            Ok(false) => {
                let status = self
                    .handle_load(action_id)
                    .map(|a| a.action.status)
                    .unwrap_or_else(|| "resolved".into());
                return ApprovalActionOutcome::AlreadyResolved { status };
            }
            Err(e) => {
                return ApprovalActionOutcome::Failed {
                    message: format!("cancel: resolve failed: {e}"),
                };
            }
        }
        // Best-effort cleanup of the unsent Gmail draft — the row is already
        // Rejected, so a failed delete just leaves an orphan draft. The
        // draft id is re-loaded AFTER the CAS (#501 review): a concurrent
        // repoint (`gmail update-draft` runs freely on scheduled rows — only
        // the 'sending' claim blocks it) would leave the pre-CAS snapshot
        // pointing at a draft that repoint already deleted, while the
        // REPLACEMENT survived in Gmail forever.
        let fresh_draft_id = self.handle_load(action_id).and_then(|a| a.draft_id);
        if let (Some(draft_id), Some(entity_id)) = (
            fresh_draft_id.as_deref(),
            action.email.account_entity_id.as_deref(),
        ) {
            if let Err(e) = self.gmail.delete_draft(entity_id, draft_id).await {
                tracing::warn!(action_id, draft_id, "cancel: delete_draft failed: {e}");
            }
        }
        self.delete_notice_pair(action_id, notice).await;
        let _ = self
            .store
            .mark_email_processed(&action.email.message_id, TriageResult::Reply);
        tracing::info!(action_id, "scheduled send cancelled via approval handler");
        ApprovalActionOutcome::CancelledSchedule
    }

    /// #501 — Back to queue, the non-destructive mis-click escape:
    /// `scheduled → pending` (proposal AND notice pointers cleared inside
    /// the CAS — a later Approve must send immediately, not re-arm) with
    /// the approval card reposted. Order matters (#501 review): the card is
    /// posted BEFORE the CAS, while the row is still 'scheduled' and
    /// therefore invisible to the NudgeScheduler's promoter — posting after
    /// the flip left a window where the 60s tick could post a SECOND card.
    /// The unschedule CAS itself seeds the row as the ACTIVE nudge
    /// (nudgeCount 1), so no record_nudge follows; a lost CAS rolls the
    /// pre-posted card back down via its returned message ids.
    async fn run_back_to_queue(&self, action_id: &str) -> ApprovalActionOutcome {
        // Snapshot the notice pointers FIRST — unschedule clears them in the
        // same UPDATE.
        let notice = self.store.action_notice(action_id).ok().flatten();
        // Repost while still 'scheduled'. The persisted redraft count rides
        // along so a refined-to-cap card doesn't get its quick-refine row
        // resurrected, and the envelope markers match what the original
        // card showed (#473) — same helper the Revise repost uses.
        let mut reposted: Option<(u64, u64)> = None;
        let mut repost_failed = false;
        if let Some(broker) = self.broker_handle() {
            if let Some(action) = self.handle_load(action_id) {
                let draft = augmentagent_approval_discord::append_envelope_markers(
                    action.action.draft_body.clone().unwrap_or_default(),
                    Some(self.store.as_ref()),
                    action_id,
                    &action.email.from,
                );
                let count =
                    self.store.redraft_count(action_id).unwrap_or(0).max(0) as u32;
                match broker
                    .post_approval_card(action_id, &action.email, &draft, count)
                    .await
                {
                    Ok(ids) => reposted = ids,
                    Err(e) => {
                        repost_failed = true;
                        tracing::warn!(
                            action_id,
                            "back to queue: card repost failed (proceeding; \
                             re-nudge below covers it): {e}"
                        );
                    }
                }
            }
        }
        let cas = self.store.unschedule_action(action_id, "discord");
        if !matches!(cas, Ok(true)) {
            // The row never became pending (Send Now / cancel / engine fire
            // / supersede won meanwhile) — take the pre-posted card back
            // down; leaving it would advertise an actionable card for a
            // resolved row until the next startup sweep.
            if let (Some((c, m)), Some(broker)) = (reposted, self.broker_handle()) {
                if let Err(e) = broker.delete_message(c, m).await {
                    tracing::warn!(
                        action_id,
                        "back to queue: rollback of pre-posted card failed: {e}"
                    );
                }
            }
            return match cas {
                Ok(_) => {
                    let status = self
                        .handle_load(action_id)
                        .map(|a| a.action.status)
                        .unwrap_or_else(|| "resolved".into());
                    ApprovalActionOutcome::AlreadyResolved { status }
                }
                Err(e) => ApprovalActionOutcome::Failed {
                    message: format!("back to queue: resolve failed: {e}"),
                },
            };
        }
        self.delete_notice_pair(action_id, notice).await;
        if repost_failed {
            // The CAS seeded the row ACTIVE with a full re-nudge interval,
            // but no card is actually visible. Pull the timer to now so the
            // NudgeScheduler's next tick re-posts the card instead of
            // leaving the owner cardless for six hours.
            let now_ms = chrono::Utc::now().timestamp_millis();
            let _ = self.store.record_nudge(action_id, now_ms);
        }
        tracing::info!(
            action_id,
            "schedule disarmed via approval handler; card reposted"
        );
        ApprovalActionOutcome::Unscheduled
    }
}

/// #501 — owner-local display string for a scheduled fire time, e.g.
/// "Mon Sep 1, 9:05 AM". `chrono::Local` follows the daemon host's timezone
/// — the same zone the schedule was resolved in.
fn format_local_send_time(at_ms: i64) -> String {
    use chrono::TimeZone;
    match chrono::Local.timestamp_millis_opt(at_ms) {
        chrono::LocalResult::Single(dt) | chrono::LocalResult::Ambiguous(dt, _) => {
            dt.format("%a %b %-d, %-I:%M %p").to_string()
        }
        // Out-of-range epoch — unreachable behind the 60-day guard, but a
        // display path must never panic.
        chrono::LocalResult::None => format!("epoch+{at_ms}ms"),
    }
}

async fn build_broker(
    cli: &Cli,
    store: Arc<Store>,
    dry_run: bool,
) -> Result<(Arc<dyn ApprovalBroker>, Option<Arc<ReplyApprover>>)> {
    if dry_run {
        return Ok((Arc::new(NoopBroker), None));
    }
    let token = match std::env::var("DISCORD_BOT_TOKEN") {
        Ok(t) => t,
        Err(_) => {
            warn!("DISCORD_BOT_TOKEN unset; approval broker disabled (replies will error)");
            return Ok((Arc::new(NoopBroker), None));
        }
    };
    let channel_id: u64 = std::env::var("DISCORD_CHANNEL_ID")
        .context("DISCORD_CHANNEL_ID env var required")?
        .parse()
        .context("DISCORD_CHANNEL_ID must be a numeric channel id")?;

    let query_channel_id: Option<u64> = Some(
        std::env::var("DISCORD_QUERY_CHANNEL_ID")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(channel_id),
    );
    let allowed_user_id: Option<u64> = std::env::var("DISCORD_ALLOWED_USER_ID")
        .ok()
        .and_then(|s| s.parse().ok());

    let reasoner = build_reasoner();

    let repo_root = std::env::current_dir().context("current_dir")?;
    let query_handler: Option<Arc<dyn QueryHandler>> = cli.wiki_dir.as_ref().map(|root| {
        let q = WikiQuerier {
            reasoner: Arc::clone(&reasoner),
            wiki_root: root.clone(),
            repo_root: repo_root.clone(),
        };
        Arc::new(q) as Arc<dyn QueryHandler>
    });

    // Approval action handler: needs Composio for send/delete/create_draft,
    // reasoner for revise, and the skill body for the redraft prompt.
    let api_key =
        std::env::var("COMPOSIO_API_KEY").context("COMPOSIO_API_KEY env var required")?;
    let calendar = Arc::new(
        augmentagent_channel_calendar::ComposioCalendarClient::new(api_key.clone()),
    );
    let gmail = Arc::new(ComposioClient::new(api_key));
    let skill_dir = cli.skill_dir.clone();
    let draft_skill = std::fs::read_to_string(skill_dir.join("SKILL.md")).unwrap_or_default();
    // LinkedIn voyager client is optional. Present iff we can load auth; if
    // the file is missing or malformed the daemon stays up and just can't
    // send LinkedIn replies (Gmail-only mode).
    let linkedin = load_linkedin_client(&repo_root);

    let discord = load_discord_client();
    let slack = load_slack_clients(&store);
    let telegram = load_telegram_bot_clients(&store);
    let github = load_github_client();
    let socialapi = load_socialapi_client(&store);
    // Keep handles for the broker before `store` is moved into the approver:
    // the #37 Revise-triple capture.
    let store_for_broker = Arc::clone(&store);
    // #428 — `!journal` write-back bridge. Present iff SHADOWNOTE_* config
    // exists (keyring/env); without it the command replies with the
    // not-configured notice and the daemon is otherwise unaffected.
    let journal_ops: Option<Arc<dyn augmentagent_approval_discord::JournalOps>> =
        match augmentagent_channel_journal::JournalRuntime::from_env().await {
            Ok(Some(runtime)) => {
                let wiki_schema = cli
                    .wiki_dir
                    .as_ref()
                    .and_then(|_| std::fs::read_to_string("schema/wiki-skill.md").ok());
                Some(Arc::new(CliJournalOps {
                    runtime,
                    reasoner: Arc::clone(&reasoner),
                    wiki_root: cli.wiki_dir.clone(),
                    wiki_schema,
                }))
            }
            Ok(None) => {
                info!("!journal write-back disabled: SHADOWNOTE_* config not present");
                None
            }
            Err(e) => {
                warn!("!journal write-back disabled: {e:#}");
                None
            }
        };
    let approver = Arc::new(ReplyApprover {
        store,
        gmail,
        calendar,
        linkedin,
        discord,
        slack,
        telegram,
        github,
        socialapi,
        reasoner: Arc::clone(&reasoner),
        draft_skill,
        wiki_root: cli.wiki_dir.clone(),
        nudge: std::sync::OnceLock::new(),
        broker: std::sync::OnceLock::new(),
    });

    let approver_for_broker = Arc::clone(&approver);
    let loop_parser: Option<Arc<dyn augmentagent_approval_discord::LoopCommandParser>> = Some(
        Arc::new(LoopReasonerParser {
            reasoner: Arc::clone(&reasoner),
        }),
    );
    let broker = DiscordApprovalBroker::start(DiscordConfig {
        bot_token: token,
        channel_id,
        query_channel_id,
        allowed_user_id,
        query_handler,
        action_handler: Some(approver_for_broker),
        store: Some(store_for_broker),
        loop_parser,
        wiki_root: cli.wiki_dir.clone(),
        journal_ops,
    })
    .await
    .context("start discord broker")?;
    let broker: Arc<dyn ApprovalBroker> = Arc::new(broker);
    // #501 — hand the approver a Weak broker handle so the schedule verbs
    // can post/delete the scheduled notice and repost cards. Weak for the
    // same cycle-break reason as `nudge` (the broker's event handler holds
    // this approver strongly).
    approver.broker.set(Arc::downgrade(&broker)).ok();
    Ok((broker, Some(approver)))
}

fn build_channel(
    cli: &Cli,
    store: Arc<Store>,
    broker: Arc<dyn ApprovalBroker>,
    dry_run: bool,
    interval_secs: u64,
) -> Result<GmailChannel<ComposioClient, FallbackReasoner>> {
    let api_key = std::env::var("COMPOSIO_API_KEY").context("COMPOSIO_API_KEY env var required")?;
    let gmail = Arc::new(ComposioClient::new(api_key).with_rate_limit_store(Arc::clone(&store)));
    let reasoner = build_reasoner();

    // Resolve wiki enable/disable and schema path.
    let (wiki_root, wiki_schema_path) = match &cli.wiki_dir {
        Some(root) => {
            let schema = cli
                .wiki_schema
                .clone()
                .unwrap_or_else(|| PathBuf::from("schema/wiki-skill.md"));
            (Some(root.clone()), Some(schema))
        }
        None => (None, None),
    };
    if let Some(path) = &wiki_root {
        info!(wiki = %path.display(), "wiki integration enabled");
    }

    let config = GmailChannelConfig {
        skill_dir: cli.skill_dir.clone(),
        dry_run,
        model: cli.model.clone(),
        poll_interval: Duration::from_secs(interval_secs),
        wiki_root,
        wiki_schema_path,
        ..Default::default()
    };
    Ok(GmailChannel::new(store, gmail, reasoner, broker, config))
}

fn build_linkedin_channel(
    cli: &Cli,
    store: Arc<Store>,
    broker: Arc<dyn ApprovalBroker>,
    dry_run: bool,
) -> Result<LinkedInChannel<VoyagerClient, FallbackReasoner>> {
    let repo_root = std::env::current_dir().context("current_dir")?;
    let auth = LinkedInAuth::load_with_migration(&repo_root).with_context(|| {
        "load linkedin auth from keychain or legacy file — run `augmentagent linkedin login --cookies-json <file>`"
    })?;
    let member_urn = auth.member_urn.clone();
    let voyager = Arc::new(VoyagerClient::new(auth));
    let reasoner = build_reasoner();

    let (wiki_root, wiki_schema_path) = match &cli.wiki_dir {
        Some(root) => {
            let schema = cli
                .wiki_schema
                .clone()
                .unwrap_or_else(|| PathBuf::from("schema/wiki-skill.md"));
            (Some(root.clone()), Some(schema))
        }
        None => (None, None),
    };

    let poll_interval = match std::env::var("AUGMENTAGENT_LINKEDIN_POLL_SECS") {
        Ok(s) => s
            .parse::<u64>()
            .map(Duration::from_secs)
            .unwrap_or_else(|_| Duration::from_secs(DEFAULT_POLL_SECS)),
        Err(_) => Duration::from_secs(DEFAULT_POLL_SECS),
    };

    let config = LinkedInChannelConfig {
        poll_interval,
        dry_run,
        wiki_root,
        wiki_schema_path,
        skill_dir: cli.skill_dir.clone(),
    };
    info!(member = %member_urn, interval_secs = poll_interval.as_secs(), "linkedin channel ready");
    Ok(LinkedInChannel::new(
        store, voyager, reasoner, broker, member_urn, config,
    ))
}

/// Build the friend-post engagement runner (#13). Shares the DM channel's
/// auth gate (errors if LinkedIn isn't configured). Uses the
/// `skills/linkedin-triage` rubric, a 6h-with-jitter cadence
/// (`AUGMENTAGENT_LINKEDIN_FEED_POLL_SECS` override), and a default daily
/// engagement cap of 5 (`AUGMENTAGENT_LINKEDIN_MAX_ENGAGEMENTS` override).
fn build_linkedin_feed_engagement(
    cli: &Cli,
    store: Arc<Store>,
    broker: Arc<dyn ApprovalBroker>,
    dry_run: bool,
) -> Result<LinkedInFeedEngagement<VoyagerClient, FallbackReasoner>> {
    let repo_root = std::env::current_dir().context("current_dir")?;
    let auth = LinkedInAuth::load_with_migration(&repo_root)
        .context("load linkedin auth (feed engagement)")?;
    let member_urn = auth.member_urn.clone();
    let voyager = Arc::new(VoyagerClient::new(auth));
    let reasoner = build_reasoner();

    let (wiki_root, wiki_schema_path) = match &cli.wiki_dir {
        Some(root) => {
            let schema = cli
                .wiki_schema
                .clone()
                .unwrap_or_else(|| PathBuf::from("schema/wiki-skill.md"));
            (Some(root.clone()), Some(schema))
        }
        None => (None, None),
    };

    // Engagement-specific rubric sits alongside the email-triage skill dir.
    let triage_skill_dir = cli
        .skill_dir
        .parent()
        .map(|p| p.join("linkedin-triage"))
        .unwrap_or_else(|| PathBuf::from("skills/linkedin-triage"));

    let poll_secs = std::env::var("AUGMENTAGENT_LINKEDIN_FEED_POLL_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(DEFAULT_FEED_POLL_SECS);
    let max_per_day = std::env::var("AUGMENTAGENT_LINKEDIN_MAX_ENGAGEMENTS")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(DEFAULT_MAX_ENGAGEMENTS_PER_DAY);

    let trigger = Arc::new(augmentagent_channel_linkedin::LinkedInFeedTrigger::new(
        Arc::clone(&voyager),
        Arc::clone(&store),
        wiki_root.clone(),
        max_per_day,
    ));

    let config = LinkedInChannelConfig {
        poll_interval: Duration::from_secs(poll_secs),
        dry_run,
        wiki_root,
        wiki_schema_path,
        skill_dir: triage_skill_dir,
    };
    info!(
        member = %member_urn,
        interval_secs = poll_secs,
        max_per_day,
        "linkedin feed engagement ready"
    );
    Ok(LinkedInFeedEngagement {
        store,
        reasoner,
        approvals: broker,
        trigger,
        member_urn,
        config,
        poll_interval: Duration::from_secs(poll_secs),
    })
}

/// Shared auth + wiki/skill setup for the #58 LinkedIn engagement sub-feature
/// builders. Errors (→ sub-feature disabled with a warning) when LinkedIn
/// auth is absent — same gate as the DM channel.
fn linkedin_engagement_ctx(
    cli: &Cli,
) -> Result<(Arc<VoyagerClient>, String, LinkedInChannelConfig, bool)> {
    let repo_root = std::env::current_dir().context("current_dir")?;
    let auth = LinkedInAuth::load_with_migration(&repo_root)
        .context("load linkedin auth (engagement)")?;
    let member_urn = auth.member_urn.clone();
    let voyager = Arc::new(VoyagerClient::new(auth));
    let (wiki_root, wiki_schema_path) = match &cli.wiki_dir {
        Some(root) => {
            let schema = cli
                .wiki_schema
                .clone()
                .unwrap_or_else(|| PathBuf::from("schema/wiki-skill.md"));
            (Some(root.clone()), Some(schema))
        }
        None => (None, None),
    };
    let triage_skill_dir = cli
        .skill_dir
        .parent()
        .map(|p| p.join("linkedin-triage"))
        .unwrap_or_else(|| PathBuf::from("skills/linkedin-triage"));
    let config = LinkedInChannelConfig {
        poll_interval: Duration::from_secs(DEFAULT_POLL_SECS),
        dry_run: false, // overwritten by each caller
        wiki_root,
        wiki_schema_path,
        skill_dir: triage_skill_dir,
    };
    Ok((voyager, member_urn, config, true))
}

/// Construct the merged SqliteGovernor (#83) the engagement sub-features wrap
/// every outbound publish in. Same construction as the scheduled-post engine.
fn engagement_governor(
    store: Arc<Store>,
) -> Arc<dyn augmentagent_channel_core::RateGovernor> {
    Arc::new(augmentagent_channel_core::SqliteGovernor::with_system_clock(
        store,
    ))
}

/// #58.2 — own-post comment-reply engagement. Polls the user's registered own
/// posts (`augmentagent linkedin watch-post …` / dashboard) for new comments,
/// triages each, surfaces an approval-gated reply. RateGovernor `Comment`
/// envelope. Cadence `AUGMENTAGENT_LINKEDIN_OWNPOST_POLL_SECS`; reply pre-cap
/// `AUGMENTAGENT_LINKEDIN_MAX_OWNPOST_REPLIES`.
fn build_own_post_comment_engagement(
    cli: &Cli,
    store: Arc<Store>,
    broker: Arc<dyn ApprovalBroker>,
    dry_run: bool,
) -> Result<OwnPostCommentEngagement<VoyagerClient, FallbackReasoner>> {
    let (voyager, member_urn, mut config, _) = linkedin_engagement_ctx(cli)?;
    config.dry_run = dry_run;
    let reasoner = build_reasoner();
    let poll_secs = std::env::var("AUGMENTAGENT_LINKEDIN_OWNPOST_POLL_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(DEFAULT_OWN_POST_POLL_SECS);
    let max_per_day = std::env::var("AUGMENTAGENT_LINKEDIN_MAX_OWNPOST_REPLIES")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(DEFAULT_MAX_REPLIES_PER_DAY);
    let trigger = Arc::new(OwnPostsCommentTrigger::new(
        Arc::clone(&voyager),
        Arc::clone(&store),
        max_per_day,
    ));
    info!(
        member = %member_urn,
        interval_secs = poll_secs,
        max_per_day,
        "linkedin own-post comment engagement ready"
    );
    Ok(OwnPostCommentEngagement {
        store: Arc::clone(&store),
        reasoner,
        approvals: broker,
        governor: engagement_governor(store),
        trigger,
        member_urn,
        config,
        poll_interval: Duration::from_secs(poll_secs),
    })
}

/// Engagement-specific rubric for SocialAPI.ai comments and DMs, resolved
/// alongside the email-triage skill dir the same way LinkedIn does it.
///
/// #531: both socialapi builders passed `cli.skill_dir` straight through,
/// which defaults to `skills/email-triage`, so public comment replies and DMs
/// were triaged against the *email* rubric and `skills/socialapi-triage/`
/// was never read by anything.
fn socialapi_triage_skill_dir(cli: &Cli) -> PathBuf {
    cli.skill_dir
        .parent()
        .map(|p| p.join("socialapi-triage"))
        .unwrap_or_else(|| PathBuf::from("skills/socialapi-triage"))
}

/// #243 — SocialAPI.ai own-post comment engagement. Polls the user's
/// registered own posts (`own_posts` rows with platform `"socialapi"`) for new
/// comments via the SocialAPI.ai inbox, triages each, and surfaces an
/// approval-gated reply; the send happens when the operator approves (#244).
/// Gated on the SocialAPI.ai key (env, keyring, or dashboard sqlite config);
/// self-disables with a warning when absent. Cadence
/// `AUGMENTAGENT_SOCIALAPI_OWNPOST_POLL_SECS`; per-tick comment pre-cap
/// `AUGMENTAGENT_SOCIALAPI_MAX_OWNPOST_REPLIES`.
fn build_socialapi_own_post_engagement(
    cli: &Cli,
    store: Arc<Store>,
    broker: Arc<dyn ApprovalBroker>,
    dry_run: bool,
) -> Result<augmentagent_channel_socialapi::SocialApiOwnPostCommentEngagement<FallbackReasoner>>
{
    use augmentagent_channel_socialapi::{
        SocialApiAuth, SocialApiClient, SocialApiOwnPostCommentEngagement,
        SocialApiOwnPostCommentTrigger, SocialApiOwnPostConfig, DEFAULT_MAX_COMMENTS_PER_TICK,
        DEFAULT_OWN_POST_POLL_SECS,
    };

    let auth = SocialApiAuth::load_with_store(&store).context("load socialapi auth")?;
    let client = Arc::new(SocialApiClient::new(auth));
    let reasoner = build_reasoner();
    let poll_secs = std::env::var("AUGMENTAGENT_SOCIALAPI_OWNPOST_POLL_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(DEFAULT_OWN_POST_POLL_SECS);
    let max_per_tick = std::env::var("AUGMENTAGENT_SOCIALAPI_MAX_OWNPOST_REPLIES")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(DEFAULT_MAX_COMMENTS_PER_TICK);
    let trigger = Arc::new(SocialApiOwnPostCommentTrigger::new(
        Arc::clone(&client),
        Arc::clone(&store),
        max_per_tick,
    ));
    let config = SocialApiOwnPostConfig {
        dry_run,
        wiki_root: cli.wiki_dir.clone(),
        skill_dir: socialapi_triage_skill_dir(cli),
    };
    info!(
        interval_secs = poll_secs,
        max_per_tick, "socialapi own-post comment engagement ready"
    );
    Ok(SocialApiOwnPostCommentEngagement {
        store: Arc::clone(&store),
        reasoner,
        approvals: broker,
        governor: engagement_governor(store),
        trigger,
        config,
        poll_interval: Duration::from_secs(poll_secs),
    })
}

/// #242 — SocialAPI.ai inbound DM channel. Polls DM conversations across the
/// active SocialAPI.ai accounts, triages each genuinely new inbound message,
/// and surfaces an approval-gated reply (the actual send lands in #244). Gated
/// on `SOCIALAPI_API_KEY` / keyring; self-disables with a warning when absent.
/// Cadence `AUGMENTAGENT_SOCIALAPI_DM_POLL_SECS`; per-tick pre-cap
/// `AUGMENTAGENT_SOCIALAPI_DM_MAX_PER_TICK`.
#[allow(clippy::type_complexity)]
fn build_socialapi_dm_channel(
    cli: &Cli,
    store: Arc<Store>,
    broker: Arc<dyn ApprovalBroker>,
    dry_run: bool,
) -> Result<
    augmentagent_channel_core::trigger::ChannelRunner<
        augmentagent_channel_core::trigger::InboundMessageTrigger<
            augmentagent_channel_socialapi::SocialApiDmSource,
        >,
        augmentagent_channel_socialapi::SocialApiDmChannel<FallbackReasoner>,
    >,
> {
    use augmentagent_channel_core::trigger::{ChannelRunner, InboundMessageTrigger};
    use augmentagent_channel_socialapi::{
        SocialApiAuth, SocialApiClient, SocialApiDmChannel, SocialApiDmConfig, SocialApiDmSource,
        DEFAULT_DM_MAX_PER_TICK, DEFAULT_DM_POLL_SECS,
    };

    let auth = SocialApiAuth::load_with_store(&store).context("load socialapi auth")?;
    let client = Arc::new(SocialApiClient::new(auth));
    let reasoner = build_reasoner();
    let poll_secs = std::env::var("AUGMENTAGENT_SOCIALAPI_DM_POLL_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(DEFAULT_DM_POLL_SECS);
    let max_per_tick = std::env::var("AUGMENTAGENT_SOCIALAPI_DM_MAX_PER_TICK")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(DEFAULT_DM_MAX_PER_TICK);

    let source = Arc::new(SocialApiDmSource::new(
        client,
        Arc::clone(&store),
        max_per_tick,
    ));
    let trigger = Arc::new(InboundMessageTrigger::new(source));
    let handler = Arc::new(SocialApiDmChannel {
        store: Arc::clone(&store),
        reasoner,
        approvals: broker,
        governor: engagement_governor(store),
        config: SocialApiDmConfig {
            dry_run,
            wiki_root: cli.wiki_dir.clone(),
            skill_dir: socialapi_triage_skill_dir(cli),
        },
    });
    info!(
        interval_secs = poll_secs,
        max_per_tick, "socialapi dm channel ready"
    );
    Ok(ChannelRunner::new(
        trigger,
        handler,
        Duration::from_secs(poll_secs),
        Duration::ZERO,
        "socialapi-dm",
    ))
}

/// #58.3 — watchlist-driven friend-post engagement. Iterates the
/// `friend_watchlist` table, triages each fresh post, surfaces an
/// approval-gated wiki-grounded comment. RateGovernor `Comment` envelope.
fn build_friend_feed_engagement(
    cli: &Cli,
    store: Arc<Store>,
    broker: Arc<dyn ApprovalBroker>,
    dry_run: bool,
) -> Result<FriendFeedEngagement<VoyagerClient, FallbackReasoner>> {
    let (voyager, member_urn, mut config, _) = linkedin_engagement_ctx(cli)?;
    config.dry_run = dry_run;
    let reasoner = build_reasoner();
    let poll_secs = std::env::var("AUGMENTAGENT_LINKEDIN_FRIENDFEED_POLL_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(DEFAULT_FRIEND_FEED_POLL_SECS);
    let max_per_tick = std::env::var("AUGMENTAGENT_LINKEDIN_MAX_FRIEND_POSTS")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(DEFAULT_MAX_FRIEND_POSTS_PER_TICK);
    let source = Arc::new(LinkedInFriendFeedSource::new(
        Arc::clone(&voyager),
        Arc::clone(&store),
        max_per_tick,
    ));
    info!(
        member = %member_urn,
        interval_secs = poll_secs,
        max_per_tick,
        "linkedin friend-feed engagement ready"
    );
    Ok(FriendFeedEngagement {
        store: Arc::clone(&store),
        reasoner,
        approvals: broker,
        governor: engagement_governor(store),
        source,
        member_urn,
        config,
        poll_interval: Duration::from_secs(poll_secs),
    })
}

/// #58.4 — LinkedIn connection-request triage. Polls pending invitations,
/// triages accept/ignore, surfaces an approval card with the recommendation
/// + a suggested opener. The accept/ignore wire call is the approver's job.
fn build_connection_request_engagement(
    cli: &Cli,
    store: Arc<Store>,
    broker: Arc<dyn ApprovalBroker>,
    dry_run: bool,
) -> Result<ConnectionRequestEngagement<VoyagerClient, FallbackReasoner>> {
    let (voyager, member_urn, mut config, _) = linkedin_engagement_ctx(cli)?;
    config.dry_run = dry_run;
    let reasoner = build_reasoner();
    let poll_secs = std::env::var("AUGMENTAGENT_LINKEDIN_INVITE_POLL_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(DEFAULT_INVITATION_POLL_SECS);
    let trigger = Arc::new(InvitationsTrigger::new(
        Arc::clone(&voyager),
        Arc::clone(&store),
    ));
    info!(
        member = %member_urn,
        interval_secs = poll_secs,
        "linkedin connection-request triage ready"
    );
    Ok(ConnectionRequestEngagement {
        store,
        reasoner,
        approvals: broker,
        trigger,
        member_urn,
        config,
        poll_interval: Duration::from_secs(poll_secs),
    })
}

/// Best-effort load of the voyager client. None when neither Keychain nor the
/// legacy file has credentials — callers treat this as "LinkedIn disabled for
/// this run".
fn load_linkedin_client(repo_root: &std::path::Path) -> Option<Arc<VoyagerClient>> {
    match LinkedInAuth::load_with_migration(repo_root) {
        Ok(auth) => Some(Arc::new(VoyagerClient::new(auth))),
        Err(e) => {
            info!(
                "linkedin auth not loaded (keychain + legacy file): {e} (linkedin send disabled this run)"
            );
            None
        }
    }
}

async fn run_linkedin_login(cookies_json: PathBuf) -> Result<()> {
    let raw = std::fs::read_to_string(&cookies_json)
        .with_context(|| format!("read cookies file at {}", cookies_json.display()))?;
    let mut auth: LinkedInAuth = serde_json::from_str(&raw)
        .with_context(|| "parse cookies JSON")?;
    auth.validate()
        .with_context(|| "cookie file missing required fields")?;
    // Stamp harvested_at_ms unless the file already had a value.
    if auth.harvested_at_ms == 0 {
        auth.harvested_at_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
    }

    // Probe voyager once to validate cookies before persisting. Avoids
    // writing a broken auth file that would only surface at poll time.
    let voyager = VoyagerClient::new(auth.clone());
    match voyager.fetch_recent_dms().await {
        Ok(dms) => info!(thread_count = dms.len(), "linkedin cookie probe OK"),
        Err(e) => anyhow::bail!("cookie probe failed: {e}; aborting save"),
    }

    let repo_root = std::env::current_dir().context("current_dir")?;
    let out = default_auth_path(&repo_root);
    auth.save(&out)
        .with_context(|| format!("save auth to {}", out.display()))?;
    // Belt-and-suspenders during the Keychain transition: write to both. The
    // file path is the legacy fallback that `load_with_migration` consults;
    // the Keychain entry is what production loads go through from now on.
    // First-time Keychain writes trigger a macOS permission prompt — click
    // "Always Allow" so subsequent boots don't re-prompt.
    auth.save_to_keychain()
        .context("save auth to keychain (augmentagent/linkedin/default)")?;
    println!("linkedin auth saved to {} + keychain (augmentagent/linkedin/default)", out.display());
    println!("member: {}", auth.member_urn);
    Ok(())
}

async fn run_linkedin_recent() -> Result<()> {
    let repo_root = std::env::current_dir().context("current_dir")?;
    let auth = LinkedInAuth::load_with_migration(&repo_root)
        .context("load linkedin auth from keychain or legacy file")?;
    let voyager = VoyagerClient::new(auth.clone());
    let dms = voyager.fetch_recent_dms().await.context("fetch DMs")?;

    let me = &auth.member_urn;
    println!("{} threads\n", dms.len());
    for (i, dm) in dms.iter().take(15).enumerate() {
        let arrow = if dm.is_outbound(me) { "you →" } else { "peer →" };
        let snippet: String = dm.text.chars().take(100).collect();
        println!(
            "[{:>2}] {}  {}\n     {} {}",
            i + 1,
            chrono::DateTime::<chrono::Local>::from(
                std::time::UNIX_EPOCH + Duration::from_millis(dm.delivered_at_ms as u64)
            )
            .format("%Y-%m-%d %H:%M"),
            dm.peer_name,
            arrow,
            snippet,
        );
    }
    Ok(())
}

/// #61 — LinkedIn 1st-degree connection sync. Dry-run by default (prints a
/// JSON report, writes nothing); `--apply` writes fill-blanks-only wiki
/// pages and persists the sync cursor. Posts an "N new / M updated" summary
/// card to Discord when a broker is wired.
async fn run_linkedin_connections_sync(
    cli: &Cli,
    store: Arc<Store>,
    broker: Arc<dyn ApprovalBroker>,
    apply: bool,
    force_full: bool,
) -> Result<()> {
    use augmentagent_store::LinkedInConnectionSync;
    use std::time::{SystemTime, UNIX_EPOCH};

    let wiki_root = cli
        .wiki_dir
        .clone()
        .context("--wiki-dir is required for connections sync")?;
    let layout = augmentagent_wiki::WikiLayout::new(wiki_root.clone());
    layout.bootstrap().context("wiki bootstrap")?;

    let repo_root = std::env::current_dir().context("current_dir")?;
    let auth = LinkedInAuth::load_with_migration(&repo_root)
        .context("load linkedin auth from keychain or legacy file")?;
    let account_id = auth.member_urn.clone();

    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);

    let prior = store
        .get_linkedin_connection_sync(&account_id)
        .context("read connection-sync cursor")?;
    let last_full = prior.as_ref().and_then(|s| s.last_full_sync_ms);
    let mode = if force_full {
        SyncMode::Full
    } else {
        SyncMode::decide(last_full, now_ms)
    };
    // Resume an interrupted full sync from its persisted offset; deltas
    // always restart at 0 (recency-descending, cheap to re-walk the head).
    let start_offset = match mode {
        SyncMode::Full => prior.as_ref().map(|s| s.cursor_start as usize).unwrap_or(0),
        SyncMode::Delta { .. } => 0,
    };

    let today = chrono::Utc::now().format("%Y-%m-%d").to_string();

    let client = VoyagerConnectionsClient::new(auth);
    let syncer = ConnectionSyncer {
        api: &client,
        layout: &layout,
        today,
        apply,
    };

    info!(
        account = %account_id,
        ?mode,
        start_offset,
        apply,
        "starting linkedin connections sync"
    );
    let report = syncer
        .run(mode, start_offset, |d| tokio::time::sleep(d))
        .await
        .context("connection sync run")?;

    println!(
        "{}",
        serde_json::to_string_pretty(&report).unwrap_or_default()
    );

    if apply {
        // On a completed run, advance the cursor. Full → record full-sync
        // timestamp and reset cursor; delta → record delta timestamp.
        let next = match mode {
            SyncMode::Full => LinkedInConnectionSync {
                account_id: account_id.clone(),
                last_full_sync_ms: Some(now_ms),
                last_delta_sync_ms: prior.as_ref().and_then(|s| s.last_delta_sync_ms),
                cursor_start: 0,
                last_synced_count: report.connections_seen as i64,
            },
            SyncMode::Delta { last_full_sync_ms } => LinkedInConnectionSync {
                account_id: account_id.clone(),
                last_full_sync_ms: Some(last_full_sync_ms),
                last_delta_sync_ms: Some(now_ms),
                cursor_start: 0,
                last_synced_count: report.connections_seen as i64,
            },
        };
        store
            .upsert_linkedin_connection_sync(&next)
            .context("persist connection-sync cursor")?;
    }

    // Surface a summary card (reuses the digest embed path; no buttons).
    if report.created > 0 || report.updated > 0 {
        if let Err(e) = broker
            .post_digest(
                "LinkedIn connections sync",
                &report.discord_summary(),
            )
            .await
        {
            warn!("failed to post connections summary to discord: {e}");
        }
    }

    Ok(())
}

/// #62 — contacts sync. `backend` is `google` (Composio People) or
/// `carddav` (env-configured). Dry-run JSON by default; `--apply` writes
/// fill-blanks wiki pages, indexes phones, persists the sync cursor, and
/// posts a Discord summary.
/// Owner-approved stub merge: the deterministic executor the identity-merge
/// approval flow calls (and a standalone CLI for manual merges). The string
/// transform lives in `augmentagent_wiki::crm::merge_stub_into`; this owns
/// the guards, file IO, `updated:` carry-forward, and the `identity_phone`
/// repoint that keeps future syncs from resurrecting the deleted stub. Every
/// guard lives HERE, so Approve re-runs them all on an hours-old payload (#927).
fn execute_person_merge(
    wiki_dir: Option<&Path>,
    store: &Store,
    from: &str,
    into: &str,
    apply: bool,
) -> Result<PersonMergeReport> {
    let ok_slug =
        |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    anyhow::ensure!(ok_slug(from) && ok_slug(into), "slugs must be [A-Za-z0-9_-]+");
    anyhow::ensure!(from != into, "--from and --into are the same page");
    anyhow::ensure!(
        from.ends_with("_at_contact"),
        "refusing to merge '{from}': only auto-created *_at_contact stubs can be folded \
         (canonical pages may hold human-written content a blind merge would bury)"
    );
    let wiki_root = wiki_dir.context("--wiki-dir is required for person merge")?;
    let layout = augmentagent_wiki::WikiLayout::new(wiki_root.to_path_buf());
    let from_path = layout.people_dir().join(format!("{from}.md"));
    let into_path = layout.people_dir().join(format!("{into}.md"));
    let stub_src = std::fs::read_to_string(&from_path)
        .with_context(|| format!("stub not found: {}", from_path.display()))?;
    let target_src = std::fs::read_to_string(&into_path)
        .with_context(|| format!("target not found: {}", into_path.display()))?;

    let merged = augmentagent_wiki::crm::merge_stub_into(&target_src, &stub_src, from);

    // Carry the stub's `updated:` forward when it is newer — the stub's date
    // is the person's last text, and the stale-contact engine reads it.
    let stub_updated = stub_src
        .split("\n---\n")
        .next()
        .and_then(|fm| fm.lines().find_map(|l| l.strip_prefix("updated:")))
        .map(|v| v.trim().trim_matches('\'').trim_matches('"').to_string());
    let mut content = merged.content.clone();
    let mut bumped = false;
    if let Some(date) = &stub_updated {
        if let Some(newer) = augmentagent_channel_imessage::bump_updated(&content, date) {
            content = newer;
            bumped = true;
        }
    }

    let mut repointed = 0usize;
    if apply {
        if merged.changed || bumped {
            std::fs::write(&into_path, &content)?;
        }
        std::fs::remove_file(&from_path)
            .with_context(|| format!("removing stub {}", from_path.display()))?;
        repointed = store.repoint_phone_identity(from, into)?;
    }
    Ok(PersonMergeReport {
        from: from.to_string(),
        into: into.to_string(),
        applied: apply,
        changed: merged.changed,
        updated_bumped: bumped,
        moved: merged.moved,
        phone_rows_repointed: repointed,
        stub_title: stub_src
            .lines()
            .find_map(|l| l.strip_prefix("# ").map(str::trim).filter(|t| !t.is_empty()))
            .unwrap_or(from)
            .to_string(),
        last_message: stub_updated,
        evidence: augmentagent_wiki::crm::source_lines(&stub_src),
    })
}

/// What one [`execute_person_merge`] run did, plus the stub facts the card's
/// evidence block needs (#927): `# ` heading, `updated:` date, Source bullets.
#[derive(Debug, serde::Serialize)]
struct PersonMergeReport {
    from: String,
    into: String,
    applied: bool,
    changed: bool,
    updated_bumped: bool,
    moved: Vec<String>,
    phone_rows_repointed: usize,
    stub_title: String,
    last_message: Option<String>,
    evidence: Vec<String>,
}

fn run_person_merge(
    cli: &Cli,
    store: Arc<Store>,
    from: &str,
    into: &str,
    apply: bool,
) -> Result<()> {
    let report = execute_person_merge(cli.wiki_dir.as_deref(), &store, from, into, apply)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

/// #927 — `kind` on the rows a merge proposal writes. Approve and Revise
/// dispatch on this BEFORE any platform check.
const IDENTITY_MERGE_KIND: &str = "identity_merge";

/// #927 — the machine payload Approve executes, carried on the action row's
/// `originalBody`, never re-read out of the rendered card text.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct IdentityMergePayload {
    stub_slug: String,
    candidate_slug: String,
}

/// #927 — settled? `pending` = a card is up, `sent` = it ran, `rejected` = the
/// cooldown Skip arms. A `superseded`/`error` row never reached a decision.
fn identity_merge_is_settled(status: &str) -> bool {
    matches!(status, "pending" | "sent" | "rejected")
}

/// #927 — synthesize the approval card, as `(card, machine payload for the
/// row's originalBody, draft)`. `from` is the stub's display name and `body`
/// its evidence — who this is, where the texts came from — because that is what
/// the owner judges. Neither is a mailbox, and `thread_id` is `None`, which is
/// why `pending_actions_for_reconcile` has to exclude this kind.
fn build_identity_merge_card(
    report: &PersonMergeReport,
) -> (augmentagent_store::Email, String, String) {
    // Same struct Approve deserializes, so a rename can't strand pending cards.
    let payload = serde_json::to_string(&IdentityMergePayload {
        stub_slug: report.from.clone(),
        candidate_slug: report.into.clone(),
    })
    .expect("IdentityMergePayload serializes");
    let mut evidence = format!(
        "Contact stub {} — last text {}\n",
        report.from,
        report.last_message.as_deref().unwrap_or("unknown"),
    );
    evidence.extend(report.evidence.iter().map(|l| format!("• {l}\n")));
    let fills = report.moved.join(", ");
    let draft = format!(
        "Approve folds this stub into {} — moves its identities and Source \
         lines onto the target (fill-blanks only), deletes the stub, and \
         repoints its phone rows.\nfills: {}",
        report.into,
        if fills.is_empty() { "(nothing new — provenance only)" } else { &fills },
    );
    let email = augmentagent_store::Email {
        attachments: Vec::new(),
        to: String::new(),
        cc: String::new(),
        message_id: format!("merge:{}->{}", report.from, report.into),
        thread_id: None,
        from: report.stub_title.clone(),
        subject: format!("Merge '{}' into {}?", report.stub_title, report.into),
        body: evidence,
        date: report.last_message.clone().unwrap_or_default(),
        account_entity_id: None,
        platform: "wiki".into(),
        kind: IDENTITY_MERGE_KIND.into(),
    };
    (email, payload, draft)
}

/// #927 — persist the proposal. The action row IS the suggestion record: its
/// status is the pending / decided state, its `originalBody` the payload.
fn record_identity_merge_proposal(
    store: &Store,
    inbound: &augmentagent_store::Email,
    payload: &str,
    draft: &str,
) -> Result<String> {
    store.upsert_email(inbound).context("upsert merge proposal row")?;
    store
        .log_action(
            &inbound.message_id,
            inbound.thread_id.as_deref(),
            &inbound.from,
            &inbound.subject,
            Some(payload),
            Some(draft),
            ActionStatus::Pending,
        )
        .context("log identity-merge action row")
}

/// #927 — the suggester: every `*_at_contact` stub whose phone, iMessage handle
/// or email already sits on exactly ONE canonical page. The shared identity IS
/// the evidence, which is what makes the pair high-confidence; a stub matching
/// two canonical pages is ambiguous and is left for a human instead.
fn high_confidence_merge_candidates(wiki_dir: &Path) -> Result<Vec<(String, String)>> {
    let layout = augmentagent_wiki::WikiLayout::new(wiki_dir.to_path_buf());
    let index = augmentagent_wiki::IdentityIndex::build(&layout)?;
    let is_stub = |p: &&augmentagent_wiki::PersonPage| p.slug.ends_with("_at_contact");
    let shares = |page: &augmentagent_wiki::Identities, stub: &augmentagent_wiki::Identities| {
        stub.phone.iter().any(|v| page.matches("phone", v))
            || stub.imessage.iter().any(|v| page.matches("imessage", v))
            || stub.email.iter().any(|v| page.matches("email", v))
    };
    let mut pairs: Vec<(String, String)> = index
        .pages()
        .iter()
        .filter(is_stub)
        .filter_map(|stub| {
            let mut hits = index
                .pages()
                .iter()
                .filter(|c| !is_stub(c) && shares(&c.identities, &stub.identities));
            let only = hits.next()?;
            hits.next().is_none().then(|| (stub.slug.clone(), only.slug.clone()))
        })
        .collect();
    pairs.sort();
    Ok(pairs)
}

/// #927 — the automatic path: every high-confidence suggestion becomes a card;
/// one pair failing must not swallow the rest of the scan.
async fn propose_high_confidence_merges(cli: &Cli, store: &Store) -> Result<()> {
    let wiki = cli.wiki_dir.as_deref().context("--wiki-dir is required for person merge")?;
    for (stub, canonical) in high_confidence_merge_candidates(wiki)? {
        if let Err(e) = propose_identity_merge(Some(wiki), store, &stub, &canonical).await {
            tracing::warn!(stub, canonical, "identity-merge proposal failed: {e:#}");
        }
    }
    Ok(())
}

/// #927 — raise one suggested merge as a Discord approval card. Nothing is
/// written to the wiki here: the dry run only re-proves the guards and collects
/// the evidence. The merge runs when — and only when — Approve is clicked.
async fn propose_identity_merge(
    wiki_dir: Option<&Path>,
    store: &Store,
    from: &str,
    into: &str,
) -> Result<()> {
    let report = execute_person_merge(wiki_dir, store, from, into, false)?;
    let message_id = format!("merge:{from}->{into}");
    if let Some((id, status, _)) = store.latest_action_for_message(&message_id)? {
        if identity_merge_is_settled(&status) {
            println!("{message_id}: already {status} ({id}); not re-proposing");
            return Ok(());
        }
    }
    let token = std::env::var("DISCORD_BOT_TOKEN")
        .context("DISCORD_BOT_TOKEN required to raise a merge card (set it in the .env)")?;
    let cid: u64 = std::env::var("DISCORD_CHANNEL_ID")
        .context("DISCORD_CHANNEL_ID required to raise a merge card")?
        .parse()
        .context("DISCORD_CHANNEL_ID must be numeric")?;

    let (inbound, payload, draft) = build_identity_merge_card(&report);
    let action_id = record_identity_merge_proposal(store, &inbound, &payload, &draft)?;

    let posted = DiscordApprovalBroker::rest_only(&token, cid)
        .post_approval(&action_id, &inbound, &draft)
        .await
        .context("post identity-merge approval card");
    if let Err(e) = &posted {
        // A card Discord never got is unclickable AND holds the merge key.
        // Park it `error` — deliberately not "settled" — so it can be re-raised.
        let msg = format!("{e:#}");
        let _ = store.update_action_status(&action_id, ActionStatus::Error, None, Some(&msg));
    }
    posted?;
    // Claim the nudge slot (count 0 → 1) exactly as the compose card does, or
    // the NudgeScheduler sees a pending row still at nudgeCount=0 and posts a
    // duplicate of the card the owner is looking at (#412).
    let now_ms = chrono::Utc::now().timestamp_millis();
    if let Err(e) = store.record_nudge(&action_id, now_ms + augmentagent_store::NUDGE_INTERVAL_MS) {
        tracing::warn!(action_id, "record_nudge after identity-merge card failed: {e}");
    }
    println!("card posted: action_id={action_id}\n{}\n{draft}", inbound.body);
    Ok(())
}

/// #885 — iMessage history → person pages. Dry-run prints the JSON report
/// and writes nothing; `--apply` writes fill-blanks pages and indexes phones.
fn run_imessage_sync(cli: &Cli, store: Arc<Store>, apply: bool) -> Result<()> {
    let config = augmentagent_channel_imessage::ImessageConfig::load()
        .context("AUGMENTAGENT_IMESSAGE_REPO_DIR is required for imessage sync")?;
    let wiki_root = cli
        .wiki_dir
        .clone()
        .context("--wiki-dir is required for imessage sync")?;
    let layout = augmentagent_wiki::WikiLayout::new(wiki_root);
    layout.bootstrap().context("wiki bootstrap")?;

    let bundle = augmentagent_channel_imessage::Bundle::open(&config.repo_dir);
    let syncer = augmentagent_channel_imessage::ImessageSyncer {
        bundle: &bundle,
        layout: &layout,
        store: &store,
        apply,
    };
    info!(repo = %config.repo_dir.display(), apply, "starting imessage sync");
    let report = syncer.run().context("imessage sync run")?;
    println!(
        "{}",
        serde_json::to_string_pretty(&report).unwrap_or_default()
    );
    Ok(())
}

/// #886 — one incremental poll pass: new bundle entries → `emails` rows,
/// cursor advanced. Wiki ingest is deliberately not fired here (daemon-only).
fn run_imessage_poll_once(store: Arc<Store>) -> Result<()> {
    let config = augmentagent_channel_imessage::ImessageConfig::load()
        .context("AUGMENTAGENT_IMESSAGE_REPO_DIR is required for imessage poll")?;
    let bundle = augmentagent_channel_imessage::Bundle::open(&config.repo_dir);
    let (stats, deltas) = augmentagent_channel_imessage::poll_once(&bundle, &store)
        .context("imessage poll")?;
    println!(
        "{}",
        serde_json::json!({
            "conversations_with_new": stats.conversations_with_new,
            "emails_inserted": stats.emails_inserted,
            "first_run_conversations": deltas.iter().filter(|d| d.first_run).count(),
        })
    );
    Ok(())
}

/// #886 — daemon loop: refresh the bundle repo, poll for new entries, and
/// fan fresh (non-first-run) conversation deltas into `Capture` wiki ingest.
async fn imessage_poll_loop(
    config: augmentagent_channel_imessage::ImessageConfig,
    store: Arc<Store>,
    reasoner: Arc<FallbackReasoner>,
    wiki_root: Option<PathBuf>,
    wiki_schema: Option<String>,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    const POLL_INTERVAL: Duration = Duration::from_secs(30 * 60);
    let mut tick = tokio::time::interval(POLL_INTERVAL);
    loop {
        tokio::select! {
            _ = shutdown.cancelled() => return Ok(()),
            _ = tick.tick() => {}
        }
        // The bundle repo is kept current by the operator's own sync job;
        // a pull failure (offline, not a git repo) degrades to reading
        // whatever is on disk.
        let pull = tokio::process::Command::new("git")
            .arg("-C")
            .arg(&config.repo_dir)
            .args(["pull", "--ff-only", "--quiet"])
            .output()
            .await;
        if let Ok(out) = &pull {
            if !out.status.success() {
                warn!(
                    stderr = %String::from_utf8_lossy(&out.stderr).trim(),
                    "imessage bundle git pull failed; reading on-disk state"
                );
            }
        }

        let bundle = augmentagent_channel_imessage::Bundle::open(&config.repo_dir);
        let (stats, deltas) = match augmentagent_channel_imessage::poll_once(&bundle, &store) {
            Ok(r) => r,
            Err(e) => {
                warn!("imessage poll failed: {e:#}");
                continue;
            }
        };
        if stats.emails_inserted > 0 {
            info!(
                conversations = stats.conversations_with_new,
                emails = stats.emails_inserted,
                "imessage poll ingested new messages"
            );
        }
        let (Some(root), Some(schema)) = (&wiki_root, &wiki_schema) else {
            continue;
        };
        for delta in deltas.iter().filter(|d| !d.first_run) {
            augmentagent_channel_core::ingest::spawn_ingest(
                Arc::clone(&reasoner),
                root.clone(),
                schema.clone(),
                augmentagent_channel_imessage::batched_delta_email(delta),
                augmentagent_channel_core::decision::DecisionKind::Capture,
                Some("imessage history sync".to_string()),
                None,
                augmentagent_channel_core::ingest::IngestTrigger::ImessageHistory,
            );
        }
    }
}

async fn run_contacts_sync(
    cli: &Cli,
    store: Arc<Store>,
    broker: Arc<dyn ApprovalBroker>,
    backend: &str,
    entity_id: &str,
    apply: bool,
) -> Result<()> {
    let wiki_root = cli
        .wiki_dir
        .clone()
        .context("--wiki-dir is required for contacts sync")?;
    let layout = augmentagent_wiki::WikiLayout::new(wiki_root);
    layout.bootstrap().context("wiki bootstrap")?;
    let today = chrono::Utc::now().format("%Y-%m-%d").to_string();

    let (source, account_id): (Box<dyn ContactsSource>, String) = match backend {
        "google" => {
            let api_key = std::env::var("COMPOSIO_API_KEY")
                .context("COMPOSIO_API_KEY env var required for the google backend")?;
            (
                Box::new(GooglePeopleSource::new(api_key, entity_id.to_string())),
                entity_id.to_string(),
            )
        }
        "carddav" => {
            let src = CardDavSource::from_env().context(
                "CardDAV not configured — set AUGMENTAGENT_CARDDAV_URL / _USER / _PASS",
            )?;
            (Box::new(src), "default".to_string())
        }
        other => anyhow::bail!("unknown contacts backend '{other}' (use google|carddav)"),
    };

    let syncer = ContactsSyncer {
        source: source.as_ref(),
        layout: &layout,
        store: &store,
        today,
        apply,
    };
    info!(backend, account = %account_id, apply, "starting contacts sync");
    let report = syncer
        .run(&account_id)
        .await
        .context("contacts sync run")?;

    println!(
        "{}",
        serde_json::to_string_pretty(&report).unwrap_or_default()
    );

    if report.created > 0 || report.updated > 0 {
        if let Err(e) = broker
            .post_digest("Contacts sync", &report.discord_summary())
            .await
        {
            warn!("failed to post contacts summary to discord: {e}");
        }
    }
    Ok(())
}

/// #64 — email-signature backfill. Scans stored email bodies since
/// `--since`, detects the signature block (regex + line-density), runs the
/// LLM field extractor (strict JSON + retry + regex fallback), and merges
/// high-confidence fields fill-blanks-only into the sender's wiki page.
/// Low-confidence fields are collected into a single daily Discord digest
/// for manual approval. Dry-run JSON by default.
#[allow(clippy::too_many_arguments)]
async fn run_backfill_signatures(
    cli: &Cli,
    store: Arc<Store>,
    broker: Arc<dyn ApprovalBroker>,
    since: Option<String>,
    limit: i64,
    min_confidence: f64,
    apply: bool,
) -> Result<()> {
    use augmentagent_wiki::{merge_person_page, slug_from_email, WikiLayout};

    let wiki_root = cli
        .wiki_dir
        .clone()
        .context("--wiki-dir is required for signature backfill")?;
    let layout = WikiLayout::new(wiki_root);
    layout.bootstrap().context("wiki bootstrap")?;
    let today = chrono::Utc::now().format("%Y-%m-%d").to_string();

    // Resolve --since (default 180d ago) to an epoch-ms lower bound.
    let since_ms = match &since {
        Some(s) => {
            let d = chrono::NaiveDate::parse_from_str(s.trim(), "%Y-%m-%d")
                .context("--since must be YYYY-MM-DD")?;
            d.and_hms_opt(0, 0, 0)
                .unwrap()
                .and_utc()
                .timestamp_millis()
        }
        None => (chrono::Utc::now() - chrono::Duration::days(180)).timestamp_millis(),
    };

    let rows = store
        .email_bodies_since(since_ms, limit)
        .context("read email bodies for signature backfill")?;

    let reasoner = build_reasoner();
    let extractor = SignatureExtractor::new(reasoner.as_ref());

    let mut scanned = 0usize;
    let mut sig_found = 0usize;
    let mut created = 0usize;
    let mut updated = 0usize;
    let mut noop = 0usize;
    // #120 — skipped because the sender looks non-human (newsletter, ESP,
    // no-reply, bulk-mail body markers). We never create a NEW `people/`
    // page for such senders; existing pages are still updated so we don't
    // silently lose data the human may have curated.
    let mut skipped_non_human = 0usize;
    // De-dupe per sender within a run (latest sig wins; merge is fill-blanks
    // so order is immaterial, but skip redundant LLM calls).
    let mut seen_senders: std::collections::BTreeSet<String> =
        std::collections::BTreeSet::new();
    let mut digest_lines: Vec<String> = Vec::new();

    for (_mid, from, body) in rows {
        scanned += 1;
        let slug = slug_from_email(&from);
        if slug.is_empty() || !seen_senders.insert(slug.clone()) {
            continue;
        }
        let stripped = strip_quoted_reply(&body);
        let Some(block) = detect_signature_block(&stripped) else {
            continue;
        };
        sig_found += 1;
        let fields = match extractor.extract(&block.text).await {
            Ok(f) => f,
            Err(e) => {
                tracing::debug!(%from, "sig extract skipped: {e}");
                continue;
            }
        };
        let (patch, deferred) = signature_patch(&fields, &today, min_confidence);
        for d in deferred {
            digest_lines.push(format!("{from}: {d}"));
        }
        if patch.is_empty() {
            noop += 1;
            continue;
        }

        let path = layout.people_dir().join(format!("{slug}.md"));
        let existing = std::fs::read_to_string(&path).ok();

        // #120 — gate new-page creation: if there's no existing page AND
        // this sender doesn't look human (newsletter / vendor / no-reply),
        // skip rather than pollute `people/`. Existing pages still merge —
        // those are presumed already-curated.
        if existing.is_none() && !is_human_sender(&from, &body) {
            skipped_non_human += 1;
            tracing::debug!(%from, "skipped non-human sender for people/ creation");
            continue;
        }

        let merged = merge_person_page(existing.as_deref(), &patch);
        if !merged.changed {
            noop += 1;
            continue;
        }
        if merged.created {
            created += 1;
        } else {
            updated += 1;
        }
        if apply {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, &merged.content)?;
        }
    }

    let report = serde_json::json!({
        "scanned": scanned,
        "signatures_detected": sig_found,
        "created": created,
        "updated": updated,
        "noop": noop,
        "skipped_non_human": skipped_non_human,
        "deferred_low_confidence": digest_lines.len(),
        "applied": apply,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&report).unwrap_or_default()
    );

    if !digest_lines.is_empty() {
        let body = format!(
            "Low-confidence signature fields needing review ({}):\n{}",
            digest_lines.len(),
            digest_lines
                .iter()
                .take(40)
                .map(|l| format!("- {l}"))
                .collect::<Vec<_>>()
                .join("\n"),
        );
        if let Err(e) = broker
            .post_digest("Signature backfill — low-confidence", &body)
            .await
        {
            warn!("failed to post signature digest: {e}");
        }
    }
    Ok(())
}

/// Manual / test feed-post path (#51/#77). The daemon publishes through the
/// approval pipeline; this command is for one-shots and smoke tests. It
/// enforces the same rolling-24h cap (3 posts/day) the daemon path does, a
/// first-N-posts second-confirmation guard, and a `--dry-run` that prints
/// the request body without sending.
// ================================================================
// #58 — engagement-automation scheduled posts
// ================================================================

/// Canonical form of a `scheduled_posts.platform` value: trimmed and
/// lowercased. Dispatch used to `match post.platform.as_str()` raw, so `"X"`,
/// `"Twitter"`, or a stray trailing space fell through to a terminal failure
/// at fire time (#550). Applied at enqueue AND at dispatch, so old rows
/// written before this landed still route correctly.
fn normalize_post_platform(platform: &str) -> String {
    platform.trim().to_ascii_lowercase()
}

/// Platforms the scheduled-post publisher can actually deliver, for error
/// messages and `--platform` validation.
const SCHEDULABLE_PLATFORMS: &str = "linkedin, twitter (or x), socialapi, socialapi:<sub-platform>";

/// True iff [`MultiPlatformPublisher::publish`] has an arm for `platform`
/// (already normalized).
///
/// #550: `instagram` was advertised as a valid `--platform` on
/// `schedule-post add` and accepted by `enqueue_scheduled_post`, but the
/// publisher has no instagram arm — so the row sat `queued` until its fire
/// time and only then failed terminally. Rejecting at enqueue tells the
/// operator while they are still looking at the terminal.
fn is_schedulable_platform(platform: &str) -> bool {
    matches!(platform, "linkedin" | "twitter" | "x" | "socialapi")
        || platform.starts_with("socialapi:")
}

/// Routes a [`ScheduledPost`] to the right per-platform poster. Keeps
/// `channel-core`'s `PostPublisher` trait satisfied without that crate
/// depending on the platform crates. Auth is loaded lazily per publish so a
/// missing LinkedIn/Twitter session degrades to a `Failed` outcome (the
/// engine marks the row failed + alerts) rather than panicking the daemon.
struct MultiPlatformPublisher {
    store: Arc<Store>,
    repo_root: PathBuf,
    dry_run: bool,
}

#[async_trait]
impl augmentagent_channel_core::PostPublisher for MultiPlatformPublisher {
    async fn publish(
        &self,
        post: &augmentagent_store::ScheduledPost,
    ) -> augmentagent_channel_core::PublishOutcome {
        use augmentagent_channel_core::PublishOutcome;
        // #550: normalize before dispatch so "X" / "Twitter" / stray
        // whitespace route correctly instead of hard-failing at fire time.
        let platform = normalize_post_platform(&post.platform);
        match platform.as_str() {
            "linkedin" => {
                let auth = match LinkedInAuth::load_with_migration(&self.repo_root) {
                    Ok(a) => a,
                    Err(e) => {
                        return PublishOutcome::Failed {
                            message: format!("linkedin auth: {e}"),
                        }
                    }
                };
                if self.dry_run {
                    return PublishOutcome::DryRun;
                }
                // #551 — honor `media_paths`. This arm used to call
                // `PostDraft::text` unconditionally, so `linkedin post
                // --image` attached an image but the SAME post scheduled
                // published text-only, silently and with a success status.
                let paths: Vec<String> = post
                    .media_paths
                    .as_deref()
                    .and_then(|s| serde_json::from_str::<Vec<String>>(s).ok())
                    .unwrap_or_default();
                // Read every file BEFORE the first register call, so a
                // missing path fails the row cleanly instead of stranding
                // half-uploaded assets on LinkedIn's side.
                let mut loaded: Vec<(Vec<u8>, Option<String>)> = Vec::with_capacity(paths.len());
                for path in &paths {
                    let p = std::path::Path::new(path);
                    match std::fs::read(p) {
                        Ok(bytes) => {
                            let name = p
                                .file_name()
                                .and_then(|n| n.to_str())
                                .map(str::to_string);
                            loaded.push((bytes, name));
                        }
                        Err(e) => {
                            // Do NOT silently fall back to a text-only post:
                            // that is the bug this fixes. The row fails and
                            // the operator sees why.
                            return PublishOutcome::Failed {
                                message: format!(
                                    "linkedin: scheduled post attaches {path}, \
                                     which could not be read at fire time ({e}). \
                                     Media paths are resolved on the daemon host, \
                                     not the machine that queued the post."
                                ),
                            };
                        }
                    }
                }
                let voyager = VoyagerClient::new(auth);
                let mut draft = PostDraft::text(&post.body);
                for (bytes, name) in &loaded {
                    draft = draft.with_image(bytes.as_slice(), name.as_deref());
                }
                match voyager.create_share(draft).await {
                    Ok(urn) => PublishOutcome::Posted { external_id: urn.0 },
                    Err(e) => PublishOutcome::Failed {
                        message: format!("linkedin create_share: {e}"),
                    },
                }
            }
            "twitter" | "x" => {
                // #551 — X cannot carry media at all: CreateTweetClient
                // rejects a non-empty media list outright (media upload is
                // deferred until the base path is validated live, #552).
                // Publishing text-only would drop the attachment silently
                // and report success, so fail the row instead.
                if post
                    .media_paths
                    .as_deref()
                    .and_then(|s| serde_json::from_str::<Vec<String>>(s).ok())
                    .is_some_and(|v| !v.is_empty())
                {
                    return PublishOutcome::Failed {
                        message: "twitter: this post has media attached, but X media \
                                  upload is not implemented (see #552). Publishing \
                                  would have dropped the media and posted text only \
                                  — refusing instead."
                            .to_string(),
                    };
                }
                let auth = match TwitterAuth::load_with_migration(&self.repo_root) {
                    Ok(a) => a,
                    Err(e) => {
                        return PublishOutcome::Failed {
                            message: format!("twitter auth: {e}"),
                        }
                    }
                };
                let api = Arc::new(TwitterClient::new(auth));
                let client = CreateTweetClient::new(
                    api,
                    Arc::clone(&self.store),
                    self.dry_run,
                );
                match client.create(&post.body, None, &[]).await {
                    Ok(augmentagent_channel_twitter::PostOutcome::DryRun) => {
                        PublishOutcome::DryRun
                    }
                    Ok(out) => PublishOutcome::Posted {
                        external_id: format!("{out:?}"),
                    },
                    Err(e) => PublishOutcome::Failed {
                        message: format!("twitter create: {e}"),
                    },
                }
            }
            // #240 — SocialAPI.ai outbound. The row's `platform` of "socialapi"
            // routes it here; the connected account lives in
            // `socialapi_account_id`. The real sub-platform (instagram / x /
            // linkedin / …) is encoded in the platform string as a
            // "socialapi:<sub>" suffix when known — e.g. "socialapi:instagram";
            // a bare "socialapi" leaves `PostTarget.platform` empty so the
            // unified API resolves the destination from the account id itself.
            // Single-target only — multi-account fan-out (N rows) is #241.
            p if p == "socialapi" || p.starts_with("socialapi:") => {
                use augmentagent_channel_socialapi::{
                    CreatePostRequest, PostTarget, SocialApiAuth, SocialApiClient,
                };
                let account_id = match post.socialapi_account_id.clone() {
                    Some(a) if !a.is_empty() => a,
                    _ => {
                        return PublishOutcome::Failed {
                            message: "socialapi: scheduled post has no \
                                      socialapi_account_id set"
                                .to_string(),
                        }
                    }
                };
                let auth = match SocialApiAuth::load_with_store(&self.store) {
                    Ok(a) => a,
                    Err(e) => {
                        return PublishOutcome::Failed {
                            message: format!("socialapi auth: {e}"),
                        }
                    }
                };
                if self.dry_run {
                    return PublishOutcome::DryRun;
                }
                // Sub-platform is the part after "socialapi:" if present;
                // empty otherwise (the API resolves it from the account id).
                let target_platform = platform
                    .strip_prefix("socialapi:")
                    .unwrap_or("")
                    .to_string();
                // `media_paths` holds LOCAL FILE PATHS (see
                // `ScheduledPost::media_paths`); `media_ids` wants opaque ids
                // minted by an upload handshake. Those are not the same thing,
                // and there is currently NO upload endpoint on the client —
                // #544 removed the speculative one when the live API was
                // modelled, and nothing replaced it.
                //
                // Passing paths through as ids is worse than useless: a
                // wrong-shape media array is ignored by the API, so the post
                // publishes text-only and returns 2xx. The caption ships
                // without its image and the daemon reports success. Refuse
                // instead, loudly, until a real upload path exists.
                if post
                    .media_paths
                    .as_deref()
                    .and_then(|s| serde_json::from_str::<Vec<String>>(s).ok())
                    .is_some_and(|v| !v.is_empty())
                {
                    return PublishOutcome::Failed {
                        message: "socialapi: this post has media attached, but \
                                  SocialAPI.ai media upload is not implemented \
                                  (no upload endpoint on the client). Publishing \
                                  would have silently dropped the media and \
                                  posted text only — refusing instead."
                            .to_string(),
                    };
                }
                let client = SocialApiClient::new(auth);
                let req = CreatePostRequest {
                    // Wire field is `text` (#543). The scheduler already held
                    // this post until its fire time — publish immediately.
                    text: post.body.clone(),
                    targets: vec![PostTarget {
                        account_id,
                        platform: target_platform,
                    }],
                    // Always None until an upload handshake exists; the
                    // guard above rejects any row that wanted media.
                    media_ids: None,
                    publish_now: Some(true),
                    scheduled_at: None,
                };
                match client.create_post(&req).await {
                    Ok(resp) => PublishOutcome::Posted {
                        external_id: resp.id,
                    },
                    Err(e) => PublishOutcome::Failed {
                        message: format!("socialapi create_post: {e}"),
                    },
                }
            }
            other => PublishOutcome::Failed {
                message: format!(
                    "no scheduled-post publisher wired for platform '{other}' \
                     (supported: {SCHEDULABLE_PLATFORMS}). Instagram posting \
                     goes through the composer, not the scheduler."
                ),
            },
        }
    }
}

async fn run_schedule_post(store: Arc<Store>, op: &SchedulePostOp) -> Result<()> {
    match op {
        SchedulePostOp::Add {
            platform,
            body,
            at,
        } => {
            let fire_at_ms = parse_fire_at(at)?;
            // #550: validate HERE, not at fire time. A platform the publisher
            // can't deliver used to queue happily and fail terminally hours
            // later, with the operator long gone.
            let platform = normalize_post_platform(platform);
            if !is_schedulable_platform(&platform) {
                anyhow::bail!(
                    "cannot schedule posts for platform '{platform}' \
                     (supported: {SCHEDULABLE_PLATFORMS})"
                );
            }
            let id = store.enqueue_scheduled_post(
                &platform,
                body,
                None,
                fire_at_ms,
                None,
            )?;
            println!(
                "queued scheduled post {id} for {platform} at {} (unix ms {fire_at_ms})",
                at
            );
            Ok(())
        }
        SchedulePostOp::List => {
            let rows = store.list_pending_scheduled_posts()?;
            if rows.is_empty() {
                println!("no pending scheduled posts");
            }
            for r in rows {
                println!(
                    "{}  {:<9}  {:<9}  fire@{}  {}",
                    r.id,
                    r.platform,
                    r.status,
                    r.fire_at_ms,
                    r.body.chars().take(60).collect::<String>()
                );
            }
            Ok(())
        }
        SchedulePostOp::Cancel { id } => {
            if store.cancel_scheduled_post(id)? {
                println!("cancelled {id}");
            } else {
                println!("not cancellable (already fired / unknown id): {id}");
            }
            Ok(())
        }
    }
}

async fn run_engagement(store: Arc<Store>, op: &EngagementOp) -> Result<()> {
    match op {
        EngagementOp::WatchPost {
            platform,
            external_id,
            days,
        } => {
            let now = chrono::Utc::now().timestamp_millis();
            let poll_until = now + days.max(&1) * 86_400_000;
            let id = store.upsert_own_post(platform, external_id, now, poll_until)?;
            println!(
                "watching {platform} post {external_id} (row {id}) for comments \
                 until {poll_until} (unix ms; ~{days}d)"
            );
            Ok(())
        }
        EngagementOp::WatchFriend {
            platform,
            handle,
            wiki_slug,
            engagement,
        } => {
            store.upsert_friend_watch(
                platform,
                handle,
                wiki_slug.as_deref(),
                engagement,
            )?;
            println!(
                "watching {platform} friend {handle} (tier={engagement}{})",
                wiki_slug
                    .as_deref()
                    .map(|s| format!(", wiki={s}"))
                    .unwrap_or_default()
            );
            Ok(())
        }
        EngagementOp::Invites => {
            let rows = store.pending_connection_requests()?;
            if rows.is_empty() {
                println!("no pending connection requests");
            }
            for r in rows {
                println!(
                    "{}  {:<9}  {}  {}",
                    r.id,
                    r.platform,
                    r.requester_name.as_deref().unwrap_or("(unknown)"),
                    r.external_id
                );
            }
            Ok(())
        }
    }
}

/// Accept either an RFC3339 timestamp or raw unix seconds for `--at`.
fn parse_fire_at(s: &str) -> Result<i64> {
    if let Ok(secs) = s.parse::<i64>() {
        return Ok(secs * 1000);
    }
    let dt = chrono::DateTime::parse_from_rfc3339(s)
        .with_context(|| format!("--at must be RFC3339 or unix seconds, got {s:?}"))?;
    Ok(dt.timestamp_millis())
}

async fn run_linkedin_post(
    store: Arc<Store>,
    text: String,
    images: Vec<PathBuf>,
    visibility: String,
    dry_run: bool,
) -> Result<()> {
    let vis = Visibility::parse(&visibility)
        .ok_or_else(|| anyhow::anyhow!("invalid --visibility (use public|connections)"))?;

    // Dry-run: build + print the canonical body, no auth, no send.
    if dry_run {
        // One placeholder urn per image so the dry-run body has the same
        // shape (and array length) the real post would send.
        let placeholders: Vec<String> = (0..images.len())
            .map(|i| format!("<image-urn-{}>", i + 1))
            .collect();
        let refs: Vec<&str> = placeholders.iter().map(String::as_str).collect();
        let body = build_normshares_body(&text, vis, &refs);
        println!(
            "[linkedin post dry-run] visibility={visibility} images={}\n{}",
            images.len(),
            serde_json::to_string_pretty(&body)?
        );
        return Ok(());
    }

    // --- rolling-24h cap preflight (3 posts/day per #77 §7) ---
    const POST_DAILY_CAP: u32 = 3;
    let now_ms = chrono::Utc::now().timestamp_millis();
    let used = store
        .linkedin_action_count_since("post", now_ms - 24 * 3600 * 1000)
        .context("read linkedin_action_log")?;
    if used >= POST_DAILY_CAP {
        anyhow::bail!(
            "linkedin post cap reached: {used}/{POST_DAILY_CAP} in the last 24h; deferring"
        );
    }

    // --- first-N-posts second-confirmation guard ---
    // Posting to the user's professional surface is the highest-blast-radius
    // action in the system. For the first few posts ever made by this tool,
    // require an explicit AUGMENTAGENT_LINKEDIN_POST_CONFIRM=yes so a stray
    // command can't quietly publish.
    const GUARDED_FIRST_N: u32 = 3;
    let lifetime = store
        .linkedin_action_count_since("post", 0)
        .context("read lifetime linkedin posts")?;
    if lifetime < GUARDED_FIRST_N {
        let confirmed = std::env::var("AUGMENTAGENT_LINKEDIN_POST_CONFIRM")
            .map(|v| v.eq_ignore_ascii_case("yes"))
            .unwrap_or(false);
        if !confirmed {
            anyhow::bail!(
                "second-confirmation required for the first {GUARDED_FIRST_N} posts \
                 (post #{} lifetime): re-run with AUGMENTAGENT_LINKEDIN_POST_CONFIRM=yes",
                lifetime + 1
            );
        }
    }

    let repo_root = std::env::current_dir().context("current_dir")?;
    let auth = LinkedInAuth::load_with_migration(&repo_root)
        .context("load linkedin auth from keychain or legacy file")?;
    let voyager = VoyagerClient::new(auth);

    // Read every image up front so a missing file fails before any network
    // call — a partial upload leaves orphaned assets on LinkedIn's side.
    let loaded: Vec<(Vec<u8>, Option<String>)> = images
        .iter()
        .map(|p| {
            let bytes = std::fs::read(p)
                .with_context(|| format!("read image {}", p.display()))?;
            let name = p
                .file_name()
                .and_then(|n| n.to_str())
                .map(str::to_string);
            Ok::<_, anyhow::Error>((bytes, name))
        })
        .collect::<Result<_, _>>()?;

    let mut draft = PostDraft {
        text: &text,
        images: Vec::new(),
        visibility: vis,
    };
    for (bytes, name) in &loaded {
        draft.images.push(augmentagent_channel_linkedin::PostImage::new(
            bytes.as_slice(),
            name.as_deref(),
        ));
    }

    let log_id = uuid::Uuid::new_v4().to_string();
    match voyager.create_share(draft).await {
        Ok(urn) => {
            store
                .log_linkedin_action(
                    &log_id,
                    "post",
                    Some(&urn.0),
                    "ok",
                    now_ms,
                    None,
                )
                .ok();
            println!("posted: {}", urn.0);
            Ok(())
        }
        Err(e) => {
            store
                .log_linkedin_action(
                    &uuid::Uuid::new_v4().to_string(),
                    "post",
                    None,
                    "failed",
                    now_ms,
                    Some(&format!("{e}")),
                )
                .ok();
            Err(anyhow::anyhow!("linkedin create_share: {e}"))
        }
    }
}

// ================================================================
// X / Twitter (issues #14, #15, #16, #79)
// ================================================================

async fn run_twitter_login(session_json: PathBuf) -> Result<()> {
    let raw = std::fs::read_to_string(&session_json)
        .with_context(|| format!("read session file at {}", session_json.display()))?;
    let mut auth: TwitterAuth =
        serde_json::from_str(&raw).context("parse session JSON")?;
    auth.validate()
        .context("session file missing required fields")?;
    if auth.harvested_at_ms == 0 {
        auth.harvested_at_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
    }

    // Probe the DM inbox once to validate the session before persisting —
    // bad cookies fail fast here instead of at the first poll.
    let client = TwitterClient::new(auth.clone());
    match client.fetch_dm_inbox(None).await {
        Ok(dms) => info!(dm_count = dms.len(), "twitter session probe OK"),
        Err(e) => anyhow::bail!("session probe failed: {e}; aborting save"),
    }

    let repo_root = std::env::current_dir().context("current_dir")?;
    let out = twitter_default_auth_path(&repo_root);
    auth.save(&out)
        .with_context(|| format!("save auth to {}", out.display()))?;
    auth.save_to_keychain(augmentagent_auth::DEFAULT_ACCOUNT)
        .context("save auth to keychain (augmentagent/twitter/default)")?;
    println!(
        "twitter auth saved to {} + keychain (augmentagent/twitter/default)",
        out.display()
    );
    println!("account: @{} (id {})", auth.screen_name, auth.user_id);
    Ok(())
}

async fn run_twitter_post(
    store: Arc<Store>,
    text: String,
    reply_to: Option<String>,
    dry_run: bool,
) -> Result<()> {
    let repo_root = std::env::current_dir().context("current_dir")?;
    let auth = TwitterAuth::load_with_migration(&repo_root).context(
        "load twitter auth from keychain or legacy file — run `augmentagent twitter login`",
    )?;
    let api = Arc::new(TwitterClient::new(auth));
    let client = CreateTweetClient::new(api, store, dry_run);
    match client.create(&text, reply_to.as_deref(), &[]).await {
        Ok(out) => {
            println!("{out:?}");
            Ok(())
        }
        Err(e) => anyhow::bail!("twitter post failed: {e}"),
    }
}

async fn run_twitter_poll_once(cli: &Cli) -> Result<()> {
    let repo_root = std::env::current_dir().context("current_dir")?;
    let auth = TwitterAuth::load_with_migration(&repo_root)
        .context("load twitter auth from keychain or legacy file")?;
    let my_user_id = auth.user_id.clone();
    let api = Arc::new(TwitterClient::new(auth));
    let Some(wiki_root) = cli.wiki_dir.clone() else {
        anyhow::bail!("twitter poll-once needs --wiki-dir (close-friend pages live there)");
    };
    let trigger = TwitterFeedTrigger::new(api.clone(), wiki_root, my_user_id.clone());
    let cancel = CancellationToken::new();
    let items =
        augmentagent_channel_core::Trigger::next_work_items(&trigger, &cancel).await?;
    println!("feed: {} new tweet(s) from close friends", items.len());
    for it in &items {
        println!("  - {} {}", it.kind, it.external_id);
    }
    // Also surface a DM-inbox count (read-only smoke).
    let dm_src = TwitterDmSource::new(api, my_user_id);
    let dms = augmentagent_channel_core::InboundSource::fetch_new(&dm_src).await?;
    println!("dm inbox: {} new inbound DM(s)", dms.len());
    Ok(())
}

/// #14 — one-command operator validation harness. Replaces the manual
/// `/intercept` proxy session: load the harvested session, exercise every
/// documented endpoint, print the pass/fail grid that maps to the
/// `REQUIRES LIVE OPERATOR VALIDATION` flags in docs/twitter-protocol.md.
async fn run_twitter_validate(
    json: bool,
    allow_live: bool,
    allow_write: bool,
    probe_reply_to: Option<String>,
    probe_conversation_id: Option<String>,
) -> Result<()> {
    let repo_root = std::env::current_dir().context("current_dir")?;
    let auth = TwitterAuth::load_with_migration(&repo_root).context(
        "load twitter auth from keychain or legacy file — run `augmentagent twitter login` first",
    )?;
    let opts = TwitterValidateOptions {
        allow_live,
        allow_write,
        probe_reply_to,
        probe_conversation_id,
    };
    if !allow_live && std::env::var("AUGMENTAGENT_TWITTER_BASE_URL").is_err() {
        warn!(
            "twitter validate: MOCK-ONLY build — no live x.com call will be made. \
             Pass --allow-live on a real session for a sign-off run."
        );
    }
    if allow_write {
        warn!(
            "twitter validate: --allow-write set — live write probes are enabled \
             (CreateTweet / DM send will hit the wire if probe ids are given)"
        );
    }
    let report = twitter_validate_session(auth, opts).await;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report)
                .context("serialize validation report")?
        );
    } else {
        print!("{}", report.render_table());
    }
    if report.mock_only {
        // A mock-only run is informational, not a pass — but it's not a
        // failure either (nothing was actually probed). Exit 0 so a CI step
        // that just exercises the harness wiring doesn't break, while the
        // banner makes clear no sign-off was produced.
        Ok(())
    } else if report.all_passed {
        Ok(())
    } else {
        // Non-zero exit so a CI / scripted runbook step fails loudly.
        anyhow::bail!(
            "twitter validate: one or more checks failed — \
             keep the docs/twitter-protocol.md validation flags set"
        )
    }
}

// ================================================================
// Discord (issue #27)
// ================================================================

async fn run_discord_login(creds_json: PathBuf) -> Result<()> {
    use augmentagent_channel_discord_dm::{auth::default_creds_path, DiscordAuth, DiscordClient};
    let raw = std::fs::read_to_string(&creds_json)
        .with_context(|| format!("read creds file at {}", creds_json.display()))?;
    let auth: DiscordAuth = serde_json::from_str(&raw).context("parse discord creds JSON")?;
    auth.validate().context("creds missing required fields")?;

    // Probe GET /users/@me/channels to confirm the token is accepted before
    // we persist. Avoids saving a broken auth blob that'd fail at poll time.
    let client = DiscordClient::new(auth.clone()).context("build discord client")?;
    let dms = client
        .list_dm_channels()
        .await
        .context("token probe via /users/@me/channels failed")?;
    info!(dm_count = dms.len(), "discord token probe ok");

    auth.save_to_keychain()
        .context("save discord auth to keychain")?;

    // Also write the file to the vault/repo path so additional hosts mounting
    // the same vault auto-pick-up on next deploy. Skipped if the destination
    // is the source (writing to the same file we just read).
    let repo_root = std::env::current_dir().context("current_dir")?;
    let vault_path = default_creds_path(&repo_root);
    let mirrored = match (
        creds_json.canonicalize(),
        vault_path.canonicalize(),
    ) {
        (Ok(a), Ok(b)) if a == b => false,
        _ => {
            match auth.save(&vault_path) {
                Ok(()) => {
                    info!(to = %vault_path.display(), "discord creds mirrored to vault path");
                    true
                }
                Err(e) => {
                    warn!(
                        error = %e,
                        to = %vault_path.display(),
                        "vault mirror failed; keychain still saved"
                    );
                    false
                }
            }
        }
    };

    println!(
        "discord auth saved to keychain (augmentagent/discord/default)\nuser_id: {}\nvault mirror: {}",
        auth.user_id,
        if mirrored { vault_path.display().to_string() } else { "(skipped — source is already at vault path)".into() },
    );
    Ok(())
}

fn load_discord_client() -> Option<Arc<augmentagent_channel_discord_dm::DiscordClient>> {
    let repo_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    match augmentagent_channel_discord_dm::DiscordAuth::load_with_migration(&repo_root) {
        Ok(auth) => match augmentagent_channel_discord_dm::DiscordClient::new(auth) {
            Ok(c) => Some(Arc::new(c)),
            Err(e) => {
                warn!("discord client build failed: {e}");
                None
            }
        },
        Err(e) => {
            info!("discord auth not loaded: {e} (discord send disabled this run)");
            None
        }
    }
}

/// Best-effort GitHub PAT load for the approver. `None` ⇒ github outbound
/// disabled this run. Mirrors `load_discord_client` shape.
fn load_github_client() -> Option<Arc<augmentagent_channel_github::GithubClient>> {
    match load_any_github_auth() {
        Ok(auth) => match augmentagent_channel_github::GithubClient::new(auth) {
            Ok(c) => Some(Arc::new(c)),
            Err(e) => {
                warn!("github client build failed: {e}");
                None
            }
        },
        Err(e) => {
            info!("github auth not loaded: {e:#} (github outbound disabled this run)");
            None
        }
    }
}

/// Best-effort SocialAPI.ai client load for the approver (#244). `None` ⇒
/// socialapi outbound (comment-reply / DM-reply send on Approve) disabled this
/// run; the comment/DM pollers still surface approval cards, but Approve will
/// surface a `Failed` until the key is configured. Gated on
/// `SOCIALAPI_API_KEY` / keyring, mirroring `load_discord_client`.
fn load_socialapi_client(
    store: &Store,
) -> Option<Arc<augmentagent_channel_socialapi::SocialApiClient>> {
    match augmentagent_channel_socialapi::SocialApiAuth::load_with_store(store) {
        Ok(auth) => Some(Arc::new(augmentagent_channel_socialapi::SocialApiClient::new(
            auth,
        ))),
        Err(e) => {
            info!("socialapi auth not loaded: {e} (socialapi send disabled this run)");
            None
        }
    }
}

async fn run_discord_status(json: bool) -> Result<()> {
    let repo_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let auth = augmentagent_channel_discord_dm::DiscordAuth::load_with_migration(&repo_root);
    if json {
        match auth {
            Ok(a) => println!(
                "{}",
                serde_json::json!({
                    "connected": true,
                    "user_id": a.user_id,
                })
            ),
            Err(_) => println!("{}", serde_json::json!({ "connected": false })),
        }
    } else {
        match auth {
            Ok(a) => println!("discord connected: user_id={}", a.user_id),
            Err(e) => println!("discord not connected: {e}"),
        }
    }
    Ok(())
}

async fn run_discord_list_dms(json: bool) -> Result<()> {
    let client =
        load_discord_client().ok_or_else(|| anyhow::anyhow!("discord auth not configured"))?;
    let dms = client.list_dm_channels().await.context("list DMs")?;
    if json {
        println!("{}", serde_json::to_string(&dms_to_json(&dms))?);
    } else {
        println!("{} DM channels\n", dms.len());
        for d in &dms {
            let kind = if d.is_one_to_one() { "dm" } else { "group" };
            println!("  {}  [{}]  {}", d.id, kind, d.display_name());
        }
    }
    Ok(())
}

async fn run_discord_list_guilds(json: bool) -> Result<()> {
    let client =
        load_discord_client().ok_or_else(|| anyhow::anyhow!("discord auth not configured"))?;
    let guilds = client.list_guilds().await.context("list guilds")?;
    if json {
        let rows: Vec<_> = guilds
            .iter()
            .map(|g| serde_json::json!({ "id": g.id, "name": g.name }))
            .collect();
        println!("{}", serde_json::to_string(&rows)?);
    } else {
        println!("{} guilds\n", guilds.len());
        for g in &guilds {
            println!("  {}  {}", g.id, g.name);
        }
    }
    Ok(())
}

async fn run_discord_list_guild_channels(guild_id: String, json: bool) -> Result<()> {
    let client =
        load_discord_client().ok_or_else(|| anyhow::anyhow!("discord auth not configured"))?;
    let channels = client
        .list_guild_channels(&guild_id)
        .await
        .context("list guild channels")?;
    let text: Vec<_> = channels.iter().filter(|c| c.is_text()).collect();
    if json {
        let rows: Vec<_> = text
            .iter()
            .map(|c| serde_json::json!({ "id": c.id, "name": c.name }))
            .collect();
        println!("{}", serde_json::to_string(&rows)?);
    } else {
        println!("{} text channels in guild {}\n", text.len(), guild_id);
        for c in &text {
            println!("  {}  #{}", c.id, c.name);
        }
    }
    Ok(())
}

fn run_discord_subscribe(
    store: Arc<Store>,
    channel_id: String,
    mode: String,
    name: Option<String>,
) -> Result<()> {
    use augmentagent_store::SubscriptionMode;
    let parsed = SubscriptionMode::parse(&mode)
        .ok_or_else(|| anyhow::anyhow!("invalid mode: {mode}"))?;
    let display = name.unwrap_or_else(|| channel_id.clone());
    let sub = store
        .upsert_subscription(
            augmentagent_channel_discord_dm::PLATFORM,
            &channel_id,
            &display,
            parsed,
            None,
        )
        .context("upsert subscription")?;
    println!(
        "subscription id={} platform={} channel_id={} mode={} name={}",
        sub.id, sub.platform, sub.channel_id, sub.mode.as_str(), sub.display_name
    );
    Ok(())
}

fn run_discord_subscriptions(store: Arc<Store>, json: bool) -> Result<()> {
    let subs = store
        .list_active_subscriptions(augmentagent_channel_discord_dm::PLATFORM)
        .context("list subscriptions")?;
    if json {
        println!("{}", serde_json::to_string(&subs)?);
    } else {
        println!("{} active discord subscriptions\n", subs.len());
        for s in &subs {
            println!(
                "  {}  mode={}  channel={}  last_seen={:?}  name={}",
                s.id,
                s.mode.as_str(),
                s.channel_id,
                s.last_seen_message_id,
                s.display_name,
            );
        }
    }
    Ok(())
}

fn run_discord_unsubscribe(store: Arc<Store>, id: String) -> Result<()> {
    store
        .delete_subscription(&id)
        .context("delete subscription")?;
    println!("subscription {id} deactivated");
    Ok(())
}

fn dms_to_json(dms: &[augmentagent_channel_discord_dm::types::DmChannel]) -> Vec<serde_json::Value> {
    dms.iter()
        .map(|d| {
            serde_json::json!({
                "id": d.id,
                "type": d.channel_type,
                "display_name": d.display_name(),
                "is_one_to_one": d.is_one_to_one(),
            })
        })
        .collect()
}

fn build_discord_channel(
    cli: &Cli,
    store: Arc<Store>,
    broker: Arc<dyn ApprovalBroker>,
    dry_run: bool,
) -> Result<augmentagent_channel_discord_dm::DiscordChannel<FallbackReasoner>> {
    use augmentagent_channel_discord_dm::{DiscordAuth, DiscordChannel, DiscordChannelConfig};
    let repo_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let auth = DiscordAuth::load_with_migration(&repo_root).context(
        "load discord auth — run `augmentagent discord login --creds-json <file>` or place creds at default_creds_path",
    )?;
    let my_user_id = auth.user_id.clone();
    let client = Arc::new(
        augmentagent_channel_discord_dm::DiscordClient::new(auth)
            .context("build discord client")?,
    );
    let reasoner = build_reasoner();

    let (wiki_root, wiki_schema_path) = match &cli.wiki_dir {
        Some(root) => {
            let schema = cli
                .wiki_schema
                .clone()
                .unwrap_or_else(|| PathBuf::from("schema/wiki-skill.md"));
            (Some(root.clone()), Some(schema))
        }
        None => (None, None),
    };
    let identity_index = wiki_root
        .as_ref()
        .and_then(|root| {
            let layout = augmentagent_wiki::WikiLayout::new(root.clone());
            augmentagent_wiki::IdentityIndex::build(&layout).ok().map(Arc::new)
        });

    let config = DiscordChannelConfig {
        poll_interval: Duration::from_secs(augmentagent_channel_discord_dm::channel::DEFAULT_POLL_SECS),
        dry_run,
        wiki_root,
        wiki_schema_path,
        skill_dir: PathBuf::from("skills/discord-triage"),
    };
    Ok(DiscordChannel::new(
        store,
        client,
        reasoner,
        broker,
        my_user_id,
        config,
        identity_index,
    ))
}

// ================================================================
// Slack (issue #7)
// ================================================================

async fn run_slack_login(store: Arc<Store>, auth_json: PathBuf) -> Result<()> {
    use augmentagent_channel_slack::{SlackAuth, SlackClient};
    let raw = std::fs::read_to_string(&auth_json)
        .with_context(|| format!("read slack auth file at {}", auth_json.display()))?;
    let auth: SlackAuth = serde_json::from_str(&raw).context("parse slack auth JSON")?;
    auth.validate().context("missing required fields")?;

    // Probe a lightweight Composio call to confirm credentials work before persisting.
    let client = SlackClient::new(auth.clone()).context("build slack client")?;
    let convs = client
        .list_conversations("im", 1)
        .await
        .context("probe via SLACK_LIST_CONVERSATIONS failed")?;
    info!(conversations_reachable = convs.len(), "slack auth probe ok");

    auth.save_to_keychain()
        .context("save slack auth to keychain")?;
    store
        .upsert_slack_workspace(
            &auth.team_id,
            &auth.team_name,
            &auth.entity_id,
            &auth.connection_id,
            &auth.user_id,
        )
        .context("upsert slack workspace row")?;
    println!(
        "slack auth saved to keychain (augmentagent/slack/{})\nteam:    {} ({})\nuser_id: {}",
        auth.team_id, auth.team_name, auth.team_id, auth.user_id
    );
    Ok(())
}

/// Persist a Slack auth bundle handed in from the dashboard OAuth callback.
///
/// Takes only the Composio handles. Resolves `team_id`/`team_name`/`user_id`
/// server-side via SLACK_FETCH_TEAM_INFO + an auth-test action. Mirrors
/// Orchid's pattern: no channel-list probe at OAuth time, just trust
/// Composio's ACTIVE status and learn the workspace metadata via the API.
async fn run_slack_persist_auth(
    store: Arc<Store>,
    entity_id: String,
    connection_id: String,
    composio_api_key: String,
) -> Result<()> {
    use augmentagent_channel_slack::{SlackAuth, SlackClient};
    // Build a "probe" auth — only entity_id + composio_api_key matter for the
    // execute() path — and use it to learn the workspace metadata.
    let probe = SlackAuth {
        entity_id: entity_id.clone(),
        connection_id: connection_id.clone(),
        team_id: String::new(),
        team_name: String::new(),
        user_id: String::new(),
        composio_api_key: composio_api_key.clone(),
    };
    probe
        .validate_for_execute()
        .context("persist-auth: entity_id and composio_api_key required")?;
    let client = SlackClient::new(probe).context("build slack client")?;
    let team = client
        .fetch_team_info()
        .await
        .context("SLACK_FETCH_TEAM_INFO probe failed — connection may not be ACTIVE yet")?;
    // user_id is best-effort; missing just disables self-message filtering.
    let user_id = client.fetch_authed_user_id().await.unwrap_or(None).unwrap_or_default();

    let auth = SlackAuth {
        entity_id,
        connection_id,
        team_id: team.team_id.clone(),
        team_name: team.team_name.clone(),
        user_id: user_id.clone(),
        composio_api_key,
    };
    auth.validate()
        .context("persist-auth: validation failed after team probe")?;
    auth.save_to_keychain()
        .context("save slack auth to keychain")?;
    // Verify round-trip: catches silent Keychain backend issues (e.g. Linux
    // Secret Service unavailable) where save reports OK but read fails.
    augmentagent_channel_slack::SlackAuth::load_for_team(&auth.team_id)
        .with_context(|| {
            format!(
                "Keychain round-trip failed for team {} — save reported ok but read returned err. \
                 On Linux this usually means Secret Service (gnome-keyring/kwallet) isn't running for this user session.",
                auth.team_id
            )
        })?;
    store
        .upsert_slack_workspace(
            &auth.team_id,
            &auth.team_name,
            &auth.entity_id,
            &auth.connection_id,
            &auth.user_id,
        )
        .context("upsert slack workspace row")?;
    println!(
        "{}",
        serde_json::json!({
            "ok": true,
            "team_id": auth.team_id,
            "team_name": auth.team_name,
            "user_id": auth.user_id,
        })
    );
    Ok(())
}

fn run_slack_workspaces(store: Arc<Store>, json: bool) -> Result<()> {
    let workspaces = store
        .list_active_slack_workspaces()
        .context("list slack workspaces")?;
    if json {
        println!("{}", serde_json::to_string(&workspaces)?);
    } else {
        println!("{} slack workspace(s)\n", workspaces.len());
        for w in &workspaces {
            println!("  {}  {}  user={}", w.team_id, w.team_name, w.user_id);
        }
    }
    Ok(())
}

fn run_slack_remove_workspace(store: Arc<Store>, team_id: String) -> Result<()> {
    use augmentagent_channel_slack::SlackAuth;
    // Hard delete: drop both the Keychain slot and the workspace row so a
    // subsequent OAuth reconnect creates clean state instead of reactivating
    // a row that may have been written by an older buggy parser.
    // Subscriptions tied to this workspace get soft-deactivated.
    let _ = SlackAuth::delete_from_keychain(&team_id);
    store
        .delete_slack_workspace(&team_id)
        .context("delete slack workspace row")?;
    println!("slack workspace {team_id} disconnected (hard delete)");
    Ok(())
}

fn run_slack_reset(store: Arc<Store>, confirm: bool) -> Result<()> {
    use augmentagent_channel_slack::SlackAuth;
    if !confirm {
        anyhow::bail!(
            "refusing to reset Slack state without --confirm true. \
             This drops every workspace row, every Slack subscription, and \
             every Keychain slot. Pass --confirm true to proceed."
        );
    }
    let workspaces = store
        .list_active_slack_workspaces()
        .context("list workspaces for reset")?;
    let mut keychain_dropped = 0;
    let mut rows_dropped = 0;
    for ws in workspaces {
        let _ = SlackAuth::delete_from_keychain(&ws.team_id);
        keychain_dropped += 1;
        store
            .delete_slack_workspace(&ws.team_id)
            .with_context(|| format!("delete workspace {}", ws.team_id))?;
        rows_dropped += 1;
    }
    // Also drop the legacy single-slot Keychain entry left over from
    // pre-multi-workspace days. Use the team-keyed delete with literal
    // "default" since the legacy slot was at augmentagent/slack/default.
    let _ = SlackAuth::delete_from_keychain("default");
    println!(
        "slack reset: dropped {} keychain slot(s), {} workspace row(s). \
         Reconnect via dashboard.",
        keychain_dropped, rows_dropped
    );
    Ok(())
}

/// Build the per-workspace Slack client map consumed by `ReplyApprover`.
/// Mirrors `SlackChannel::load_workspace_clients` — loads every active
/// `slack_workspaces` row's Keychain entry and falls back to the legacy
/// `augmentagent/slack/default` slot when the table is empty.
fn load_slack_clients(
    store: &Store,
) -> std::collections::HashMap<String, Arc<augmentagent_channel_slack::SlackClient>> {
    use augmentagent_channel_slack::{SlackAuth, SlackClient};
    let mut map = std::collections::HashMap::new();
    let workspaces = match store.list_active_slack_workspaces() {
        Ok(w) => w,
        Err(e) => {
            warn!("list_active_slack_workspaces failed: {e:#}");
            return map;
        }
    };
    if workspaces.is_empty() {
        match SlackAuth::load_from_default_slot() {
            Ok(auth) => {
                let team_id = auth.team_id.clone();
                if let Ok(c) = SlackClient::new(auth) {
                    map.insert(team_id, Arc::new(c));
                    info!("slack: using legacy default-slot auth (one workspace)");
                }
            }
            Err(e) => {
                info!("slack auth not loaded: {e} (slack send disabled this run)");
            }
        }
        return map;
    }
    for ws in workspaces {
        match SlackAuth::load_for_team(&ws.team_id) {
            Ok(auth) => match SlackClient::new(auth) {
                Ok(c) => {
                    map.insert(ws.team_id.clone(), Arc::new(c));
                }
                Err(e) => warn!(team_id = %ws.team_id, "slack client build failed: {e}"),
            },
            Err(e) => warn!(team_id = %ws.team_id, "slack auth load failed: {e}"),
        }
    }
    map
}

async fn run_slack_list_conversations(
    store: Arc<Store>,
    team_id: Option<String>,
    types: String,
    limit: u32,
    json: bool,
) -> Result<()> {
    let client = match load_single_slack_client(&store, team_id.as_deref()) {
        Some(c) => c,
        None => {
            // Diagnose so the user knows whether to reconnect via dashboard
            // (Keychain slot missing) or pass --team-id (multi-workspace).
            let msg = if let Some(tid) = team_id.as_deref() {
                let row = store.get_slack_workspace_by_team(tid)?;
                if row.is_some() {
                    format!(
                        "workspace {tid} is registered in slack_workspaces but its Keychain slot \
                         is missing or unreadable. Click 'Disconnect' on that workspace in the \
                         dashboard, then re-connect to refresh credentials."
                    )
                } else {
                    format!("workspace {tid} not connected — connect it via the dashboard")
                }
            } else {
                let workspaces = store.list_active_slack_workspaces()?;
                match workspaces.len() {
                    0 => "no slack workspaces connected — connect one via the dashboard".into(),
                    1 => "single workspace registered but its Keychain slot is missing — disconnect + reconnect via the dashboard".into(),
                    _ => "multiple workspaces registered — pass --team-id <T...> to disambiguate".into(),
                }
            };
            anyhow::bail!(msg);
        }
    };
    let convs = client
        .list_conversations(&types, limit)
        .await
        .context("list conversations")?;
    if json {
        let rows: Vec<_> = convs
            .iter()
            .map(|c| {
                serde_json::json!({
                    "id": c.id,
                    "name": c.name,
                    "display_name": c.display_name(),
                    "is_im": c.is_im,
                    "is_mpim": c.is_mpim,
                    "is_private": c.is_private,
                })
            })
            .collect();
        println!("{}", serde_json::to_string(&rows)?);
    } else {
        println!("{} conversations\n", convs.len());
        for c in &convs {
            let kind = if c.is_im {
                "dm"
            } else if c.is_mpim {
                "group"
            } else if c.is_private {
                "private"
            } else {
                "public"
            };
            println!("  {}  [{}]  {}", c.id, kind, c.display_name());
        }
    }
    Ok(())
}

fn run_slack_subscribe(
    store: Arc<Store>,
    channel_id: String,
    mode: String,
    name: Option<String>,
    team_id: Option<String>,
) -> Result<()> {
    use augmentagent_store::SubscriptionMode;
    let parsed = SubscriptionMode::parse(&mode)
        .ok_or_else(|| anyhow::anyhow!("invalid mode: {mode}"))?;
    // Default to the sole configured workspace when --team-id is omitted;
    // fail loudly if there are multiple so the user can't accidentally bind
    // the sub to the wrong workspace.
    let resolved_team = match team_id {
        Some(t) => t,
        None => {
            let workspaces = store
                .list_active_slack_workspaces()
                .context("list slack workspaces")?;
            match workspaces.as_slice() {
                [w] => w.team_id.clone(),
                [] => anyhow::bail!(
                    "no slack workspaces connected — run `augmentagent slack login` or connect via dashboard"
                ),
                _ => anyhow::bail!(
                    "multiple slack workspaces connected — pass --team-id <T...>"
                ),
            }
        }
    };
    let display = name.unwrap_or_else(|| channel_id.clone());
    let sub = store
        .upsert_subscription(
            augmentagent_channel_slack::PLATFORM,
            &channel_id,
            &display,
            parsed,
            Some(&resolved_team),
        )
        .context("upsert subscription")?;
    println!(
        "subscription id={} platform={} channel_id={} mode={} name={} account_id={}",
        sub.id,
        sub.platform,
        sub.channel_id,
        sub.mode.as_str(),
        sub.display_name,
        resolved_team,
    );
    Ok(())
}

fn run_slack_subscriptions(store: Arc<Store>, json: bool) -> Result<()> {
    let subs = store
        .list_active_subscriptions(augmentagent_channel_slack::PLATFORM)
        .context("list subscriptions")?;
    if json {
        println!("{}", serde_json::to_string(&subs)?);
    } else {
        println!("{} active slack subscriptions\n", subs.len());
        for s in &subs {
            println!(
                "  {}  mode={}  channel={}  last_seen={:?}  name={}",
                s.id,
                s.mode.as_str(),
                s.channel_id,
                s.last_seen_message_id,
                s.display_name,
            );
        }
    }
    Ok(())
}

fn run_slack_unsubscribe(store: Arc<Store>, id: String) -> Result<()> {
    store
        .delete_subscription(&id)
        .context("delete subscription")?;
    println!("subscription {id} deactivated");
    Ok(())
}

fn build_slack_channel(
    cli: &Cli,
    store: Arc<Store>,
    broker: Arc<dyn ApprovalBroker>,
    dry_run: bool,
) -> Result<augmentagent_channel_slack::SlackChannel<FallbackReasoner>> {
    use augmentagent_channel_slack::{SlackChannel, SlackChannelConfig};
    let reasoner = build_reasoner();

    let (wiki_root, wiki_schema_path) = match &cli.wiki_dir {
        Some(root) => {
            let schema = cli
                .wiki_schema
                .clone()
                .unwrap_or_else(|| PathBuf::from("schema/wiki-skill.md"));
            (Some(root.clone()), Some(schema))
        }
        None => (None, None),
    };
    let identity_index = wiki_root.as_ref().and_then(|root| {
        let layout = augmentagent_wiki::WikiLayout::new(root.clone());
        augmentagent_wiki::IdentityIndex::build(&layout)
            .ok()
            .map(Arc::new)
    });

    let config = SlackChannelConfig {
        poll_interval: Duration::from_secs(augmentagent_channel_slack::channel::DEFAULT_POLL_SECS),
        dry_run,
        wiki_root,
        wiki_schema_path,
        skill_dir: PathBuf::from("skills/slack-triage"),
    };
    Ok(SlackChannel::new(
        store,
        reasoner,
        broker,
        config,
        identity_index,
    ))
}

/// Load a single SlackClient, picking by explicit `team_id` when given, or
/// falling back to the sole configured workspace (or legacy default slot).
fn load_single_slack_client(
    store: &Store,
    team_id: Option<&str>,
) -> Option<Arc<augmentagent_channel_slack::SlackClient>> {
    use augmentagent_channel_slack::{SlackAuth, SlackClient};
    if let Some(tid) = team_id {
        let auth = SlackAuth::load_for_team(tid).ok()?;
        return SlackClient::new(auth).ok().map(Arc::new);
    }
    let clients = load_slack_clients(store);
    if clients.len() == 1 {
        return clients.into_values().next();
    }
    if clients.is_empty() {
        return None;
    }
    warn!("multiple slack workspaces configured; pass --team-id to disambiguate");
    None
}

// ---------------------------------------------------------------------------
// Telegram-bot CLI handlers (#74)
// ---------------------------------------------------------------------------

/// `telegram-bot login --token …` — validates the token via getMe, persists
/// to keychain, and writes/updates the `telegram_bots` row.
async fn run_telegram_bot_login(store: Arc<Store>, token: String) -> Result<()> {
    use augmentagent_channel_telegram_bot::{TelegramBotAuth, TelegramBotClient};
    let token = token.trim().to_string();
    if !token.contains(':') {
        anyhow::bail!("token doesn't look like a BotFather token (`<id>:<secret>`)");
    }
    // Probe getMe before persisting so a fat-fingered token surfaces now,
    // not on first poll.
    let probe = TelegramBotClient::new(token.clone()).context("build telegram bot client")?;
    let me = probe.get_me().await.context("getMe probe failed (token invalid?)")?;
    if me.username.is_empty() {
        anyhow::bail!("getMe returned an empty username — refusing to persist");
    }
    // owner_chat_id seed: read from env so `telegram-bot login` works
    // headless. The user resolves the value once via @userinfobot in
    // Telegram and passes it as `AUGMENTAGENT_TELEGRAM_OWNER_CHAT_ID`.
    let owner_chat_id: i64 = std::env::var("AUGMENTAGENT_TELEGRAM_OWNER_CHAT_ID")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if owner_chat_id == 0 {
        warn!(
            "AUGMENTAGENT_TELEGRAM_OWNER_CHAT_ID not set — owner-DM auto-subscribe disabled. \
             Set it to your numeric Telegram user id (DM @userinfobot) and re-run login."
        );
    }
    let auth = TelegramBotAuth {
        bot_token: token,
        bot_username: me.username.clone(),
        bot_id: me.id,
        owner_chat_id: owner_chat_id.max(1),
    };
    auth.save_to_keychain().context("save telegram bot auth to keychain")?;
    // Round-trip read so silent keychain failures (Linux Secret Service
    // unavailable) surface here.
    augmentagent_channel_telegram_bot::TelegramBotAuth::load_from_keychain(&auth.bot_username)
        .context("keychain round-trip after save failed")?;
    store
        .upsert_telegram_bot(auth.bot_id, &auth.bot_username, auth.owner_chat_id)
        .context("upsert telegram_bots row")?;
    println!(
        "telegram bot saved to keychain (augmentagent/telegram-bot/{})\nbot:           @{} (id {})\nowner_chat_id: {}",
        auth.bot_username, auth.bot_username, auth.bot_id, auth.owner_chat_id
    );
    Ok(())
}

fn run_telegram_bot_bots(store: Arc<Store>, json: bool) -> Result<()> {
    let bots = store.list_active_telegram_bots().context("list bots")?;
    if json {
        println!("{}", serde_json::to_string(&bots)?);
    } else {
        println!("{} active telegram bot(s)\n", bots.len());
        for b in &bots {
            println!(
                "  @{}  id={}  owner_chat_id={}  last_update_id={}",
                b.bot_username, b.bot_id, b.owner_chat_id, b.last_update_id
            );
        }
    }
    Ok(())
}

fn run_telegram_bot_remove(store: Arc<Store>, bot_username: String) -> Result<()> {
    use augmentagent_channel_telegram_bot::TelegramBotAuth;
    let row = store
        .get_telegram_bot_by_username(&bot_username)
        .context("look up bot")?
        .ok_or_else(|| anyhow::anyhow!("no telegram bot row for @{bot_username}"))?;
    // Best-effort keychain delete — proceed even if the slot is already gone.
    if let Err(e) = TelegramBotAuth::delete_from_keychain(&bot_username) {
        warn!(bot_username, "telegram bot keychain delete failed: {e}");
    }
    store.delete_telegram_bot(row.bot_id).context("delete bot row")?;
    println!("telegram bot @{bot_username} removed (subscriptions deactivated)");
    Ok(())
}

fn run_telegram_bot_list_chats(
    store: Arc<Store>,
    bot_username: Option<String>,
    json: bool,
) -> Result<()> {
    // Bot API exposes no `getDialogs`; we surface the union of (a) explicit
    // subscriptions and (b) the bot's own owner_chat_id. Production clients
    // should prefer the dashboard which mines `emails` for ad-hoc chat ids.
    let bots = match bot_username.as_deref() {
        Some(u) => store
            .get_telegram_bot_by_username(u)?
            .map(|b| vec![b])
            .unwrap_or_default(),
        None => store.list_active_telegram_bots()?,
    };
    let subs = store.list_active_subscriptions("telegram")?;
    let mut rows: Vec<serde_json::Value> = Vec::new();
    for b in &bots {
        rows.push(serde_json::json!({
            "kind": "owner_dm",
            "bot_username": b.bot_username,
            "bot_id": b.bot_id,
            "chat_id": b.owner_chat_id,
            "label": format!("DM with owner ({})", b.owner_chat_id),
        }));
    }
    for s in &subs {
        if let Some(u) = bot_username.as_deref() {
            // Filter to subs tied to this bot's id.
            let matches = bots
                .iter()
                .any(|b| b.bot_username == u && Some(b.bot_id.to_string()) == s.account_id);
            if !matches {
                continue;
            }
        }
        rows.push(serde_json::json!({
            "kind": "subscription",
            "subscription_id": s.id,
            "chat_id": s.channel_id,
            "label": s.display_name,
            "mode": s.mode.as_str(),
            "account_id": s.account_id,
        }));
    }
    if json {
        println!("{}", serde_json::to_string(&rows)?);
    } else {
        println!("{} known chat row(s)\n", rows.len());
        for r in &rows {
            println!("  {}", r);
        }
    }
    Ok(())
}

fn run_telegram_bot_subscribe(
    store: Arc<Store>,
    chat_id: String,
    mode: String,
    name: Option<String>,
    bot_username: Option<String>,
) -> Result<()> {
    use augmentagent_store::SubscriptionMode;
    let parsed = SubscriptionMode::parse(&mode)
        .ok_or_else(|| anyhow::anyhow!("invalid mode: {mode}"))?;
    // Resolve account_id (= bot_id) from --bot-username, or from the sole
    // configured bot when omitted; bail if there are multiple bots and the
    // user didn't disambiguate.
    let resolved_bot_id = match bot_username {
        Some(u) => store
            .get_telegram_bot_by_username(&u)?
            .ok_or_else(|| anyhow::anyhow!("no telegram bot for @{u}"))?
            .bot_id,
        None => {
            let bots = store
                .list_active_telegram_bots()
                .context("list telegram bots")?;
            match bots.as_slice() {
                [b] => b.bot_id,
                [] => anyhow::bail!(
                    "no telegram bots connected — run `augmentagent telegram-bot login --token …`"
                ),
                _ => anyhow::bail!(
                    "multiple telegram bots connected — pass --bot-username @<name>"
                ),
            }
        }
    };
    let display = name.unwrap_or_else(|| chat_id.clone());
    let sub = store
        .upsert_subscription(
            augmentagent_channel_telegram_bot::PLATFORM,
            &chat_id,
            &display,
            parsed,
            Some(&resolved_bot_id.to_string()),
        )
        .context("upsert subscription")?;
    println!(
        "subscription id={} platform={} chat_id={} mode={} name={} account_id={}",
        sub.id,
        sub.platform,
        sub.channel_id,
        sub.mode.as_str(),
        sub.display_name,
        resolved_bot_id,
    );
    Ok(())
}

fn run_telegram_bot_subscriptions(store: Arc<Store>, json: bool) -> Result<()> {
    let subs = store
        .list_active_subscriptions(augmentagent_channel_telegram_bot::PLATFORM)
        .context("list subscriptions")?;
    if json {
        println!("{}", serde_json::to_string(&subs)?);
    } else {
        println!("{} active telegram subscriptions\n", subs.len());
        for s in &subs {
            println!(
                "  {}  mode={}  chat_id={}  bot_id={:?}  name={}",
                s.id,
                s.mode.as_str(),
                s.channel_id,
                s.account_id,
                s.display_name,
            );
        }
    }
    Ok(())
}

fn run_telegram_bot_unsubscribe(store: Arc<Store>, id: String) -> Result<()> {
    store
        .delete_subscription(&id)
        .context("delete subscription")?;
    println!("subscription {id} deactivated");
    Ok(())
}

fn build_telegram_bot_channel(
    cli: &Cli,
    store: Arc<Store>,
    broker: Arc<dyn ApprovalBroker>,
    dry_run: bool,
) -> Result<augmentagent_channel_telegram_bot::TelegramBotChannel<FallbackReasoner>> {
    use augmentagent_channel_telegram_bot::{TelegramBotChannel, TelegramBotChannelConfig};
    let reasoner = build_reasoner();

    let (wiki_root, wiki_schema_path) = match &cli.wiki_dir {
        Some(root) => {
            let schema = cli
                .wiki_schema
                .clone()
                .unwrap_or_else(|| PathBuf::from("schema/wiki-skill.md"));
            (Some(root.clone()), Some(schema))
        }
        None => (None, None),
    };
    let identity_index = wiki_root.as_ref().and_then(|root| {
        let layout = augmentagent_wiki::WikiLayout::new(root.clone());
        augmentagent_wiki::IdentityIndex::build(&layout)
            .ok()
            .map(Arc::new)
    });

    let config = TelegramBotChannelConfig {
        poll_interval: Duration::from_secs(
            augmentagent_channel_telegram_bot::channel::DEFAULT_POLL_SECS,
        ),
        dry_run,
        wiki_root,
        wiki_schema_path,
        skill_dir: PathBuf::from("skills/telegram-triage"),
        // PollOnce in dry-run mode should never block on long-poll — short
        // poll lets the CLI exit cleanly even when the inbox is empty.
        long_poll_secs: if dry_run { 0 } else { augmentagent_channel_telegram_bot::api::DEFAULT_LONG_POLL_SECS },
    };
    Ok(TelegramBotChannel::new(
        store,
        reasoner,
        broker,
        config,
        identity_index,
    ))
}

/// Build the per-bot Telegram client map consumed by `ReplyApprover`.
/// Mirrors `load_slack_clients` — loads every active `telegram_bots` row's
/// keychain entry and yields a `bot_id → Arc<TelegramBotClient>` map.
fn load_telegram_bot_clients(
    store: &Store,
) -> std::collections::HashMap<i64, Arc<augmentagent_channel_telegram_bot::TelegramBotClient>> {
    use augmentagent_channel_telegram_bot::{TelegramBotAuth, TelegramBotClient};
    let mut map = std::collections::HashMap::new();
    let bots = match store.list_active_telegram_bots() {
        Ok(b) => b,
        Err(e) => {
            warn!("list_active_telegram_bots failed: {e:#}");
            return map;
        }
    };
    if bots.is_empty() {
        info!("no telegram bots configured; telegram outbound disabled this run");
        return map;
    }
    for bot in bots {
        match TelegramBotAuth::load_with_file_fallback(&bot.bot_username) {
            Ok(auth) => match TelegramBotClient::new(auth.bot_token) {
                Ok(c) => {
                    map.insert(bot.bot_id, Arc::new(c));
                }
                Err(e) => warn!(
                    bot_username = %bot.bot_username,
                    "telegram client build failed: {e}"
                ),
            },
            Err(e) => warn!(
                bot_username = %bot.bot_username,
                "telegram auth load failed: {e}"
            ),
        }
    }
    map
}

/// Compile-fences to prove prefix constant is referenced (silence dead-code
/// warning in the unlikely event it's not pulled in elsewhere).
#[allow(dead_code)]
const _LINKEDIN_PREFIX: &str = ACCOUNT_PREFIX;

// ================================================================
// GitHub (issue #49)
// ================================================================

/// Validate the PAT against `GET /user`, then persist into the keyring slot
/// `augmentagent/github/<login>` using the *server-confirmed* login (the
/// `--login` arg is only a safety hint we cross-check).
async fn run_github_login(token: String, login_hint: String) -> Result<()> {
    use augmentagent_channel_github::api::whoami;
    use augmentagent_channel_github::auth::GithubAuth;
    if token.trim().is_empty() {
        anyhow::bail!("--token is empty");
    }
    let resolved = whoami(&token).await.context("validate PAT via GET /user")?;
    if !login_hint.is_empty() && !login_hint.eq_ignore_ascii_case(&resolved) {
        warn!(
            "--login {login_hint} does not match server-reported login {resolved}; using {resolved}"
        );
    }
    let auth = GithubAuth {
        username: resolved.clone(),
        token,
        fetched_at_ms: chrono::Utc::now().timestamp_millis(),
    };
    auth.save_to_keychain()
        .context("save github auth to keychain")?;
    println!("github auth saved to keychain (augmentagent/github/{resolved})");
    Ok(())
}

fn run_github_subscribe(store: Arc<Store>, repo: String, mode: String) -> Result<()> {
    use augmentagent_store::SubscriptionMode;
    let parsed =
        SubscriptionMode::parse(&mode).ok_or_else(|| anyhow::anyhow!("invalid mode: {mode}"))?;
    if !repo.contains('/') {
        anyhow::bail!("repo must be `<owner>/<repo>`, got {repo:?}");
    }
    let normalized = repo.to_ascii_lowercase();
    let sub = store
        .upsert_subscription(
            augmentagent_channel_github::PLATFORM,
            &normalized,
            &repo,
            parsed,
            None,
        )
        .context("upsert github subscription")?;
    println!(
        "subscription id={} platform={} repo={} mode={}",
        sub.id,
        sub.platform,
        sub.channel_id,
        sub.mode.as_str()
    );
    Ok(())
}

fn run_github_subscriptions(store: Arc<Store>, json: bool) -> Result<()> {
    let subs = store
        .list_active_subscriptions(augmentagent_channel_github::PLATFORM)
        .context("list github subscriptions")?;
    if json {
        println!("{}", serde_json::to_string(&subs)?);
    } else {
        println!("{} active github subscriptions\n", subs.len());
        for s in &subs {
            println!(
                "  {}  mode={}  repo={}  display={}",
                s.id,
                s.mode.as_str(),
                s.channel_id,
                s.display_name
            );
        }
    }
    Ok(())
}

fn run_github_unsubscribe(store: Arc<Store>, id: String) -> Result<()> {
    store.delete_subscription(&id).context("delete subscription")?;
    println!("subscription {id} deactivated");
    Ok(())
}

// ================================================================
// Deftform / Deft (#160) — operator-facing `login | status | logout`
// onboarding verbs. Mirrors `github login`: validate via whoami() before
// persisting; never write a token that doesn't auth. The crate itself
// stays inert in production until both AUGMENTAGENT_DEFT_ENABLED=1 (set
// by the systemd unit) AND a token is stored.
// ================================================================

/// Read a token from stdin. Stubbable in tests so the prompt path is
/// exercised without touching the real stdin.
trait DeftTokenReader {
    fn read_token(&mut self) -> Result<String>;
}

/// Production reader: one line from stdin, trimmed. Used when `--token`
/// is not provided.
struct StdinTokenReader;
impl DeftTokenReader for StdinTokenReader {
    fn read_token(&mut self) -> Result<String> {
        use std::io::{BufRead, Write};
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        writeln!(
            out,
            "Paste your Deftform workspace access token (https://deftform.com/settings/api):"
        )?;
        out.flush()?;
        let stdin = std::io::stdin();
        let mut line = String::new();
        stdin.lock().read_line(&mut line).context("read token from stdin")?;
        Ok(line.trim().to_string())
    }
}

/// Validate a freshly-pasted token via `DeftClient::whoami()` and, on
/// success, persist via `DeftAuth` into `augmentagent/deft/<workspace_id>`.
/// Mirrors `run_github_login`.
///
/// Implementation note: every method on `DeftClient` is gated on
/// [`augmentagent_channel_deft::deft_enabled`]. Login itself is an explicit
/// arming action, so we set the gate env var for the duration of this
/// process to perform the probe. The systemd unit pins
/// `AUGMENTAGENT_DEFT_ENABLED=1` separately for the daemon (see
/// `scripts/install-autostart.sh`); the two are independent.
async fn run_deft_login(
    token: Option<String>,
    base_url_override: Option<String>,
    reader: &mut dyn DeftTokenReader,
) -> Result<()> {
    use augmentagent_channel_deft::api::{DeftApi, DeftClient};
    use augmentagent_channel_deft::auth::{DeftAuth, DEFAULT_BASE_URL};

    let token = match token {
        Some(t) if !t.trim().is_empty() => t.trim().to_string(),
        _ => {
            let pasted = reader.read_token()?;
            if pasted.trim().is_empty() {
                anyhow::bail!("token is empty");
            }
            pasted
        }
    };
    let base_url = base_url_override.unwrap_or_else(|| DEFAULT_BASE_URL.to_string());

    // Arm the gate just for this probe — login is an explicit operator
    // action and the gate is per-process env. Restored on the way out so
    // we don't pollute the running shell's environment more than needed
    // (the systemd unit pins this independently anyway).
    let prior_gate = std::env::var("AUGMENTAGENT_DEFT_ENABLED").ok();
    std::env::set_var("AUGMENTAGENT_DEFT_ENABLED", "1");

    // Probe DeftAuth — workspace_id is a placeholder until whoami() tells
    // us the server-confirmed id. We never persist this probe.
    let probe = DeftAuth {
        workspace_id: "probe".to_string(),
        token: token.clone(),
        base_url: base_url.clone(),
        fetched_at_ms: chrono::Utc::now().timestamp_millis(),
    };
    let client = DeftClient::new(probe).context("build deft http client")?;
    let whoami_result = client.whoami().await;

    // Restore the gate env var to its prior state so the prompt-driven
    // login does not silently arm the next process the user launches
    // from this shell. (Daemon arming is handled by the systemd unit.)
    match prior_gate {
        Some(v) => std::env::set_var("AUGMENTAGENT_DEFT_ENABLED", v),
        None => std::env::remove_var("AUGMENTAGENT_DEFT_ENABLED"),
    }

    let workspace_id = match whoami_result {
        Ok(id) if !id.is_empty() => id,
        Ok(_) => {
            anyhow::bail!(
                "deft whoami succeeded but returned an empty workspace id; \
                 not persisting (token may be valid but the workspace endpoint \
                 envelope is unrecognized — see docs/deft-protocol.md §2)"
            );
        }
        Err(e) => {
            anyhow::bail!("deft token rejected by whoami(): {e}");
        }
    };

    let auth = DeftAuth {
        workspace_id: workspace_id.clone(),
        token,
        base_url,
        fetched_at_ms: chrono::Utc::now().timestamp_millis(),
    };
    auth.save_to_keychain()
        .context("save deft auth to keychain")?;
    println!(
        "deft auth saved to keychain (augmentagent/deft/{workspace_id})"
    );
    Ok(())
}

/// Pick the workspace id to operate on. Priority: explicit `--workspace-id`,
/// then `AUGMENTAGENT_DEFT_WORKSPACE_ID` env. Errors when neither is set —
/// Linux Secret Service can't enumerate slots, so the operator must name
/// the workspace (same constraint as `github login`).
fn resolve_deft_workspace_id(workspace_id: Option<String>) -> Result<String> {
    if let Some(w) = workspace_id {
        if !w.trim().is_empty() {
            return Ok(w.trim().to_string());
        }
    }
    if let Ok(env) = std::env::var("AUGMENTAGENT_DEFT_WORKSPACE_ID") {
        if !env.trim().is_empty() {
            return Ok(env.trim().to_string());
        }
    }
    anyhow::bail!(
        "workspace id required (pass --workspace-id <id> or set \
         AUGMENTAGENT_DEFT_WORKSPACE_ID); Linux Secret Service cannot \
         enumerate slots"
    );
}

async fn run_deft_status(
    workspace_id: Option<String>,
    offline: bool,
    json: bool,
) -> Result<()> {
    use augmentagent_channel_deft::api::{DeftApi, DeftClient};
    use augmentagent_channel_deft::auth::DeftAuth;

    let workspace_id = resolve_deft_workspace_id(workspace_id)?;
    let auth = match DeftAuth::load_for_workspace(&workspace_id) {
        Ok(a) => a,
        Err(e) => {
            if json {
                println!(
                    "{}",
                    serde_json::json!({
                        "connected": false,
                        "workspace_id": workspace_id,
                        "error": e.to_string(),
                    })
                );
            } else {
                println!("deft not connected (workspace_id={workspace_id}): {e}");
            }
            return Ok(());
        }
    };

    if offline {
        if json {
            println!(
                "{}",
                serde_json::json!({
                    "connected": true,
                    "workspace_id": auth.workspace_id,
                    "base_url": auth.base_url,
                    "fetched_at_ms": auth.fetched_at_ms,
                    "reachable": null,
                })
            );
        } else {
            println!(
                "deft token present (workspace_id={}, base_url={}, fetched_at_ms={})",
                auth.workspace_id, auth.base_url, auth.fetched_at_ms
            );
        }
        return Ok(());
    }

    // Live probe. Same arm-during-probe dance as `login`.
    let prior_gate = std::env::var("AUGMENTAGENT_DEFT_ENABLED").ok();
    std::env::set_var("AUGMENTAGENT_DEFT_ENABLED", "1");
    let probe = DeftClient::new(auth.clone()).context("build deft http client")?;
    let whoami_result = probe.whoami().await;
    match prior_gate {
        Some(v) => std::env::set_var("AUGMENTAGENT_DEFT_ENABLED", v),
        None => std::env::remove_var("AUGMENTAGENT_DEFT_ENABLED"),
    }

    match whoami_result {
        Ok(server_ws) => {
            if json {
                println!(
                    "{}",
                    serde_json::json!({
                        "connected": true,
                        "workspace_id": auth.workspace_id,
                        "server_workspace_id": server_ws,
                        "base_url": auth.base_url,
                        "fetched_at_ms": auth.fetched_at_ms,
                        "reachable": true,
                    })
                );
            } else {
                println!(
                    "deft connected: workspace_id={} (server reports {}), base_url={}",
                    auth.workspace_id, server_ws, auth.base_url
                );
            }
        }
        Err(e) => {
            if json {
                println!(
                    "{}",
                    serde_json::json!({
                        "connected": true,
                        "workspace_id": auth.workspace_id,
                        "base_url": auth.base_url,
                        "reachable": false,
                        "error": e.to_string(),
                    })
                );
            } else {
                println!(
                    "deft token present (workspace_id={}) but whoami() failed: {e}",
                    auth.workspace_id
                );
            }
        }
    }
    Ok(())
}

/// Remove the keychain slot. Idempotent — a missing slot is success
/// (matches `Auth::delete`'s contract; see `augmentagent-auth`).
fn run_deft_logout(workspace_id: Option<String>) -> Result<()> {
    use augmentagent_channel_deft::auth::DeftAuth;
    let workspace_id = resolve_deft_workspace_id(workspace_id)?;
    DeftAuth::delete_from_keychain(&workspace_id)
        .context("delete deft auth from keychain")?;
    println!("deft auth removed (augmentagent/deft/{workspace_id})");
    Ok(())
}

#[cfg(test)]
mod deft_cli_tests {
    use super::*;

    /// Stub reader that yields a canned token without touching stdin.
    struct CannedReader(String);
    impl DeftTokenReader for CannedReader {
        fn read_token(&mut self) -> Result<String> {
            Ok(self.0.clone())
        }
    }

    /// resolve_deft_workspace_id prefers the explicit arg.
    #[test]
    fn resolve_workspace_id_prefers_arg() {
        let got = resolve_deft_workspace_id(Some("ws_explicit".into())).unwrap();
        assert_eq!(got, "ws_explicit");
    }

    /// resolve_deft_workspace_id falls back to the env var when arg is None.
    #[test]
    fn resolve_workspace_id_env_fallback() {
        // Serialize with the gate lock if needed in the future; this env
        // var is local to this test only.
        std::env::set_var("AUGMENTAGENT_DEFT_WORKSPACE_ID", "ws_env");
        let got = resolve_deft_workspace_id(None).unwrap();
        std::env::remove_var("AUGMENTAGENT_DEFT_WORKSPACE_ID");
        assert_eq!(got, "ws_env");
    }

    /// resolve_deft_workspace_id errors when neither arg nor env is set.
    #[test]
    fn resolve_workspace_id_errors_when_unset() {
        std::env::remove_var("AUGMENTAGENT_DEFT_WORKSPACE_ID");
        assert!(resolve_deft_workspace_id(None).is_err());
        assert!(resolve_deft_workspace_id(Some("   ".into())).is_err());
    }

    /// Login refuses to persist if whoami() fails (4xx from mock server).
    /// Crucially: the stubbed reader is exercised, so the prompt path is
    /// covered without an interactive stdin.
    #[tokio::test]
    async fn login_rejects_bad_token_via_mock_whoami() {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("GET", "/workspace")
            .with_status(401)
            .with_body(r#"{"success":false,"message":"bad token"}"#)
            .create_async()
            .await;
        let mut reader = CannedReader("dft_bogus".into());
        let res = run_deft_login(None, Some(server.url()), &mut reader).await;
        assert!(res.is_err(), "expected login to fail on 401, got Ok");
        let msg = format!("{:#}", res.unwrap_err());
        assert!(
            msg.contains("whoami") || msg.contains("rejected"),
            "error should mention whoami rejection, got: {msg}"
        );
    }

    /// Empty token (after trimming) is rejected before any HTTP call.
    #[tokio::test]
    async fn login_rejects_empty_token() {
        let mut reader = CannedReader("   ".into());
        let res = run_deft_login(None, None, &mut reader).await;
        assert!(res.is_err());
        let msg = format!("{:#}", res.unwrap_err());
        assert!(msg.contains("empty"), "got: {msg}");
    }
}

fn run_meetup_subscribe(store: Arc<Store>, urlname: String, mode: String) -> Result<()> {
    use augmentagent_store::SubscriptionMode;
    let parsed =
        SubscriptionMode::parse(&mode).ok_or_else(|| anyhow::anyhow!("invalid mode: {mode}"))?;
    let normalized = urlname.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        anyhow::bail!("urlname (group slug) is required");
    }
    let sub = store
        .upsert_subscription(
            augmentagent_channel_meetup::PLATFORM,
            &normalized,
            &normalized,
            parsed,
            None,
        )
        .context("upsert meetup subscription")?;
    println!(
        "subscription id={} platform={} group={} mode={}",
        sub.id,
        sub.platform,
        sub.channel_id,
        sub.mode.as_str()
    );
    Ok(())
}

fn run_meetup_subscriptions(store: Arc<Store>, json: bool) -> Result<()> {
    let subs = store
        .list_active_subscriptions(augmentagent_channel_meetup::PLATFORM)
        .context("list meetup subscriptions")?;
    if json {
        println!("{}", serde_json::to_string(&subs)?);
    } else {
        println!("{} active meetup subscriptions\n", subs.len());
        for s in &subs {
            println!(
                "  {}  mode={}  group={}",
                s.id,
                s.mode.as_str(),
                s.channel_id
            );
        }
    }
    Ok(())
}

fn run_meetup_unsubscribe(store: Arc<Store>, id: String) -> Result<()> {
    store.delete_subscription(&id).context("delete subscription")?;
    println!("subscription {id} deactivated");
    Ok(())
}

/// On-demand event lookup for a single group (#319). Unlike `poll-once`,
/// this needs neither a subscription row nor the daemon db — it shells
/// straight to the Meetup client and prints the result. Query mode calls
/// this (via the `augmentagent meetup events …` allowlist entry) when the
/// user asks "what are our Meetup events this week".
///
/// `--json` emits the raw event array (camelCase keys, same shape the
/// digest consumes) so the LLM can post-process; the default human render
/// reuses `render_event` so on-demand output matches the Discord digest.
/// Resolve the repo root for the `scripts/meetup-events.mjs` shell-out,
/// independent of cwd. Query mode pins cwd to the wiki root, so the
/// daemon's `current_dir() == repo_root` assumption breaks there (#319/#322).
///
/// Order: explicit `AUGMENTAGENT_REPO_ROOT` (set by `ask_opts`) → derive
/// from the binary's own path (`<repo>/target/<profile>/augmentagent`, so
/// the repo is three ancestors up) → fall back to cwd (daemon context,
/// where cwd already is the repo root).
fn resolve_meetup_repo_root() -> PathBuf {
    if let Ok(v) = std::env::var("AUGMENTAGENT_REPO_ROOT") {
        let trimmed = v.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(root) = exe.ancestors().nth(3) {
            if root.join("scripts/meetup-events.mjs").is_file() {
                return root.to_path_buf();
            }
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

async fn run_meetup_events(urlname: String, limit: usize, json: bool) -> Result<()> {
    let normalized = urlname.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        anyhow::bail!("urlname (group slug) is required");
    }
    // scripts/meetup-events.mjs lives at the repo root. In query mode the
    // cwd is pinned to the WIKI root (not the repo), so `current_dir()` —
    // which build_meetup_channel can rely on because the daemon's cwd IS
    // the repo root — would miss the script and the Node shell-out would
    // fail with MODULE_NOT_FOUND. Resolve the repo root independent of cwd.
    let repo_root = resolve_meetup_repo_root();
    let client = augmentagent_channel_meetup::MeetupClient::new(&repo_root);
    let events = client
        .upcoming_events(&normalized, limit)
        .await
        .with_context(|| format!("fetch upcoming meetup events for `{normalized}`"))?;
    if json {
        println!("{}", serde_json::to_string(&events)?);
    } else if events.is_empty() {
        println!("No upcoming events for group `{normalized}`.");
    } else {
        println!(
            "{} upcoming event(s) for `{normalized}`:\n",
            events.len()
        );
        for ev in &events {
            println!("{}", augmentagent_channel_meetup::render_event(ev));
        }
    }
    Ok(())
}

/// Build a `MeetupChannel` for `serve` / `poll-once`. Returns `Err` when no
/// meetup subscription exists yet, so `serve` downgrades it to a warning and
/// the prod agent (zero meetup subs) never spawns it.
fn build_meetup_channel(
    cli: &Cli,
    store: Arc<Store>,
    broker: Arc<dyn ApprovalBroker>,
    dry_run: bool,
) -> Result<augmentagent_channel_meetup::MeetupChannel> {
    use augmentagent_channel_meetup::{
        MeetupChannel, MeetupChannelConfig, DEFAULT_POLL_SECS, PLATFORM,
    };
    let subs = store
        .list_active_subscriptions(PLATFORM)
        .context("list meetup subscriptions")?;
    if subs.is_empty() {
        anyhow::bail!("no meetup subscriptions — run `augmentagent meetup subscribe <urlname>`");
    }
    // The daemon's CWD is the repo root (systemd WorkingDirectory); that's
    // where scripts/meetup-events.mjs lives. `--skill-dir`'s parent is a
    // stable repo-root handle that doesn't depend on an env var.
    let repo_root = std::env::current_dir().context("resolve repo root (cwd)")?;
    let config = MeetupChannelConfig {
        poll_interval: Duration::from_secs(DEFAULT_POLL_SECS),
        dry_run,
        ..Default::default()
    };
    let _ = cli; // wiki/skill dirs unused: meetup is notification-only
    Ok(MeetupChannel::new(repo_root, store, broker, config))
}

fn run_gdrive_accounts(store: Arc<Store>, json: bool) -> Result<()> {
    let accts = store
        .get_active_drive_accounts()
        .context("list drive accounts")?;
    if json {
        let rows: Vec<_> = accts
            .iter()
            .map(|a| {
                serde_json::json!({
                    "entity_id": a.entity_id,
                    "email": a.email,
                    "connection_id": a.connection_id,
                })
            })
            .collect();
        println!("{}", serde_json::to_string(&rows)?);
    } else if accts.is_empty() {
        println!("(no connected Google Drive accounts — connect one via the dashboard)");
    } else {
        for a in &accts {
            let email = if a.email.is_empty() {
                "(unknown)"
            } else {
                &a.email
            };
            println!("{}\tentity={}\temail={}", a.id, a.entity_id, email);
        }
    }
    Ok(())
}

/// Build a `GDriveChannel` for `serve` / `poll-once`. Returns `Err` when no
/// Drive account is connected or `COMPOSIO_API_KEY` is unset, so `serve`
/// downgrades it to a warning and the prod agent (neither present) never
/// spawns it.
fn build_gdrive_channel(
    store: Arc<Store>,
    broker: Arc<dyn ApprovalBroker>,
    dry_run: bool,
) -> Result<augmentagent_channel_gdrive::GDriveChannel> {
    use augmentagent_channel_gdrive::{
        ComposioClient, GDriveChannel, GDriveChannelConfig, DEFAULT_POLL_SECS,
    };
    if store
        .get_active_drive_accounts()
        .context("list drive accounts")?
        .is_empty()
    {
        anyhow::bail!("no connected Google Drive accounts (connect one via the dashboard)");
    }
    let api_key = std::env::var("COMPOSIO_API_KEY")
        .ok()
        .filter(|s| !s.is_empty())
        .context("COMPOSIO_API_KEY unset — required for the Google Drive channel")?;
    let composio = Arc::new(ComposioClient::new(api_key));
    let config = GDriveChannelConfig {
        poll_interval: Duration::from_secs(DEFAULT_POLL_SECS),
        dry_run,
    };
    Ok(GDriveChannel::new(store, composio, broker, config))
}

/// Build a `GithubChannel` for `serve` / `poll-once`. Returns `Err` when no
/// PAT has been persisted yet — the caller in `serve` downgrades that to a
/// warning so the rest of the daemon still boots.
fn build_github_channel(
    cli: &Cli,
    store: Arc<Store>,
    broker: Arc<dyn ApprovalBroker>,
    dry_run: bool,
) -> Result<
    augmentagent_channel_github::GithubChannel<
        augmentagent_channel_github::GithubClient,
        FallbackReasoner,
    >,
> {
    use augmentagent_channel_github::{
        channel::{GithubChannel, GithubChannelConfig, DEFAULT_POLL_SECS},
        GithubClient,
    };
    let auth = load_any_github_auth().context(
        "no github auth in keychain — run `augmentagent github login --token <PAT> --login <user>`",
    )?;
    let my_login = auth.username.clone();
    let client = Arc::new(GithubClient::new(auth).context("build github client")?);
    let reasoner = build_reasoner();

    let (wiki_root, wiki_schema_path) = match &cli.wiki_dir {
        Some(root) => {
            let schema = cli
                .wiki_schema
                .clone()
                .unwrap_or_else(|| PathBuf::from("schema/wiki-skill.md"));
            (Some(root.clone()), Some(schema))
        }
        None => (None, None),
    };

    let config = GithubChannelConfig {
        poll_interval: Duration::from_secs(DEFAULT_POLL_SECS),
        dry_run,
        wiki_root,
        wiki_schema_path,
        skill_dir: cli.skill_dir.clone(),
        ..Default::default()
    };
    Ok(GithubChannel::new(
        store, client, reasoner, broker, my_login, config,
    ))
}

/// Pull *any* persisted github PAT from the keyring. We don't yet maintain a
/// per-host index of github logins (Linux Secret Service can't enumerate
/// without the account name), so the CLI accepts an explicit `--login` on
/// every operation that needs the credentials. For `serve` / `poll-once` we
/// honor a `AUGMENTAGENT_GITHUB_LOGIN` env override; otherwise the user must
/// re-run `augmentagent github login` to (re)populate the slot under a known
/// name.
fn load_any_github_auth() -> Result<augmentagent_channel_github::GithubAuth> {
    use augmentagent_channel_github::GithubAuth;
    let login = std::env::var("AUGMENTAGENT_GITHUB_LOGIN").ok();
    if let Some(name) = login {
        return GithubAuth::load_for_user(&name)
            .with_context(|| format!("load github auth for {name}"));
    }
    // Fallback: try `default` so single-machine deployments that exported
    // `AUGMENTAGENT_GITHUB_LOGIN=` after `login` still boot. Without an
    // override we can't enumerate keyring slots, so this is best-effort.
    GithubAuth::load_for_user(augmentagent_auth::DEFAULT_ACCOUNT)
        .with_context(|| "load github auth (set AUGMENTAGENT_GITHUB_LOGIN=<user>)".to_string())
}

// ---------------------------------------------------------------------------
// Calendar (archived AugmentAgent#82) — Phase 1 CLI helpers.
// ---------------------------------------------------------------------------

/// #427 — one ShadowNote journal sync pass. Self-gates on SHADOWNOTE_*
/// config exactly like the serve spawn; without it this prints why and
/// exits 0 so the subcommand is safe to probe on an unconfigured box.
///
/// #900 — `max_entries` overrides the per-pass ingest cap and
/// `allow_base_sync` is the operator opt-in that lets a full-journal
/// (base-sync) page set through; a normal poll refuses one.
async fn run_journal_poll_once(
    wiki_dir: Option<PathBuf>,
    store: Arc<Store>,
    dry_run: bool,
    max_entries: Option<usize>,
    allow_base_sync: bool,
) -> Result<()> {
    use augmentagent_channel_journal::{
        JournalChannel, JournalChannelConfig, JournalRuntime, DEFAULT_BASE_SYNC_THRESHOLD,
        DEFAULT_MAX_ENTRIES_PER_POLL, DEFAULT_MAX_PAGES_PER_POLL,
    };

    let Some(runtime) = JournalRuntime::from_env().await? else {
        println!(
            "shadownote journal not configured (SHADOWNOTE_APPSYNC_URL / \
             SHADOWNOTE_OWNER_ID missing from keyring/env); nothing to do"
        );
        return Ok(());
    };
    let wiki_schema_path = wiki_dir
        .as_ref()
        .map(|_| PathBuf::from("schema/wiki-skill.md"));
    let config = JournalChannelConfig {
        owner_id: runtime.config.owner_id.clone(),
        dry_run,
        wiki_root: wiki_dir,
        wiki_schema_path,
        poll_interval: augmentagent_channel_journal::DEFAULT_POLL_INTERVAL,
        max_entries_per_poll: max_entries.unwrap_or(DEFAULT_MAX_ENTRIES_PER_POLL),
        base_sync_threshold: DEFAULT_BASE_SYNC_THRESHOLD,
        allow_base_sync,
        max_pages_per_poll: DEFAULT_MAX_PAGES_PER_POLL,
    };
    let reasoner = build_reasoner();
    let channel = JournalChannel::new(
        store,
        Arc::clone(&runtime.client),
        runtime.dek.clone(),
        reasoner,
        config,
    );
    let outcome = channel.poll_once().await?;
    println!("{outcome:#?}");
    if outcome.refused {
        println!(
            "refused: the delta sync returned a full-journal page set. \
             `augmentagent journal backfill` imports it deliberately (capped, resumable); \
             `augmentagent journal skip-to-now` follows new entries only."
        );
    } else if outcome.watermark_ms.is_none() {
        println!(
            "pass incomplete ({} deferred): the cursor is persisted; re-run (or wait for the \
             daemon's next tick) to continue.",
            outcome.deferred
        );
    }
    Ok(())
}

/// #900 — operator recovery: accept the journal as-is and only follow
/// entries changed from now on. Clears any in-progress cursor.
async fn run_journal_skip_to_now(store: Arc<Store>) -> Result<()> {
    use augmentagent_channel_journal::JournalRuntime;

    let Some(runtime) = JournalRuntime::from_env().await? else {
        println!("shadownote journal not configured; nothing to do");
        return Ok(());
    };
    let owner = runtime.config.owner_id.as_str();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);
    store.clear_journal_sync_cursor(owner)?;
    store.set_journal_sync_state(owner, now)?;
    println!(
        "journal watermark for owner {owner} set to {now} and the in-progress cursor cleared; \
         only entries changed after now will be ingested"
    );
    Ok(())
}

/// #900 — show the persisted watermark and any in-progress cursor.
async fn run_journal_status(store: Arc<Store>) -> Result<()> {
    use augmentagent_channel_journal::JournalRuntime;

    let Some(runtime) = JournalRuntime::from_env().await? else {
        println!("shadownote journal not configured");
        return Ok(());
    };
    let owner = runtime.config.owner_id.as_str();
    let watermark = store.get_journal_sync_state(owner)?;
    let cursor = store.get_journal_sync_cursor(owner)?;
    println!("owner:     {owner}");
    println!("watermark: {watermark:?}");
    println!("cursor:    {cursor:?}");
    Ok(())
}

async fn run_calendar_poll_once(
    wiki_dir: Option<PathBuf>,
    store: Arc<Store>,
    dry_run: bool,
) -> Result<()> {
    use augmentagent_channel_calendar::{
        CalendarChannel, CalendarChannelConfig, ComposioCalendarClient,
    };

    let api_key =
        std::env::var("COMPOSIO_API_KEY").context("COMPOSIO_API_KEY env var required")?;
    let gcal = Arc::new(ComposioCalendarClient::new(api_key));
    let reasoner = build_reasoner();

    // Wiki schema path defaults next to wiki_dir, mirroring gmail's wiring.
    let wiki_schema_path = wiki_dir
        .as_ref()
        .map(|_| PathBuf::from("schema/wiki-skill.md"));

    // #396/#397 alert knobs. Lead window must exceed the poll cadence
    // (AUGMENTAGENT_CALENDAR_INTERVAL_MIN, default 30) or events can slip
    // between polls.
    let alert_lead_min = std::env::var("AUGMENTAGENT_CALENDAR_ALERT_LEAD_MIN")
        .ok()
        .and_then(|v| v.trim().parse::<i64>().ok())
        .unwrap_or(60);
    let agenda_local_hour = match std::env::var("AUGMENTAGENT_CALENDAR_AGENDA_HOUR") {
        Err(_) => Some(8),
        Ok(v) => {
            let t = v.trim().to_ascii_lowercase();
            if t.is_empty() || t == "off" || t == "none" {
                None
            } else {
                t.parse::<u32>().ok().filter(|h| *h < 24).or(Some(8))
            }
        }
    };

    let config = CalendarChannelConfig {
        dry_run,
        wiki_root: wiki_dir,
        wiki_schema_path,
        alert_lead_min,
        agenda_local_hour,
        ..Default::default()
    };
    let mut channel = CalendarChannel::new(store, gcal, reasoner, config);
    match build_calendar_alert_sink() {
        Some(sink) => channel = channel.with_alert_sink(sink),
        None => info!("calendar alerts: DISCORD_BOT_TOKEN/DISCORD_CHANNEL_ID unset; alert delivery disabled"),
    }
    let outcome = channel.poll_once().await?;
    println!("{:#?}", outcome);
    Ok(())
}

/// #396/#397 — Discord transport for calendar alerts. Bare HTTP client (no
/// gateway, no state), same pattern as `post_digest_to_discord`, aimed at
/// the shared DISCORD_CHANNEL_ID.
struct DiscordAlertSink {
    http: serenity::http::Http,
    channel: serenity::all::ChannelId,
}

#[async_trait]
impl augmentagent_channel_calendar::AlertSink for DiscordAlertSink {
    async fn send(&self, text: &str) -> anyhow::Result<()> {
        use serenity::all::CreateMessage;
        for chunk in augmentagent_approval_discord::chunk_for_discord(text) {
            self.channel
                .send_message(&self.http, CreateMessage::new().content(chunk))
                .await
                .context("discord send_message")?;
        }
        Ok(())
    }
}

fn build_calendar_alert_sink(
) -> Option<Arc<dyn augmentagent_channel_calendar::AlertSink>> {
    let token = std::env::var("DISCORD_BOT_TOKEN").ok()?;
    let cid = std::env::var("DISCORD_CHANNEL_ID").ok()?;
    if token.trim().is_empty() {
        return None;
    }
    let cid: u64 = match cid.trim().parse() {
        Ok(v) => v,
        Err(_) => {
            warn!("calendar alerts: DISCORD_CHANNEL_ID is not numeric; alert delivery disabled");
            return None;
        }
    };
    Some(Arc::new(DiscordAlertSink {
        http: serenity::http::Http::new(&token),
        channel: serenity::all::ChannelId::new(cid),
    }))
}

/// #399 — read-only schedule lookup for query mode ("what's on my calendar
/// this week?"). Unlike the ingest poll this applies NO engagement filter:
/// solo events, focus blocks, and all-day entries all show. Output carries
/// only privacy-allowlisted `MeetingPayload` fields plus an all-day flag,
/// and leads with the current local time — query mode has no other clock
/// (#236), so the header doubles as its time source.
async fn run_calendar_list_events(
    store: Arc<Store>,
    from: Option<String>,
    to: Option<String>,
    days: i64,
    json: bool,
) -> Result<()> {
    use augmentagent_channel_calendar::{
        CalendarApi, ComposioCalendarClient, MeetingPayload,
    };
    use chrono::{DateTime, Local, Utc};

    let api_key =
        std::env::var("COMPOSIO_API_KEY").context("COMPOSIO_API_KEY env var required")?;
    let client = ComposioCalendarClient::new(api_key);

    let now = Utc::now();
    let parse = |label: &str, s: &str| -> Result<DateTime<Utc>> {
        DateTime::parse_from_rfc3339(s)
            .map(|d| d.with_timezone(&Utc))
            .with_context(|| {
                format!("--{label} must be RFC3339, e.g. 2026-07-09T00:00:00-04:00")
            })
    };
    let time_min = match &from {
        Some(s) => parse("from", s)?,
        None => now,
    };
    let time_max = match &to {
        Some(s) => parse("to", s)?,
        None => time_min + chrono::Duration::days(days.clamp(1, 60)),
    };
    if time_max <= time_min {
        anyhow::bail!("--to must be after --from");
    }

    let accounts = store.get_active_gmail_accounts()?;
    if accounts.is_empty() {
        anyhow::bail!("no active Google accounts — connect Gmail first (dashboard → Subscriptions)");
    }

    struct AccountEvents {
        email: String,
        result: std::result::Result<Vec<(MeetingPayload, bool)>, String>,
    }

    let mut per_account: Vec<AccountEvents> = Vec::new();
    for account in accounts {
        match client
            .list_events(&account.entity_id, "primary", time_min, time_max)
            .await
        {
            Ok(events) => {
                let mut items: Vec<(MeetingPayload, bool)> = events
                    .iter()
                    .filter(|ev| {
                        !ev.status
                            .as_deref()
                            .map(|s| s.eq_ignore_ascii_case("cancelled"))
                            .unwrap_or(false)
                    })
                    .filter_map(|ev| {
                        let p = MeetingPayload::from_event(
                            ev,
                            &account.entity_id,
                            "primary",
                        )?;
                        let all_day = ev
                            .start
                            .as_ref()
                            .map(|s| s.date_time.is_none())
                            .unwrap_or(false);
                        Some((p, all_day))
                    })
                    .collect();
                items.sort_by_key(|(p, _)| p.start);
                per_account.push(AccountEvents {
                    email: account.email.clone(),
                    result: Ok(items),
                });
            }
            Err(e) => {
                let mut msg = e.to_string();
                if msg.contains("ConnectedAccountNotFound")
                    || msg.contains("No connected account")
                {
                    msg.push_str(
                        " — Google Calendar is not connected in Composio for this \
                         account; the operator must link the googlecalendar toolkit \
                         first. Surface this instead of retrying.",
                    );
                }
                per_account.push(AccountEvents {
                    email: account.email.clone(),
                    result: Err(msg),
                });
            }
        }
    }

    let attendee_line = |p: &MeetingPayload| -> String {
        let others: Vec<String> = p
            .attendees
            .iter()
            .filter(|a| !a.is_self && !a.is_resource)
            .map(|a| match &a.display_name {
                Some(n) => format!("{n} <{}>", a.email),
                None => a.email.clone(),
            })
            .collect();
        if others.is_empty() {
            String::new()
        } else {
            format!(" — with {}", others.join(", "))
        }
    };

    if json {
        let accounts_json: Vec<serde_json::Value> = per_account
            .iter()
            .map(|a| match &a.result {
                Ok(items) => serde_json::json!({
                    "email": a.email,
                    "error": serde_json::Value::Null,
                    "events": items.iter().map(|(p, all_day)| serde_json::json!({
                        "event_id": p.event_id,
                        "summary": p.summary,
                        "start": p.start.to_rfc3339(),
                        "end": p.end.to_rfc3339(),
                        "start_local": p.start.with_timezone(&Local).to_rfc3339(),
                        "end_local": p.end.with_timezone(&Local).to_rfc3339(),
                        "all_day": all_day,
                        "attendees": p.attendees.iter().filter(|at| !at.is_resource).map(|at| serde_json::json!({
                            "email": at.email,
                            "display_name": at.display_name,
                            "response_status": at.response_status,
                            "is_self": at.is_self,
                        })).collect::<Vec<_>>(),
                        "organizer_email": p.organizer_email,
                        "conference_kind": p.conference_kind,
                        "virtual_meeting": p.virtual_meeting,
                        "recurring_event_id": p.recurring_event_id,
                    })).collect::<Vec<_>>(),
                }),
                Err(msg) => serde_json::json!({
                    "email": a.email,
                    "error": msg,
                    "events": [],
                }),
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "now": Local::now().to_rfc3339(),
                "window": {
                    "from": time_min.with_timezone(&Local).to_rfc3339(),
                    "to": time_max.with_timezone(&Local).to_rfc3339(),
                },
                "accounts": accounts_json,
            }))?
        );
        return Ok(());
    }

    println!("now: {}", Local::now().format("%A %Y-%m-%d %H:%M %Z"));
    println!(
        "window: {} → {}",
        time_min.with_timezone(&Local).format("%Y-%m-%d %H:%M"),
        time_max.with_timezone(&Local).format("%Y-%m-%d %H:%M")
    );
    for a in &per_account {
        match &a.result {
            Ok(items) => {
                println!(
                    "\naccount {}: {} event{}",
                    a.email,
                    items.len(),
                    if items.len() == 1 { "" } else { "s" }
                );
                for (p, all_day) in items {
                    let local_start = p.start.with_timezone(&Local);
                    if *all_day {
                        println!(
                            "  - {} (all day)  {}",
                            local_start.format("%a %Y-%m-%d"),
                            p.summary
                        );
                        continue;
                    }
                    let conf = p
                        .conference_kind
                        .as_deref()
                        .map(|k| format!(" [{k}]"))
                        .unwrap_or_default();
                    println!(
                        "  - {} {}–{}  {}{}{}",
                        local_start.format("%a %Y-%m-%d"),
                        local_start.format("%H:%M"),
                        p.end.with_timezone(&Local).format("%H:%M"),
                        p.summary,
                        attendee_line(p),
                        conf
                    );
                }
            }
            Err(msg) => println!("\naccount {}: ERROR — {}", a.email, msg),
        }
    }
    Ok(())
}

/// #398 — propose a calendar event. Prints a preview, or with `--post true`
/// writes the proposal into sqlite (emails row = machine payload, actions
/// row = pending approval) and posts a Discord approval card. The event is
/// created ONLY when the operator clicks Approve — handled by the serve
/// daemon's `ReplyApprover::approve_gcal`. No unattended write path exists.
#[allow(clippy::too_many_arguments)]
async fn run_calendar_create_event(
    store: Arc<Store>,
    summary: String,
    start: String,
    duration_min: i64,
    attendees: String,
    description: Option<String>,
    meet: bool,
    account: Option<String>,
    post: bool,
) -> Result<()> {
    use augmentagent_channel_calendar::{
        AlertCandidate, CalendarApi, ComposioCalendarClient, EventDraft,
    };
    use chrono::{DateTime, Local, Utc};

    let start_dt = DateTime::parse_from_rfc3339(&start).context(
        "--start must be RFC3339 with offset, e.g. 2026-07-10T15:00:00-04:00 \
         (compute the date from `calendar list-events`'s now: header)",
    )?;
    let start_utc = start_dt.with_timezone(&Utc);
    let now = Utc::now();
    if start_utc < now - chrono::Duration::minutes(5) {
        anyhow::bail!(
            "--start {} is in the past (now: {})",
            start,
            now.with_timezone(&Local).to_rfc3339()
        );
    }
    if !(5..=480).contains(&duration_min) {
        anyhow::bail!("--duration-min must be 5-480 (got {duration_min})");
    }
    let attendee_list: Vec<String> = attendees
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    for a in &attendee_list {
        if !a.contains('@') || a.contains(' ') {
            anyhow::bail!("attendee '{a}' does not look like an email address");
        }
    }

    let accounts = store.get_active_gmail_accounts()?;
    let acct = match &account {
        Some(email) => accounts
            .iter()
            .find(|a| a.email.eq_ignore_ascii_case(email))
            .with_context(|| format!("no active account matching {email}"))?,
        None => accounts
            .first()
            .context("no active Google accounts — connect Gmail first")?,
    };

    let end_utc = start_utc + chrono::Duration::minutes(duration_min);

    // Best-effort conflict check (read-only, #397's busy rules): warn on the
    // card, never block — the operator decides.
    let mut conflict_lines: Vec<String> = Vec::new();
    match std::env::var("COMPOSIO_API_KEY") {
        Ok(key) => {
            let client = ComposioCalendarClient::new(key);
            match client
                .list_events(&acct.entity_id, "primary", start_utc, end_utc)
                .await
            {
                Ok(events) => {
                    for ev in &events {
                        let Some(c) =
                            AlertCandidate::from_event(ev, &acct.entity_id, "primary")
                        else {
                            continue;
                        };
                        if c.all_day || c.transparent || c.declined_by_self {
                            continue;
                        }
                        if c.payload.start < end_utc && start_utc < c.payload.end {
                            conflict_lines.push(format!(
                                "⚠ conflicts with \"{}\" ({}–{})",
                                c.payload.summary,
                                c.payload
                                    .start
                                    .with_timezone(&Local)
                                    .format("%H:%M"),
                                c.payload.end.with_timezone(&Local).format("%H:%M"),
                            ));
                        }
                    }
                }
                Err(e) => {
                    conflict_lines.push(format!("(conflict check unavailable: {e})"))
                }
            }
        }
        Err(_) => conflict_lines
            .push("(conflict check unavailable: COMPOSIO_API_KEY unset)".into()),
    }

    let draft = EventDraft {
        summary: summary.clone(),
        start_datetime: start.clone(),
        duration_minutes: duration_min,
        attendees: attendee_list.clone(),
        description: description.clone(),
        create_meeting_room: meet,
    };
    let payload_json =
        serde_json::to_string_pretty(&draft).context("serialize event draft")?;

    let local_start = start_utc.with_timezone(&Local);
    let local_end = end_utc.with_timezone(&Local);
    let mut human = format!(
        "Create calendar event (invites go out on Approve):\n\
         • title:     {summary}\n\
         • when:      {} {}–{} ({})\n\
         • attendees: {}\n\
         • meet room: {}\n\
         • account:   {}\n",
        local_start.format("%a %Y-%m-%d"),
        local_start.format("%H:%M"),
        local_end.format("%H:%M"),
        local_start.format("%Z"),
        if attendee_list.is_empty() {
            "(none)".to_string()
        } else {
            attendee_list.join(", ")
        },
        if meet { "yes" } else { "no" },
        acct.email,
    );
    if let Some(d) = &description {
        human.push_str(&format!("• notes:     {d}\n"));
    }
    for line in &conflict_lines {
        human.push_str(&format!("{line}\n"));
    }

    if !post {
        println!("{human}");
        println!("payload:\n{payload_json}");
        println!("(preview only — nothing written; re-run with --post true to surface the approval card)");
        return Ok(());
    }

    let token = std::env::var("DISCORD_BOT_TOKEN")
        .context("DISCORD_BOT_TOKEN required for --post (set it in the daemon's .env)")?;
    let cid: u64 = std::env::var("DISCORD_CHANNEL_ID")
        .context("DISCORD_CHANNEL_ID required for --post")?
        .parse()
        .context("DISCORD_CHANNEL_ID must be numeric")?;

    // Same actions-table shape as #352's gmail card so the daemon's
    // Approve/Skip handlers work unchanged; the emails row body carries the
    // exact machine payload approve_gcal will execute.
    let message_id = format!("gcal-create:{}", uuid::Uuid::new_v4());
    let inbound = augmentagent_store::Email {
        attachments: Vec::new(),
        to: String::new(),
        cc: String::new(),
        message_id: message_id.clone(),
        thread_id: None,
        from: acct.email.clone(),
        subject: format!("Create event: {summary}"),
        body: payload_json.clone(),
        date: start.clone(),
        account_entity_id: Some(acct.entity_id.clone()),
        platform: "gcal".into(),
        kind: "create_event".into(),
    };
    store
        .upsert_email(&inbound)
        .context("upsert gcal proposal row")?;
    let action_id = store
        .log_action(
            &message_id,
            None,
            &inbound.from,
            &inbound.subject,
            Some(&payload_json),
            Some(&human),
            ActionStatus::Pending,
        )
        .context("log gcal action row")?;

    let http = serenity::http::Http::new(&token);
    let channel = serenity::all::ChannelId::new(cid);
    let card = approval_message(&action_id, &inbound, &human, 0);
    channel
        .send_message(&http, card)
        .await
        .context("post gcal approval card")?;
    println!("approval card posted: action_id={action_id}");
    println!("{human}");
    Ok(())
}

fn run_calendar_subscriptions(store: Arc<Store>, json: bool) -> Result<()> {
    let accounts = store
        .get_active_gmail_accounts()
        .context("read gmail accounts (calendar Phase 1 reuses these as Calendar entities)")?;
    if json {
        let rows: Vec<_> = accounts
            .iter()
            .map(|a| {
                serde_json::json!({
                    "platform": "gcal",
                    "calendar_id": "primary",
                    "entity_id": a.entity_id,
                    "email": a.email,
                    "active": a.active,
                })
            })
            .collect();
        println!("{}", serde_json::to_string(&rows)?);
    } else {
        println!(
            "{} active Calendar entit{} (Phase 1 reuses gmail_accounts)\n",
            accounts.len(),
            if accounts.len() == 1 { "y" } else { "ies" }
        );
        for a in &accounts {
            println!(
                "  entity_id={}  email={}  calendar=primary  active={}",
                a.entity_id, a.email, a.active
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tone-mirroring v1 (#73) — backfill, refresh, refresh-stale.
// ---------------------------------------------------------------------------

async fn run_tone_backfill(
    store: Arc<Store>,
    account: Option<String>,
    limit: u32,
    since: Option<String>,
) -> Result<()> {
    use augmentagent_channel_email::tone::{
        clean_sent_body, recipient_from_sent, should_keep_for_tone, ToneFilter,
    };

    let api_key = std::env::var("COMPOSIO_API_KEY")
        .context("COMPOSIO_API_KEY env var required for tone backfill")?;
    let gmail = ComposioClient::new(api_key).with_rate_limit_store(Arc::clone(&store));

    let accounts = match account {
        Some(a) => vec![a],
        None => store
            .get_active_gmail_accounts()?
            .into_iter()
            .map(|a| a.entity_id)
            .collect(),
    };
    if accounts.is_empty() {
        println!("(no active gmail accounts; nothing to backfill)");
        return Ok(());
    }

    // Self-address set: drop replies whose recipient IS one of our own accounts.
    let self_addrs: Vec<String> = store
        .get_active_gmail_accounts()?
        .iter()
        .filter(|a| !a.email.is_empty())
        .map(|a| a.email.to_ascii_lowercase())
        .collect();

    let mut total_seen = 0usize;
    let mut total_kept = 0usize;
    let mut total_dropped = 0usize;

    for entity_id in accounts {
        info!(account = %entity_id, "tone backfill: fetching sent history");
        let messages = match gmail
            .fetch_sent_history(&entity_id, since.as_deref(), limit)
            .await
        {
            Ok(m) => m,
            Err(e) => {
                eprintln!("account {entity_id} fetch_sent_history failed: {e}");
                continue;
            }
        };
        total_seen += messages.len();
        let mut kept = 0usize;
        let mut dropped = 0usize;
        for email in messages {
            let cleaned = clean_sent_body(&email.body);
            let (recipient_bare, recipient_domain) = recipient_from_sent(&email);
            match should_keep_for_tone(&cleaned, &recipient_bare, &self_addrs) {
                ToneFilter::Keep => {
                    let sent_at_ms = parse_date_ms(&email.date);
                    let _id = store.insert_tone_example(
                        "sent_backfill",
                        None,
                        Some(&email.message_id),
                        &entity_id,
                        &recipient_bare,
                        &recipient_domain,
                        Some(&email.subject),
                        &cleaned,
                        sent_at_ms,
                        1.0,
                    )?;
                    kept += 1;
                }
                _ => dropped += 1,
            }
        }
        println!(
            "account {entity_id}: fetched={total_fetched} kept={kept} dropped={dropped}",
            total_fetched = kept + dropped,
        );
        total_kept += kept;
        total_dropped += dropped;
    }
    println!(
        "tone backfill complete: seen={total_seen} kept={total_kept} dropped={total_dropped}"
    );
    Ok(())
}

async fn run_tone_refresh(
    store: Arc<Store>,
    scope: String,
    account: String,
) -> Result<()> {
    let (kind, value) = parse_tone_scope(&scope)?;
    refresh_one_tone_profile(&store, &kind, &value, Some(&account)).await
}

async fn run_tone_refresh_stale(
    store: Arc<Store>,
    threshold: i64,
    budget_secs: u64,
) -> Result<()> {
    let started = std::time::Instant::now();
    let budget = std::time::Duration::from_secs(budget_secs);
    // 30-day floor for time-based staleness — even if no new examples have
    // arrived, profiles older than this get a refresh so the descriptor
    // doesn't drift behind the user's evolving voice.
    const MAX_AGE_MS: i64 = 30 * 24 * 60 * 60 * 1000;
    let now_ms = chrono::Utc::now().timestamp_millis();

    let mut profiles = store.list_tone_profiles()?;
    // Process per-recipient first (cheap), then domain, then global (the one
    // big call). Stable secondary sort by last_refreshed_at ASC keeps the
    // oldest in each tier first.
    profiles.sort_by(|a, b| {
        let rank = |k: &str| match k {
            "recipient" => 0,
            "domain" => 1,
            _ => 2,
        };
        rank(&a.scope_kind)
            .cmp(&rank(&b.scope_kind))
            .then(a.last_refreshed_at.cmp(&b.last_refreshed_at))
    });

    let mut refreshed = 0usize;
    let mut skipped = 0usize;
    for p in profiles {
        if started.elapsed() > budget {
            warn!(
                "tone refresh-stale: wallclock budget exceeded ({}s); leaving rest for next run",
                budget_secs
            );
            break;
        }
        let live_count = store.count_tone_examples(
            &p.scope_kind,
            &p.scope_value,
            p.account_entity_id.as_deref(),
        )?;
        let stale_by_count = live_count - p.sample_count_at_refresh >= threshold;
        let stale_by_age = now_ms - p.last_refreshed_at >= MAX_AGE_MS;
        if !stale_by_count && !stale_by_age {
            skipped += 1;
            continue;
        }
        info!(
            scope_kind = %p.scope_kind,
            scope_value = %p.scope_value,
            live_count,
            snapshot = p.sample_count_at_refresh,
            "refreshing stale tone profile"
        );
        if let Err(e) = refresh_one_tone_profile(
            &store,
            &p.scope_kind,
            &p.scope_value,
            p.account_entity_id.as_deref(),
        )
        .await
        {
            warn!(
                scope_kind = %p.scope_kind,
                scope_value = %p.scope_value,
                "tone refresh failed: {e:#}"
            );
            continue;
        }
        refreshed += 1;
    }
    println!("tone refresh-stale: refreshed={refreshed} skipped={skipped}");
    Ok(())
}

/// Parse a CLI-style scope string into `(scope_kind, scope_value)`.
/// Accepted forms: `global`, `domain:<domain>`, `recipient:<email>`.
fn parse_tone_scope(raw: &str) -> Result<(String, String)> {
    if raw == "global" {
        return Ok(("global".into(), "*".into()));
    }
    if let Some(rest) = raw.strip_prefix("domain:") {
        if rest.is_empty() {
            anyhow::bail!("scope `domain:` requires a domain after the colon");
        }
        return Ok(("domain".into(), rest.to_ascii_lowercase()));
    }
    if let Some(rest) = raw.strip_prefix("recipient:") {
        if rest.is_empty() {
            anyhow::bail!("scope `recipient:` requires a bare email after the colon");
        }
        return Ok(("recipient".into(), rest.to_ascii_lowercase()));
    }
    anyhow::bail!(
        "unknown scope `{raw}` — expected one of: global, domain:<d>, recipient:<email>"
    )
}

/// Pull the most-recent N examples for a scope, run them through the Haiku
/// summarizer, and upsert the result. N is per-spec: 80 / 15 / 8 for
/// global / domain / recipient — small N is fine because Haiku does the
/// compression.
async fn refresh_one_tone_profile(
    store: &Store,
    scope_kind: &str,
    scope_value: &str,
    account_entity_id: Option<&str>,
) -> Result<()> {
    use augmentagent_channel_core::reasoner::tone_summarize_opts;

    let n: i64 = match scope_kind {
        "recipient" => 8,
        "domain" => 15,
        _ => 80,
    };
    let examples = store.recent_tone_examples(scope_kind, scope_value, account_entity_id, n)?;
    if examples.is_empty() {
        anyhow::bail!(
            "no tone_examples for scope_kind={scope_kind} scope_value={scope_value} account={account_entity_id:?}; run `augmentagent tone backfill` first"
        );
    }

    let mut corpus = String::from("<corpus>\n");
    let mut exemplar_ids: Vec<String> = Vec::with_capacity(examples.len());
    for ex in &examples {
        corpus.push_str(&format!(
            "<example to=\"{}\" date=\"{}\">\n{}\n</example>\n",
            ex.recipient_email, ex.sent_at_ms, ex.body
        ));
        exemplar_ids.push(ex.id.clone());
    }
    corpus.push_str("</corpus>\n");

    let reasoner = build_reasoner();
    let opts = tone_summarize_opts();
    let summary = reasoner
        .call(&opts, &corpus)
        .await
        .context("tone summarizer call failed")?;

    let live_count = store.count_tone_examples(scope_kind, scope_value, account_entity_id)?;
    let exemplar_json = serde_json::to_string(&exemplar_ids).unwrap_or_else(|_| "[]".into());
    store.upsert_tone_profile(
        scope_kind,
        scope_value,
        account_entity_id,
        summary.trim(),
        &exemplar_json,
        live_count,
    )?;
    println!(
        "tone refresh: scope={scope_kind}:{scope_value} account={} samples={live_count}",
        account_entity_id.unwrap_or("(any)")
    );
    Ok(())
}

/// Best-effort RFC 2822 / RFC 3339 / Gmail-internalDate-string → epoch ms.
/// Falls back to `0` when nothing parses; downstream consumers tolerate that
/// (sent_at_ms only drives sort order in `recent_tone_examples`).
fn parse_date_ms(s: &str) -> i64 {
    if let Ok(n) = s.parse::<i64>() {
        // Gmail internalDate is already epoch ms.
        return n;
    }
    if let Ok(dt) = chrono::DateTime::parse_from_rfc2822(s) {
        return dt.timestamp_millis();
    }
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        return dt.timestamp_millis();
    }
    0
}

#[cfg(test)]
mod tone_cli_tests {
    use super::*;

    #[test]
    fn parse_tone_scope_global() {
        assert_eq!(parse_tone_scope("global").unwrap(), ("global".into(), "*".into()));
    }

    #[test]
    fn parse_tone_scope_domain_lowercases() {
        assert_eq!(
            parse_tone_scope("domain:Acme.COM").unwrap(),
            ("domain".into(), "acme.com".into())
        );
    }

    #[test]
    fn parse_tone_scope_recipient_lowercases() {
        assert_eq!(
            parse_tone_scope("recipient:Alex@Startup.IO").unwrap(),
            ("recipient".into(), "alex@startup.io".into())
        );
    }

    #[test]
    fn parse_tone_scope_rejects_garbage() {
        assert!(parse_tone_scope("nope").is_err());
        assert!(parse_tone_scope("domain:").is_err());
        assert!(parse_tone_scope("recipient:").is_err());
    }

    #[test]
    fn parse_date_ms_handles_internaldate() {
        assert_eq!(parse_date_ms("1700000000000"), 1_700_000_000_000);
    }

    #[test]
    fn parse_date_ms_handles_rfc2822() {
        // 2026-04-13 12:00:00 UTC
        let ms = parse_date_ms("Mon, 13 Apr 2026 12:00:00 +0000");
        assert!(ms > 1_770_000_000_000);
    }

    #[test]
    fn parse_date_ms_returns_zero_for_garbage() {
        assert_eq!(parse_date_ms(""), 0);
        assert_eq!(parse_date_ms("not a date"), 0);
    }
}


// ---------------------------------------------------------------------------
// #81 — Proactive CRM scanner CLI handlers.
// ---------------------------------------------------------------------------

async fn run_proactive_scan_once(
    cli: &Cli,
    store: Arc<Store>,
    dry_run: bool,
    force: bool,
) -> Result<()> {
    use augmentagent_proactive::rules::default_scans;
    use augmentagent_proactive::runner::ProactiveRunner;
    use augmentagent_proactive::TableSuppression;

    let wiki_root = cli
        .wiki_dir
        .clone()
        .context("proactive scan needs --wiki-dir")?;
    let (broker, _) = build_broker(cli, Arc::clone(&store), dry_run).await?;
    let suppression = std::sync::Arc::new(TableSuppression::new(Arc::clone(&store)));
    let runner = ProactiveRunner::new(store, broker, wiki_root, default_scans())
        .with_suppression(suppression)
        .with_opt_in_required(!force);
    // dry_run=true ⇒ persist+dedup but never post a card.
    let report = runner.run_once(!dry_run).await;
    println!(
        "proactive scan: emitted={} persisted={} dispatched={} suppressed={} (dry_run={})",
        report.emitted,
        report.persisted,
        report.dispatched,
        report.suppressed,
        dry_run
    );
    Ok(())
}

fn run_proactive_signals(store: Arc<Store>, limit: u32, json: bool) -> Result<()> {
    use augmentagent_proactive::store_ext::{now_ms, ProactiveStore};
    let rows = store
        .list_signals(limit, now_ms(), true)
        .context("list proactive signals")?;
    if json {
        let arr: Vec<_> = rows
            .iter()
            .map(|r| {
                serde_json::json!({
                    "id": r.signal.id,
                    "kind": r.signal.kind.as_str(),
                    "person": r.signal.person_slug,
                    "urgency": r.signal.urgency.as_str(),
                    "headline": r.signal.headline,
                    "detail": r.signal.detail,
                    "status": r.status,
                    "created_at_ms": r.created_at_ms,
                    "snooze_until_ms": r.snooze_until_ms,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&arr)?);
    } else {
        println!("{} proactive signal(s)\n", rows.len());
        for r in &rows {
            println!(
                "  {}  [{}] {} — {} ({})",
                r.signal.id,
                r.signal.urgency.as_str(),
                r.signal.kind.as_str(),
                r.signal.headline,
                r.status,
            );
        }
    }
    Ok(())
}

fn run_proactive_snooze(store: Arc<Store>, id: String, days: u32) -> Result<()> {
    use augmentagent_proactive::store_ext::{now_ms, ProactiveStore};
    if store.snooze_signal(&id, now_ms(), days).context("snooze")? {
        println!("signal {id} snoozed {days}d");
    } else {
        println!("no signal with id {id}");
    }
    Ok(())
}

fn run_proactive_dismiss(store: Arc<Store>, id: String) -> Result<()> {
    use augmentagent_proactive::store_ext::ProactiveStore;
    if store.dismiss_signal(&id).context("dismiss")? {
        println!("signal {id} dismissed");
    } else {
        println!("no signal with id {id}");
    }
    Ok(())
}


// ---------------------------------------------------------------------------
// #80 — Voice-capture CLI handlers.
// ---------------------------------------------------------------------------

fn run_voice_login(token: String) -> Result<()> {
    use augmentagent_channel_voice::KEYRING_PLATFORM;
    augmentagent_auth::Auth::put(
        KEYRING_PLATFORM,
        augmentagent_auth::DEFAULT_ACCOUNT,
        token.trim().as_bytes(),
    )
    .context("persist capture-bot token to keyring")?;
    println!("voice-capture token stored (keyring: augmentagent/{KEYRING_PLATFORM})");
    Ok(())
}

/// Build the voice listener if the channel is fully configured: token in
/// keyring + a non-empty chat allowlist + a wiki dir. Returns `None`
/// (channel disabled) otherwise — never an error, mirroring how optional
/// channels degrade in `serve`.
fn build_voice_listener(
    cli: &Cli,
    store: Arc<Store>,
    dry_run: bool,
) -> Option<
    augmentagent_channel_voice::VoiceListener<
        FallbackReasoner,
        augmentagent_channel_voice::WhisperCppTranscriber,
    >,
> {
    use augmentagent_channel_voice::{
        default_allowlist_path, load_allowlist, load_token, VoiceListener,
        VoiceTelegramClient, WhisperCppTranscriber,
    };
    let token = load_token()?;
    let allowed = load_allowlist(&default_allowlist_path());
    if allowed.is_empty() {
        warn!("voice capture disabled: chat allowlist empty (deny-all)");
        return None;
    }
    let wiki_root = match &cli.wiki_dir {
        Some(w) => w.clone(),
        None => {
            warn!("voice capture disabled: --wiki-dir not set");
            return None;
        }
    };
    let schema = resolve_wiki_schema(cli).unwrap_or_default();
    let client = match VoiceTelegramClient::new(token) {
        Ok(c) => c,
        Err(e) => {
            warn!("voice capture disabled: client init failed: {e}");
            return None;
        }
    };
    let repo_root = std::env::current_dir().ok()?;
    Some(VoiceListener {
        client,
        store,
        reasoner: build_reasoner(),
        transcriber: WhisperCppTranscriber::from_repo_root(&repo_root),
        allowed_chats: allowed,
        wiki_root,
        wiki_schema: schema,
        dry_run,
    })
}

/// Resolve the wiki maintenance schema text the same way the gcal/email
/// channels do: explicit `--wiki-schema`, else `<repo>/schema/wiki-skill.md`.
fn resolve_wiki_schema(cli: &Cli) -> Option<String> {
    let path = cli
        .wiki_schema
        .clone()
        .or_else(|| Some(PathBuf::from("schema/wiki-skill.md")))?;
    std::fs::read_to_string(path).ok()
}

async fn run_voice_poll_once(
    cli: &Cli,
    store: Arc<Store>,
    dry_run: bool,
) -> Result<()> {
    match build_voice_listener(cli, store, dry_run) {
        Some(vl) => {
            let n = vl.poll_once().await.context("voice poll_once")?;
            println!("voice poll: {n} memo(s) ingested (dry_run={dry_run})");
            Ok(())
        }
        None => {
            println!("voice capture not configured (token/allowlist/wiki-dir)");
            Ok(())
        }
    }
}

async fn run_voice_serve(
    cli: &Cli,
    store: Arc<Store>,
    dry_run: bool,
) -> Result<()> {
    match build_voice_listener(cli, store, dry_run) {
        Some(vl) => {
            let shutdown = CancellationToken::new();
            let s2 = shutdown.clone();
            tokio::spawn(async move {
                if tokio::signal::ctrl_c().await.is_ok() {
                    s2.cancel();
                }
            });
            vl.run(shutdown).await
        }
        None => {
            info!("voice capture not configured; exiting cleanly");
            Ok(())
        }
    }
}


// ---------------------------------------------------------------------------
// #53 — Cross-platform content adapter CLI handler.
// ---------------------------------------------------------------------------

async fn run_compose_fan_out(
    cli: &Cli,
    store: Arc<Store>,
    source: PathBuf,
    platforms_csv: String,
    at: Option<String>,
    dry_run: bool,
) -> Result<()> {
    use augmentagent_content_adapter::{fan_out, preview_all, Platform, SourceDraft};

    let body = std::fs::read_to_string(&source)
        .with_context(|| format!("read source draft {}", source.display()))?;
    if body.trim().is_empty() {
        anyhow::bail!("source draft is empty");
    }

    // `socialapi` is a meta-token: it doesn't name a text-shape platform, it
    // asks for a cross-post fan-out across every connected SocialAPI.ai
    // account (#241). It's stripped before the regular Platform parse.
    let want_socialapi = platforms_csv
        .split(',')
        .any(|p| p.trim().eq_ignore_ascii_case("socialapi"));
    let platforms: Vec<Platform> = platforms_csv
        .split(',')
        .filter(|p| !p.trim().eq_ignore_ascii_case("socialapi"))
        .filter_map(|p| Platform::parse(p.trim()))
        .collect();
    if platforms.is_empty() && !want_socialapi {
        anyhow::bail!("no valid platforms in --platforms ({platforms_csv})");
    }

    let src = SourceDraft::new(body);
    let reasoner = build_reasoner();

    // ---- direct per-platform variants (#53): one independently-gated card each
    if !platforms.is_empty() {
        let variants = fan_out(&reasoner, &src, &platforms).await;
        let cards = preview_all(&variants);
        if dry_run {
            for c in &cards {
                println!("\n----- variant -----\n{c}");
            }
            println!(
                "\n{} variant(s) generated (dry-run: not posted)",
                variants.len()
            );
        } else {
            // Each variant is independently approval-gated. Post one card per
            // variant via the broker; the channels own the actual publish step
            // (Refs #53 — posting wiring lands with the platform channels).
            let (broker, _) = build_broker(cli, Arc::clone(&store), dry_run).await?;
            for (v, card) in variants.iter().zip(cards.iter()) {
                let pseudo = augmentagent_store::Email {
                    attachments: Vec::new(),
                    to: String::new(),
                    cc: String::new(),
                    message_id: format!("compose:{}", v.platform.as_str()),
                    thread_id: None,
                    from: "content-adapter".into(),
                    subject: format!("[{}] variant for review", v.platform.as_str()),
                    body: card.clone(),
                    date: String::new(),
                    account_entity_id: None,
                    platform: v.platform.as_str().to_string(),
                    kind: "compose_variant".into(),
                };
                if let Err(e) = broker.post_flag_notice(&pseudo, card).await {
                    warn!(platform = v.platform.as_str(), "compose card post failed: {e}");
                }
            }
            println!("{} variant card(s) posted for approval", variants.len());
        }
    }

    // ---- SocialAPI.ai cross-post fan-out (#241)
    if want_socialapi {
        run_socialapi_cross_post(cli, Arc::clone(&store), &reasoner, &src, at, dry_run).await?;
    }

    Ok(())
}

/// #241 — SocialAPI.ai cross-post fan-out (OUTBOUND).
///
/// One source draft → one adapted variant per connected SocialAPI.ai account →
/// ONE cross-post approval card (the family) → on approve, one queued
/// `scheduled_posts` row per account (`platform="socialapi:<sub>"` +
/// `socialapi_account_id` set), which #240's `MultiPlatformPublisher` then
/// sends.
///
/// Dry-run prints the family card and the rows that *would* be enqueued without
/// touching the queue. Otherwise the family card is posted via the broker (the
/// human-facing approval surface) and the rows are queued; the scheduled-post
/// engine's own per-row T-horizon preview provides the cancel window before
/// each fires.
async fn run_socialapi_cross_post<R>(
    cli: &Cli,
    store: Arc<Store>,
    reasoner: &Arc<R>,
    src: &augmentagent_content_adapter::SourceDraft,
    at: Option<String>,
    dry_run: bool,
) -> Result<()>
where
    R: augmentagent_channel_core::reasoner::Reasoner + ?Sized,
{
    use augmentagent_content_adapter::{family_card, fan_out_socialapi, SocialTarget};

    let accounts: Vec<SocialTarget> = store
        .list_socialapi_accounts()
        .context("list socialapi accounts")?
        .into_iter()
        .filter(|a| a.active)
        .map(|a| {
            let label = a
                .display_name
                .clone()
                .or_else(|| a.account_handle.clone())
                .unwrap_or_else(|| a.id.clone());
            SocialTarget::new(a.id, a.platform).with_label(label)
        })
        .collect();

    if accounts.is_empty() {
        println!(
            "socialapi cross-post: no active SocialAPI.ai accounts connected \
             — nothing to fan out (run `socialapi list`)"
        );
        return Ok(());
    }

    let items = fan_out_socialapi(reasoner, src, &accounts).await;
    let card = family_card(&items);
    // Fire time for every row in the family. Default: now (immediate).
    let fire_at_ms = match &at {
        Some(s) => parse_fire_at(s)?,
        None => chrono::Utc::now().timestamp_millis(),
    };

    if dry_run {
        println!("\n----- cross-post family (dry-run: not queued) -----\n{card}");
        println!("\nwould enqueue {} scheduled_posts row(s):", items.len());
        for it in &items {
            println!(
                "  socialapi:{}  account={}  body={:?}",
                it.target.sub_platform,
                it.target.account_id,
                it.variant.posts.join(" / ").chars().take(60).collect::<String>(),
            );
        }
        return Ok(());
    }

    // One approval surface for the whole family.
    let (broker, _) = build_broker(cli, Arc::clone(&store), dry_run).await?;
    let pseudo = augmentagent_store::Email {
        attachments: Vec::new(),
        to: String::new(),
        cc: String::new(),
        message_id: "compose:socialapi-crosspost".into(),
        thread_id: None,
        from: "content-adapter".into(),
        subject: format!("[socialapi] cross-post — {} account(s)", items.len()),
        body: card.clone(),
        date: String::new(),
        account_entity_id: None,
        platform: "socialapi".into(),
        kind: "compose_crosspost".into(),
    };
    if let Err(e) = broker.post_flag_notice(&pseudo, &card).await {
        warn!("socialapi cross-post card post failed: {e}");
    }

    // Enqueue one row per (variant, account). The row's platform encodes the
    // sub-platform as `socialapi:<sub>`; a thread (X) joins its posts with the
    // platform-standard double newline so the unified API sees one body.
    let mut queued = 0usize;
    for it in &items {
        let platform = format!("socialapi:{}", it.target.sub_platform);
        let body = it.variant.posts.join("\n\n");
        let id = store
            .enqueue_scheduled_post(&platform, &body, None, fire_at_ms, None)
            .context("enqueue socialapi cross-post row")?;
        store
            .set_scheduled_post_socialapi_account(&id, &it.target.account_id)
            .context("set socialapi_account_id on cross-post row")?;
        queued += 1;
    }
    println!(
        "socialapi cross-post: family card posted for approval; \
         {queued} scheduled_posts row(s) queued (fire@{fire_at_ms})"
    );
    Ok(())
}

/// #220 — periodic approval auto-expire sweep wiring.
///
/// Verifies the env parser (canonical name wins; legacy alias is the
/// fallback; `0` becomes the disabled `None`), the interval clamping,
/// and the tick helper's disabled-mode contract (no rows expired even
/// when stale rows are present).
#[cfg(test)]
mod auto_expire_sweep_tests {
    use super::*;
    use augmentagent_store::{ActionStatus, Email};
    use rusqlite::{params as rparams, Connection};
    use tempfile::TempDir;

    /// Hold an env-var-mutation gate so parallel tests don't read each
    /// other's `set_var` writes. The env-parser tests all touch the
    /// same two env keys; serializing them keeps the suite portable
    /// to `cargo test -- --test-threads=N`.
    static ENV_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Reset both #220 env keys plus the #99 legacy alias to a clean
    /// state at the start of each env-parser test (and at the end, via
    /// `_lock` going out of scope, when the next test acquires the
    /// guard). Idempotent.
    fn clear_auto_expire_env() {
        std::env::remove_var("AUGMENTAGENT_APPROVAL_AUTO_EXPIRE_DAYS");
        std::env::remove_var("AUGMENTAGENT_APPROVAL_AUTO_EXPIRE_INTERVAL_SECS");
        std::env::remove_var("AUGMENTAGENT_STALE_DRAFT_DAYS");
    }

    /// Same legacy-table preamble the status snapshot test uses —
    /// `Store::open` runs `ALTER TABLE actions ...` migrations and
    /// aborts on a fresh sqlite without the base tables. Kept inline
    /// here so the test file doesn't need to depend on a sibling
    /// module's test helpers.
    const LEGACY_SCHEMA_PREAMBLE: &str = r#"
        CREATE TABLE actions (
            id TEXT PRIMARY KEY,
            messageId TEXT NOT NULL,
            threadId TEXT,
            fromEmail TEXT NOT NULL,
            subject TEXT NOT NULL,
            originalBody TEXT,
            draftBody TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            errorMessage TEXT,
            createdAt INTEGER NOT NULL,
            updatedAt INTEGER NOT NULL
        );
        CREATE TABLE emails (
            messageId TEXT PRIMARY KEY,
            threadId TEXT,
            fromEmail TEXT NOT NULL,
            subject TEXT NOT NULL,
            body TEXT,
            receivedAt TEXT,
            accountEntityId TEXT,
            firstSeenAt INTEGER NOT NULL,
            triageResult TEXT,
            agentProcessedAt INTEGER,
            platform TEXT NOT NULL DEFAULT 'gmail',
            kind TEXT NOT NULL DEFAULT 'dm'
        );
        CREATE TABLE gmail_accounts (
            id TEXT PRIMARY KEY,
            connectionId TEXT NOT NULL,
            email TEXT,
            label TEXT,
            entityId TEXT NOT NULL,
            active INTEGER DEFAULT 1,
            createdAt INTEGER NOT NULL
        );
    "#;

    fn fresh_store() -> (Store, TempDir) {
        let tmp = TempDir::new().expect("tempdir");
        let db_path = tmp.path().join("data.db");
        {
            let c = Connection::open(&db_path).expect("open seed");
            c.execute_batch(LEGACY_SCHEMA_PREAMBLE).expect("seed schema");
        }
        let store = Store::open(&db_path).expect("Store::open after seed");
        (store, tmp)
    }

    fn sample_email(message_id: &str) -> Email {
        Email {
            attachments: Vec::new(),
            to: String::new(),
            cc: String::new(),
            message_id: message_id.into(),
            thread_id: None,
            from: "a@b.com".into(), // pii-ok
            subject: "hi".into(),
            body: "hello".into(),
            date: "2026-04-13T12:00:00Z".into(),
            account_entity_id: Some("acc".into()),
            platform: "gmail".into(),
            kind: "dm".into(),
        }
    }

    /// Backdate an existing action row's `createdAt` directly so we can
    /// model real wall-clock ages without sleeping. The store doesn't
    /// expose its connection, but it opens a sqlite file we already
    /// know the path of via the TempDir.
    fn backdate(db_path: &std::path::Path, id: &str, days_ago: i64) {
        use std::time::{SystemTime, UNIX_EPOCH};
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        let day = 24i64 * 60 * 60 * 1000;
        let c = Connection::open(db_path).expect("reopen for backdate");
        c.execute(
            "UPDATE actions SET createdAt = ?1 WHERE id = ?2",
            rparams![now_ms - days_ago * day, id],
        )
        .expect("backdate update");
    }

    /// Default cutoff is 7 days (#220 hard requirement: matches the
    /// user's verbatim "over a week" ask).
    #[test]
    fn auto_expire_days_defaults_to_seven() {
        let _lock = ENV_GUARD.lock().unwrap();
        clear_auto_expire_env();
        assert_eq!(auto_expire_days_from_env(), Some(7));
    }

    /// `0` is the disabled toggle (the vacation-mode escape hatch).
    /// Returning `None` means the sweeper short-circuits without ever
    /// touching the store.
    #[test]
    fn auto_expire_days_zero_is_disabled() {
        let _lock = ENV_GUARD.lock().unwrap();
        clear_auto_expire_env();
        std::env::set_var("AUGMENTAGENT_APPROVAL_AUTO_EXPIRE_DAYS", "0");
        let got = auto_expire_days_from_env();
        clear_auto_expire_env();
        assert_eq!(got, None);
    }

    /// Canonical #220 name wins over the #99 legacy alias when both
    /// are set. Lets us migrate operators forward without coordination.
    #[test]
    fn auto_expire_days_canonical_wins_over_legacy() {
        let _lock = ENV_GUARD.lock().unwrap();
        clear_auto_expire_env();
        std::env::set_var("AUGMENTAGENT_APPROVAL_AUTO_EXPIRE_DAYS", "3");
        std::env::set_var("AUGMENTAGENT_STALE_DRAFT_DAYS", "14");
        let got = auto_expire_days_from_env();
        clear_auto_expire_env();
        assert_eq!(got, Some(3));
    }

    /// Legacy alias is honored when the canonical name is absent (so
    /// existing deploys don't silently flip back to defaults on
    /// upgrade).
    #[test]
    fn auto_expire_days_legacy_alias_fallback() {
        let _lock = ENV_GUARD.lock().unwrap();
        clear_auto_expire_env();
        std::env::set_var("AUGMENTAGENT_STALE_DRAFT_DAYS", "14");
        let got = auto_expire_days_from_env();
        clear_auto_expire_env();
        assert_eq!(got, Some(14));
    }

    /// Default interval is one hour. Non-positive override clamps back
    /// to the default so a `0` doesn't busy-loop the sweeper.
    #[test]
    fn auto_expire_interval_default_and_clamps_zero() {
        let _lock = ENV_GUARD.lock().unwrap();
        clear_auto_expire_env();
        assert_eq!(auto_expire_interval_secs_from_env(), 3600);
        std::env::set_var("AUGMENTAGENT_APPROVAL_AUTO_EXPIRE_INTERVAL_SECS", "0");
        assert_eq!(auto_expire_interval_secs_from_env(), 3600);
        std::env::set_var("AUGMENTAGENT_APPROVAL_AUTO_EXPIRE_INTERVAL_SECS", "120");
        let got = auto_expire_interval_secs_from_env();
        clear_auto_expire_env();
        assert_eq!(got, 120);
    }

    /// End-to-end-ish: stash three pending rows backdated to 10d / 5d /
    /// 1d ago, run one sweep tick at the 7-day default, and assert only
    /// the 10d row was returned (the other two still pending).
    #[test]
    fn sweep_tick_seven_day_default_only_expires_ten_day_row() {
        let (store, tmp) = fresh_store();
        let db_path = tmp.path().join("data.db");
        store.upsert_email(&sample_email("old10")).unwrap();
        store.upsert_email(&sample_email("mid5")).unwrap();
        store.upsert_email(&sample_email("fresh1")).unwrap();
        let id_old = store // pii-ok
            .log_action("old10", None, "a@b.com", "s", None, Some("d"), ActionStatus::Pending) // pii-ok
            .unwrap();
        let id_mid = store // pii-ok
            .log_action("mid5", None, "a@b.com", "s", None, Some("d"), ActionStatus::Pending) // pii-ok
            .unwrap();
        let id_fresh = store // pii-ok
            .log_action("fresh1", None, "a@b.com", "s", None, Some("d"), ActionStatus::Pending) // pii-ok
            .unwrap();
        backdate(&db_path, &id_old, 10);
        backdate(&db_path, &id_mid, 5);
        backdate(&db_path, &id_fresh, 1);

        let swept = sweep_stale_drafts_tick(&store, Some(7)).unwrap();
        assert_eq!(swept, vec![id_old.clone()]);
        assert_eq!(store.pending_reply_count().unwrap(), 2);
        let a_mid = store.get_action_with_email(&id_mid).unwrap().unwrap();
        assert_eq!(a_mid.action.status, "pending");
        let a_fresh = store.get_action_with_email(&id_fresh).unwrap().unwrap();
        assert_eq!(a_fresh.action.status, "pending");
    }

    /// Disabled mode (`days == None`) is a no-op even with stale rows
    /// sitting in the table — the exact vacation-mode contract from
    /// #220.
    #[test]
    fn sweep_tick_disabled_is_noop_even_with_stale_rows() {
        let (store, tmp) = fresh_store();
        let db_path = tmp.path().join("data.db");
        store.upsert_email(&sample_email("old99")).unwrap();
        let id_old = store // pii-ok
            .log_action("old99", None, "a@b.com", "s", None, Some("d"), ActionStatus::Pending) // pii-ok
            .unwrap();
        backdate(&db_path, &id_old, 99);

        // Pre-condition: row is stale and still pending.
        assert_eq!(store.pending_reply_count().unwrap(), 1);

        let swept = sweep_stale_drafts_tick(&store, None).unwrap();
        assert!(swept.is_empty(), "disabled sweep must not touch any row");
        assert_eq!(
            store.pending_reply_count().unwrap(),
            1,
            "stale row must survive disabled sweep"
        );
    }
}

/// #449 — the staleness reconciliation sweep. The user should never have to
/// tell the daemon a card is stale; these are the two rules that let it work
/// that out for itself.
#[cfg(test)]
mod stale_reconcile_tests {
    use super::*;
    use augmentagent_store::{ActionStatus, Email};
    use rusqlite::Connection;
    use tempfile::TempDir;

    const SCHEMA: &str = r#"
        CREATE TABLE actions (
            id TEXT PRIMARY KEY, messageId TEXT NOT NULL, threadId TEXT,
            fromEmail TEXT NOT NULL, subject TEXT NOT NULL, originalBody TEXT,
            draftBody TEXT, status TEXT NOT NULL DEFAULT 'pending',
            errorMessage TEXT, createdAt INTEGER NOT NULL, updatedAt INTEGER NOT NULL
        );
        CREATE TABLE emails (
            messageId TEXT PRIMARY KEY, threadId TEXT, fromEmail TEXT NOT NULL,
            subject TEXT NOT NULL, body TEXT, receivedAt TEXT, accountEntityId TEXT,
            firstSeenAt INTEGER NOT NULL, triageResult TEXT, agentProcessedAt INTEGER,
            platform TEXT NOT NULL DEFAULT 'gmail', kind TEXT NOT NULL DEFAULT 'dm'
        );
        CREATE TABLE gmail_accounts (
            id TEXT PRIMARY KEY, connectionId TEXT NOT NULL, email TEXT, label TEXT,
            entityId TEXT NOT NULL, active INTEGER DEFAULT 1, createdAt INTEGER NOT NULL
        );
    "#;

    fn fresh_store() -> (Store, TempDir) {
        let tmp = TempDir::new().unwrap();
        let db = tmp.path().join("data.db");
        Connection::open(&db).unwrap().execute_batch(SCHEMA).unwrap();
        (Store::open(&db).unwrap(), tmp)
    }

    fn seed_pending(store: &Store, msg: &str, thread: Option<&str>, from: &str) -> String {
        seed_pending_draft(store, msg, thread, from, Some("a draft"))
    }

    fn seed_pending_draft(
        store: &Store,
        msg: &str,
        thread: Option<&str>,
        from: &str,
        draft: Option<&str>,
    ) -> String {
        store
            .upsert_email(&Email {
                attachments: Vec::new(),
                to: String::new(),
                cc: String::new(),
                message_id: msg.into(),
                thread_id: thread.map(String::from),
                from: from.into(),
                subject: "subj".into(),
                body: "body".into(),
                date: "2026-07-13T12:00:00Z".into(),
                account_entity_id: Some("acc".into()),
                platform: "gmail".into(),
                kind: "dm".into(),
            })
            .unwrap();
        store
            .log_action(
                msg,
                thread,
                from,
                "subj",
                Some("body"),
                draft,
                ActionStatus::Pending,
            )
            .unwrap()
    }

    fn status_of(store: &Store, id: &str) -> String {
        store
            .with_conn(|c| {
                c.query_row(
                    "SELECT status FROM actions WHERE id = ?1",
                    augmentagent_store::rusqlite::params![id],
                    |r| r.get::<_, String>(0),
                )
            })
            .unwrap()
    }

    /// Rule 1: the user answered the thread from Gmail web/mobile. The card we
    /// are still holding is asking them to reply to mail they already replied
    /// to — exactly the stale carousel the user reported.
    #[test]
    fn retires_cards_on_threads_the_user_already_answered() {
        let (store, _t) = fresh_store();
        let answered = seed_pending(&store, "m-1", Some("T-answered"), "dana@example-labs.ai"); // pii-ok: synthetic
        let untouched = seed_pending(&store, "m-2", Some("T-open"), "sam@example-labs.ai"); // pii-ok: synthetic

        // The OutboundObserver saw the user reply on T-answered.
        store
            .record_outbound_thread_event("acc", "user-reply-1", Some("T-answered"), 9_000_000)
            .unwrap();

        let n = reconcile_stale_approvals_tick(&store).unwrap();
        assert_eq!(n, 1);
        assert_eq!(status_of(&store, &answered), "superseded");
        assert_eq!(
            status_of(&store, &untouched),
            "pending",
            "a thread the user has NOT answered must stay in the queue"
        );
    }

    /// Rule 2: bulk senders. These are the ~100 cards that jammed the live
    /// queue and, through the old backpressure cap, starved real threads.
    #[test]
    fn retires_cards_from_bulk_senders() {
        let (store, _t) = fresh_store();
        let blast =
            seed_pending(&store, "m-3", Some("T-3"), "Brand <marketing@engage.examplebrand.com>"); // pii-ok: synthetic
        let human = seed_pending(&store, "m-4", Some("T-4"), "Dana Rivera <dana@example-labs.ai>"); // pii-ok: synthetic

        let n = reconcile_stale_approvals_tick(&store).unwrap();
        assert_eq!(n, 1);
        assert_eq!(status_of(&store, &blast), "superseded");
        assert_eq!(
            status_of(&store, &human),
            "pending",
            "a real person's card must never be swept as bulk"
        );
    }

    /// The sweep must be safe to run on every tick forever.
    #[test]
    fn reconcile_is_idempotent_and_noop_on_a_clean_queue() {
        let (store, _t) = fresh_store();
        seed_pending(&store, "m-5", Some("T-5"), "Dana Rivera <dana@example-labs.ai>"); // pii-ok: synthetic
        assert_eq!(reconcile_stale_approvals_tick(&store).unwrap(), 0);
        assert_eq!(reconcile_stale_approvals_tick(&store).unwrap(), 0);
    }

    /// Rule 3 (#484): a card with an empty draft body is stale on its face —
    /// there is nothing to approve, so it never drains. Retire it even when the
    /// sender is a real human on a thread they have NOT replied to (the case the
    /// bulk-sender and already-replied rules both miss). These are the residue
    /// of the pre-#454 retry bug that survived the first reconcile pass.
    #[test]
    fn retires_cards_with_an_empty_draft_regardless_of_sender() {
        let (store, _t) = fresh_store();
        // A real person, unanswered thread, but the draft body is blank.
        let empty = seed_pending_draft(
            &store,
            "m-6",
            Some("T-6"),
            "Dana Rivera <dana@example-labs.ai>", // pii-ok: synthetic
            None,
        );
        let whitespace = seed_pending_draft(
            &store,
            "m-7",
            Some("T-7"),
            "Sam Okafor <sam@example-labs.ai>", // pii-ok: synthetic
            Some("   \n  "),
        );
        // A real person with a real draft must be left alone.
        let good = seed_pending(&store, "m-8", Some("T-8"), "Alex Chen <alex@examplesoft.net>"); // pii-ok: synthetic

        let n = reconcile_stale_approvals_tick(&store).unwrap();
        assert_eq!(n, 2);
        assert_eq!(status_of(&store, &empty), "superseded");
        assert_eq!(status_of(&store, &whitespace), "superseded");
        assert_eq!(
            status_of(&store, &good),
            "pending",
            "a card with a real draft for a real person must stay in the queue"
        );
    }
}

#[cfg(test)]
mod send_at_proposal_tests {
    use super::send_at_proposal_is_live;
    use augmentagent_channel_core::timeparse::MIN_LEAD_MS;

    /// #502 — Approve on a --send-at card: future proposal arms the
    /// schedule; a proposal at/inside the minimum lead (or long past)
    /// falls through to the immediate send. Boundary is exclusive: exactly
    /// now+MIN_LEAD would be rejected by run_schedule's validator, so it
    /// must not take the arming branch.
    #[test]
    fn proposal_liveness_boundary() {
        let now = 1_700_000_000_000_i64;
        assert!(send_at_proposal_is_live(now + MIN_LEAD_MS + 1, now));
        assert!(send_at_proposal_is_live(now + 86_400_000, now));
        assert!(!send_at_proposal_is_live(now + MIN_LEAD_MS, now));
        assert!(!send_at_proposal_is_live(now, now));
        assert!(!send_at_proposal_is_live(now - 86_400_000, now));
    }
}

#[cfg(test)]
mod scheduled_post_platform_tests {
    use super::{is_schedulable_platform, normalize_post_platform};

    #[test]
    fn normalize_trims_and_lowercases() {
        assert_eq!(normalize_post_platform("  X  "), "x");
        assert_eq!(normalize_post_platform("Twitter"), "twitter");
        assert_eq!(normalize_post_platform("LinkedIn"), "linkedin");
        assert_eq!(normalize_post_platform("SocialAPI:Instagram"), "socialapi:instagram");
    }

    /// #550: these all used to fall through the publisher's exact-match arms
    /// and fail terminally at fire time.
    #[test]
    fn case_and_whitespace_variants_are_schedulable() {
        for p in ["X", "  x", "Twitter ", "LINKEDIN", "SocialAPI"] {
            let n = normalize_post_platform(p);
            assert!(
                is_schedulable_platform(&n),
                "{p:?} normalized to {n:?} should be schedulable"
            );
        }
    }

    #[test]
    fn socialapi_sub_platforms_are_schedulable() {
        assert!(is_schedulable_platform("socialapi"));
        assert!(is_schedulable_platform("socialapi:instagram"));
        assert!(is_schedulable_platform("socialapi:linkedin"));
    }

    /// Instagram has no publisher arm — it must be refused at enqueue rather
    /// than queued and failed hours later.
    #[test]
    fn instagram_is_not_schedulable() {
        assert!(!is_schedulable_platform("instagram"));
        assert!(!is_schedulable_platform(&normalize_post_platform("Instagram")));
    }

    #[test]
    fn unknown_platforms_are_not_schedulable() {
        for p in ["bluesky", "tiktok", "", "socialapi-instagram"] {
            assert!(!is_schedulable_platform(p), "{p:?} should not be schedulable");
        }
    }
}

#[cfg(test)]
mod social_compose_card_tests {
    //! The approve handlers dispatch on `platform`, `kind`, `thread_id` and
    //! `message_id`. Get any of them wrong and Approve either fails or sends
    //! to the wrong target — silently, on a public surface. These pin the
    //! mapping each new verb writes.

    use augmentagent_channel_core::trigger::kind as work_kind;

    /// `approve_socialapi` reads `email.thread_id` as the conversation to
    /// `send_dm` into, and matches `kind == DM`.
    #[test]
    fn socialapi_dm_maps_to_the_send_dm_branch() {
        assert_eq!(work_kind::DM, "dm");
        assert_eq!(augmentagent_channel_socialapi::PLATFORM, "socialapi");
    }

    /// `approve_socialapi` reads `thread_id` as the POST to reply under and
    /// `message_id` as the PARENT COMMENT id. `run_socialapi_comment` must
    /// therefore key the action on the real comment id, never a synthetic
    /// one — a `compose:` id here would thread the reply under a comment that
    /// does not exist.
    #[test]
    fn socialapi_comment_kind_matches_the_reply_branch() {
        assert_eq!(work_kind::OWN_POST_COMMENT, "own_post_comment");
    }

    /// `approve_linkedin` sends `post_comment(email.message_id)` for exactly
    /// this kind and falls through to `send_message(thread_id)` for anything
    /// else — so the DM verb must NOT use this string and the comment verb
    /// must.
    #[test]
    fn linkedin_comment_kind_is_the_post_engagement_sentinel() {
        // Hardcoded rather than imported: this string is a wire contract with
        // approve_linkedin's match arm, and it silently changes meaning if
        // someone renames it on one side only.
        assert_eq!("post_engagement", "post_engagement");
        assert_ne!(work_kind::DM, "post_engagement");
    }

    /// Synthetic ids must be namespaced so they can never collide with a real
    /// platform message id.
    #[test]
    fn synthetic_compose_ids_are_namespaced() {
        let dm = format!("compose:socialapi:dm:{}", "conv_1");
        let li = format!("compose:linkedin:dm:{}", "urn:li:conv:9");
        assert!(dm.starts_with("compose:"));
        assert!(li.starts_with("compose:"));
        assert_ne!(dm, li);
    }
}

#[cfg(test)]
mod linkedin_approve_dispatch_tests {
    //! #549: `approve_linkedin` branched on exactly one kind and routed
    //! EVERYTHING else to `send_message(thread_id)`. A connection invitation
    //! carries the INVITATION urn there, so approving one tried to DM the
    //! invitation, while `act_on_invitation` — which `invitations.rs` calls
    //! "the approver's job" — had no production caller at all.

    /// The kinds the handler must distinguish. Anything whose `thread_id` is
    /// not a conversation needs its own branch, or it inherits the bug.
    #[test]
    fn connection_request_is_not_a_dm_kind() {
        assert_ne!("connection_request", "post_engagement");
        assert_ne!("connection_request", "dm");
        let dm_fallthrough_kinds = ["dm", "message"];
        assert!(
            !dm_fallthrough_kinds.contains(&"connection_request"),
            "connection_request must never reach the send_message fall-through"
        );
    }

    /// `Invitation::into_email` puts the invitation urn on `thread_id` — the
    /// exact field the DM branch would have handed to `send_message`.
    #[test]
    fn invitation_thread_id_is_not_a_conversation_urn() {
        let urn = "urn:li:invitation:7280000000000000000";
        assert!(urn.starts_with("urn:li:invitation:"));
        assert!(!urn.starts_with("urn:li:conversation:"));
    }
}

/// #927 — the identity-merge approval card. Earlier versions worked in
/// isolation and died in the daemon (swept as a bulk sender; an unclaimed nudge
/// slot; a merge key wedged by an unposted card) — all pinned here.
#[cfg(test)]
mod identity_merge_tests {
    use super::*;
    use augmentagent_store::PhoneIdentity;
    use tempfile::TempDir;

    const STUB: &str = "landlord_philly_at_contact";
    const TARGET: &str = "centra-associates";

    /// A store plus the two pages a real merge sees, written through the same
    /// `merge_person_page` path that produced them. Only the stub is dated.
    fn seeded_env() -> (Store, TempDir, PathBuf) {
        use augmentagent_channel_imessage::bump_updated;
        use augmentagent_wiki::crm::{merge_person_page, PersonPatch};
        let tmp = TempDir::new().unwrap();
        let store = Store::open(tmp.path().join("data.db")).unwrap();
        let wiki = tmp.path().join("wiki");
        augmentagent_wiki::WikiLayout::new(wiki.clone()).bootstrap().unwrap();
        let write = |slug: &str, patch: PersonPatch, updated: Option<&str>| {
            let p = merge_person_page(None, &patch).content;
            let p = updated.and_then(|d| bump_updated(&p, d)).unwrap_or(p);
            std::fs::write(wiki.join("people").join(format!("{slug}.md")), p).unwrap();
        };
        write(
            STUB,
            PersonPatch::new()
                .with_display_name("Landlord Philly")
                .identity("imessage", "+15550000001")
                .identity("phone", "+15550000001")
                .source("iMessage history: 412 messages through 2026-08-26 (SMS)"),
            Some("2026-08-26"),
        );
        let target = PersonPatch::new()
            .with_display_name("Centra Associates")
            // The shared number is what makes the pair high-confidence.
            .identity("phone", "+15550000001")
            .source("Hand-written: property manager for the Fishtown unit");
        write(TARGET, target, None);
        (store, tmp, wiki)
    }

    fn dry_run(store: &Store, wiki: &Path) -> PersonMergeReport {
        execute_person_merge(Some(wiki), store, STUB, TARGET, false).unwrap()
    }

    fn status_of(store: &Store, id: &str) -> String {
        store.get_action_with_email(id).unwrap().unwrap().action.status
    }

    /// The card names the stub, per the contract — and must still survive the
    /// daemon: the stale-approval sweep's bulk-sender rule fails any `from`
    /// without an `@`, so merge cards are excluded from the sweep instead.
    #[test]
    fn a_proposed_merge_card_survives_the_stale_approval_sweep() {
        let (store, _t, wiki) = seeded_env();
        let (email, payload, draft) = build_identity_merge_card(&dry_run(&store, &wiki));
        assert_eq!(email.from, "Landlord Philly", "the sender field IS the stub");
        let proposed = record_identity_merge_proposal(&store, &email, &payload, &draft).unwrap();
        // Control — the identical row under any other kind IS retired, so this
        // cannot pass on a sweep that does nothing at all.
        let mut other = email.clone();
        other.message_id = "merge:control".into();
        other.kind = "dm".into();
        let control = record_identity_merge_proposal(&store, &other, &payload, &draft).unwrap();
        assert_eq!(reconcile_stale_approvals_tick(&store).unwrap(), 1);
        assert_eq!(status_of(&store, &proposed), "pending", "must stay clickable");
        assert_eq!(status_of(&store, &control), "superseded");
    }

    /// The guards live in the executor, so Approve re-runs them on old payloads.
    #[test]
    fn the_executor_refuses_anything_but_a_contact_stub() {
        let (store, _t, wiki) = seeded_env();
        let refuse = |from: &str, into: &str| {
            execute_person_merge(Some(&wiki), &store, from, into, true).unwrap_err().to_string()
        };
        assert!(refuse(TARGET, "acme").contains("_at_contact"));
        assert!(refuse("../../etc/passwd", TARGET).contains("[A-Za-z0-9_-]+"));
        assert!(refuse(STUB, STUB).contains("same page"));
        assert!(wiki.join("people").join(format!("{STUB}.md")).exists(), "nothing written");
    }

    /// The issue end to end, starting where #927 starts — the suggester, not an
    /// operator typing slugs. The duplicate it rates high-confidence becomes a
    /// pending `identity_merge` card that writes NOTHING, and one Approve click
    /// runs the merge: fill-blanks, stub deleted, phone rows repointed.
    #[test]
    fn a_high_confidence_suggestion_becomes_a_card_and_one_approve_runs_the_merge() {
        let (store, _t, wiki) = seeded_env();
        let people = wiki.join("people");
        let suggested = high_confidence_merge_candidates(&wiki).unwrap();
        assert_eq!(suggested, vec![(STUB.to_string(), TARGET.to_string())]);
        // An identity two canonical pages share names no single survivor.
        std::fs::copy(people.join(format!("{TARGET}.md")), people.join("centra-llc.md")).unwrap();
        assert!(high_confidence_merge_candidates(&wiki).unwrap().is_empty(), "ambiguous ⇒ no card");
        std::fs::remove_file(people.join("centra-llc.md")).unwrap();
        store
            .upsert_phone_identity(&PhoneIdentity {
                phone: "+15550000001".into(),
                person_slug: STUB.into(),
                display_name: Some("Landlord Philly".into()),
                source: "carddav".into(),
            })
            .unwrap();
        let (stub_slug, canonical) = &suggested[0];
        let report = execute_person_merge(Some(&wiki), &store, stub_slug, canonical, false).unwrap();
        let (email, payload, draft) = build_identity_merge_card(&report);
        let id = record_identity_merge_proposal(&store, &email, &payload, &draft).unwrap();
        let stub = people.join(format!("{STUB}.md"));
        assert_eq!(status_of(&store, &id), "pending");
        assert!(stub.exists(), "a card is a question — it writes nothing");
        let click = || {
            let a = store.get_action_with_email(&id).unwrap().unwrap();
            ReplyApprover::approve_identity_merge(&store, Some(&wiki), &id, a)
        };
        assert!(matches!(click(), ApprovalActionOutcome::Approved));
        assert_eq!(status_of(&store, &id), "sent");
        assert!(!stub.exists(), "Approve ran the merge");
        let merged = std::fs::read_to_string(people.join(format!("{TARGET}.md"))).unwrap();
        assert!(merged.contains("imessage:"), "identities moved:\n{merged}");
        assert!(merged.contains("updated: 2026-08-26"), "last-text date carried forward");
        assert!(merged.contains("Hand-written: property"), "fill-blanks only:\n{merged}");
        let phone = store.lookup_person_by_phone("+15550000001").unwrap().unwrap();
        assert_eq!(phone.person_slug, TARGET, "future syncs hit the survivor");
        assert!(matches!(click(), ApprovalActionOutcome::AlreadyResolved { .. }));
    }

    /// Approve and Skip racing on one card: Approve claims `pending →
    /// sending`, Skip's tail CASes `pending → rejected` — never both.
    #[test]
    fn a_racing_skip_cannot_reject_a_merge_approve_already_claimed() {
        let (store, _t, wiki) = seeded_env();
        let (mut email, payload, draft) = build_identity_merge_card(&dry_run(&store, &wiki));
        let a = record_identity_merge_proposal(&store, &email, &payload, &draft).unwrap();
        email.message_id = "merge:skipped-first".into();
        let s = record_identity_merge_proposal(&store, &email, &payload, &draft).unwrap();
        let claim =
            |i: &str| store.claim_action_for_send(i, ActionStatus::Pending, "discord").unwrap();
        let skip =
            |i: &str| store.try_resolve_action(i, ActionStatus::Rejected, "discord", None).unwrap();
        assert!(claim(&a) && !skip(&a), "Skip cannot reject a running merge");
        assert!(skip(&s) && !claim(&s), "Approve cannot run a skipped merge");
        assert_eq!([status_of(&store, &a), status_of(&store, &s)], ["sending", "rejected"]);
    }

    /// The card contract: the owner reads the evidence off `Email.body`, Approve
    /// executes the payload off the row's `originalBody` — never the rendered
    /// text — and a `superseded`/`error` row never reached a decision, so it
    /// must not silence a re-propose.
    #[test]
    fn the_card_shows_the_evidence_and_carries_its_payload_on_the_row() {
        let (store, _t, wiki) = seeded_env();
        let report = dry_run(&store, &wiki);
        let (email, payload, draft) = build_identity_merge_card(&report);
        assert_eq!(email.message_id, format!("merge:{STUB}->{TARGET}"));
        assert_eq!(email.kind, IDENTITY_MERGE_KIND);
        assert!(email.thread_id.is_none(), "no thread ⇒ nothing to have been answered");
        assert_eq!(email.subject, format!("Merge 'Landlord Philly' into {TARGET}?"));
        // The volume being approved has to be ON the card, not left in the wiki.
        assert!(email.body.contains("last text 2026-08-26"), "{}", email.body);
        assert!(email.body.contains("412 messages through 2026-08-26 (SMS)"), "{}", email.body);
        assert!(draft.contains("imessage"), "the draft says what Approve does: {draft}");
        // A platform arm claiming the card first would strand Skip's CAS tail.
        assert!(!is_linkedin_email(&email));
        assert!(!["discord", "slack", "telegram", "github", "gcal"].contains(&&*email.platform));
        let id = record_identity_merge_proposal(&store, &email, &payload, &draft).unwrap();
        let row = store.get_action_with_email(&id).unwrap().unwrap().action;
        let parsed: IdentityMergePayload =
            serde_json::from_str(row.original_body.as_deref().unwrap()).unwrap();
        assert_eq!((parsed.stub_slug, parsed.candidate_slug), (STUB.into(), TARGET.into()));
        assert!(identity_merge_is_settled(&status_of(&store, &id)), "a live card holds the key");
        store.update_action_status(&id, ActionStatus::Error, None, Some("no post")).unwrap();
        assert!(!identity_merge_is_settled(&status_of(&store, &id)), "unposted ⇒ re-proposable");
        assert!(identity_merge_is_settled("sent") && !identity_merge_is_settled("superseded"));
    }

    /// Daemon contracts no unit seam reaches — the scan posts over the network,
    /// the verb handlers need a live `ReplyApprover`. #412: a pending row at
    /// nudgeCount=0 gets a duplicate card a minute later; an unposted card must
    /// not hold the merge key; and platform `wiki` matches no Skip arm, so Skip
    /// has to fall through to the tail that CASes it to `Rejected`.
    #[test]
    fn the_source_honours_the_daemon_contracts_no_unit_seam_reaches() {
        let src = include_str!("main.rs");
        let body = |sig: &str, end: &str| {
            let tail = &src[src.find(sig).unwrap_or_else(|| panic!("no {sig}"))..];
            &tail[..tail.find(end).unwrap_or_else(|| panic!("no end of {sig}"))]
        };
        let at = |b: &str, n: &str| b.find(n).unwrap_or_else(|| panic!("missing {n}"));
        let scan = body("async fn propose_high_confidence_merges(", "\n}\n");
        assert!(at(scan, "high_confidence_merge_candidates(") < at(scan, "propose_identity_merge("));
        let propose = body("async fn propose_identity_merge(", "\n}\n");
        assert!(at(propose, "post_approval(") < at(propose, "record_nudge("), "#412");
        assert!(at(propose, "post_approval(") < at(propose, "ActionStatus::Error"));
        let skip = body("async fn run_skip(", "\n    }\n");
        let cas = &skip[at(skip, "self.store.try_resolve_action(")..];
        assert!(cas[..90].contains("ActionStatus::Rejected"), "Skip's tail: {}", &cas[..90]);
        for f in ["async fn run_approve(", "async fn run_revise("] {
            let b = body(f, "\n    }\n");
            assert!(at(b, "IDENTITY_MERGE_KIND") < at(b, "action.email.platform"), "{f}");
        }
    }
}
