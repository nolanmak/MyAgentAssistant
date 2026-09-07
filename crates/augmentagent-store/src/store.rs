use std::path::Path;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection, OptionalExtension};
use thiserror::Error;
use uuid::Uuid;

use crate::redact;

use crate::models::{
    Account, ActionRecord, ActionStatus, AgentPrRun, AgentRepo, ChannelSubscription,
    ConnectionRequestRow, DriveAccount, Email, FriendWatch, LearnedPattern,
    LinkedInConnectionSync, OwnPost, PhoneIdentity, RateAuditRow, RateHalt,
    RateWarmup, ScheduledPost, ScheduledPostStatus, SlackWorkspace, SocialapiAccount,
    SocialapiWebhookEvent, SubscriptionMode, TelegramBot,
    ToneExample, ToneProfile, TriageResult, UserLoop, WhatsappDevice,
};

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

pub type StoreResult<T> = Result<T, StoreError>;

/// #900 — an interrupted ShadowNote sync pass, persisted after every page so
/// a restart resumes pagination instead of replaying the whole batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JournalSyncCursor {
    /// The `lastSync` the in-progress query was issued with (`None` = base
    /// sync). A resumed query must reuse it — AppSync page tokens belong to
    /// that query.
    pub last_sync_ms: Option<i64>,
    /// The server's `startedAt` for this pass; becomes the watermark when
    /// the pass completes.
    pub started_at_ms: i64,
    /// Token of the next unprocessed page (`None` = start from the first).
    pub next_token: Option<String>,
}

/// The only `kind` values `socialapi_webhook_events` accepts (#529). Kept as a
/// constant so the runtime guard, the sqlite CHECK, and the drain's filter
/// can't drift apart — a row whose kind matches none of these is never
/// selected by the drain and would strand at `processed = 0` forever.
pub const SOCIALAPI_WEBHOOK_KINDS: [&str; 2] = ["dm", "comment"];

pub struct Store {
    conn: Mutex<Connection>,
    path: std::path::PathBuf,
}

impl Store {
    pub fn open(path: impl AsRef<Path>) -> StoreResult<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let conn = Connection::open(path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "busy_timeout", 5000)?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        Self::migrate(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
            path: path_buf,
        })
    }

    /// On-disk path this store was opened from. Lets extension crates
    /// (e.g. `augmentagent-proactive`) run their own additive queries
    /// against the same db file without threading the path through every
    /// constructor.
    pub fn db_path(&self) -> &Path {
        &self.path
    }

    /// Run a closure with the locked connection. Used by extension traits in
    /// sibling crates that need bespoke queries the core `Store` API doesn't
    /// expose. Keeps the single-connection WAL invariant intact (no second
    /// writer connection racing the daemon's).
    pub fn with_conn<T>(
        &self,
        f: impl FnOnce(&Connection) -> rusqlite::Result<T>,
    ) -> StoreResult<T> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        Ok(f(&guard)?)
    }

    /// #80 — read the last acked Telegram update_id for the voice-capture
    /// bot. `None` (treated as 0 by callers) before the first poll.
    pub fn voice_capture_offset(&self, bot_key: &str) -> StoreResult<Option<i64>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let v: Option<i64> = guard
            .query_row(
                "SELECT last_update_id FROM voice_capture_state WHERE bot_key = ?1",
                params![bot_key],
                |r| r.get(0),
            )
            .optional()?;
        Ok(v)
    }

    /// #80 — persist the last acked Telegram update_id (monotonic upsert).
    pub fn set_voice_capture_offset(
        &self,
        bot_key: &str,
        last_update_id: i64,
    ) -> StoreResult<()> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "INSERT INTO voice_capture_state (bot_key, last_update_id, updated_at_ms) \
             VALUES (?1, ?2, ?3) \
             ON CONFLICT(bot_key) DO UPDATE SET \
                last_update_id = MAX(last_update_id, excluded.last_update_id), \
                updated_at_ms = excluded.updated_at_ms",
            params![bot_key, last_update_id, now],
        )?;
        Ok(())
    }

    /// #396/#397 — calendar alert dedup: the fingerprint recorded when the
    /// alert `key` last fired, or `None` if it never has.
    pub fn calendar_alert_fingerprint(&self, key: &str) -> StoreResult<Option<String>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let v: Option<String> = guard
            .query_row(
                "SELECT fingerprint FROM calendar_alerts WHERE key = ?1",
                params![key],
                |r| r.get(0),
            )
            .optional()?;
        Ok(v)
    }

    /// #396/#397 — record that alert `key` fired describing `fingerprint`
    /// (event start/end times). A later fingerprint change (reschedule)
    /// re-arms the alert; an identical one stays deduped.
    pub fn upsert_calendar_alert(&self, key: &str, fingerprint: &str) -> StoreResult<()> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "INSERT INTO calendar_alerts (key, fingerprint, sent_at_ms) \
             VALUES (?1, ?2, ?3) \
             ON CONFLICT(key) DO UPDATE SET \
                fingerprint = excluded.fingerprint, \
                sent_at_ms  = excluded.sent_at_ms",
            params![key, fingerprint, now],
        )?;
        Ok(())
    }

    /// Daily research pipeline — has this arXiv paper id already been
    /// surfaced on a previous run? Drives dedup so a re-run never re-issues
    /// the same paper.
    pub fn research_seen(&self, arxiv_id: &str) -> StoreResult<bool> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let hit: Option<i64> = guard
            .query_row(
                "SELECT 1 FROM research_seen WHERE arxiv_id = ?1",
                params![arxiv_id],
                |r| r.get(0),
            )
            .optional()?;
        Ok(hit.is_some())
    }

    /// Daily research pipeline — mark an arXiv paper id as surfaced. Idempotent
    /// (ON CONFLICT DO NOTHING) so a partial earlier run can be safely retried.
    pub fn mark_research_seen(&self, arxiv_id: &str) -> StoreResult<()> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "INSERT INTO research_seen (arxiv_id, seen_at) VALUES (?1, ?2) \
             ON CONFLICT(arxiv_id) DO NOTHING",
            params![arxiv_id, now],
        )?;
        Ok(())
    }

    /// #35 — record one detected ask (shadow telemetry). Cheap insert; never
    /// dedups (we want the full shadow stream for analysis).
    #[allow(clippy::too_many_arguments)]
    pub fn record_detected_ask(
        &self,
        message_id: &str,
        platform: &str,
        ask_text: &str,
        resolver_kind: &str,
        auto_fillable: bool,
        confidence: Option<f64>,
        raw_json: Option<&str>,
    ) -> StoreResult<String> {
        let id = Uuid::new_v4().to_string();
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "INSERT INTO detected_asks \
                (id, message_id, platform, ask_text, resolver_kind, \
                 auto_fillable, confidence, raw_json, detected_at_ms) \
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
            params![
                id,
                message_id,
                platform,
                ask_text,
                resolver_kind,
                auto_fillable as i64,
                confidence,
                raw_json,
                now,
            ],
        )?;
        Ok(id)
    }

    /// #35 — count detected asks since `since_ms` (shadow-mode dashboards).
    pub fn detected_asks_since(&self, since_ms: i64) -> StoreResult<i64> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n: i64 = guard.query_row(
            "SELECT COUNT(*) FROM detected_asks WHERE detected_at_ms >= ?1",
            params![since_ms],
            |r| r.get(0),
        )?;
        Ok(n)
    }

    /// Additive, idempotent schema migrations. Safe to run against databases
    /// that were created by the original Node daemon (they just lack some of
    /// the newer columns).
    ///
    /// SCHEMA OWNERSHIP (#45): This crate is the authoritative source for the
    /// 9 base tables + 10 indexes that used to live in `src/db.ts`. The block
    /// of `CREATE TABLE IF NOT EXISTS` statements below mirrors `initDb()`
    /// column-for-column and runs BEFORE the additive `ALTER TABLE` blocks so
    /// the daemon can come up cleanly on an empty file without waiting for the
    /// Node dashboard to bootstrap the schema first (the race that PR #39
    /// papered over). Every CREATE is `IF NOT EXISTS` and every CREATE INDEX
    /// is `IF NOT EXISTS`, which makes this safe to run against the live
    /// production DB (~1.16GB, WAL-active) where every table+index already
    /// exists — the statements become no-ops. Node still defensively re-runs
    /// its own CREATE TABLE IF NOT EXISTS in `initDb()` so the dashboard can
    /// boot even if the Rust daemon hasn't run yet (concurrent systemd start).
    fn migrate(conn: &Connection) -> StoreResult<()> {
        // -------------------------------------------------------------------
        // #45 — Rust-owned schema. Mirrors `src/db.ts::initDb()` exactly
        // (column names, types, NOT NULL, DEFAULT, PRIMARY KEY). Do NOT
        // diverge from Node here; the `storeSchemaParity.test.js` test pins
        // the column lists, and Rust's `store_open_creates_all_node_owned_tables`
        // pins table presence. Indexes for these tables are created at the
        // END of migrate() (after the ALTERs) so legacy DBs that pre-date a
        // referenced column (e.g. `emails.platform`) still upgrade cleanly.
        // -------------------------------------------------------------------
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS actions (\
                 id TEXT PRIMARY KEY,\
                 messageId TEXT NOT NULL,\
                 threadId TEXT,\
                 fromEmail TEXT NOT NULL,\
                 recipientEmail TEXT,\
                 subject TEXT NOT NULL,\
                 originalBody TEXT,\
                 draftBody TEXT,\
                 status TEXT NOT NULL DEFAULT 'pending',\
                 errorMessage TEXT,\
                 createdAt INTEGER NOT NULL,\
                 updatedAt INTEGER NOT NULL\
             );\
             CREATE TABLE IF NOT EXISTS senders (\
                 id TEXT PRIMARY KEY,\
                 email TEXT UNIQUE NOT NULL,\
                 label TEXT,\
                 active INTEGER DEFAULT 1,\
                 createdAt INTEGER NOT NULL\
             );\
             CREATE TABLE IF NOT EXISTS config (\
                 key TEXT PRIMARY KEY,\
                 value TEXT NOT NULL,\
                 updatedAt INTEGER NOT NULL\
             );\
             CREATE TABLE IF NOT EXISTS gmail_accounts (\
                 id TEXT PRIMARY KEY,\
                 connectionId TEXT NOT NULL,\
                 email TEXT,\
                 label TEXT,\
                 entityId TEXT NOT NULL,\
                 active INTEGER DEFAULT 1,\
                 createdAt INTEGER NOT NULL\
             );\
             CREATE TABLE IF NOT EXISTS emails (\
                 messageId TEXT PRIMARY KEY,\
                 threadId TEXT,\
                 fromEmail TEXT NOT NULL,\
                 subject TEXT NOT NULL,\
                 body TEXT,\
                 receivedAt TEXT,\
                 accountEntityId TEXT,\
                 firstSeenAt INTEGER NOT NULL,\
                 triageResult TEXT,\
                 agentProcessedAt INTEGER,\
                 platform TEXT NOT NULL DEFAULT 'gmail',\
                 kind TEXT NOT NULL DEFAULT 'dm'\
             );\
             CREATE TABLE IF NOT EXISTS channel_subscriptions (\
                 id                   TEXT PRIMARY KEY,\
                 platform             TEXT NOT NULL,\
                 channel_id           TEXT NOT NULL,\
                 display_name         TEXT NOT NULL,\
                 mode                 TEXT NOT NULL,\
                 active               INTEGER NOT NULL DEFAULT 1,\
                 account_id           TEXT,\
                 last_seen_message_id TEXT,\
                 last_digest_at_ms    INTEGER,\
                 created_at_ms        INTEGER NOT NULL,\
                 updated_at_ms        INTEGER NOT NULL\
             );\
             CREATE TABLE IF NOT EXISTS slack_workspaces (\
                 id              TEXT PRIMARY KEY,\
                 team_id         TEXT NOT NULL UNIQUE,\
                 team_name       TEXT NOT NULL,\
                 entity_id       TEXT NOT NULL,\
                 connection_id   TEXT NOT NULL,\
                 user_id         TEXT NOT NULL,\
                 active          INTEGER NOT NULL DEFAULT 1,\
                 created_at_ms   INTEGER NOT NULL\
             );\
             CREATE TABLE IF NOT EXISTS drive_accounts (\
                 id            TEXT PRIMARY KEY,\
                 connection_id TEXT NOT NULL,\
                 entity_id     TEXT NOT NULL,\
                 email         TEXT,\
                 label         TEXT,\
                 active        INTEGER NOT NULL DEFAULT 1,\
                 created_at_ms INTEGER NOT NULL\
             );\
             CREATE TABLE IF NOT EXISTS drive_sync_state (\
                 entity_id     TEXT PRIMARY KEY,\
                 page_token    TEXT NOT NULL,\
                 updated_at_ms INTEGER NOT NULL\
             );",
        )?;

        if !column_exists(conn, "actions", "retryCount")? {
            conn.execute("ALTER TABLE actions ADD COLUMN retryCount INTEGER DEFAULT 0", [])?;
        }
        if !column_exists(conn, "actions", "draftId")? {
            conn.execute("ALTER TABLE actions ADD COLUMN draftId TEXT", [])?;
        }
        if !column_exists(conn, "actions", "nudgeCount")? {
            conn.execute(
                "ALTER TABLE actions ADD COLUMN nudgeCount INTEGER NOT NULL DEFAULT 0",
                [],
            )?;
        }
        // #34: quick-refine analytics + redraft-iteration cap. `lastPresetId`
        // records the most-recent canned preset chosen (NULL for free-form
        // Revise / never refined); `redraftCount` is the stacked-iteration
        // counter the approval card uses to enforce MAX_REDRAFT_ITERATIONS.
        if !column_exists(conn, "actions", "lastPresetId")? {
            conn.execute("ALTER TABLE actions ADD COLUMN lastPresetId TEXT", [])?;
        }
        if !column_exists(conn, "actions", "redraftCount")? {
            conn.execute(
                "ALTER TABLE actions ADD COLUMN redraftCount INTEGER NOT NULL DEFAULT 0",
                [],
            )?;
        }
        // #473: the outbound envelope of a compose-originated card, so the
        // Revise redraft can re-send to the SAME recipients instead of
        // falling back to `emails.from` (which is the original sender on
        // reply cards — dropping an overridden To and any cc/bcc). All three
        // are comma-joined bare-address lists; NULL on rows from flows that
        // never set an envelope (auto-triage replies, non-gmail platforms),
        // which keep the pre-#473 from-based behavior.
        if !column_exists(conn, "actions", "toEmails")? {
            conn.execute("ALTER TABLE actions ADD COLUMN toEmails TEXT", [])?;
        }
        if !column_exists(conn, "actions", "ccEmails")? {
            conn.execute("ALTER TABLE actions ADD COLUMN ccEmails TEXT", [])?;
        }
        if !column_exists(conn, "actions", "bccEmails")? {
            conn.execute("ALTER TABLE actions ADD COLUMN bccEmails TEXT", [])?;
        }
        // #652: the subject the outgoing draft actually carries, once a Revise
        // has changed it. NULL = never overridden, which keeps the derived
        // `Re: <inbound subject>` behavior.
        if !column_exists(conn, "actions", "envelopeSubject")? {
            conn.execute("ALTER TABLE actions ADD COLUMN envelopeSubject TEXT", [])?;
        }
        if !column_exists(conn, "actions", "nextNudgeAtMs")? {
            conn.execute("ALTER TABLE actions ADD COLUMN nextNudgeAtMs INTEGER", [])?;
            // One-shot backfill: rows still in 'pending' from before the
            // nudge-loop ship need a timer or they'll never get reminded.
            // Seeded at createdAt + 6h, matching the steady-state invariant
            // log_action maintains for fresh rows. Old backlog items will
            // therefore fire on the first scheduler tick after upgrade.
            conn.execute(
                "UPDATE actions \
                    SET nextNudgeAtMs = createdAt + ?1 \
                  WHERE status = 'pending' AND nextNudgeAtMs IS NULL",
                params![NUDGE_INTERVAL_MS],
            )?;
        }
        // #500 — scheduled email send. `scheduledAtMs` is authoritative only
        // while status is 'pending' (a --send-at proposal awaiting approval)
        // or 'scheduled' (armed); terminal rows may retain it as audit.
        // `noticeChannelId`/`noticeMessageId` point at the Discord
        // scheduled-notice message so the fire/cancel paths can delete it —
        // actions.messageId is the INBOUND email id and must not be reused
        // for Discord ids (#449 id-space lesson).
        if !column_exists(conn, "actions", "scheduledAtMs")? {
            conn.execute("ALTER TABLE actions ADD COLUMN scheduledAtMs INTEGER", [])?;
        }
        if !column_exists(conn, "actions", "noticeChannelId")? {
            conn.execute("ALTER TABLE actions ADD COLUMN noticeChannelId TEXT", [])?;
        }
        if !column_exists(conn, "actions", "noticeMessageId")? {
            conn.execute("ALTER TABLE actions ADD COLUMN noticeMessageId TEXT", [])?;
        }
        // Mirrors idx_scheduled_posts_fire: the engine's due query is
        // `status = 'scheduled' AND scheduledAtMs <= now`.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_actions_scheduled \
                 ON actions(status, scheduledAtMs)",
            [],
        )?;
        if !column_exists(conn, "emails", "platform")? {
            conn.execute(
                "ALTER TABLE emails ADD COLUMN platform TEXT NOT NULL DEFAULT 'gmail'",
                [],
            )?;
            // One-shot backfill for pre-platform-column rows: any row whose
            // accountEntityId looks like a LinkedIn URN is tagged 'linkedin'.
            // Safe to run once at column-add time — fresh rows from the channels
            // write their platform directly.
            conn.execute(
                "UPDATE emails SET platform = 'linkedin' WHERE accountEntityId LIKE 'urn:li:%'",
                [],
            )?;
        }
        if !column_exists(conn, "emails", "kind")? {
            conn.execute(
                "ALTER TABLE emails ADD COLUMN kind TEXT NOT NULL DEFAULT 'dm'",
                [],
            )?;
        }
        // Issue #27: per-channel subscription registry (platform-agnostic).
        // Rows control which Discord/Slack/etc channels the poller watches and
        // which mode (priority / digest / store_only) they route through.
        // Uniqueness is enforced at the upsert layer (platform, channel_id,
        // account_id) rather than via a SQL UNIQUE so multi-workspace Slack
        // can carry the same channel id across teams.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS channel_subscriptions (\
                 id                   TEXT PRIMARY KEY,\
                 platform             TEXT NOT NULL,\
                 channel_id           TEXT NOT NULL,\
                 display_name         TEXT NOT NULL,\
                 mode                 TEXT NOT NULL,\
                 active               INTEGER NOT NULL DEFAULT 1,\
                 account_id           TEXT,\
                 last_seen_message_id TEXT,\
                 last_digest_at_ms    INTEGER,\
                 created_at_ms        INTEGER NOT NULL,\
                 updated_at_ms        INTEGER NOT NULL\
             )",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_channel_subs_active_mode \
                ON channel_subscriptions(active, mode)",
            [],
        )?;
        // Multi-workspace Slack: each subscription may belong to a specific
        // account (for Slack, the workspace `team_id`). Nullable so existing
        // Discord rows migrate cleanly.
        if !column_exists(conn, "channel_subscriptions", "account_id")? {
            conn.execute(
                "ALTER TABLE channel_subscriptions ADD COLUMN account_id TEXT",
                [],
            )?;
        }
        // Older DBs still carry the legacy UNIQUE(platform, channel_id). Detect
        // it and rebuild without the constraint so multi-workspace rows can
        // coexist. SQLite can't ALTER away a constraint in place.
        if table_has_unique(conn, "channel_subscriptions", "platform", "channel_id")? {
            conn.execute_batch(
                "BEGIN TRANSACTION;\n\
                 CREATE TABLE channel_subscriptions_new (\
                   id                   TEXT PRIMARY KEY,\
                   platform             TEXT NOT NULL,\
                   channel_id           TEXT NOT NULL,\
                   display_name         TEXT NOT NULL,\
                   mode                 TEXT NOT NULL,\
                   active               INTEGER NOT NULL DEFAULT 1,\
                   account_id           TEXT,\
                   last_seen_message_id TEXT,\
                   last_digest_at_ms    INTEGER,\
                   created_at_ms        INTEGER NOT NULL,\
                   updated_at_ms        INTEGER NOT NULL\
                 );\n\
                 INSERT INTO channel_subscriptions_new \
                   (id, platform, channel_id, display_name, mode, active, \
                    account_id, last_seen_message_id, last_digest_at_ms, \
                    created_at_ms, updated_at_ms) \
                   SELECT id, platform, channel_id, display_name, mode, active, \
                          account_id, last_seen_message_id, last_digest_at_ms, \
                          created_at_ms, updated_at_ms \
                     FROM channel_subscriptions;\n\
                 DROP TABLE channel_subscriptions;\n\
                 ALTER TABLE channel_subscriptions_new RENAME TO channel_subscriptions;\n\
                 CREATE INDEX IF NOT EXISTS idx_channel_subs_active_mode \
                   ON channel_subscriptions(active, mode);\n\
                 COMMIT;",
            )?;
        }
        conn.execute(
            "CREATE TABLE IF NOT EXISTS slack_workspaces (\
                 id              TEXT PRIMARY KEY,\
                 team_id         TEXT NOT NULL UNIQUE,\
                 team_name       TEXT NOT NULL,\
                 entity_id       TEXT NOT NULL,\
                 connection_id   TEXT NOT NULL,\
                 user_id         TEXT NOT NULL,\
                 active          INTEGER NOT NULL DEFAULT 1,\
                 created_at_ms   INTEGER NOT NULL\
             )",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_slack_workspaces_active \
                ON slack_workspaces(active)",
            [],
        )?;

        // ----------------------------------------------------------------
        // Wave-A foundation: tables for the parallel feature PRs branching
        // off `foundation/swarm-v1`. Schemas are pulled verbatim from each
        // research issue body. Tables are independent — order doesn't matter.
        // Every CREATE is `IF NOT EXISTS`, so re-running migrate is a no-op.
        // ----------------------------------------------------------------

        // #73 — per-recipient tone-mirroring v1.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS tone_profiles (\
                 id                       TEXT PRIMARY KEY,\
                 scope_kind               TEXT NOT NULL CHECK (scope_kind IN ('global','domain','recipient')),\
                 scope_value              TEXT NOT NULL,\
                 account_entity_id        TEXT,\
                 summary                  TEXT NOT NULL,\
                 exemplar_ids             TEXT NOT NULL DEFAULT '[]',\
                 sample_count             INTEGER NOT NULL DEFAULT 0,\
                 sample_count_at_refresh  INTEGER NOT NULL DEFAULT 0,\
                 last_refreshed_at        INTEGER NOT NULL,\
                 created_at_ms            INTEGER NOT NULL,\
                 updated_at_ms            INTEGER NOT NULL,\
                 UNIQUE(scope_kind, scope_value, account_entity_id)\
             )",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tone_profiles_scope \
                ON tone_profiles(scope_kind, scope_value)",
            [],
        )?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS tone_examples (\
                 id                  TEXT PRIMARY KEY,\
                 source              TEXT NOT NULL CHECK (source IN ('sent_backfill','user_edit','approved_clean')),\
                 action_id           TEXT,\
                 message_id          TEXT,\
                 account_entity_id   TEXT NOT NULL,\
                 recipient_email     TEXT NOT NULL,\
                 recipient_domain    TEXT NOT NULL,\
                 subject             TEXT,\
                 body                TEXT NOT NULL,\
                 body_chars          INTEGER NOT NULL,\
                 sent_at_ms          INTEGER NOT NULL,\
                 ingested_at_ms      INTEGER NOT NULL,\
                 weight              REAL NOT NULL DEFAULT 1.0,\
                 FOREIGN KEY (action_id) REFERENCES actions(id) ON DELETE SET NULL\
             )",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tone_examples_recipient \
                ON tone_examples(recipient_email, sent_at_ms DESC)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tone_examples_domain \
                ON tone_examples(recipient_domain, sent_at_ms DESC)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tone_examples_account_recent \
                ON tone_examples(account_entity_id, sent_at_ms DESC)",
            [],
        )?;

        // #37 — draft revision history for tone learning + eval.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS draft_revisions (\
                 id                 TEXT PRIMARY KEY,\
                 actionId           TEXT NOT NULL REFERENCES actions(id) ON DELETE CASCADE,\
                 iteration          INTEGER NOT NULL,\
                 draftBody          TEXT NOT NULL,\
                 feedbackText       TEXT,\
                 presetId           TEXT,\
                 outcome            TEXT NOT NULL,\
                 modelId            TEXT NOT NULL,\
                 promptTokens       INTEGER,\
                 completionTokens   INTEGER,\
                 createdAt          INTEGER NOT NULL,\
                 UNIQUE(actionId, iteration)\
             )",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_draft_revisions_outcome \
                ON draft_revisions(outcome, createdAt)",
            [],
        )?;

        // #83 — RateGovernor (per-platform rate events, halts, warmup).
        conn.execute(
            "CREATE TABLE IF NOT EXISTS rate_events (\
                 id              TEXT PRIMARY KEY,\
                 platform        TEXT NOT NULL,\
                 action_kind     TEXT NOT NULL,\
                 account_id      TEXT NOT NULL,\
                 occurred_at_ms  INTEGER NOT NULL,\
                 status          TEXT NOT NULL,\
                 cause           TEXT NOT NULL,\
                 target_id       TEXT,\
                 meta_json       TEXT\
             )",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rate_events_window \
                ON rate_events(platform, action_kind, account_id, occurred_at_ms)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rate_events_audit \
                ON rate_events(platform, occurred_at_ms)",
            [],
        )?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS rate_halts (\
                 platform               TEXT PRIMARY KEY,\
                 paused_until_ms        INTEGER NOT NULL,\
                 reason                 TEXT NOT NULL,\
                 triggered_by_event_id  TEXT,\
                 created_at_ms          INTEGER NOT NULL,\
                 acknowledged_at_ms     INTEGER\
             )",
            [],
        )?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS rate_warmup (\
                 platform              TEXT NOT NULL,\
                 account_id            TEXT NOT NULL,\
                 warmup_started_at_ms  INTEGER NOT NULL,\
                 PRIMARY KEY (platform, account_id)\
             )",
            [],
        )?;

        // #74 — Telegram Bot API per-bot state (long-poll cursor).
        conn.execute(
            "CREATE TABLE IF NOT EXISTS telegram_bots (\
                 id              TEXT PRIMARY KEY,\
                 bot_id          INTEGER NOT NULL UNIQUE,\
                 bot_username    TEXT NOT NULL,\
                 owner_chat_id   INTEGER NOT NULL,\
                 last_update_id  INTEGER NOT NULL DEFAULT 0,\
                 active          INTEGER NOT NULL DEFAULT 1,\
                 created_at_ms   INTEGER NOT NULL\
             )",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_telegram_bots_active \
                ON telegram_bots(active)",
            [],
        )?;

        // #74 — WhatsApp linked devices + per-chat outbound/inbound allowlists.
        // Inbound allowlist comes from review feedback: even reading a chat
        // requires explicit opt-in for ban-risk reasons.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS whatsapp_devices (\
                 id                TEXT PRIMARY KEY,\
                 phone             TEXT NOT NULL UNIQUE,\
                 device_jid        TEXT NOT NULL,\
                 user_jid          TEXT NOT NULL,\
                 paired_at_ms      INTEGER NOT NULL,\
                 last_event_at_ms  INTEGER NOT NULL DEFAULT 0,\
                 session_status    TEXT NOT NULL DEFAULT 'paired',\
                 active            INTEGER NOT NULL DEFAULT 1,\
                 created_at_ms     INTEGER NOT NULL\
             )",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_whatsapp_devices_active \
                ON whatsapp_devices(active)",
            [],
        )?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS whatsapp_outbound_allowlist (\
                 chat_jid       TEXT PRIMARY KEY,\
                 enabled_at_ms  INTEGER NOT NULL\
             )",
            [],
        )?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS whatsapp_inbound_allowlist (\
                 chat_jid       TEXT PRIMARY KEY,\
                 enabled_at_ms  INTEGER NOT NULL\
             )",
            [],
        )?;

        // #104 — user-defined scheduled tasks (`/loop`). Channel-agnostic:
        // `channel` is the surface the loop was created from (`discord` today)
        // and `channel_ref` is the originating channel/DM id the scheduler
        // posts results back to. `interval_secs` is enforced against a floor at
        // the command layer; `fail_count` drives pause-on-repeated-failure.
        // `status` is `active` | `paused` | `stopped`. Survives restarts — the
        // scheduler reloads `active` rows on boot.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS user_loops (\
                 id             TEXT PRIMARY KEY,\
                 owner          TEXT NOT NULL,\
                 channel        TEXT NOT NULL,\
                 channel_ref    TEXT NOT NULL,\
                 interval_secs  INTEGER NOT NULL,\
                 prompt         TEXT NOT NULL,\
                 status         TEXT NOT NULL DEFAULT 'active',\
                 last_run_ms    INTEGER,\
                 last_status    TEXT,\
                 fail_count     INTEGER NOT NULL DEFAULT 0,\
                 created_at_ms  INTEGER NOT NULL,\
                 updated_at_ms  INTEGER NOT NULL,\
                 expires_at_ms  INTEGER\
             )",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_user_loops_owner_status \
                ON user_loops(owner, status)",
            [],
        )?;
        // Auto-stop deadline (#104 follow-up). Nullable; pre-existing rows
        // migrate cleanly with `None` (run forever until manually stopped).
        if !column_exists(conn, "user_loops", "expires_at_ms")? {
            conn.execute(
                "ALTER TABLE user_loops ADD COLUMN expires_at_ms INTEGER",
                [],
            )?;
        }
        // #231 — cron-style scheduling. When `cron_expr` is set,
        // `interval_secs` is ignored and `tz` is required for next-firing
        // computation. Both nullable so existing interval-only rows
        // migrate cleanly. The scheduler tick picks the cron branch
        // whenever `cron_expr.is_some()`.
        if !column_exists(conn, "user_loops", "cron_expr")? {
            conn.execute(
                "ALTER TABLE user_loops ADD COLUMN cron_expr TEXT",
                [],
            )?;
        }
        if !column_exists(conn, "user_loops", "tz")? {
            conn.execute(
                "ALTER TABLE user_loops ADD COLUMN tz TEXT",
                [],
            )?;
        }

        // #47 — cross-surface state sync. `status_source` records which surface
        // resolved an action (discord / dashboard / telegram / cli / nudge) so
        // the originating surface can suppress its own echo;
        // `status_updated_at` timestamps the last transition for the SSE feed.
        // Nullable so pre-existing rows migrate cleanly.
        if !column_exists(conn, "actions", "status_source")? {
            conn.execute("ALTER TABLE actions ADD COLUMN status_source TEXT", [])?;
        }
        if !column_exists(conn, "actions", "status_updated_at")? {
            conn.execute(
                "ALTER TABLE actions ADD COLUMN status_updated_at INTEGER",
                [],
            )?;
        }

        // #48 — Code Mode for AugmentAgent. Three additive columns on `actions`
        // so a code-mode draft can persist the generated TypeScript program and
        // its tool-call trace next to the existing draft body. `mode` is the
        // discriminator — `'classic'` (the existing one-shot draft prompt) or
        // `'code'` (the Cloudflare-style TS program executed in the Deno
        // sidecar); defaults to `'classic'` so pre-existing rows + every classic
        // path call site stay byte-identical. `generatedSource` is the full TS
        // program (only populated when `mode = 'code'`, or when a failed
        // code-mode program is recorded after fallback for audit); nullable for
        // classic rows. `toolCallTrace` is a JSON array of
        // `{call, args_summary, result_summary, error?}` records (see
        // `ToolCallRecord` in `augmentagent-channel-core`). Nullable for
        // classic rows.
        if !column_exists(conn, "actions", "mode")? {
            conn.execute(
                "ALTER TABLE actions ADD COLUMN mode TEXT NOT NULL DEFAULT 'classic'",
                [],
            )?;
        }
        if !column_exists(conn, "actions", "generatedSource")? {
            conn.execute("ALTER TABLE actions ADD COLUMN generatedSource TEXT", [])?;
        }
        if !column_exists(conn, "actions", "toolCallTrace")? {
            conn.execute("ALTER TABLE actions ADD COLUMN toolCallTrace TEXT", [])?;
        }

        // #179 — Liveness signal for `gmail_accounts`. `active=1` only tells
        // us the operator connected this account at some point, NOT that
        // the connection still works at Composio (project switches, key
        // rotations, revoked OAuth grants all leave `active=1` forever).
        // The Rust gmail poller writes these on every cycle so the
        // dashboard can compute "actually connected" = `active=1` AND
        // `lastPollOk=1` AND `now - lastPolledAt < STALE_MS`. Column names
        // are camelCase to match the rest of the `gmail_accounts` table
        // (`connectionId`, `entityId`, `createdAt`).
        if !column_exists(conn, "gmail_accounts", "lastPolledAt")? {
            conn.execute(
                "ALTER TABLE gmail_accounts ADD COLUMN lastPolledAt INTEGER",
                [],
            )?;
        }
        if !column_exists(conn, "gmail_accounts", "lastPollOk")? {
            conn.execute(
                "ALTER TABLE gmail_accounts ADD COLUMN lastPollOk INTEGER",
                [],
            )?;
        }

        // #797 — Composio returns Gmail user rate limits in an HTTP-success
        // envelope. This is intentionally separate from account liveness: a
        // cooling account remains connected and recovers automatically when
        // the provider's window ends.
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS gmail_fetch_cooldowns (\
                 entityId TEXT PRIMARY KEY,\
                 retryAfterMs INTEGER NOT NULL,\
                 logId TEXT,\
                 observedAtMs INTEGER NOT NULL\
             );",
        )?;

        // #81 — Proactive CRM signals + per-scan run cursor.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS proactive_signals (\
                 id                     TEXT PRIMARY KEY,\
                 kind                   TEXT NOT NULL,\
                 person_slug            TEXT,\
                 urgency                TEXT NOT NULL,\
                 headline               TEXT NOT NULL,\
                 detail                 TEXT NOT NULL,\
                 suggested_action_json  TEXT,\
                 status                 TEXT NOT NULL,\
                 snooze_until_ms        INTEGER,\
                 dedup_key              TEXT NOT NULL,\
                 created_at_ms          INTEGER NOT NULL,\
                 dispatched_at_ms       INTEGER\
             )",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_proactive_signals_status_created \
                ON proactive_signals(status, created_at_ms)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_proactive_signals_dedup_recent \
                ON proactive_signals(dedup_key, created_at_ms)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_proactive_signals_person \
                ON proactive_signals(person_slug)",
            [],
        )?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS proactive_scan_runs (\
                 scan_id         TEXT PRIMARY KEY,\
                 last_run_at_ms  INTEGER NOT NULL\
             )",
            [],
        )?;

        // #79 — Twitter/X GraphQL queryId cache (rotated by X every 2-6 wk).
        conn.execute(
            "CREATE TABLE IF NOT EXISTS twitter_query_ids (\
                 operation      TEXT PRIMARY KEY,\
                 query_id       TEXT NOT NULL,\
                 last_seen_at   INTEGER NOT NULL\
             )",
            [],
        )?;

        // #79 — Twitter/X outbound posting audit log. Drives the hard
        // 15-posts/day quota preflight (separate from the #83 RateGovernor
        // soft caps — this is the platform's own free-tier ceiling).
        conn.execute(
            "CREATE TABLE IF NOT EXISTS twitter_post_log (\
                 id              TEXT PRIMARY KEY,\
                 kind            TEXT NOT NULL,\
                 reply_to        TEXT,\
                 status          TEXT NOT NULL,\
                 tweet_id        TEXT,\
                 occurred_at_ms  INTEGER NOT NULL,\
                 meta_json       TEXT\
             )",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_twitter_post_log_window \
                ON twitter_post_log(occurred_at_ms)",
            [],
        )?;

        // #77 — LinkedIn outbound action audit log (post / comment / like /
        // connection_invite / dm / profile_view), drives daily/hourly caps.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS linkedin_action_log (\
                 id              TEXT PRIMARY KEY,\
                 action_kind     TEXT NOT NULL,\
                 target_urn      TEXT,\
                 status          TEXT NOT NULL,\
                 occurred_at_ms  INTEGER NOT NULL,\
                 meta_json       TEXT\
             )",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_linkedin_action_log_window \
                ON linkedin_action_log(action_kind, occurred_at_ms)",
            [],
        )?;

        // #58 / #74-engagement — scheduled outbound posts (cross-platform).
        conn.execute(
            "CREATE TABLE IF NOT EXISTS scheduled_posts (\
                 id              TEXT PRIMARY KEY,\
                 platform        TEXT NOT NULL,\
                 body            TEXT NOT NULL,\
                 media_paths     TEXT,\
                 fire_at_ms      INTEGER NOT NULL,\
                 status          TEXT NOT NULL,\
                 approval_msg    TEXT,\
                 posted_at_ms    INTEGER,\
                 external_id     TEXT,\
                 thread_parent   TEXT REFERENCES scheduled_posts(id) ON DELETE SET NULL,\
                 created_at_ms   INTEGER NOT NULL\
             )",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_scheduled_posts_fire \
                ON scheduled_posts(status, fire_at_ms)",
            [],
        )?;
        // #240 — SocialAPI.ai outbound. When set, the publisher routes the row
        // through SocialAPI.ai (platform = "socialapi") to this connected
        // account; `platform` then carries the real sub-platform
        // (instagram / x / linkedin / …) so the publisher knows the target.
        // Nullable + additive: pre-existing rows migrate cleanly with `None`
        // and keep using the native LinkedIn / Twitter arms. Multi-account
        // fan-out (N rows) is #241.
        if !column_exists(conn, "scheduled_posts", "socialapi_account_id")? {
            conn.execute(
                "ALTER TABLE scheduled_posts ADD COLUMN socialapi_account_id TEXT",
                [],
            )?;
        }

        // ----------------------------------------------------------------
        // #58 — engagement-automation spine. Additive + dormant in prod:
        // empty + unwritten unless the engagement loops / CLI populate them.
        // Same proven-safe pattern as the wave-A tables above. Schemas
        // mirror the #58 research-issue body.
        // ----------------------------------------------------------------

        // #58.2 — the user's own watched posts + already-seen comment ids.
        // `OwnPostsSource` polls the last N posts and diffs incoming
        // comments against `seen_comments` so a new comment becomes one
        // `own_post_comment` WorkItem exactly once.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS own_posts (\
                 id            TEXT PRIMARY KEY,\
                 platform      TEXT NOT NULL,\
                 external_id   TEXT NOT NULL,\
                 posted_at_ms  INTEGER NOT NULL,\
                 poll_until_ms INTEGER NOT NULL,\
                 last_polled_ms INTEGER,\
                 created_at_ms INTEGER NOT NULL,\
                 UNIQUE (platform, external_id)\
             )",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_own_posts_poll \
                ON own_posts(platform, poll_until_ms)",
            [],
        )?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS seen_comments (\
                 id            TEXT PRIMARY KEY,\
                 own_post_id   TEXT NOT NULL REFERENCES own_posts(id) ON DELETE CASCADE,\
                 external_id   TEXT NOT NULL,\
                 author_handle TEXT,\
                 body          TEXT,\
                 triage_id     TEXT,\
                 created_at_ms INTEGER NOT NULL,\
                 UNIQUE (own_post_id, external_id)\
             )",
            [],
        )?;

        // #58.3 — friend watchlist + seen friend posts. `engagement` is
        // 'high' (every post) | 'medium' (weekly digest) | 'low' (only on
        // milestone keywords). `wiki_slug` grounds the draft prompt.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS friend_watchlist (\
                 id            TEXT PRIMARY KEY,\
                 platform      TEXT NOT NULL,\
                 handle        TEXT NOT NULL,\
                 wiki_slug     TEXT,\
                 engagement    TEXT NOT NULL DEFAULT 'medium',\
                 added_at_ms   INTEGER NOT NULL,\
                 paused_until_ms INTEGER,\
                 UNIQUE (platform, handle)\
             )",
            [],
        )?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS friend_posts_seen (\
                 id            TEXT PRIMARY KEY,\
                 watchlist_id  TEXT NOT NULL REFERENCES friend_watchlist(id) ON DELETE CASCADE,\
                 external_id   TEXT NOT NULL,\
                 posted_at_ms  INTEGER NOT NULL,\
                 triage_id     TEXT,\
                 UNIQUE (watchlist_id, external_id)\
             )",
            [],
        )?;

        // #58.4 — inbound LinkedIn (and future-platform) connection-request
        // triage queue. `decision` is one of
        // accept|decline|accept_and_dm|pending.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS connection_requests (\
                 id              TEXT PRIMARY KEY,\
                 platform        TEXT NOT NULL,\
                 external_id     TEXT NOT NULL,\
                 requester_name  TEXT,\
                 requester_url   TEXT,\
                 message         TEXT,\
                 decision        TEXT NOT NULL DEFAULT 'pending',\
                 decided_at_ms   INTEGER,\
                 triage_id       TEXT,\
                 created_at_ms   INTEGER NOT NULL,\
                 UNIQUE (platform, external_id)\
             )",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_connection_requests_decision \
                ON connection_requests(decision, created_at_ms)",
            [],
        )?;

        // #58.5 — warm-touch per-contact state. The actual cadence scoring
        // is the merged #81 proactive `StaleContactScan`; this table only
        // carries the per-slug nudge/snooze bookkeeping the #58 card needs
        // (so we never duplicate the proactive scoring engine).
        conn.execute(
            "CREATE TABLE IF NOT EXISTS warm_touch_state (\
                 wiki_slug            TEXT PRIMARY KEY,\
                 last_interaction_ms  INTEGER,\
                 last_nudged_ms       INTEGER,\
                 snoozed_until_ms     INTEGER,\
                 cadence_days         INTEGER\
             )",
            [],
        )?;

        // Multi-tenant Google Drive (Composio). Inert in prod: empty + unread
        // unless a tenant connects a Drive account. Same proven-safe pattern
        // as the dormant wave-A tables already shipping in prod.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS drive_accounts (\
                 id            TEXT PRIMARY KEY,\
                 connection_id TEXT NOT NULL,\
                 entity_id     TEXT NOT NULL,\
                 email         TEXT,\
                 label         TEXT,\
                 active        INTEGER NOT NULL DEFAULT 1,\
                 created_at_ms INTEGER NOT NULL\
             )",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_drive_accounts_active \
                ON drive_accounts(active)",
            [],
        )?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS drive_sync_state (\
                 entity_id     TEXT PRIMARY KEY,\
                 page_token    TEXT NOT NULL,\
                 updated_at_ms INTEGER NOT NULL\
             )",
            [],
        )?;

        // #427 — ShadowNote journal DataStore delta-sync watermark. One row
        // per ownerId; stores the `startedAt` epoch-ms the last completed
        // sync returned, passed back as `lastSync` so a restart only
        // re-ingests entries changed since the previous full pass.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS journal_sync_state (\
                 owner_id      TEXT PRIMARY KEY,\
                 last_sync_ms  INTEGER NOT NULL,\
                 updated_at_ms INTEGER NOT NULL\
             )",
            [],
        )?;

        // #900 — in-progress journal sync pass (resumable pagination cursor)
        // and the per-(entry, _version) ingest ledger that makes a replayed
        // page set idempotent. Both keyed by ownerId like journal_sync_state.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS journal_sync_cursor (\
                 owner_id      TEXT PRIMARY KEY,\
                 last_sync_ms  INTEGER,\
                 started_at_ms INTEGER NOT NULL,\
                 next_token    TEXT,\
                 updated_at_ms INTEGER NOT NULL\
             )",
            [],
        )?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS journal_ingested (\
                 owner_id       TEXT NOT NULL,\
                 entry_id       TEXT NOT NULL,\
                 version        INTEGER NOT NULL,\
                 ingested_at_ms INTEGER NOT NULL,\
                 PRIMARY KEY (owner_id, entry_id, version)\
             )",
            [],
        )?;

        // #886 — iMessage bundle incremental cursor. One row per
        // conversation identifier; stores how many entries of that
        // conversation's append-only messages.md have been ingested, so a
        // poll only processes the tail.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS imessage_sync_state (\
                 conversation  TEXT PRIMARY KEY,\
                 entries_seen  INTEGER NOT NULL,\
                 updated_at_ms INTEGER NOT NULL\
             )",
            [],
        )?;

        // #80 — voice-capture Telegram long-poll cursor. Single-row table
        // keyed by a logical capture-bot id; stores the last acked update_id
        // so a daemon restart never re-ingests an already-transcribed memo.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS voice_capture_state (\
                 bot_key         TEXT PRIMARY KEY,\
                 last_update_id  INTEGER NOT NULL DEFAULT 0,\
                 updated_at_ms   INTEGER NOT NULL\
             )",
            [],
        )?;

        // #396/#397 — calendar alert dedup ledger. One row per alert key
        // (`upcoming:<event_id>`, `conflict:<idA>:<idB>`, `agenda:<date>`);
        // the fingerprint captures the event time(s) the alert described,
        // so a reschedule re-arms the alert while a re-poll stays silent.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS calendar_alerts (\
                 key          TEXT PRIMARY KEY,\
                 fingerprint  TEXT NOT NULL,\
                 sent_at_ms   INTEGER NOT NULL\
             )",
            [],
        )?;

        // #219 — outbound-mail observer cursor. One row per Gmail account
        // entity; `last_seen_sent_at_ms` is the high-water timestamp of the
        // newest SENT message we've already classified, so a daemon restart
        // never re-emits an OutboundEvent for the same reply. Mirrors the
        // `voice_capture_state` single-row-per-source shape.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS outbound_state (\
                 entity_id              TEXT PRIMARY KEY,\
                 last_seen_sent_at_ms   INTEGER NOT NULL DEFAULT 0,\
                 updated_at_ms          INTEGER NOT NULL\
             )",
            [],
        )?;

        // #218 — per-thread log of outbound messages the OutboundObserver has
        // classified. Lets the inbound triage gate cheaply answer "has the
        // user already replied on this thread after timestamp X?" without a
        // live Gmail API call. One row per (entity_id, message_id); duplicates
        // are ignored (idempotent re-poll). The cheap-signal companion to
        // `outbound_state`'s cursor. Index on (thread_id, sent_at_ms) makes
        // the "any newer reply?" query a covered point-lookup.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS outbound_thread_log (\
                 entity_id    TEXT NOT NULL,\
                 message_id   TEXT NOT NULL,\
                 thread_id    TEXT,\
                 sent_at_ms   INTEGER NOT NULL,\
                 recorded_at_ms INTEGER NOT NULL,\
                 PRIMARY KEY (entity_id, message_id)\
             )",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_outbound_thread_log_thread_sent \
             ON outbound_thread_log (thread_id, sent_at_ms)",
            [],
        )?;

        // #449 — every message the DAEMON itself put in the SENT folder, keyed
        // by the Gmail message id the send call returned. The OutboundObserver
        // scans SENT to detect replies the *user* wrote in Gmail web/mobile;
        // without this table it cannot tell those apart from its own sends.
        //
        // The pre-#449 code tried to do that by matching SENT-folder ids
        // against `actions.messageId`, but that column holds the *inbound*
        // email's id — the two id spaces never intersect, so every agent reply
        // was misread as a manual user reply. That made enabling the observer
        // actively harmful (it would supersede live drafts and suppress
        // drafting on any thread the agent had ever answered), which is why it
        // shipped default-OFF and stayed off.
        //
        // `thread_id` + `sent_at_ms` back the proximity fallback used when
        // Composio's send response omits the message id (see
        // `has_daemon_send_near`), so a decoding gap degrades to "don't
        // supersede" rather than "corrupt the user-reply signal".
        conn.execute(
            "CREATE TABLE IF NOT EXISTS self_sent_messages (\
                 message_id   TEXT PRIMARY KEY,\
                 thread_id    TEXT,\
                 entity_id    TEXT,\
                 action_id    TEXT,\
                 sent_at_ms   INTEGER NOT NULL\
             )",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_self_sent_messages_thread_sent \
             ON self_sent_messages (thread_id, sent_at_ms)",
            [],
        )?;

        // #57 — proactive-nudge user actions. One row per user gesture
        // (snooze a signal, dismiss it, mute a person, mute a rule). The
        // proactive runner read-throughs this before dispatch; the dashboard
        // /relationships page writes it. `scope` is the target the action
        // applies to: a signal id, a person slug, or a rule kind. NULL
        // expires_at = permanent.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS proactive_user_actions (\
                 id            TEXT PRIMARY KEY,\
                 action        TEXT NOT NULL,\
                 scope         TEXT NOT NULL,\
                 created_at_ms INTEGER NOT NULL,\
                 expires_at_ms INTEGER\
             )",
            [],
        )?;

        // #45 — Web Push subscriptions for the PWA approval surface. One row
        // per browser push endpoint; `p256dh`/`auth` are the VAPID client
        // keys. Inert until the user installs the PWA + grants notifications.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS pwa_subscriptions (\
                 id            TEXT PRIMARY KEY,\
                 endpoint      TEXT NOT NULL UNIQUE,\
                 p256dh        TEXT NOT NULL,\
                 auth          TEXT NOT NULL,\
                 created_at_ms INTEGER NOT NULL\
             )",
            [],
        )?;

        // ---------------------------------------------------------------
        // CRM ingestion (#61 LinkedIn connections / #62 contacts / #64
        // signature backfill). All additive + dormant in prod until the
        // respective CLI command runs; same proven-safe pattern as the
        // wave-A tables above.
        // ---------------------------------------------------------------

        // #61 — LinkedIn 1st-degree connection sync cursor. One row keyed by
        // the user's own member urn (`account_id`); `last_full_sync_ms`
        // gates full-vs-delta mode, `cursor_start` resumes a paginated full
        // sync that was interrupted mid-run.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS linkedin_connection_sync (\
                 account_id          TEXT PRIMARY KEY,\
                 last_full_sync_ms   INTEGER,\
                 last_delta_sync_ms  INTEGER,\
                 cursor_start        INTEGER NOT NULL DEFAULT 0,\
                 last_synced_count   INTEGER NOT NULL DEFAULT 0,\
                 updated_at_ms       INTEGER NOT NULL\
             )",
            [],
        )?;

        // #62 — generic contacts sync token (Google People `syncToken` or
        // CardDAV `getctag`), keyed by `(backend, account_id)`.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS contacts_sync_state (\
                 backend       TEXT NOT NULL,\
                 account_id    TEXT NOT NULL,\
                 sync_token    TEXT,\
                 updated_at_ms INTEGER NOT NULL,\
                 PRIMARY KEY (backend, account_id)\
             )",
            [],
        )?;

        // #62 — phone→person reverse index consulted by message-triage
        // before creating a new wiki page. `phone` is E.164-normalized;
        // unique so re-ingest is an upsert, not a duplicate.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS identity_phone (\
                 phone         TEXT PRIMARY KEY,\
                 person_slug   TEXT NOT NULL,\
                 display_name  TEXT,\
                 source        TEXT NOT NULL,\
                 updated_at_ms INTEGER NOT NULL\
             )",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_proactive_user_actions_lookup \
                ON proactive_user_actions(action, scope, expires_at_ms)",
            [],
        )?;

        // #35 — structured ask-detection telemetry (Phase 1: shadow mode,
        // log-only, never injected). One row per detected ask in an inbound
        // message. `resolver_kind` is the would-be resolver
        // (scheduling|calendly|share_doc|intro|none); `auto_fillable` records
        // whether the shadow extractor judged it resolvable. No FK to
        // `actions` — asks are detected on the raw message before any action
        // row may exist.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS detected_asks (\
                 id              TEXT PRIMARY KEY,\
                 message_id      TEXT NOT NULL,\
                 platform        TEXT NOT NULL,\
                 ask_text        TEXT NOT NULL,\
                 resolver_kind   TEXT NOT NULL,\
                 auto_fillable   INTEGER NOT NULL DEFAULT 0,\
                 confidence      REAL,\
                 raw_json        TEXT,\
                 detected_at_ms  INTEGER NOT NULL\
             )",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_detected_asks_msg \
                ON detected_asks(message_id)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_detected_asks_recent \
                ON detected_asks(detected_at_ms)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_identity_phone_slug \
                ON identity_phone(person_slug)",
            [],
        )?;

        // #117 — multi-repo agent-coding allowlist + audit. Both additive +
        // dormant in prod until a repo is granted via the dashboard; same
        // proven-safe pattern as the wave-A tables above. `full_name` is
        // UNIQUE NOCASE so `Owner/Repo` and `owner/repo` can't both be
        // allowlisted. Default-deny: an empty table means the loop touches
        // nothing.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS agent_repos (\
                 id                 TEXT PRIMARY KEY,\
                 full_name          TEXT NOT NULL UNIQUE COLLATE NOCASE,\
                 base_branch        TEXT NOT NULL DEFAULT 'main',\
                 build_cmd          TEXT NOT NULL DEFAULT '',\
                 blast_radius_extra TEXT NOT NULL DEFAULT '',\
                 max_diff_lines     INTEGER NOT NULL DEFAULT 600,\
                 enabled            INTEGER NOT NULL DEFAULT 1,\
                 created_at_ms      INTEGER NOT NULL,\
                 updated_at_ms      INTEGER NOT NULL\
             )",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_repos_enabled \
                ON agent_repos(enabled)",
            [],
        )?;
        // One row per attempt. The gate (`pending_approval`) lives here so a
        // daemon restart never loses an awaiting-approval PR and the
        // dashboard can render full per-repo history.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS agent_pr_runs (\
                 id             TEXT PRIMARY KEY,\
                 repo_full_name TEXT NOT NULL,\
                 issue_number   INTEGER NOT NULL,\
                 branch         TEXT NOT NULL,\
                 summary        TEXT NOT NULL DEFAULT '',\
                 diff_lines     INTEGER NOT NULL DEFAULT 0,\
                 status         TEXT NOT NULL,\
                 pr_url         TEXT,\
                 error          TEXT,\
                 created_at_ms  INTEGER NOT NULL,\
                 updated_at_ms  INTEGER NOT NULL\
             )",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_pr_runs_repo \
                ON agent_pr_runs(repo_full_name, created_at_ms)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_pr_runs_status \
                ON agent_pr_runs(status)",
            [],
        )?;

        // -------------------------------------------------------------------
        // #111 — FTS5-backed cross-session memory. Owned by Rust (Cluster G).
        // `memory` is the durable storage; `memory_fts` is the FTS5 virtual
        // table the MCP memory server (crates/augmentagent-mcp-memory) reads
        // for `memory_search`. Triggers keep the two in lockstep so callers
        // can write to `memory` and search through `memory_fts` without a
        // separate index step.
        //
        // Schema mirrors issue #111: id (TEXT, opaque uuid), created_at_ms,
        // surface (email/slack/discord/ask/digest/other), subject, body,
        // tags (comma-separated, optional). `tags` is a single TEXT column
        // rather than a join table because we expect single-digit tags per
        // entry and the MCP layer just wants prefix-search behavior; an
        // FTS5 index over the joined tag string subsumes both needs.
        //
        // FTS5 is compiled into the bundled libsqlite3-sys (see the
        // -DSQLITE_ENABLE_FTS5 flag in the bundled build), so no extension
        // load is required at runtime — `CREATE VIRTUAL TABLE` Just Works.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS memory (\
                 id            TEXT PRIMARY KEY,\
                 created_at_ms INTEGER NOT NULL,\
                 surface       TEXT NOT NULL,\
                 subject       TEXT NOT NULL,\
                 body          TEXT NOT NULL,\
                 tags          TEXT NOT NULL DEFAULT ''\
             )",
            [],
        )?;
        // External-content FTS5: the `memory` table is the source of truth;
        // the FTS table is rebuildable. `content_rowid` ties FTS rows to
        // `memory.rowid` so the `delete-by-rowid` trigger can clean up FTS
        // when memory rows are pruned. Tokenizer = `porter unicode61` for
        // sensible stemming on English text without locale baggage.
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(\
                 subject, body, tags, surface UNINDEXED, \
                 content='memory', content_rowid='rowid', \
                 tokenize='porter unicode61'\
             )",
            [],
        )?;
        // Keep FTS in sync. INSERT/DELETE/UPDATE triggers mirror SQLite's
        // documented external-content recipe. `UPDATE` is implemented as
        // DELETE-then-INSERT so the FTS internal docid tracking stays
        // correct across edits.
        conn.execute_batch(
            "CREATE TRIGGER IF NOT EXISTS memory_ai AFTER INSERT ON memory BEGIN \
                 INSERT INTO memory_fts(rowid, subject, body, tags, surface) \
                 VALUES (new.rowid, new.subject, new.body, new.tags, new.surface); \
             END;\
             CREATE TRIGGER IF NOT EXISTS memory_ad AFTER DELETE ON memory BEGIN \
                 INSERT INTO memory_fts(memory_fts, rowid, subject, body, tags, surface) \
                 VALUES('delete', old.rowid, old.subject, old.body, old.tags, old.surface); \
             END;\
             CREATE TRIGGER IF NOT EXISTS memory_au AFTER UPDATE ON memory BEGIN \
                 INSERT INTO memory_fts(memory_fts, rowid, subject, body, tags, surface) \
                 VALUES('delete', old.rowid, old.subject, old.body, old.tags, old.surface); \
                 INSERT INTO memory_fts(rowid, subject, body, tags, surface) \
                 VALUES (new.rowid, new.subject, new.body, new.tags, new.surface); \
             END;",
        )?;
        // Chronological index for `memory_recent` (the MCP server's
        // search-isn't-the-right-shape path). Surface-filtered index too
        // because most reads scope to one channel.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_created ON memory(created_at_ms DESC)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_surface_created \
                ON memory(surface, created_at_ms DESC)",
            [],
        )?;

        // #238 — SocialAPI.ai integration. `socialapi_accounts` is the local
        // registry of managed social accounts (one row per platform handle);
        // `active` gates polling/posting. `socialapi_seen_comments` dedupes
        // inbound comments per post so the engagement loop only triages each
        // comment once. Both nullable-where-noted so onboarding can backfill
        // metadata later.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS socialapi_accounts (\
                 id             TEXT PRIMARY KEY,\
                 brand_id       TEXT,\
                 platform       TEXT NOT NULL,\
                 display_name   TEXT,\
                 account_handle TEXT,\
                 active         INTEGER NOT NULL DEFAULT 1,\
                 created_at_ms  INTEGER NOT NULL,\
                 updated_at_ms  INTEGER NOT NULL\
             )",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_socialapi_accounts_active \
                ON socialapi_accounts(active)",
            [],
        )?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS socialapi_seen_comments (\
                 post_id       TEXT NOT NULL,\
                 comment_id    TEXT NOT NULL,\
                 author        TEXT,\
                 text          TEXT,\
                 seen_at_ms    INTEGER NOT NULL,\
                 PRIMARY KEY (post_id, comment_id)\
             )",
            [],
        )?;
        // #242 — inbound DM dedup. `socialapi_seen_dms` records each inbound
        // DM message once, keyed on (conversation_id, message_id), so the DM
        // poller only surfaces a genuinely new inbound message a single time,
        // even across daemon restarts. Mirrors `socialapi_seen_comments`.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS socialapi_seen_dms (\
                 conversation_id TEXT NOT NULL,\
                 message_id      TEXT NOT NULL,\
                 author          TEXT,\
                 text            TEXT,\
                 seen_at_ms      INTEGER NOT NULL,\
                 PRIMARY KEY (conversation_id, message_id)\
             )",
            [],
        )?;
        // #249 — SocialAPI.ai inbound webhook events (near-real-time inbox).
        // The Express dashboard receives + verifies + persists each pushed
        // comment/DM event here; the daemon DRAINS unprocessed rows as a
        // fast-path alongside its API poll, reusing socialapi_seen_{dms,
        // comments} for downstream dedup so a webhook-delivered item and a
        // later poll of the same item don't both produce a draft. `id` is
        // SYNTHESIZED by the TS receiver from the item identity
        // (`socialapi:dm:<conversation>:<message>` /
        // `socialapi:comment:<post>:<comment>`), NOT the provider's event id
        // (#529) — so a provider re-send under a new event id still collapses,
        // and two distinct events about the same message also collapse.
        // `kind` is 'dm' | 'comment'; the CHECK below plus the guard in
        // `insert_socialapi_webhook_event` keep an unrecognized value out,
        // since the drain filters on `kind` and would strand it forever.
        // `processed` flips to 1 once the daemon emits the WorkItem. Mirrors
        // the TS schema in src/db.ts so both daemons share an identical shape.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS socialapi_webhook_events (\
                 id             TEXT PRIMARY KEY,\
                 kind           TEXT NOT NULL CHECK (kind IN ('dm', 'comment')),\
                 account_id     TEXT,\
                 payload_json   TEXT NOT NULL,\
                 received_at_ms INTEGER NOT NULL,\
                 processed      INTEGER NOT NULL DEFAULT 0\
             )",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_socialapi_webhook_events_unprocessed \
                ON socialapi_webhook_events(processed, received_at_ms)",
            [],
        )?;

        // #173 — channel-draft lifecycle for high-stakes channels (email,
        // linkedin, slack DMs). State machine: pending → approved → published,
        // with pending|approved → discarded as the bail path. The `payload_json`
        // column is opaque to the store — channel crates serialize their own
        // shape. Status is enforced in Rust (`crates/augmentagent-proactive::
        // drafts`), not via CHECK, so we can add new states without ALTER.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS channel_drafts (\
                 id                  TEXT PRIMARY KEY,\
                 target_channel      TEXT NOT NULL,\
                 payload_json        TEXT NOT NULL,\
                 status              TEXT NOT NULL DEFAULT 'pending',\
                 note                TEXT,\
                 created_at_ms       INTEGER NOT NULL,\
                 updated_at_ms       INTEGER NOT NULL,\
                 approved_at_ms      INTEGER,\
                 published_at_ms     INTEGER,\
                 discarded_at_ms     INTEGER,\
                 publish_result_json TEXT,\
                 error_message       TEXT\
             )",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_channel_drafts_status \
                ON channel_drafts(status, created_at_ms DESC)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_channel_drafts_target \
                ON channel_drafts(target_channel, status)",
            [],
        )?;

        // Daily research pipeline — dedup ledger. One row per arXiv paper id
        // we've already surfaced, so a daily re-run never re-issues a GitHub
        // issue (or re-summarizes) the same paper. Rust-only (not mirrored in
        // Node's initDb), additive, idempotent — same proven-safe shape as
        // `voice_capture_state`.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS research_seen (\
                 arxiv_id  TEXT PRIMARY KEY,\
                 seen_at   INTEGER NOT NULL\
             )",
            [],
        )?;

        // -------------------------------------------------------------------
        // #45 — indexes for Node-owned tables. Created at the END of migrate
        // so they run AFTER the additive ALTERs above have added any columns
        // they reference (most notably `emails.platform` on legacy DBs).
        // Every CREATE INDEX uses IF NOT EXISTS so this is a no-op on the
        // production DB where the indexes already exist.
        // -------------------------------------------------------------------
        conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_actions_status ON actions(status);\
             CREATE INDEX IF NOT EXISTS idx_actions_created ON actions(createdAt);\
             CREATE INDEX IF NOT EXISTS idx_actions_messageId ON actions(messageId);\
             CREATE INDEX IF NOT EXISTS idx_gmail_accounts_active ON gmail_accounts(active);\
             CREATE INDEX IF NOT EXISTS idx_emails_triage ON emails(triageResult);\
             CREATE INDEX IF NOT EXISTS idx_emails_seen ON emails(firstSeenAt);\
             CREATE INDEX IF NOT EXISTS idx_emails_platform ON emails(platform);\
             CREATE INDEX IF NOT EXISTS idx_channel_subs_active_mode ON channel_subscriptions(active, mode);\
             CREATE INDEX IF NOT EXISTS idx_slack_workspaces_active ON slack_workspaces(active);\
             CREATE INDEX IF NOT EXISTS idx_drive_accounts_active ON drive_accounts(active);",
        )?;

        Ok(())
    }

    /// Insert email if new. Returns `true` when the row did not previously exist.
    /// Matches Node `upsertEmail` behavior in src/db.ts: preserves firstSeenAt on re-seen messages.
    pub fn upsert_email(&self, email: &Email) -> StoreResult<bool> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let existed: Option<i64> = guard
            .query_row(
                "SELECT 1 FROM emails WHERE messageId = ?1",
                params![email.message_id],
                |r| r.get(0),
            )
            .optional()?;
        // #165: redact secrets/PII at the persistence boundary. The
        // channel layer still operates on plaintext for API calls; only
        // the persisted copy is masked.
        let body_masked = redact::mask(&email.body);
        let body_param: &str = &body_masked;
        if existed.is_some() {
            guard.execute(
                "UPDATE emails SET threadId = ?2, fromEmail = ?3, subject = ?4, body = ?5, receivedAt = ?6 WHERE messageId = ?1",
                params![
                    email.message_id,
                    email.thread_id,
                    email.from,
                    email.subject,
                    body_param,
                    email.date,
                ],
            )?;
            Ok(false)
        } else {
            let now = now_millis();
            guard.execute(
                "INSERT INTO emails (messageId, threadId, fromEmail, subject, body, receivedAt, accountEntityId, firstSeenAt, platform, kind) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                params![
                    email.message_id,
                    email.thread_id,
                    email.from,
                    email.subject,
                    body_param,
                    email.date,
                    email.account_entity_id,
                    now,
                    email.platform,
                    email.kind,
                ],
            )?;
            Ok(true)
        }
    }

    /// [`upsert_email`] variant for historical imports (#886): a *new* row's
    /// `firstSeenAt` is the caller-supplied message timestamp rather than
    /// now, so downstream freshness rendering ("facts as of …") reflects
    /// when the message was actually sent. Existing rows update like
    /// [`upsert_email`] and keep their original `firstSeenAt`.
    pub fn upsert_email_backfill(&self, email: &Email, first_seen_ms: i64) -> StoreResult<bool> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let existed: Option<i64> = guard
            .query_row(
                "SELECT 1 FROM emails WHERE messageId = ?1",
                params![email.message_id],
                |r| r.get(0),
            )
            .optional()?;
        let body_masked = redact::mask(&email.body);
        let body_param: &str = &body_masked;
        if existed.is_some() {
            guard.execute(
                "UPDATE emails SET threadId = ?2, fromEmail = ?3, subject = ?4, body = ?5, receivedAt = ?6 WHERE messageId = ?1",
                params![
                    email.message_id,
                    email.thread_id,
                    email.from,
                    email.subject,
                    body_param,
                    email.date,
                ],
            )?;
            Ok(false)
        } else {
            guard.execute(
                "INSERT INTO emails (messageId, threadId, fromEmail, subject, body, receivedAt, accountEntityId, firstSeenAt, platform, kind) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                params![
                    email.message_id,
                    email.thread_id,
                    email.from,
                    email.subject,
                    body_param,
                    email.date,
                    email.account_entity_id,
                    first_seen_ms,
                    email.platform,
                    email.kind,
                ],
            )?;
            Ok(true)
        }
    }

    /// True iff the email has been carried to a terminal outcome — skip, flag,
    /// dry-run reply, successful send, or an explicit rejection/timeout from
    /// the approver. Transient errors leave `agentProcessedAt = NULL`, which
    /// makes them retryable.
    /// Every per-attendee row the calendar channel wrote, as
    /// `(messageId, fromEmail, receivedAt)`.
    ///
    /// Those rows are keyed `gcal:{event_id}:{attendee_email}` and carry the
    /// event start in `receivedAt`, which makes them a complete record of who
    /// was invited to what — recoverable with no Google API call. #915's
    /// transcript bridge reads this to attach a roster to a recorded meeting.
    ///
    /// # Errors
    ///
    /// Whatever sqlite failed with.
    pub fn gcal_attendee_rows(&self) -> StoreResult<Vec<(String, String, String)>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT messageId, fromEmail, COALESCE(receivedAt, '') FROM emails \
             WHERE platform = 'gcal' AND messageId LIKE 'gcal:%'",
        )?;
        let rows = stmt
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    pub fn is_email_complete(&self, message_id: &str) -> StoreResult<bool> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let row: Option<Option<i64>> = guard
            .query_row(
                "SELECT agentProcessedAt FROM emails WHERE messageId = ?1",
                params![message_id],
                |r| r.get::<_, Option<i64>>(0),
            )
            .optional()?;
        Ok(matches!(row, Some(Some(_))))
    }

    /// `firstSeenAt` (epoch ms) for a known message id. Powers wiki page
    /// freshness (#642): pages cite messageIds in `sources:` / inline `m:`
    /// cites, and the first time we saw a message bounds how old the facts
    /// derived from it are. `None` = unknown id (the caller must treat that
    /// as "unknown", never "fresh").
    pub fn email_first_seen_at(&self, message_id: &str) -> StoreResult<Option<i64>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let row: Option<i64> = guard
            .query_row(
                "SELECT firstSeenAt FROM emails WHERE messageId = ?1",
                params![message_id],
                |r| r.get(0),
            )
            .optional()?;
        Ok(row)
    }

    pub fn is_message_processed(&self, message_id: &str) -> StoreResult<bool> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let row: Option<i64> = guard
            .query_row(
                "SELECT 1 FROM actions WHERE messageId = ?1 LIMIT 1",
                params![message_id],
                |r| r.get(0),
            )
            .optional()?;
        Ok(row.is_some())
    }

    /// #795 — the newest action row for a message as
    /// `(id, status, retry_count)`, or `None` when the message has never
    /// been actioned. Callers need the *status* (not just existence) to tell
    /// "a card is up / this is settled" from "triage errored with no card,
    /// so this still needs another attempt".
    pub fn latest_action_for_message(
        &self,
        message_id: &str,
    ) -> StoreResult<Option<(String, String, i64)>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let row = guard
            .query_row(
                "SELECT id, status, COALESCE(retryCount, 0) FROM actions \
                 WHERE messageId = ?1 ORDER BY createdAt DESC LIMIT 1",
                params![message_id],
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, i64>(2)?,
                    ))
                },
            )
            .optional()?;
        Ok(row)
    }

    /// True iff there's already an in-flight action for this message —
    /// `pending` (awaiting Discord approval), `error` (will be picked up by
    /// the retry tick), or `scheduled`/`sending` (#500 — armed for a future
    /// send / mid-send). The poll loop uses this to avoid spawning duplicate
    /// actions for the same email while one is still mid-flight.
    pub fn has_open_action(&self, message_id: &str) -> StoreResult<bool> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let row: Option<i64> = guard
            .query_row(
                "SELECT 1 FROM actions \
                 WHERE messageId = ?1 \
                   AND status IN ('pending', 'error', 'scheduled', 'sending') \
                 LIMIT 1",
                params![message_id],
                |r| r.get(0),
            )
            .optional()?;
        Ok(row.is_some())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn log_action(
        &self,
        message_id: &str,
        thread_id: Option<&str>,
        from_email: &str,
        subject: &str,
        original_body: Option<&str>,
        draft_body: Option<&str>,
        status: ActionStatus,
    ) -> StoreResult<String> {
        let id = Uuid::new_v4().to_string();
        let now = now_millis();
        let next_nudge_at_ms = match status {
            ActionStatus::Pending => Some(now + NUDGE_INTERVAL_MS),
            _ => None,
        };
        let guard = self.conn.lock().expect("store mutex poisoned");
        // #165: mask free-form bodies before persisting. Borrowed when no
        // secrets matched (zero-alloc fast path), owned otherwise.
        let original_body_masked = original_body.map(redact::mask);
        let draft_body_masked = draft_body.map(redact::mask);
        let original_body_param: Option<&str> = original_body_masked.as_deref();
        let draft_body_param: Option<&str> = draft_body_masked.as_deref();
        // #48: `mode`/`generatedSource`/`toolCallTrace` are intentionally
        // omitted from the column list — the migration's column defaults
        // populate them as 'classic' / NULL / NULL so the classic path stays
        // byte-identical post-migration. Code-mode callers use the dedicated
        // `log_action_code_mode` helper.
        guard.execute(
            "INSERT INTO actions (id, messageId, threadId, fromEmail, subject, originalBody, draftBody, status, errorMessage, createdAt, updatedAt, nudgeCount, nextNudgeAtMs) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, NULL, ?9, ?9, 0, ?10)",
            params![
                id,
                message_id,
                thread_id,
                from_email,
                subject,
                original_body_param,
                draft_body_param,
                status.as_str(),
                now,
                next_nudge_at_ms,
            ],
        )?;
        Ok(id)
    }

    /// #48 — code-mode draft variant of [`Self::log_action`]. Persists the
    /// generated TypeScript program (`generated_source`) and the tool-call
    /// trace alongside the standard action row, with `mode = 'code'`.
    ///
    /// Call this from the code-mode dispatcher's terminal `tools.draft` path
    /// (and from the fallback bookkeeping when a failed code-mode program is
    /// recorded for audit before falling back to the classic prompt). The
    /// classic-path code keeps calling [`Self::log_action`] — no behavior
    /// change there.
    ///
    /// `trace_json` is a pre-serialized JSON string (compact array) stored
    /// verbatim in the `toolCallTrace` column. The caller is responsible for
    /// serializing the `Vec<ToolCallRecord>` to JSON before calling — matching
    /// how `meta_json`, `exemplar_ids`, and `raw_json` work throughout this
    /// file. The store is a dumb byte-bucket for this field.
    #[allow(clippy::too_many_arguments)]
    pub fn log_action_code_mode(
        &self,
        message_id: &str,
        thread_id: Option<&str>,
        from_email: &str,
        subject: &str,
        original_body: Option<&str>,
        draft_body: Option<&str>,
        status: ActionStatus,
        generated_source: &str,
        trace_json: &str,
    ) -> StoreResult<String> {
        let id = Uuid::new_v4().to_string();
        let now = now_millis();
        let next_nudge_at_ms = match status {
            ActionStatus::Pending => Some(now + NUDGE_INTERVAL_MS),
            _ => None,
        };
        let guard = self.conn.lock().expect("store mutex poisoned");
        // #165: mask free-form bodies + the code-mode tool-call trace
        // before persisting. `generated_source` is the LLM-emitted TS
        // program and is also caller-supplied free-form text, so it goes
        // through the same redactor in case the model echoed back any
        // tokens it saw in scope.
        let original_body_masked = original_body.map(redact::mask);
        let draft_body_masked = draft_body.map(redact::mask);
        let original_body_param: Option<&str> = original_body_masked.as_deref();
        let draft_body_param: Option<&str> = draft_body_masked.as_deref();
        let generated_source_masked = redact::mask(generated_source);
        let trace_json_masked = redact::mask(trace_json);
        let generated_source_param: &str = &generated_source_masked;
        let trace_json_param: &str = &trace_json_masked;
        guard.execute(
            "INSERT INTO actions (id, messageId, threadId, fromEmail, subject, originalBody, draftBody, status, errorMessage, createdAt, updatedAt, nudgeCount, nextNudgeAtMs, mode, generatedSource, toolCallTrace) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, NULL, ?9, ?9, 0, ?10, 'code', ?11, ?12)",
            params![
                id,
                message_id,
                thread_id,
                from_email,
                subject,
                original_body_param,
                draft_body_param,
                status.as_str(),
                now,
                next_nudge_at_ms,
                generated_source_param,
                trace_json_param,
            ],
        )?;
        Ok(id)
    }

    /// #48 — read the code-mode fields for a single action. `mode` is always
    /// non-NULL (defaulted to `'classic'` by the migration); `generated_source`
    /// and `tool_call_trace` are `None` for classic rows. Used by tests +
    /// downstream postmortem / audit code.
    pub fn action_code_mode_fields(
        &self,
        action_id: &str,
    ) -> StoreResult<Option<ActionCodeModeFields>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let row: Option<(String, Option<String>, Option<String>)> = guard
            .query_row(
                "SELECT mode, generatedSource, toolCallTrace FROM actions WHERE id = ?1",
                params![action_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()?;
        let Some((mode, generated_source, tool_call_trace)) = row else {
            return Ok(None);
        };
        Ok(Some(ActionCodeModeFields {
            mode,
            generated_source,
            tool_call_trace,
        }))
    }

    /// Log a `flagged` action and persist the triage flag `reason` into the
    /// `errorMessage` column (unused for non-error statuses). The morning
    /// digest (#100) reads this back via `flagged_actions_since` so it can
    /// enumerate *why* each item was flagged, not just that it was.
    pub fn log_flagged_action(
        &self,
        message_id: &str,
        thread_id: Option<&str>,
        from_email: &str,
        subject: &str,
        original_body: Option<&str>,
        reason: &str,
    ) -> StoreResult<String> {
        let id = Uuid::new_v4().to_string();
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "INSERT INTO actions (id, messageId, threadId, fromEmail, subject, originalBody, draftBody, status, errorMessage, createdAt, updatedAt, nudgeCount, nextNudgeAtMs) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, NULL, 'flagged', ?7, ?8, ?8, 0, NULL)",
            params![
                id,
                message_id,
                thread_id,
                from_email,
                subject,
                original_body,
                reason,
                now,
            ],
        )?;
        Ok(id)
    }

    pub fn update_action_status(
        &self,
        action_id: &str,
        status: ActionStatus,
        draft_body: Option<&str>,
        error_message: Option<&str>,
    ) -> StoreResult<()> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "UPDATE actions SET status = ?2, draftBody = COALESCE(?3, draftBody), errorMessage = COALESCE(?4, errorMessage), updatedAt = ?5 WHERE id = ?1",
            params![action_id, status.as_str(), draft_body, error_message, now],
        )?;
        Ok(())
    }

    /// Persist a (draft, feedback, revised_draft) triple for action `action_id`
    /// (#37 — draft-quality feedback loop).
    ///
    /// Writes two rows to `draft_revisions` in a single transaction:
    /// 1. The pre-Revise draft as iteration N with `outcome = 'superseded'`
    ///    and `feedbackText = NULL`.
    /// 2. The post-Revise draft as iteration N+1 with `outcome = 'pending'`
    ///    and `feedbackText = feedback`.
    ///
    /// Iteration numbering is contiguous per action: it picks `MAX(iteration)+1`
    /// from existing rows, defaulting to 0 when this is the first revise on a
    /// fresh action. Using two rows keeps the schema's `(actionId, iteration)`
    /// invariant clean and lets downstream consumers read the chronology
    /// without inferring it.
    ///
    /// Returns the id of the **revised** (newest) row — that's the one
    /// downstream tone-mirror / clusterer code refers to.
    pub fn record_revision_triple(
        &self,
        action_id: &str,
        original_draft: &str,
        feedback: &str,
        revised_draft: &str,
    ) -> StoreResult<String> {
        let now = now_millis();
        let revised_id = Uuid::new_v4().to_string();
        let mut guard = self.conn.lock().expect("store mutex poisoned");
        let tx = guard.transaction()?;
        // Pick the next iteration. If no prior rows exist for this action the
        // pre-Revise draft is iteration 0 and the revised one is iteration 1.
        let max_iter: Option<i64> = tx
            .query_row(
                "SELECT MAX(iteration) FROM draft_revisions WHERE actionId = ?1",
                params![action_id],
                |r| r.get::<_, Option<i64>>(0),
            )
            .optional()?
            .flatten();
        let next_iter = max_iter.map(|i| i + 1).unwrap_or(0);
        // The pre-Revise draft is only inserted when there is no existing row
        // at iteration `next_iter - 0` for this action — i.e. on the first
        // Revise. On subsequent revises the previous iteration's row is
        // already there (with outcome = 'pending'); we just flip it to
        // 'superseded' so the chain stays consistent.
        if next_iter == 0 {
            let original_id = Uuid::new_v4().to_string();
            tx.execute(
                "INSERT INTO draft_revisions \
                   (id, actionId, iteration, draftBody, feedbackText, presetId, \
                    outcome, modelId, promptTokens, completionTokens, createdAt) \
                 VALUES (?1, ?2, 0, ?3, NULL, NULL, 'superseded', '', NULL, NULL, ?4)",
                params![original_id, action_id, original_draft, now],
            )?;
        } else {
            tx.execute(
                "UPDATE draft_revisions \
                    SET outcome = 'superseded' \
                  WHERE actionId = ?1 AND iteration = ?2 AND outcome = 'pending'",
                params![action_id, next_iter - 1],
            )?;
        }
        let revised_iter = if next_iter == 0 { 1 } else { next_iter };
        tx.execute(
            "INSERT INTO draft_revisions \
               (id, actionId, iteration, draftBody, feedbackText, presetId, \
                outcome, modelId, promptTokens, completionTokens, createdAt) \
             VALUES (?1, ?2, ?3, ?4, ?5, NULL, 'pending', '', NULL, NULL, ?6)",
            params![revised_id, action_id, revised_iter, revised_draft, feedback, now],
        )?;
        tx.commit()?;
        Ok(revised_id)
    }

    /// All revision rows for `action_id`, oldest iteration first. Backs the
    /// downstream tone-mirror corpus (#73) and the recurring-feedback
    /// clusterer (#37 Phase 3).
    pub fn list_revisions_for_action(
        &self,
        action_id: &str,
    ) -> StoreResult<Vec<RevisionRecord>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT id, actionId, iteration, draftBody, feedbackText, presetId, \
                    outcome, modelId, promptTokens, completionTokens, createdAt \
               FROM draft_revisions \
              WHERE actionId = ?1 \
              ORDER BY iteration ASC",
        )?;
        let rows = stmt.query_map(params![action_id], row_to_revision_record)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Revision rows whose `feedbackText` is non-NULL, created within the last
    /// `since_ms` milliseconds. Backs the recurring-feedback clusterer
    /// (`augmentagent drafts feedback-clusters`).
    pub fn list_recent_feedback(
        &self,
        since_ms: i64,
    ) -> StoreResult<Vec<RevisionRecord>> {
        if since_ms <= 0 {
            return Ok(Vec::new());
        }
        let cutoff = now_millis() - since_ms;
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT id, actionId, iteration, draftBody, feedbackText, presetId, \
                    outcome, modelId, promptTokens, completionTokens, createdAt \
               FROM draft_revisions \
              WHERE feedbackText IS NOT NULL \
                AND createdAt >= ?1 \
              ORDER BY createdAt DESC",
        )?;
        let rows = stmt.query_map(params![cutoff], row_to_revision_record)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn mark_email_processed(&self, message_id: &str, triage: TriageResult) -> StoreResult<()> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "UPDATE emails SET triageResult = ?2, agentProcessedAt = ?3 WHERE messageId = ?1",
            params![message_id, triage.as_str(), now],
        )?;
        Ok(())
    }

    /// #179 — record the outcome of the most recent gmail poll for an entity
    /// so the dashboard can show a *live* connected indicator instead of
    /// trusting the static `active` flag. `ok=true` after a successful
    /// `GMAIL_FETCH_EMAILS`, `false` after any 4xx/5xx. UPDATE-by-entity
    /// (no-op if the entity isn't in `gmail_accounts`, which only happens
    /// in tests with mock stores).
    pub fn set_gmail_account_poll_outcome(
        &self,
        entity_id: &str,
        ok: bool,
        at_ms: i64,
    ) -> StoreResult<()> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "UPDATE gmail_accounts SET lastPolledAt = ?1, lastPollOk = ?2 \
             WHERE entityId = ?3",
            rusqlite::params![at_ms, if ok { 1 } else { 0 }, entity_id],
        )?;
        Ok(())
    }

    /// #797 — persist Composio's per-entity Gmail fetch cooldown so polling,
    /// background observers, and one-shot CLI processes share one boundary.
    pub fn set_gmail_fetch_cooldown(
        &self,
        entity_id: &str,
        retry_after_ms: i64,
        log_id: Option<&str>,
    ) -> StoreResult<()> {
        let observed_at_ms = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "INSERT INTO gmail_fetch_cooldowns \
                (entityId, retryAfterMs, logId, observedAtMs) VALUES (?1, ?2, ?3, ?4) \
             ON CONFLICT(entityId) DO UPDATE SET \
                retryAfterMs = MAX(gmail_fetch_cooldowns.retryAfterMs, excluded.retryAfterMs), \
                logId = COALESCE(excluded.logId, gmail_fetch_cooldowns.logId), \
                observedAtMs = excluded.observedAtMs",
            params![entity_id, retry_after_ms, log_id, observed_at_ms],
        )?;
        Ok(())
    }

    /// Returns only an active boundary; expired rows are harmless audit data.
    pub fn gmail_fetch_cooldown_until(
        &self,
        entity_id: &str,
        now_ms: i64,
    ) -> StoreResult<Option<i64>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let retry_after_ms: Option<i64> = guard
            .query_row(
                "SELECT retryAfterMs FROM gmail_fetch_cooldowns \
                 WHERE entityId = ?1 AND retryAfterMs > ?2",
                params![entity_id, now_ms],
                |r| r.get(0),
            )
            .optional()?;
        Ok(retry_after_ms)
    }

    /// A successful fetch proves the provider accepted the request.
    pub fn clear_gmail_fetch_cooldown(&self, entity_id: &str) -> StoreResult<()> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "DELETE FROM gmail_fetch_cooldowns WHERE entityId = ?1",
            params![entity_id],
        )?;
        Ok(())
    }

    pub fn get_active_gmail_accounts(&self) -> StoreResult<Vec<Account>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT id, connectionId, entityId, email, active FROM gmail_accounts WHERE active = 1",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(Account {
                id: r.get(0)?,
                connection_id: r.get::<_, Option<String>>(1)?,
                entity_id: r.get(2)?,
                email: r.get::<_, Option<String>>(3)?.unwrap_or_default(),
                active: r.get::<_, i64>(4)? != 0,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    // --- Multi-tenant Google Drive (Composio) ---------------------------
    // Inert in prod (no rows) — only a tenant that connects Drive uses these.

    /// Insert/replace a connected Drive account (dedup by `connection_id`).
    pub fn add_drive_account(
        &self,
        connection_id: &str,
        entity_id: &str,
        email: Option<&str>,
        label: Option<&str>,
    ) -> StoreResult<String> {
        let id = format!("drive-{connection_id}");
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "INSERT INTO drive_accounts \
                 (id, connection_id, entity_id, email, label, active, created_at_ms) \
                 VALUES (?1, ?2, ?3, ?4, ?5, 1, ?6) \
             ON CONFLICT(id) DO UPDATE SET \
                 entity_id = excluded.entity_id, email = excluded.email, \
                 label = excluded.label, active = 1",
            params![id, connection_id, entity_id, email, label, now_millis()],
        )?;
        Ok(id)
    }

    pub fn get_active_drive_accounts(&self) -> StoreResult<Vec<DriveAccount>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT id, connection_id, entity_id, email, label, active \
               FROM drive_accounts WHERE active = 1 ORDER BY created_at_ms ASC",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(DriveAccount {
                id: r.get(0)?,
                connection_id: r.get(1)?,
                entity_id: r.get(2)?,
                email: r.get::<_, Option<String>>(3)?.unwrap_or_default(),
                label: r.get::<_, Option<String>>(4)?,
                active: r.get::<_, i64>(5)? != 0,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Persisted Drive `changes.list` cursor for an entity (None on first poll).
    pub fn get_drive_sync_token(&self, entity_id: &str) -> StoreResult<Option<String>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let tok: Option<String> = guard
            .query_row(
                "SELECT page_token FROM drive_sync_state WHERE entity_id = ?1",
                params![entity_id],
                |r| r.get(0),
            )
            .optional()?;
        Ok(tok)
    }

    pub fn set_drive_sync_token(&self, entity_id: &str, page_token: &str) -> StoreResult<()> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "INSERT INTO drive_sync_state (entity_id, page_token, updated_at_ms) \
                 VALUES (?1, ?2, ?3) \
             ON CONFLICT(entity_id) DO UPDATE SET \
                 page_token = excluded.page_token, updated_at_ms = excluded.updated_at_ms",
            params![entity_id, page_token, now_millis()],
        )?;
        Ok(())
    }

    /// ShadowNote journal delta-sync watermark (#427). `None` = never
    /// synced — caller runs a base sync (full backfill).
    pub fn get_journal_sync_state(&self, owner_id: &str) -> StoreResult<Option<i64>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let ms: Option<i64> = guard
            .query_row(
                "SELECT last_sync_ms FROM journal_sync_state WHERE owner_id = ?1",
                params![owner_id],
                |r| r.get(0),
            )
            .optional()?;
        Ok(ms)
    }

    pub fn set_journal_sync_state(&self, owner_id: &str, last_sync_ms: i64) -> StoreResult<()> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "INSERT INTO journal_sync_state (owner_id, last_sync_ms, updated_at_ms) \
                 VALUES (?1, ?2, ?3) \
             ON CONFLICT(owner_id) DO UPDATE SET \
                 last_sync_ms = excluded.last_sync_ms, updated_at_ms = excluded.updated_at_ms",
            params![owner_id, last_sync_ms, now_millis()],
        )?;
        Ok(())
    }

    /// #900 — the interrupted sync pass for `owner_id`, if any (crash,
    /// restart, budget exhausted, page-fetch error).
    pub fn get_journal_sync_cursor(
        &self,
        owner_id: &str,
    ) -> StoreResult<Option<JournalSyncCursor>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let cursor = guard
            .query_row(
                "SELECT last_sync_ms, started_at_ms, next_token \
                   FROM journal_sync_cursor WHERE owner_id = ?1",
                params![owner_id],
                |r| {
                    Ok(JournalSyncCursor {
                        last_sync_ms: r.get(0)?,
                        started_at_ms: r.get(1)?,
                        next_token: r.get(2)?,
                    })
                },
            )
            .optional()?;
        Ok(cursor)
    }

    /// #900 — persist the in-progress pass after a page is processed.
    pub fn set_journal_sync_cursor(
        &self,
        owner_id: &str,
        cursor: &JournalSyncCursor,
    ) -> StoreResult<()> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "INSERT INTO journal_sync_cursor \
                 (owner_id, last_sync_ms, started_at_ms, next_token, updated_at_ms) \
                 VALUES (?1, ?2, ?3, ?4, ?5) \
             ON CONFLICT(owner_id) DO UPDATE SET \
                 last_sync_ms = excluded.last_sync_ms, \
                 started_at_ms = excluded.started_at_ms, \
                 next_token = excluded.next_token, \
                 updated_at_ms = excluded.updated_at_ms",
            params![
                owner_id,
                cursor.last_sync_ms,
                cursor.started_at_ms,
                cursor.next_token,
                now_millis()
            ],
        )?;
        Ok(())
    }

    /// #900 — the pass completed (or was abandoned); no cursor to resume.
    pub fn clear_journal_sync_cursor(&self, owner_id: &str) -> StoreResult<()> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "DELETE FROM journal_sync_cursor WHERE owner_id = ?1",
            params![owner_id],
        )?;
        Ok(())
    }

    /// #900 — has this `(entry, _version)` already been handed to ingest?
    pub fn journal_entry_ingested(
        &self,
        owner_id: &str,
        entry_id: &str,
        version: i64,
    ) -> StoreResult<bool> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let hit: Option<i64> = guard
            .query_row(
                "SELECT 1 FROM journal_ingested \
                  WHERE owner_id = ?1 AND entry_id = ?2 AND version = ?3",
                params![owner_id, entry_id, version],
                |r| r.get(0),
            )
            .optional()?;
        Ok(hit.is_some())
    }

    /// #900 — record an `(entry, _version)` as handed to ingest. Returns
    /// `true` when the row is new, `false` when it was already recorded.
    pub fn mark_journal_ingested(
        &self,
        owner_id: &str,
        entry_id: &str,
        version: i64,
    ) -> StoreResult<bool> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let inserted = guard.execute(
            "INSERT OR IGNORE INTO journal_ingested \
                 (owner_id, entry_id, version, ingested_at_ms) VALUES (?1, ?2, ?3, ?4)",
            params![owner_id, entry_id, version, now_millis()],
        )?;
        Ok(inserted > 0)
    }

    // ---------------------------------------------------------------
    // #886 — iMessage bundle incremental cursor.
    // ---------------------------------------------------------------

    /// Entries of this conversation already ingested. 0 = never synced.
    pub fn get_imessage_entries_seen(&self, conversation: &str) -> StoreResult<i64> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n: Option<i64> = guard
            .query_row(
                "SELECT entries_seen FROM imessage_sync_state WHERE conversation = ?1",
                params![conversation],
                |r| r.get(0),
            )
            .optional()?;
        Ok(n.unwrap_or(0))
    }

    pub fn set_imessage_entries_seen(&self, conversation: &str, entries_seen: i64) -> StoreResult<()> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "INSERT INTO imessage_sync_state (conversation, entries_seen, updated_at_ms) \
                 VALUES (?1, ?2, ?3) \
             ON CONFLICT(conversation) DO UPDATE SET \
                 entries_seen = excluded.entries_seen, updated_at_ms = excluded.updated_at_ms",
            params![conversation, entries_seen, now_millis()],
        )?;
        Ok(())
    }

    // ---------------------------------------------------------------
    // #61 — LinkedIn connection-sync cursor.
    // ---------------------------------------------------------------

    /// Read the connection-sync cursor for `account_id` (the user's own
    /// member urn). `None` means this account has never synced — caller
    /// should run a full sync.
    pub fn get_linkedin_connection_sync(
        &self,
        account_id: &str,
    ) -> StoreResult<Option<LinkedInConnectionSync>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let row = guard
            .query_row(
                "SELECT account_id, last_full_sync_ms, last_delta_sync_ms, \
                        cursor_start, last_synced_count \
                   FROM linkedin_connection_sync WHERE account_id = ?1",
                params![account_id],
                |r| {
                    Ok(LinkedInConnectionSync {
                        account_id: r.get(0)?,
                        last_full_sync_ms: r.get(1)?,
                        last_delta_sync_ms: r.get(2)?,
                        cursor_start: r.get(3)?,
                        last_synced_count: r.get(4)?,
                    })
                },
            )
            .optional()?;
        Ok(row)
    }

    /// Upsert the connection-sync cursor. Pass the full desired state — this
    /// is a blind overwrite of the mutable columns (the caller owns the
    /// full-vs-delta decision).
    pub fn upsert_linkedin_connection_sync(
        &self,
        s: &LinkedInConnectionSync,
    ) -> StoreResult<()> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "INSERT INTO linkedin_connection_sync \
                 (account_id, last_full_sync_ms, last_delta_sync_ms, \
                  cursor_start, last_synced_count, updated_at_ms) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6) \
             ON CONFLICT(account_id) DO UPDATE SET \
                 last_full_sync_ms = excluded.last_full_sync_ms, \
                 last_delta_sync_ms = excluded.last_delta_sync_ms, \
                 cursor_start = excluded.cursor_start, \
                 last_synced_count = excluded.last_synced_count, \
                 updated_at_ms = excluded.updated_at_ms",
            params![
                s.account_id,
                s.last_full_sync_ms,
                s.last_delta_sync_ms,
                s.cursor_start,
                s.last_synced_count,
                now_millis(),
            ],
        )?;
        Ok(())
    }

    // ---------------------------------------------------------------
    // #62 — contacts sync token + phone reverse index.
    // ---------------------------------------------------------------

    /// Read the sync token (Google People `syncToken` / CardDAV `getctag`)
    /// for `(backend, account_id)`. `None` → full sync on next run.
    pub fn get_contacts_sync_token(
        &self,
        backend: &str,
        account_id: &str,
    ) -> StoreResult<Option<String>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let tok: Option<String> = guard
            .query_row(
                "SELECT sync_token FROM contacts_sync_state \
                   WHERE backend = ?1 AND account_id = ?2",
                params![backend, account_id],
                |r| r.get(0),
            )
            .optional()?
            .flatten();
        Ok(tok)
    }

    pub fn set_contacts_sync_token(
        &self,
        backend: &str,
        account_id: &str,
        token: &str,
    ) -> StoreResult<()> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "INSERT INTO contacts_sync_state \
                 (backend, account_id, sync_token, updated_at_ms) \
                 VALUES (?1, ?2, ?3, ?4) \
             ON CONFLICT(backend, account_id) DO UPDATE SET \
                 sync_token = excluded.sync_token, \
                 updated_at_ms = excluded.updated_at_ms",
            params![backend, account_id, token, now_millis()],
        )?;
        Ok(())
    }

    /// Reverse-lookup a person by E.164 phone. The message-triage path calls
    /// this *before* creating a new wiki page so a known phone resolves to an
    /// existing contact instead of fragmenting identity.
    pub fn lookup_person_by_phone(
        &self,
        phone_e164: &str,
    ) -> StoreResult<Option<PhoneIdentity>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let row = guard
            .query_row(
                "SELECT phone, person_slug, display_name, source \
                   FROM identity_phone WHERE phone = ?1",
                params![phone_e164],
                |r| {
                    Ok(PhoneIdentity {
                        phone: r.get(0)?,
                        person_slug: r.get(1)?,
                        display_name: r.get(2)?,
                        source: r.get(3)?,
                    })
                },
            )
            .optional()?;
        Ok(row)
    }

    /// Upsert a phone→person index row. Idempotent: re-ingesting the same
    /// contact rewrites the (slug, name) for that phone rather than
    /// duplicating.
    pub fn upsert_phone_identity(&self, p: &PhoneIdentity) -> StoreResult<()> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "INSERT INTO identity_phone \
                 (phone, person_slug, display_name, source, updated_at_ms) \
                 VALUES (?1, ?2, ?3, ?4, ?5) \
             ON CONFLICT(phone) DO UPDATE SET \
                 person_slug = excluded.person_slug, \
                 display_name = excluded.display_name, \
                 source = excluded.source, \
                 updated_at_ms = excluded.updated_at_ms",
            params![
                p.phone,
                p.person_slug,
                p.display_name,
                p.source,
                now_millis(),
            ],
        )?;
        Ok(())
    }

    /// Repoint every phone-index row from one person slug to another — the
    /// store half of an owner-approved page merge (`person merge`). Without
    /// this, a deleted stub's rows keep resolving future syncs to a missing
    /// page. Returns the number of rows moved.
    pub fn repoint_phone_identity(&self, from_slug: &str, to_slug: &str) -> StoreResult<usize> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n = guard.execute(
            "UPDATE identity_phone SET person_slug = ?2, updated_at_ms = ?3 \
             WHERE person_slug = ?1",
            params![from_slug, to_slug, now_millis()],
        )?;
        Ok(n)
    }

    /// Backfill the human-readable Gmail address for a connected account.
    /// The OAuth connect flow never captured it (Composio doesn't return it
    /// on the connection), so the dashboard + entity picker show opaque IDs
    /// until this is populated from a `GMAIL_GET_PROFILE` lookup.
    pub fn update_gmail_account_email(&self, id: &str, email: &str) -> StoreResult<()> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "UPDATE gmail_accounts SET email = ?2 WHERE id = ?1",
            params![id, email],
        )?;
        Ok(())
    }

    pub fn save_learned_pattern(&self, _pattern: &LearnedPattern) -> StoreResult<()> {
        // Node writes these as JSON under skills/email-triage/learned/*.json, not sqlite.
        // Phase 3 decides final home. For Phase 1 this is a no-op; channel adapter logs instead.
        Ok(())
    }

    /// Count actions grouped by status for rows created in the last `since_ms`
    /// milliseconds. Pairs the `actions.status` text with its count.
    pub fn action_counts_since(
        &self,
        since_ms: i64,
    ) -> StoreResult<Vec<(String, i64)>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT status, COUNT(*) FROM actions WHERE createdAt >= ?1 GROUP BY status",
        )?;
        let rows = stmt.query_map(params![since_ms], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Most recently processed emails. Each row: (from, subject, triageResult).
    /// `triageResult` is `None` for rows still awaiting processing.
    pub fn recent_emails_since(
        &self,
        since_ms: i64,
        limit: i64,
    ) -> StoreResult<Vec<(String, String, Option<String>)>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT fromEmail, subject, triageResult \
             FROM emails \
             WHERE firstSeenAt >= ?1 \
             ORDER BY firstSeenAt DESC \
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![since_ms, limit], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, Option<String>>(2)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// #64 — `(message_id, from_email, body)` for emails first seen on/after
    /// `since_ms`, newest first. Backs `backfill signatures`: it mines each
    /// body's signature block for role/title/company/phone. Idempotent at
    /// the call site (the wiki merge is fill-blanks-only).
    pub fn email_bodies_since(
        &self,
        since_ms: i64,
        limit: i64,
    ) -> StoreResult<Vec<(String, String, String)>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT messageId, fromEmail, body \
             FROM emails \
             WHERE firstSeenAt >= ?1 \
             ORDER BY firstSeenAt DESC \
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![since_ms, limit], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, Option<String>>(2)?.unwrap_or_default(),
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// How many reply actions are currently sitting in `pending` status
    /// (awaiting the user's Discord click). Useful as a digest metric.
    pub fn pending_reply_count(&self) -> StoreResult<i64> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n: i64 = guard.query_row(
            "SELECT COUNT(*) FROM actions WHERE status = 'pending'",
            [],
            |r| r.get(0),
        )?;
        Ok(n)
    }

    /// Every action that landed in `flagged` status within the window. Each
    /// row: (from, subject, reason). The flag reason is stashed in
    /// `errorMessage` at log time (it is otherwise unused for non-error
    /// statuses); empty/NULL collapses to "flagged". No LIMIT — the digest
    /// (#100) needs an exhaustive list, and flagged volume is small by
    /// construction (triage flags are the exception, not the rule).
    pub fn flagged_actions_since(
        &self,
        since_ms: i64,
    ) -> StoreResult<Vec<(String, String, String)>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT fromEmail, subject, COALESCE(NULLIF(errorMessage, ''), 'flagged') \
             FROM actions \
             WHERE status = 'flagged' AND createdAt >= ?1 \
             ORDER BY createdAt DESC",
        )?;
        let rows = stmt.query_map(params![since_ms], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Every action currently sitting in `pending` (awaiting the user's
    /// Discord click), oldest first. Each row: (from, subject, age_ms). No
    /// LIMIT — the digest (#100) must enumerate the entire approval backlog,
    /// and #99's backpressure keeps this set bounded.
    pub fn pending_actions(&self) -> StoreResult<Vec<(String, String, i64)>> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT fromEmail, subject, createdAt \
             FROM actions \
             WHERE status = 'pending' \
             ORDER BY createdAt ASC",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                (now - r.get::<_, i64>(2)?).max(0),
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// The `limit` oldest pending actions, oldest first. Each row:
    /// (action_id, from, subject, age_ms). Backs `approvals list` and the
    /// `discard-older` bulk-clear path (#99).
    pub fn oldest_pending_actions(
        &self,
        limit: i64,
    ) -> StoreResult<Vec<(String, String, String, i64)>> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT id, fromEmail, subject, createdAt \
             FROM actions \
             WHERE status = 'pending' \
             ORDER BY createdAt ASC \
             LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                (now - r.get::<_, i64>(3)?).max(0),
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Expire every pending action created on or before `cutoff_ms` by
    /// flipping it to `timed_out` (the existing terminal status for
    /// abandoned approvals). Returns the ids of the rows that were swept,
    /// so callers can audit-log per-row (the Serve-loop periodic sweep
    /// emits one line per id; #220) or summarize by `.len()` (the
    /// `approvals discard-older` CLI; #99).
    ///
    /// Implementation: select the matching ids first, then UPDATE only
    /// those ids in a single statement, so the returned set and the set
    /// flipped are guaranteed to be the same rows (no torn read where a
    /// row transitions between the SELECT and the UPDATE).
    pub fn expire_pending_older_than(&self, cutoff_ms: i64) -> StoreResult<Vec<String>> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT id FROM actions \
             WHERE status = 'pending' AND createdAt <= ?1",
        )?;
        let ids: Vec<String> = stmt
            .query_map(params![cutoff_ms], |r| r.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        if ids.is_empty() {
            return Ok(ids);
        }
        // Build the IN-list placeholder string for the UPDATE. Rusqlite
        // doesn't expand `Vec` into an IN-clause; do it manually. Safe
        // because the placeholders are `?N`, not user input.
        let placeholders: String = (0..ids.len())
            .map(|i| format!("?{}", i + 2))
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "UPDATE actions \
             SET status = 'timed_out', \
                 errorMessage = COALESCE(NULLIF(errorMessage, ''), 'expired: stale pending draft'), \
                 updatedAt = ?1 \
             WHERE status = 'pending' AND id IN ({placeholders})"
        );
        let mut params_vec: Vec<&dyn rusqlite::ToSql> = Vec::with_capacity(ids.len() + 1);
        params_vec.push(&now);
        for id in &ids {
            params_vec.push(id);
        }
        guard.execute(&sql, params_vec.as_slice())?;
        Ok(ids)
    }

    /// Resolve a single pending action to `approved` (used by
    /// `approvals approve-all`). Returns true if a pending row was flipped.
    /// This only flips the status row — it does NOT send the Gmail draft;
    /// the existing Discord approve handler owns the send path. `approve-all`
    /// is a queue-hygiene escape hatch ("I've handled these out of band"),
    /// not a bulk-send.
    pub fn mark_pending_approved(&self, action_id: &str) -> StoreResult<bool> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n = guard.execute(
            "UPDATE actions \
             SET status = 'approved', updatedAt = ?2 \
             WHERE id = ?1 AND status = 'pending'",
            params![action_id, now],
        )?;
        Ok(n > 0)
    }

    /// #219 — flip every `pending`/`dry_run` action on `thread_id` to
    /// `superseded` because the user replied out-of-band (Gmail web / mobile).
    /// Returns the list of affected action ids so the caller can edit the
    /// matching Discord approval cards in a follow-on (the card-edit step is
    /// intentionally NOT done here; this method only owns the queue-state
    /// transition so #220's auto-expire and the outbound observer can both
    /// call it without dragging the Discord http client through the store).
    /// `reason` is stashed in `errorMessage` (same convention as
    /// `expire_pending_older_than`) so the dashboard can surface why the row
    /// went terminal.
    pub fn mark_pending_drafts_superseded_by_thread(
        &self,
        thread_id: &str,
        reason: &str,
    ) -> StoreResult<Vec<String>> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        // Collect the ids first so we can return them. A single transaction
        // (implicit on a single UPDATE with RETURNING-like read) isn't worth
        // the ceremony: at worst a concurrent writer races us and the second
        // UPDATE no-ops on already-superseded rows. The SELECT scope matches
        // the UPDATE WHERE exactly, so the ids we hand back are precisely
        // the ones the UPDATE will (or just did) flip.
        // #500 — 'scheduled' is deliberately NOT in this set. This thread-wide
        // flip is invoked from contexts with no time bound (the reconcile
        // sweep's pending-row rule asks "any user reply EVER"; the observer
        // processes replies that may predate the schedule being armed), and an
        // armed scheduled send must only be cancelled by a reply that came
        // AFTER the owner armed it. Scheduled rows are cancelled through the
        // bounded, per-row [`Self::mark_scheduled_superseded`] instead — from
        // the engine's fire-time guard and the reconcile sweep's dedicated
        // scheduled pass. 'sending' is also excluded — the Composio call is
        // already in flight and flipping the row under it would let the
        // conditional finish_send report a phantom failure for a send that
        // landed.
        let mut stmt = guard.prepare(
            "SELECT id FROM actions \
             WHERE threadId = ?1 AND status IN ('pending', 'dry_run')",
        )?;
        let ids: Vec<String> = stmt
            .query_map(params![thread_id], |r| r.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        drop(stmt);
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        guard.execute(
            "UPDATE actions \
             SET status = 'superseded', \
                 errorMessage = COALESCE(NULLIF(?2, ''), 'superseded by manual reply'), \
                 updatedAt = ?3 \
             WHERE threadId = ?1 AND status IN ('pending', 'dry_run')",
            params![thread_id, reason, now],
        )?;
        Ok(ids)
    }

    /// #449 — record a message the daemon itself just sent, so the
    /// OutboundObserver can skip it when it scans the SENT folder instead of
    /// misreading it as a reply the user typed by hand.
    ///
    /// Idempotent: re-sending the same Gmail message id is a no-op, so a
    /// retried send or a re-polled observer tick can't double-insert.
    pub fn record_self_sent_message(
        &self,
        message_id: &str,
        thread_id: Option<&str>,
        entity_id: Option<&str>,
        action_id: Option<&str>,
    ) -> StoreResult<()> {
        // Stamped here rather than by the caller: this is called immediately
        // after the send returns, and every caller would otherwise need the
        // store's private clock.
        let sent_at_ms = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "INSERT OR IGNORE INTO self_sent_messages \
                 (message_id, thread_id, entity_id, action_id, sent_at_ms) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![message_id, thread_id, entity_id, action_id, sent_at_ms],
        )?;
        Ok(())
    }

    /// #455 — `createdAt` of the oldest still-open approval card, if any.
    ///
    /// This is exactly how far back the OutboundObserver needs to look on its
    /// very first tick. A reply the user sent BEFORE the oldest open card
    /// was raised cannot make any current card stale, so there is no reason to
    /// walk further back through their SENT history than this.
    ///
    /// #500 — includes `scheduled`/`sending` rows: a scheduled send must be
    /// cancellable by a manual reply, so when the only open work is scheduled
    /// the cursor must still reach back to when it was raised, not seed to
    /// "now" and go blind to replies the user already sent.
    pub fn oldest_pending_action_created_at(&self) -> StoreResult<Option<i64>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let v: Option<i64> = guard
            .query_row(
                "SELECT MIN(createdAt) FROM actions \
                  WHERE status IN ('pending', 'scheduled', 'sending')",
                [],
                |r| r.get::<_, Option<i64>>(0),
            )
            .optional()?
            .flatten();
        Ok(v)
    }

    /// #449 — the Gmail message ids of daemon sends within the last
    /// `cutoff_ms`. Feeds `classify_outbound`'s `SkipDaemonSent` arm.
    pub fn self_sent_message_ids_since(
        &self,
        cutoff_ms: i64,
    ) -> StoreResult<std::collections::HashSet<String>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT message_id FROM self_sent_messages WHERE sent_at_ms >= ?1",
        )?;
        let rows = stmt.query_map(params![cutoff_ms], |r| r.get::<_, String>(0))?;
        let mut out = std::collections::HashSet::new();
        for row in rows {
            out.insert(row?);
        }
        Ok(out)
    }

    /// #449 — proximity fallback for the id-matching path above. If Composio's
    /// send response omitted the message id we have no exact id to match on,
    /// but we still logged the thread and the send time. A SENT-folder message
    /// on the same thread within `window_ms` of one of our own sends is far
    /// more likely to BE that send than a user reply typed in the same instant,
    /// so we treat it as ours.
    ///
    /// The failure mode is deliberately one-sided: a false positive means we
    /// skip superseding a card (the user sees one stale card), whereas a false
    /// negative would mean recording an agent send as a user reply and
    /// suppressing drafts on that thread forever. Prefer the harmless error.
    pub fn has_daemon_send_near(
        &self,
        thread_id: &str,
        at_ms: i64,
        window_ms: i64,
    ) -> StoreResult<bool> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let hit: Option<i64> = guard
            .query_row(
                "SELECT 1 FROM self_sent_messages \
                 WHERE thread_id = ?1 AND ABS(sent_at_ms - ?2) <= ?3 LIMIT 1",
                params![thread_id, at_ms, window_ms],
                |r| r.get(0),
            )
            .optional()?;
        Ok(hit.is_some())
    }

    /// #449 — every pending approval card, with the sender/subject/body the
    /// staleness rules need. Powers the reconciliation sweep that retires
    /// cards the user has already dealt with (or that should never have been
    /// raised) without the user having to ask.
    ///
    /// #927 — `identity_merge` cards are excluded: all three staleness rules
    /// read an inbound *email* (thread answered? bulk sender? empty draft?),
    /// and a merge card has no thread and a display name rather than a mailbox
    /// in `fromEmail`, which the bulk-sender rule retires on sight.
    pub fn pending_actions_for_reconcile(&self) -> StoreResult<Vec<PendingActionRow>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT a.id, a.threadId, a.fromEmail, a.subject, \
                    COALESCE(a.originalBody, ''), \
                    (a.draftBody IS NULL OR TRIM(a.draftBody, ' \t\r\n') = '') \
               FROM actions a \
               LEFT JOIN emails e ON a.messageId = e.messageId \
              WHERE a.status = 'pending' \
                AND COALESCE(e.kind, '') <> 'identity_merge' \
              ORDER BY a.createdAt ASC",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(PendingActionRow {
                id: r.get(0)?,
                thread_id: r.get(1)?,
                from_email: r.get(2)?,
                subject: r.get(3)?,
                body: r.get(4)?,
                draft_empty: r.get(5)?,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(StoreError::from)
    }

    /// #449 — flip an explicit set of pending action ids to `superseded`.
    /// Companion to `mark_pending_drafts_superseded_by_thread` for the cases
    /// where staleness is decided per-card (bulk sender) rather than
    /// per-thread (user already replied). Returns the number of rows flipped.
    pub fn mark_pending_superseded_by_ids(
        &self,
        ids: &[String],
        reason: &str,
    ) -> StoreResult<usize> {
        if ids.is_empty() {
            return Ok(0);
        }
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        let placeholders: String = (0..ids.len())
            .map(|i| format!("?{}", i + 3))
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "UPDATE actions \
             SET status = 'superseded', \
                 errorMessage = COALESCE(NULLIF(?1, ''), 'superseded: stale'), \
                 updatedAt = ?2 \
             WHERE status = 'pending' AND id IN ({placeholders})"
        );
        let mut params_vec: Vec<&dyn rusqlite::ToSql> = Vec::with_capacity(ids.len() + 2);
        params_vec.push(&reason);
        params_vec.push(&now);
        for id in ids {
            params_vec.push(id);
        }
        let n = guard.execute(&sql, params_vec.as_slice())?;
        Ok(n)
    }

    /// #219 — read the high-water `sent` timestamp the outbound observer has
    /// already classified for `entity_id`. `None` (treated as 0 by the
    /// observer) before the first poll on that account, so the first tick
    /// after enabling the observer doesn't backfill the entire SENT folder.
    pub fn outbound_last_seen(&self, entity_id: &str) -> StoreResult<Option<i64>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let v: Option<i64> = guard
            .query_row(
                "SELECT last_seen_sent_at_ms FROM outbound_state WHERE entity_id = ?1",
                params![entity_id],
                |r| r.get(0),
            )
            .optional()?;
        Ok(v)
    }

    /// #219 — persist the high-water `sent` timestamp for `entity_id`.
    /// Monotonic upsert (same pattern as `set_voice_capture_offset`): a
    /// stale write never rewinds the cursor, so an out-of-order page from
    /// Composio can't cause us to re-emit OutboundEvents we already handled.
    pub fn set_outbound_last_seen(
        &self,
        entity_id: &str,
        last_seen_sent_at_ms: i64,
    ) -> StoreResult<()> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "INSERT INTO outbound_state (entity_id, last_seen_sent_at_ms, updated_at_ms) \
             VALUES (?1, ?2, ?3) \
             ON CONFLICT(entity_id) DO UPDATE SET \
                last_seen_sent_at_ms = MAX(last_seen_sent_at_ms, excluded.last_seen_sent_at_ms), \
                updated_at_ms = excluded.updated_at_ms",
            params![entity_id, last_seen_sent_at_ms, now],
        )?;
        Ok(())
    }

    /// #218 — record one classified outbound message in the per-thread log.
    /// Called by the OutboundObserver on every `Classification::Emit`.
    /// Idempotent: re-inserting the same `(entity_id, message_id)` is a
    /// no-op (`INSERT OR IGNORE`), so the observer can safely re-poll a
    /// page without producing duplicate rows. Messages without a thread_id
    /// are still recorded — they just won't satisfy the inbound
    /// already-replied lookup (which requires a thread match).
    pub fn record_outbound_thread_event(
        &self,
        entity_id: &str,
        message_id: &str,
        thread_id: Option<&str>,
        sent_at_ms: i64,
    ) -> StoreResult<()> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "INSERT OR IGNORE INTO outbound_thread_log \
                 (entity_id, message_id, thread_id, sent_at_ms, recorded_at_ms) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![entity_id, message_id, thread_id, sent_at_ms, now],
        )?;
        Ok(())
    }

    /// #218 — has the user sent any outbound message on `thread_id` whose
    /// `sent_at_ms` is strictly greater than `after_ms`? The cheap-signal
    /// half of the inbound-side "skip drafting when the user already
    /// replied" check; the live Gmail thread-fetch fallback is deferred.
    ///
    /// Returns `Ok(false)` (not an error) when the `outbound_thread_log`
    /// table doesn't exist yet — keeps the inbound gate working on old
    /// stores that pre-date this migration. Same defensive shape used by
    /// other recently-added read methods that probe migration-new tables.
    pub fn thread_has_user_reply_after(
        &self,
        thread_id: &str,
        after_ms: i64,
    ) -> StoreResult<bool> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let res = guard.query_row(
            "SELECT 1 FROM outbound_thread_log \
                 WHERE thread_id = ?1 AND sent_at_ms > ?2 LIMIT 1",
            params![thread_id, after_ms],
            |_| Ok(()),
        );
        match res {
            Ok(()) => Ok(true),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(false),
            Err(e) => {
                // Old stores may not yet have the migration applied (e.g.
                // an external snapshot opened read-only). Treat
                // missing-table as "no recorded reply" rather than an
                // error so the caller still drafts as before.
                let msg = e.to_string();
                if msg.contains("no such table") {
                    Ok(false)
                } else {
                    Err(e.into())
                }
            }
        }
    }

    /// Load a single action row plus its email body. Used by the Discord
    /// event handler on approve/revise/skip clicks to reconstruct context.
    pub fn get_action_with_email(&self, action_id: &str) -> StoreResult<Option<ActionWithEmail>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let row: Option<ActionWithEmail> = guard
            .query_row(
                "SELECT \
                   a.id, a.messageId, a.threadId, a.fromEmail, a.subject, \
                   a.originalBody, a.draftBody, a.status, a.errorMessage, \
                   a.createdAt, a.updatedAt, COALESCE(a.retryCount, 0), a.draftId, \
                   e.body, e.receivedAt, e.accountEntityId, e.platform, e.kind \
                 FROM actions a \
                 LEFT JOIN emails e ON a.messageId = e.messageId \
                 WHERE a.id = ?1",
                params![action_id],
                |r| {
                    Ok(ActionWithEmail {
                        action: ActionRecord {
                            id: r.get(0)?,
                            message_id: r.get(1)?,
                            thread_id: r.get(2)?,
                            from_email: r.get(3)?,
                            subject: r.get(4)?,
                            original_body: r.get(5)?,
                            draft_body: r.get(6)?,
                            status: r.get(7)?,
                            error_message: r.get(8)?,
                            created_at: ms_to_rfc3339(r.get::<_, i64>(9)?),
                            updated_at: ms_to_rfc3339(r.get::<_, i64>(10)?),
                        },
                        retry_count: r.get::<_, i64>(11)?,
                        draft_id: r.get::<_, Option<String>>(12)?,
                        email: Email {
                            attachments: Vec::new(),
                            to: String::new(),
                            cc: String::new(),
                            message_id: r.get(1)?,
                            thread_id: r.get(2)?,
                            from: r.get(3)?,
                            subject: r.get(4)?,
                            body: r.get::<_, Option<String>>(13)?.unwrap_or_default(),
                            date: r.get::<_, Option<String>>(14)?.unwrap_or_default(),
                            account_entity_id: r.get::<_, Option<String>>(15)?,
                            platform: r.get::<_, Option<String>>(16)?.unwrap_or_else(|| "gmail".into()),
                            kind: r.get::<_, Option<String>>(17)?.unwrap_or_else(|| "dm".into()),
                        },
                    })
                },
            )
            .optional()?;
        Ok(row)
    }

    /// Store the Gmail-side draft id alongside an action. Called right after
    /// create_draft succeeds.
    pub fn set_action_draft_id(&self, action_id: &str, draft_id: &str) -> StoreResult<()> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "UPDATE actions SET draftId = ?2, updatedAt = ?3 WHERE id = ?1",
            params![action_id, draft_id, now],
        )?;
        Ok(())
    }

    /// #473 — persist the outbound envelope (To/cc/bcc, comma-joined bare
    /// addresses) of a compose-originated card. Empty strings are stored as
    /// NULL so `get_action_envelope` cleanly reports "no envelope recorded".
    pub fn set_action_envelope(
        &self,
        action_id: &str,
        to: Option<&str>,
        cc: Option<&str>,
        bcc: Option<&str>,
    ) -> StoreResult<()> {
        fn nz(v: Option<&str>) -> Option<&str> {
            v.filter(|s| !s.trim().is_empty())
        }
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "UPDATE actions SET toEmails = ?2, ccEmails = ?3, bccEmails = ?4, updatedAt = ?5 \
             WHERE id = ?1",
            params![action_id, nz(to), nz(cc), nz(bcc), now],
        )?;
        Ok(())
    }

    /// #652 — persist the subject the outgoing draft now carries, after a
    /// Revise changed it. Kept separate from [`set_action_envelope`] so a
    /// later recipient write can't clobber it (and vice versa).
    pub fn set_action_subject(&self, action_id: &str, subject: Option<&str>) -> StoreResult<()> {
        let subject = subject.filter(|s| !s.trim().is_empty());
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "UPDATE actions SET envelopeSubject = ?2, updatedAt = ?3 WHERE id = ?1",
            params![action_id, subject, now],
        )?;
        Ok(())
    }

    /// #473 — the envelope recorded by [`set_action_envelope`] (and the #652
    /// subject override), or `None` when the row doesn't exist or no field was
    /// ever set (pre-#473 rows, auto-triage replies): callers then fall back
    /// to `emails.from` and the derived reply subject.
    pub fn get_action_envelope(
        &self,
        action_id: &str,
    ) -> StoreResult<Option<ActionEnvelope>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let row = guard
            .query_row(
                "SELECT toEmails, ccEmails, bccEmails, envelopeSubject FROM actions WHERE id = ?1",
                params![action_id],
                |r| {
                    Ok(ActionEnvelope {
                        to: r.get::<_, Option<String>>(0)?,
                        cc: r.get::<_, Option<String>>(1)?,
                        bcc: r.get::<_, Option<String>>(2)?,
                        subject: r.get::<_, Option<String>>(3)?,
                    })
                },
            )
            .optional()?;
        Ok(row.filter(|e| {
            e.to.is_some() || e.cc.is_some() || e.bcc.is_some() || e.subject.is_some()
        }))
    }

    /// #419 duplicate guard — the newest open (pending or #500 scheduled)
    /// action for this account + recipient + subject, as
    /// `(action_id, draft_id, status)`. The status lets the compose path
    /// distinguish a replaceable pending card from an ARMED scheduled send —
    /// the latter must be refused, not silently replaced: the supersede in
    /// the Replace arm is pending-only, so "replacing" a scheduled row would
    /// leave the old schedule armed alongside the new card (double send).
    /// `fromEmail` holds the recipient for compose-originated cards (the
    /// card's From field shows who the mail goes to), so matching on it plus
    /// the emails-row entity id catches "the same email asked twice".
    pub fn find_pending_action_for_recipient(
        &self,
        account_entity_id: &str,
        recipient: &str,
        subject: &str,
    ) -> StoreResult<Option<(String, Option<String>, String)>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let row = guard
            .query_row(
                "SELECT a.id, a.draftId, a.status FROM actions a \
                 JOIN emails e ON a.messageId = e.messageId \
                 WHERE a.status IN ('pending', 'scheduled') \
                   AND e.accountEntityId = ?1 \
                   AND LOWER(a.fromEmail) = LOWER(?2) \
                   AND a.subject = ?3 \
                 ORDER BY a.createdAt DESC LIMIT 1",
                params![account_entity_id, recipient, subject],
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, Option<String>>(1)?,
                        r.get::<_, String>(2)?,
                    ))
                },
            )
            .optional()?;
        Ok(row)
    }

    /// #419 card-sync — ids of open actions currently pointing at
    /// `draft_id`. `update-draft` replaces a Gmail draft with a new id; any
    /// live approval card for the old draft must be repointed or its Approve
    /// button sends a deleted draft.
    ///
    /// #500 — 'scheduled' rows are included for the same reason: the engine
    /// would otherwise fire GMAIL_SEND_DRAFT on the deleted old id at the
    /// scheduled time ("fix the typo in that email" is a mainline flow).
    /// 'sending' is deliberately excluded — repointing a row mid-send is
    /// wrong in both directions.
    pub fn find_pending_action_ids_by_draft_id(
        &self,
        draft_id: &str,
    ) -> StoreResult<Vec<String>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT id FROM actions \
              WHERE status IN ('pending', 'scheduled') AND draftId = ?1",
        )?;
        let ids = stmt
            .query_map(params![draft_id], |r| r.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ids)
    }

    /// #419 card-sync — refresh the stored draft body WITHOUT touching
    /// status/error (update_action_status would also stamp those). Keeps the
    /// card's Revise context in step with the Gmail draft after update-draft.
    pub fn set_action_draft_body(&self, action_id: &str, body: &str) -> StoreResult<()> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "UPDATE actions SET draftBody = ?2, updatedAt = ?3 WHERE id = ?1",
            params![action_id, body, now],
        )?;
        Ok(())
    }

    /// Record a quick-refine preset choice and bump the redraft counter (#34).
    ///
    /// Called once per Quick-refine select. `preset_id` is `None` for a
    /// free-form Revise (still counts toward the iteration cap). Returns the
    /// post-increment `redraftCount` so the caller can decide whether the cap
    /// ([`MAX_REDRAFT_ITERATIONS`-equivalent](crate)) has been hit.
    pub fn record_redraft(
        &self,
        action_id: &str,
        preset_id: Option<&str>,
    ) -> StoreResult<i64> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "UPDATE actions \
                SET redraftCount = COALESCE(redraftCount, 0) + 1, \
                    lastPresetId = ?2, \
                    updatedAt = ?3 \
              WHERE id = ?1",
            params![action_id, preset_id, now],
        )?;
        let count: i64 = guard.query_row(
            "SELECT COALESCE(redraftCount, 0) FROM actions WHERE id = ?1",
            params![action_id],
            |r| r.get(0),
        )?;
        Ok(count)
    }

    /// Read the current redraft iteration count for an action (#34). Returns 0
    /// for a never-refined action or one that predates the column.
    pub fn redraft_count(&self, action_id: &str) -> StoreResult<i64> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let count: Option<i64> = guard
            .query_row(
                "SELECT COALESCE(redraftCount, 0) FROM actions WHERE id = ?1",
                params![action_id],
                |r| r.get(0),
            )
            .optional()?;
        Ok(count.unwrap_or(0))
    }

    /// Find reply-intent actions that errored out and deserve another try.
    ///
    /// Criteria:
    /// - `actions.status = 'error'` (not `permanent_error`, not terminal)
    /// - `actions.createdAt` within `max_age_ms` (don't retry ancient errors forever)
    /// - `actions.updatedAt` older than `min_gap_ms` ago (space attempts out)
    /// - `actions.retryCount < max_attempts`
    /// - `emails.platform` equals the caller's channel — each channel's retry
    ///   tick only sees its own rows (#670); otherwise e.g. a socialapi Error
    ///   row lands in the gmail tick and is dispatched through gmail plumbing
    /// - Joined with `emails` so the caller has the email body to retry with
    pub fn list_retryable_replies(
        &self,
        platform: &str,
        now_ms: i64,
        max_age_ms: i64,
        min_gap_ms: i64,
        max_attempts: i64,
        limit: i64,
    ) -> StoreResult<Vec<RetryableReply>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT \
               a.id, a.messageId, a.threadId, a.fromEmail, a.subject, \
               a.originalBody, a.draftBody, a.status, a.errorMessage, \
               a.createdAt, a.updatedAt, COALESCE(a.retryCount, 0), \
               e.body, e.receivedAt, e.accountEntityId, e.platform, e.kind \
             FROM actions a \
             JOIN emails e ON a.messageId = e.messageId \
             WHERE a.status = 'error' \
               AND e.platform = ?1 \
               AND a.createdAt >= ?2 \
               AND a.updatedAt <= ?3 \
               AND COALESCE(a.retryCount, 0) < ?4 \
             ORDER BY a.createdAt ASC \
             LIMIT ?5",
        )?;
        let rows = stmt.query_map(
            params![
                platform,
                now_ms - max_age_ms,
                now_ms - min_gap_ms,
                max_attempts,
                limit,
            ],
            |r| {
                Ok(RetryableReply {
                    action: ActionRecord {
                        id: r.get(0)?,
                        message_id: r.get(1)?,
                        thread_id: r.get(2)?,
                        from_email: r.get(3)?,
                        subject: r.get(4)?,
                        original_body: r.get(5)?,
                        draft_body: r.get(6)?,
                        status: r.get(7)?,
                        error_message: r.get(8)?,
                        created_at: ms_to_rfc3339(r.get::<_, i64>(9)?),
                        updated_at: ms_to_rfc3339(r.get::<_, i64>(10)?),
                    },
                    retry_count: r.get::<_, i64>(11)?,
                    email: Email {
                        attachments: Vec::new(),
                        to: String::new(),
                        cc: String::new(),
                        message_id: r.get(1)?,
                        thread_id: r.get(2)?,
                        from: r.get(3)?,
                        subject: r.get(4)?,
                        body: r.get::<_, Option<String>>(12)?.unwrap_or_default(),
                        date: r.get::<_, Option<String>>(13)?.unwrap_or_default(),
                        account_entity_id: r.get::<_, Option<String>>(14)?,
                        platform: r.get::<_, Option<String>>(15)?.unwrap_or_else(|| "gmail".into()),
                        kind: r.get::<_, Option<String>>(16)?.unwrap_or_else(|| "dm".into()),
                    },
                })
            },
        )?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Increment an action's retry counter. When it crosses `max_attempts`,
    /// flip the status to `permanent_error` so the retry loop stops touching it.
    pub fn increment_retry_count(
        &self,
        action_id: &str,
        max_attempts: i64,
    ) -> StoreResult<i64> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "UPDATE actions SET retryCount = COALESCE(retryCount, 0) + 1, updatedAt = ?2 WHERE id = ?1",
            params![action_id, now],
        )?;
        let new_count: i64 = guard.query_row(
            "SELECT COALESCE(retryCount, 0) FROM actions WHERE id = ?1",
            params![action_id],
            |r| r.get(0),
        )?;
        if new_count >= max_attempts {
            guard.execute(
                "UPDATE actions SET status = 'permanent_error', updatedAt = ?2 WHERE id = ?1",
                params![action_id, now],
            )?;
        }
        Ok(new_count)
    }

    // --- pending-action nudge loop (serial queue) ---

    /// Count of pending actions currently in the nudge queue: rows whose
    /// `nextNudgeAtMs` has fired (i.e. they're either active or due to be
    /// promoted). Used by the scheduler to compute the X/Y queue counter.
    pub fn count_pending_overdue(&self, now_ms: i64) -> StoreResult<i64> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n: i64 = guard.query_row(
            "SELECT COUNT(*) FROM actions \
              WHERE status = 'pending' \
                AND nextNudgeAtMs IS NOT NULL \
                AND nextNudgeAtMs <= ?1",
            params![now_ms],
            |r| r.get(0),
        )?;
        Ok(n)
    }

    /// The currently-active card in the nudge queue, if any. "Active" means a
    /// pending row that has already been promoted (`nudgeCount > 0`) — the
    /// card the user is currently looking at. There is at most one.
    pub fn find_active_nudge(&self) -> StoreResult<Option<PendingNudge>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let row = guard
            .query_row(
                "SELECT \
                   a.id, a.messageId, a.threadId, a.fromEmail, a.subject, \
                   a.originalBody, a.draftBody, a.status, a.errorMessage, \
                   a.createdAt, a.updatedAt, COALESCE(a.retryCount, 0), a.draftId, \
                   COALESCE(a.nudgeCount, 0), a.nextNudgeAtMs, \
                   e.body, e.receivedAt, e.accountEntityId, e.platform, e.kind \
                 FROM actions a \
                 JOIN emails e ON a.messageId = e.messageId \
                 WHERE a.status = 'pending' AND COALESCE(a.nudgeCount, 0) > 0 \
                 ORDER BY a.createdAt ASC \
                 LIMIT 1",
                [],
                row_to_pending_nudge,
            )
            .optional()?;
        Ok(row)
    }

    /// The next pending row eligible for promotion to active: oldest pending
    /// row with `nudgeCount = 0`. Returns None when the backlog is empty.
    ///
    /// The 6h `nextNudgeAtMs` interval only throttles *re-nudges* of the
    /// already-active card (see `find_active_nudge` + `record_nudge`). Initial
    /// promotion has no throttle — when the user resolves a card we want the
    /// next one surfaced immediately, regardless of its age. `now_ms` is
    /// retained for API stability and possible future filters.
    pub fn find_next_to_promote(&self, _now_ms: i64) -> StoreResult<Option<PendingNudge>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let row = guard
            .query_row(
                "SELECT \
                   a.id, a.messageId, a.threadId, a.fromEmail, a.subject, \
                   a.originalBody, a.draftBody, a.status, a.errorMessage, \
                   a.createdAt, a.updatedAt, COALESCE(a.retryCount, 0), a.draftId, \
                   COALESCE(a.nudgeCount, 0), a.nextNudgeAtMs, \
                   e.body, e.receivedAt, e.accountEntityId, e.platform, e.kind \
                 FROM actions a \
                 JOIN emails e ON a.messageId = e.messageId \
                 WHERE a.status = 'pending' \
                   AND COALESCE(a.nudgeCount, 0) = 0 \
                 ORDER BY a.createdAt ASC \
                 LIMIT 1",
                [],
                row_to_pending_nudge,
            )
            .optional()?;
        Ok(row)
    }

    /// Mark a nudge as delivered: bump `nudgeCount` and schedule the next
    /// reminder at `next_at_ms`. Caller computes the next time (typically
    /// `now + NUDGE_INTERVAL_MS`). Used both for initial promotion (count
    /// goes 0 → 1) and re-nudges of the active card (1 → 2 → ...).
    pub fn record_nudge(&self, action_id: &str, next_at_ms: i64) -> StoreResult<()> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "UPDATE actions \
                SET nudgeCount = COALESCE(nudgeCount, 0) + 1, \
                    nextNudgeAtMs = ?2, \
                    updatedAt = ?3 \
              WHERE id = ?1",
            params![action_id, next_at_ms, now],
        )?;
        Ok(())
    }

    /// Defer the next nudge when the user engages mid-flow (e.g. revises).
    /// Pushes `nextNudgeAtMs` out by one full interval but **does not** zero
    /// `nudgeCount` — under serial-queue mode that would kick the card back
    /// into the backlog and yank the user between drafts. The card stays the
    /// active one until the user finally approves or skips.
    pub fn reset_nudge_schedule(&self, action_id: &str) -> StoreResult<()> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "UPDATE actions \
                SET nextNudgeAtMs = ?2, \
                    updatedAt = ?3 \
              WHERE id = ?1",
            params![action_id, now + NUDGE_INTERVAL_MS, now],
        )?;
        Ok(())
    }

    // --- channel_subscriptions (issue #27) ---

    /// Create or update a subscription. Keyed on `(platform, channel_id, account_id)`
    /// so the same channel id can coexist across Slack workspaces. Re-running
    /// with the same triple upserts in place.
    pub fn upsert_subscription(
        &self,
        platform: &str,
        channel_id: &str,
        display_name: &str,
        mode: SubscriptionMode,
        account_id: Option<&str>,
    ) -> StoreResult<ChannelSubscription> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        // NULLs don't equate in SQL; IS is NULL-safe. Use it so lookup matches
        // existing rows whose account_id is NULL (Discord, pre-migration).
        let existing: Option<String> = guard
            .query_row(
                "SELECT id FROM channel_subscriptions \
                 WHERE platform = ?1 AND channel_id = ?2 AND account_id IS ?3",
                params![platform, channel_id, account_id],
                |r| r.get(0),
            )
            .optional()?;
        let id = match existing {
            Some(id) => {
                guard.execute(
                    "UPDATE channel_subscriptions \
                        SET display_name = ?2, mode = ?3, active = 1, updated_at_ms = ?4 \
                      WHERE id = ?1",
                    params![id, display_name, mode.as_str(), now],
                )?;
                id
            }
            None => {
                let id = Uuid::new_v4().to_string();
                guard.execute(
                    "INSERT INTO channel_subscriptions \
                        (id, platform, channel_id, display_name, mode, active, \
                         account_id, created_at_ms, updated_at_ms) \
                     VALUES (?1, ?2, ?3, ?4, ?5, 1, ?6, ?7, ?7)",
                    params![
                        id,
                        platform,
                        channel_id,
                        display_name,
                        mode.as_str(),
                        account_id,
                        now,
                    ],
                )?;
                id
            }
        };
        drop(guard);
        self.get_subscription(&id)?
            .ok_or_else(|| StoreError::Sqlite(rusqlite::Error::QueryReturnedNoRows))
    }

    pub fn get_subscription(&self, id: &str) -> StoreResult<Option<ChannelSubscription>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let row: Option<ChannelSubscription> = guard
            .query_row(
                "SELECT id, platform, channel_id, display_name, mode, active, \
                        account_id, last_seen_message_id, last_digest_at_ms, \
                        created_at_ms, updated_at_ms \
                   FROM channel_subscriptions \
                  WHERE id = ?1",
                params![id],
                row_to_subscription,
            )
            .optional()?;
        Ok(row)
    }

    /// List active subscriptions for a platform. Callers iterate these per
    /// poll tick. Returns deterministic order by `created_at_ms ASC` so tests
    /// are stable.
    pub fn list_active_subscriptions(
        &self,
        platform: &str,
    ) -> StoreResult<Vec<ChannelSubscription>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT id, platform, channel_id, display_name, mode, active, \
                    account_id, last_seen_message_id, last_digest_at_ms, \
                    created_at_ms, updated_at_ms \
               FROM channel_subscriptions \
              WHERE active = 1 AND platform = ?1 \
              ORDER BY created_at_ms ASC",
        )?;
        let rows = stmt.query_map(params![platform], row_to_subscription)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn update_subscription_mode(
        &self,
        id: &str,
        mode: SubscriptionMode,
    ) -> StoreResult<()> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "UPDATE channel_subscriptions \
                SET mode = ?2, updated_at_ms = ?3 \
              WHERE id = ?1",
            params![id, mode.as_str(), now],
        )?;
        Ok(())
    }

    /// Soft delete — flips `active = 0`. Kept around for audit + to prevent
    /// the unique-pair constraint blocking a later re-subscribe.
    pub fn delete_subscription(&self, id: &str) -> StoreResult<()> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "UPDATE channel_subscriptions \
                SET active = 0, updated_at_ms = ?2 \
              WHERE id = ?1",
            params![id, now],
        )?;
        Ok(())
    }

    /// Update `last_seen_message_id` after a successful poll. Snowflakes are
    /// time-sortable, so the caller passes the newest message id seen this
    /// tick.
    pub fn update_last_seen_message(
        &self,
        id: &str,
        message_id: &str,
    ) -> StoreResult<()> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "UPDATE channel_subscriptions \
                SET last_seen_message_id = ?2, updated_at_ms = ?3 \
              WHERE id = ?1",
            params![id, message_id, now],
        )?;
        Ok(())
    }

    /// Fetch (from, subject, body) rows for messages in a thread since
    /// `since_ms`. Used by the digest scheduler to aggregate one channel's
    /// recent activity. Oldest first so the prompt reads top-down.
    pub fn recent_emails_for_thread(
        &self,
        thread_id: &str,
        since_ms: i64,
    ) -> StoreResult<Vec<(String, String, String)>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT fromEmail, subject, COALESCE(body, '') \
               FROM emails \
              WHERE threadId = ?1 AND firstSeenAt >= ?2 \
              ORDER BY firstSeenAt ASC",
        )?;
        let rows = stmt.query_map(params![thread_id, since_ms], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn mark_digest_posted(&self, id: &str, at_ms: i64) -> StoreResult<()> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "UPDATE channel_subscriptions \
                SET last_digest_at_ms = ?2, updated_at_ms = ?3 \
              WHERE id = ?1",
            params![id, at_ms, now],
        )?;
        Ok(())
    }

    // --- slack_workspaces ---

    pub fn upsert_slack_workspace(
        &self,
        team_id: &str,
        team_name: &str,
        entity_id: &str,
        connection_id: &str,
        user_id: &str,
    ) -> StoreResult<SlackWorkspace> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        let existing: Option<String> = guard
            .query_row(
                "SELECT id FROM slack_workspaces WHERE team_id = ?1",
                params![team_id],
                |r| r.get(0),
            )
            .optional()?;
        match existing {
            Some(id) => {
                guard.execute(
                    "UPDATE slack_workspaces \
                        SET team_name = ?2, entity_id = ?3, connection_id = ?4, \
                            user_id = ?5, active = 1 \
                      WHERE id = ?1",
                    params![id, team_name, entity_id, connection_id, user_id],
                )?;
            }
            None => {
                let id = Uuid::new_v4().to_string();
                guard.execute(
                    "INSERT INTO slack_workspaces \
                        (id, team_id, team_name, entity_id, connection_id, \
                         user_id, active, created_at_ms) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, 1, ?7)",
                    params![
                        id,
                        team_id,
                        team_name,
                        entity_id,
                        connection_id,
                        user_id,
                        now,
                    ],
                )?;
            }
        };
        drop(guard);
        self.get_slack_workspace_by_team(team_id)?
            .ok_or_else(|| StoreError::Sqlite(rusqlite::Error::QueryReturnedNoRows))
    }

    pub fn list_active_slack_workspaces(&self) -> StoreResult<Vec<SlackWorkspace>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT id, team_id, team_name, entity_id, connection_id, \
                    user_id, active, created_at_ms \
               FROM slack_workspaces \
              WHERE active = 1 \
              ORDER BY created_at_ms ASC",
        )?;
        let rows = stmt.query_map([], row_to_slack_workspace)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn get_slack_workspace_by_team(
        &self,
        team_id: &str,
    ) -> StoreResult<Option<SlackWorkspace>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let row = guard
            .query_row(
                "SELECT id, team_id, team_name, entity_id, connection_id, \
                        user_id, active, created_at_ms \
                   FROM slack_workspaces \
                  WHERE team_id = ?1",
                params![team_id],
                row_to_slack_workspace,
            )
            .optional()?;
        Ok(row)
    }

    pub fn deactivate_slack_workspace(&self, team_id: &str) -> StoreResult<()> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "UPDATE slack_workspaces SET active = 0 WHERE team_id = ?1",
            params![team_id],
        )?;
        Ok(())
    }

    /// Hard delete — used by Disconnect so a subsequent OAuth re-creates a
    /// fresh row instead of reactivating a stale one. We also soft-delete
    /// any subscriptions tied to this workspace so the poll loop stops
    /// trying to read them with credentials that just got nuked.
    pub fn delete_slack_workspace(&self, team_id: &str) -> StoreResult<()> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "UPDATE channel_subscriptions \
                SET active = 0, updated_at_ms = ?2 \
              WHERE platform = 'slack' AND account_id = ?1",
            params![team_id, now],
        )?;
        guard.execute(
            "DELETE FROM slack_workspaces WHERE team_id = ?1",
            params![team_id],
        )?;
        Ok(())
    }

    // --- telegram_bots (#74) ---

    /// Insert a fresh `telegram_bots` row, or — if a row with this `bot_id`
    /// already exists — update its `bot_username` / `owner_chat_id` and
    /// re-activate it. `last_update_id` is preserved on update so a re-login
    /// doesn't reset the long-poll cursor.
    pub fn upsert_telegram_bot(
        &self,
        bot_id: i64,
        bot_username: &str,
        owner_chat_id: i64,
    ) -> StoreResult<TelegramBot> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        let existing: Option<String> = guard
            .query_row(
                "SELECT id FROM telegram_bots WHERE bot_id = ?1",
                params![bot_id],
                |r| r.get(0),
            )
            .optional()?;
        match existing {
            Some(id) => {
                guard.execute(
                    "UPDATE telegram_bots \
                        SET bot_username = ?2, owner_chat_id = ?3, active = 1 \
                      WHERE id = ?1",
                    params![id, bot_username, owner_chat_id],
                )?;
            }
            None => {
                let id = Uuid::new_v4().to_string();
                guard.execute(
                    "INSERT INTO telegram_bots \
                        (id, bot_id, bot_username, owner_chat_id, last_update_id, \
                         active, created_at_ms) \
                     VALUES (?1, ?2, ?3, ?4, 0, 1, ?5)",
                    params![id, bot_id, bot_username, owner_chat_id, now],
                )?;
            }
        };
        drop(guard);
        self.get_telegram_bot_by_id(bot_id)?
            .ok_or_else(|| StoreError::Sqlite(rusqlite::Error::QueryReturnedNoRows))
    }

    pub fn list_active_telegram_bots(&self) -> StoreResult<Vec<TelegramBot>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT id, bot_id, bot_username, owner_chat_id, last_update_id, \
                    active, created_at_ms \
               FROM telegram_bots \
              WHERE active = 1 \
              ORDER BY created_at_ms ASC",
        )?;
        let rows = stmt.query_map([], row_to_telegram_bot)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn get_telegram_bot_by_id(&self, bot_id: i64) -> StoreResult<Option<TelegramBot>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let row = guard
            .query_row(
                "SELECT id, bot_id, bot_username, owner_chat_id, last_update_id, \
                        active, created_at_ms \
                   FROM telegram_bots \
                  WHERE bot_id = ?1",
                params![bot_id],
                row_to_telegram_bot,
            )
            .optional()?;
        Ok(row)
    }

    pub fn get_telegram_bot_by_username(
        &self,
        bot_username: &str,
    ) -> StoreResult<Option<TelegramBot>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let row = guard
            .query_row(
                "SELECT id, bot_id, bot_username, owner_chat_id, last_update_id, \
                        active, created_at_ms \
                   FROM telegram_bots \
                  WHERE bot_username = ?1 \
                  ORDER BY created_at_ms DESC \
                  LIMIT 1",
                params![bot_username],
                row_to_telegram_bot,
            )
            .optional()?;
        Ok(row)
    }

    /// Bump the long-poll cursor. Called once per successful `getUpdates`
    /// batch with the largest `update_id` returned in that batch.
    pub fn update_telegram_bot_last_update_id(
        &self,
        bot_id: i64,
        last_update_id: i64,
    ) -> StoreResult<()> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "UPDATE telegram_bots \
                SET last_update_id = ?2 \
              WHERE bot_id = ?1 AND last_update_id < ?2",
            params![bot_id, last_update_id],
        )?;
        Ok(())
    }

    /// Hard delete + soft-deactivate all subscriptions tied to this bot, so
    /// the poll loop stops trying to read with credentials we just nuked.
    pub fn delete_telegram_bot(&self, bot_id: i64) -> StoreResult<()> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        let bot_id_str = bot_id.to_string();
        guard.execute(
            "UPDATE channel_subscriptions \
                SET active = 0, updated_at_ms = ?2 \
              WHERE platform = 'telegram' AND account_id = ?1",
            params![bot_id_str, now],
        )?;
        guard.execute(
            "DELETE FROM telegram_bots WHERE bot_id = ?1",
            params![bot_id],
        )?;
        Ok(())
    }

    // --- whatsapp_devices + allowlists (#74 / #102) ---

    /// Insert a fresh `whatsapp_devices` row, or — if a row with this `phone`
    /// already exists — refresh its JIDs / status and re-activate it.
    /// `paired_at_ms` is preserved on update so the original pairing time
    /// stays meaningful across re-pairs.
    pub fn upsert_whatsapp_device(
        &self,
        phone: &str,
        device_jid: &str,
        user_jid: &str,
    ) -> StoreResult<WhatsappDevice> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        let existing: Option<String> = guard
            .query_row(
                "SELECT id FROM whatsapp_devices WHERE phone = ?1",
                params![phone],
                |r| r.get(0),
            )
            .optional()?;
        match existing {
            Some(id) => {
                guard.execute(
                    "UPDATE whatsapp_devices \
                        SET device_jid = ?2, user_jid = ?3, \
                            session_status = 'paired', active = 1 \
                      WHERE id = ?1",
                    params![id, device_jid, user_jid],
                )?;
            }
            None => {
                let id = Uuid::new_v4().to_string();
                guard.execute(
                    "INSERT INTO whatsapp_devices \
                        (id, phone, device_jid, user_jid, paired_at_ms, \
                         last_event_at_ms, session_status, active, created_at_ms) \
                     VALUES (?1, ?2, ?3, ?4, ?5, 0, 'paired', 1, ?5)",
                    params![id, phone, device_jid, user_jid, now],
                )?;
            }
        };
        drop(guard);
        self.get_whatsapp_device_by_phone(phone)?
            .ok_or(StoreError::Sqlite(rusqlite::Error::QueryReturnedNoRows))
    }

    pub fn list_active_whatsapp_devices(&self) -> StoreResult<Vec<WhatsappDevice>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT id, phone, device_jid, user_jid, paired_at_ms, \
                    last_event_at_ms, session_status, active, created_at_ms \
               FROM whatsapp_devices \
              WHERE active = 1 \
              ORDER BY created_at_ms ASC",
        )?;
        let rows = stmt.query_map([], row_to_whatsapp_device)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn get_whatsapp_device_by_phone(
        &self,
        phone: &str,
    ) -> StoreResult<Option<WhatsappDevice>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let row = guard
            .query_row(
                "SELECT id, phone, device_jid, user_jid, paired_at_ms, \
                        last_event_at_ms, session_status, active, created_at_ms \
                   FROM whatsapp_devices WHERE phone = ?1",
                params![phone],
                row_to_whatsapp_device,
            )
            .optional()?;
        Ok(row)
    }

    /// Mark a device logged-out (sidecar emitted `logged-out`). Keeps the row
    /// for audit; the channel skips logged-out devices at send time.
    pub fn mark_whatsapp_device_logged_out(&self, phone: &str) -> StoreResult<()> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "UPDATE whatsapp_devices \
                SET session_status = 'logged_out', active = 0 \
              WHERE phone = ?1",
            params![phone],
        )?;
        Ok(())
    }

    pub fn touch_whatsapp_device_event(&self, phone: &str) -> StoreResult<()> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "UPDATE whatsapp_devices SET last_event_at_ms = ?2 WHERE phone = ?1",
            params![phone, now_millis()],
        )?;
        Ok(())
    }

    /// Hard delete + deactivate subscriptions for one device (unlink).
    pub fn delete_whatsapp_device(&self, phone: &str) -> StoreResult<()> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "UPDATE channel_subscriptions \
                SET active = 0, updated_at_ms = ?2 \
              WHERE platform = 'whatsapp' AND account_id = ?1",
            params![phone, now],
        )?;
        guard.execute(
            "DELETE FROM whatsapp_devices WHERE phone = ?1",
            params![phone],
        )?;
        Ok(())
    }

    /// Opt a chat into outbound sends. Idempotent.
    pub fn allow_whatsapp_outbound(&self, chat_jid: &str) -> StoreResult<()> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "INSERT INTO whatsapp_outbound_allowlist (chat_jid, enabled_at_ms) \
             VALUES (?1, ?2) ON CONFLICT(chat_jid) DO NOTHING",
            params![chat_jid, now_millis()],
        )?;
        Ok(())
    }

    pub fn deny_whatsapp_outbound(&self, chat_jid: &str) -> StoreResult<()> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "DELETE FROM whatsapp_outbound_allowlist WHERE chat_jid = ?1",
            params![chat_jid],
        )?;
        Ok(())
    }

    pub fn is_whatsapp_outbound_allowed(&self, chat_jid: &str) -> StoreResult<bool> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n: i64 = guard.query_row(
            "SELECT COUNT(*) FROM whatsapp_outbound_allowlist WHERE chat_jid = ?1",
            params![chat_jid],
            |r| r.get(0),
        )?;
        Ok(n > 0)
    }

    /// Opt a chat into inbound triage. Per review feedback even *reading* a
    /// chat requires explicit opt-in for ban-risk reasons. Idempotent.
    pub fn allow_whatsapp_inbound(&self, chat_jid: &str) -> StoreResult<()> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "INSERT INTO whatsapp_inbound_allowlist (chat_jid, enabled_at_ms) \
             VALUES (?1, ?2) ON CONFLICT(chat_jid) DO NOTHING",
            params![chat_jid, now_millis()],
        )?;
        Ok(())
    }

    pub fn deny_whatsapp_inbound(&self, chat_jid: &str) -> StoreResult<()> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "DELETE FROM whatsapp_inbound_allowlist WHERE chat_jid = ?1",
            params![chat_jid],
        )?;
        Ok(())
    }

    pub fn is_whatsapp_inbound_allowed(&self, chat_jid: &str) -> StoreResult<bool> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n: i64 = guard.query_row(
            "SELECT COUNT(*) FROM whatsapp_inbound_allowlist WHERE chat_jid = ?1",
            params![chat_jid],
            |r| r.get(0),
        )?;
        Ok(n > 0)
    }

    // --- tone profiles & examples (issue #73) ---

    /// Insert one tone example. Returns the new row's `id` (uuid).
    ///
    /// All filtering — empty body, too short, no-reply recipient — is the
    /// caller's responsibility (see `tone::should_keep_for_tone` in
    /// `augmentagent-channel-email`). This helper is a dumb writer.
    #[allow(clippy::too_many_arguments)]
    pub fn insert_tone_example(
        &self,
        source: &str,
        action_id: Option<&str>,
        message_id: Option<&str>,
        account_entity_id: &str,
        recipient_email: &str,
        recipient_domain: &str,
        subject: Option<&str>,
        body: &str,
        sent_at_ms: i64,
        weight: f64,
    ) -> StoreResult<String> {
        let id = Uuid::new_v4().to_string();
        let now = now_millis();
        let body_chars = body.chars().count() as i64;
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "INSERT INTO tone_examples \
                (id, source, action_id, message_id, account_entity_id, \
                 recipient_email, recipient_domain, subject, body, body_chars, \
                 sent_at_ms, ingested_at_ms, weight) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            params![
                id,
                source,
                action_id,
                message_id,
                account_entity_id,
                recipient_email,
                recipient_domain,
                subject,
                body,
                body_chars,
                sent_at_ms,
                now,
                weight,
            ],
        )?;
        Ok(id)
    }

    /// Look up a tone profile by `(scope_kind, scope_value, account_entity_id)`.
    /// `account_entity_id = None` matches the `NULL` cross-account row used
    /// for the global scope when no per-account split is needed.
    pub fn get_tone_profile(
        &self,
        scope_kind: &str,
        scope_value: &str,
        account_entity_id: Option<&str>,
    ) -> StoreResult<Option<ToneProfile>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        // `IS` is NULL-safe (`= NULL` would never match), matching the same
        // pattern `upsert_subscription` uses for the optional account_id key.
        let row = guard
            .query_row(
                "SELECT id, scope_kind, scope_value, account_entity_id, summary, \
                        exemplar_ids, sample_count, sample_count_at_refresh, \
                        last_refreshed_at, created_at_ms, updated_at_ms \
                   FROM tone_profiles \
                  WHERE scope_kind = ?1 AND scope_value = ?2 \
                    AND account_entity_id IS ?3",
                params![scope_kind, scope_value, account_entity_id],
                row_to_tone_profile,
            )
            .optional()?;
        Ok(row)
    }

    // -----------------------------------------------------------------
    // #83 — RateGovernor helpers (rate_events / rate_halts / rate_warmup).
    //
    // These are the store-side primitives the SqliteGovernor in
    // augmentagent-channel-core leans on. Kept here (and not on a separate
    // RateStore facade) so the single Mutex<Connection> still serializes
    // all rate writes against everything else hitting the same `data.db`.
    // The governor module owns the cap math; this layer just talks to SQL.
    // -----------------------------------------------------------------

    /// Insert a rate-event row. `status` is the snake-case form of the
    /// outcome (`ok` | `failed` | `rolled_back` | `suspicion`).
    /// `RolledBack` rows are still persisted (audit) but the count helpers
    /// below filter them out so they don't burn quota.
    #[allow(clippy::too_many_arguments)]
    pub fn insert_rate_event(
        &self,
        id: &str,
        platform: &str,
        action_kind: &str,
        account_id: &str,
        occurred_at_ms: i64,
        status: &str,
        cause: &str,
        target_id: Option<&str>,
        meta_json: Option<&str>,
    ) -> StoreResult<()> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "INSERT INTO rate_events \
                 (id, platform, action_kind, account_id, occurred_at_ms, \
                  status, cause, target_id, meta_json) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                id,
                platform,
                action_kind,
                account_id,
                occurred_at_ms,
                status,
                cause,
                target_id,
                meta_json,
            ],
        )?;
        Ok(())
    }

    /// Sliding-window count of "quota-burning" events for a (platform,
    /// action, account) tuple in `[since_ms, now_ms]`. Excludes
    /// `rolled_back` rows by definition (the action never executed).
    pub fn rate_event_count_in_window(
        &self,
        platform: &str,
        action_kind: &str,
        account_id: &str,
        since_ms: i64,
        now_ms: i64,
    ) -> StoreResult<u32> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n: i64 = guard.query_row(
            "SELECT COUNT(*) FROM rate_events \
              WHERE platform = ?1 \
                AND action_kind = ?2 \
                AND account_id = ?3 \
                AND occurred_at_ms >= ?4 \
                AND occurred_at_ms <= ?5 \
                AND status != 'rolled_back'",
            params![platform, action_kind, account_id, since_ms, now_ms],
            |r| r.get(0),
        )?;
        Ok(n.max(0) as u32)
    }

    /// Most recent quota-burning event timestamp for the (platform, action,
    /// account) tuple — drives min-gap enforcement. Returns `None` when no
    /// such event has ever happened.
    pub fn rate_last_event_at(
        &self,
        platform: &str,
        action_kind: &str,
        account_id: &str,
    ) -> StoreResult<Option<i64>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let v: Option<i64> = guard
            .query_row(
                "SELECT MAX(occurred_at_ms) FROM rate_events \
                  WHERE platform = ?1 \
                    AND action_kind = ?2 \
                    AND account_id = ?3 \
                    AND status != 'rolled_back'",
                params![platform, action_kind, account_id],
                |r| r.get::<_, Option<i64>>(0),
            )
            .optional()?
            .flatten();
        Ok(v)
    }

    /// Read the active halt row for a platform, if any. Caller compares
    /// `paused_until_ms` against the clock to decide whether the halt is
    /// still in effect.
    pub fn rate_halt_state(&self, platform: &str) -> StoreResult<Option<RateHalt>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let row = guard
            .query_row(
                "SELECT platform, paused_until_ms, reason, triggered_by_event_id, \
                        created_at_ms, acknowledged_at_ms \
                   FROM rate_halts WHERE platform = ?1",
                params![platform],
                |r| {
                    Ok(RateHalt {
                        platform: r.get(0)?,
                        paused_until_ms: r.get(1)?,
                        reason: r.get(2)?,
                        triggered_by_event_id: r.get(3)?,
                        created_at_ms: r.get(4)?,
                        acknowledged_at_ms: r.get(5)?,
                    })
                },
            )
            .optional()?;
        Ok(row)
    }

    /// Upsert a tone profile keyed on `(scope_kind, scope_value, account_entity_id)`.
    ///
    /// On insert: stores all fields and stamps timestamps.
    /// On update: refreshes summary/exemplar_ids/sample_count plus the
    /// `sample_count_at_refresh` snapshot so the staleness predicate
    /// (`sample_count - sample_count_at_refresh >= threshold`) resets.
    #[allow(clippy::too_many_arguments)]
    pub fn upsert_tone_profile(
        &self,
        scope_kind: &str,
        scope_value: &str,
        account_entity_id: Option<&str>,
        summary: &str,
        exemplar_ids: &str,
        sample_count: i64,
    ) -> StoreResult<ToneProfile> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        let existing: Option<String> = guard
            .query_row(
                "SELECT id FROM tone_profiles \
                 WHERE scope_kind = ?1 AND scope_value = ?2 \
                   AND account_entity_id IS ?3",
                params![scope_kind, scope_value, account_entity_id],
                |r| r.get(0),
            )
            .optional()?;
        match existing {
            Some(id) => {
                guard.execute(
                    "UPDATE tone_profiles \
                        SET summary = ?2, exemplar_ids = ?3, sample_count = ?4, \
                            sample_count_at_refresh = ?4, last_refreshed_at = ?5, \
                            updated_at_ms = ?5 \
                      WHERE id = ?1",
                    params![id, summary, exemplar_ids, sample_count, now],
                )?;
            }
            None => {
                let id = Uuid::new_v4().to_string();
                guard.execute(
                    "INSERT INTO tone_profiles \
                        (id, scope_kind, scope_value, account_entity_id, summary, \
                         exemplar_ids, sample_count, sample_count_at_refresh, \
                         last_refreshed_at, created_at_ms, updated_at_ms) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7, ?8, ?8, ?8)",
                    params![
                        id,
                        scope_kind,
                        scope_value,
                        account_entity_id,
                        summary,
                        exemplar_ids,
                        sample_count,
                        now,
                    ],
                )?;
            }
        }
        drop(guard);
        self.get_tone_profile(scope_kind, scope_value, account_entity_id)?
            .ok_or_else(|| StoreError::Sqlite(rusqlite::Error::QueryReturnedNoRows))
    }

    /// Capture the post-edit body of a `Sent` action as a tone example.
    ///
    /// This is the gold-standard signal: the user's voice corrected on top of
    /// the model's draft. Called from the email channel adapter right after
    /// `send_draft` succeeds. Source is `user_edit`, weight=1.5 to bias the
    /// summarizer toward these over backfilled history. No-op (Ok) if the
    /// action doesn't exist or is missing `draftBody`.
    pub fn record_user_edit_as_tone_example(&self, action_id: &str) -> StoreResult<Option<String>> {
        let row: Option<(
            Option<String>,
            String,
            Option<String>,
            String,
            String,
            Option<String>,
            Option<String>,
            Option<i64>,
        )>;
        {
            let guard = self.conn.lock().expect("store mutex poisoned");
            row = guard
                .query_row(
                    "SELECT a.draftBody, a.fromEmail, a.threadId, a.messageId, a.subject, \
                            e.body, e.accountEntityId, e.firstSeenAt \
                       FROM actions a \
                       LEFT JOIN emails e ON a.messageId = e.messageId \
                      WHERE a.id = ?1 AND a.status = 'sent'",
                    params![action_id],
                    |r| {
                        Ok((
                            r.get::<_, Option<String>>(0)?,
                            r.get::<_, String>(1)?,
                            r.get::<_, Option<String>>(2)?,
                            r.get::<_, String>(3)?,
                            r.get::<_, String>(4)?,
                            r.get::<_, Option<String>>(5)?,
                            r.get::<_, Option<String>>(6)?,
                            r.get::<_, Option<i64>>(7)?,
                        ))
                    },
                )
                .optional()?;
        }
        let Some((
            draft_body,
            from_email,
            _thread_id,
            message_id,
            subject,
            _orig_body,
            account_entity_id,
            received_at_ms,
        )) = row
        else {
            return Ok(None);
        };
        let Some(body) = draft_body.filter(|b| !b.trim().is_empty()) else {
            return Ok(None);
        };
        let Some(account) = account_entity_id else {
            // Fallback for actions without account context — skip silently;
            // the per-account scoping invariant on tone_examples is intentional.
            return Ok(None);
        };
        // Recipient is the row we replied TO. `actions.fromEmail` is the
        // sender of the inbound mail; that IS the address we just replied to.
        let recipient_email = bare_lower(&from_email);
        let recipient_domain = recipient_email
            .split_once('@')
            .map(|(_, d)| d.to_string())
            .unwrap_or_default();
        if recipient_email.is_empty() || recipient_domain.is_empty() {
            return Ok(None);
        }
        let sent_at_ms = received_at_ms.unwrap_or_else(now_millis);
        let id = self.insert_tone_example(
            "user_edit",
            Some(action_id),
            Some(&message_id),
            &account,
            &recipient_email,
            &recipient_domain,
            Some(&subject),
            &body,
            sent_at_ms,
            1.5,
        )?;
        Ok(Some(id))
    }

    /// Pull the most-recent N example bodies for a scope, oldest→newest.
    /// Used by the summarizer to assemble the corpus prompt.
    pub fn recent_tone_examples(
        &self,
        scope_kind: &str,
        scope_value: &str,
        account_entity_id: Option<&str>,
        limit: i64,
    ) -> StoreResult<Vec<ToneExample>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        // Branch on scope_kind to use the right index. All three queries
        // return the same column order so they share `row_to_tone_example`.
        let sql = match scope_kind {
            "recipient" => {
                "SELECT id, source, action_id, message_id, account_entity_id, \
                        recipient_email, recipient_domain, subject, body, body_chars, \
                        sent_at_ms, ingested_at_ms, weight \
                   FROM tone_examples \
                  WHERE recipient_email = ?1 AND account_entity_id IS ?2 \
                  ORDER BY sent_at_ms DESC LIMIT ?3"
            }
            "domain" => {
                "SELECT id, source, action_id, message_id, account_entity_id, \
                        recipient_email, recipient_domain, subject, body, body_chars, \
                        sent_at_ms, ingested_at_ms, weight \
                   FROM tone_examples \
                  WHERE recipient_domain = ?1 AND account_entity_id IS ?2 \
                  ORDER BY sent_at_ms DESC LIMIT ?3"
            }
            // global / anything else: filter only by account.
            _ => {
                "SELECT id, source, action_id, message_id, account_entity_id, \
                        recipient_email, recipient_domain, subject, body, body_chars, \
                        sent_at_ms, ingested_at_ms, weight \
                   FROM tone_examples \
                  WHERE account_entity_id IS ?2 \
                  ORDER BY sent_at_ms DESC LIMIT ?3"
            }
        };
        let mut stmt = guard.prepare(sql)?;
        let rows = stmt.query_map(
            params![scope_value, account_entity_id, limit],
            row_to_tone_example,
        )?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// List every tone profile, ordered by `last_refreshed_at` ascending so
    /// the staleness scan in `tone refresh-stale` walks the oldest first.
    pub fn list_tone_profiles(&self) -> StoreResult<Vec<ToneProfile>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT id, scope_kind, scope_value, account_entity_id, summary, \
                    exemplar_ids, sample_count, sample_count_at_refresh, \
                    last_refreshed_at, created_at_ms, updated_at_ms \
               FROM tone_profiles \
              ORDER BY last_refreshed_at ASC",
        )?;
        let rows = stmt.query_map([], row_to_tone_profile)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Count of tone_examples rows currently keyed against a scope. Used by
    /// the staleness predicate to refresh `sample_count` to ground truth
    /// before comparing against the snapshot.
    pub fn count_tone_examples(
        &self,
        scope_kind: &str,
        scope_value: &str,
        account_entity_id: Option<&str>,
    ) -> StoreResult<i64> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n: i64 = match scope_kind {
            "recipient" => guard.query_row(
                "SELECT COUNT(*) FROM tone_examples \
                  WHERE recipient_email = ?1 AND account_entity_id IS ?2",
                params![scope_value, account_entity_id],
                |r| r.get(0),
            )?,
            "domain" => guard.query_row(
                "SELECT COUNT(*) FROM tone_examples \
                  WHERE recipient_domain = ?1 AND account_entity_id IS ?2",
                params![scope_value, account_entity_id],
                |r| r.get(0),
            )?,
            _ => guard.query_row(
                "SELECT COUNT(*) FROM tone_examples WHERE account_entity_id IS ?1",
                params![account_entity_id],
                |r| r.get(0),
            )?,
        };
        Ok(n)
    }

    /// Upsert a halt for `platform`. Replaces any existing row — `permit()`
    /// only ever cares about the most recent halt window per platform.
    pub fn rate_set_halt(
        &self,
        platform: &str,
        paused_until_ms: i64,
        reason: &str,
        triggered_by_event_id: Option<&str>,
        now_ms: i64,
    ) -> StoreResult<()> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "INSERT INTO rate_halts \
                 (platform, paused_until_ms, reason, triggered_by_event_id, \
                  created_at_ms, acknowledged_at_ms) \
             VALUES (?1, ?2, ?3, ?4, ?5, NULL) \
             ON CONFLICT(platform) DO UPDATE SET \
                 paused_until_ms = excluded.paused_until_ms, \
                 reason = excluded.reason, \
                 triggered_by_event_id = excluded.triggered_by_event_id, \
                 created_at_ms = excluded.created_at_ms, \
                 acknowledged_at_ms = NULL",
            params![
                platform,
                paused_until_ms,
                reason,
                triggered_by_event_id,
                now_ms
            ],
        )?;
        Ok(())
    }

    /// Mark the active halt as acknowledged by the user (Discord button /
    /// dashboard). Doesn't lift the halt — the halt stays until
    /// `paused_until_ms` passes — but suppresses re-pinging the user.
    pub fn rate_acknowledge_halt(&self, platform: &str, now_ms: i64) -> StoreResult<()> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "UPDATE rate_halts SET acknowledged_at_ms = ?2 WHERE platform = ?1",
            params![platform, now_ms],
        )?;
        Ok(())
    }

    /// Read the warmup-start timestamp for a (platform, account) pair, if
    /// it has been seeded.
    pub fn rate_get_warmup(
        &self,
        platform: &str,
        account_id: &str,
    ) -> StoreResult<Option<RateWarmup>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let row = guard
            .query_row(
                "SELECT platform, account_id, warmup_started_at_ms \
                   FROM rate_warmup WHERE platform = ?1 AND account_id = ?2",
                params![platform, account_id],
                |r| {
                    Ok(RateWarmup {
                        platform: r.get(0)?,
                        account_id: r.get(1)?,
                        warmup_started_at_ms: r.get(2)?,
                    })
                },
            )
            .optional()?;
        Ok(row)
    }

    /// Idempotently seed (platform, account) → `warmup_started_at_ms = now`.
    /// Existing rows are left alone so warmup math doesn't get reset by a
    /// repeated `permit()` call on a known account.
    pub fn rate_seed_warmup(
        &self,
        platform: &str,
        account_id: &str,
        warmup_started_at_ms: i64,
    ) -> StoreResult<()> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "INSERT INTO rate_warmup (platform, account_id, warmup_started_at_ms) \
             VALUES (?1, ?2, ?3) \
             ON CONFLICT(platform, account_id) DO NOTHING",
            params![platform, account_id, warmup_started_at_ms],
        )?;
        Ok(())
    }

    /// Override a warmup start time. Used by the dashboard's
    /// "skip warmup, this account is well-aged" button (sets the timestamp
    /// 28 days into the past so the multiplier reads 1.0).
    pub fn rate_override_warmup(
        &self,
        platform: &str,
        account_id: &str,
        warmup_started_at_ms: i64,
    ) -> StoreResult<()> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "INSERT INTO rate_warmup (platform, account_id, warmup_started_at_ms) \
             VALUES (?1, ?2, ?3) \
             ON CONFLICT(platform, account_id) DO UPDATE SET \
                 warmup_started_at_ms = excluded.warmup_started_at_ms",
            params![platform, account_id, warmup_started_at_ms],
        )?;
        Ok(())
    }

    /// `[since_ms, until_ms]` audit dump for one account. Optional
    /// `platform` filter narrows to a single platform; `None` returns all
    /// platforms for the account (useful for an "everything I did" query).
    /// Ordered newest-first so the dashboard table renders without a sort.
    pub fn rate_audit_query(
        &self,
        account_id: &str,
        platform: Option<&str>,
        since_ms: i64,
        until_ms: i64,
    ) -> StoreResult<Vec<RateAuditRow>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let rows = match platform {
            Some(p) => {
                let mut stmt = guard.prepare(
                    "SELECT id, platform, action_kind, account_id, occurred_at_ms, \
                            status, cause, target_id, meta_json \
                       FROM rate_events \
                      WHERE account_id = ?1 \
                        AND platform = ?2 \
                        AND occurred_at_ms >= ?3 \
                        AND occurred_at_ms <= ?4 \
                      ORDER BY occurred_at_ms DESC",
                )?;
                let it = stmt.query_map(params![account_id, p, since_ms, until_ms], |r| {
                    Ok(RateAuditRow {
                        id: r.get(0)?,
                        platform: r.get(1)?,
                        action_kind: r.get(2)?,
                        account_id: r.get(3)?,
                        occurred_at_ms: r.get(4)?,
                        status: r.get(5)?,
                        cause: r.get(6)?,
                        target_id: r.get(7)?,
                        meta_json: r.get(8)?,
                    })
                })?;
                it.collect::<rusqlite::Result<Vec<_>>>()?
            }
            None => {
                let mut stmt = guard.prepare(
                    "SELECT id, platform, action_kind, account_id, occurred_at_ms, \
                            status, cause, target_id, meta_json \
                       FROM rate_events \
                      WHERE account_id = ?1 \
                        AND occurred_at_ms >= ?2 \
                        AND occurred_at_ms <= ?3 \
                      ORDER BY occurred_at_ms DESC",
                )?;
                let it = stmt.query_map(params![account_id, since_ms, until_ms], |r| {
                    Ok(RateAuditRow {
                        id: r.get(0)?,
                        platform: r.get(1)?,
                        action_kind: r.get(2)?,
                        account_id: r.get(3)?,
                        occurred_at_ms: r.get(4)?,
                        status: r.get(5)?,
                        cause: r.get(6)?,
                        target_id: r.get(7)?,
                        meta_json: r.get(8)?,
                    })
                })?;
                it.collect::<rusqlite::Result<Vec<_>>>()?
            }
        };
        Ok(rows)
    }

    /// Housekeeping helper — prune rate_events older than `older_than_ms`
    /// (90d retention per #83). Returns rows deleted for logging.
    pub fn rate_prune_events(&self, older_than_ms: i64) -> StoreResult<usize> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n = guard.execute(
            "DELETE FROM rate_events WHERE occurred_at_ms < ?1",
            params![older_than_ms],
        )?;
        Ok(n)
    }

    // ---------------------------------------------------------------
    // #104 — user-defined scheduled tasks (`/loop`).
    // ---------------------------------------------------------------

    /// Create a loop. `id` is generated; returns it.
    ///
    /// - `interval_secs` — fixed-interval cadence; pass `0` when using
    ///   `cron_expr` (the scheduler ignores it in that case).
    /// - `expires_at_ms` — optional auto-stop deadline; `None` means run
    ///   forever (until manually stopped or auto-paused on failures).
    /// - `cron_expr` + `tz` (#231) — cron-style cadence anchored to a
    ///   timezone. Must be both `Some` or both `None`; callers validate
    ///   the cron expression and tz string before reaching this method.
    pub fn create_user_loop(
        &self,
        owner: &str,
        channel: &str,
        channel_ref: &str,
        interval_secs: i64,
        prompt: &str,
        expires_at_ms: Option<i64>,
        cron_expr: Option<&str>,
        tz: Option<&str>,
    ) -> StoreResult<String> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let id = Uuid::new_v4().to_string();
        let now = now_millis();
        guard.execute(
            "INSERT INTO user_loops \
                 (id, owner, channel, channel_ref, interval_secs, prompt, \
                  status, fail_count, created_at_ms, updated_at_ms, \
                  expires_at_ms, cron_expr, tz) \
               VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'active', 0, ?7, ?7, ?8, ?9, ?10)",
            params![
                id,
                owner,
                channel,
                channel_ref,
                interval_secs,
                prompt,
                now,
                expires_at_ms,
                cron_expr,
                tz,
            ],
        )?;
        Ok(id)
    }

    /// All loops for an owner (any status), newest first.
    pub fn list_user_loops(&self, owner: &str) -> StoreResult<Vec<UserLoop>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT id, owner, channel, channel_ref, interval_secs, prompt, \
                    status, last_run_ms, last_status, fail_count, \
                    created_at_ms, updated_at_ms, expires_at_ms, \
                    cron_expr, tz \
               FROM user_loops \
              WHERE owner = ?1 AND status != 'stopped' \
              ORDER BY created_at_ms DESC",
        )?;
        let rows = stmt.query_map(params![owner], row_to_user_loop)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Every loop the scheduler should tick (status = 'active'), across owners.
    /// Used on boot to rehydrate and on each scheduler pass.
    pub fn list_active_user_loops(&self) -> StoreResult<Vec<UserLoop>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT id, owner, channel, channel_ref, interval_secs, prompt, \
                    status, last_run_ms, last_status, fail_count, \
                    created_at_ms, updated_at_ms, expires_at_ms, \
                    cron_expr, tz \
               FROM user_loops \
              WHERE status = 'active' \
              ORDER BY created_at_ms ASC",
        )?;
        let rows = stmt.query_map([], row_to_user_loop)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Count active loops for an owner — backs the per-user max.
    pub fn count_active_user_loops(&self, owner: &str) -> StoreResult<i64> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n: i64 = guard.query_row(
            "SELECT COUNT(*) FROM user_loops WHERE owner = ?1 AND status = 'active'",
            params![owner],
            |r| r.get(0),
        )?;
        Ok(n)
    }

    /// Transition a loop to `stopped`. Scoped by owner so a user can only
    /// stop their own. Returns true if a row changed.
    pub fn stop_user_loop(&self, owner: &str, id: &str) -> StoreResult<bool> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n = guard.execute(
            "UPDATE user_loops SET status = 'stopped', updated_at_ms = ?3 \
              WHERE id = ?1 AND owner = ?2 AND status != 'stopped'",
            params![id, owner, now_millis()],
        )?;
        Ok(n == 1)
    }

    /// Record the outcome of a loop run. `ok=false` increments `fail_count`;
    /// on reaching `pause_at` consecutive failures the loop is auto-paused.
    /// A success resets `fail_count` to 0.
    pub fn record_user_loop_run(
        &self,
        id: &str,
        ok: bool,
        status_text: &str,
        pause_at: i64,
    ) -> StoreResult<()> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let now = now_millis();
        if ok {
            guard.execute(
                "UPDATE user_loops \
                    SET last_run_ms = ?2, last_status = ?3, fail_count = 0, \
                        updated_at_ms = ?2 \
                  WHERE id = ?1",
                params![id, now, status_text],
            )?;
        } else {
            guard.execute(
                "UPDATE user_loops \
                    SET last_run_ms = ?2, last_status = ?3, \
                        fail_count = fail_count + 1, \
                        status = CASE WHEN fail_count + 1 >= ?4 THEN 'paused' \
                                      ELSE status END, \
                        updated_at_ms = ?2 \
                  WHERE id = ?1",
                params![id, now, status_text, pause_at],
            )?;
        }
        Ok(())
    }

    /// Transition every active loop whose `expires_at_ms <= now` to
    /// `stopped`, in a single statement. Returns the `(id, channel, channel_ref)`
    /// tuples of the rows we just stopped so the caller can post an
    /// "expired" notice back to the originating surface. Idempotent — a
    /// second call is a no-op once everything's already stopped.
    pub fn stop_expired_user_loops(
        &self,
        now_ms: i64,
    ) -> StoreResult<Vec<(String, String, String)>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        // Select first so we can return the surface info; then stop in the
        // same lock acquisition so a racing scheduler tick can't double-post.
        let mut stmt = guard.prepare(
            "SELECT id, channel, channel_ref FROM user_loops \
              WHERE status = 'active' \
                AND expires_at_ms IS NOT NULL \
                AND expires_at_ms <= ?1",
        )?;
        let rows = stmt.query_map(params![now_ms], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
            ))
        })?;
        let mut out: Vec<(String, String, String)> = Vec::new();
        for row in rows {
            out.push(row?);
        }
        drop(stmt);
        if !out.is_empty() {
            guard.execute(
                "UPDATE user_loops \
                    SET status = 'stopped', last_status = 'expired', \
                        updated_at_ms = ?1 \
                  WHERE status = 'active' \
                    AND expires_at_ms IS NOT NULL \
                    AND expires_at_ms <= ?1",
                params![now_ms],
            )?;
        }
        Ok(out)
    }

    // ---------------------------------------------------------------
    // #47 — cross-surface state sync: compare-and-swap status mutation.
    // ---------------------------------------------------------------

    /// Atomically flip a pending action to a terminal status, recording the
    /// resolving surface. Returns true only when this call actually performed
    /// the transition (exactly one row changed). A second surface racing on
    /// the same action gets `false` and must NOT re-run side effects.
    ///
    /// Distinct from `update_action_status`, which is unconditional and used
    /// for re-draft / pending bookkeeping. This is the resolve gate.
    ///
    /// `reason`, when given, lands in `errorMessage` (the usual reason slot
    /// for non-error terminal states — same convention as
    /// `expire_pending_older_than` / the supersede paths).
    pub fn try_resolve_action(
        &self,
        action_id: &str,
        new_status: ActionStatus,
        source: &str,
        reason: Option<&str>,
    ) -> StoreResult<bool> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n = guard.execute(
            "UPDATE actions \
                SET status = ?2, status_source = ?3, status_updated_at = ?4, \
                    updatedAt = ?4, \
                    errorMessage = COALESCE(?5, errorMessage) \
              WHERE id = ?1 AND status = 'pending'",
            params![
                action_id,
                new_status.as_str(),
                source,
                now_millis(),
                reason,
            ],
        )?;
        Ok(n == 1)
    }

    /// The surface that last resolved an action (NULL if still pending or
    /// pre-migration). Drives the Discord echo-suppression on the broadcast.
    pub fn action_status_source(
        &self,
        action_id: &str,
    ) -> StoreResult<Option<String>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let v: Option<Option<String>> = guard
            .query_row(
                "SELECT status_source FROM actions WHERE id = ?1",
                params![action_id],
                |r| r.get(0),
            )
            .optional()?;
        Ok(v.flatten())
    }

    // ---------------------------------------------------------------
    // #500 — scheduled email send: CAS state machine + engine queries.
    //
    // pending ──schedule──► scheduled ──claim──► sending ──finish──► sent/error
    //              ▲            │ unschedule (back to queue, proposal cleared)
    //              └────────────┘
    //
    // Every mutation here is a single conditional UPDATE returning
    // rows_affected == 1, so racing surfaces (Discord click, engine tick,
    // dashboard) get exactly one winner and losers must not run side
    // effects. `scheduledAtMs` is authoritative only on 'pending'
    // (a --send-at proposal) and 'scheduled' (armed) rows; it is cleared on
    // every exit that can lead back to an approvable state and retained on
    // sent rows as audit ("this send was fired by schedule").
    // ---------------------------------------------------------------

    /// Arm a schedule: `pending → scheduled` with the fire time. Used by the
    /// carousel Schedule control and by Approve on a card carrying a
    /// --send-at proposal. Returns false if the row was no longer pending
    /// (approved / superseded / a second click won).
    pub fn schedule_action(
        &self,
        action_id: &str,
        at_ms: i64,
        source: &str,
    ) -> StoreResult<bool> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n = guard.execute(
            "UPDATE actions \
                SET status = 'scheduled', scheduledAtMs = ?2, \
                    status_source = ?3, status_updated_at = ?4, updatedAt = ?4 \
              WHERE id = ?1 AND status = 'pending'",
            params![action_id, at_ms, source, now],
        )?;
        Ok(n == 1)
    }

    /// Claim a row for one send attempt: `from_status → sending`. The engine
    /// and Send Now claim from 'scheduled'; the hardened Approve path claims
    /// from 'pending'. The claim is what makes the final Sent/Error flip
    /// conditional and a concurrent Schedule/Approve/supersede race lose
    /// cleanly. `scheduledAtMs` is retained (audit: when it was due).
    pub fn claim_action_for_send(
        &self,
        action_id: &str,
        from_status: ActionStatus,
        source: &str,
    ) -> StoreResult<bool> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n = guard.execute(
            "UPDATE actions \
                SET status = 'sending', \
                    status_source = ?3, status_updated_at = ?4, updatedAt = ?4 \
              WHERE id = ?1 AND status = ?2",
            params![action_id, from_status.as_str(), source, now],
        )?;
        Ok(n == 1)
    }

    /// Back to queue: `scheduled → pending`, clearing the proposal AND the
    /// notice pointers. The row re-enters the queue as the ACTIVE card
    /// (nudgeCount 1, next re-nudge one interval out) because the caller has
    /// already reposted the approval card by the time this CAS runs —
    /// leaving nudgeCount at 0 would make the row instantly promotable and
    /// the NudgeScheduler's 60s tick could post a SECOND card in the window
    /// before the caller recorded its own (#501 review).
    /// Clearing `scheduledAtMs` here is load-bearing: a later Approve on the
    /// reposted card must send immediately, not re-arm a stale proposal.
    pub fn unschedule_action(&self, action_id: &str, source: &str) -> StoreResult<bool> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n = guard.execute(
            "UPDATE actions \
                SET status = 'pending', scheduledAtMs = NULL, \
                    noticeChannelId = NULL, noticeMessageId = NULL, \
                    nudgeCount = 1, nextNudgeAtMs = ?2, \
                    status_source = ?3, status_updated_at = ?4, updatedAt = ?4 \
              WHERE id = ?1 AND status = 'scheduled'",
            params![action_id, now + NUDGE_INTERVAL_MS, source, now],
        )?;
        Ok(n == 1)
    }

    /// Engine-only claim: `scheduled → sending`, additionally gated on the
    /// fire time still being due. The tick works from a due-list snapshot
    /// that can be minutes old (earlier rows' sends are wall-clock bounded
    /// but slow); without the `scheduledAtMs <= now` predicate, a
    /// back-to-queue + re-schedule landing in that window would be fired at
    /// the OLD due moment — up to a day early (#501 review). Send Now keeps
    /// using the plain claim: firing NOW regardless of the armed time is its
    /// whole point.
    pub fn claim_due_action_for_send(
        &self,
        action_id: &str,
        now_ms: i64,
        source: &str,
    ) -> StoreResult<bool> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n = guard.execute(
            "UPDATE actions \
                SET status = 'sending', \
                    status_source = ?3, status_updated_at = ?4, updatedAt = ?4 \
              WHERE id = ?1 AND status = 'scheduled' \
                AND scheduledAtMs IS NOT NULL AND scheduledAtMs <= ?2",
            params![action_id, now_ms, source, now],
        )?;
        Ok(n == 1)
    }

    /// Cancel: `scheduled → rejected`. The caller owns the Gmail-draft delete
    /// (best-effort, run_skip convention) and the notice delete; the pointers
    /// are cleared here so a failed notice-delete can't be retried against a
    /// resolved row forever.
    pub fn cancel_scheduled_action(
        &self,
        action_id: &str,
        reason: &str,
        source: &str,
    ) -> StoreResult<bool> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n = guard.execute(
            "UPDATE actions \
                SET status = 'rejected', scheduledAtMs = NULL, \
                    noticeChannelId = NULL, noticeMessageId = NULL, \
                    errorMessage = COALESCE(NULLIF(?2, ''), 'schedule cancelled'), \
                    status_source = ?3, status_updated_at = ?4, updatedAt = ?4 \
              WHERE id = ?1 AND status = 'scheduled'",
            params![action_id, reason, source, now],
        )?;
        Ok(n == 1)
    }

    /// Terminal success flip after a claimed send: `sending → sent`.
    /// Conditional on the claim so a racing surface that already resolved the
    /// row can never be overwritten (the "never flip a Sent row" guarantee).
    pub fn finish_send_sent(&self, action_id: &str, source: &str) -> StoreResult<bool> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n = guard.execute(
            "UPDATE actions \
                SET status = 'sent', \
                    status_source = ?2, status_updated_at = ?3, updatedAt = ?3 \
              WHERE id = ?1 AND status = 'sending'",
            params![action_id, source, now],
        )?;
        Ok(n == 1)
    }

    /// Terminal failure flip after a claimed send: `sending → error`.
    ///
    /// `retry_count_override` decides who owns recovery:
    /// - `None` (Approve path): leave retryCount alone — the generic retry
    ///   tick may re-dispatch as before.
    /// - `Some(cap)` (engine / stuck-claim reconcile): stamp retryCount to
    ///   the retry cap so `list_retryable_replies` NEVER picks the row up.
    ///   The generic retry path routes through `dispatch_reply`, which would
    ///   repost an approval card for a send that may have actually landed
    ///   (Composio timeout-but-delivered) — the double-send the scheduled
    ///   pipeline exists to avoid. Recovery is the owner's, via the failure
    ///   notice.
    pub fn finish_send_error(
        &self,
        action_id: &str,
        error_message: &str,
        retry_count_override: Option<i64>,
        source: &str,
    ) -> StoreResult<bool> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n = guard.execute(
            "UPDATE actions \
                SET status = 'error', \
                    errorMessage = ?2, \
                    retryCount = COALESCE(?3, retryCount), \
                    status_source = ?4, status_updated_at = ?5, updatedAt = ?5 \
              WHERE id = ?1 AND status = 'sending'",
            params![action_id, error_message, retry_count_override, source, now],
        )?;
        Ok(n == 1)
    }

    /// Due scheduled sends, oldest fire time first. `limit` bounds one tick's
    /// work; the engine logs when a backlog is truncated so a burst is never
    /// silently dropped. Rows are
    /// (action_id, scheduledAtMs, armed_at_ms, threadId): `armed_at_ms` is
    /// the moment the schedule was armed (`status_updated_at` stamped by
    /// `schedule_action`, falling back to `createdAt` for safety) — the
    /// correct lower bound for the fire-time "did the owner reply since?"
    /// guard. Bounding by `createdAt` instead would cancel a schedule the
    /// owner armed AFTER sending their own quick manual reply.
    pub fn due_scheduled_actions(
        &self,
        now_ms: i64,
        limit: i64,
    ) -> StoreResult<Vec<(String, i64, i64, Option<String>)>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT id, scheduledAtMs, COALESCE(status_updated_at, createdAt), \
                    threadId \
               FROM actions \
              WHERE status = 'scheduled' AND scheduledAtMs IS NOT NULL \
                AND scheduledAtMs <= ?1 \
              ORDER BY scheduledAtMs ASC \
              LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(params![now_ms, limit], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, i64>(2)?,
                    r.get::<_, Option<String>>(3)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Bounded, per-row cancellation of an armed scheduled send:
    /// `scheduled → superseded`. This — never the thread-wide
    /// `mark_pending_drafts_superseded_by_thread` — is how scheduled rows are
    /// retired when the owner replied manually, because the callers
    /// (engine fire-time guard, reconcile scheduled pass) bound the reply
    /// check to the row's own arming moment first. Notice pointers are left
    /// intact so the notice message can still be cleaned up afterwards.
    pub fn mark_scheduled_superseded(
        &self,
        action_id: &str,
        reason: &str,
        source: &str,
    ) -> StoreResult<bool> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n = guard.execute(
            "UPDATE actions \
                SET status = 'superseded', scheduledAtMs = NULL, \
                    errorMessage = COALESCE(NULLIF(?2, ''), 'superseded by manual reply'), \
                    status_source = ?3, status_updated_at = ?4, updatedAt = ?4 \
              WHERE id = ?1 AND status = 'scheduled'",
            params![action_id, reason, source, now],
        )?;
        Ok(n == 1)
    }

    /// #500 — conditional draft refresh for the Revise tail: writes the new
    /// draft body only while the row is STILL pending. The reasoner + Gmail
    /// round-trips inside run_revise take seconds; a Schedule/Approve/
    /// supersede landing meanwhile must not be stomped back to pending by an
    /// unconditional write. Returns false when the row moved on (caller
    /// cleans up its freshly created Gmail draft and reports AlreadyResolved).
    pub fn refresh_pending_draft(
        &self,
        action_id: &str,
        draft_body: &str,
    ) -> StoreResult<bool> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        let draft_masked = redact::mask(draft_body);
        let n = guard.execute(
            "UPDATE actions \
                SET draftBody = ?2, updatedAt = ?3 \
              WHERE id = ?1 AND status = 'pending'",
            params![action_id, &*draft_masked, now],
        )?;
        Ok(n == 1)
    }

    /// True while some action row holds `draft_id` in the `sending` claim —
    /// i.e. a Composio send of exactly this draft is in flight.
    /// `gmail update-draft` refuses in that window: its create-replacement +
    /// delete-old sequence would yank the draft out from under the send.
    pub fn draft_id_in_flight(&self, draft_id: &str) -> StoreResult<bool> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let row: Option<i64> = guard
            .query_row(
                "SELECT 1 FROM actions \
                  WHERE draftId = ?1 AND status = 'sending' LIMIT 1",
                params![draft_id],
                |r| r.get(0),
            )
            .optional()?;
        Ok(row.is_some())
    }

    /// Count of all armed scheduled sends, regardless of due-ness
    /// (dashboard/status surfaces; the engine uses `due_scheduled_actions`).
    pub fn count_scheduled_actions(&self) -> StoreResult<i64> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n: i64 = guard.query_row(
            "SELECT COUNT(*) FROM actions WHERE status = 'scheduled'",
            [],
            |r| r.get(0),
        )?;
        Ok(n)
    }

    /// Rows stuck in the 'sending' claim longer than the grace window —
    /// i.e. the daemon crashed (or was killed) mid-send. The engine flips
    /// these to a retry-exempt 'error' and notifies; it must NEVER resend
    /// them, because the crash window includes "Composio accepted the send
    /// and we died before recording it".
    pub fn stuck_sending_actions(
        &self,
        now_ms: i64,
        grace_ms: i64,
    ) -> StoreResult<Vec<String>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT id FROM actions \
              WHERE status = 'sending' AND updatedAt <= ?1",
        )?;
        let ids = stmt
            .query_map(params![now_ms - grace_ms], |r| r.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ids)
    }

    /// Armed scheduled rows with a real thread, for the reconcile sweep's
    /// Rule-1-only pass (#500): "did the user reply on this thread since the
    /// schedule was ARMED?" Rule 2 (bulk-sender heuristic) must NOT run on
    /// these — `fromEmail` on compose cards holds the RECIPIENT, and a
    /// scheduled send to a newsletter-looking address would be wrongly
    /// cancelled. Rows are (action_id, thread_id, armed_at_ms) —
    /// `armed_at_ms` = `COALESCE(status_updated_at, createdAt)`, the same
    /// arming-moment bound the engine's fire-time guard uses, so a reply the
    /// owner sent BEFORE deliberately arming the schedule never cancels it.
    pub fn scheduled_actions_for_reconcile(
        &self,
    ) -> StoreResult<Vec<(String, String, i64)>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT id, threadId, COALESCE(status_updated_at, createdAt) \
               FROM actions \
              WHERE status = 'scheduled' AND threadId IS NOT NULL \
              ORDER BY createdAt ASC",
        )?;
        let rows = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, i64>(2)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Persist the Discord scheduled-notice pointers so the engine can
    /// delete/update the notice at fire/cancel time (the broker itself is
    /// post-only and keeps no per-action state).
    pub fn set_action_notice(
        &self,
        action_id: &str,
        channel_id: &str,
        message_id: &str,
    ) -> StoreResult<()> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "UPDATE actions \
                SET noticeChannelId = ?2, noticeMessageId = ?3, updatedAt = ?4 \
              WHERE id = ?1",
            params![action_id, channel_id, message_id, now],
        )?;
        Ok(())
    }

    /// The stored scheduled-notice pointers, if any.
    pub fn action_notice(
        &self,
        action_id: &str,
    ) -> StoreResult<Option<(String, String)>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let row: Option<(Option<String>, Option<String>)> = guard
            .query_row(
                "SELECT noticeChannelId, noticeMessageId FROM actions WHERE id = ?1",
                params![action_id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        Ok(row.and_then(|(c, m)| Some((c?, m?))))
    }

    /// Clear the notice pointers after the notice message was deleted.
    pub fn clear_action_notice(&self, action_id: &str) -> StoreResult<()> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "UPDATE actions \
                SET noticeChannelId = NULL, noticeMessageId = NULL, updatedAt = ?2 \
              WHERE id = ?1",
            params![action_id, now],
        )?;
        Ok(())
    }

    /// The armed fire time of an action (`scheduledAtMs`), if set. Callers
    /// gate on status themselves — a pending row's value is a --send-at
    /// proposal, a scheduled row's value is the armed fire time.
    pub fn action_scheduled_at(&self, action_id: &str) -> StoreResult<Option<i64>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let v: Option<Option<i64>> = guard
            .query_row(
                "SELECT scheduledAtMs FROM actions WHERE id = ?1",
                params![action_id],
                |r| r.get(0),
            )
            .optional()?;
        Ok(v.flatten())
    }

    /// Store a --send-at proposal on a still-pending action (#502 uses this
    /// at card-post time; Approve then arms it via `schedule_action`).
    pub fn set_action_scheduled_at(
        &self,
        action_id: &str,
        at_ms: Option<i64>,
    ) -> StoreResult<()> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "UPDATE actions SET scheduledAtMs = ?2, updatedAt = ?3 WHERE id = ?1",
            params![action_id, at_ms, now],
        )?;
        Ok(())
    }

    // ---------------------------------------------------------------
    // #45 — PWA Web Push subscriptions.
    // ---------------------------------------------------------------

    pub fn add_pwa_subscription(
        &self,
        endpoint: &str,
        p256dh: &str,
        auth: &str,
    ) -> StoreResult<()> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "INSERT INTO pwa_subscriptions (id, endpoint, p256dh, auth, created_at_ms)                VALUES (?1, ?2, ?3, ?4, ?5)              ON CONFLICT(endpoint) DO UPDATE SET p256dh = ?3, auth = ?4",
            params![
                Uuid::new_v4().to_string(),
                endpoint,
                p256dh,
                auth,
                now_millis()
            ],
        )?;
        Ok(())
    }

    pub fn list_pwa_subscriptions(
        &self,
    ) -> StoreResult<Vec<(String, String, String)>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard
            .prepare("SELECT endpoint, p256dh, auth FROM pwa_subscriptions")?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn remove_pwa_subscription(&self, endpoint: &str) -> StoreResult<()> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "DELETE FROM pwa_subscriptions WHERE endpoint = ?1",
            params![endpoint],
        )?;
        Ok(())
    }

    // -----------------------------------------------------------------
    // #79 — Twitter/X GraphQL queryId cache + outbound post log.
    // -----------------------------------------------------------------

    /// Read the cached queryId for a GraphQL operation (e.g. `CreateTweet`).
    /// `None` => never observed; caller falls back to a static default.
    pub fn twitter_query_id(&self, operation: &str) -> StoreResult<Option<String>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let v: Option<String> = guard
            .query_row(
                "SELECT query_id FROM twitter_query_ids WHERE operation = ?1",
                params![operation],
                |r| r.get(0),
            )
            .optional()?;
        Ok(v)
    }

    /// Upsert an observed queryId for a GraphQL operation. Called whenever a
    /// fresher id is harvested (env override / network capture) so the next
    /// boot uses it without a recompile.
    pub fn put_twitter_query_id(
        &self,
        operation: &str,
        query_id: &str,
        last_seen_at: i64,
    ) -> StoreResult<()> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "INSERT INTO twitter_query_ids (operation, query_id, last_seen_at) \
             VALUES (?1, ?2, ?3) \
             ON CONFLICT(operation) DO UPDATE SET \
                 query_id = excluded.query_id, \
                 last_seen_at = excluded.last_seen_at",
            params![operation, query_id, last_seen_at],
        )?;
        Ok(())
    }

    /// Append a Twitter outbound-post audit row. `kind` is `tweet` | `reply`,
    /// `status` is `ok` | `failed` | `dry_run`.
    pub fn log_twitter_post(
        &self,
        id: &str,
        kind: &str,
        reply_to: Option<&str>,
        status: &str,
        tweet_id: Option<&str>,
        occurred_at_ms: i64,
        meta_json: Option<&str>,
    ) -> StoreResult<()> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "INSERT INTO twitter_post_log \
                 (id, kind, reply_to, status, tweet_id, occurred_at_ms, meta_json) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![id, kind, reply_to, status, tweet_id, occurred_at_ms, meta_json],
        )?;
        Ok(())
    }

    /// Count "real" outbound posts (status `ok` or `failed` — both burn the
    /// platform quota; `dry_run` rows are excluded) in `[since_ms, now_ms]`.
    /// Drives the hard 15/day preflight in the Twitter posting client.
    pub fn twitter_post_count_in_window(
        &self,
        since_ms: i64,
        now_ms: i64,
    ) -> StoreResult<u32> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n: i64 = guard.query_row(
            "SELECT COUNT(*) FROM twitter_post_log \
              WHERE occurred_at_ms >= ?1 \
                AND occurred_at_ms <= ?2 \
                AND status != 'dry_run'",
            params![since_ms, now_ms],
            |r| r.get(0),
        )?;
        Ok(n.max(0) as u32)
    }

    // -----------------------------------------------------------------
    // #77 / #13 — linkedin_action_log helpers.
    //
    // A LinkedIn-scoped, action-keyed audit log distinct from the
    // cross-platform rate_events table. It exists so the posting + feed
    // engagement paths can enforce their own rolling-window caps (3
    // posts/day, 5 engagements/day) durably across daemon restarts
    // *without* depending on the RateGovernor's full permit/record
    // lifecycle — the governor still gates the *decision*; this table is
    // the cheap, LinkedIn-only counter the channel reads directly.
    // -----------------------------------------------------------------

    /// Append a LinkedIn outbound-action row. `status` is free-form
    /// (`ok` | `failed` | `pending`); only `ok` rows count toward caps via
    /// [`Store::linkedin_action_count_since`].
    pub fn log_linkedin_action(
        &self,
        id: &str,
        action_kind: &str,
        target_urn: Option<&str>,
        status: &str,
        occurred_at_ms: i64,
        meta_json: Option<&str>,
    ) -> StoreResult<()> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "INSERT INTO linkedin_action_log \
                 (id, action_kind, target_urn, status, occurred_at_ms, meta_json) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![id, action_kind, target_urn, status, occurred_at_ms, meta_json],
        )?;
        Ok(())
    }

    /// Count successful (`status = 'ok'`) actions of `action_kind` since
    /// `since_ms` (inclusive). Backs the rolling-24h post cap and the
    /// daily engagement cap. Non-`ok` rows (failed / pending) are excluded
    /// so a failed dispatch doesn't permanently consume a daily slot.
    pub fn linkedin_action_count_since(
        &self,
        action_kind: &str,
        since_ms: i64,
    ) -> StoreResult<u32> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n: i64 = guard.query_row(
            "SELECT COUNT(*) FROM linkedin_action_log \
              WHERE action_kind = ?1 \
                AND status = 'ok' \
                AND occurred_at_ms >= ?2",
            params![action_kind, since_ms],
            |r| r.get(0),
        )?;
        Ok(n.max(0) as u32)
    }

    /// True iff any `ok` row already exists for this (`action_kind`,
    /// `target_urn`) pair — used to suppress duplicate engagement on the
    /// same post across feed polls.
    pub fn linkedin_action_exists(
        &self,
        action_kind: &str,
        target_urn: &str,
    ) -> StoreResult<bool> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let found: Option<i64> = guard
            .query_row(
                "SELECT 1 FROM linkedin_action_log \
                  WHERE action_kind = ?1 AND target_urn = ?2 AND status = 'ok' \
                  LIMIT 1",
                params![action_kind, target_urn],
                |r| r.get(0),
            )
            .optional()?;
        Ok(found.is_some())
    }

    // --- #117 multi-repo agent-coding allowlist + audit ----------------

    /// Allowlist (or update) a repo. Idempotent on `full_name` (case-insens):
    /// re-granting an existing repo updates its config + re-enables it
    /// without resetting its PR-run history.
    #[allow(clippy::too_many_arguments)]
    pub fn upsert_agent_repo(
        &self,
        full_name: &str,
        base_branch: &str,
        build_cmd: &str,
        blast_radius_extra: &str,
        max_diff_lines: i64,
    ) -> StoreResult<AgentRepo> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        let existing: Option<String> = guard
            .query_row(
                "SELECT id FROM agent_repos WHERE full_name = ?1 COLLATE NOCASE",
                params![full_name],
                |r| r.get(0),
            )
            .optional()?;
        match existing {
            Some(id) => {
                guard.execute(
                    "UPDATE agent_repos SET base_branch = ?2, build_cmd = ?3, \
                            blast_radius_extra = ?4, max_diff_lines = ?5, \
                            enabled = 1, updated_at_ms = ?6 \
                      WHERE id = ?1",
                    params![
                        id,
                        base_branch,
                        build_cmd,
                        blast_radius_extra,
                        max_diff_lines,
                        now
                    ],
                )?;
            }
            None => {
                let id = Uuid::new_v4().to_string();
                guard.execute(
                    "INSERT INTO agent_repos \
                        (id, full_name, base_branch, build_cmd, \
                         blast_radius_extra, max_diff_lines, enabled, \
                         created_at_ms, updated_at_ms) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, 1, ?7, ?7)",
                    params![
                        id,
                        full_name,
                        base_branch,
                        build_cmd,
                        blast_radius_extra,
                        max_diff_lines,
                        now
                    ],
                )?;
            }
        }
        drop(guard);
        self.get_agent_repo(full_name)?
            .ok_or(StoreError::Sqlite(rusqlite::Error::QueryReturnedNoRows))
    }

    pub fn get_agent_repo(&self, full_name: &str) -> StoreResult<Option<AgentRepo>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let row = guard
            .query_row(
                "SELECT id, full_name, base_branch, build_cmd, \
                        blast_radius_extra, max_diff_lines, enabled, \
                        created_at_ms, updated_at_ms \
                   FROM agent_repos WHERE full_name = ?1 COLLATE NOCASE",
                params![full_name],
                row_to_agent_repo,
            )
            .optional()?;
        Ok(row)
    }

    /// List allowlisted repos. `enabled_only` filters to active grants — the
    /// loop always passes `true` (default-deny); the dashboard passes `false`
    /// to also show revoked rows.
    pub fn list_agent_repos(&self, enabled_only: bool) -> StoreResult<Vec<AgentRepo>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let sql = if enabled_only {
            "SELECT id, full_name, base_branch, build_cmd, blast_radius_extra, \
                    max_diff_lines, enabled, created_at_ms, updated_at_ms \
               FROM agent_repos WHERE enabled = 1 ORDER BY full_name ASC"
        } else {
            "SELECT id, full_name, base_branch, build_cmd, blast_radius_extra, \
                    max_diff_lines, enabled, created_at_ms, updated_at_ms \
               FROM agent_repos ORDER BY full_name ASC"
        };
        let mut stmt = guard.prepare(sql)?;
        let rows = stmt.query_map([], row_to_agent_repo)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Revoke a repo (soft: `enabled = 0`) AND auto-reject any of its
    /// in-flight `pending_approval` gate rows so a revoked repo can never get
    /// a PR opened from a stale awaiting-approval card. Returns the number of
    /// gate rows that were cancelled.
    pub fn revoke_agent_repo(&self, full_name: &str) -> StoreResult<usize> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "UPDATE agent_repos SET enabled = 0, updated_at_ms = ?2 \
              WHERE full_name = ?1 COLLATE NOCASE",
            params![full_name, now],
        )?;
        let cancelled = guard.execute(
            "UPDATE agent_pr_runs \
                SET status = 'rejected', \
                    error = 'repo access revoked', \
                    updated_at_ms = ?2 \
              WHERE repo_full_name = ?1 COLLATE NOCASE \
                AND status = 'pending_approval'",
            params![full_name, now],
        )?;
        Ok(cancelled)
    }

    /// Insert a fresh PR-run audit row (called once the verification gate
    /// passes, in `pending_approval`).
    pub fn insert_agent_pr_run(
        &self,
        repo_full_name: &str,
        issue_number: i64,
        branch: &str,
        summary: &str,
        diff_lines: i64,
        status: &str,
    ) -> StoreResult<AgentPrRun> {
        let id = Uuid::new_v4().to_string();
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "INSERT INTO agent_pr_runs \
                (id, repo_full_name, issue_number, branch, summary, \
                 diff_lines, status, created_at_ms, updated_at_ms) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?8)",
            params![
                id,
                repo_full_name,
                issue_number,
                branch,
                summary,
                diff_lines,
                status,
                now
            ],
        )?;
        drop(guard);
        self.get_agent_pr_run(&id)?
            .ok_or(StoreError::Sqlite(rusqlite::Error::QueryReturnedNoRows))
    }

    pub fn get_agent_pr_run(&self, id: &str) -> StoreResult<Option<AgentPrRun>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let row = guard
            .query_row(
                "SELECT id, repo_full_name, issue_number, branch, summary, \
                        diff_lines, status, pr_url, error, created_at_ms, \
                        updated_at_ms \
                   FROM agent_pr_runs WHERE id = ?1",
                params![id],
                row_to_agent_pr_run,
            )
            .optional()?;
        Ok(row)
    }

    /// Approve a `pending_approval` gate row. Returns the freshly-approved
    /// row, or `None` if it wasn't pending (already resolved / not found) so
    /// callers can render a 409-style "already resolved" without racing two
    /// surfaces (mirrors the reply-approval CAS guard).
    pub fn approve_agent_pr_run(&self, id: &str) -> StoreResult<Option<AgentPrRun>> {
        self.transition_gate(id, "approved", None, None)
    }

    /// Reject a `pending_approval` gate row. Same CAS semantics as approve.
    pub fn reject_agent_pr_run(&self, id: &str) -> StoreResult<Option<AgentPrRun>> {
        self.transition_gate(id, "rejected", None, Some("rejected by reviewer"))
    }

    /// Mark a gate row `pr_opened` with its URL (terminal). Unconditional —
    /// only the loop calls this, right after it opens the draft PR.
    pub fn mark_agent_pr_opened(&self, id: &str, pr_url: &str) -> StoreResult<()> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "UPDATE agent_pr_runs SET status = 'pr_opened', pr_url = ?2, \
                    updated_at_ms = ?3 WHERE id = ?1",
            params![id, pr_url, now],
        )?;
        Ok(())
    }

    /// Mark a gate row `failed` with an error (terminal).
    pub fn mark_agent_pr_failed(&self, id: &str, error: &str) -> StoreResult<()> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "UPDATE agent_pr_runs SET status = 'failed', error = ?2, \
                    updated_at_ms = ?3 WHERE id = ?1",
            params![id, error, now],
        )?;
        Ok(())
    }

    /// CAS-style transition: only mutate when still `pending_approval`.
    fn transition_gate(
        &self,
        id: &str,
        new_status: &str,
        pr_url: Option<&str>,
        error: Option<&str>,
    ) -> StoreResult<Option<AgentPrRun>> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n = guard.execute(
            "UPDATE agent_pr_runs \
                SET status = ?2, pr_url = COALESCE(?3, pr_url), \
                    error = COALESCE(?4, error), updated_at_ms = ?5 \
              WHERE id = ?1 AND status = 'pending_approval'",
            params![id, new_status, pr_url, error, now],
        )?;
        drop(guard);
        if n == 0 {
            return Ok(None);
        }
        self.get_agent_pr_run(id)
    }

    /// True if an open (non-terminal) gate row already exists for this
    /// (repo, issue). Dedup guard so the loop doesn't queue two approval
    /// cards for the same issue. `pending_approval` + `approved` (queued for
    /// the open-PR step) both count as open.
    pub fn has_open_agent_pr_run(
        &self,
        repo_full_name: &str,
        issue_number: i64,
    ) -> StoreResult<bool> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let found: Option<i64> = guard
            .query_row(
                "SELECT 1 FROM agent_pr_runs \
                  WHERE repo_full_name = ?1 COLLATE NOCASE \
                    AND issue_number = ?2 \
                    AND status IN ('pending_approval','approved') \
                  LIMIT 1",
                params![repo_full_name, issue_number],
                |r| r.get(0),
            )
            .optional()?;
        Ok(found.is_some())
    }

    /// Per-repo PR-run history, newest first (dashboard audit view).
    pub fn list_agent_pr_runs(
        &self,
        repo_full_name: Option<&str>,
        limit: i64,
    ) -> StoreResult<Vec<AgentPrRun>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut out = Vec::new();
        match repo_full_name {
            Some(repo) => {
                let mut stmt = guard.prepare(
                    "SELECT id, repo_full_name, issue_number, branch, summary, \
                            diff_lines, status, pr_url, error, created_at_ms, \
                            updated_at_ms \
                       FROM agent_pr_runs \
                      WHERE repo_full_name = ?1 COLLATE NOCASE \
                      ORDER BY created_at_ms DESC LIMIT ?2",
                )?;
                let rows =
                    stmt.query_map(params![repo, limit], row_to_agent_pr_run)?;
                for r in rows {
                    out.push(r?);
                }
            }
            None => {
                let mut stmt = guard.prepare(
                    "SELECT id, repo_full_name, issue_number, branch, summary, \
                            diff_lines, status, pr_url, error, created_at_ms, \
                            updated_at_ms \
                       FROM agent_pr_runs \
                      ORDER BY created_at_ms DESC LIMIT ?1",
                )?;
                let rows = stmt.query_map(params![limit], row_to_agent_pr_run)?;
                for r in rows {
                    out.push(r?);
                }
            }
        }
        Ok(out)
    }

    // =====================================================================
    // #58.1 — scheduled outbound posts (cross-platform queue)
    // =====================================================================

    /// Queue a new outbound post. `media_paths` is a JSON array string or
    /// `None` for a text post. Returns the generated row id. Status starts
    /// at `queued`; the serve-tick fire loop drives it through the
    /// `previewed` → `posted`/`failed`/`cancelled` lifecycle.
    pub fn enqueue_scheduled_post(
        &self,
        platform: &str,
        body: &str,
        media_paths: Option<&str>,
        fire_at_ms: i64,
        thread_parent: Option<&str>,
    ) -> StoreResult<String> {
        let id = Uuid::new_v4().to_string();
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "INSERT INTO scheduled_posts \
                (id, platform, body, media_paths, fire_at_ms, status, \
                 thread_parent, created_at_ms) \
             VALUES (?1,?2,?3,?4,?5,'queued',?6,?7)",
            params![id, platform, body, media_paths, fire_at_ms, thread_parent, now],
        )?;
        Ok(id)
    }

    /// #240 — route a queued post through SocialAPI.ai to a specific connected
    /// account. `platform` on the row should already hold the real
    /// sub-platform (instagram / x / linkedin / …). No-op-safe to call before
    /// firing. Returns whether a row was updated.
    pub fn set_scheduled_post_socialapi_account(
        &self,
        id: &str,
        socialapi_account_id: &str,
    ) -> StoreResult<bool> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n = guard.execute(
            "UPDATE scheduled_posts SET socialapi_account_id = ?2 WHERE id = ?1",
            params![id, socialapi_account_id],
        )?;
        Ok(n > 0)
    }

    /// Posts that are `queued` and within `horizon_ms` of firing but have no
    /// preview card yet — the T-30min preview batch.
    pub fn scheduled_posts_due_for_preview(
        &self,
        now_ms: i64,
        horizon_ms: i64,
    ) -> StoreResult<Vec<ScheduledPost>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT id, platform, body, media_paths, fire_at_ms, status, \
                    approval_msg, posted_at_ms, external_id, thread_parent, \
                    created_at_ms, socialapi_account_id \
               FROM scheduled_posts \
              WHERE status = 'queued' AND approval_msg IS NULL \
                AND fire_at_ms <= ?1 \
              ORDER BY fire_at_ms ASC",
        )?;
        let rows = stmt
            .query_map(params![now_ms + horizon_ms], row_to_scheduled_post)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Posts whose `fire_at_ms` has arrived and are still `previewed`
    /// (user did not cancel) or `queued` (post-silently mode skipped the
    /// preview) — the T-0 publish batch.
    pub fn scheduled_posts_due_to_fire(
        &self,
        now_ms: i64,
    ) -> StoreResult<Vec<ScheduledPost>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT id, platform, body, media_paths, fire_at_ms, status, \
                    approval_msg, posted_at_ms, external_id, thread_parent, \
                    created_at_ms, socialapi_account_id \
               FROM scheduled_posts \
              WHERE status IN ('previewed','queued') AND fire_at_ms <= ?1 \
              ORDER BY fire_at_ms ASC",
        )?;
        let rows = stmt
            .query_map(params![now_ms], row_to_scheduled_post)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Move a post to `previewed` and record the Discord preview message id.
    pub fn mark_scheduled_post_previewed(
        &self,
        id: &str,
        approval_msg: Option<&str>,
    ) -> StoreResult<()> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "UPDATE scheduled_posts SET status = 'previewed', approval_msg = ?2 \
              WHERE id = ?1",
            params![id, approval_msg],
        )?;
        Ok(())
    }

    /// Terminal transition: `posted` (with the platform's external id) or a
    /// non-ok status (`failed` / `cancelled`).
    pub fn mark_scheduled_post_status(
        &self,
        id: &str,
        status: ScheduledPostStatus,
        external_id: Option<&str>,
    ) -> StoreResult<()> {
        let posted_at = if status == ScheduledPostStatus::Posted {
            Some(now_millis())
        } else {
            None
        };
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "UPDATE scheduled_posts \
                SET status = ?2, external_id = COALESCE(?3, external_id), \
                    posted_at_ms = COALESCE(?4, posted_at_ms) \
              WHERE id = ?1",
            params![id, status.as_str(), external_id, posted_at],
        )?;
        Ok(())
    }

    /// Cancel a still-pending (`queued`/`previewed`) post. No-op (returns
    /// `false`) if it already fired or was cancelled.
    pub fn cancel_scheduled_post(&self, id: &str) -> StoreResult<bool> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n = guard.execute(
            "UPDATE scheduled_posts SET status = 'cancelled' \
              WHERE id = ?1 AND status IN ('queued','previewed')",
            params![id],
        )?;
        Ok(n > 0)
    }

    /// All not-yet-terminal posts, soonest first — the dashboard / CLI queue.
    pub fn list_pending_scheduled_posts(&self) -> StoreResult<Vec<ScheduledPost>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT id, platform, body, media_paths, fire_at_ms, status, \
                    approval_msg, posted_at_ms, external_id, thread_parent, \
                    created_at_ms, socialapi_account_id \
               FROM scheduled_posts \
              WHERE status IN ('queued','previewed') \
              ORDER BY fire_at_ms ASC",
        )?;
        let rows = stmt
            .query_map([], row_to_scheduled_post)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    // =====================================================================
    // #58.2-.4 — spine accessors (own-post comments, friend feed,
    // connection-request queue). Minimal dedup-key surface so the sources
    // can be implemented incrementally without further migrations.
    // =====================================================================

    /// Register one of the user's own posts to watch for comments. Idempotent
    /// on `(platform, external_id)`.
    pub fn upsert_own_post(
        &self,
        platform: &str,
        external_id: &str,
        posted_at_ms: i64,
        poll_until_ms: i64,
    ) -> StoreResult<String> {
        let id = Uuid::new_v4().to_string();
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "INSERT INTO own_posts \
                (id, platform, external_id, posted_at_ms, poll_until_ms, created_at_ms) \
             VALUES (?1,?2,?3,?4,?5,?6) \
             ON CONFLICT(platform, external_id) DO UPDATE SET \
                poll_until_ms = MAX(poll_until_ms, excluded.poll_until_ms)",
            params![id, platform, external_id, posted_at_ms, poll_until_ms, now],
        )?;
        let row_id: String = guard.query_row(
            "SELECT id FROM own_posts WHERE platform = ?1 AND external_id = ?2",
            params![platform, external_id],
            |r| r.get(0),
        )?;
        Ok(row_id)
    }

    /// Record a freshly-seen comment. Returns `false` if it was already seen
    /// (the `(own_post_id, external_id)` unique guard tripped) so the caller
    /// only synthesizes a WorkItem for genuinely new comments.
    pub fn record_seen_comment(
        &self,
        own_post_id: &str,
        external_id: &str,
        author_handle: Option<&str>,
        body: Option<&str>,
    ) -> StoreResult<bool> {
        let id = Uuid::new_v4().to_string();
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n = guard.execute(
            "INSERT OR IGNORE INTO seen_comments \
                (id, own_post_id, external_id, author_handle, body, created_at_ms) \
             VALUES (?1,?2,?3,?4,?5,?6)",
            params![id, own_post_id, external_id, author_handle, body, now],
        )?;
        Ok(n > 0)
    }

    /// Add (or refresh) a friend to the engagement watchlist.
    pub fn upsert_friend_watch(
        &self,
        platform: &str,
        handle: &str,
        wiki_slug: Option<&str>,
        engagement: &str,
    ) -> StoreResult<()> {
        let id = Uuid::new_v4().to_string();
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "INSERT INTO friend_watchlist \
                (id, platform, handle, wiki_slug, engagement, added_at_ms) \
             VALUES (?1,?2,?3,?4,?5,?6) \
             ON CONFLICT(platform, handle) DO UPDATE SET \
                wiki_slug = excluded.wiki_slug, \
                engagement = excluded.engagement",
            params![id, platform, handle, wiki_slug, engagement, now],
        )?;
        Ok(())
    }

    /// Queue an inbound connection request for triage. Idempotent on
    /// `(platform, external_id)`; returns `false` if already queued.
    pub fn record_connection_request(
        &self,
        platform: &str,
        external_id: &str,
        requester_name: Option<&str>,
        requester_url: Option<&str>,
        message: Option<&str>,
    ) -> StoreResult<bool> {
        let id = Uuid::new_v4().to_string();
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n = guard.execute(
            "INSERT OR IGNORE INTO connection_requests \
                (id, platform, external_id, requester_name, requester_url, \
                 message, created_at_ms) \
             VALUES (?1,?2,?3,?4,?5,?6,?7)",
            params![
                id, platform, external_id, requester_name, requester_url, message, now
            ],
        )?;
        Ok(n > 0)
    }

    /// Resolve a queued connection request to a terminal decision.
    pub fn decide_connection_request(
        &self,
        id: &str,
        decision: &str,
    ) -> StoreResult<()> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "UPDATE connection_requests \
                SET decision = ?2, decided_at_ms = ?3 WHERE id = ?1",
            params![id, decision, now],
        )?;
        Ok(())
    }

    // ---- #58.2 own-post comment poller query surface ----

    /// Own posts still inside their poll window for `platform`, least-recently
    /// polled first so a tick spreads load. The poller fetches comments for
    /// each and diffs them against `seen_comments`.
    pub fn own_posts_due_for_poll(
        &self,
        platform: &str,
        now_ms: i64,
    ) -> StoreResult<Vec<OwnPost>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT id, platform, external_id, posted_at_ms, poll_until_ms, \
                    last_polled_ms, created_at_ms \
               FROM own_posts \
              WHERE platform = ?1 AND poll_until_ms >= ?2 \
              ORDER BY COALESCE(last_polled_ms, 0) ASC",
        )?;
        let rows = stmt
            .query_map(params![platform, now_ms], row_to_own_post)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Stamp `last_polled_ms = now` after a comment-poll pass for this post.
    pub fn mark_own_post_polled(&self, id: &str) -> StoreResult<()> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        guard.execute(
            "UPDATE own_posts SET last_polled_ms = ?2 WHERE id = ?1",
            params![id, now],
        )?;
        Ok(())
    }

    // ---- #243 SocialAPI.ai own-post comment engagement query surface ----

    /// Record a freshly-seen SocialAPI.ai comment. Returns `false` if the
    /// `(post_id, comment_id)` pair was already seen so the own-post poller
    /// only synthesizes a WorkItem for genuinely new comments. Mirrors
    /// [`Store::record_seen_comment`] but against the `socialapi_seen_comments`
    /// ledger added in #238.
    pub fn record_seen_socialapi_comment(
        &self,
        post_id: &str,
        comment_id: &str,
        author: Option<&str>,
        text: Option<&str>,
    ) -> StoreResult<bool> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n = guard.execute(
            "INSERT OR IGNORE INTO socialapi_seen_comments \
                (post_id, comment_id, author, text, seen_at_ms) \
             VALUES (?1,?2,?3,?4,?5)",
            params![post_id, comment_id, author, text, now],
        )?;
        Ok(n > 0)
    }

    /// One-shot dedup for inbound SocialAPI.ai DM messages (#242). Returns
    /// `true` the first time a `(conversation_id, message_id)` pair is seen and
    /// `false` on every subsequent call. Mirrors
    /// [`Store::record_seen_socialapi_comment`] but against the
    /// `socialapi_seen_dms` ledger.
    ///
    /// #671: this is written when a DM reaches a TERMINAL outcome (skipped,
    /// carded, dry-run, already-actioned) rather than when it is emitted, so a
    /// transient triage/draft failure leaves the DM unledgered and the next
    /// poll re-feeds it. Read side: [`Store::is_socialapi_dm_seen`].
    pub fn record_seen_socialapi_dm(
        &self,
        conversation_id: &str,
        message_id: &str,
        author: Option<&str>,
        text: Option<&str>,
    ) -> StoreResult<bool> {
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n = guard.execute(
            "INSERT OR IGNORE INTO socialapi_seen_dms \
                (conversation_id, message_id, author, text, seen_at_ms) \
             VALUES (?1,?2,?3,?4,?5)",
            params![conversation_id, message_id, author, text, now],
        )?;
        Ok(n > 0)
    }

    /// True iff `(conversation_id, message_id)` is already in the
    /// `socialapi_seen_dms` ledger. Read-only counterpart of
    /// [`Store::record_seen_socialapi_dm`]: the DM source gates emission on
    /// this so it no longer claims a message it hasn't carried to a terminal
    /// outcome (#671).
    pub fn is_socialapi_dm_seen(
        &self,
        conversation_id: &str,
        message_id: &str,
    ) -> StoreResult<bool> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let row: Option<i64> = guard
            .query_row(
                "SELECT 1 FROM socialapi_seen_dms \
                 WHERE conversation_id = ?1 AND message_id = ?2",
                params![conversation_id, message_id],
                |r| r.get(0),
            )
            .optional()?;
        Ok(row.is_some())
    }

    /// Insert one inbound SocialAPI.ai webhook event (#249) idempotently,
    /// de-duped by the receiver-synthesized `id`. Returns `true` when the row
    /// was newly inserted (`false` on a duplicate id). The Express webhook
    /// receiver is the normal writer; this method exists so the Rust side can
    /// also seed events in tests and stays the single schema source. New rows
    /// land with `processed = 0` and are picked up by
    /// [`Store::take_unprocessed_socialapi_webhook_events`].
    ///
    /// #529: `kind` is validated here, not just by the sqlite CHECK. The CHECK
    /// only exists on databases created after that fix, and a row with an
    /// unrecognized kind is invisible forever — the drain filters `kind = ?1`,
    /// so it would sit at `processed = 0` and never be emitted or cleaned up.
    pub fn insert_socialapi_webhook_event(
        &self,
        id: &str,
        kind: &str,
        account_id: Option<&str>,
        payload_json: &str,
    ) -> StoreResult<bool> {
        if !SOCIALAPI_WEBHOOK_KINDS.contains(&kind) {
            return Err(StoreError::InvalidInput(format!(
                "socialapi webhook event kind must be one of {SOCIALAPI_WEBHOOK_KINDS:?}, got {kind:?}"
            )));
        }
        let now = now_millis();
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n = guard.execute(
            "INSERT OR IGNORE INTO socialapi_webhook_events \
                (id, kind, account_id, payload_json, received_at_ms, processed) \
             VALUES (?1,?2,?3,?4,?5,0)",
            params![id, kind, account_id, payload_json, now],
        )?;
        Ok(n > 0)
    }

    /// Drain unprocessed inbound SocialAPI.ai webhook events (#249), oldest
    /// first. These are the near-real-time fast-path rows the Express receiver
    /// persisted; the DM source and own-post comment trigger read them ahead of
    /// the API poll and emit the corresponding WorkItems, then mark each
    /// processed via [`Store::mark_socialapi_webhook_event_processed`]. Reusing
    /// `socialapi_seen_{dms,comments}` downstream means a webhook-delivered item
    /// and a later poll of the same item collapse to a single draft. Reads do
    /// not mutate — the caller marks each row processed only after it has
    /// durably recorded the item in the seen-ledger, so a crash mid-drain just
    /// re-drains (and the seen-ledger dedups).
    pub fn take_unprocessed_socialapi_webhook_events(
        &self,
        kind: &str,
        limit: u32,
    ) -> StoreResult<Vec<SocialapiWebhookEvent>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT id, kind, account_id, payload_json, received_at_ms \
               FROM socialapi_webhook_events \
              WHERE processed = 0 AND kind = ?1 \
              ORDER BY received_at_ms ASC, id ASC \
              LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(params![kind, limit], |r| {
                Ok(SocialapiWebhookEvent {
                    id: r.get(0)?,
                    kind: r.get(1)?,
                    account_id: r.get::<_, Option<String>>(2)?,
                    payload_json: r.get(3)?,
                    received_at_ms: r.get(4)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Mark one drained SocialAPI.ai webhook event processed (#249) so the next
    /// drain skips it. Returns the rows touched (0 when `id` is unknown).
    pub fn mark_socialapi_webhook_event_processed(&self, id: &str) -> StoreResult<usize> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n = guard.execute(
            "UPDATE socialapi_webhook_events SET processed = 1 WHERE id = ?1",
            params![id],
        )?;
        Ok(n)
    }

    /// Read one row out of the `config` table — the key/value store the
    /// Express dashboard writes its pasted secrets into (`setConfig` in
    /// `src/db.ts`). `Ok(None)` covers both "no such key" and an empty value,
    /// so callers can treat a blank as unset.
    ///
    /// The store deliberately exposed no generic getter before #525, which is
    /// why every consumer that wanted a dashboard-written key reopened the db
    /// by hand with a different precedence. Route new ones through here.
    pub fn get_config(&self, key: &str) -> StoreResult<Option<String>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let val: rusqlite::Result<String> = guard.query_row(
            "SELECT value FROM config WHERE key = ?1",
            params![key],
            |r| r.get(0),
        );
        match val {
            Ok(v) if !v.trim().is_empty() => Ok(Some(v.trim().to_string())),
            Ok(_) => Ok(None),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Account handles of every registered SocialAPI.ai account, lowercased
    /// and stripped of a leading `@`. This is the set of identities that count
    /// as "us" when deciding whether an inbound-looking DM is really our own
    /// outbound message (#526). Inactive accounts are included on purpose —
    /// disabling an account must not make its past outbound messages start
    /// looking inbound.
    pub fn socialapi_account_handles(&self) -> StoreResult<Vec<String>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT account_handle FROM socialapi_accounts \
             WHERE account_handle IS NOT NULL AND TRIM(account_handle) <> ''",
        )?;
        let rows = stmt
            .query_map([], |r| r.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows
            .into_iter()
            .map(|h| h.trim().trim_start_matches('@').to_ascii_lowercase())
            .filter(|h| !h.is_empty())
            .collect())
    }

    /// Account ids of the active (polling-enabled) SocialAPI.ai accounts. The
    /// own-post comment poller iterates these to scope `list_comments` per
    /// account. Empty when no accounts are registered yet, which makes the
    /// engagement inert (always-spawn-empty-is-free).
    pub fn active_socialapi_account_ids(&self) -> StoreResult<Vec<String>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT id FROM socialapi_accounts WHERE active = 1 ORDER BY id ASC",
        )?;
        let rows = stmt
            .query_map([], |r| r.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// List every registered SocialAPI.ai account (active + inactive), in a
    /// stable order (active first, then by id). Powers the CLI `socialapi
    /// list` verb (#245), which has to surface disabled accounts too — unlike
    /// [`Store::active_socialapi_account_ids`], which only feeds the live
    /// engagement loop.
    pub fn list_socialapi_accounts(&self) -> StoreResult<Vec<SocialapiAccount>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT id, brand_id, platform, display_name, account_handle, active \
               FROM socialapi_accounts ORDER BY active DESC, id ASC",
        )?;
        let rows = stmt
            .query_map([], |r| {
                Ok(SocialapiAccount {
                    id: r.get(0)?,
                    brand_id: r.get::<_, Option<String>>(1)?,
                    platform: r.get(2)?,
                    display_name: r.get::<_, Option<String>>(3)?,
                    account_handle: r.get::<_, Option<String>>(4)?,
                    active: r.get::<_, i64>(5)? != 0,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Toggle a SocialAPI.ai account's `active` gate. Returns the number of
    /// rows touched (0 when `id` is unknown) so the CLI `socialapi disable`
    /// verb can tell the operator whether the id matched. Also bumps
    /// `updated_at_ms` (caller supplies `now_ms`, matching the store's
    /// timestamp-injection convention) so the change is observable in audit
    /// listings.
    pub fn set_socialapi_account_active(
        &self,
        id: &str,
        active: bool,
        now_ms: i64,
    ) -> StoreResult<usize> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n = guard.execute(
            "UPDATE socialapi_accounts SET active = ?2, updated_at_ms = ?3 WHERE id = ?1",
            params![id, active as i64, now_ms],
        )?;
        Ok(n)
    }

    // ---- #58.3 friend-feed engagement query surface ----

    /// Active (not paused) friend watches for `platform`. The friend-feed
    /// source iterates these and emits a `friend_post` WorkItem per fresh
    /// post.
    pub fn active_friend_watch(
        &self,
        platform: &str,
        now_ms: i64,
    ) -> StoreResult<Vec<FriendWatch>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT id, platform, handle, wiki_slug, engagement, added_at_ms, \
                    paused_until_ms \
               FROM friend_watchlist \
              WHERE platform = ?1 \
                AND (paused_until_ms IS NULL OR paused_until_ms <= ?2) \
              ORDER BY added_at_ms ASC",
        )?;
        let rows = stmt
            .query_map(params![platform, now_ms], row_to_friend_watch)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Record a freshly-seen friend post. Returns `false` if it was already
    /// seen (the `(watchlist_id, external_id)` unique guard tripped) so the
    /// caller only synthesizes a WorkItem for genuinely new posts.
    pub fn record_friend_post_seen(
        &self,
        watchlist_id: &str,
        external_id: &str,
        posted_at_ms: i64,
    ) -> StoreResult<bool> {
        let id = Uuid::new_v4().to_string();
        let guard = self.conn.lock().expect("store mutex poisoned");
        let n = guard.execute(
            "INSERT OR IGNORE INTO friend_posts_seen \
                (id, watchlist_id, external_id, posted_at_ms) \
             VALUES (?1,?2,?3,?4)",
            params![id, watchlist_id, external_id, posted_at_ms],
        )?;
        Ok(n > 0)
    }

    // ---- #58.4 connection-request triage query surface ----

    /// All connection requests still awaiting a decision, oldest first.
    pub fn pending_connection_requests(
        &self,
    ) -> StoreResult<Vec<ConnectionRequestRow>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT id, platform, external_id, requester_name, requester_url, \
                    message, decision, decided_at_ms, triage_id, created_at_ms \
               FROM connection_requests \
              WHERE decision = 'pending' \
              ORDER BY created_at_ms ASC",
        )?;
        let rows = stmt
            .query_map([], row_to_connection_request)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Fetch one connection request by row id (the approver re-hydrates the
    /// invitation urn from this on a button click).
    pub fn connection_request_by_id(
        &self,
        id: &str,
    ) -> StoreResult<Option<ConnectionRequestRow>> {
        let guard = self.conn.lock().expect("store mutex poisoned");
        let row = guard
            .query_row(
                "SELECT id, platform, external_id, requester_name, \
                        requester_url, message, decision, decided_at_ms, \
                        triage_id, created_at_ms \
                   FROM connection_requests WHERE id = ?1",
                params![id],
                row_to_connection_request,
            )
            .optional()?;
        Ok(row)
    }
}

fn row_to_tone_profile(r: &rusqlite::Row) -> rusqlite::Result<ToneProfile> {
    Ok(ToneProfile {
        id: r.get(0)?,
        scope_kind: r.get(1)?,
        scope_value: r.get(2)?,
        account_entity_id: r.get::<_, Option<String>>(3)?,
        summary: r.get(4)?,
        exemplar_ids: r.get(5)?,
        sample_count: r.get(6)?,
        sample_count_at_refresh: r.get(7)?,
        last_refreshed_at: r.get(8)?,
        created_at_ms: r.get(9)?,
        updated_at_ms: r.get(10)?,
    })
}

fn row_to_tone_example(r: &rusqlite::Row) -> rusqlite::Result<ToneExample> {
    Ok(ToneExample {
        id: r.get(0)?,
        source: r.get(1)?,
        action_id: r.get::<_, Option<String>>(2)?,
        message_id: r.get::<_, Option<String>>(3)?,
        account_entity_id: r.get(4)?,
        recipient_email: r.get(5)?,
        recipient_domain: r.get(6)?,
        subject: r.get::<_, Option<String>>(7)?,
        body: r.get(8)?,
        body_chars: r.get(9)?,
        sent_at_ms: r.get(10)?,
        ingested_at_ms: r.get(11)?,
        weight: r.get(12)?,
    })
}

fn bare_lower(raw: &str) -> String {
    let s = if let (Some(open), Some(close)) = (raw.find('<'), raw.rfind('>')) {
        if open < close {
            &raw[open + 1..close]
        } else {
            raw
        }
    } else {
        raw
    };
    s.trim().to_ascii_lowercase()
}

fn row_to_slack_workspace(r: &rusqlite::Row) -> rusqlite::Result<SlackWorkspace> {
    Ok(SlackWorkspace {
        id: r.get(0)?,
        team_id: r.get(1)?,
        team_name: r.get(2)?,
        entity_id: r.get(3)?,
        connection_id: r.get(4)?,
        user_id: r.get(5)?,
        active: r.get::<_, i64>(6)? != 0,
        created_at_ms: r.get(7)?,
    })
}

fn row_to_telegram_bot(r: &rusqlite::Row) -> rusqlite::Result<TelegramBot> {
    Ok(TelegramBot {
        id: r.get(0)?,
        bot_id: r.get(1)?,
        bot_username: r.get(2)?,
        owner_chat_id: r.get(3)?,
        last_update_id: r.get(4)?,
        active: r.get::<_, i64>(5)? != 0,
        created_at_ms: r.get(6)?,
    })
}

fn row_to_whatsapp_device(r: &rusqlite::Row) -> rusqlite::Result<WhatsappDevice> {
    Ok(WhatsappDevice {
        id: r.get(0)?,
        phone: r.get(1)?,
        device_jid: r.get(2)?,
        user_jid: r.get(3)?,
        paired_at_ms: r.get(4)?,
        last_event_at_ms: r.get(5)?,
        session_status: r.get(6)?,
        active: r.get::<_, i64>(7)? != 0,
        created_at_ms: r.get(8)?,
    })
}

fn row_to_user_loop(r: &rusqlite::Row) -> rusqlite::Result<UserLoop> {
    Ok(UserLoop {
        id: r.get(0)?,
        owner: r.get(1)?,
        channel: r.get(2)?,
        channel_ref: r.get(3)?,
        interval_secs: r.get(4)?,
        prompt: r.get(5)?,
        status: r.get(6)?,
        last_run_ms: r.get(7)?,
        last_status: r.get(8)?,
        fail_count: r.get(9)?,
        created_at_ms: r.get(10)?,
        updated_at_ms: r.get(11)?,
        expires_at_ms: r.get(12)?,
        cron_expr: r.get(13)?,
        tz: r.get(14)?,
    })
}

fn row_to_agent_repo(r: &rusqlite::Row) -> rusqlite::Result<AgentRepo> {
    Ok(AgentRepo {
        id: r.get(0)?,
        full_name: r.get(1)?,
        base_branch: r.get(2)?,
        build_cmd: r.get(3)?,
        blast_radius_extra: r.get(4)?,
        max_diff_lines: r.get(5)?,
        enabled: r.get::<_, i64>(6)? != 0,
        created_at_ms: r.get(7)?,
        updated_at_ms: r.get(8)?,
    })
}

fn row_to_agent_pr_run(r: &rusqlite::Row) -> rusqlite::Result<AgentPrRun> {
    Ok(AgentPrRun {
        id: r.get(0)?,
        repo_full_name: r.get(1)?,
        issue_number: r.get(2)?,
        branch: r.get(3)?,
        summary: r.get(4)?,
        diff_lines: r.get(5)?,
        status: r.get(6)?,
        pr_url: r.get(7)?,
        error: r.get(8)?,
        created_at_ms: r.get(9)?,
        updated_at_ms: r.get(10)?,
    })
}

fn row_to_scheduled_post(r: &rusqlite::Row) -> rusqlite::Result<ScheduledPost> {
    Ok(ScheduledPost {
        id: r.get(0)?,
        platform: r.get(1)?,
        body: r.get(2)?,
        media_paths: r.get::<_, Option<String>>(3)?,
        fire_at_ms: r.get(4)?,
        status: r.get(5)?,
        approval_msg: r.get::<_, Option<String>>(6)?,
        posted_at_ms: r.get::<_, Option<i64>>(7)?,
        external_id: r.get::<_, Option<String>>(8)?,
        thread_parent: r.get::<_, Option<String>>(9)?,
        created_at_ms: r.get(10)?,
        socialapi_account_id: r.get::<_, Option<String>>(11)?,
    })
}

fn row_to_own_post(r: &rusqlite::Row) -> rusqlite::Result<OwnPost> {
    Ok(OwnPost {
        id: r.get(0)?,
        platform: r.get(1)?,
        external_id: r.get(2)?,
        posted_at_ms: r.get(3)?,
        poll_until_ms: r.get(4)?,
        last_polled_ms: r.get::<_, Option<i64>>(5)?,
        created_at_ms: r.get(6)?,
    })
}

fn row_to_friend_watch(r: &rusqlite::Row) -> rusqlite::Result<FriendWatch> {
    Ok(FriendWatch {
        id: r.get(0)?,
        platform: r.get(1)?,
        handle: r.get(2)?,
        wiki_slug: r.get::<_, Option<String>>(3)?,
        engagement: r.get(4)?,
        added_at_ms: r.get(5)?,
        paused_until_ms: r.get::<_, Option<i64>>(6)?,
    })
}

fn row_to_connection_request(
    r: &rusqlite::Row,
) -> rusqlite::Result<ConnectionRequestRow> {
    Ok(ConnectionRequestRow {
        id: r.get(0)?,
        platform: r.get(1)?,
        external_id: r.get(2)?,
        requester_name: r.get::<_, Option<String>>(3)?,
        requester_url: r.get::<_, Option<String>>(4)?,
        message: r.get::<_, Option<String>>(5)?,
        decision: r.get(6)?,
        decided_at_ms: r.get::<_, Option<i64>>(7)?,
        triage_id: r.get::<_, Option<String>>(8)?,
        created_at_ms: r.get(9)?,
    })
}

fn row_to_subscription(r: &rusqlite::Row) -> rusqlite::Result<ChannelSubscription> {
    let mode_str: String = r.get(4)?;
    let mode = SubscriptionMode::parse(&mode_str).ok_or_else(|| {
        rusqlite::Error::FromSqlConversionFailure(
            4,
            rusqlite::types::Type::Text,
            format!("unknown subscription mode: {mode_str}").into(),
        )
    })?;
    Ok(ChannelSubscription {
        id: r.get(0)?,
        platform: r.get(1)?,
        channel_id: r.get(2)?,
        display_name: r.get(3)?,
        mode,
        active: r.get::<_, i64>(5)? != 0,
        account_id: r.get::<_, Option<String>>(6)?,
        last_seen_message_id: r.get::<_, Option<String>>(7)?,
        last_digest_at_ms: r.get::<_, Option<i64>>(8)?,
        created_at_ms: r.get(9)?,
        updated_at_ms: r.get(10)?,
    })
}

#[derive(Debug, Clone)]
pub struct RetryableReply {
    pub action: ActionRecord,
    pub retry_count: i64,
    pub email: Email,
}

/// #449 — one pending approval card, reduced to the fields the staleness
/// reconciliation sweep needs to decide whether it still deserves the user's
/// attention. Returned by [`Store::pending_actions_for_reconcile`].
#[derive(Debug, Clone)]
pub struct PendingActionRow {
    pub id: String,
    pub thread_id: Option<String>,
    pub from_email: String,
    pub subject: String,
    pub body: String,
    /// True when this card has no draft to approve (draftBody NULL/blank).
    /// A card with nothing to approve is stale on its face (#484).
    pub draft_empty: bool,
}

/// #48 — the three code-mode columns on `actions`, returned by
/// [`Store::action_code_mode_fields`]. `mode` is `'classic'` or `'code'` (never
/// NULL post-migration). For classic rows `generated_source` and
/// `tool_call_trace` are `None`; for code-mode rows both are populated.
/// `tool_call_trace` is the raw JSON string as stored — callers (channel-core)
/// are responsible for deserializing it into `Vec<ToolCallRecord>`.
#[derive(Debug, Clone)]
pub struct ActionCodeModeFields {
    pub mode: String,
    pub generated_source: Option<String>,
    pub tool_call_trace: Option<String>,
}

/// One row from `draft_revisions` (#37). The full chain for a given action is
/// `iteration = 0, 1, 2, ...` with `outcome ∈ { superseded, pending, approved,
/// skipped }`. `feedbackText` is the user's Revise feedback for this draft;
/// it's NULL on the iteration-0 (auto-generated) draft and Some on every
/// revised draft thereafter.
#[derive(Debug, Clone)]
pub struct RevisionRecord {
    pub id: String,
    pub action_id: String,
    pub iteration: i64,
    pub draft_body: String,
    pub feedback_text: Option<String>,
    pub preset_id: Option<String>,
    pub outcome: String,
    pub model_id: String,
    pub prompt_tokens: Option<i64>,
    pub completion_tokens: Option<i64>,
    pub created_at_ms: i64,
}

fn row_to_revision_record(r: &rusqlite::Row) -> rusqlite::Result<RevisionRecord> {
    Ok(RevisionRecord {
        id: r.get(0)?,
        action_id: r.get(1)?,
        iteration: r.get(2)?,
        draft_body: r.get(3)?,
        feedback_text: r.get::<_, Option<String>>(4)?,
        preset_id: r.get::<_, Option<String>>(5)?,
        outcome: r.get(6)?,
        model_id: r.get(7)?,
        prompt_tokens: r.get::<_, Option<i64>>(8)?,
        completion_tokens: r.get::<_, Option<i64>>(9)?,
        created_at_ms: r.get(10)?,
    })
}

/// Fixed interval between nudges for a pending approval card. 6 hours.
pub const NUDGE_INTERVAL_MS: i64 = 6 * 60 * 60 * 1000;

/// A pending action in the nudge queue, packaged with the email body the
/// approval card was rendered from. `nudge_count` is how many times this card
/// has been surfaced (0 = still in backlog, ≥1 = currently active);
/// `next_nudge_at_ms` is when it next becomes eligible for posting/re-posting.
#[derive(Debug, Clone)]
pub struct PendingNudge {
    pub action: ActionWithEmail,
    pub nudge_count: i64,
    pub next_nudge_at_ms: Option<i64>,
}

fn row_to_pending_nudge(r: &rusqlite::Row) -> rusqlite::Result<PendingNudge> {
    Ok(PendingNudge {
        action: ActionWithEmail {
            action: ActionRecord {
                id: r.get(0)?,
                message_id: r.get(1)?,
                thread_id: r.get(2)?,
                from_email: r.get(3)?,
                subject: r.get(4)?,
                original_body: r.get(5)?,
                draft_body: r.get(6)?,
                status: r.get(7)?,
                error_message: r.get(8)?,
                created_at: ms_to_rfc3339(r.get::<_, i64>(9)?),
                updated_at: ms_to_rfc3339(r.get::<_, i64>(10)?),
            },
            retry_count: r.get::<_, i64>(11)?,
            draft_id: r.get::<_, Option<String>>(12)?,
            email: Email {
                attachments: Vec::new(),
                to: String::new(),
                cc: String::new(),
                message_id: r.get(1)?,
                thread_id: r.get(2)?,
                from: r.get(3)?,
                subject: r.get(4)?,
                body: r.get::<_, Option<String>>(15)?.unwrap_or_default(),
                date: r.get::<_, Option<String>>(16)?.unwrap_or_default(),
                account_entity_id: r.get::<_, Option<String>>(17)?,
                platform: r.get::<_, Option<String>>(18)?.unwrap_or_else(|| "gmail".into()),
                kind: r.get::<_, Option<String>>(19)?.unwrap_or_else(|| "dm".into()),
            },
        },
        nudge_count: r.get::<_, i64>(13)?,
        next_nudge_at_ms: r.get::<_, Option<i64>>(14)?,
    })
}

/// #473 — the outbound envelope a compose-originated card was created with.
/// The recipient fields are comma-joined bare-address lists as passed at
/// compose time; `None` = never set (fall back to `emails.from`, the pre-#473
/// behavior). `subject` (#652) is the header a Revise overrode, `None` when
/// the derived reply subject still applies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionEnvelope {
    pub to: Option<String>,
    pub cc: Option<String>,
    pub bcc: Option<String>,
    pub subject: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ActionWithEmail {
    pub action: ActionRecord,
    pub retry_count: i64,
    pub draft_id: Option<String>,
    pub email: Email,
}

fn column_exists(conn: &Connection, table: &str, column: &str) -> StoreResult<bool> {
    let sql = format!("PRAGMA table_info({table})");
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |r| r.get::<_, String>(1))?;
    for name in rows {
        if name? == column {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Detect whether `table` has a UNIQUE index covering exactly `(col_a, col_b)`.
/// Uses sqlite's PRAGMA index_list + PRAGMA index_info to walk the schema.
fn table_has_unique(
    conn: &Connection,
    table: &str,
    col_a: &str,
    col_b: &str,
) -> StoreResult<bool> {
    let index_list_sql = format!("PRAGMA index_list({table})");
    let mut stmt = conn.prepare(&index_list_sql)?;
    // index_list columns: seq, name, unique, origin, partial
    let rows = stmt.query_map([], |r| {
        Ok((
            r.get::<_, String>(1)?,
            r.get::<_, i64>(2)?,
            r.get::<_, String>(3)?,
        ))
    })?;
    for row in rows {
        let (index_name, is_unique, _origin) = row?;
        if is_unique == 0 {
            continue;
        }
        let info_sql = format!("PRAGMA index_info({index_name})");
        let mut info = conn.prepare(&info_sql)?;
        let cols: Vec<String> = info
            .query_map([], |r| r.get::<_, String>(2))?
            .collect::<Result<Vec<_>, _>>()?;
        if cols.len() == 2
            && ((cols[0] == col_a && cols[1] == col_b)
                || (cols[0] == col_b && cols[1] == col_a))
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn ms_to_rfc3339(ms: i64) -> String {
    use time::{format_description::well_known::Rfc3339, OffsetDateTime};
    OffsetDateTime::from_unix_timestamp(ms / 1000)
        .ok()
        .and_then(|dt| dt.format(&Rfc3339).ok())
        .unwrap_or_default()
}

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use serde_json;

    /// Store on a tempdir-backed file, NOT a `NamedTempFile` (#877).
    ///
    /// `NamedTempFile` deletes only its own file on drop; SQLite in WAL mode
    /// creates `-wal`/`-shm` SIDECARS next to it that nothing removes. With
    /// the verification gate running this suite on every pipeline round,
    /// /tmp accumulated 22,006 leaked files (19 GB) in three days. A tempdir
    /// removes everything under it, sidecars included.
    fn fresh_store() -> (Store, TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("store-test.db");
        {
            let conn = Connection::open(&db).unwrap();
            conn.execute_batch(
                r#"
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
                "#,
            )
            .unwrap();
        }
        (Store::open(&db).unwrap(), dir)
    }

    fn sample_email(message_id: &str) -> Email {
        Email {
            attachments: Vec::new(),
            to: String::new(),
            cc: String::new(),
            message_id: message_id.into(),
            thread_id: None,
            from: "a@b.com".into(),
            subject: "hi".into(),
            body: "hello".into(),
            date: "2026-04-13T12:00:00Z".into(),
            account_entity_id: Some("acc".into()),
            platform: "gmail".into(),
            kind: "dm".into(),
        }
    }

    #[test]
    fn upsert_email_returns_is_new() {
        let (s, _f) = fresh_store();
        let e = sample_email("m1");
        assert!(s.upsert_email(&e).unwrap());
        assert!(!s.upsert_email(&e).unwrap());
    }

    #[test]
    fn imessage_entries_seen_roundtrip() {
        let (store, _d) = fresh_store();
        // never-synced conversation reads as 0, not an error
        assert_eq!(store.get_imessage_entries_seen("+14155550123").unwrap(), 0);
        store.set_imessage_entries_seen("+14155550123", 42).unwrap();
        assert_eq!(store.get_imessage_entries_seen("+14155550123").unwrap(), 42);
        store.set_imessage_entries_seen("+14155550123", 57).unwrap();
        assert_eq!(store.get_imessage_entries_seen("+14155550123").unwrap(), 57);
    }

    #[test]
    fn upsert_email_backfill_preserves_historical_first_seen() {
        let (store, _d) = fresh_store();
        let email = sample_email("imessage:+14155550123:0");
        // historical import: firstSeenAt is the message timestamp, so the
        // wiki index renders honest "facts as of" dates (#886)
        assert!(store.upsert_email_backfill(&email, 1_600_000_000_000).unwrap());
        assert_eq!(
            store.email_first_seen_at("imessage:+14155550123:0").unwrap(),
            Some(1_600_000_000_000)
        );
        // re-import is an update, and never rewrites firstSeenAt
        assert!(!store.upsert_email_backfill(&email, 1_700_000_000_000).unwrap());
        assert_eq!(
            store.email_first_seen_at("imessage:+14155550123:0").unwrap(),
            Some(1_600_000_000_000)
        );
    }

    #[test]
    fn socialapi_tables_exist_after_migrate() {
        let (s, _f) = fresh_store();
        let guard = s.conn.lock().unwrap();
        for tbl in [
            "socialapi_accounts",
            "socialapi_seen_comments",
            "socialapi_seen_dms",
            "socialapi_webhook_events",
        ] {
            let n: i64 = guard
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                    params![tbl],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(n, 1, "expected table {tbl} to exist");
        }
    }

    #[test]
    fn gmail_fetch_cooldown_is_persisted_per_entity_and_never_moves_backward() {
        let (store, _file) = fresh_store();

        store
            .set_gmail_fetch_cooldown("rate-limited", 2_000, Some("log-first"))
            .expect("record initial cooldown");
        store
            .set_gmail_fetch_cooldown("rate-limited", 1_500, Some("log-stale"))
            .expect("an older provider timestamp cannot shorten a cooldown");
        store
            .set_gmail_fetch_cooldown("healthy", 3_000, None)
            .expect("record another entity independently");

        assert_eq!(
            store.gmail_fetch_cooldown_until("rate-limited", 1_000).unwrap(),
            Some(2_000)
        );
        assert_eq!(
            store.gmail_fetch_cooldown_until("healthy", 1_000).unwrap(),
            Some(3_000)
        );
        assert_eq!(
            store.gmail_fetch_cooldown_until("rate-limited", 2_000).unwrap(),
            None,
            "expired cooldowns must stop blocking requests"
        );
    }

    #[test]
    fn log_and_update_action_status() {
        let (s, _f) = fresh_store();
        let id = s
            .log_action(
                "m1",
                None,
                "a@b.com",
                "subj",
                None,
                None,
                ActionStatus::Pending,
            )
            .unwrap();
        s.update_action_status(&id, ActionStatus::Sent, Some("draft"), None)
            .unwrap();
    }

    #[test]
    fn mark_email_processed_sets_triage() {
        let (s, _f) = fresh_store();
        s.upsert_email(&sample_email("m1")).unwrap();
        s.mark_email_processed("m1", TriageResult::Skip).unwrap();
    }

    #[test]
    fn is_message_processed_reflects_action_existence() {
        let (s, _f) = fresh_store();
        assert!(!s.is_message_processed("nope").unwrap());
        s.log_action("m1", None, "a@b.com", "s", None, None, ActionStatus::DryRun)
            .unwrap();
        assert!(s.is_message_processed("m1").unwrap());
    }

    #[test]
    fn record_redraft_increments_and_persists_preset() {
        // #34: quick-refine analytics + iteration cap counter.
        let (s, _f) = fresh_store();
        let id = s
            .log_action("m1", None, "a@b.com", "s", None, Some("v0"), ActionStatus::Pending)
            .unwrap();
        assert_eq!(s.redraft_count(&id).unwrap(), 0);

        let c1 = s.record_redraft(&id, Some("shorter")).unwrap();
        assert_eq!(c1, 1);
        let c2 = s.record_redraft(&id, Some("warmer")).unwrap();
        assert_eq!(c2, 2);
        // Free-form Revise records no preset but still counts.
        let c3 = s.record_redraft(&id, None).unwrap();
        assert_eq!(c3, 3);
        assert_eq!(s.redraft_count(&id).unwrap(), 3);
    }

    #[test]
    fn redraft_count_zero_for_unknown_action() {
        let (s, _f) = fresh_store();
        assert_eq!(s.redraft_count("does-not-exist").unwrap(), 0);
    }

    // --- #473: compose envelope round-trip ---

    #[test]
    fn action_envelope_round_trips_and_defaults_to_none() {
        let (s, _f) = fresh_store();
        let id = s
            .log_action("m1", None, "josh@x.com", "intro", None, Some("v0"), ActionStatus::Pending)
            .unwrap();
        // Pre-#473 shape: nothing recorded → None, so revise falls back to from.
        assert_eq!(s.get_action_envelope(&id).unwrap(), None);

        s.set_action_envelope(&id, Some("omer@y.com"), None, Some("josh@x.com"))
            .unwrap();
        let env = s.get_action_envelope(&id).unwrap().expect("envelope set");
        assert_eq!(env.to.as_deref(), Some("omer@y.com"));
        assert_eq!(env.cc, None);
        assert_eq!(env.bcc.as_deref(), Some("josh@x.com"));

        // Empty strings are normalized to NULL, not stored as "".
        s.set_action_envelope(&id, Some("a@b.com, c@d.com"), Some(""), Some("  "))
            .unwrap();
        let env = s.get_action_envelope(&id).unwrap().expect("envelope set");
        assert_eq!(env.to.as_deref(), Some("a@b.com, c@d.com"));
        assert_eq!(env.cc, None);
        assert_eq!(env.bcc, None);

        // Unknown action id → None, not an error.
        assert_eq!(s.get_action_envelope("nope").unwrap(), None);
    }

    // --- #652: subject override survives Revise ---

    #[test]
    fn action_subject_round_trips_independently_of_the_recipients() {
        let (s, _f) = fresh_store();
        let id = s
            .log_action(
                "m1",
                None,
                "alice@example.com",
                "intro",
                None,
                Some("v0"),
                ActionStatus::Pending,
            )
            .unwrap();

        // Subject-only override: the envelope must become visible, or the
        // redraft and the reposted card never see the new subject.
        s.set_action_subject(&id, Some("Invoice for July")).unwrap();
        let env = s.get_action_envelope(&id).unwrap().expect("envelope set");
        assert_eq!(env.subject.as_deref(), Some("Invoice for July"));
        assert_eq!(env.to, None);

        // A later recipient write must not clobber the subject.
        s.set_action_envelope(&id, Some("bob@example.com"), None, None)
            .unwrap();
        let env = s.get_action_envelope(&id).unwrap().expect("envelope set");
        assert_eq!(env.subject.as_deref(), Some("Invoice for July"));
        assert_eq!(env.to.as_deref(), Some("bob@example.com"));

        // Blank normalizes to NULL, same as the recipient fields.
        s.set_action_subject(&id, Some("  ")).unwrap();
        assert_eq!(s.get_action_envelope(&id).unwrap().unwrap().subject, None);
    }

    // --- #419: duplicate guard + card-sync queries ---

    #[test]
    fn find_pending_action_for_recipient_matches_case_insensitive_and_status() {
        let (s, _f) = fresh_store();
        s.upsert_email(&sample_email("m1")).unwrap();
        let id = s
            .log_action("m1", None, "John@Example.com", "Call Follow-up", None, Some("d"), ActionStatus::Pending)
            .unwrap();
        s.set_action_draft_id(&id, "r-draft-1").unwrap();

        // Case-insensitive recipient match, exact subject, entity from the
        // joined emails row ("acc" in sample_email).
        let hit = s
            .find_pending_action_for_recipient("acc", "john@example.com", "Call Follow-up")
            .unwrap();
        assert_eq!(
            hit,
            Some((id.clone(), Some("r-draft-1".into()), "pending".into()))
        );

        // Different subject, different entity, or resolved status → no hit.
        assert!(s.find_pending_action_for_recipient("acc", "john@example.com", "Other").unwrap().is_none());
        assert!(s.find_pending_action_for_recipient("acc2", "john@example.com", "Call Follow-up").unwrap().is_none());
        s.update_action_status(&id, ActionStatus::Sent, None, None).unwrap();
        assert!(
            s.find_pending_action_for_recipient("acc", "john@example.com", "Call Follow-up").unwrap().is_none(),
            "resolved actions must not trigger the duplicate guard"
        );
    }

    #[test]
    fn card_sync_queries_find_and_update_pending_actions_by_draft() {
        let (s, _f) = fresh_store();
        s.upsert_email(&sample_email("m1")).unwrap();
        s.upsert_email(&sample_email("m2")).unwrap();
        let a1 = s
            .log_action("m1", None, "a@b.com", "s1", None, Some("old body"), ActionStatus::Pending)
            .unwrap();
        let a2 = s
            .log_action("m2", None, "a@b.com", "s2", None, Some("x"), ActionStatus::Pending)
            .unwrap();
        s.set_action_draft_id(&a1, "r-old").unwrap();
        s.set_action_draft_id(&a2, "r-other").unwrap();

        let ids = s.find_pending_action_ids_by_draft_id("r-old").unwrap();
        assert_eq!(ids, vec![a1.clone()]);

        // Repoint + refresh body — the update-draft card-sync sequence.
        s.set_action_draft_id(&a1, "r-new").unwrap();
        s.set_action_draft_body(&a1, "new body").unwrap();
        let got = s.get_action_with_email(&a1).unwrap().unwrap();
        assert_eq!(got.draft_id.as_deref(), Some("r-new"));
        assert_eq!(got.action.draft_body.as_deref(), Some("new body"));
        assert_eq!(got.action.status, "pending", "body refresh must not touch status");

        // A resolved action no longer follows draft-id lookups.
        s.update_action_status(&a2, ActionStatus::Rejected, None, None).unwrap();
        assert!(s.find_pending_action_ids_by_draft_id("r-other").unwrap().is_empty());
    }

    // --- nudge loop ---

    #[test]
    fn pending_action_seeds_nudge_schedule() {
        let (s, _f) = fresh_store();
        s.upsert_email(&sample_email("m1")).unwrap();
        let id = s
            .log_action("m1", None, "a@b.com", "s", None, Some("draft"), ActionStatus::Pending)
            .unwrap();
        // Next nudge is roughly 6h out — query directly to verify.
        let conn = Connection::open(_f.path().join("store-test.db")).unwrap();
        let next: Option<i64> = conn
            .query_row(
                "SELECT nextNudgeAtMs FROM actions WHERE id = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert!(next.is_some(), "pending action must have a nudge timer");
        let count: i64 = conn
            .query_row(
                "SELECT nudgeCount FROM actions WHERE id = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn non_pending_action_skips_nudge_schedule() {
        let (s, _f) = fresh_store();
        s.upsert_email(&sample_email("m1")).unwrap();
        let id = s
            .log_action("m1", None, "a@b.com", "s", None, None, ActionStatus::DryRun)
            .unwrap();
        let conn = Connection::open(_f.path().join("store-test.db")).unwrap();
        let next: Option<i64> = conn
            .query_row(
                "SELECT nextNudgeAtMs FROM actions WHERE id = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert!(next.is_none(), "dry-run actions should never be nudged");
    }

    #[test]
    fn find_next_to_promote_returns_oldest_unpromoted_immediately() {
        let (s, _f) = fresh_store();
        s.upsert_email(&sample_email("m1")).unwrap();
        s.upsert_email(&sample_email("m2")).unwrap();
        let id1 = s
            .log_action("m1", None, "a@b.com", "s1", None, Some("d1"), ActionStatus::Pending)
            .unwrap();
        // Slight delay so m2's createdAt > m1's.
        std::thread::sleep(std::time::Duration::from_millis(2));
        let _id2 = s
            .log_action("m2", None, "a@b.com", "s2", None, Some("d2"), ActionStatus::Pending)
            .unwrap();
        // Initial promotion is no longer gated by the 6h timer — fresh
        // backlog rows are eligible immediately. Oldest createdAt wins.
        let nxt = s
            .find_next_to_promote(now_millis())
            .unwrap()
            .expect("expected next");
        assert_eq!(nxt.action.action.id, id1, "oldest createdAt wins");
        assert_eq!(nxt.nudge_count, 0);
    }

    #[test]
    fn find_next_to_promote_skips_promoted_rows() {
        let (s, _f) = fresh_store();
        s.upsert_email(&sample_email("m1")).unwrap();
        s.upsert_email(&sample_email("m2")).unwrap();
        let id1 = s
            .log_action("m1", None, "a@b.com", "s1", None, Some("d1"), ActionStatus::Pending)
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let id2 = s
            .log_action("m2", None, "a@b.com", "s2", None, Some("d2"), ActionStatus::Pending)
            .unwrap();
        // m1 is the active card (nudgeCount=1); next promotion should pick m2.
        s.record_nudge(&id1, now_millis() + NUDGE_INTERVAL_MS).unwrap();
        let nxt = s
            .find_next_to_promote(now_millis())
            .unwrap()
            .expect("expected next");
        assert_eq!(nxt.action.action.id, id2);
    }

    #[test]
    fn find_active_nudge_returns_promoted_pending() {
        let (s, _f) = fresh_store();
        s.upsert_email(&sample_email("m1")).unwrap();
        let id = s
            .log_action("m1", None, "a@b.com", "s", None, Some("d"), ActionStatus::Pending)
            .unwrap();
        assert!(s.find_active_nudge().unwrap().is_none());
        s.record_nudge(&id, now_millis() + NUDGE_INTERVAL_MS).unwrap();
        let active = s.find_active_nudge().unwrap().expect("expected active card");
        assert_eq!(active.action.action.id, id);
        assert_eq!(active.nudge_count, 1);
        assert!(active.next_nudge_at_ms.is_some());
    }

    #[test]
    fn find_active_nudge_excludes_terminal_status() {
        let (s, _f) = fresh_store();
        s.upsert_email(&sample_email("m1")).unwrap();
        let id = s
            .log_action("m1", None, "a@b.com", "s", None, Some("d"), ActionStatus::Pending)
            .unwrap();
        s.record_nudge(&id, now_millis() + NUDGE_INTERVAL_MS).unwrap();
        s.update_action_status(&id, ActionStatus::Approved, None, None)
            .unwrap();
        assert!(
            s.find_active_nudge().unwrap().is_none(),
            "approved card should no longer be active"
        );
    }

    #[test]
    fn count_pending_overdue_counts_all_due() {
        let (s, _f) = fresh_store();
        s.upsert_email(&sample_email("m1")).unwrap();
        s.upsert_email(&sample_email("m2")).unwrap();
        s.log_action("m1", None, "a@b.com", "s", None, None, ActionStatus::Pending)
            .unwrap();
        let id2 = s
            .log_action("m2", None, "a@b.com", "s", None, None, ActionStatus::Pending)
            .unwrap();
        // Promote m2 so it's active. Both should still count as overdue when
        // we query past the timer.
        s.record_nudge(&id2, now_millis() + NUDGE_INTERVAL_MS).unwrap();
        let future = now_millis() + 2 * NUDGE_INTERVAL_MS;
        assert_eq!(s.count_pending_overdue(future).unwrap(), 2);
    }

    #[test]
    fn record_nudge_bumps_count_and_pushes_next() {
        let (s, _f) = fresh_store();
        s.upsert_email(&sample_email("m1")).unwrap();
        let id = s
            .log_action("m1", None, "a@b.com", "s", None, Some("d"), ActionStatus::Pending)
            .unwrap();
        let now = now_millis();
        s.record_nudge(&id, now + NUDGE_INTERVAL_MS).unwrap();
        s.record_nudge(&id, now + 2 * NUDGE_INTERVAL_MS).unwrap();
        let conn = Connection::open(_f.path().join("store-test.db")).unwrap();
        let (count, next): (i64, i64) = conn
            .query_row(
                "SELECT nudgeCount, nextNudgeAtMs FROM actions WHERE id = ?1",
                params![id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(count, 2);
        assert_eq!(next, now + 2 * NUDGE_INTERVAL_MS);
    }

    #[test]
    fn reset_nudge_schedule_defers_without_zeroing_count() {
        let (s, _f) = fresh_store();
        s.upsert_email(&sample_email("m1")).unwrap();
        let id = s
            .log_action("m1", None, "a@b.com", "s", None, Some("d"), ActionStatus::Pending)
            .unwrap();
        // Promote, then re-nudge once → count = 2.
        s.record_nudge(&id, now_millis()).unwrap();
        s.record_nudge(&id, now_millis()).unwrap();
        // Revise: defer timer 6h, KEEP count (card stays the active one).
        s.reset_nudge_schedule(&id).unwrap();
        let conn = Connection::open(_f.path().join("store-test.db")).unwrap();
        let (count, next): (i64, i64) = conn
            .query_row(
                "SELECT nudgeCount, nextNudgeAtMs FROM actions WHERE id = ?1",
                params![id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(count, 2, "revise must not zero nudgeCount");
        let now = now_millis();
        assert!(next >= now + NUDGE_INTERVAL_MS - 5_000);
        assert!(next <= now + NUDGE_INTERVAL_MS + 5_000);
    }

    // --- channel_subscriptions ---

    #[test]
    fn upsert_subscription_creates_then_updates() {
        let (s, _f) = fresh_store();
        let sub = s
            .upsert_subscription(
                "discord",
                "ch1",
                "DM with alice",
                SubscriptionMode::Priority,
                None,
            )
            .unwrap();
        assert_eq!(sub.platform, "discord");
        assert_eq!(sub.channel_id, "ch1");
        assert_eq!(sub.mode, SubscriptionMode::Priority);
        assert!(sub.active);
        assert!(sub.last_seen_message_id.is_none());

        let updated = s
            .upsert_subscription(
                "discord",
                "ch1",
                "DM with alice (renamed)",
                SubscriptionMode::Digest,
                None,
            )
            .unwrap();
        assert_eq!(updated.id, sub.id, "same (platform, channel_id) re-upserts in place");
        assert_eq!(updated.display_name, "DM with alice (renamed)");
        assert_eq!(updated.mode, SubscriptionMode::Digest);
    }

    #[test]
    fn list_active_subscriptions_filters_by_platform_and_active() {
        let (s, _f) = fresh_store();
        let d1 = s
            .upsert_subscription("discord", "d1", "d1", SubscriptionMode::Priority, None)
            .unwrap();
        s.upsert_subscription("discord", "d2", "d2", SubscriptionMode::Digest, None)
            .unwrap();
        s.upsert_subscription("slack", "s1", "s1", SubscriptionMode::StoreOnly, Some("T1"))
            .unwrap();

        let discord_subs = s.list_active_subscriptions("discord").unwrap();
        assert_eq!(discord_subs.len(), 2);
        assert!(discord_subs.iter().all(|x| x.platform == "discord"));

        s.delete_subscription(&d1.id).unwrap();
        let after_delete = s.list_active_subscriptions("discord").unwrap();
        assert_eq!(after_delete.len(), 1, "soft-deleted subs excluded");
        assert_eq!(after_delete[0].channel_id, "d2");
    }

    #[test]
    fn update_subscription_mode_persists() {
        let (s, _f) = fresh_store();
        let sub = s
            .upsert_subscription("discord", "ch1", "dm", SubscriptionMode::Priority, None)
            .unwrap();
        s.update_subscription_mode(&sub.id, SubscriptionMode::StoreOnly)
            .unwrap();
        let reloaded = s.get_subscription(&sub.id).unwrap().unwrap();
        assert_eq!(reloaded.mode, SubscriptionMode::StoreOnly);
    }

    #[test]
    fn update_last_seen_message_persists() {
        let (s, _f) = fresh_store();
        let sub = s
            .upsert_subscription("discord", "ch1", "dm", SubscriptionMode::Priority, None)
            .unwrap();
        s.update_last_seen_message(&sub.id, "1234567890").unwrap();
        let reloaded = s.get_subscription(&sub.id).unwrap().unwrap();
        assert_eq!(reloaded.last_seen_message_id.as_deref(), Some("1234567890"));
    }

    #[test]
    fn mark_digest_posted_persists() {
        let (s, _f) = fresh_store();
        let sub = s
            .upsert_subscription("discord", "ch1", "dm", SubscriptionMode::Digest, None)
            .unwrap();
        s.mark_digest_posted(&sub.id, 1776806000000).unwrap();
        let reloaded = s.get_subscription(&sub.id).unwrap().unwrap();
        assert_eq!(reloaded.last_digest_at_ms, Some(1776806000000));
    }

    #[test]
    fn delete_then_reupsert_restores_same_row() {
        // Soft-delete preserves the (platform, channel_id, account_id) unique
        // triple; a subsequent upsert should flip active back to 1 and
        // overwrite fields.
        let (s, _f) = fresh_store();
        let sub = s
            .upsert_subscription("discord", "ch1", "first", SubscriptionMode::Priority, None)
            .unwrap();
        s.delete_subscription(&sub.id).unwrap();
        let restored = s
            .upsert_subscription("discord", "ch1", "second", SubscriptionMode::Digest, None)
            .unwrap();
        assert_eq!(restored.id, sub.id);
        assert_eq!(restored.display_name, "second");
        assert_eq!(restored.mode, SubscriptionMode::Digest);
        assert!(restored.active);
    }

    #[test]
    fn same_channel_distinct_workspaces_coexist() {
        let (s, _f) = fresh_store();
        let a = s
            .upsert_subscription("slack", "C_SAME", "general@A", SubscriptionMode::Priority, Some("T_A"))
            .unwrap();
        let b = s
            .upsert_subscription("slack", "C_SAME", "general@B", SubscriptionMode::Priority, Some("T_B"))
            .unwrap();
        assert_ne!(a.id, b.id, "same channel_id across workspaces yields distinct rows");
        let list = s.list_active_subscriptions("slack").unwrap();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn slack_workspace_upsert_is_idempotent() {
        let (s, _f) = fresh_store();
        let w1 = s
            .upsert_slack_workspace("T1", "Team1", "e1", "c1", "U1")
            .unwrap();
        let w2 = s
            .upsert_slack_workspace("T1", "Team1 renamed", "e1", "c1", "U1")
            .unwrap();
        assert_eq!(w1.id, w2.id);
        assert_eq!(w2.team_name, "Team1 renamed");
        let list = s.list_active_slack_workspaces().unwrap();
        assert_eq!(list.len(), 1);
    }

    // --- draft_revisions (#37) ---

    #[test]
    fn record_revision_triple_writes_two_rows_first_time() {
        let (s, _f) = fresh_store();
        s.upsert_email(&sample_email("m1")).unwrap();
        let action_id = s
            .log_action("m1", None, "a@b.com", "s", None, Some("v0"), ActionStatus::Pending)
            .unwrap();
        let revised_id = s
            .record_revision_triple(&action_id, "v0", "less formal please", "v1")
            .unwrap();
        let rows = s.list_revisions_for_action(&action_id).unwrap();
        assert_eq!(rows.len(), 2, "first revise writes original + revised rows");
        assert_eq!(rows[0].iteration, 0);
        assert_eq!(rows[0].draft_body, "v0");
        assert_eq!(rows[0].feedback_text, None);
        assert_eq!(rows[0].outcome, "superseded");
        assert_eq!(rows[1].iteration, 1);
        assert_eq!(rows[1].draft_body, "v1");
        assert_eq!(rows[1].feedback_text.as_deref(), Some("less formal please"));
        assert_eq!(rows[1].outcome, "pending");
        assert_eq!(rows[1].id, revised_id);
    }

    #[test]
    fn record_revision_triple_chains_subsequent_revises() {
        let (s, _f) = fresh_store();
        s.upsert_email(&sample_email("m1")).unwrap();
        let action_id = s
            .log_action("m1", None, "a@b.com", "s", None, Some("v0"), ActionStatus::Pending)
            .unwrap();
        s.record_revision_triple(&action_id, "v0", "less formal", "v1")
            .unwrap();
        s.record_revision_triple(&action_id, "v1", "shorter", "v2")
            .unwrap();
        let rows = s.list_revisions_for_action(&action_id).unwrap();
        assert_eq!(rows.len(), 3, "second revise appends one row, supersedes prior pending");
        assert_eq!(rows[0].outcome, "superseded");
        assert_eq!(rows[1].outcome, "superseded", "prior pending flips to superseded");
        assert_eq!(rows[1].draft_body, "v1");
        assert_eq!(rows[2].outcome, "pending");
        assert_eq!(rows[2].draft_body, "v2");
        assert_eq!(rows[2].feedback_text.as_deref(), Some("shorter"));
        // Iterations stay contiguous + UNIQUE.
        assert_eq!(rows[0].iteration, 0);
        assert_eq!(rows[1].iteration, 1);
        assert_eq!(rows[2].iteration, 2);
    }

    #[test]
    fn list_recent_feedback_filters_by_age_and_skips_iteration_zero() {
        let (s, _f) = fresh_store();
        s.upsert_email(&sample_email("m1")).unwrap();
        let action_id = s
            .log_action("m1", None, "a@b.com", "s", None, Some("v0"), ActionStatus::Pending)
            .unwrap();
        s.record_revision_triple(&action_id, "v0", "less formal", "v1")
            .unwrap();
        let recent = s.list_recent_feedback(60_000).unwrap();
        assert_eq!(recent.len(), 1, "only the iteration-1 row has feedback text");
        assert_eq!(recent[0].feedback_text.as_deref(), Some("less formal"));
        // A zero-window query excludes everything.
        let none = s.list_recent_feedback(0).unwrap();
        assert!(none.is_empty());
    }

    #[test]
    fn telegram_bot_upsert_preserves_last_update_id_on_relogin() {
        let (s, _f) = fresh_store();
        let bot = s
            .upsert_telegram_bot(99, "triage_bot", 5555)
            .unwrap();
        assert_eq!(bot.last_update_id, 0);
        s.update_telegram_bot_last_update_id(99, 4242).unwrap();
        // Re-login should not reset the cursor.
        let bot2 = s
            .upsert_telegram_bot(99, "triage_bot_renamed", 5555)
            .unwrap();
        assert_eq!(bot2.last_update_id, 4242);
        assert_eq!(bot2.bot_username, "triage_bot_renamed");
    }

    #[test]
    fn telegram_bot_update_last_update_id_is_monotonic() {
        let (s, _f) = fresh_store();
        s.upsert_telegram_bot(7, "b", 1).unwrap();
        s.update_telegram_bot_last_update_id(7, 100).unwrap();
        // A stale write (lower id) must not move the cursor backward.
        s.update_telegram_bot_last_update_id(7, 50).unwrap();
        let bot = s.get_telegram_bot_by_id(7).unwrap().unwrap();
        assert_eq!(bot.last_update_id, 100);
    }

    #[test]
    fn telegram_bot_delete_deactivates_subscriptions() {
        let (s, _f) = fresh_store();
        s.upsert_telegram_bot(7, "b", 1).unwrap();
        s.upsert_subscription("telegram", "12345", "alice", SubscriptionMode::Priority, Some("7"))
            .unwrap();
        s.delete_telegram_bot(7).unwrap();
        let subs = s.list_active_subscriptions("telegram").unwrap();
        assert!(subs.is_empty());
        assert!(s.get_telegram_bot_by_id(7).unwrap().is_none());
    }

    #[test]
    fn subscription_mode_parse_round_trip() {
        for m in [
            SubscriptionMode::Priority,
            SubscriptionMode::Digest,
            SubscriptionMode::StoreOnly,
        ] {
            assert_eq!(SubscriptionMode::parse(m.as_str()), Some(m));
        }
        assert_eq!(SubscriptionMode::parse("garbage"), None);
    }

    // --- tone profiles & examples (issue #73) ---

    #[test]
    fn insert_and_query_tone_example() {
        let (s, _f) = fresh_store();
        let id = s
            .insert_tone_example(
                "sent_backfill",
                None,
                Some("gmail-msg-1"),
                "acc1",
                "jeremy@acme.com",
                "acme.com",
                Some("Re: stuff"),
                "Hey — quick reply.",
                1_700_000_000_000,
                1.0,
            )
            .unwrap();
        assert!(!id.is_empty(), "insert returns a non-empty uuid");
        let recents = s
            .recent_tone_examples("recipient", "jeremy@acme.com", Some("acc1"), 10)
            .unwrap();
        assert_eq!(recents.len(), 1);
        assert_eq!(recents[0].source, "sent_backfill");
        assert_eq!(recents[0].body, "Hey — quick reply.");
        // body_chars is character count, not byte count.
        assert_eq!(recents[0].body_chars, "Hey — quick reply.".chars().count() as i64);
    }

    #[test]
    fn get_tone_profile_returns_none_for_missing() {
        let (s, _f) = fresh_store();
        let p = s.get_tone_profile("global", "*", Some("acc1")).unwrap();
        assert!(p.is_none());
    }

    #[test]
    fn upsert_tone_profile_inserts_then_updates_in_place() {
        let (s, _f) = fresh_store();
        let p1 = s
            .upsert_tone_profile(
                "global",
                "*",
                Some("acc1"),
                "{\"register\":\"casual\"}",
                "[]",
                10,
            )
            .unwrap();
        assert_eq!(p1.scope_kind, "global");
        assert_eq!(p1.sample_count, 10);
        assert_eq!(p1.sample_count_at_refresh, 10);

        // Re-upsert with new summary and higher sample_count: SAME id, fields refresh.
        let p2 = s
            .upsert_tone_profile(
                "global",
                "*",
                Some("acc1"),
                "{\"register\":\"professional\"}",
                "[\"x\"]",
                25,
            )
            .unwrap();
        assert_eq!(p1.id, p2.id, "upsert is in-place keyed by (scope,scope_value,acct)");
        assert_eq!(p2.summary, "{\"register\":\"professional\"}");
        assert_eq!(p2.sample_count, 25);
        // Snapshot is reset to current sample_count on refresh — caller can
        // then compare future inserts against this for staleness.
        assert_eq!(p2.sample_count_at_refresh, 25);
    }

    #[test]
    fn upsert_tone_profile_keys_account_distinctly() {
        let (s, _f) = fresh_store();
        let p_a = s
            .upsert_tone_profile("global", "*", Some("acc-A"), "A", "[]", 1)
            .unwrap();
        let p_b = s
            .upsert_tone_profile("global", "*", Some("acc-B"), "B", "[]", 1)
            .unwrap();
        assert_ne!(p_a.id, p_b.id, "different accounts → different rows");
        let all = s.list_tone_profiles().unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn recent_tone_examples_orders_newest_first_and_limits() {
        let (s, _f) = fresh_store();
        for (i, ts) in [1_000_i64, 3_000, 2_000].iter().enumerate() {
            s.insert_tone_example(
                "sent_backfill",
                None,
                Some(&format!("m{i}")),
                "acc1",
                "x@y.com",
                "y.com",
                None,
                "body",
                *ts,
                1.0,
            )
            .unwrap();
        }
        let recents = s
            .recent_tone_examples("recipient", "x@y.com", Some("acc1"), 2)
            .unwrap();
        assert_eq!(recents.len(), 2);
        assert_eq!(recents[0].sent_at_ms, 3000);
        assert_eq!(recents[1].sent_at_ms, 2000);
    }

    #[test]
    fn count_tone_examples_per_scope() {
        let (s, _f) = fresh_store();
        for to in ["a@acme.com", "b@acme.com", "c@other.com"] {
            let domain = to.split('@').nth(1).unwrap();
            s.insert_tone_example(
                "sent_backfill",
                None,
                None,
                "acc1",
                to,
                domain,
                None,
                "body body body",
                1_000,
                1.0,
            )
            .unwrap();
        }
        assert_eq!(
            s.count_tone_examples("recipient", "a@acme.com", Some("acc1")).unwrap(),
            1
        );
        assert_eq!(
            s.count_tone_examples("domain", "acme.com", Some("acc1")).unwrap(),
            2
        );
        assert_eq!(
            s.count_tone_examples("global", "*", Some("acc1")).unwrap(),
            3
        );
    }

    #[test]
    fn record_user_edit_as_tone_example_captures_sent_draft() {
        let (s, _f) = fresh_store();
        // Seed an email + an action that's been transitioned to Sent.
        let mut email = sample_email("m-edit");
        email.from = "Alex <alex@startup.io>".into();
        email.subject = "Re: launch".into();
        s.upsert_email(&email).unwrap();
        let action_id = s
            .log_action(
                "m-edit",
                None,
                "Alex <alex@startup.io>",
                "Re: launch",
                Some("inbound body"),
                Some("This is the post-edit draft the user actually sent."),
                ActionStatus::Pending,
            )
            .unwrap();
        s.update_action_status(&action_id, ActionStatus::Sent, None, None)
            .unwrap();

        let new_id = s.record_user_edit_as_tone_example(&action_id).unwrap();
        assert!(new_id.is_some(), "expected a tone example to be recorded");
        let recents = s
            .recent_tone_examples("recipient", "alex@startup.io", Some("acc"), 10)
            .unwrap();
        assert_eq!(recents.len(), 1);
        assert_eq!(recents[0].source, "user_edit");
        assert!((recents[0].weight - 1.5).abs() < f64::EPSILON);
        assert_eq!(recents[0].recipient_domain, "startup.io");
        assert_eq!(
            recents[0].body,
            "This is the post-edit draft the user actually sent."
        );
    }

    #[test]
    fn record_user_edit_skips_non_sent_or_empty_draft() {
        let (s, _f) = fresh_store();
        let email = sample_email("m-pending");
        s.upsert_email(&email).unwrap();
        let id = s
            .log_action(
                "m-pending",
                None,
                "a@b.com",
                "subj",
                None,
                Some("draft"),
                ActionStatus::Pending,
            )
            .unwrap();
        // Status is Pending, not Sent → no-op.
        assert!(s.record_user_edit_as_tone_example(&id).unwrap().is_none());
        // Now flip to Sent but with no draftBody on a separate action.
        let id2 = s
            .log_action(
                "m-pending",
                None,
                "a@b.com",
                "subj",
                None,
                None,
                ActionStatus::Pending,
            )
            .unwrap();
        s.update_action_status(&id2, ActionStatus::Sent, None, None)
            .unwrap();
        assert!(
            s.record_user_edit_as_tone_example(&id2).unwrap().is_none(),
            "missing draftBody → skip"
        );
    }

    #[test]
    fn linkedin_connection_sync_roundtrips() {
        let (s, _f) = fresh_store();
        assert!(s
            .get_linkedin_connection_sync("urn:li:fsd_profile:ME")
            .unwrap()
            .is_none());
        let cur = LinkedInConnectionSync {
            account_id: "urn:li:fsd_profile:ME".into(),
            last_full_sync_ms: Some(1_700_000_000_000),
            last_delta_sync_ms: None,
            cursor_start: 80,
            last_synced_count: 562,
        };
        s.upsert_linkedin_connection_sync(&cur).unwrap();
        let got = s
            .get_linkedin_connection_sync("urn:li:fsd_profile:ME")
            .unwrap()
            .unwrap();
        assert_eq!(got.cursor_start, 80);
        assert_eq!(got.last_full_sync_ms, Some(1_700_000_000_000));
        assert_eq!(got.last_synced_count, 562);
        // Upsert overwrites mutable columns.
        let cur2 = LinkedInConnectionSync {
            cursor_start: 0,
            last_delta_sync_ms: Some(1_700_100_000_000),
            ..cur
        };
        s.upsert_linkedin_connection_sync(&cur2).unwrap();
        let got2 = s
            .get_linkedin_connection_sync("urn:li:fsd_profile:ME")
            .unwrap()
            .unwrap();
        assert_eq!(got2.cursor_start, 0);
        assert_eq!(got2.last_delta_sync_ms, Some(1_700_100_000_000));
    }

    #[test]
    fn contacts_sync_token_roundtrips() {
        let (s, _f) = fresh_store();
        assert!(s
            .get_contacts_sync_token("google_people", "acc1")
            .unwrap()
            .is_none());
        s.set_contacts_sync_token("google_people", "acc1", "tok-abc")
            .unwrap();
        assert_eq!(
            s.get_contacts_sync_token("google_people", "acc1").unwrap(),
            Some("tok-abc".to_string())
        );
        // Distinct (backend, account) key is independent.
        assert!(s
            .get_contacts_sync_token("carddav", "acc1")
            .unwrap()
            .is_none());
        s.set_contacts_sync_token("google_people", "acc1", "tok-def")
            .unwrap();
        assert_eq!(
            s.get_contacts_sync_token("google_people", "acc1").unwrap(),
            Some("tok-def".to_string())
        );
    }

    #[test]
    fn phone_identity_reverse_lookup() {
        let (s, _f) = fresh_store();
        assert!(s.lookup_person_by_phone("+14155550100").unwrap().is_none());
        s.upsert_phone_identity(&PhoneIdentity {
            phone: "+14155550100".into(),
            person_slug: "jane_doe".into(),
            display_name: Some("Jane Doe".into()),
            source: "google_people".into(),
        })
        .unwrap();
        let p = s.lookup_person_by_phone("+14155550100").unwrap().unwrap();
        assert_eq!(p.person_slug, "jane_doe");
        assert_eq!(p.display_name.as_deref(), Some("Jane Doe"));
        // Re-ingest same phone → upsert, not duplicate; slug can move.
        s.upsert_phone_identity(&PhoneIdentity {
            phone: "+14155550100".into(),
            person_slug: "jane_d".into(),
            display_name: Some("Jane D.".into()),
            source: "carddav".into(),
        })
        .unwrap();
        let p2 = s.lookup_person_by_phone("+14155550100").unwrap().unwrap();
        assert_eq!(p2.person_slug, "jane_d");
        assert_eq!(p2.source, "carddav");
    }

    // --- user loops (#104) ---

    #[test]
    fn user_loop_create_list_stop_roundtrip() {
        let (s, _f) = fresh_store();
        let id = s
            .create_user_loop("u1", "discord", "chan-7", 300, "/digest", None, None, None)
            .unwrap();
        let loops = s.list_user_loops("u1").unwrap();
        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].id, id);
        assert_eq!(loops[0].interval_secs, 300);
        assert_eq!(loops[0].status, "active");
        assert!(loops[0].expires_at_ms.is_none());
        assert_eq!(s.count_active_user_loops("u1").unwrap(), 1);

        // scoped: another user can't stop it
        assert!(!s.stop_user_loop("u2", &id).unwrap());
        assert!(s.stop_user_loop("u1", &id).unwrap());
        // stopped rows drop out of the owner listing
        assert!(s.list_user_loops("u1").unwrap().is_empty());
        assert_eq!(s.count_active_user_loops("u1").unwrap(), 0);
    }

    #[test]
    fn user_loop_pauses_after_repeated_failures() {
        let (s, _f) = fresh_store();
        let id = s
            .create_user_loop("u1", "discord", "c", 300, "/x", None, None, None)
            .unwrap();
        s.record_user_loop_run(&id, false, "boom", 3).unwrap();
        s.record_user_loop_run(&id, false, "boom", 3).unwrap();
        // still active after 2 failures
        assert_eq!(s.list_active_user_loops().unwrap().len(), 1);
        s.record_user_loop_run(&id, false, "boom", 3).unwrap();
        // 3rd failure auto-pauses
        assert!(s.list_active_user_loops().unwrap().is_empty());
        let l = &s.list_user_loops("u1").unwrap()[0];
        assert_eq!(l.status, "paused");
        assert_eq!(l.fail_count, 3);
    }

    #[test]
    fn user_loop_success_resets_fail_count() {
        let (s, _f) = fresh_store();
        let id = s
            .create_user_loop("u1", "discord", "c", 300, "/x", None, None, None)
            .unwrap();
        s.record_user_loop_run(&id, false, "boom", 5).unwrap();
        s.record_user_loop_run(&id, true, "ok", 5).unwrap();
        let l = &s.list_user_loops("u1").unwrap()[0];
        assert_eq!(l.fail_count, 0);
        assert_eq!(l.last_status.as_deref(), Some("ok"));
        assert!(l.last_run_ms.is_some());
    }

    #[test]
    fn user_loop_expires_at_stops_at_deadline() {
        let (s, _f) = fresh_store();
        // Past deadline → should be swept on the next stop_expired_user_loops().
        let past = 1_000_i64;
        let id_expired = s
            .create_user_loop("u1", "discord", "chan-a", 300, "ping", Some(past), None, None)
            .unwrap();
        // Future deadline → should be left alone.
        let id_live = s
            .create_user_loop("u1", "discord", "chan-b", 300, "pong", Some(i64::MAX), None, None)
            .unwrap();
        // No deadline → also left alone.
        let id_forever = s
            .create_user_loop("u1", "discord", "chan-c", 300, "forever", None, None, None)
            .unwrap();

        let stopped = s.stop_expired_user_loops(2_000).unwrap();
        let ids: Vec<&str> = stopped.iter().map(|(i, _, _)| i.as_str()).collect();
        assert_eq!(ids, vec![id_expired.as_str()]);
        // Returned tuple carries the surface info needed to post the notice.
        assert_eq!(stopped[0].1, "discord");
        assert_eq!(stopped[0].2, "chan-a");

        // Row is now `stopped` with last_status='expired'.
        let all = s.list_user_loops("u1").unwrap();
        // listing drops stopped rows, so only two remain
        assert_eq!(all.len(), 2);
        assert!(all.iter().any(|l| l.id == id_live));
        assert!(all.iter().any(|l| l.id == id_forever));

        // Second sweep is a no-op (idempotent).
        assert!(s.stop_expired_user_loops(2_000).unwrap().is_empty());
    }

    // --- cross-surface CAS resolve (#47) ---

    #[test]
    fn pwa_subscription_upsert_list_remove() {
        let (s, _f) = fresh_store();
        s.add_pwa_subscription("https://push/ep1", "p1", "a1").unwrap();
        s.add_pwa_subscription("https://push/ep1", "p2", "a2").unwrap(); // upsert
        s.add_pwa_subscription("https://push/ep2", "p3", "a3").unwrap();
        let subs = s.list_pwa_subscriptions().unwrap();
        assert_eq!(subs.len(), 2);
        let ep1 = subs.iter().find(|(e, _, _)| e == "https://push/ep1").unwrap();
        assert_eq!(ep1.1, "p2", "upsert replaced keys");
        s.remove_pwa_subscription("https://push/ep1").unwrap();
        assert_eq!(s.list_pwa_subscriptions().unwrap().len(), 1);
    }

    #[test]
    fn try_resolve_action_is_compare_and_swap() {
        let (s, _f) = fresh_store();
        let id = s
            .log_action(
                "m-cas",
                None,
                "a@b.com",
                "subj",
                None,
                None,
                ActionStatus::Pending,
            )
            .unwrap();
        // First resolver wins, and its reason lands in errorMessage.
        assert!(s
            .try_resolve_action(&id, ActionStatus::Sent, "discord", Some("done"))
            .unwrap());
        // Second resolver (racing surface) loses — no double side effect.
        assert!(!s
            .try_resolve_action(&id, ActionStatus::Skipped, "dashboard", None)
            .unwrap());
        assert_eq!(
            s.action_status_source(&id).unwrap().as_deref(),
            Some("discord")
        );
        let guard = s.conn.lock().unwrap();
        let reason: Option<String> = guard
            .query_row(
                "SELECT errorMessage FROM actions WHERE id = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(reason.as_deref(), Some("done"));
    }

    // ---- #500: scheduled email send — CAS state machine + engine queries ----

    fn raw_action_row(s: &Store, id: &str) -> (String, Option<i64>, Option<i64>) {
        let guard = s.conn.lock().unwrap();
        guard
            .query_row(
                "SELECT status, scheduledAtMs, retryCount FROM actions WHERE id = ?1",
                params![id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap()
    }

    fn pending_action(s: &Store, message_id: &str) -> String {
        s.log_action(
            message_id,
            Some("thread-500"),
            "peer@example.com",
            "subj",
            None,
            Some("draft body"),
            ActionStatus::Pending,
        )
        .unwrap()
    }

    #[test]
    fn schedule_claim_finish_happy_path_is_cas_gated() {
        let (s, _f) = fresh_store();
        let id = pending_action(&s, "m-sched-1");

        assert!(s.schedule_action(&id, 1_000_000, "discord").unwrap());
        // Second schedule (double pick) loses.
        assert!(!s.schedule_action(&id, 2_000_000, "discord").unwrap());
        let (status, at, _) = raw_action_row(&s, &id);
        assert_eq!(status, "scheduled");
        assert_eq!(at, Some(1_000_000), "loser must not overwrite the fire time");

        // Engine claims from 'scheduled'; a second claim loses.
        assert!(s
            .claim_action_for_send(&id, ActionStatus::Scheduled, "engine")
            .unwrap());
        assert!(!s
            .claim_action_for_send(&id, ActionStatus::Scheduled, "engine")
            .unwrap());
        // Approve-style claim from 'pending' also loses now.
        assert!(!s
            .claim_action_for_send(&id, ActionStatus::Pending, "discord")
            .unwrap());

        assert!(s.finish_send_sent(&id, "engine").unwrap());
        // Terminal flip is one-shot.
        assert!(!s.finish_send_sent(&id, "engine").unwrap());
        let (status, at, _) = raw_action_row(&s, &id);
        assert_eq!(status, "sent");
        assert_eq!(at, Some(1_000_000), "sent rows keep scheduledAtMs as audit");
    }

    #[test]
    fn unschedule_clears_proposal_and_reseeds_nudge_queue() {
        let (s, _f) = fresh_store();
        let id = pending_action(&s, "m-sched-2");
        s.schedule_action(&id, 1_000_000, "discord").unwrap();
        s.set_action_notice(&id, "chan-1", "msg-1").unwrap();

        assert!(s.unschedule_action(&id, "discord").unwrap());
        let (status, at, _) = raw_action_row(&s, &id);
        assert_eq!(status, "pending");
        assert_eq!(
            at, None,
            "back-to-queue MUST clear the proposal or a later Approve re-arms it"
        );
        assert_eq!(s.action_notice(&id).unwrap(), None);
        // Row re-enters the queue as the ACTIVE card (caller reposted it) —
        // nudgeCount 0 would let the NudgeScheduler race a second card in.
        let guard = s.conn.lock().unwrap();
        let (nudge_count, next_at): (i64, Option<i64>) = guard
            .query_row(
                "SELECT COALESCE(nudgeCount,0), nextNudgeAtMs FROM actions WHERE id = ?1",
                params![id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(nudge_count, 1);
        assert!(next_at.is_some());
        drop(guard);
        // Unschedule on a non-scheduled row is a no-op.
        assert!(!s.unschedule_action(&id, "discord").unwrap());
    }

    #[test]
    fn claim_due_rejects_rearmed_future_schedule() {
        let (s, _f) = fresh_store();
        let id = pending_action(&s, "m-aba-1");
        let now = now_millis();
        s.schedule_action(&id, now - 1_000, "t").unwrap();
        // ABA: back-to-queue + re-arm for tomorrow while the engine's due
        // snapshot still lists the row.
        assert!(s.unschedule_action(&id, "t").unwrap());
        assert!(s.schedule_action(&id, now + 86_400_000, "t").unwrap());
        assert!(
            !s.claim_due_action_for_send(&id, now, "engine").unwrap(),
            "a re-armed future schedule must not be claimable from a stale \
             due snapshot"
        );
        // Still armed for tomorrow; and once due, the claim works.
        assert!(s
            .claim_due_action_for_send(&id, now + 86_500_000, "engine")
            .unwrap());
    }

    #[test]
    fn cancel_scheduled_action_is_terminal_and_clears_pointers() {
        let (s, _f) = fresh_store();
        let id = pending_action(&s, "m-sched-3");
        s.schedule_action(&id, 1_000_000, "discord").unwrap();
        s.set_action_notice(&id, "chan-1", "msg-1").unwrap();

        assert!(s.cancel_scheduled_action(&id, "", "discord").unwrap());
        let (status, at, _) = raw_action_row(&s, &id);
        assert_eq!(status, "rejected");
        assert_eq!(at, None);
        assert_eq!(s.action_notice(&id).unwrap(), None);
        assert!(!s.cancel_scheduled_action(&id, "", "discord").unwrap());
    }

    #[test]
    fn finish_send_error_override_exempts_row_from_retry_queue() {
        let (s, _f) = fresh_store();
        let id = pending_action(&s, "m-sched-4");
        let email = sample_email("m-sched-4");
        s.upsert_email(&email).unwrap();
        s.schedule_action(&id, 1_000, "discord").unwrap();
        s.claim_action_for_send(&id, ActionStatus::Scheduled, "engine")
            .unwrap();

        assert!(s
            .finish_send_error(&id, "send_draft: boom", Some(5), "engine")
            .unwrap());
        let (status, _, retry_count) = raw_action_row(&s, &id);
        assert_eq!(status, "error");
        assert_eq!(retry_count, Some(5), "engine failures must stamp the cap");
        // With retryCount at the cap the generic retry tick must skip it.
        let now = now_millis();
        let retryable = s
            .list_retryable_replies("gmail", now, 86_400_000, 0, 5, 10)
            .unwrap();
        assert!(
            retryable.iter().all(|r| r.action.id != id),
            "a capped scheduled-send failure must never re-enter dispatch_reply"
        );
    }

    /// #670 REGRESSION — the retry queue is per-channel. A socialapi Error row
    /// (logged by the DM handler's parse-failure / post_approval-failure paths)
    /// must never surface to the gmail tick, whose recovery path dispatches
    /// through gmail plumbing.
    #[test]
    fn list_retryable_replies_is_scoped_to_caller_platform() {
        let (s, _f) = fresh_store();
        let gmail_email = sample_email("m-plat-gmail");
        let sapi_email = Email {
            platform: "socialapi".into(),
            ..sample_email("m-plat-sapi")
        };
        for e in [&gmail_email, &sapi_email] {
            s.upsert_email(e).unwrap();
            s.log_action(
                &e.message_id,
                None,
                &e.from,
                &e.subject,
                Some(&e.body),
                Some("draft"),
                ActionStatus::Error,
            )
            .unwrap();
        }

        let now = now_millis();
        let ids = |rows: Vec<RetryableReply>| {
            rows.into_iter()
                .map(|r| r.action.message_id)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            ids(s
                .list_retryable_replies("gmail", now, 86_400_000, 0, 5, 10)
                .unwrap()),
            ["m-plat-gmail"],
        );
        assert_eq!(
            ids(s
                .list_retryable_replies("socialapi", now, 86_400_000, 0, 5, 10)
                .unwrap()),
            ["m-plat-sapi"],
        );
    }

    #[test]
    fn approve_path_error_without_override_stays_retryable() {
        let (s, _f) = fresh_store();
        let id = pending_action(&s, "m-sched-5");
        let email = sample_email("m-sched-5");
        s.upsert_email(&email).unwrap();
        s.claim_action_for_send(&id, ActionStatus::Pending, "discord")
            .unwrap();
        assert!(s
            .finish_send_error(&id, "send_draft: transient", None, "discord")
            .unwrap());
        let (status, _, retry_count) = raw_action_row(&s, &id);
        assert_eq!(status, "error");
        assert_eq!(retry_count, Some(0), "approve failures keep the retry path");
    }

    #[test]
    fn due_and_stuck_queries_respect_status_and_time_bounds() {
        let (s, _f) = fresh_store();
        let due = pending_action(&s, "m-due");
        let future = pending_action(&s, "m-future");
        let now = now_millis();
        s.schedule_action(&due, now - 1_000, "t").unwrap();
        s.schedule_action(&future, now + 3_600_000, "t").unwrap();

        let rows = s.due_scheduled_actions(now, 10).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].0, due);

        // Claim it, backdate the claim, and it becomes a stuck row.
        s.claim_action_for_send(&due, ActionStatus::Scheduled, "engine")
            .unwrap();
        assert!(s.stuck_sending_actions(now, 600_000).unwrap().is_empty());
        {
            let guard = s.conn.lock().unwrap();
            guard
                .execute(
                    "UPDATE actions SET updatedAt = ?2 WHERE id = ?1",
                    params![due, now - 700_000],
                )
                .unwrap();
        }
        assert_eq!(s.stuck_sending_actions(now, 600_000).unwrap(), vec![due]);
    }

    #[test]
    fn open_action_queries_see_scheduled_and_sending_rows() {
        let (s, _f) = fresh_store();
        let id = pending_action(&s, "m-open-1");
        s.schedule_action(&id, 1_000, "t").unwrap();
        assert!(
            s.has_open_action("m-open-1").unwrap(),
            "poll loop would draft a duplicate if scheduled rows were invisible"
        );
        let oldest = s.oldest_pending_action_created_at().unwrap();
        assert!(
            oldest.is_some(),
            "observer cursor must reach back to scheduled rows"
        );
        s.claim_action_for_send(&id, ActionStatus::Scheduled, "t")
            .unwrap();
        assert!(s.has_open_action("m-open-1").unwrap());
        assert!(s.oldest_pending_action_created_at().unwrap().is_some());
    }

    #[test]
    fn supersede_by_thread_skips_scheduled_and_sending_rows() {
        let (s, _f) = fresh_store();
        let plain = pending_action(&s, "m-th-0");
        let scheduled = pending_action(&s, "m-th-1");
        let sending = pending_action(&s, "m-th-2");
        s.schedule_action(&scheduled, 1_000, "t").unwrap();
        s.schedule_action(&sending, 1_000, "t").unwrap();
        s.claim_action_for_send(&sending, ActionStatus::Scheduled, "t")
            .unwrap();

        let ids = s
            .mark_pending_drafts_superseded_by_thread("thread-500", "manual reply")
            .unwrap();
        assert!(ids.contains(&plain));
        // The thread-wide flip has no time bound (reconcile Rule 1 asks "any
        // reply EVER"), so it must never touch armed schedules — those are
        // cancelled only through the bounded mark_scheduled_superseded.
        assert!(!ids.contains(&scheduled));
        assert!(
            !ids.contains(&sending),
            "a row mid-send must not be flipped under the Composio call"
        );
        let (status, at, _) = raw_action_row(&s, &scheduled);
        assert_eq!(status, "scheduled");
        assert_eq!(at, Some(1_000));
    }

    #[test]
    fn mark_scheduled_superseded_is_targeted_and_cas_gated() {
        let (s, _f) = fresh_store();
        let id = pending_action(&s, "m-ss-1");
        s.schedule_action(&id, 1_000, "t").unwrap();
        s.set_action_notice(&id, "chan", "msg").unwrap();

        assert!(s
            .mark_scheduled_superseded(&id, "owner replied after arming", "engine")
            .unwrap());
        let (status, at, _) = raw_action_row(&s, &id);
        assert_eq!(status, "superseded");
        assert_eq!(at, None);
        // Notice pointers survive so the notice message can still be cleaned.
        assert!(s.action_notice(&id).unwrap().is_some());
        // Second call loses; the fire-time claim also loses.
        assert!(!s
            .mark_scheduled_superseded(&id, "again", "engine")
            .unwrap());
        assert!(!s
            .claim_action_for_send(&id, ActionStatus::Scheduled, "engine")
            .unwrap());
    }

    #[test]
    fn refresh_pending_draft_only_writes_while_pending() {
        let (s, _f) = fresh_store();
        let id = pending_action(&s, "m-rp-1");
        assert!(s.refresh_pending_draft(&id, "new body").unwrap());
        let guard = s.conn.lock().unwrap();
        let body: String = guard
            .query_row(
                "SELECT draftBody FROM actions WHERE id = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        drop(guard);
        assert_eq!(body, "new body");
        // Resolved row: the revise tail must lose, not resurrect the row.
        s.schedule_action(&id, 1_000, "t").unwrap();
        assert!(!s.refresh_pending_draft(&id, "stomp").unwrap());
        let (status, ..) = raw_action_row(&s, &id);
        assert_eq!(status, "scheduled");
    }

    #[test]
    fn draft_id_in_flight_tracks_sending_claim() {
        let (s, _f) = fresh_store();
        let id = pending_action(&s, "m-if-1");
        s.set_action_draft_id(&id, "draft-if").unwrap();
        assert!(!s.draft_id_in_flight("draft-if").unwrap());
        s.claim_action_for_send(&id, ActionStatus::Pending, "discord")
            .unwrap();
        assert!(s.draft_id_in_flight("draft-if").unwrap());
        s.finish_send_sent(&id, "discord").unwrap();
        assert!(!s.draft_id_in_flight("draft-if").unwrap());
    }

    #[test]
    fn draft_repoint_query_sees_scheduled_not_sending() {
        let (s, _f) = fresh_store();
        let a = pending_action(&s, "m-dr-1");
        let b = pending_action(&s, "m-dr-2");
        s.set_action_draft_id(&a, "draft-x").unwrap();
        s.set_action_draft_id(&b, "draft-x").unwrap();
        s.schedule_action(&a, 1_000, "t").unwrap();
        s.schedule_action(&b, 1_000, "t").unwrap();
        s.claim_action_for_send(&b, ActionStatus::Scheduled, "t")
            .unwrap();
        let ids = s.find_pending_action_ids_by_draft_id("draft-x").unwrap();
        assert_eq!(ids, vec![a], "update-draft repoints scheduled, skips sending");
    }

    #[test]
    fn scheduled_actions_for_reconcile_lists_only_threaded_scheduled_rows() {
        let (s, _f) = fresh_store();
        let threaded = pending_action(&s, "m-rec-1");
        let unthreaded = s
            .log_action(
                "m-rec-2",
                None,
                "peer@example.com",
                "subj",
                None,
                Some("d"),
                ActionStatus::Pending,
            )
            .unwrap();
        s.schedule_action(&threaded, 1_000, "t").unwrap();
        s.schedule_action(&unthreaded, 1_000, "t").unwrap();
        let rows = s.scheduled_actions_for_reconcile().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].0, threaded);
        assert_eq!(rows[0].1, "thread-500");
    }

    #[test]
    fn migration_is_idempotent_for_scheduled_columns() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("store-test.db");
        {
            let s1 = Store::open(&db).unwrap();
            drop(s1);
        }
        // Second open re-runs migrate() over the already-altered schema.
        let s2 = Store::open(&db).unwrap();
        let guard = s2.conn.lock().unwrap();
        let n: i64 = guard
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('actions') \
                  WHERE name IN ('scheduledAtMs','noticeChannelId','noticeMessageId')",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 3);
    }

    // ---- #99 / #100: queue backpressure + exhaustive digest ----

    #[test]
    fn log_flagged_action_persists_reason_and_flagged_enumerates_all() {
        let (s, _f) = fresh_store();
        let e1 = sample_email("f1");
        let e2 = sample_email("f2");
        s.upsert_email(&e1).unwrap();
        s.upsert_email(&e2).unwrap();
        s.log_flagged_action("f1", None, "alice@x.com", "Re: contract", None, "needs sign-off")
            .unwrap();
        s.log_flagged_action("f2", None, "bob@x.com", "Payout failed", None, "")
            .unwrap();

        let flagged = s.flagged_actions_since(0).unwrap();
        assert_eq!(flagged.len(), 2, "both flagged rows must be enumerated, no LIMIT");
        // Reason persisted; empty reason collapses to "flagged".
        let by_from: std::collections::HashMap<_, _> = flagged
            .iter()
            .map(|(f, _s, r)| (f.as_str(), r.as_str()))
            .collect();
        assert_eq!(by_from["alice@x.com"], "needs sign-off");
        assert_eq!(by_from["bob@x.com"], "flagged");
    }

    #[test]
    fn flagged_actions_since_respects_window() {
        let (s, _f) = fresh_store();
        let e = sample_email("fw");
        s.upsert_email(&e).unwrap();
        s.log_flagged_action("fw", None, "a@b.com", "s", None, "r").unwrap();
        // A far-future `since` excludes everything.
        let future = now_millis() + 60_000;
        assert!(s.flagged_actions_since(future).unwrap().is_empty());
        assert_eq!(s.flagged_actions_since(0).unwrap().len(), 1);
    }

    #[test]
    fn pending_actions_enumerates_entire_backlog_oldest_first() {
        let (s, _f) = fresh_store();
        for i in 0..3 {
            let mid = format!("p{i}");
            let e = sample_email(&mid);
            s.upsert_email(&e).unwrap();
            s.log_action(&mid, None, &format!("u{i}@x.com"), "s", None, Some("d"), ActionStatus::Pending)
                .unwrap();
        }
        // A flagged row must not show up in the pending list.
        let ef = sample_email("pf");
        s.upsert_email(&ef).unwrap();
        s.log_flagged_action("pf", None, "z@x.com", "s", None, "r").unwrap();

        let pending = s.pending_actions().unwrap();
        assert_eq!(pending.len(), 3, "all pending, no LIMIT; flagged excluded");
        for (_f, _s, age) in &pending {
            assert!(*age >= 0);
        }
        let oldest = s.oldest_pending_actions(2).unwrap();
        assert_eq!(oldest.len(), 2, "limit honored");
    }

    #[test]
    fn expire_pending_older_than_only_touches_old_pending() {
        let (s, _f) = fresh_store();
        let e = sample_email("ex");
        s.upsert_email(&e).unwrap();
        let id = s
            .log_action("ex", None, "a@b.com", "s", None, Some("d"), ActionStatus::Pending)
            .unwrap();
        // Cutoff in the past → nothing expired (row is fresh).
        assert!(s.expire_pending_older_than(0).unwrap().is_empty());
        assert_eq!(s.pending_reply_count().unwrap(), 1);
        // Cutoff in the future → the fresh row is now "older than" and swept.
        let future = now_millis() + 60_000;
        let swept = s.expire_pending_older_than(future).unwrap();
        assert_eq!(swept, vec![id.clone()]);
        assert_eq!(s.pending_reply_count().unwrap(), 0);
        // Re-sweep is a no-op (already terminal).
        assert!(s.expire_pending_older_than(future).unwrap().is_empty());
        let a = s.get_action_with_email(&id).unwrap().unwrap();
        assert_eq!(a.action.status, "timed_out");
    }

    /// #220 — the periodic sweep cares about a *cohort* of rows with
    /// different ages, not just one. Build three pending rows
    /// backdated to 10d, 5d, and 1d ago; assert that with a 7-day
    /// cutoff only the 10d row transitions and its id is returned.
    /// Then assert that re-running with cutoff far in the past (the
    /// "disabled equivalent" at the store layer) is a no-op even with
    /// the 5d and 1d rows still pending.
    #[test]
    fn expire_pending_older_than_picks_only_the_stale_cohort() {
        let (s, _f) = fresh_store();
        s.upsert_email(&sample_email("old10")).unwrap();
        s.upsert_email(&sample_email("mid5")).unwrap();
        s.upsert_email(&sample_email("fresh1")).unwrap();
        let id_old = s // pii-ok
            .log_action("old10", None, "a@b.com", "s", None, Some("d"), ActionStatus::Pending) // pii-ok
            .unwrap();
        let id_mid = s // pii-ok
            .log_action("mid5", None, "a@b.com", "s", None, Some("d"), ActionStatus::Pending) // pii-ok
            .unwrap();
        let id_fresh = s // pii-ok
            .log_action("fresh1", None, "a@b.com", "s", None, Some("d"), ActionStatus::Pending) // pii-ok
            .unwrap();

        // Backdate createdAt directly so we can model real wall-clock
        // ages without sleeping. log_action stamps `now`; rewrite each
        // row's createdAt to the desired age below.
        let now = now_millis();
        let day = 24i64 * 60 * 60 * 1000;
        {
            let guard = s.conn.lock().unwrap();
            guard
                .execute(
                    "UPDATE actions SET createdAt = ?1 WHERE id = ?2",
                    params![now - 10 * day, id_old],
                )
                .unwrap();
            guard
                .execute(
                    "UPDATE actions SET createdAt = ?1 WHERE id = ?2",
                    params![now - 5 * day, id_mid],
                )
                .unwrap();
            guard
                .execute(
                    "UPDATE actions SET createdAt = ?1 WHERE id = ?2",
                    params![now - 1 * day, id_fresh],
                )
                .unwrap();
        }

        // 7-day cutoff → only the 10d-old row crosses it.
        let cutoff_7d = now - 7 * day;
        let swept = s.expire_pending_older_than(cutoff_7d).unwrap();
        assert_eq!(swept, vec![id_old.clone()]);
        assert_eq!(s.pending_reply_count().unwrap(), 2);

        // Re-run with a cutoff so far in the past that no surviving
        // pending row qualifies (the "0-day = disabled" equivalent at
        // the sweep layer: see [`sweep_stale_drafts_tick`] for the
        // env-level disable). Must be a no-op even though stale rows
        // are still pending — proves the cutoff is honored strictly.
        let far_past = i64::MIN / 2;
        assert!(s.expire_pending_older_than(far_past).unwrap().is_empty());
        assert_eq!(s.pending_reply_count().unwrap(), 2);

        // Sanity: the 10d row is `timed_out`, the others stay `pending`.
        let a_old = s.get_action_with_email(&id_old).unwrap().unwrap();
        assert_eq!(a_old.action.status, "timed_out");
        let a_mid = s.get_action_with_email(&id_mid).unwrap().unwrap();
        assert_eq!(a_mid.action.status, "pending");
        let a_fresh = s.get_action_with_email(&id_fresh).unwrap().unwrap();
        assert_eq!(a_fresh.action.status, "pending");
    }

    #[test]
    fn mark_pending_approved_only_flips_pending_rows() {
        let (s, _f) = fresh_store();
        let e = sample_email("ap");
        s.upsert_email(&e).unwrap();
        let id = s
            .log_action("ap", None, "a@b.com", "s", None, Some("d"), ActionStatus::Pending)
            .unwrap();
        assert!(s.mark_pending_approved(&id).unwrap());
        assert_eq!(s.pending_reply_count().unwrap(), 0);
        // Second call is a no-op (no longer pending).
        assert!(!s.mark_pending_approved(&id).unwrap());
        let a = s.get_action_with_email(&id).unwrap().unwrap();
        assert_eq!(a.action.status, "approved");
    }

    // --- #117 multi-repo allowlist + gate -----------------------------

    #[test]
    fn agent_repo_allowlist_is_default_deny_and_idempotent() {
        let (s, _f) = fresh_store();
        // Default-deny: nothing allowlisted out of the box.
        assert!(s.list_agent_repos(true).unwrap().is_empty());
        assert!(s.get_agent_repo("acme/widgets").unwrap().is_none());

        let r = s
            .upsert_agent_repo("acme/widgets", "main", "cargo test", "infra/", 400)
            .unwrap();
        assert_eq!(r.full_name, "acme/widgets");
        assert!(r.enabled);
        assert_eq!(r.max_diff_lines, 400);

        // Case-insensitive uniqueness: re-granting updates in place, no dup.
        let r2 = s
            .upsert_agent_repo("ACME/Widgets", "develop", "make test", "", 999)
            .unwrap();
        assert_eq!(r2.id, r.id);
        assert_eq!(r2.base_branch, "develop");
        assert_eq!(s.list_agent_repos(false).unwrap().len(), 1);
    }

    #[test]
    fn revoking_repo_cancels_inflight_gate_rows() {
        let (s, _f) = fresh_store();
        s.upsert_agent_repo("acme/widgets", "main", "", "", 600)
            .unwrap();
        let run = s
            .insert_agent_pr_run("acme/widgets", 42, "agent-fix/issue-42", "fix", 12, "pending_approval")
            .unwrap();
        assert!(s.has_open_agent_pr_run("acme/widgets", 42).unwrap());

        let cancelled = s.revoke_agent_repo("acme/widgets").unwrap();
        assert_eq!(cancelled, 1);
        assert!(!s.get_agent_repo("acme/widgets").unwrap().unwrap().enabled);
        let after = s.get_agent_pr_run(&run.id).unwrap().unwrap();
        assert_eq!(after.status, "rejected");
        // Loop default-deny no longer sees it.
        assert!(s.list_agent_repos(true).unwrap().is_empty());
    }

    #[test]
    fn gate_approve_is_cas_guarded() {
        let (s, _f) = fresh_store();
        s.upsert_agent_repo("acme/widgets", "main", "", "", 600)
            .unwrap();
        let run = s
            .insert_agent_pr_run("acme/widgets", 7, "agent-fix/issue-7", "sum", 3, "pending_approval")
            .unwrap();

        let approved = s.approve_agent_pr_run(&run.id).unwrap();
        assert_eq!(approved.unwrap().status, "approved");
        // Second transition is rejected (no longer pending) — no double-fire.
        assert!(s.approve_agent_pr_run(&run.id).unwrap().is_none());
        assert!(s.reject_agent_pr_run(&run.id).unwrap().is_none());

        s.mark_agent_pr_opened(&run.id, "https://github.com/acme/widgets/pull/9")
            .unwrap();
        let done = s.get_agent_pr_run(&run.id).unwrap().unwrap();
        assert_eq!(done.status, "pr_opened");
        assert_eq!(
            done.pr_url.as_deref(),
            Some("https://github.com/acme/widgets/pull/9")
        );

        let hist = s.list_agent_pr_runs(Some("acme/widgets"), 10).unwrap();
        assert_eq!(hist.len(), 1);
        assert_eq!(s.list_agent_pr_runs(None, 10).unwrap().len(), 1);
    }

    // ---- #58 engagement sub-feature query surface ----

    #[test]
    fn own_posts_due_for_poll_respects_horizon_and_platform() {
        let (s, _f) = fresh_store();
        let now = 1_700_000_000_000_i64;
        // In-window LinkedIn post.
        let live = s
            .upsert_own_post("linkedin", "urn:li:activity:1", now, now + 86_400_000)
            .unwrap();
        // Expired (poll_until in the past) — must be excluded.
        s.upsert_own_post("linkedin", "urn:li:activity:2", now, now - 1)
            .unwrap();
        // Different platform — must be excluded from a linkedin query.
        s.upsert_own_post("twitter", "tweet-9", now, now + 86_400_000)
            .unwrap();

        let due = s.own_posts_due_for_poll("linkedin", now).unwrap();
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].id, live);
        assert_eq!(due[0].external_id, "urn:li:activity:1");
        assert!(due[0].last_polled_ms.is_none());

        // After a poll pass the row is stamped.
        s.mark_own_post_polled(&live).unwrap();
        let again = s.own_posts_due_for_poll("linkedin", now).unwrap();
        assert!(again[0].last_polled_ms.is_some());
    }

    #[test]
    fn seen_comment_dedup_is_one_shot() {
        let (s, _f) = fresh_store();
        let now = 1_700_000_000_000_i64;
        let post = s
            .upsert_own_post("linkedin", "urn:li:activity:7", now, now + 1)
            .unwrap();
        // First sighting → true (synthesize a WorkItem). Repeat → false.
        assert!(s
            .record_seen_comment(&post, "c1", Some("jane"), Some("nice!"))
            .unwrap());
        assert!(!s
            .record_seen_comment(&post, "c1", Some("jane"), Some("nice!"))
            .unwrap());
        // A genuinely new comment is still surfaced.
        assert!(s
            .record_seen_comment(&post, "c2", Some("bob"), Some("congrats"))
            .unwrap());
    }

    #[test]
    fn socialapi_seen_comment_dedup_is_one_shot() {
        let (s, _f) = fresh_store();
        // First sighting of (post, comment) → true; repeat → false.
        assert!(s
            .record_seen_socialapi_comment("post_1", "cmt_1", Some("jane"), Some("nice!"))
            .unwrap());
        assert!(!s
            .record_seen_socialapi_comment("post_1", "cmt_1", Some("jane"), Some("nice!"))
            .unwrap());
        // A new comment on the same post is still surfaced.
        assert!(s
            .record_seen_socialapi_comment("post_1", "cmt_2", Some("bob"), Some("gg"))
            .unwrap());
        // Same comment id on a *different* post is a distinct ledger key.
        assert!(s
            .record_seen_socialapi_comment("post_2", "cmt_1", None, None)
            .unwrap());
    }

    #[test]
    fn socialapi_seen_dm_dedup_is_one_shot() {
        let (s, _f) = fresh_store();
        // First sighting of (conversation, message) → true; repeat → false.
        assert!(s
            .record_seen_socialapi_dm("conv_1", "msg_1", Some("jane"), Some("hi!"))
            .unwrap());
        assert!(!s
            .record_seen_socialapi_dm("conv_1", "msg_1", Some("jane"), Some("hi!"))
            .unwrap());
        // A new message in the same conversation is still surfaced.
        assert!(s
            .record_seen_socialapi_dm("conv_1", "msg_2", Some("jane"), Some("again"))
            .unwrap());
        // Same message id in a *different* conversation is a distinct key.
        assert!(s
            .record_seen_socialapi_dm("conv_2", "msg_1", None, None)
            .unwrap());
    }

    #[test]
    fn is_socialapi_dm_seen_reflects_ledger() {
        let (s, _f) = fresh_store();
        assert!(!s.is_socialapi_dm_seen("conv_1", "msg_1").unwrap());
        s.record_seen_socialapi_dm("conv_1", "msg_1", Some("jane"), Some("hi!"))
            .unwrap();
        assert!(s.is_socialapi_dm_seen("conv_1", "msg_1").unwrap());
        // Keyed on the pair, not either half alone.
        assert!(!s.is_socialapi_dm_seen("conv_1", "msg_2").unwrap());
        assert!(!s.is_socialapi_dm_seen("conv_2", "msg_1").unwrap());
    }

    #[test]
    fn socialapi_webhook_event_insert_dedup_drain_and_mark() {
        let (s, _f) = fresh_store();
        // Insert two dm events + one comment event; a duplicate id is ignored.
        assert!(s
            .insert_socialapi_webhook_event("e1", "dm", Some("acc_1"), "{\"id\":\"m1\"}")
            .unwrap());
        assert!(s
            .insert_socialapi_webhook_event("e2", "dm", None, "{\"id\":\"m2\"}")
            .unwrap());
        assert!(s
            .insert_socialapi_webhook_event("e3", "comment", None, "{\"id\":\"c1\"}")
            .unwrap());
        // Duplicate id → false, no second row.
        assert!(!s
            .insert_socialapi_webhook_event("e1", "dm", Some("acc_1"), "{\"id\":\"m1\"}")
            .unwrap());

        // Drain is kind-scoped: only the two dm events come back, oldest first.
        let dms = s
            .take_unprocessed_socialapi_webhook_events("dm", 10)
            .unwrap();
        assert_eq!(dms.len(), 2);
        assert_eq!(dms[0].id, "e1");
        assert_eq!(dms[0].account_id.as_deref(), Some("acc_1"));
        assert_eq!(dms[1].id, "e2");

        // Limit is honored.
        let one = s
            .take_unprocessed_socialapi_webhook_events("dm", 1)
            .unwrap();
        assert_eq!(one.len(), 1);

        // Marking processed removes it from the next drain.
        assert_eq!(s.mark_socialapi_webhook_event_processed("e1").unwrap(), 1);
        let after = s
            .take_unprocessed_socialapi_webhook_events("dm", 10)
            .unwrap();
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].id, "e2");
        // Unknown id → 0 rows touched.
        assert_eq!(s.mark_socialapi_webhook_event_processed("nope").unwrap(), 0);

        // Comment kind still drains independently.
        let comments = s
            .take_unprocessed_socialapi_webhook_events("comment", 10)
            .unwrap();
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].id, "e3");
    }

    #[test]
    fn active_socialapi_account_ids_lists_only_active() {
        let (s, _f) = fresh_store();
        assert!(s.active_socialapi_account_ids().unwrap().is_empty());
        let guard = s.conn.lock().unwrap();
        guard
            .execute(
                "INSERT INTO socialapi_accounts \
                    (id, platform, active, created_at_ms, updated_at_ms) \
                 VALUES ('acc_a','twitter',1,0,0), \
                        ('acc_b','instagram',0,0,0), \
                        ('acc_c','linkedin',1,0,0)",
                [],
            )
            .unwrap();
        drop(guard);
        let ids = s.active_socialapi_account_ids().unwrap();
        assert_eq!(ids, vec!["acc_a".to_string(), "acc_c".to_string()]);
    }

    #[test]
    fn list_and_disable_socialapi_accounts() {
        let (s, _f) = fresh_store();
        assert!(s.list_socialapi_accounts().unwrap().is_empty());
        {
            let guard = s.conn.lock().unwrap();
            guard
                .execute(
                    "INSERT INTO socialapi_accounts \
                        (id, platform, display_name, account_handle, active, created_at_ms, updated_at_ms) \
                     VALUES ('acc_a','twitter','Acme','@acme',1,0,0), \
                            ('acc_b','instagram',NULL,NULL,0,0,0)",
                    [],
                )
                .unwrap();
        }
        // list returns active-first, then by id; surfaces inactive too.
        let all = s.list_socialapi_accounts().unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].id, "acc_a");
        assert!(all[0].active);
        assert_eq!(all[0].display_name.as_deref(), Some("Acme"));
        assert_eq!(all[1].id, "acc_b");
        assert!(!all[1].active);

        // disabling a known id flips active and reports 1 row touched.
        assert_eq!(s.set_socialapi_account_active("acc_a", false, 123).unwrap(), 1);
        assert!(s.active_socialapi_account_ids().unwrap().is_empty());
        // unknown id touches nothing.
        assert_eq!(s.set_socialapi_account_active("nope", false, 123).unwrap(), 0);
    }

    #[test]
    fn active_friend_watch_excludes_paused_and_other_platform() {
        let (s, _f) = fresh_store();
        let now = 1_700_000_000_000_i64;
        s.upsert_friend_watch("linkedin", "urn:li:fsd_profile:A", Some("alex"), "high")
            .unwrap();
        s.upsert_friend_watch("twitter", "@b", None, "medium")
            .unwrap();
        let watch = s.active_friend_watch("linkedin", now).unwrap();
        assert_eq!(watch.len(), 1);
        assert_eq!(watch[0].handle, "urn:li:fsd_profile:A");
        assert_eq!(watch[0].wiki_slug.as_deref(), Some("alex"));
        assert_eq!(watch[0].engagement, "high");

        // friend-post dedup: first sighting true, repeat false.
        assert!(s
            .record_friend_post_seen(&watch[0].id, "urn:li:activity:9", now)
            .unwrap());
        assert!(!s
            .record_friend_post_seen(&watch[0].id, "urn:li:activity:9", now)
            .unwrap());
    }

    #[test]
    fn pending_connection_requests_round_trip() {
        let (s, _f) = fresh_store();
        assert!(s
            .record_connection_request(
                "linkedin",
                "urn:li:invitation:1",
                Some("Jane Doe"),
                Some("https://linkedin.com/in/jane"),
                Some("worked together at Acme"),
            )
            .unwrap());
        // Idempotent on (platform, external_id).
        assert!(!s
            .record_connection_request(
                "linkedin",
                "urn:li:invitation:1",
                Some("Jane Doe"),
                None,
                None,
            )
            .unwrap());

        let pending = s.pending_connection_requests().unwrap();
        assert_eq!(pending.len(), 1);
        let id = pending[0].id.clone();
        assert_eq!(pending[0].requester_name.as_deref(), Some("Jane Doe"));
        assert_eq!(pending[0].decision, "pending");

        let by_id = s.connection_request_by_id(&id).unwrap().unwrap();
        assert_eq!(by_id.external_id, "urn:li:invitation:1");

        // Decision moves it out of the pending queue.
        s.decide_connection_request(&id, "accept").unwrap();
        assert!(s.pending_connection_requests().unwrap().is_empty());
        let decided = s.connection_request_by_id(&id).unwrap().unwrap();
        assert_eq!(decided.decision, "accept");
        assert!(decided.decided_at_ms.is_some());
    }

    // ---- #48 code-mode actions schema --------------------------------

    #[test]
    fn migration_adds_code_mode_columns_on_fresh_store() {
        // `fresh_store` creates the pre-#48 `actions` schema by hand, then
        // opens Store which runs `migrate()`. The new columns must be present
        // and `mode` must default to 'classic' for any inserted row that
        // doesn't set it.
        let (s, f) = fresh_store();
        let id = s
            .log_action(
                "m1",
                None,
                "alice@example.com",
                "s",
                None,
                Some("d"),
                ActionStatus::Pending,
            )
            .unwrap();
        let conn = Connection::open(f.path().join("store-test.db")).unwrap();
        let (mode, source, trace): (String, Option<String>, Option<String>) = conn
            .query_row(
                "SELECT mode, generatedSource, toolCallTrace FROM actions WHERE id = ?1",
                params![id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!(mode, "classic", "classic log_action defaults mode to 'classic'");
        assert!(source.is_none(), "classic rows leave generatedSource NULL");
        assert!(trace.is_none(), "classic rows leave toolCallTrace NULL");
    }

    #[test]
    fn migration_is_idempotent() {
        // Opening Store twice must not error on the second run — every
        // ALTER TABLE is guarded by `column_exists`.
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("store-test.db");
        {
            let conn = Connection::open(&db).unwrap();
            conn.execute_batch(
                r#"
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
                "#,
            )
            .unwrap();
        }
        let _s1 = Store::open(&db).unwrap();
        let _s2 = Store::open(&db).unwrap();
    }

    #[test]
    fn log_action_code_mode_persists_source_and_trace_round_trip() {
        let (s, _f) = fresh_store();
        s.upsert_email(&sample_email("c1")).unwrap();
        let trace_json = r#"[{"call":"db.recentEmailsFrom","args_summary":"sender=alice@example.com, days=30","result_summary":"3 results"},{"call":"tools.draft","args_summary":"channel=gmail, body_len=142","result_summary":"ok"}]"#;
        let source = "async function main() { /* ... */ }\nmain();";
        let id = s
            .log_action_code_mode(
                "c1",
                Some("thr1"),
                "alice@example.com",
                "Re: thing",
                Some("orig body"),
                Some("drafted body"),
                ActionStatus::Pending,
                source,
                trace_json,
            )
            .unwrap();
        let got = s
            .action_code_mode_fields(&id)
            .unwrap()
            .expect("row must exist");
        assert_eq!(got.mode, "code");
        assert_eq!(got.generated_source.as_deref(), Some(source));
        // Assert raw JSON string equality — the store must return exactly what was passed in.
        assert_eq!(got.tool_call_trace.as_deref(), Some(trace_json));
    }

    #[test]
    fn log_action_code_mode_stores_error_field_in_trace_json() {
        // The trace JSON (pre-serialized by the caller) is stored verbatim and
        // returned as-is. Parse both sides to serde_json::Value to confirm the
        // stored bytes represent the same structure that was passed in.
        let (s, _f) = fresh_store();
        s.upsert_email(&sample_email("c2")).unwrap();
        let trace_json = r#"[{"call":"tools.notAThing","args_summary":"","result_summary":"","error":"unknown tool: tools.notAThing"}]"#;
        let id = s
            .log_action_code_mode(
                "c2",
                None,
                "x@example.com",
                "s",
                None,
                None,
                ActionStatus::Error,
                "// failed program",
                trace_json,
            )
            .unwrap();
        let got = s
            .action_code_mode_fields(&id)
            .unwrap()
            .expect("row must exist");
        let stored = got.tool_call_trace.expect("trace populated");
        // Parse both sides to confirm structural equality (allows whitespace differences).
        let expected: serde_json::Value = serde_json::from_str(trace_json).unwrap();
        let actual: serde_json::Value = serde_json::from_str(&stored).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn classic_log_action_leaves_code_mode_fields_null() {
        // Belt-and-suspenders: a classic-path action read back via the new
        // helper returns mode='classic' with no source / no trace.
        let (s, _f) = fresh_store();
        let id = s
            .log_action(
                "cl",
                None,
                "a@example.com",
                "s",
                None,
                Some("d"),
                ActionStatus::Pending,
            )
            .unwrap();
        let got = s
            .action_code_mode_fields(&id)
            .unwrap()
            .expect("row must exist");
        assert_eq!(got.mode, "classic");
        assert!(got.generated_source.is_none());
        assert!(got.tool_call_trace.is_none());
    }

    // ----- #45: Rust-owned schema tests -----

    /// The 9 Node-owned base tables that, as of #45, the Rust crate MUST
    /// create itself on an empty file. Keep this list in sync with the
    /// corresponding parity reference in `test/storeSchemaParity.test.js`.
    const NODE_OWNED_TABLES: &[&str] = &[
        "actions",
        "senders",
        "config",
        "gmail_accounts",
        "emails",
        "channel_subscriptions",
        "slack_workspaces",
        "drive_accounts",
        "drive_sync_state",
    ];

    /// The 10 Node-owned indexes mirrored into Rust as of #45.
    const NODE_OWNED_INDEXES: &[&str] = &[
        "idx_actions_status",
        "idx_actions_created",
        "idx_actions_messageId",
        "idx_gmail_accounts_active",
        "idx_emails_triage",
        "idx_emails_seen",
        "idx_emails_platform",
        "idx_channel_subs_active_mode",
        "idx_slack_workspaces_active",
        "idx_drive_accounts_active",
    ];

    fn table_names(conn: &Connection) -> Vec<String> {
        let mut stmt = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table'")
            .unwrap();
        let rows = stmt
            .query_map([], |r| r.get::<_, String>(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        rows
    }

    fn index_names(conn: &Connection) -> Vec<String> {
        let mut stmt = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='index'")
            .unwrap();
        let rows = stmt
            .query_map([], |r| r.get::<_, String>(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        rows
    }

    /// #45 — opening Store on a fresh tempfile (no Node ever touched it)
    /// must create every Node-owned base table itself.
    #[test]
    fn store_open_creates_all_node_owned_tables() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("store-test.db");
        let _store = Store::open(&db).expect("open on empty file should succeed");
        let conn = Connection::open(&db).unwrap();
        let tables = table_names(&conn);
        for t in NODE_OWNED_TABLES {
            assert!(
                tables.iter().any(|n| n == *t),
                "missing Node-owned base table '{t}'. Present: {tables:?}",
            );
        }
    }

    /// #45 — every Node-owned index must be present after Store::open on an
    /// otherwise-empty tempfile.
    #[test]
    fn store_open_creates_all_required_indexes() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("store-test.db");
        let _store = Store::open(&db).expect("open on empty file should succeed");
        let conn = Connection::open(&db).unwrap();
        let indexes = index_names(&conn);
        for i in NODE_OWNED_INDEXES {
            assert!(
                indexes.iter().any(|n| n == *i),
                "missing Node-owned index '{i}'. Present: {indexes:?}",
            );
        }
    }

    /// #45 — if Node has already partially initialized the DB (the
    /// concurrent-startup case the comment in `initDb()` calls out), the
    /// Rust migration must not error. Simulate by pre-creating one of the
    /// base tables with the minimal Node shape, then open Store.
    #[test]
    fn store_open_is_idempotent_against_node_initialized_db() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("store-test.db");
        {
            let conn = Connection::open(&db).unwrap();
            // Pre-seed actions exactly as Node's initDb would (older shape:
            // no recipientEmail yet — that's what the ALTER guards exist for).
            conn.execute_batch(
                "CREATE TABLE actions (
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
                 );",
            )
            .unwrap();
        }
        let _store = Store::open(&db)
            .expect("open against Node-style pre-seeded db should succeed");
        // And re-opening must be safe (steady-state autostart case).
        let _store2 = Store::open(&db).expect("re-open should be a no-op");
        let conn = Connection::open(&db).unwrap();
        let tables = table_names(&conn);
        for t in NODE_OWNED_TABLES {
            assert!(
                tables.iter().any(|n| n == *t),
                "missing Node-owned base table '{t}' after re-open. Present: {tables:?}",
            );
        }
    }

    // ---- #219 — outbound observer: supersede pending drafts by thread ----

    /// Two pending drafts on T1 + one pending on T2 ⇒ only T1's flip;
    /// T2 untouched; returns the two ids. Also asserts errorMessage carries
    /// the reason so the dashboard can render context.
    // pii-ok — synthetic test fixtures (a@b.com is the local fresh_store convention).
    #[test]
    fn mark_pending_drafts_superseded_by_thread_only_flips_matching_thread() {
        let (s, _f) = fresh_store();
        let id1 = s
            .log_action("m1", Some("T1"), "a@b.com", "s1", None, Some("d1"), ActionStatus::Pending) // pii-ok
            .unwrap();
        let id2 = s
            .log_action("m2", Some("T1"), "a@b.com", "s2", None, Some("d2"), ActionStatus::Pending) // pii-ok
            .unwrap();
        let id3 = s
            .log_action("m3", Some("T2"), "a@b.com", "s3", None, Some("d3"), ActionStatus::Pending) // pii-ok
            .unwrap();

        let affected = s
            .mark_pending_drafts_superseded_by_thread("T1", "superseded by manual reply")
            .unwrap();
        let mut got: Vec<String> = affected;
        got.sort();
        let mut want = vec![id1.clone(), id2.clone()];
        want.sort();
        assert_eq!(got, want, "only T1's pending rows return");

        // Status side-effect check.
        let conn = Connection::open(_f.path().join("store-test.db")).unwrap();
        let status_of = |id: &str| -> (String, Option<String>) {
            conn.query_row(
                "SELECT status, errorMessage FROM actions WHERE id = ?1",
                params![id],
                |r| Ok((r.get::<_, String>(0)?, r.get::<_, Option<String>>(1)?)),
            )
            .unwrap()
        };
        let (s1, e1) = status_of(&id1);
        let (s2, _e2) = status_of(&id2);
        let (s3, _e3) = status_of(&id3);
        assert_eq!(s1, "superseded");
        assert_eq!(s2, "superseded");
        assert_eq!(s3, "pending", "T2 must be untouched");
        assert_eq!(e1.as_deref(), Some("superseded by manual reply"));
    }

    /// `dry_run` rows on the same thread also get swept — they're equivalent
    /// to pending from the user's perspective (a card the daemon would
    /// surface if dry_run flipped off).
    // pii-ok — synthetic test fixture (a@b.com is the local fresh_store convention).
    #[test]
    fn mark_pending_drafts_superseded_by_thread_includes_dry_run() {
        let (s, _f) = fresh_store();
        let id_dry = s
            .log_action("m1", Some("T1"), "a@b.com", "s1", None, Some("d1"), ActionStatus::DryRun) // pii-ok
            .unwrap();
        let affected = s
            .mark_pending_drafts_superseded_by_thread("T1", "")
            .unwrap();
        assert_eq!(affected, vec![id_dry]);
    }

    /// Non-pending rows (already sent / rejected / superseded) on the same
    /// thread stay put — the observer must not "un-send" history.
    // pii-ok — synthetic test fixtures (a@b.com is the local fresh_store convention).
    #[test]
    fn mark_pending_drafts_superseded_by_thread_leaves_terminal_rows_alone() {
        let (s, _f) = fresh_store();
        let id_pending = s
            .log_action("m1", Some("T1"), "a@b.com", "s1", None, Some("d1"), ActionStatus::Pending) // pii-ok
            .unwrap();
        let id_sent = s
            .log_action("m2", Some("T1"), "a@b.com", "s2", None, Some("d2"), ActionStatus::Sent) // pii-ok
            .unwrap();
        let id_rej = s
            .log_action("m3", Some("T1"), "a@b.com", "s3", None, Some("d3"), ActionStatus::Rejected) // pii-ok
            .unwrap();

        let affected = s
            .mark_pending_drafts_superseded_by_thread("T1", "rsn")
            .unwrap();
        assert_eq!(affected, vec![id_pending.clone()]);

        let conn = Connection::open(_f.path().join("store-test.db")).unwrap();
        let st = |id: &str| -> String {
            conn.query_row(
                "SELECT status FROM actions WHERE id = ?1",
                params![id],
                |r| r.get::<_, String>(0),
            )
            .unwrap()
        };
        assert_eq!(st(&id_sent), "sent");
        assert_eq!(st(&id_rej), "rejected");
    }

    /// No matching rows ⇒ empty Vec, no error. Also covers the
    /// short-circuit path that skips the UPDATE entirely.
    #[test]
    fn mark_pending_drafts_superseded_by_thread_empty_when_no_match() {
        let (s, _f) = fresh_store();
        let affected = s
            .mark_pending_drafts_superseded_by_thread("nope", "rsn")
            .unwrap();
        assert!(affected.is_empty());
    }

    /// Cursor starts unset and survives a monotonic upsert; an older write
    /// cannot rewind the high-water timestamp.
    #[test]
    fn outbound_last_seen_is_monotonic() {
        let (s, _f) = fresh_store();
        assert_eq!(s.outbound_last_seen("ent-1").unwrap(), None);
        s.set_outbound_last_seen("ent-1", 1_000).unwrap();
        assert_eq!(s.outbound_last_seen("ent-1").unwrap(), Some(1_000));
        s.set_outbound_last_seen("ent-1", 5_000).unwrap();
        assert_eq!(s.outbound_last_seen("ent-1").unwrap(), Some(5_000));
        // Stale write must NOT rewind.
        s.set_outbound_last_seen("ent-1", 2_000).unwrap();
        assert_eq!(s.outbound_last_seen("ent-1").unwrap(), Some(5_000));
        // Independent per entity.
        assert_eq!(s.outbound_last_seen("ent-2").unwrap(), None);
        s.set_outbound_last_seen("ent-2", 42).unwrap();
        assert_eq!(s.outbound_last_seen("ent-2").unwrap(), Some(42));
    }

    /// #218 — the per-thread outbound log answers the "user already
    /// replied?" question with strict `> after_ms` semantics, is keyed by
    /// thread so other threads don't shadow it, and is idempotent on
    /// repeated inserts of the same `(entity_id, message_id)`.
    #[test]
    fn thread_has_user_reply_after_strict_and_per_thread() {
        let (s, _f) = fresh_store();

        // Empty log: every query is false.
        assert!(!s.thread_has_user_reply_after("T1", 0).unwrap());
        assert!(!s.thread_has_user_reply_after("T1", i64::MAX).unwrap());

        // Record an outbound on T1 at 2_000.
        s.record_outbound_thread_event("ent-1", "msg-A", Some("T1"), 2_000)
            .unwrap();

        // Strictly greater than: equal does NOT match.
        assert!(s.thread_has_user_reply_after("T1", 1_999).unwrap());
        assert!(!s.thread_has_user_reply_after("T1", 2_000).unwrap());
        assert!(!s.thread_has_user_reply_after("T1", 2_001).unwrap());

        // Different thread: never matches.
        assert!(!s.thread_has_user_reply_after("T2", 0).unwrap());

        // Add a second outbound on T2 — T1 lookups still see only T1 rows.
        s.record_outbound_thread_event("ent-1", "msg-B", Some("T2"), 5_000)
            .unwrap();
        assert!(s.thread_has_user_reply_after("T2", 4_999).unwrap());
        assert!(!s.thread_has_user_reply_after("T1", 4_999).unwrap());

        // Idempotent: re-inserting the same (entity_id, message_id) is a
        // no-op. The original timestamp wins.
        s.record_outbound_thread_event("ent-1", "msg-A", Some("T1"), 99_999)
            .unwrap();
        assert!(!s.thread_has_user_reply_after("T1", 50_000).unwrap());

        // A second outbound on T1 from a DIFFERENT entity with a newer
        // timestamp is recorded independently and DOES make the lookup
        // true past the original timestamp.
        s.record_outbound_thread_event("ent-2", "msg-C", Some("T1"), 10_000)
            .unwrap();
        assert!(s.thread_has_user_reply_after("T1", 9_999).unwrap());

        // NULL thread_id is recordable but never satisfies a thread
        // lookup (SQL: `thread_id = ?` is unknown for NULL). After
        // inserting a NULL-thread row at t=20_000, T-other still has no
        // matching reply even though a numerically-newer row exists.
        s.record_outbound_thread_event("ent-1", "msg-noth", None, 20_000)
            .unwrap();
        assert!(!s.thread_has_user_reply_after("T-other", 0).unwrap());
    }

    /// #525 — the dashboard's paste-your-key card writes here; the daemon has
    /// to be able to read it back or the documented primary setup flow leaves
    /// the channel dead.
    #[test]
    fn get_config_reads_dashboard_written_keys() {
        let (s, _f) = fresh_store();
        assert_eq!(s.get_config("socialapi_api_key").unwrap(), None);
        s.with_conn(|c| {
            c.execute(
                "INSERT INTO config (key, value, updatedAt) VALUES \
                    ('socialapi_api_key','  sk_live_abc  ',0), \
                    ('blank_key','   ',0)",
                [],
            )
        })
        .unwrap();
        // Trimmed on the way out.
        assert_eq!(
            s.get_config("socialapi_api_key").unwrap().as_deref(),
            Some("sk_live_abc")
        );
        // A whitespace-only value is "unset", not a key made of spaces.
        assert_eq!(s.get_config("blank_key").unwrap(), None);
        assert_eq!(s.get_config("never_set").unwrap(), None);
    }

    /// #526 — the ownership signal for the DM webhook fast path.
    #[test]
    fn socialapi_account_handles_normalizes_and_includes_inactive() {
        let (s, _f) = fresh_store();
        assert!(s.socialapi_account_handles().unwrap().is_empty());
        s.with_conn(|c| {
            c.execute(
                "INSERT INTO socialapi_accounts \
                    (id, platform, account_handle, active, created_at_ms, updated_at_ms) \
                 VALUES ('a','instagram','@Acme',1,0,0), \
                        ('b','twitter','  BrandX ',0,0,0), \
                        ('c','linkedin',NULL,1,0,0), \
                        ('d','tiktok','',1,0,0)",
                [],
            )
        })
        .unwrap();
        let mut handles = s.socialapi_account_handles().unwrap();
        handles.sort();
        // Lowercased, '@' stripped, trimmed; NULL/empty dropped; the INACTIVE
        // account is still included — disabling it must not make its past
        // outbound messages start looking inbound.
        assert_eq!(handles, vec!["acme".to_string(), "brandx".to_string()]);
    }

    /// #529 — an unrecognized kind is invisible to the drain (which filters on
    /// it), so it would sit at processed=0 forever. Reject at the door.
    #[test]
    fn insert_socialapi_webhook_event_rejects_unknown_kind() {
        let (s, _f) = fresh_store();
        let err = s
            .insert_socialapi_webhook_event("ev_1", "reaction", None, "{}")
            .unwrap_err();
        assert!(
            matches!(err, StoreError::InvalidInput(ref m) if m.contains("reaction")),
            "expected InvalidInput naming the bad kind, got {err:?}"
        );
        // And nothing was written.
        assert!(s
            .take_unprocessed_socialapi_webhook_events("reaction", 10)
            .unwrap()
            .is_empty());
        // Both legal kinds still insert.
        assert!(s
            .insert_socialapi_webhook_event("ev_2", "dm", None, "{}")
            .unwrap());
        assert!(s
            .insert_socialapi_webhook_event("ev_3", "comment", None, "{}")
            .unwrap());
        // Idempotent on repeat id.
        assert!(!s
            .insert_socialapi_webhook_event("ev_2", "dm", None, "{}")
            .unwrap());
    }
    #[test]
    fn journal_sync_cursor_roundtrip_and_ingested_marks() {
        let (store, _dir) = fresh_store();
        assert_eq!(store.get_journal_sync_cursor("o").unwrap(), None);
        let cur = JournalSyncCursor {
            last_sync_ms: Some(5),
            started_at_ms: 10,
            next_token: Some("2".into()),
        };
        store.set_journal_sync_cursor("o", &cur).unwrap();
        assert_eq!(store.get_journal_sync_cursor("o").unwrap(), Some(cur.clone()));
        let cur2 = JournalSyncCursor {
            next_token: None,
            ..cur
        };
        store.set_journal_sync_cursor("o", &cur2).unwrap();
        assert_eq!(store.get_journal_sync_cursor("o").unwrap(), Some(cur2));
        store.clear_journal_sync_cursor("o").unwrap();
        assert_eq!(store.get_journal_sync_cursor("o").unwrap(), None);

        assert!(!store.journal_entry_ingested("o", "e1", 3).unwrap());
        assert!(store.mark_journal_ingested("o", "e1", 3).unwrap());
        assert!(
            !store.mark_journal_ingested("o", "e1", 3).unwrap(),
            "second mark is a no-op"
        );
        assert!(store.journal_entry_ingested("o", "e1", 3).unwrap());
        assert!(
            !store.journal_entry_ingested("o", "e1", 4).unwrap(),
            "a bumped _version is unseen"
        );
        assert!(
            !store.journal_entry_ingested("other", "e1", 3).unwrap(),
            "scoped per owner"
        );
    }
}
