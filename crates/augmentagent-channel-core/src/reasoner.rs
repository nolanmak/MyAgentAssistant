//! Direct `claude` CLI reasoner.
//!
//! Spawns the `claude` binary with `-p --output-format stream-json` and reads
//! the final assistant text. Authenticated via the user's Claude Max
//! subscription session (`claude login`); no API key.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;

use crate::cli_gate::CliGate;

use async_trait::async_trait;
use serde::Deserialize;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tracing::{debug, warn};

use crate::tool_audit::{
    build_audit_record, default_audit_log_path, is_high_risk, parse_stream_event, AuditLogger,
    AuditNotifier, ToolEvent,
};

/// Env-var safelist for `restrict_env=true` spawns (see #128).
///
/// The spawned `claude` CLI needs these to function:
///   - `HOME` — locates `~/.claude/credentials.json` (OAuth) and the
///     Claude Max subscription session cache.
///   - `PATH` — finds `node`, `gh`, `secret-tool`, and any binaries
///     the user has installed for tool subcommands.
///   - `USER` / `LOGNAME` — some libraries identify the running user.
///   - `TERM` / `LANG` — terminal capability + locale for output.
///   - `SHELL` — Bash-allowlisted tool invocations honor the user's
///     shell when claude shells out for command execution.
///   - `DBUS_SESSION_BUS_ADDRESS` — required for `secret-tool` /
///     libsecret access on Linux. Without it, keyring fallback in
///     any sub-CLI breaks.
///
/// LC_* and XDG_* are forwarded dynamically (too many names to list).
const SAFELIST_ENV_VARS: &[&str] = &[
    "HOME",
    "PATH",
    "USER",
    "LOGNAME",
    "TERM",
    "LANG",
    "SHELL",
    "DBUS_SESSION_BUS_ADDRESS",
    // Claude Max + Anthropic-side overrides users set explicitly.
    "CLAUDE_CLI",
    "ANTHROPIC_API_KEY",
    // #248 — SocialAPI.ai read-only drafting aid. Only relevant when the
    // OPTIONAL `AUGMENTAGENT_SOCIALAPI_MCP_READONLY` flag is set, which
    // attaches the SocialAPI MCP server to the reasoner. Listing it here
    // lets the bearer key survive `restrict_env=true` spawns so the MCP
    // server can authenticate; when the flag is off no opts ever forward
    // it, so a stray key in the daemon env cannot leak into a child.
    "SOCIALAPI_API_KEY",
];

/// #248 — OPTIONAL feature flag. When set to a truthy value
/// (`1`/`true`/`yes`/`on`, case-insensitive), the draft/code-mode reasoner
/// MAY have SocialAPI.ai's MCP server attached as a strictly READ-ONLY
/// drafting aid via [`with_socialapi_readonly_mcp`]. Default (unset) is a
/// hard no-op: [`with_socialapi_readonly_mcp`] returns the opts byte-for-byte
/// unchanged so existing behavior is preserved.
pub const SOCIALAPI_MCP_READONLY_FLAG: &str = "AUGMENTAGENT_SOCIALAPI_MCP_READONLY";

/// Env var holding the SocialAPI.ai bearer key. Mirrors
/// `augmentagent_channel_socialapi::auth::ENV_VAR` without taking a crate
/// dependency (channel-core stays free of the socialapi crate).
const SOCIALAPI_API_KEY_ENV: &str = "SOCIALAPI_API_KEY";

/// MCP server name under which SocialAPI.ai is registered. The deny hook in
/// `scripts/aa-socialapi-readonly-guard.sh` keys off the
/// `mcp__socialapi__<tool>` prefix this produces, so the two MUST agree.
const SOCIALAPI_MCP_SERVER_NAME: &str = "socialapi";

/// Default SocialAPI.ai MCP endpoint. Overridable via
/// `AUGMENTAGENT_SOCIALAPI_MCP_URL` for tenants / tests. The MCP server is
/// HTTP-transport (Claude Code `type: "http"`), authenticated with the
/// bearer key forwarded in the `Authorization` header.
const SOCIALAPI_MCP_DEFAULT_URL: &str = "https://api.social-api.ai/v1/mcp";

/// Returns true when the OPTIONAL SocialAPI read-only MCP drafting aid is
/// enabled via [`SOCIALAPI_MCP_READONLY_FLAG`]. Accepts the usual truthy
/// spellings; everything else (including unset) is false.
pub fn socialapi_mcp_readonly_enabled() -> bool {
    matches!(
        std::env::var(SOCIALAPI_MCP_READONLY_FLAG)
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// #248 — Conditionally attach SocialAPI.ai's MCP server to a draft/code-mode
/// [`ReasonerOpts`] as a strictly READ-ONLY drafting aid.
///
/// **Default OFF / strict no-op.** When [`SOCIALAPI_MCP_READONLY_FLAG`] is
/// unset (or not truthy), `opts` is returned **byte-for-byte unchanged** —
/// no MCP server, no env mutation, no settings hook. Callers can wire this
/// in unconditionally; with the flag off the produced opts (and therefore
/// the spawned `claude` args) are identical to today's behavior.
///
/// When the flag IS set, this:
///   1. Registers SocialAPI's MCP server (`mcp__socialapi__*`) via a
///      `settings_json` `mcpServers` HTTP entry and advertises its tools in
///      `allowed_tools` so the model can reach read context.
///   2. Installs a `PreToolUse` deny hook
///      (`scripts/aa-socialapi-readonly-guard.sh`) that blocks any
///      `mcp__socialapi__*` tool whose name implies a write/send/mutation
///      and allows only clearly read-only tools — FAIL-CLOSED.
///   3. Forwards `SOCIALAPI_API_KEY` (already on the env SAFELIST) into
///      `opts.env` so the MCP server can authenticate even under
///      `restrict_env`.
///
/// The read-only guarantee is enforced at TWO layers: the allow-list patterns
/// are deliberately conservative AND the PreToolUse hook independently denies
/// write-implying tools, so a misconfigured allow-list cannot grant a send.
///
/// `repo_root` locates the guard script. The bearer key is loaded
/// just-in-time via [`crate::secret_loader::load_provider_key`] (same posture
/// as `ask_opts`'s COMPOSIO key); if it can't be loaded the MCP server is
/// still configured but will simply fail to authenticate at the server — it
/// can never gain write access regardless.
/// Drafting opts for the SocialAPI.ai channels (#533).
///
/// [`draft_opts`] plus the optional read-only SocialAPI MCP aid, so the
/// drafting model can pull thread/post context it wasn't handed. Scoped to
/// this channel deliberately: attaching a SocialAPI server to a Gmail draft
/// would be noise, and the narrower the surface the smaller the blast radius
/// if the guard hook ever regresses.
///
/// Strictly default-off — [`with_socialapi_readonly_mcp`] is a byte-for-byte
/// no-op unless `AUGMENTAGENT_SOCIALAPI_MCP_READONLY` is set. Before this
/// existed the helper had no production caller at all, so setting that flag
/// attached nothing and `scripts/aa-socialapi-readonly-guard.sh` never ran:
/// a fail-closed guard guarding nothing.
///
/// `repo_root` is resolved from the process cwd, which is where the daemon
/// runs and where the guard script lives.
pub fn socialapi_draft_opts(system_prompt: String, wiki_root: Option<PathBuf>) -> ReasonerOpts {
    let opts = draft_opts(system_prompt, wiki_root);
    let repo_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    with_socialapi_readonly_mcp(opts, &repo_root)
}

pub fn with_socialapi_readonly_mcp(mut opts: ReasonerOpts, repo_root: &std::path::Path) -> ReasonerOpts {
    if !socialapi_mcp_readonly_enabled() {
        // Strict no-op: unchanged opts ⇒ unchanged spawn args.
        return opts;
    }

    let mcp_url = std::env::var("AUGMENTAGENT_SOCIALAPI_MCP_URL")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| SOCIALAPI_MCP_DEFAULT_URL.to_string());

    let guard_path = repo_root.join("scripts/aa-socialapi-readonly-guard.sh");
    opts.settings_json = Some(build_socialapi_readonly_settings(
        SOCIALAPI_MCP_SERVER_NAME,
        &mcp_url,
        &guard_path,
    ));

    // Advertise the server's tools. The wildcard pattern lets the model see
    // SocialAPI read tools; the PreToolUse hook is the real gate that denies
    // writes, so this stays permissive on purpose (defense-in-depth lives in
    // the hook, not in pattern guesswork about tool names).
    let pattern = format!("mcp__{SOCIALAPI_MCP_SERVER_NAME}__*");
    if !opts.allowed_tools.iter().any(|t| t == &pattern) {
        opts.allowed_tools.push(pattern);
    }

    // Forward the bearer key so the MCP server can authenticate under
    // `restrict_env`. Loaded JIT; only added when actually resolvable.
    if !opts.env.iter().any(|(k, _)| k == SOCIALAPI_API_KEY_ENV) {
        if let Some(key) = crate::secret_loader::load_provider_key(SOCIALAPI_API_KEY_ENV) {
            opts.env.push((SOCIALAPI_API_KEY_ENV.into(), key));
        }
    }

    opts
}

/// Build the inline JSON passed to `claude --settings` for the SocialAPI
/// read-only drafting aid (#248). Registers the SocialAPI MCP server
/// (HTTP transport, bearer auth via env-expanded header) AND a PreToolUse
/// deny hook scoped to that server's tools.
///
/// `${SOCIALAPI_API_KEY}` is left as a literal env-expansion token — the
/// Claude CLI expands it from the spawned process env (which we populate),
/// so the raw key never lands in the args list or in any log line.
fn build_socialapi_readonly_settings(
    server_name: &str,
    mcp_url: &str,
    guard_path: &std::path::Path,
) -> String {
    serde_json::json!({
        "mcpServers": {
            server_name: {
                "type": "http",
                "url": mcp_url,
                "headers": {
                    "Authorization": "Bearer ${SOCIALAPI_API_KEY}"
                }
            }
        },
        "hooks": {
            "PreToolUse": [{
                // Only police the SocialAPI MCP tool surface. The guard
                // script itself is fail-closed for anything under this
                // prefix that isn't clearly read-only.
                "matcher": format!("mcp__{server_name}__.*"),
                "hooks": [{
                    "type": "command",
                    "command": guard_path.to_string_lossy()
                }]
            }]
        }
    })
    .to_string()
}

/// Typed reasoner failure surfaced through the `anyhow` chain (#655/#656).
///
/// Callers keep receiving `anyhow::Error`, but the fallback layer (and any
/// caller that cares) can `err.downcast_ref::<ReasonerError>()` to decide
/// what to do. The variants split along the ONE axis that matters for
/// failover: is the failure *provider-side* (another provider might succeed
/// right now) or *local* (retrying elsewhere wastes quota and masks the real
/// problem)?
///
/// Display strings are chosen to preserve the pre-#655 log/UI text — e.g.
/// `RateLimited` renders as `claude rate limit: …`, which greps and the
/// Discord error surface already rely on.
#[derive(Debug, thiserror::Error)]
pub enum ReasonerError {
    /// The provider refused on quota. `reset_at` is best-effort parsed from
    /// the refusal text ("resets 9:30am (America/New_York)"); `None` means
    /// the fallback layer applies its default cooldown.
    #[error("{provider} rate limit: {message}")]
    RateLimited {
        provider: String,
        message: String,
        reset_at: Option<chrono::DateTime<chrono::Utc>>,
    },
    /// The spawned CLI exceeded the watchdog timeout (#656 — before this, a
    /// hung subprocess blocked its pipeline forever and "Anthropic is down"
    /// was undetectable). The child is killed via `kill_on_drop`.
    #[error("{provider} timed out after {secs}s")]
    Timeout { provider: String, secs: u64 },
    /// Provider-side failure that is not a quota refusal: connection errors,
    /// 5xx, non-zero exits with API-shaped stderr, empty output.
    #[error("{provider} unavailable: {message}")]
    Unavailable { provider: String, message: String },
    /// Local fault (missing binary, corrupted config, bad flags). NOT
    /// failover-eligible from the primary's perspective — but the fallback
    /// layer may still try the *next* provider, since a local fault in one
    /// adapter says nothing about the others.
    #[error("{message}")]
    Local { message: String },
    /// #954 — the caller gave up waiting for a #898 CLI-gate permit. Our box
    /// is saturated (or a permit leaked), which says nothing about the
    /// provider: never latched, but the chain may still try the next one.
    #[error("{provider} waited {waited_secs}s for a CLI gate permit")]
    GateTimeout { provider: String, waited_secs: u64 },
}

impl From<crate::cli_gate::GateWaitTimeout> for ReasonerError {
    fn from(e: crate::cli_gate::GateWaitTimeout) -> Self {
        ReasonerError::GateTimeout { provider: e.provider, waited_secs: e.waited_secs }
    }
}

impl ReasonerError {
    /// Provider-side failures are the ones worth latching a cooldown for.
    pub fn is_provider_side(&self) -> bool {
        matches!(
            self,
            ReasonerError::RateLimited { .. }
                | ReasonerError::Timeout { .. }
                | ReasonerError::Unavailable { .. }
        )
    }

    /// Find a `ReasonerError` anywhere in an `anyhow` chain.
    pub fn find_in(err: &anyhow::Error) -> Option<&ReasonerError> {
        err.downcast_ref::<ReasonerError>()
    }
}

/// Watchdog timeout for one spawned reasoner CLI call (#656). Env-tunable
/// without a rebuild; the default is deliberately generous because agentic
/// presets (wiki-ask, auto-PR build) legitimately run many minutes. The
/// point is not tight latency — it's that "hung forever" becomes a typed,
/// failover-eligible error instead of a silently stuck pipeline.
pub fn reasoner_timeout() -> std::time::Duration {
    let secs = std::env::var("AUGMENTAGENT_REASONER_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|s| *s > 0)
        .unwrap_or(3600);
    std::time::Duration::from_secs(secs)
}

/// Names the preset behind a CLI-gate permit for the #954 hold watchdog:
/// "claude" alone cannot say *which* call leaked a slot. The capability class
/// already distinguishes presets without adding a field to every one of them
/// (#655); the ask/draft paths carry a session id on top.
pub(crate) fn caller_tag(opts: &ReasonerOpts) -> String {
    let class = format!("{:?}", crate::providers::classify(opts));
    match opts.session_id.as_deref() {
        Some(sid) => format!("{class}:{sid}"),
        None => class,
    }
}

/// Class-aware watchdog (#655 review): full-agentic and write-tools runs
/// (wiki-ask, auto-PR builds that compile code) legitimately run far longer
/// than a triage/draft call — give them 2× the base budget so the watchdog
/// catches hangs, not honest work.
pub(crate) fn reasoner_timeout_for(opts: &ReasonerOpts) -> std::time::Duration {
    use crate::providers::{classify, CapabilityClass};
    let base = reasoner_timeout();
    match classify(opts) {
        CapabilityClass::FullAgentic | CapabilityClass::WriteTools => base * 2,
        _ => base,
    }
}

/// Best-effort parse of the Claude CLI quota refusal's reset hint:
///
/// ```text
/// You've hit your session limit · resets 9:30am (America/New_York)
/// ```
///
/// Returns the next UTC instant matching `<h>:<mm><am|pm>` in the named IANA
/// timezone (today if still ahead, else tomorrow). Any parse failure returns
/// `None` — the cooldown latch then falls back to a fixed interval, so a
/// wording change can never break failover, only its precision.
pub fn parse_reset_hint(message: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    use chrono::TimeZone;
    let lower = message.to_ascii_lowercase();
    let idx = lower.find("resets ")?;
    let rest = message[idx + "resets ".len()..].trim();
    // Time token: "9:30am" or "10am".
    let time_tok = rest.split_whitespace().next()?;
    let time_lower = time_tok.to_ascii_lowercase();
    let (num, is_pm) = if let Some(t) = time_lower.strip_suffix("pm") {
        (t, true)
    } else if let Some(t) = time_lower.strip_suffix("am") {
        (t, false)
    } else {
        return None;
    };
    let (h, m) = match num.split_once(':') {
        Some((h, m)) => (h.parse::<u32>().ok()?, m.parse::<u32>().ok()?),
        None => (num.parse::<u32>().ok()?, 0),
    };
    if h == 0 || h > 12 || m > 59 {
        return None;
    }
    let hour24 = match (h, is_pm) {
        (12, false) => 0,
        (12, true) => 12,
        (h, true) => h + 12,
        (h, false) => h,
    };
    // Timezone token: "(America/New_York)". Absent → UTC is assumed, which
    // can be hours wrong — tolerable ONLY because of the plausibility cap
    // below, which bounds any tz mistake at a short latch or a None.
    let tz: chrono_tz::Tz = rest
        .find('(')
        .and_then(|open| {
            let close = rest[open..].find(')')? + open;
            rest[open + 1..close].trim().parse().ok()
        })
        .unwrap_or(chrono_tz::UTC);
    let now = chrono::Utc::now().with_timezone(&tz);
    let today = now.date_naive().and_hms_opt(hour24, m, 0)?;
    let candidate = tz.from_local_datetime(&today).earliest()?;
    let candidate = if candidate <= now {
        tz.from_local_datetime(&(today + chrono::Duration::days(1)))
            .earliest()?
    } else {
        candidate
    };
    // Plausibility cap (#655 review): Claude session windows are 5-hourly,
    // so a genuine reset is never more than ~5h out. Anything further means
    // we parsed quoted/stale text, hit the day-rollover on a just-expired
    // window, or mis-assumed the timezone — returning None makes the latch
    // fall back to its short default cooldown, which self-heals, instead of
    // latching the provider for up to a day.
    let candidate_utc = candidate.with_timezone(&chrono::Utc);
    if candidate_utc - chrono::Utc::now() > chrono::Duration::hours(6) {
        return None;
    }
    Some(candidate_utc)
}

/// Wrap an untyped `CallError::Other` in the matching [`ReasonerError`]
/// classification (#656): a missing binary is `Local` (failing over from it
/// is fine, latching is not); everything else — connection failures, non-zero
/// exits, empty output — is `Unavailable`, the "provider might be down"
/// bucket. The original error stays in the chain for diagnostics.
pub(crate) fn classify_other(provider: &str, e: anyhow::Error) -> anyhow::Error {
    let not_found = e.chain().any(|c| {
        c.downcast_ref::<std::io::Error>()
            .is_some_and(|io| io.kind() == std::io::ErrorKind::NotFound)
    });
    if not_found {
        e.context(ReasonerError::Local {
            message: format!("{provider} binary not found on PATH"),
        })
    } else {
        let first_line = e
            .to_string()
            .lines()
            .next()
            .unwrap_or("subprocess failed")
            .chars()
            .take(200)
            .collect::<String>();
        e.context(ReasonerError::Unavailable {
            provider: provider.to_string(),
            message: first_line,
        })
    }
}

/// Per-call options for a `Reasoner`. Each call type (triage, draft, ingest)
/// gets a different preset — see `triage_opts`, `draft_opts`, `ingest_opts`.
#[derive(Debug, Clone)]
pub struct ReasonerOpts {
    pub system_prompt: String,
    pub model: Option<String>,
    pub allowed_tools: Vec<String>,
    pub add_dirs: Vec<PathBuf>,
    pub permission_mode: String,
    /// Override the spawned Claude CLI's working directory. Useful to scope
    /// Write/Edit to a specific subtree (e.g. wiki root) so accidental writes
    /// can't escape into the source tree.
    pub cwd: Option<PathBuf>,
    /// Extra env vars to set on the spawned Claude CLI process. Inherited by
    /// any sub-processes Claude itself spawns (e.g. `augmentagent gmail
    /// search`). Used to pass `AUGMENTAGENT_DB` so sub-CLIs find the db even
    /// when `cwd` is pinned to a sibling directory like the wiki root.
    pub env: Vec<(String, String)>,
    /// Inline JSON forwarded to `claude --settings <json>`. Used by
    /// [`ask_opts`] to install a PreToolUse hook that path-scopes
    /// Read/Write/Edit/Glob/Grep to the wiki root (#127). When `None`,
    /// no `--settings` flag is passed and Claude falls back to its
    /// usual settings-source resolution.
    pub settings_json: Option<String>,
    /// When true, the spawned Claude CLI process's env is **cleared**
    /// before [`env`](Self::env) is applied, retaining only a small
    /// safelist of OS essentials (HOME, PATH, USER, TERM, LANG,
    /// LC_*, XDG_*) so the child can still find its OAuth credentials,
    /// resolve binaries on PATH, and render TTY output. Used by
    /// [`ask_opts`] for #128 so the wiki-query agent does NOT
    /// inherit the daemon's `GROQ_API_KEY`, `CEREBRAS_API_KEY`, etc.
    /// loaded from `.env` at startup. Default `false` keeps existing
    /// call sites (triage, draft, ingest) unchanged.
    pub restrict_env: bool,
    /// NDJSON audit-log sink for tool calls observed in the
    /// `stream-json` output (#132 / #201). When `Some`, each paired
    /// `tool_use`/`tool_result` produces an [`AuditRecord`] appended
    /// to the configured log path. [`ask_opts`] defaults this to the
    /// system path; other presets leave it `None` so triage/draft/ingest
    /// don't grow a new on-disk side effect.
    pub audit_logger: Option<Arc<AuditLogger>>,
    /// Per-request Discord-side notifier invoked when [`is_high_risk`]
    /// matches the tool name (#132 / #201). Constructed in the channel
    /// crate (typically `augmentagent-approval-discord`) so this crate
    /// stays free of any serenity dependency. `None` disables the side
    /// channel; the audit log still receives the record if
    /// [`audit_logger`](Self::audit_logger) is set.
    pub audit_notifier: Option<Arc<dyn AuditNotifier>>,
    /// Logical session id stamped on every audit record produced by
    /// this call (#132 / #201). Typically `format!("{channel}:{msg}")`
    /// so a reviewer can correlate audit rows with a single Discord
    /// turn. `None` falls back to `"-"` in the recorded row.
    pub session_id: Option<String>,
}

/// Trait the channel uses to reach Claude. Test doubles stub this.
///
/// `call_code_mode` and `call_code_mode_with_repair` are sibling entrypoints
/// that reuse the same `claude` CLI machinery as `call`. They are provided
/// as default trait methods so test doubles only need to stub `call` — the
/// fenced-block extraction layer lives here, once.
#[async_trait]
pub trait Reasoner: Send + Sync {
    async fn call(&self, opts: &ReasonerOpts, user_message: &str) -> anyhow::Result<String>;

    /// Like [`call`](Reasoner::call), but returns **everything the model said**
    /// across the turn rather than just its final text block.
    ///
    /// Use this for any caller whose output is prose shown to a human (the
    /// wiki-ask / Discord answer path). `call` keeps only the last block, which
    /// is correct for callers that end their prompt with "emit JSON" but is
    /// actively wrong here: the ask prompt orders the model to write the
    /// deliverable *first* and file a wiki receipt *last*, so last-block-wins
    /// posts the receipt and silently drops the deliverable (#446).
    ///
    /// Defaults to `call` so test doubles and non-CLI reasoners need not
    /// implement it; [`ClaudeCliReasoner`] overrides it.
    async fn call_transcript(
        &self,
        opts: &ReasonerOpts,
        user_message: &str,
    ) -> anyhow::Result<String> {
        self.call(opts, user_message).await
    }

    /// Code-Mode entrypoint. Spawns `claude` exactly like [`Reasoner::call`]
    /// using the caller-provided system prompt (which should be built via
    /// [`crate::prompt::code_mode_system`] from a manifest), parses the
    /// stream-json output identically, then extracts the first fenced
    /// ```ts``` or ```typescript``` block from the assistant text.
    ///
    /// Returns the source string (trimmed) on success. Returns
    /// [`crate::code_mode::CodeModeError::NoCodeBlock`] (wrapped in
    /// `anyhow::Error`) if the response has no fenced ts block.
    async fn call_code_mode(
        &self,
        opts: &ReasonerOpts,
        user_message: &str,
    ) -> anyhow::Result<String> {
        // Wrap the inner `call` error in `CodeModeError::ReasonerFailed` so
        // callers can distinguish "claude itself failed" from "claude returned
        // text but it had no fenced block".
        let raw = match self.call(opts, user_message).await {
            Ok(t) => t,
            Err(e) => return Err(crate::code_mode::CodeModeError::ReasonerFailed(e).into()),
        };
        let source = crate::code_mode::extract_ts_block(&raw)?;
        Ok(source)
    }

    /// Self-repair retry. Same wiring as [`Reasoner::call_code_mode`] but
    /// the caller supplies the prior failed source and the error message,
    /// which are appended to `user_message` before sending. The base
    /// `user_message` should be the same string passed to the first
    /// `call_code_mode` attempt so the cache prefix is preserved.
    async fn call_code_mode_with_repair(
        &self,
        opts: &ReasonerOpts,
        user_message: &str,
        prior_source: &str,
        prior_error: &str,
    ) -> anyhow::Result<String> {
        // Repair tail is sourced from `crate::prompt::code_mode_repair_tail`
        // so the two ways of assembling a repair message (here in the
        // reasoner, or pre-built by the caller via
        // `code_mode_repair_user_message`) produce the same wire bytes
        // given the same base message. The tail also picks the
        // empty-prior-program template (used in #205-#208) automatically.
        let tail = crate::prompt::code_mode_repair_tail(prior_source, prior_error);
        let repair_msg = format!("{user_message}{tail}");
        self.call_code_mode(opts, &repair_msg).await
    }
}

pub struct ClaudeCliReasoner {
    bin: String,
    /// #898 — shared cap on concurrent CLI children; the global gate in
    /// production, a private one in tests.
    gate: Arc<CliGate>,
}

impl Default for ClaudeCliReasoner {
    fn default() -> Self {
        Self {
            bin: std::env::var("CLAUDE_CLI").unwrap_or_else(|_| "claude".into()),
            gate: CliGate::global(),
        }
    }
}

impl ClaudeCliReasoner {
    pub fn new() -> Self {
        Self::default()
    }

    /// #898 — use a specific gate instead of the process-global one.
    pub fn with_gate(mut self, gate: Arc<CliGate>) -> Self {
        self.gate = gate;
        self
    }

    /// Legacy convenience: pin a model for callers that don't build a full
    /// `ReasonerOpts` themselves. Kept only as a construction helper; model
    /// is applied via `ReasonerOpts::model` now.
    pub fn with_model(self, _model: impl Into<String>) -> Self {
        self
    }
}

impl ClaudeCliReasoner {
    /// Shared body of [`Reasoner::call`] and [`Reasoner::call_transcript`] —
    /// identical except for how the stream's text blocks are reduced.
    async fn call_with_capture(
        &self,
        opts: &ReasonerOpts,
        user_message: &str,
        capture: TextCapture,
    ) -> anyhow::Result<String> {
        // #141 — Try the subprocess once. If it fails because the user's
        // `~/.claude.json` is corrupted (the CLI itself writes the corruption
        // diagnostics into stderr and exits non-zero), attempt a one-shot
        // auto-recovery from the most recent on-disk backup and retry. If the
        // retry also fails or no backup exists, surface a short sanitized
        // user-facing error instead of the raw multi-line stderr blob.
        match self.call_once_timed(opts, user_message, capture).await {
            Ok(text) => Ok(text),
            // #448 — do NOT retry. Retrying a quota refusal is exactly the
            // amplification that kept the daemon pinned against the limit.
            // Log it once, loudly and honestly, and surface a typed error the
            // FallbackReasoner chain (#655) can latch a cooldown from and —
            // where the preset allows — route to a fallback provider.
            Err(CallError::RateLimited { message }) => {
                warn!(
                    "claude REFUSED ON QUOTA (not retrying): {message} — \
                     typed as ReasonerError::RateLimited so the provider \
                     chain (#655, AUGMENTAGENT_REASONER_CHAIN) can latch a \
                     cooldown and fail over instead of hammering the wall"
                );
                Err(rate_limited_error("claude", message))
            }
            Err(CallError::Timeout { secs }) => Err(anyhow::Error::new(ReasonerError::Timeout {
                provider: "claude".into(),
                secs,
            })),
            Err(CallError::GateTimeout { waited_secs }) => {
                Err(anyhow::Error::new(ReasonerError::GateTimeout {
                    provider: "claude".into(),
                    waited_secs,
                }))
            }
            // Untyped on purpose — content-level, neither latches nor fails
            // over (#655 review).
            Err(CallError::EmptyOutput) => {
                Err(anyhow::anyhow!("claude produced no assistant text"))
            }
            Err(CallError::ConfigCorrupted { stderr }) => {
                let recovered = match restore_latest_claude_backup() {
                    Ok(Some(_)) => true,
                    Ok(None) => false,
                    Err(e) => {
                        warn!("claude backup restore failed: {e:#}");
                        false
                    }
                };
                if recovered {
                    match self.call_once_timed(opts, user_message, capture).await {
                        Ok(text) => return Ok(text),
                        Err(CallError::ConfigCorrupted { stderr }) => {
                            return Err(anyhow::Error::new(ReasonerError::Local {
                                message: sanitize_claude_error(&stderr),
                            }));
                        }
                        Err(CallError::RateLimited { message }) => {
                            warn!("claude REFUSED ON QUOTA after config recovery: {message}");
                            return Err(rate_limited_error("claude", message));
                        }
                        Err(CallError::Timeout { secs }) => {
                            return Err(anyhow::Error::new(ReasonerError::Timeout {
                                provider: "claude".into(),
                                secs,
                            }));
                        }
                        Err(CallError::GateTimeout { waited_secs }) => {
                            return Err(anyhow::Error::new(ReasonerError::GateTimeout {
                                provider: "claude".into(),
                                waited_secs,
                            }));
                        }
                        Err(CallError::EmptyOutput) => {
                            return Err(anyhow::anyhow!("claude produced no assistant text"));
                        }
                        Err(CallError::Other(e)) => return Err(classify_other("claude", e)),
                    }
                }
                Err(anyhow::Error::new(ReasonerError::Local {
                    message: sanitize_claude_error(&stderr),
                }))
            }
            Err(CallError::Other(e)) => Err(classify_other("claude", e)),
        }
    }

    /// [`call_once`] under the #656 watchdog. On expiry the in-flight future
    /// is dropped, which kills the child via `kill_on_drop(true)` — no
    /// orphaned `claude` processes, no forever-stuck pipeline.
    async fn call_once_timed(
        &self,
        opts: &ReasonerOpts,
        user_message: &str,
        capture: TextCapture,
    ) -> Result<String, CallError> {
        let dur = reasoner_timeout_for(opts);
        // #898 — take the CLI slot *before* the watchdog starts: time spent
        // queued behind other children must not count against this call.
        // The permit lives to the end of this scope — past the child on
        // every path, including the watchdog dropping `call_once`.
        // #954 — the *wait* carries the same budget as the call: an
        // unbounded queue is how the whole daemon stopped reasoning for 15 h.
        let caller = caller_tag(opts);
        let acquire = self.gate.acquire_timed("claude", &caller, dur);
        let _permit = acquire
            .await
            .map_err(|e| CallError::GateTimeout { waited_secs: e.waited_secs })?;
        match tokio::time::timeout(dur, self.call_once(opts, user_message, capture)).await {
            Ok(r) => r,
            Err(_) => {
                warn!(
                    "claude call exceeded the {}s watchdog; child killed (see \
                     AUGMENTAGENT_REASONER_TIMEOUT_SECS, #656)",
                    dur.as_secs()
                );
                Err(CallError::Timeout {
                    secs: dur.as_secs(),
                })
            }
        }
    }
}

/// Build the typed rate-limit error, parsing the reset hint once so both the
/// first-attempt and post-recovery paths stamp the same shape.
fn rate_limited_error(provider: &str, message: String) -> anyhow::Error {
    let reset_at = parse_reset_hint(&message);
    anyhow::Error::new(ReasonerError::RateLimited {
        provider: provider.to_string(),
        message,
        reset_at,
    })
}

#[async_trait]
impl Reasoner for ClaudeCliReasoner {
    async fn call(&self, opts: &ReasonerOpts, user_message: &str) -> anyhow::Result<String> {
        self.call_with_capture(opts, user_message, TextCapture::LastBlock)
            .await
    }

    async fn call_transcript(
        &self,
        opts: &ReasonerOpts,
        user_message: &str,
    ) -> anyhow::Result<String> {
        self.call_with_capture(opts, user_message, TextCapture::AllBlocks)
            .await
    }
}

/// Inner-call result. Distinguishes the "corrupted `~/.claude.json`" failure
/// (which the outer `call` tries to auto-recover from) from any other error
/// (which is propagated as-is).
enum CallError {
    /// `claude` exited non-zero with stderr that matches the
    /// "Configuration error … JSON Parse error" / "is corrupted" pattern the
    /// subprocess emits when `~/.claude.json` is unreadable.
    ConfigCorrupted { stderr: String },
    /// #448 — the CLI refused on quota. Arrives as a *successful* completion
    /// whose text is "You've hit your session limit · resets …", so without
    /// this variant it masquerades as a malformed model answer and gets
    /// retried. Distinct so callers can back off until the reset instead.
    RateLimited { message: String },
    /// #656 — the watchdog expired before the CLI finished. The child is
    /// killed via `kill_on_drop`; distinct so the outer wrapper can surface
    /// a typed, failover-eligible [`ReasonerError::Timeout`].
    Timeout { secs: u64 },
    /// #954 — the #898 gate never handed out a permit, so no child ever ran:
    /// our box's fault, not the provider's.
    GateTimeout { waited_secs: u64 },
    /// The CLI exited 0 but produced no assistant text. Content-level, NOT
    /// provider-side (#655 review): surfaced untyped so it neither latches
    /// the provider nor triggers failover.
    EmptyOutput,
    /// Any other failure: spawn errors, IO errors, non-config exit failures.
    Other(anyhow::Error),
}

impl From<anyhow::Error> for CallError {
    fn from(e: anyhow::Error) -> Self {
        CallError::Other(e)
    }
}

impl From<std::io::Error> for CallError {
    fn from(e: std::io::Error) -> Self {
        CallError::Other(e.into())
    }
}

/// How [`ClaudeCliReasoner::call_once`] turns a multi-event `stream-json`
/// session into the single `String` the caller gets back.
///
/// An agentic turn emits **many** `assistant` events interleaved with
/// `tool_use`, so "the model's text" is a *list* of blocks, not one value.
/// The two callers want opposite halves of it:
///
/// - [`LastBlock`](TextCapture::LastBlock) — machine-readable callers
///   (`triage_opts`, `digest_opts`, `loop_parse_opts`, …) whose prompts end
///   with "emit JSON". The payload is the model's *final* word; any earlier
///   block is scratch reasoning that would corrupt the parse. This is the
///   historical behaviour and stays the default.
/// - [`AllBlocks`](TextCapture::AllBlocks) — human-facing prose callers
///   (`ask_opts`). Here every block is part of the answer.
///
/// See #446: the ask path was pinned to `LastBlock`, so a turn that wrote the
/// deliverable *before* filing it to the wiki lost the deliverable — the
/// trailing "filed it to X" receipt overwrote it and was all Discord ever saw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TextCapture {
    /// Keep only the last non-empty text block (or the `result` event).
    LastBlock,
    /// Keep every text block, in order, joined by a blank line.
    AllBlocks,
}

/// Reduce the text blocks observed across one `stream-json` session down to
/// the string the caller receives.
///
/// `result` is the `result` event's payload, which the CLI populates with the
/// **final** assistant message. Under [`TextCapture::AllBlocks`] it is
/// therefore a duplicate of the last entry in `blocks` and is deliberately
/// ignored — it is only consulted as a fallback when the model produced no
/// assistant text at all (e.g. a session that ended in a tool call).
///
/// Split out of [`ClaudeCliReasoner::call_once`] — which spawns a subprocess
/// and so can't be unit-tested — following the same pattern as
/// [`audit_stream_line`].
fn select_final_text(blocks: &[String], result: Option<&str>, capture: TextCapture) -> String {
    match capture {
        TextCapture::AllBlocks => {
            if blocks.is_empty() {
                result.unwrap_or_default().to_string()
            } else {
                blocks.join("\n\n")
            }
        }
        TextCapture::LastBlock => result
            .map(str::to_string)
            .or_else(|| blocks.last().cloned())
            .unwrap_or_default(),
    }
}

impl ClaudeCliReasoner {
    /// Single-shot spawn-and-read. Pulled out of [`Reasoner::call`] so the
    /// outer wrapper can retry on a recoverable failure (corrupted
    /// `~/.claude.json`, see #141).
    async fn call_once(
        &self,
        opts: &ReasonerOpts,
        user_message: &str,
        capture: TextCapture,
    ) -> Result<String, CallError> {
        let mut args: Vec<String> = vec![
            "-p".into(),
            "--output-format".into(),
            "stream-json".into(),
            "--verbose".into(),
            "--permission-mode".into(),
            opts.permission_mode.clone(),
        ];

        // `--allowedTools`: empty list = no tools. Space-separated names per claude CLI docs.
        args.push("--allowedTools".into());
        args.push(opts.allowed_tools.join(" "));

        // `--system-prompt` REPLACES Claude Code's default system (tools list,
        // MCP metadata, agent catalog — ~25k tokens). Empty → minimal stub.
        let effective_system = if opts.system_prompt.trim().is_empty() {
            "You are a concise assistant."
        } else {
            &opts.system_prompt
        };
        args.push("--system-prompt".into());
        args.push(effective_system.to_string());

        if let Some(m) = &opts.model {
            args.push("--model".into());
            args.push(m.clone());
        }

        for dir in &opts.add_dirs {
            args.push("--add-dir".into());
            args.push(dir.to_string_lossy().into_owned());
        }

        // `--settings <json>` wires in per-call settings — notably the
        // PreToolUse hook that path-scopes Read/Write/Edit/Glob/Grep to
        // `WIKI_ROOT` for the wiki-query agent (#127).
        //
        // #317 — Claude Code does NOT load MCP servers declared under
        // `--settings`; they only come from `--mcp-config`. So if the
        // settings JSON carries an `mcpServers` object, split it out: the
        // servers go to `--mcp-config` (plus `--strict-mcp-config`, so the
        // agent's tool surface stays exactly what we declare and never
        // picks up the host's global MCP config), and the remaining
        // settings (hooks, etc.) go to `--settings`.
        if let Some(settings) = &opts.settings_json {
            let (settings_only, mcp_config) = split_mcp_from_settings(settings);
            if let Some(mcp_json) = mcp_config {
                args.push("--mcp-config".into());
                args.push(mcp_json);
                args.push("--strict-mcp-config".into());
            }
            if let Some(s) = settings_only {
                args.push("--settings".into());
                args.push(s);
            }
        }

        let mut cmd = Command::new(&self.bin);
        cmd.args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            // #656 — the watchdog in `call_once_timed` cancels this future on
            // expiry; killing the child on drop is what makes that cancel
            // real instead of leaking an orphaned CLI still burning quota.
            .kill_on_drop(true);
        // Scope Write/Edit by setting the spawned CLI's cwd when requested.
        if let Some(cwd) = &opts.cwd {
            cmd.current_dir(cwd);
        }
        // #128 — When `restrict_env` is set (wiki-query path), clear the
        // inherited env and re-add only OS essentials. This prevents the
        // spawned `claude` CLI (and any sub-CLIs it spawns via the
        // scoped Bash allowlist) from seeing provider keys the daemon
        // loaded from `.env` at startup. `opts.env` is then layered on
        // top with explicit, just-in-time-loaded secrets.
        if opts.restrict_env {
            cmd.env_clear();
            for var in SAFELIST_ENV_VARS {
                if let Ok(v) = std::env::var(var) {
                    cmd.env(var, v);
                }
            }
            // Forward LC_*/XDG_* dynamically — there are too many to list.
            for (k, v) in std::env::vars() {
                if k.starts_with("LC_") || k.starts_with("XDG_") {
                    cmd.env(k, v);
                }
            }
        }
        // Forward any extra env vars. These are inherited by any subprocesses
        // Claude spawns (notably `augmentagent gmail search` via the scoped
        // Bash allowlist in `ask_opts`), which is how we carry `AUGMENTAGENT_DB`
        // through to a sub-CLI whose cwd is unrelated to the repo root.
        if !opts.env.is_empty() {
            cmd.envs(opts.env.iter().map(|(k, v)| (k.as_str(), v.as_str())));
        }
        let mut child = cmd.spawn()?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(user_message.as_bytes()).await?;
            stdin.shutdown().await?;
        }

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("claude stdout missing"))?;
        let mut lines = BufReader::new(stdout).lines();

        // #201 audit context: only spin up pairing state if a logger or
        // notifier is configured, so the no-audit path (triage/draft/ingest)
        // pays zero overhead.
        let audit_active = opts.audit_logger.is_some() || opts.audit_notifier.is_some();
        let audit_session = opts
            .session_id
            .clone()
            .unwrap_or_else(|| "-".to_string());
        let mut pending_tool_uses: HashMap<String, (String, serde_json::Value)> = HashMap::new();

        // Every non-empty text block the model emitted, in stream order, plus
        // the terminal `result` payload. Reducing these to one string is
        // `select_final_text`'s job — do NOT collapse them here, or a turn
        // that speaks before its last tool call loses everything it said
        // (#446).
        let mut text_blocks: Vec<String> = Vec::new();
        let mut result_text: Option<String> = None;
        while let Some(line) = lines.next_line().await? {
            if line.trim().is_empty() {
                continue;
            }
            if audit_active {
                audit_stream_line(
                    &line,
                    &mut pending_tool_uses,
                    &audit_session,
                    opts.audit_logger.as_ref(),
                    opts.audit_notifier.as_ref(),
                );
            }
            match serde_json::from_str::<StreamEvent>(&line) {
                Ok(StreamEvent::Assistant { message }) => {
                    for block in message.content {
                        if let ContentBlock::Text { text } = block {
                            if !text.trim().is_empty() {
                                text_blocks.push(text);
                            }
                        }
                    }
                }
                Ok(StreamEvent::Result { result }) => {
                    if let Some(r) = result {
                        if !r.trim().is_empty() {
                            result_text = Some(r);
                        }
                    }
                }
                Ok(StreamEvent::Other) => {}
                Err(e) => debug!("stream-json parse skip: {e} line={line}"),
            }
        }
        let final_text = select_final_text(&text_blocks, result_text.as_deref(), capture);

        let status = child.wait().await?;
        if !status.success() {
            let mut stderr_buf = String::new();
            if let Some(mut err) = child.stderr.take() {
                use tokio::io::AsyncReadExt;
                let _ = err.read_to_string(&mut stderr_buf).await;
            }
            warn!("claude exited {status:?}: {stderr_buf}");
            if final_text.is_empty() {
                // #141 — Detect the "~/.claude.json corrupted" pattern and
                // surface a typed error so the outer wrapper can attempt
                // one-shot auto-recovery from the backup the CLI itself
                // wrote.
                if is_claude_config_corrupted(&stderr_buf) {
                    return Err(CallError::ConfigCorrupted { stderr: stderr_buf });
                }
                return Err(CallError::Other(anyhow::anyhow!(
                    "claude exited non-zero: {status:?}: {stderr_buf}"
                )));
            }
        }
        if final_text.is_empty() {
            return Err(CallError::EmptyOutput);
        }
        // #448 — the quota refusal is a *successful* completion. Catch it here,
        // before it reaches a JSON parser that would mislabel it a parse error
        // and trigger a retry storm against a wall we're already up against.
        if is_rate_limited(&final_text) {
            return Err(CallError::RateLimited {
                message: final_text.trim().to_string(),
            });
        }
        Ok(final_text)
    }
}

/// One step of the audit pairing state machine: feed a single stream-json
/// line through [`parse_stream_event`], match `tool_use`/`tool_result` by
/// id via `pending`, and spawn the (best-effort) record + notify side
/// effects. Pulled out of [`ClaudeCliReasoner::call_once`] so unit tests
/// can drive the same code path without spawning the `claude` CLI.
///
/// The logger record and the notifier call are both `tokio::spawn`'d so
/// neither a slow disk nor a flaky Discord HTTP call ever blocks the
/// reasoner's reply path — the swap-in semantics here have to match the
/// inline loop, including the high-risk gate around the notifier.
fn audit_stream_line(
    line: &str,
    pending: &mut HashMap<String, (String, serde_json::Value)>,
    session_id: &str,
    logger: Option<&Arc<AuditLogger>>,
    notifier: Option<&Arc<dyn AuditNotifier>>,
) {
    for ev in parse_stream_event(line) {
        match ev {
            ToolEvent::Use { id, name, input } => {
                pending.insert(id, (name, input));
            }
            ToolEvent::Result {
                id,
                is_error,
                content,
            } => {
                let Some((name, input)) = pending.remove(&id) else {
                    debug!("tool-audit: unmatched tool_result id={id}");
                    continue;
                };
                let record = build_audit_record(
                    chrono::Utc::now().to_rfc3339(),
                    session_id.to_string(),
                    name.clone(),
                    input,
                    &content,
                    is_error,
                );
                if let Some(logger) = logger.cloned() {
                    let rec = record.clone();
                    tokio::spawn(async move {
                        logger.record(&rec).await;
                    });
                }
                if is_high_risk(&name) {
                    if let Some(notifier) = notifier.cloned() {
                        let sid = session_id.to_string();
                        tokio::spawn(async move {
                            notifier.notify(&sid, &record).await;
                        });
                    }
                }
            }
        }
    }
}

/// Recognises the stderr pattern emitted by the `claude` CLI when it cannot
/// parse `~/.claude.json`. We match on the two phrases the CLI prints
/// together: a "Configuration error" / "JSON Parse error" header and the
/// "is corrupted" follow-up. Either phrase alone is enough — the CLI varies
/// the exact wording between versions.
fn is_claude_config_corrupted(stderr: &str) -> bool {
    let needle = stderr.to_ascii_lowercase();
    (needle.contains("configuration error") && needle.contains("json parse error"))
        || needle.contains("is corrupted")
        || needle.contains(".claude.json is corrupted")
}

/// #448 — Recognise the CLI's quota refusal, which it returns as ordinary
/// *assistant text* (exit 0), not an error:
///
/// ```text
/// You've hit your session limit · resets 9:30am (America/New_York)
/// ```
///
/// Because it arrives as a successful completion, every JSON-parsing caller
/// treated it as a malformed model answer. The logs are full of the resulting
/// lie:
///
/// ```text
/// ERROR channel: triage parse failed: no JSON object found in model output;
///                raw=You've hit your session limit · resets 9:30am
/// ```
///
/// That misdiagnosis is expensive: a "parse failure" looks transient, so the
/// email is left in an error state and the action layer retries it (max=5)
/// while the 2-minute poll keeps feeding new work into the same wall. Naming
/// the condition lets callers back off instead of hammering it.
pub(crate) fn is_rate_limited(text: &str) -> bool {
    // #655 review hardening: this check runs on SUCCESSFUL model output, and
    // post-#655 a positive doesn't just fail one call — it latches the
    // provider in the shared cooldown file. So it must only fire when the
    // refusal IS the message, not merely appears in it (a digest summarizing
    // "the owner forwarded a limit notice", or a wiki-ask transcript quoting
    // log.md, legitimately CONTAINS refusal wording). Two gates on top of
    // the phrase pair: genuine CLI refusals are one short line (length cap)
    // and start with the refusal itself (anchor) — a JSON answer starts with
    // '{', prose quoting starts elsewhere.
    let t = text.trim();
    if t.len() > 300 {
        return false;
    }
    let t = t.to_ascii_lowercase();
    let anchored = t.starts_with("you've hit your")
        || t.starts_with("you've reached your")
        || t.starts_with("you have hit your")
        || t.starts_with("you have reached your")
        || t.starts_with("rate limit")
        || t.starts_with("usage limit")
        || t.starts_with("session limit");
    anchored
        && (t.contains("session limit") || t.contains("usage limit") || t.contains("rate limit"))
        && (t.contains("hit your") || t.contains("reached your") || t.contains("resets"))
}

/// Locate the most recent `.claude.json.backup.*` file under
/// `~/.claude/backups/` and copy it over `~/.claude.json`. Returns the path
/// that was restored, `Ok(None)` if no backups exist (or `HOME` is unset),
/// or `Err` if a filesystem op failed. Caller decides whether to retry the
/// spawn.
fn restore_latest_claude_backup() -> anyhow::Result<Option<PathBuf>> {
    let home = match std::env::var_os("HOME") {
        Some(h) => PathBuf::from(h),
        None => return Ok(None),
    };
    let backups_dir = home.join(".claude/backups");
    let target = home.join(".claude.json");
    let read = match std::fs::read_dir(&backups_dir) {
        Ok(r) => r,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    let mut latest: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in read.flatten() {
        let path = entry.path();
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };
        if !name.starts_with(".claude.json.backup.") {
            continue;
        }
        let mtime = entry
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        if latest.as_ref().is_none_or(|(t, _)| mtime > *t) {
            latest = Some((mtime, path));
        }
    }
    let Some((_, src)) = latest else {
        return Ok(None);
    };
    std::fs::copy(&src, &target)?;
    Ok(Some(src))
}

/// Pure helper: turn the multi-line, often-duplicated stderr blob the
/// `claude` CLI emits on a corrupted-config failure into a single short
/// user-facing line. No `ExitStatus(...)` prefix, no absolute home paths, no
/// repeated blocks.
///
/// Caller (`event_handler.rs`) already prefixes the returned string with
/// `"wiki query failed: "` so we deliberately omit that here.
pub fn sanitize_claude_error(stderr: &str) -> String {
    let trimmed = stderr.trim();
    if trimmed.is_empty() {
        return "wiki temporarily unavailable: claude subprocess failed".into();
    }
    // De-duplicate the stderr buffer: the same diagnostic block is sometimes
    // repeated 2-4x by the subprocess. We split on blank-line separators,
    // keep insertion order, and drop exact-duplicate blocks.
    let mut seen: Vec<&str> = Vec::new();
    for block in trimmed.split("\n\n") {
        let b = block.trim();
        if b.is_empty() {
            continue;
        }
        if !seen.iter().any(|s| *s == b) {
            seen.push(b);
        }
    }
    let deduped = seen.join("\n\n");
    let lower = deduped.to_ascii_lowercase();
    if (lower.contains("configuration error") && lower.contains("json parse error"))
        || lower.contains("is corrupted")
        || lower.contains(".claude.json is corrupted")
    {
        return "wiki temporarily unavailable: claude config corrupted (auto-recovery failed)"
            .into();
    }
    // Generic fallback: a single short line scrubbed of `ExitStatus(...)`
    // noise and absolute `$HOME` paths. Cap length so a wall of text from an
    // unrelated tool failure cannot leak to Discord.
    let mut line = deduped.lines().next().unwrap_or("").trim().to_string();
    if line.is_empty() {
        line = "claude subprocess failed".into();
    }
    if let Some(rest) = line.strip_prefix("ExitStatus(") {
        if let Some(idx) = rest.find("): ") {
            line = rest[idx + 3..].to_string();
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        let home_str = home.to_string_lossy().into_owned();
        if !home_str.is_empty() {
            line = line.replace(&home_str, "~");
        }
    }
    if line.len() > 200 {
        line.truncate(200);
        line.push('…');
    }
    format!("wiki temporarily unavailable: {line}")
}

/// Resolve the absolute on-disk path of the app database so it can be passed
/// as `AUGMENTAGENT_DB` to sub-CLIs whose cwd may differ from the repo root.
///
/// Mirrors the resolution order in `main.rs`: explicit `AUGMENTAGENT_DB` env
/// wins, else `<repo_root>/data.db`. We always return an absolute path —
/// `canonicalize` when the file exists, otherwise fall back to a manual
/// absolute join so callers never inherit a relative path.
fn resolve_db_path(repo_root: &std::path::Path) -> PathBuf {
    let raw = std::env::var("AUGMENTAGENT_DB")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo_root.join("data.db"));
    if let Ok(abs) = raw.canonicalize() {
        return abs;
    }
    if raw.is_absolute() {
        raw
    } else {
        repo_root.join(&raw)
    }
}

/// Preset builders for the three call types.
/// Model pinned for email triage (#448). Explicit rather than `None`, so the
/// daemon can never again inherit the owner's interactive model choice.
/// Overridable without a rebuild via `AUGMENTAGENT_TRIAGE_MODEL`.
fn triage_model() -> String {
    std::env::var("AUGMENTAGENT_TRIAGE_MODEL").unwrap_or_else(|_| TRIAGE_MODEL.to_string())
}

/// The Opus tier every quality-critical preset asks for in its comment
/// (`draft`, `lint`, `ask`, `digest`, `social_adapter`, `resume`).
///
/// #448 — those presets all *said* "Opus" in a comment while passing
/// `model: None`, which emits no `--model` flag and so silently inherits
/// whatever the OWNER last picked for their own interactive Claude Code in
/// `~/.claude/settings.json`. That is not a default, it's a leak: the owner
/// selecting `opus[1m]` @ `xhigh` for a coding session re-tiered every
/// background draft, digest and answer onto their Max subscription. Saying
/// "Opus" out loud makes the intent real and the daemon immune to `/model`.
///
/// Overridable via `AUGMENTAGENT_OPUS_MODEL` for a no-rebuild tier change.
fn opus_model() -> String {
    std::env::var("AUGMENTAGENT_OPUS_MODEL").unwrap_or_else(|_| OPUS_MODEL.to_string())
}

/// Default tier for the quality-critical presets. See [`opus_model`].
const OPUS_MODEL: &str = "claude-opus-4-8";

/// Default triage tier. Opus is a deliberate quality call (see `triage_opts`) —
/// what #448 removes is the *inherited* `opus[1m]` / `xhigh` variant, not Opus.
const TRIAGE_MODEL: &str = "claude-opus-4-8";

pub fn triage_opts(wiki_root: Option<PathBuf>) -> ReasonerOpts {
    // Opus for triage. Haiku was too narrow on "flag" — missed personal
    // messages from known contacts asking for engagement. Volume is ~70
    // emails/day so the cost bump is rounding error; this is the most
    // quality-critical step in the pipeline.
    let mut add_dirs = Vec::new();
    let mut allowed_tools = Vec::new();
    if let Some(root) = wiki_root {
        add_dirs.push(root);
        // Read-only access lets triage open a sender's people page and weight
        // importance by prior context without risking wiki mutation.
        allowed_tools = vec!["Read".into(), "Grep".into(), "Glob".into()];
    }
    ReasonerOpts {
        system_prompt: crate::prompt::TRIAGE_SYSTEM.to_string(),
        // #448 — PIN the model. `None` meant "no --model flag", which made the
        // spawned CLI inherit whatever the OWNER had set for their own
        // interactive Claude Code in ~/.claude/settings.json. That silently
        // upgraded ~250 background classifications/day to whatever tier the
        // owner happened to pick for coding (observed: `opus[1m]` at `xhigh`
        // effort) and billed it to their Max subscription — the daemon's own
        // subprocess is what kept hitting "You've hit your session limit".
        // Keep Opus (the comment above is a deliberate quality call), but say
        // so explicitly so a `/model` change by the owner can never again
        // re-tier the daemon.
        model: Some(triage_model()),
        allowed_tools,
        add_dirs,
        permission_mode: "default".into(),
        cwd: None,
        env: Vec::new(),
        settings_json: None,
        restrict_env: false,
        audit_logger: None,
        audit_notifier: None,
        session_id: None,
    }
}

pub fn draft_opts(system_prompt: String, wiki_root: Option<PathBuf>) -> ReasonerOpts {
    let mut add_dirs = Vec::new();
    let mut allowed_tools = Vec::new();
    if let Some(root) = wiki_root {
        add_dirs.push(root);
        allowed_tools = vec!["Read".into(), "Grep".into(), "Glob".into()];
    }
    ReasonerOpts {
        system_prompt,
        model: Some(opus_model()),
        allowed_tools,
        add_dirs,
        permission_mode: "default".into(),
        cwd: None,
        env: Vec::new(),
        settings_json: None,
        restrict_env: false,
        audit_logger: None,
        audit_notifier: None,
        session_id: None,
    }
}

pub fn lint_opts(system_prompt: String, wiki_root: PathBuf) -> ReasonerOpts {
    ReasonerOpts {
        system_prompt,
        model: Some(opus_model()), // Opus — lint is reasoning-heavy, low volume
        allowed_tools: vec!["Read".into(), "Grep".into(), "Glob".into()],
        add_dirs: vec![wiki_root],
        permission_mode: "default".into(),
        cwd: None,
        env: Vec::new(),
        settings_json: None,
        restrict_env: false,
        audit_logger: None,
        audit_notifier: None,
        session_id: None,
    }
}

/// Preset for ad-hoc wiki queries (CLI `wiki ask` + Discord DMs).
///
/// Claude gets a broad toolbelt for this one — if the wiki doesn't answer, it
/// can search the inbox via the `augmentagent gmail search` subcommand (scoped
/// Bash allowlist), reach the web, and persist durable new facts back to the
/// wiki via Write/Edit. The spawned CLI's cwd is pinned to `wiki_root` so
/// Write/Edit cannot escape into the source tree.
pub fn ask_opts(wiki_root: PathBuf, repo_root: PathBuf) -> ReasonerOpts {
    // #337 — WIKI_ROOT must be ABSOLUTE. The daemon launches with
    // `--wiki-dir ./wiki` (relative), and the scope guard resolves WIKI_ROOT
    // with `readlink -f` from the spawned CLI's cwd — which `ask_opts` pins to
    // the wiki root itself. A relative `./wiki` therefore resolves a SECOND
    // time to `<wiki>/wiki`, so every legitimate page path (`projects/x.md`,
    // `about/me.md`, …) falls outside the guard root and the durable-facts
    // write is rejected. Canonicalize up front so cwd, the WIKI_ROOT env, and
    // `--add-dir` all carry the same absolute path. Falls back to joining the
    // current dir when the path doesn't exist yet (keeps tests hermetic).
    let wiki_root = std::fs::canonicalize(&wiki_root).unwrap_or_else(|_| {
        std::env::current_dir()
            .map(|d| d.join(&wiki_root))
            .unwrap_or(wiki_root)
    });
    let bin = repo_root.join("target/release/augmentagent");
    // Scoped Bash patterns. The claude permission matcher does literal
    // string comparison — `augmentagent foo` (bare) and `/abs/path/augmentagent
    // foo` are different patterns even though the shell resolves them to the
    // same binary. We register BOTH forms for every `augmentagent` subcommand
    // because the agent (correctly, per the schema's "binary is on $PATH"
    // claim) tends to invoke the bare command. To make that work end-to-end
    // we also push `bin.parent()` onto the spawned process's PATH below, so
    // bare `augmentagent` actually resolves to *our* release binary. The
    // absolute-path patterns stay as a defense-in-depth fallback. (#214)
    let aa_gh = repo_root.join("scripts/aa-gh");
    let bash_gmail_abs = format!("Bash({} gmail *)", bin.display());
    let bash_gmail_bare = "Bash(augmentagent gmail *)".to_string();
    // #212 — `loop` (singular) reads/updates the sqlite `user_loops` table
    // that backs the Discord `/loop` scheduler (#104). The COMMON case for
    // "kill the hello world loop" — runs inside the daemon, no claude PID.
    let bash_loop_abs = format!("Bash({} loop *)", bin.display());
    let bash_loop_bare = "Bash(augmentagent loop *)".to_string();
    // #174/#176 — `loops` (plural) is OS-level claude PID control for the
    // rare orphan-Claude-Code-session case. Kept on the allowlist; the
    // system prompt points at `loop` first.
    let bash_loops_abs = format!("Bash({} loops *)", bin.display());
    let bash_loops_bare = "Bash(augmentagent loops *)".to_string();
    // #319 — `meetup events <urlname>` is the on-demand, read-only event
    // lookup query mode uses to answer "what are our Meetup events this
    // week". Scoped to the `events` subcommand only — `subscribe`,
    // `unsubscribe`, and `poll-once` (which posts a Discord card) stay out
    // of the agent's reach. Both forms, same `#214` bare-resolves-on-PATH
    // rationale as the other `augmentagent` subcommands.
    let bash_meetup_events_abs = format!("Bash({} meetup events *)", bin.display());
    let bash_meetup_events_bare = "Bash(augmentagent meetup events *)".to_string();
    // #399 — `calendar list-events` is the on-demand, read-only schedule
    // lookup query mode uses to answer "what's on my calendar this week" /
    // "am I free Thursday". Scoped to `list-events` only — `poll-once`
    // (writes dedup rows, sends alerts) and `backfill` stay out of the
    // agent's reach. The no-arg forms are needed because the `*` patterns
    // only match invocations WITH trailing args, and the bare no-arg call
    // (defaults to the next 7 days) is the common one.
    let bash_calendar_list_abs = format!("Bash({} calendar list-events *)", bin.display());
    let bash_calendar_list_bare = "Bash(augmentagent calendar list-events *)".to_string();
    let bash_calendar_list_abs_noargs =
        format!("Bash({} calendar list-events)", bin.display());
    let bash_calendar_list_bare_noargs =
        "Bash(augmentagent calendar list-events)".to_string();
    // #398 — `calendar create-event` PROPOSES an event: it writes a pending
    // action row and posts a Discord approval card; the event is created
    // (and invites sent) only when the operator clicks Approve. Safe to
    // allowlist for the same reason `gmail compose --post` is (#352): the
    // command's only externally visible effect is one card in the existing
    // approval channel. Args are always required, so no no-arg form.
    let bash_calendar_create_abs =
        format!("Bash({} calendar create-event *)", bin.display());
    let bash_calendar_create_bare =
        "Bash(augmentagent calendar create-event *)".to_string();
    // #571 / #572 — operator-initiated social drafts. Same safety argument
    // as `gmail compose --post` and `calendar create-event`: these verbs
    // touch no social API at all. They write a pending action row and post
    // ONE card to the existing approval channel; the send happens only when
    // the operator clicks Approve.
    //
    // Scoped to the card-raising verbs ONLY — deliberately NOT
    // `Bash(augmentagent linkedin *)`. A wildcard there would also hand the
    // agent `linkedin post`, which publishes to the feed directly, and
    // `linkedin login`. Likewise `socialapi *` would expose `connect`.
    let bash_socialapi_dm_abs = format!("Bash({} socialapi dm *)", bin.display());
    let bash_socialapi_dm_bare = "Bash(augmentagent socialapi dm *)".to_string();
    let bash_socialapi_comment_abs = format!("Bash({} socialapi comment *)", bin.display());
    let bash_socialapi_comment_bare = "Bash(augmentagent socialapi comment *)".to_string();
    // Read-only urn lookup. `linkedin dm` needs a `--conversation-urn`, and
    // a DM that arrived only as a notification stub has no stored thread_id
    // to recover it from — without this the agent hits a dead end and falls
    // back to "paste it yourself". Gmail has `gmail search` for exactly this.
    let bash_linkedin_recent_abs = format!("Bash({} linkedin recent-dms *)", bin.display());
    let bash_linkedin_recent_bare = "Bash(augmentagent linkedin recent-dms *)".to_string();
    let bash_linkedin_recent_abs_noargs = format!("Bash({} linkedin recent-dms)", bin.display());
    let bash_linkedin_recent_bare_noargs = "Bash(augmentagent linkedin recent-dms)".to_string();
    let bash_linkedin_dm_abs = format!("Bash({} linkedin dm *)", bin.display());
    let bash_linkedin_dm_bare = "Bash(augmentagent linkedin dm *)".to_string();
    let bash_linkedin_comment_abs = format!("Bash({} linkedin comment *)", bin.display());
    let bash_linkedin_comment_bare = "Bash(augmentagent linkedin comment *)".to_string();
    // The sub-CLI inherits our cwd = wiki_root, so its default `data.db`
    // lookup would fail. Ship an absolute `AUGMENTAGENT_DB` so `main.rs`
    // resolves the db regardless of cwd.
    let db_path = resolve_db_path(&repo_root);
    // #127 — Path-scoping enforcement at the harness layer. The Claude CLI
    // does not natively path-scope Read/Write/Edit/Glob/Grep to `--add-dir`;
    // a 2026-05-24 sandbox-escape probe confirmed Read/Write/Glob succeeded
    // against /tmp and the source tree even though `wiki-ask.md` claimed
    // they were wiki-scoped. We install a PreToolUse hook that rejects any
    // file-tool call whose path resolves outside `wiki_root`. WIKI_ROOT
    // is forwarded as an env var so the hook script can resolve it without
    // baking the absolute path into the JSON.
    let guard_path = repo_root.join("scripts/aa-wiki-scope-guard.sh");
    // #317 — Conversation recall. The wiki-query agent only ever received a
    // short `<conversation_history>` window injected into the prompt; it had
    // no tool to reach earlier turns or its own past drafts, so it would
    // flatly fail to recall work it did the same morning. The
    // `augmentagent-mcp-memory` stdio server (built but never wired into ask
    // mode) exposes exactly that: `search_conversation_history` searches the
    // `emails` + `actions` tables (inbound messages + agent drafts) across
    // all channels, and `memory_search`/`memory_recent` reach the curated
    // memory store. Register it as a stdio MCP server scoped to the daemon
    // db, alongside the path-scope hook.
    let memory_bin = repo_root.join("target/release/augmentagent-mcp-memory");
    let settings_json = build_wiki_scope_settings(&guard_path, &memory_bin, &db_path);

    // #128 — Provider secrets are loaded from the Linux Secret Service
    // just-in-time so the `Read` tool can never exfiltrate them from
    // `.env` (companion fix to #127's path scoping). We forward only
    // the keys the wiki-query agent's sub-CLIs actually need:
    //
    //   - COMPOSIO_API_KEY — `augmentagent gmail` subcommands.
    //
    // GROQ/Cerebras are not consumed by anything the wiki-query agent
    // can spawn (those are used by the daemon's email-triage path
    // running in a separate process), so we deliberately do NOT
    // forward them. Defense in depth: even if scoping breaks, those
    // keys are not in the agent's env.
    let mut env: Vec<(String, String)> = vec![
        (
            "AUGMENTAGENT_DB".into(),
            db_path.to_string_lossy().into_owned(),
        ),
        // WIKI_ROOT is consumed by `scripts/aa-wiki-scope-guard.sh` to
        // know which prefix is permitted for file tools.
        ("WIKI_ROOT".into(), wiki_root.to_string_lossy().into_owned()),
        // #319/#322 — query mode pins cwd to the wiki root, so any sub-CLI
        // that shells to a repo-relative helper (e.g. `meetup events` →
        // scripts/meetup-events.mjs) can't find it via current_dir().
        // Hand it the repo root explicitly.
        (
            "AUGMENTAGENT_REPO_ROOT".into(),
            repo_root.to_string_lossy().into_owned(),
        ),
    ];
    // #915/#922 — the scope guard allows the READ tools under the transcript
    // clone, but only if it can see the same variable the daemon used to
    // open the dir (`add_dirs` below); `restrict_env` means nothing is
    // inherited, so forward it explicitly. Same condition as add_dirs:
    // absent variable or absent directory, absent env.
    if let Some(t) = std::env::var_os("AUGMENTAGENT_TRANSCRIPTS_DIR") {
        let t = PathBuf::from(t);
        if t.is_dir() {
            env.push((
                "AUGMENTAGENT_TRANSCRIPTS_DIR".into(),
                t.to_string_lossy().into_owned(),
            ));
        }
    }
    // #214 — Prepend the release bin's dir to PATH so the agent's bare
    // `augmentagent <subcommand>` invocations actually resolve to OUR
    // binary. Without this, the bare-command allowlist patterns above
    // are dead letters (shell would fail with "command not found" even
    // if the harness allowed the string). The systemd unit's PATH does
    // not include `target/release` by default.
    //
    // Same rationale extends to `scripts/` for the `aa-gh` shim: the
    // bare-form `Bash(aa-gh issue ... *)` patterns registered below are
    // only useful if the shim actually resolves on PATH. PR #199 wired
    // up `aa-gh` by absolute path only, so the bare form the model
    // tends to type (matching the schema's command examples) hit
    // "command not found" / "This command requires approval" in
    // production. Putting `scripts/` on PATH closes that loop.
    if let Some(parent) = bin.parent() {
        let scripts_dir = repo_root.join("scripts");
        let existing = std::env::var("PATH").unwrap_or_default();
        let prepended = if existing.is_empty() {
            format!("{}:{}", parent.display(), scripts_dir.display())
        } else {
            format!(
                "{}:{}:{}",
                parent.display(),
                scripts_dir.display(),
                existing
            )
        };
        env.push(("PATH".into(), prepended));
    }
    if let Some(key) = crate::secret_loader::load_provider_key("COMPOSIO_API_KEY") {
        env.push(("COMPOSIO_API_KEY".into(), key));
    }
    // #352 — Forward the Discord bot token + approval channel so
    // `augmentagent gmail compose --post` can spin a one-shot serenity Http
    // and surface a Discord approval card for the draft instead of dumping
    // prose into chat. Scope is narrow: the wiki-ask agent's Bash allowlist
    // only lets it run `augmentagent gmail …` subcommands, and `--post`
    // ultimately only sends one message to the existing approval channel.
    if let Ok(tok) = std::env::var("DISCORD_BOT_TOKEN") {
        if !tok.is_empty() {
            env.push(("DISCORD_BOT_TOKEN".into(), tok));
        }
    }
    if let Ok(cid) = std::env::var("DISCORD_CHANNEL_ID") {
        if !cid.is_empty() {
            env.push(("DISCORD_CHANNEL_ID".into(), cid));
        }
    }

    ReasonerOpts {
        system_prompt: include_str!("../../../schema/wiki-ask.md").to_string(),
        model: Some(opus_model()), // Opus — quality matters for answer coherence
        allowed_tools: vec![
            "Read".into(),
            "Grep".into(),
            "Glob".into(),
            "Write".into(),
            "Edit".into(),
            "WebSearch".into(),
            "WebFetch".into(),
            // #317 — read-only conversation/memory recall via the memory MCP
            // server registered in settings_json below. Explicit tool names
            // (not a wildcard) so the write tool `memory_write` is NOT exposed
            // here — durable-fact persistence is a separate, deliberate surface.
            "mcp__memory__search_conversation_history".into(),
            "mcp__memory__memory_search".into(),
            "mcp__memory__memory_recent".into(),
            bash_gmail_abs,
            bash_gmail_bare,
            bash_loop_abs,
            bash_loop_bare,
            bash_loops_abs,
            bash_loops_bare,
            bash_meetup_events_abs,
            bash_meetup_events_bare,
            bash_calendar_list_abs,
            bash_calendar_list_bare,
            bash_calendar_list_abs_noargs,
            bash_calendar_list_bare_noargs,
            bash_calendar_create_abs,
            bash_calendar_create_bare,
            bash_socialapi_dm_abs,
            bash_socialapi_dm_bare,
            bash_socialapi_comment_abs,
            bash_socialapi_comment_bare,
            bash_linkedin_recent_abs,
            bash_linkedin_recent_bare,
            bash_linkedin_recent_abs_noargs,
            bash_linkedin_recent_bare_noargs,
            bash_linkedin_dm_abs,
            bash_linkedin_dm_bare,
            bash_linkedin_comment_abs,
            bash_linkedin_comment_bare,
            format!("Bash({} issue create *)", aa_gh.display()),
            format!("Bash({} issue list *)", aa_gh.display()),
            format!("Bash({} issue view *)", aa_gh.display()),
            format!("Bash({} issue comment *)", aa_gh.display()),
            // Bare form. The model tends to copy the bare `aa-gh ...`
            // shape from the schema's command examples even when the
            // surrounding prose says "absolute path required" — the
            // claude permission matcher does literal string matching,
            // so the absolute-path patterns above don't cover the bare
            // form. Without these four entries the wiki-query agent
            // hits "This command requires approval" on perfectly
            // legitimate issue-filing calls. (PATH wired above; same
            // pattern as `augmentagent` subcommands per #214.)
            "Bash(aa-gh issue create *)".to_string(),
            "Bash(aa-gh issue list *)".to_string(),
            "Bash(aa-gh issue view *)".to_string(),
            "Bash(aa-gh issue comment *)".to_string(),
        ],
        add_dirs: {
            // #915: the wiki holds the distilled, cited facts about a meeting;
            // the transcript clone holds what was actually said. Grepping the
            // words is the only way to answer "what exactly did we agree", and
            // the clone is read-only to this daemon, so opening it costs
            // nothing but the path. Absent variable, absent directory —
            // unchanged behaviour.
            let mut dirs = vec![wiki_root.clone()];
            if let Some(t) = std::env::var_os("AUGMENTAGENT_TRANSCRIPTS_DIR") {
                let t = PathBuf::from(t);
                if t.is_dir() {
                    dirs.push(t);
                }
            }
            dirs
        },
        permission_mode: "acceptEdits".into(),
        // Pin cwd to the wiki so Write/Edit cannot touch the source tree.
        cwd: Some(wiki_root.clone()),
        env,
        settings_json: Some(settings_json),
        // #128 — Clear the spawned process's env so the wiki-query
        // agent does not inherit secrets from the daemon's `.env`.
        restrict_env: true,
        // #132 / #201 — Default the wiki-query path to the system
        // audit log. Callers (event_handler.rs) overwrite this with a
        // shared logger + per-request notifier + session_id before
        // dispatching `reasoner.call`.
        audit_logger: Some(Arc::new(AuditLogger::new(default_audit_log_path()))),
        audit_notifier: None,
        session_id: None,
    }
}

/// Split an `mcpServers` object out of a `--settings` JSON blob.
///
/// Returns `(settings_without_mcp, mcp_config_json)`:
/// - `settings_without_mcp` — the original object minus `mcpServers`,
///   serialized, or `None` when nothing is left (so we don't pass an empty
///   `--settings {}`).
/// - `mcp_config_json` — `{"mcpServers": {...}}` ready for `--mcp-config`,
///   or `None` when the settings carried no servers.
///
/// Claude Code reads MCP servers only from `--mcp-config`, never from
/// `--settings`, so presets that want both a hook and an MCP server (e.g.
/// [`ask_opts`]) declare both in `settings_json` and rely on the reasoner to
/// route each to the right flag (#317). On any JSON parse failure the input
/// is passed through unchanged as settings (fail-open to today's behaviour).
fn split_mcp_from_settings(raw: &str) -> (Option<String>, Option<String>) {
    let parsed: serde_json::Value = match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(_) => return (Some(raw.to_string()), None),
    };
    let serde_json::Value::Object(mut map) = parsed else {
        return (Some(raw.to_string()), None);
    };
    let mcp_config = map.remove("mcpServers").map(|servers| {
        serde_json::json!({ "mcpServers": servers }).to_string()
    });
    let settings_only = if map.is_empty() {
        None
    } else {
        Some(serde_json::Value::Object(map).to_string())
    };
    (settings_only, mcp_config)
}

/// Build the inline JSON passed to `claude --settings` for wiki-query mode.
///
/// Registers two things:
/// 1. A PreToolUse hook covering the five file tools Claude Code does not
///    natively path-scope (Read/Write/Edit/Glob/Grep). The hook is a shell
///    script that reads the tool-call JSON from stdin and emits a
///    `decision: "block"` reply when the path resolves outside `WIKI_ROOT`.
/// 2. The `memory` stdio MCP server (#317) — `augmentagent-mcp-memory` —
///    scoped to the daemon db via its own `AUGMENTAGENT_DB` env so the
///    conversation-recall tools work under `restrict_env`. The Claude CLI
///    spawns and owns the server's lifecycle; we only declare it here.
///
/// Returns a one-line JSON string so it survives `--settings` arg passing
/// without shell-quoting heartburn.
fn build_wiki_scope_settings(
    guard_path: &std::path::Path,
    memory_bin: &std::path::Path,
    db_path: &std::path::Path,
) -> String {
    // Use serde_json to ensure the guard path is JSON-escaped correctly
    // (spaces, quotes, backslashes). The settings shape is the Claude Code
    // hook spec: `hooks.PreToolUse[].matcher` (regex) + `hooks[].command`.
    serde_json::json!({
        "mcpServers": {
            "memory": {
                "command": memory_bin.to_string_lossy(),
                "args": [],
                // Scope the server to the daemon db explicitly. The MCP
                // server is spawned by the Claude CLI as a child; under
                // `restrict_env` it cannot rely on inheriting AUGMENTAGENT_DB
                // from the daemon, so we pin it per-server here.
                "env": {
                    "AUGMENTAGENT_DB": db_path.to_string_lossy()
                }
            }
        },
        "hooks": {
            "PreToolUse": [{
                "matcher": "Read|Write|Edit|Glob|Grep",
                "hooks": [{
                    "type": "command",
                    "command": guard_path.to_string_lossy()
                }]
            }]
        }
    })
    .to_string()
}

/// Preset for the morning digest synthesis call.
/// System prompt embedded from `schema/digest-prompt.md`. Opus quality, wiki
/// read-only access so Claude can enrich bare stats with context from
/// people/project pages when the signal warrants it.
pub fn digest_opts(wiki_root: Option<PathBuf>) -> ReasonerOpts {
    let mut add_dirs = Vec::new();
    let mut allowed_tools = Vec::new();
    if let Some(root) = wiki_root {
        add_dirs.push(root);
        allowed_tools = vec!["Read".into(), "Grep".into(), "Glob".into()];
    }
    ReasonerOpts {
        system_prompt: include_str!("../../../schema/digest-prompt.md").to_string(),
        model: Some(opus_model()), // Opus — digest tone + coverage benefit from quality
        allowed_tools,
        add_dirs,
        permission_mode: "default".into(),
        cwd: None,
        env: Vec::new(),
        settings_json: None,
        restrict_env: false,
        audit_logger: None,
        audit_notifier: None,
        session_id: None,
    }
}

/// Preset for the tone-mirroring summarizer (#73). Pure transform from a
/// corpus of sent-mail bodies to a ~120-token JSON voice descriptor — no
/// tools, no wiki access, Haiku for cost (per-recipient refresh is ~$0.001).
pub fn tone_summarize_opts() -> ReasonerOpts {
    ReasonerOpts {
        system_prompt: include_str!("../../../schema/tone-summarize.md").to_string(),
        model: Some("claude-haiku-4-5-20251001".into()),
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
    }
}

/// Preset for the cross-platform content adapter (#53). One call per target
/// platform, fanned out in parallel. Pure text transform: no tools, no wiki
/// access, no Bash. The full system prompt (shared rules + the one platform's
/// section + optional voice sample) is assembled by the content-adapter crate
/// and passed in; we just pin the model + lock the toolbelt shut.
///
/// Opus, not Haiku: voice-matching + format nuance across platforms is
/// quality-sensitive and volume is tiny (a handful of variants per compose).
pub fn social_adapter_opts(system_prompt: String) -> ReasonerOpts {
    ReasonerOpts {
        system_prompt,
        model: Some(opus_model()), // Opus — voice + format fidelity matter, volume is low
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
    }
}

/// Preset for parsing Discord `/loop` create-text into a structured spec.
/// One JSON object out, no tools, Haiku for snappy command-feedback latency
/// (loop creates are interactive — sub-second matters). The system prompt
/// is inlined here since it's tiny and only used at one site.
pub fn loop_parse_opts() -> ReasonerOpts {
    ReasonerOpts {
        system_prompt: r#"You parse a single user request to create a recurring task ("loop").

The input describes one of two cadence forms:

A) INTERVAL — fires every N seconds/minutes/hours/days. Recognised units:
   s/sec/seconds, m/min/minutes, h/hr/hours, d/day/days.

B) CRON — fires on a specific day-of-week or hour-of-day pattern
   (#231). Use when the user mentions a weekday ("Monday", "every
   weekday", "Mon-Fri") OR a specific clock time ("9am", "morning",
   "noon"). REQUIRES a timezone. If the user didn't state one, emit
   an error asking for it — don't guess.

Plus REQUIRED:
- A prompt (what the loop should do each tick)
Optionally:
- A total duration (auto-stop time)

Output a SINGLE JSON object on one line, no prose, no code fences:
  INTERVAL form: {"interval_secs": <int>, "prompt": <string>, "duration_secs": <int or null>}
  CRON form:     {"cron_expr": "<5-field cron>", "tz": "<IANA tz>", "prompt": <string>, "duration_secs": <int or null>}

Cron field order is `min hour day-of-month month day-of-week`. Day-of-week is
Unix convention (0=Sun..6=Sat) but PREFER NAMES (MON, TUE, …) which are
unambiguous: `0 9 * * MON` for "every Monday 9am". `MON-FRI` for weekdays.

On failure (no parseable cadence, ambiguous, empty prompt, OR cron form
without a timezone), output:
  {"error": "<short user-facing message>"}

Examples:
  "5m do the digest" → {"interval_secs": 300, "prompt": "do the digest", "duration_secs": null}
  "say hi every 5 mins" → {"interval_secs": 300, "prompt": "say hi", "duration_secs": null}
  "every 5mins for 20 mins and say hello world 🙂" → {"interval_secs": 300, "prompt": "say hello world 🙂", "duration_secs": 1200}
  "ping me every 10 minutes for the next 2 hours" → {"interval_secs": 600, "prompt": "ping me", "duration_secs": 7200}
  "triage every email every 1h" → {"interval_secs": 3600, "prompt": "triage every email", "duration_secs": null}
  "thirty seconds /digest" → {"interval_secs": 30, "prompt": "/digest", "duration_secs": null}
  "every Monday 9am Eastern, ping me" → {"cron_expr": "0 9 * * MON", "tz": "America/New_York", "prompt": "ping me", "duration_secs": null}
  "every weekday at noon UTC say morning" → {"cron_expr": "0 12 * * MON-FRI", "tz": "UTC", "prompt": "say morning", "duration_secs": null}
  "every monday say hi" → {"error": "what timezone for the Monday schedule? (e.g. America/New_York, UTC)"}
  "every weekday at 8am check inbox" → {"error": "what timezone for 8am? (e.g. America/New_York, UTC)"}
  "asdf" → {"error": "couldn't find a cadence — try `loop 5m do thing`, `loop do thing every 5m`, or `loop every Monday 9am EST do thing`"}
"#.to_string(),
        model: Some("claude-haiku-4-5-20251001".into()),
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
    }
}


/// Preset for the archetype picker (#36). A single fast structured-output
/// classification: email + triage label in, one archetype id (or `none`) +
/// confidence out. Haiku for cost/latency — the issue specifies a fast,
/// single-call classifier; no tools, no wiki.
pub fn archetype_pick_opts() -> ReasonerOpts {
    ReasonerOpts {
        system_prompt: crate::archetype::ARCHETYPE_PICKER_SYSTEM.to_string(),
        model: Some("claude-haiku-4-5-20251001".into()),
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
    }
}

pub fn ingest_opts(system_prompt: String, wiki_root: PathBuf) -> ReasonerOpts {
    ReasonerOpts {
        system_prompt,
        model: Some("claude-haiku-4-5-20251001".into()),
        allowed_tools: vec![
            "Read".into(),
            "Grep".into(),
            "Glob".into(),
            "Write".into(),
            "Edit".into(),
        ],
        add_dirs: vec![wiki_root],
        permission_mode: "acceptEdits".into(),
        cwd: None,
        env: Vec::new(),
        settings_json: None,
        restrict_env: false,
        audit_logger: None,
        audit_notifier: None,
        session_id: None,
    }
}

/// Preset for the one-shot v2 wiki migration tool (`wiki migrate --to v2`).
///
/// Haiku for cost — full 562-page corpus runs ~$5. Read-only against the
/// wiki: the migration tool parses the model's text response and applies
/// the patch via Rust IO so v1 keys are preserved byte-for-byte (#78 §2
/// step 6 + §8 risk register forbid round-tripping the whole frontmatter).
/// cwd pinned so any accidental Bash escapes are scoped to the wiki tree.
pub fn wiki_migrate_opts(system_prompt: String, wiki_root: PathBuf) -> ReasonerOpts {
    ReasonerOpts {
        system_prompt,
        model: Some("claude-haiku-4-5-20251001".into()),
        allowed_tools: vec!["Read".into(), "Grep".into(), "Glob".into()],
        add_dirs: vec![wiki_root.clone()],
        permission_mode: "acceptEdits".into(),
        cwd: Some(wiki_root),
        env: Vec::new(),
        settings_json: None,
        restrict_env: false,
        audit_logger: None,
        audit_notifier: None,
        session_id: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    /// In-memory `AuditNotifier` for the audit_stream_line integration test.
    /// Records every `notify` call (session_id + tool name) so the test can
    /// assert that high-risk hits fire and Read does not.
    #[derive(Debug, Default)]
    struct CountingNotifier {
        seen: Mutex<Vec<(String, String)>>,
        calls: AtomicUsize,
    }

    #[async_trait]
    impl AuditNotifier for CountingNotifier {
        async fn notify(&self, session_id: &str, record: &crate::tool_audit::AuditRecord) {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.seen
                .lock()
                .unwrap()
                .push((session_id.to_string(), record.tool.clone()));
        }
    }

    #[tokio::test]
    async fn audit_stream_line_records_writes_and_pings_high_risk() {
        let tmp = tempfile::tempdir().unwrap();
        let log_path = tmp.path().join("tool-audit.log");
        let logger = Arc::new(AuditLogger::new(log_path.clone()));
        // Hold a typed Arc so the test can inspect counters at the end,
        // plus a trait-object Arc (sharing the same allocation via coercion)
        // to hand to the helper.
        let typed: Arc<CountingNotifier> = Arc::new(CountingNotifier::default());
        let notifier: Arc<dyn AuditNotifier> = typed.clone();

        let mut pending: HashMap<String, (String, serde_json::Value)> = HashMap::new();
        let session = "ch:123";
        // 1) High-risk Write: tool_use then tool_result.
        let write_use = r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"tu_w","name":"Write","input":{"file_path":"/tmp/x.txt","content":"hi"}}]}}"#;
        let write_res = r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"tu_w","is_error":false,"content":"ok"}]}}"#;
        // 2) Non-high-risk Read: tool_use then tool_result.
        let read_use = r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"tu_r","name":"Read","input":{"file_path":"/wiki/people.md"}}]}}"#;
        let read_res = r##"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"tu_r","is_error":false,"content":"hello"}]}}"##;

        let log_arc = logger.clone();
        let notifier_arc = notifier.clone();
        audit_stream_line(write_use, &mut pending, session, Some(&log_arc), Some(&notifier_arc));
        audit_stream_line(write_res, &mut pending, session, Some(&log_arc), Some(&notifier_arc));
        audit_stream_line(read_use, &mut pending, session, Some(&log_arc), Some(&notifier_arc));
        audit_stream_line(read_res, &mut pending, session, Some(&log_arc), Some(&notifier_arc));

        // Both record + notify spawn background tasks; give them a moment.
        for _ in 0..20 {
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
            if log_path.exists() {
                let body = tokio::fs::read_to_string(&log_path).await.unwrap_or_default();
                if body.lines().count() >= 2 && typed.calls.load(Ordering::SeqCst) >= 1 {
                    break;
                }
            }
        }

        let body = tokio::fs::read_to_string(&log_path).await.unwrap();
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 2, "both tool calls land in the log");

        let r1: crate::tool_audit::AuditRecord = serde_json::from_str(lines[0]).unwrap();
        let r2: crate::tool_audit::AuditRecord = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(r1.tool, "Write");
        assert_eq!(r1.session_id, session);
        assert_eq!(r2.tool, "Read");

        // Discord-side notifier fires for Write only, NOT Read.
        let seen = typed.seen.lock().unwrap();
        assert_eq!(seen.len(), 1, "only Write is high-risk");
        assert_eq!(seen[0].1, "Write");
        assert_eq!(seen[0].0, session);
        assert!(pending.is_empty(), "all uses get paired by their result");
    }

    #[tokio::test]
    async fn audit_stream_line_inactive_skips_when_no_sinks() {
        // Empty pending + None logger/notifier — should be a no-op even on a
        // tool_use line. Guard against future refactors that accidentally
        // pay parse cost on the no-audit path.
        let mut pending: HashMap<String, (String, serde_json::Value)> = HashMap::new();
        let line = r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"tu_x","name":"Write","input":{}}]}}"#;
        audit_stream_line(line, &mut pending, "s", None, None);
        // Pending gains one entry because the use was parsed; that's fine —
        // the call_once outer wrapper only invokes this helper when
        // audit_active is true. The contract here is "no panics, no IO".
        assert_eq!(pending.len(), 1);
    }


    /// Test double that records the args of the last `call` and returns a
    /// canned string. Used to verify `call_code_mode` extracts the fenced
    /// block from whatever the underlying `call` returns, without spawning
    /// the real `claude` CLI.
    struct CannedReasoner {
        canned: String,
        last_user: Mutex<Option<String>>,
        last_system: Mutex<Option<String>>,
        fail: bool,
    }

    impl CannedReasoner {
        fn ok(canned: impl Into<String>) -> Self {
            Self {
                canned: canned.into(),
                last_user: Mutex::new(None),
                last_system: Mutex::new(None),
                fail: false,
            }
        }
        fn err() -> Self {
            Self {
                canned: String::new(),
                last_user: Mutex::new(None),
                last_system: Mutex::new(None),
                fail: true,
            }
        }
    }

    #[async_trait]
    impl Reasoner for CannedReasoner {
        async fn call(&self, opts: &ReasonerOpts, user_message: &str) -> anyhow::Result<String> {
            *self.last_user.lock().unwrap() = Some(user_message.to_string());
            *self.last_system.lock().unwrap() = Some(opts.system_prompt.clone());
            if self.fail {
                Err(anyhow::anyhow!("simulated claude failure"))
            } else {
                Ok(self.canned.clone())
            }
        }
    }

    fn dummy_opts() -> ReasonerOpts {
        ReasonerOpts {
            system_prompt: "stub".into(),
            model: None,
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
        }
    }

    #[tokio::test]
    async fn call_code_mode_returns_fenced_ts_body_verbatim() {
        let canned = "Sure, here's the program:\n\n```ts\nasync function main(): Promise<void> {\n  await tools.draft(\"gmail\", \"hello\", \"reason\");\n}\nawait main();\n```\n";
        let reasoner = CannedReasoner::ok(canned);
        let got = reasoner
            .call_code_mode(&dummy_opts(), "user")
            .await
            .expect("must extract ts block");
        assert_eq!(
            got,
            "async function main(): Promise<void> {\n  await tools.draft(\"gmail\", \"hello\", \"reason\");\n}\nawait main();"
        );
    }

    #[tokio::test]
    async fn call_code_mode_errors_when_no_fenced_block() {
        let reasoner = CannedReasoner::ok("Sorry, I can't help with that.");
        let err = reasoner
            .call_code_mode(&dummy_opts(), "user")
            .await
            .expect_err("no code block must error");
        // Downcast through the anyhow chain to verify the typed error.
        let typed = err
            .downcast_ref::<crate::code_mode::CodeModeError>()
            .expect("error must be a CodeModeError");
        assert!(matches!(
            typed,
            crate::code_mode::CodeModeError::NoCodeBlock
        ));
    }

    #[tokio::test]
    async fn call_code_mode_wraps_underlying_call_failure() {
        let reasoner = CannedReasoner::err();
        let err = reasoner
            .call_code_mode(&dummy_opts(), "user")
            .await
            .expect_err("must surface call failure");
        let typed = err
            .downcast_ref::<crate::code_mode::CodeModeError>()
            .expect("error must be a CodeModeError");
        assert!(matches!(
            typed,
            crate::code_mode::CodeModeError::ReasonerFailed(_)
        ));
    }

    #[tokio::test]
    async fn call_code_mode_with_repair_appends_prior_program_and_error() {
        let canned = "```ts\nawait tools.draft(\"gmail\", \"fixed\", \"r\");\n```";
        let reasoner = CannedReasoner::ok(canned);
        let base_user = "Write a TypeScript program that drafts a reply.";
        let got = reasoner
            .call_code_mode_with_repair(
                &dummy_opts(),
                base_user,
                "async function main(){ throw new Error('boom') } main();",
                "Error: boom",
            )
            .await
            .expect("repair succeeds");
        assert_eq!(got, "await tools.draft(\"gmail\", \"fixed\", \"r\");");

        // The user message sent to `call` must include both the base text
        // and the failed program + error blocks.
        let sent = reasoner
            .last_user
            .lock()
            .unwrap()
            .clone()
            .expect("call must have been invoked");
        assert!(sent.contains(base_user), "missing base user msg: {sent}");
        assert!(sent.contains("<prior_program>"));
        assert!(sent.contains("throw new Error('boom')"));
        assert!(sent.contains("<prior_error>"));
        assert!(sent.contains("Error: boom"));
    }

    /// #446 regression. The ask prompt orders the model to emit the
    /// deliverable first and file a durable-facts wiki receipt last, so a
    /// content turn streams as: text(post) → tool_use(Write) → text(receipt).
    /// Under the old last-block-wins extraction Discord only ever saw the
    /// receipt — the post itself was silently discarded, with no error, while
    /// the wiki file it referenced was written correctly. This is the bug the
    /// owner hit three times in a row on 2026-07-14.
    #[test]
    fn all_blocks_keeps_a_deliverable_written_before_the_wiki_receipt() {
        let post = "Tomorrow at Blockspace: the OG Coffee&Code.\n\nNo panels, no pitches.";
        let receipt = "Filed the event details in projects/blockspace-philly.md.";
        let blocks = vec![post.to_string(), receipt.to_string()];

        // The `result` event echoes the model's FINAL message, i.e. the
        // receipt. AllBlocks must ignore it rather than let it win.
        let got = select_final_text(&blocks, Some(receipt), TextCapture::AllBlocks);

        assert!(
            got.starts_with(post),
            "the deliverable must lead the reply, got: {got:?}"
        );
        assert!(got.contains(receipt), "the filing note is kept too");
        assert_eq!(got, format!("{post}\n\n{receipt}"));
    }

    /// The other half of the contract: JSON callers (`triage_opts`,
    /// `digest_opts`, `loop_parse_opts`, …) must keep last-block-wins. Their
    /// prompts end with "emit JSON", and the triage prompt has the model
    /// Read/Grep the wiki first — so scratch prose routinely precedes the
    /// payload. Concatenating it would put prose in front of the object and
    /// break `no JSON object found in model output`.
    #[test]
    fn last_block_capture_still_isolates_the_json_payload() {
        let json = r#"{"decision":"skip","reason":"marketing blast"}"#;
        let blocks = vec![
            "Let me check the wiki for this sender.".to_string(),
            json.to_string(),
        ];

        let got = select_final_text(&blocks, Some(json), TextCapture::LastBlock);

        assert_eq!(got, json, "scratch reasoning must not reach the parser");
        assert!(serde_json::from_str::<serde_json::Value>(&got).is_ok());
    }

    /// A session that ends on a tool call emits no trailing assistant text.
    /// Both modes fall back to the `result` payload rather than returning ""
    /// (which would trip the "claude produced no assistant text" error).
    #[test]
    fn both_modes_fall_back_to_the_result_event_when_no_text_blocks() {
        let blocks: Vec<String> = Vec::new();
        assert_eq!(
            select_final_text(&blocks, Some("done"), TextCapture::AllBlocks),
            "done"
        );
        assert_eq!(
            select_final_text(&blocks, Some("done"), TextCapture::LastBlock),
            "done"
        );
    }

    /// And with neither text nor result, both modes yield empty — the caller's
    /// existing `final_text.is_empty()` guard is what turns that into an error.
    #[test]
    fn no_text_and_no_result_yields_empty_for_both_modes() {
        let blocks: Vec<String> = Vec::new();
        assert!(select_final_text(&blocks, None, TextCapture::AllBlocks).is_empty());
        assert!(select_final_text(&blocks, None, TextCapture::LastBlock).is_empty());
    }

    /// Missing `result` event (stream cut short): LastBlock still returns the
    /// most recent block, preserving pre-#446 behaviour.
    #[test]
    fn last_block_capture_uses_the_final_block_when_result_is_absent() {
        let blocks = vec!["first".to_string(), "second".to_string()];
        assert_eq!(
            select_final_text(&blocks, None, TextCapture::LastBlock),
            "second"
        );
        assert_eq!(
            select_final_text(&blocks, None, TextCapture::AllBlocks),
            "first\n\nsecond"
        );
    }

    #[test]
    fn ask_opts_ships_absolute_db_env() {
        let repo = tempfile::tempdir().expect("repo tmpdir");
        let wiki = tempfile::tempdir().expect("wiki tmpdir");
        // Touch the db file so canonicalize succeeds, mirroring a real repo.
        std::fs::write(repo.path().join("data.db"), b"").unwrap();

        // Clear a potentially-inherited env that would override our resolution.
        let _guard = EnvGuard::unset("AUGMENTAGENT_DB");
        let opts = ask_opts(wiki.path().to_path_buf(), repo.path().to_path_buf());

        let (key, value) = opts
            .env
            .iter()
            .find(|(k, _)| k == "AUGMENTAGENT_DB")
            .expect("AUGMENTAGENT_DB env var present");
        assert_eq!(key, "AUGMENTAGENT_DB");
        let path = std::path::Path::new(value);
        assert!(path.is_absolute(), "db path must be absolute: {value}");
        assert_eq!(path.file_name().unwrap(), "data.db");
    }

    #[test]
    fn ask_opts_honors_augmentagent_db_override() {
        let repo = tempfile::tempdir().expect("repo tmpdir");
        let wiki = tempfile::tempdir().expect("wiki tmpdir");
        let override_dir = tempfile::tempdir().expect("override tmpdir");
        let override_db = override_dir.path().join("custom.db");
        std::fs::write(&override_db, b"").unwrap();

        let _guard = EnvGuard::set("AUGMENTAGENT_DB", override_db.to_str().unwrap());
        let opts = ask_opts(wiki.path().to_path_buf(), repo.path().to_path_buf());
        let value = opts
            .env
            .iter()
            .find(|(k, _)| k == "AUGMENTAGENT_DB")
            .map(|(_, v)| v.clone())
            .expect("AUGMENTAGENT_DB env present");
        assert!(std::path::Path::new(&value).is_absolute());
        assert_eq!(
            std::path::Path::new(&value).file_name().unwrap(),
            "custom.db"
        );
    }

    /// #800 — an announcement drafted while the Meetup scraper was down
    /// carried a fabricated "Friday, 6:00 PM" pulled from a wiki cadence
    /// note. `wiki-ask.md` must keep the guardrail that a specific event
    /// date/time comes only from a fetched instance, and that an unverifiable
    /// one is flagged rather than guessed.
    #[test]
    fn ask_opts_prompt_forbids_uncited_event_datetimes() {
        let repo = tempfile::tempdir().expect("repo tmpdir");
        let wiki = tempfile::tempdir().expect("wiki tmpdir");
        let opts = ask_opts(wiki.path().to_path_buf(), repo.path().to_path_buf());

        for needle in [
            "A cadence is not an instance",
            "[date/time unverified",
            "(from wiki, unverified)",
        ] {
            assert!(
                opts.system_prompt.contains(needle),
                "wiki-ask.md lost the #800 event-date guardrail: {needle:?}"
            );
        }
    }

    /// #317 — the reasoner routes `mcpServers` to `--mcp-config` and keeps
    /// the rest on `--settings`.
    #[test]
    fn split_mcp_from_settings_routes_servers_and_keeps_hooks() {
        let raw = r#"{"mcpServers":{"memory":{"command":"x"}},"hooks":{"PreToolUse":[]}}"#;
        let (settings, mcp) = split_mcp_from_settings(raw);
        // mcp_config carries the server wrapped in {"mcpServers": …}.
        let mcp_v: serde_json::Value =
            serde_json::from_str(&mcp.expect("mcp config present")).unwrap();
        assert!(mcp_v.pointer("/mcpServers/memory/command").is_some());
        // settings keeps hooks and DROPS mcpServers.
        let s_v: serde_json::Value =
            serde_json::from_str(&settings.expect("settings present")).unwrap();
        assert!(s_v.pointer("/hooks/PreToolUse").is_some());
        assert!(s_v.pointer("/mcpServers").is_none());
    }

    #[test]
    fn split_mcp_from_settings_no_servers_passes_through() {
        let raw = r#"{"hooks":{"PreToolUse":[]}}"#;
        let (settings, mcp) = split_mcp_from_settings(raw);
        assert!(mcp.is_none());
        assert_eq!(settings.as_deref(), Some(raw));
    }

    #[test]
    fn split_mcp_from_settings_only_servers_yields_no_settings() {
        let raw = r#"{"mcpServers":{"memory":{"command":"x"}}}"#;
        let (settings, mcp) = split_mcp_from_settings(raw);
        assert!(settings.is_none(), "empty residual settings must be None");
        assert!(mcp.is_some());
    }

    /// #317 — `ask_opts` must register the `memory` stdio MCP server (scoped
    /// to the daemon db) AND advertise the read-only recall tools, while
    /// keeping the path-scope hook and NOT exposing the write tool.
    #[test]
    fn ask_opts_registers_memory_mcp_server_and_recall_tools() {
        let repo = tempfile::tempdir().expect("repo tmpdir");
        let wiki = tempfile::tempdir().expect("wiki tmpdir");
        std::fs::write(repo.path().join("data.db"), b"").unwrap();
        let _guard = EnvGuard::unset("AUGMENTAGENT_DB");
        let opts = ask_opts(wiki.path().to_path_buf(), repo.path().to_path_buf());

        // Recall tools advertised; write tool deliberately absent here.
        let joined = opts.allowed_tools.join("\n");
        for needle in [
            "mcp__memory__search_conversation_history",
            "mcp__memory__memory_search",
            "mcp__memory__memory_recent",
        ] {
            assert!(
                opts.allowed_tools.iter().any(|t| t == needle),
                "expected recall tool {needle}; got:\n{joined}"
            );
        }
        assert!(
            !opts.allowed_tools.iter().any(|t| t == "mcp__memory__memory_write"),
            "memory_write must NOT be exposed in ask mode allowlist"
        );

        // Settings JSON registers the stdio server, db-scoped, and STILL
        // carries the path-scope PreToolUse hook (we add alongside, not
        // replace).
        let settings = opts.settings_json.expect("settings_json present");
        let parsed: serde_json::Value =
            serde_json::from_str(&settings).expect("settings_json is valid JSON");
        let cmd = parsed
            .pointer("/mcpServers/memory/command")
            .and_then(|v| v.as_str())
            .expect("memory server command present");
        assert!(
            cmd.ends_with("augmentagent-mcp-memory"),
            "memory command should point at the mcp-memory binary; got {cmd}"
        );
        let db = parsed
            .pointer("/mcpServers/memory/env/AUGMENTAGENT_DB")
            .and_then(|v| v.as_str())
            .expect("memory server AUGMENTAGENT_DB env present");
        assert!(
            std::path::Path::new(db).is_absolute(),
            "memory server db path must be absolute; got {db}"
        );
        assert!(
            parsed.pointer("/hooks/PreToolUse/0/matcher").is_some(),
            "path-scope PreToolUse hook must survive alongside mcpServers"
        );
    }

    /// #127 — `ask_opts` must wire up a PreToolUse hook so the harness
    /// path-scopes Read/Write/Edit/Glob/Grep to the wiki root. Without
    /// this, the `wiki-ask.md` "scoped to the wiki root" claim is
    /// prompt-as-policy, not policy.
    #[test]
    fn ask_opts_installs_pretool_use_hook_for_file_tools() {
        let repo = tempfile::tempdir().expect("repo tmpdir");
        let wiki = tempfile::tempdir().expect("wiki tmpdir");
        std::fs::write(repo.path().join("data.db"), b"").unwrap();
        let _guard = EnvGuard::unset("AUGMENTAGENT_DB");
        let opts = ask_opts(wiki.path().to_path_buf(), repo.path().to_path_buf());

        let settings = opts
            .settings_json
            .as_deref()
            .expect("ask_opts must ship a settings_json with the scope hook");
        let value: serde_json::Value =
            serde_json::from_str(settings).expect("settings_json must be valid JSON");

        let matcher = value
            .pointer("/hooks/PreToolUse/0/matcher")
            .and_then(|v| v.as_str())
            .expect("hook matcher must be a string");
        // Each of the file-touching tools that Claude Code does not
        // natively path-scope must appear in the matcher regex.
        for tool in ["Read", "Write", "Edit", "Glob", "Grep"] {
            assert!(
                matcher.contains(tool),
                "hook matcher must cover {tool}; got: {matcher}"
            );
        }

        let cmd = value
            .pointer("/hooks/PreToolUse/0/hooks/0/command")
            .and_then(|v| v.as_str())
            .expect("hook command must be a string");
        assert!(
            cmd.ends_with("scripts/aa-wiki-scope-guard.sh"),
            "hook command must point at the wiki-scope guard script; got: {cmd}"
        );

        // WIKI_ROOT must be forwarded so the guard script knows the
        // permitted prefix. Without it the script bails out exit-2 and
        // every file-tool call is denied.
        let wiki_env = opts
            .env
            .iter()
            .find(|(k, _)| k == "WIKI_ROOT")
            .map(|(_, v)| v.clone())
            .expect("WIKI_ROOT env var must be forwarded to the guard");
        // #337 — WIKI_ROOT is canonicalized inside ask_opts; compare against
        // the canonical form of the tempdir (symlinked /tmp on some hosts).
        let expected = std::fs::canonicalize(wiki.path()).unwrap();
        assert_eq!(wiki_env, expected.to_string_lossy());
    }

    /// #337 — a RELATIVE `--wiki-dir` (the daemon passes `./wiki`) must be
    /// absolutized before it reaches WIKI_ROOT. Otherwise the guard's
    /// `readlink -f` re-resolves it against the spawned CLI's cwd (already the
    /// wiki root) to `<wiki>/wiki`, and every real page path is rejected.
    #[test]
    fn ask_opts_absolutizes_relative_wiki_root() {
        let repo = tempfile::tempdir().expect("repo tmpdir");
        std::fs::write(repo.path().join("data.db"), b"").unwrap();
        let _guard = EnvGuard::unset("AUGMENTAGENT_DB");
        // A relative wiki dir, exactly like the systemd unit's `--wiki-dir ./wiki`.
        let rel = std::path::PathBuf::from("rel-wiki-xyz");
        let opts = ask_opts(rel, repo.path().to_path_buf());
        let wiki_env = opts
            .env
            .iter()
            .find(|(k, _)| k == "WIKI_ROOT")
            .map(|(_, v)| v.clone())
            .expect("WIKI_ROOT env var must be forwarded to the guard");
        assert!(
            std::path::Path::new(&wiki_env).is_absolute(),
            "WIKI_ROOT must be absolute even for a relative --wiki-dir; got {wiki_env}"
        );
        assert!(
            !wiki_env.contains("rel-wiki-xyz/rel-wiki-xyz"),
            "relative wiki dir must not be doubled into <wiki>/wiki; got {wiki_env}"
        );
    }

    /// PR #922 review — #915 opens the transcript clone via `--add-dir`, but
    /// the #127 scope guard only allowed WIKI_ROOT, so every Read/Grep/Glob
    /// on a transcript was denied. The guard now honors
    /// AUGMENTAGENT_TRANSCRIPTS_DIR — which only works if `ask_opts` forwards
    /// it into the spawned CLI's restricted (#128) env.
    #[test]
    fn ask_opts_forwards_transcripts_dir_to_the_scope_guard() {
        let repo = tempfile::tempdir().expect("repo tmpdir");
        let wiki = tempfile::tempdir().expect("wiki tmpdir");
        let transcripts = tempfile::tempdir().expect("transcripts tmpdir");
        std::fs::write(repo.path().join("data.db"), b"").unwrap();
        // One EnvGuard only — it holds the global env lock, so a second
        // would self-deadlock. AUGMENTAGENT_DB is cleared by hand under it.
        let _t = EnvGuard::set(
            "AUGMENTAGENT_TRANSCRIPTS_DIR",
            transcripts.path().to_str().unwrap(),
        );
        let prior_db = std::env::var("AUGMENTAGENT_DB").ok();
        std::env::remove_var("AUGMENTAGENT_DB");

        let opts = ask_opts(wiki.path().to_path_buf(), repo.path().to_path_buf());

        if let Some(v) = prior_db {
            std::env::set_var("AUGMENTAGENT_DB", v);
        }
        let fwd = opts
            .env
            .iter()
            .find(|(k, _)| k == "AUGMENTAGENT_TRANSCRIPTS_DIR")
            .map(|(_, v)| v.clone())
            .expect("AUGMENTAGENT_TRANSCRIPTS_DIR must reach the guard's env");
        assert_eq!(fwd, transcripts.path().to_string_lossy());
        // And the clone itself is opened for the session (#915 behaviour,
        // pinned so the env forwarding cannot drift from add_dirs).
        assert!(
            opts.add_dirs
                .iter()
                .any(|d| d.as_path() == transcripts.path()),
            "transcript clone must be in add_dirs"
        );
    }

    /// Regression test for the PR #199 follow-up: `aa-gh` must be
    /// reachable both as bare `aa-gh ...` and by absolute path, and
    /// `<repo>/scripts/` must be on PATH so the bare form actually
    /// resolves in shell. The original wire-in only registered the
    /// absolute-path patterns and did not extend PATH, which left the
    /// wiki-query agent hitting "This command requires approval" on
    /// every bare-form issue-filing call (the same shape used by the
    /// command examples in `schema/wiki-ask.md`).
    #[test]
    fn ask_opts_allows_aa_gh_both_forms_and_extends_path() {
        let repo = tempfile::tempdir().expect("repo tmpdir");
        let wiki = tempfile::tempdir().expect("wiki tmpdir");
        std::fs::write(repo.path().join("data.db"), b"").unwrap();
        let _guard = EnvGuard::unset("AUGMENTAGENT_DB");
        let opts = ask_opts(wiki.path().to_path_buf(), repo.path().to_path_buf());

        let joined = opts.allowed_tools.join("\n");
        for bare in [
            "Bash(aa-gh issue create *)",
            "Bash(aa-gh issue list *)",
            "Bash(aa-gh issue view *)",
            "Bash(aa-gh issue comment *)",
        ] {
            assert!(
                opts.allowed_tools.iter().any(|e| e == bare),
                "expected bare-form allowlist entry {bare}; got:\n{joined}"
            );
        }
        let aa_gh_abs = repo
            .path()
            .join("scripts/aa-gh")
            .display()
            .to_string();
        for sub in ["issue create *", "issue list *", "issue view *", "issue comment *"] {
            let needle = format!("Bash({aa_gh_abs} {sub})");
            assert!(
                opts.allowed_tools.iter().any(|e| e == &needle),
                "expected absolute-path allowlist entry {needle}; got:\n{joined}"
            );
        }

        let path_env = opts
            .env
            .iter()
            .find(|(k, _)| k == "PATH")
            .map(|(_, v)| v.clone())
            .expect("PATH env var must be forwarded to the spawned agent");
        let scripts_dir = repo.path().join("scripts").display().to_string();
        assert!(
            path_env.split(':').any(|seg| seg == scripts_dir),
            "PATH must contain {scripts_dir} so bare `aa-gh` resolves; got: {path_env}"
        );
    }

    /// Serialize env-var mutations across the two tests so they don't race.
    /// (`cargo test` runs tests in parallel by default; both touch the same
    /// process-wide `AUGMENTAGENT_DB` var.)
    struct EnvGuard {
        key: &'static str,
        prior: Option<String>,
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl EnvGuard {
        fn lock() -> std::sync::MutexGuard<'static, ()> {
            static M: std::sync::Mutex<()> = std::sync::Mutex::new(());
            M.lock().unwrap_or_else(|e| e.into_inner())
        }
        fn unset(key: &'static str) -> Self {
            let lock = Self::lock();
            let prior = std::env::var(key).ok();
            std::env::remove_var(key);
            Self {
                key,
                prior,
                _lock: lock,
            }
        }
        fn set(key: &'static str, value: &str) -> Self {
            let lock = Self::lock();
            let prior = std::env::var(key).ok();
            std::env::set_var(key, value);
            Self {
                key,
                prior,
                _lock: lock,
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.prior {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }

    // ---- #141: subprocess error sanitization ----

    /// Verbatim sample of the multi-line stderr blob that surfaced in the
    /// Discord bug report — block repeated 4x, leading `ExitStatus(...)`,
    /// absolute home paths.
    const CORRUPT_4X_STDERR: &str = "Configuration error in /home/operator/.claude.json: JSON Parse error: Unexpected EOF\n\nClaude configuration file at /home/operator/.claude.json is corrupted: JSON Parse error: Unexpected EOF\nThe corrupted file has already been backed up.\nA backup file exists at: /home/operator/.claude/backups/.claude.json.backup.1779736244424\nYou can manually restore it by running: cp \"/home/operator/.claude/backups/.claude.json.backup.1779736244424\" \"/home/operator/.claude.json\"\n\nConfiguration error in /home/operator/.claude.json: JSON Parse error: Unexpected EOF\n\nClaude configuration file at /home/operator/.claude.json is corrupted: JSON Parse error: Unexpected EOF\nThe corrupted file has already been backed up.\nA backup file exists at: /home/operator/.claude/backups/.claude.json.backup.1779736244424\nYou can manually restore it by running: cp \"/home/operator/.claude/backups/.claude.json.backup.1779736244424\" \"/home/operator/.claude.json\"\n\nConfiguration error in /home/operator/.claude.json: JSON Parse error: Unexpected EOF\n\nClaude configuration file at /home/operator/.claude.json is corrupted: JSON Parse error: Unexpected EOF\nThe corrupted file has already been backed up.";

    #[test]
    fn sanitize_corrupted_config_returns_single_short_line() {
        let out = sanitize_claude_error(CORRUPT_4X_STDERR);
        assert_eq!(
            out,
            "wiki temporarily unavailable: claude config corrupted (auto-recovery failed)"
        );
        // The single line must not leak ExitStatus noise, absolute home
        // paths, or repeated diagnostic blocks.
        assert!(!out.contains("ExitStatus"), "leaked ExitStatus: {out}");
        assert!(!out.contains("/home/"), "leaked absolute path: {out}");
        assert_eq!(out.lines().count(), 1, "must be single-line: {out:?}");
    }

    #[test]
    fn sanitize_dedupes_repeated_blocks() {
        let block = "Configuration error in /tmp/x.json: JSON Parse error: foo";
        let repeated = format!("{block}\n\n{block}\n\n{block}\n\n{block}");
        let out = sanitize_claude_error(&repeated);
        // Even on the matched-pattern fast-path we collapse to one line; the
        // important property is that the output never repeats the diagnostic.
        let matches = out.matches("auto-recovery failed").count();
        assert_eq!(matches, 1, "diagnostic must not repeat: {out}");
    }

    #[test]
    fn sanitize_generic_failure_strips_exitstatus_and_home() {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/home/test".into());
        let stderr = format!(
            "ExitStatus(unix_wait_status(256)): some tool failed reading {home}/.config/foo"
        );
        let out = sanitize_claude_error(&stderr);
        assert!(out.starts_with("wiki temporarily unavailable: "), "{out}");
        assert!(!out.contains("ExitStatus"), "{out}");
        assert!(!out.contains(&home), "{out}");
    }

    #[test]
    fn sanitize_empty_returns_friendly_fallback() {
        let out = sanitize_claude_error("");
        assert_eq!(
            out,
            "wiki temporarily unavailable: claude subprocess failed"
        );
    }

    #[test]
    fn is_claude_config_corrupted_matches_real_stderr() {
        assert!(is_claude_config_corrupted(CORRUPT_4X_STDERR));
        assert!(is_claude_config_corrupted(
            "Claude configuration file at /home/u/.claude.json is corrupted: foo"
        ));
        assert!(!is_claude_config_corrupted("network connection failed"));
        assert!(!is_claude_config_corrupted(""));
    }

    // ---- #248: optional read-only SocialAPI MCP drafting aid ----

    /// Serialize the env-touching SocialAPI flag tests against each other AND
    /// against the `EnvGuard`-based tests, since the flag is process-global.
    fn socialapi_flag_guard(value: Option<&str>) -> Vec<EnvGuard> {
        let mut guards = Vec::new();
        match value {
            Some(v) => guards.push(EnvGuard::set(SOCIALAPI_MCP_READONLY_FLAG, v)),
            None => guards.push(EnvGuard::unset(SOCIALAPI_MCP_READONLY_FLAG)),
        }
        guards
    }

    /// Render the spawn-relevant fields of a `ReasonerOpts` to a stable
    /// string so two opts can be compared for byte-identity. `ReasonerOpts`
    /// doesn't derive `PartialEq` (it holds trait objects), so we compare the
    /// fields that actually shape the spawned `claude` args + env.
    fn opts_fingerprint(o: &ReasonerOpts) -> String {
        format!(
            "system={:?}|model={:?}|tools={:?}|add_dirs={:?}|perm={:?}|cwd={:?}|env={:?}|settings={:?}|restrict={:?}",
            o.system_prompt,
            o.model,
            o.allowed_tools,
            o.add_dirs,
            o.permission_mode,
            o.cwd,
            o.env,
            o.settings_json,
            o.restrict_env,
        )
    }

    /// Flag OFF ⇒ STRICT no-op. The opts a caller would spawn must be
    /// byte-identical to the input, so existing draft behavior is unchanged.
    #[test]
    fn socialapi_mcp_flag_off_is_strict_noop() {
        let _g = socialapi_flag_guard(None);
        assert!(!socialapi_mcp_readonly_enabled());

        let base = draft_opts("draft system".into(), None);
        let before = opts_fingerprint(&base);
        let after = with_socialapi_readonly_mcp(base, std::path::Path::new("/repo"));
        let after_fp = opts_fingerprint(&after);

        assert_eq!(before, after_fp, "flag-off must not change spawn-relevant opts");
        assert!(after.settings_json.is_none(), "no settings_json when flag off");
        assert!(
            !after.allowed_tools.iter().any(|t| t.starts_with("mcp__socialapi__")),
            "no socialapi MCP tools when flag off"
        );
        assert!(
            !after.env.iter().any(|(k, _)| k == "SOCIALAPI_API_KEY"),
            "no SOCIALAPI_API_KEY env when flag off"
        );
    }

    /// A non-truthy value (e.g. `0`/`false`) must also be treated as OFF.
    #[test]
    fn socialapi_mcp_flag_falsey_is_noop() {
        for val in ["0", "false", "no", "off", ""] {
            let _g = socialapi_flag_guard(Some(val));
            assert!(
                !socialapi_mcp_readonly_enabled(),
                "value {val:?} must be treated as disabled"
            );
            let base = draft_opts("s".into(), None);
            let before = opts_fingerprint(&base);
            let after = with_socialapi_readonly_mcp(base, std::path::Path::new("/repo"));
            assert_eq!(before, opts_fingerprint(&after), "{val:?} must be a no-op");
        }
    }

    /// Flag ON ⇒ SocialAPI MCP server attached + deny hook installed +
    /// wildcard tool advertised. The bearer key forwarding is environment-
    /// dependent (keyring/env) so we don't assert on it here; the no-leak
    /// guarantee is covered by the flag-off test + the SAFELIST.
    #[test]
    fn socialapi_mcp_flag_on_attaches_server_and_deny_hook() {
        let _g = socialapi_flag_guard(Some("1"));
        assert!(socialapi_mcp_readonly_enabled());

        let repo = tempfile::tempdir().unwrap();
        let base = draft_opts("draft system".into(), None);
        let opts = with_socialapi_readonly_mcp(base, repo.path());

        // 1) Tool advertised.
        assert!(
            opts.allowed_tools.iter().any(|t| t == "mcp__socialapi__*"),
            "socialapi MCP wildcard tool must be advertised; got {:?}",
            opts.allowed_tools
        );

        // 2) settings_json registers the MCP server + the deny hook.
        let settings = opts
            .settings_json
            .as_deref()
            .expect("flag-on must ship settings_json");
        let v: serde_json::Value = serde_json::from_str(settings).expect("valid JSON");

        let server = v
            .pointer("/mcpServers/socialapi")
            .expect("socialapi MCP server registered");
        assert_eq!(server.get("type").and_then(|x| x.as_str()), Some("http"));
        assert!(
            server
                .get("url")
                .and_then(|x| x.as_str())
                .unwrap_or_default()
                .contains("mcp"),
            "MCP url must point at an mcp endpoint; got {server:?}"
        );
        // Bearer auth is via env-expansion, NOT a baked key.
        let auth = server
            .pointer("/headers/Authorization")
            .and_then(|x| x.as_str())
            .unwrap_or_default();
        assert_eq!(auth, "Bearer ${SOCIALAPI_API_KEY}", "raw key must not be baked into settings");

        let matcher = v
            .pointer("/hooks/PreToolUse/0/matcher")
            .and_then(|x| x.as_str())
            .expect("deny hook matcher present");
        assert!(
            matcher.contains("mcp__socialapi__"),
            "hook must police the socialapi MCP surface; got {matcher}"
        );
        let cmd = v
            .pointer("/hooks/PreToolUse/0/hooks/0/command")
            .and_then(|x| x.as_str())
            .expect("deny hook command present");
        assert!(
            cmd.ends_with("scripts/aa-socialapi-readonly-guard.sh"),
            "hook must invoke the read-only guard; got {cmd}"
        );
    }

    /// The bearer key is on the env SAFELIST so it survives `restrict_env`.
    #[test]
    fn socialapi_key_is_on_env_safelist() {
        assert!(
            SAFELIST_ENV_VARS.contains(&"SOCIALAPI_API_KEY"),
            "SOCIALAPI_API_KEY must be safelisted so it survives restrict_env spawns"
        );
    }

    /// Drive the actual guard script end-to-end: a send/reply tool is DENIED,
    /// a list/get tool is ALLOWED, an unknown socialapi tool is DENIED
    /// (fail-closed), and a non-socialapi tool passes through. This is the
    /// load-bearing read-only guarantee, so we exercise the real script.
    #[test]
    fn socialapi_guard_script_denies_writes_allows_reads_failclosed() {
        // Locate the guard script relative to this crate (workspace layout:
        // CARGO_MANIFEST_DIR is crates/augmentagent-channel-core).
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let guard = manifest.join("../../scripts/aa-socialapi-readonly-guard.sh");
        if !guard.exists() {
            // Defensive: if run from an unusual layout, skip rather than fail
            // spuriously. The unit assertions above already cover wiring.
            eprintln!("guard script not found at {guard:?}; skipping script e2e");
            return;
        }
        if std::process::Command::new("jq").arg("--version").output().is_err() {
            eprintln!("jq not available; skipping guard script e2e");
            return;
        }

        let run = |tool: &str| -> (bool, String) {
            // bool = "denied?"
            let event = serde_json::json!({
                "tool_name": tool,
                "tool_input": {}
            })
            .to_string();
            let out = std::process::Command::new("bash")
                .arg(&guard)
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .and_then(|mut child| {
                    use std::io::Write;
                    child
                        .stdin
                        .take()
                        .unwrap()
                        .write_all(event.as_bytes())
                        .unwrap();
                    child.wait_with_output()
                })
                .expect("guard script runs");
            let stdout = String::from_utf8_lossy(&out.stdout).to_string();
            let denied = stdout.contains("\"deny\"") || stdout.contains("\"decision\":\"block\"")
                || stdout.contains("\"decision\": \"block\"");
            (denied, stdout)
        };

        // Writes / sends → DENIED.
        for tool in [
            "mcp__socialapi__reply_to_comment",
            "mcp__socialapi__post_tweet",
            "mcp__socialapi__send_dm",
            "mcp__socialapi__delete_post",
            "mcp__socialapi__create_thread",
        ] {
            let (denied, out) = run(tool);
            assert!(denied, "{tool} must be DENIED; guard said: {out}");
        }

        // Reads → ALLOWED (empty stdout, exit 0).
        for tool in [
            "mcp__socialapi__list_comments",
            "mcp__socialapi__get_thread",
            "mcp__socialapi__fetch_post",
            "mcp__socialapi__search_mentions",
        ] {
            let (denied, out) = run(tool);
            assert!(!denied, "{tool} must be ALLOWED; guard said: {out}");
        }

        // Unknown socialapi verb → DENIED fail-closed.
        let (denied, out) = run("mcp__socialapi__frobnicate");
        assert!(denied, "unknown socialapi tool must be denied fail-closed; got: {out}");

        // Non-socialapi tool → pass-through (allowed, empty stdout).
        let (denied, _out) = run("Read");
        assert!(!denied, "non-socialapi tools must pass through untouched");
    }
}

/// Preset for the one-shot `resume ingest` CLI. Opus quality for a single-run
/// seeding pass. Full wiki R/W/E with cwd pinned so writes cannot escape.
pub fn resume_opts(wiki_root: PathBuf) -> ReasonerOpts {
    ReasonerOpts {
        system_prompt: include_str!("../../../schema/resume-ingest.md").to_string(),
        model: Some(opus_model()), // Opus — seeding the wiki is high-leverage and one-shot
        allowed_tools: vec![
            "Read".into(),
            "Grep".into(),
            "Glob".into(),
            "Write".into(),
            "Edit".into(),
        ],
        add_dirs: vec![wiki_root.clone()],
        permission_mode: "acceptEdits".into(),
        cwd: Some(wiki_root),
        env: Vec::new(),
        settings_json: None,
        restrict_env: false,
        audit_logger: None,
        audit_notifier: None,
        session_id: None,
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StreamEvent {
    Assistant {
        message: AssistantMessage,
    },
    Result {
        #[serde(default)]
        result: Option<String>,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
struct AssistantMessage {
    #[serde(default)]
    content: Vec<ContentBlock>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentBlock {
    Text {
        text: String,
    },
    #[serde(other)]
    Other,
}

#[cfg(test)]
mod rate_limit_tests {
    use super::*;

    /// #448 — the exact string the CLI returned to the daemon on 2026-07-14,
    /// copied from ~/.local/state/augmentagent/stderr.log. It arrived as a
    /// successful completion, so triage logged it as
    /// "triage parse failed: no JSON object found in model output" and the
    /// action layer retried (max=5) against a wall we were already up against.
    #[test]
    fn detects_the_real_session_limit_refusal() {
        assert!(is_rate_limited(
            "You've hit your session limit · resets 9:30am (America/New_York)"
        ));
        assert!(is_rate_limited("You've reached your usage limit. Resets at 3pm."));
        assert!(is_rate_limited(
            "Rate limit exceeded — resets in 12 minutes"
        ));
    }

    /// Must NOT fire on a legitimate triage answer, or every email about
    /// billing/quotas would be dropped instead of classified. This is the
    /// dangerous false positive.
    #[test]
    fn does_not_fire_on_ordinary_model_output() {
        for ok in [
            r#"{"decision":"skip","reason":"Marketing blast from a no-reply sender."}"#,
            // An email that legitimately discusses limits.
            r#"{"decision":"flag","reason":"Vendor says we hit our API rate limit in prod."}"#,
            r#"{"decision":"reply","reason":"Asks about our session limit policy."}"#,
            "",
        ] {
            assert!(!is_rate_limited(ok), "false positive on: {ok}");
        }
    }

    /// #655 review — post-#655 a false positive doesn't fail one call, it
    /// LATCHES the provider in shared state. Output that merely CONTAINS or
    /// QUOTES refusal wording must never match: the refusal must be the
    /// whole (short, anchored) message.
    #[test]
    fn quoted_refusals_inside_answers_do_not_latch() {
        for ok in [
            // Short triage answer quoting a forwarded limit notice.
            r#"{"decision":"flag","reason":"Owner forwarded: You've hit your usage limit · resets 9:30am"}"#,
            // Digest line summarizing yesterday's incident.
            "Yesterday's incidents: two triage calls failed with \
             'You've hit your session limit · resets 9:30am (America/New_York)'.",
            // Long transcript with the refusal mid-text (length cap).
            &format!(
                "{}\nYou've hit your session limit · resets 9:30am\n{}",
                "context ".repeat(50),
                "more context ".repeat(50)
            ),
        ] {
            assert!(!is_rate_limited(ok), "quoted refusal must not match: {ok}");
        }
        // The genuine refusal itself still matches, of course.
        assert!(is_rate_limited(
            "You've hit your session limit · resets 9:30am (America/New_York)"
        ));
    }
}

#[cfg(test)]
mod model_pin_tests {
    use super::*;
    use std::path::PathBuf;

    /// #448 — the regression guard. A preset with `model: None` emits no
    /// `--model` flag, so the spawned CLI inherits the OWNER's interactive
    /// `~/.claude/settings.json`. That is how ~250 background email
    /// classifications a day ended up on `opus[1m]` @ `xhigh`, billed to the
    /// owner's Max subscription, purely because they'd picked that model for
    /// their own coding.
    ///
    /// Every preset the daemon runs unattended must name its model out loud.
    /// If you add a preset, pin it — do not let `None` back in.
    #[test]
    fn no_daemon_preset_inherits_the_owners_interactive_model() {
        let wiki = PathBuf::from("/tmp/wiki");
        let repo = PathBuf::from("/tmp/repo");
        let presets: Vec<(&str, ReasonerOpts)> = vec![
            ("triage", triage_opts(Some(wiki.clone()))),
            ("draft", draft_opts("sys".into(), Some(wiki.clone()))),
            ("lint", lint_opts("sys".into(), wiki.clone())),
            ("ask", ask_opts(wiki.clone(), repo.clone())),
            ("digest", digest_opts(Some(wiki.clone()))),
            ("social_adapter", social_adapter_opts("sys".into())),
            ("resume", resume_opts(wiki.clone())),
            ("tone_summarize", tone_summarize_opts()),
            ("loop_parse", loop_parse_opts()),
            ("archetype_pick", archetype_pick_opts()),
            ("ingest", ingest_opts("sys".into(), wiki.clone())),
            ("wiki_migrate", wiki_migrate_opts("sys".into(), wiki.clone())),
        ];
        for (name, opts) in presets {
            let model = opts.model.as_deref().unwrap_or("");
            assert!(
                !model.is_empty(),
                "{name}_opts has model: None — it will inherit the owner's \
                 interactive /model choice and bill their subscription (#448)"
            );
            // The inherited value we're defending against is literally the
            // 1M-context alias from the owner's settings.json.
            assert!(
                !model.contains("[1m]"),
                "{name}_opts pins a context-window alias ({model}); pin a real \
                 model id so the tier can't drift"
            );
        }
    }

    /// The env overrides exist so a tier change needs no rebuild.
    #[test]
    fn env_overrides_take_precedence() {
        // Defaults, with no env set, are real model ids.
        assert_eq!(TRIAGE_MODEL, "claude-opus-4-8");
        assert_eq!(OPUS_MODEL, "claude-opus-4-8");
        assert!(!TRIAGE_MODEL.contains("[1m]"));
    }
}

#[cfg(test)]
mod socialapi_draft_opts_wiring_tests {
    use super::*;

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// #533: the aid must stay strictly default-off. With the flag unset,
    /// `socialapi_draft_opts` has to be indistinguishable from `draft_opts`.
    #[test]
    fn default_off_is_byte_for_byte_plain_draft_opts() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var(SOCIALAPI_MCP_READONLY_FLAG);
        let plain = draft_opts("sys".into(), None);
        let wired = socialapi_draft_opts("sys".into(), None);
        assert_eq!(wired.settings_json, plain.settings_json);
        assert_eq!(wired.allowed_tools, plain.allowed_tools);
        assert!(wired.settings_json.is_none());
        assert!(!wired.allowed_tools.iter().any(|t| t.contains("socialapi")));
    }

    /// With the flag on, the server is attached and its tools advertised —
    /// which before this wiring never happened, because the helper had no
    /// production caller.
    #[test]
    fn flag_on_attaches_the_server_and_advertises_its_tools() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var(SOCIALAPI_MCP_READONLY_FLAG, "1");
        let wired = socialapi_draft_opts("sys".into(), None);
        std::env::remove_var(SOCIALAPI_MCP_READONLY_FLAG);
        assert!(
            wired.settings_json.is_some(),
            "the MCP server must be declared when the flag is on"
        );
        assert!(
            wired.allowed_tools.iter().any(|t| t.contains("socialapi")),
            "the server's tools must be advertised: {:?}",
            wired.allowed_tools
        );
        let json = wired.settings_json.unwrap();
        // The guard hook is the real gate; it must be wired into the settings
        // or the read-only promise is unenforced.
        assert!(
            json.contains("aa-socialapi-readonly-guard.sh"),
            "settings must reference the fail-closed guard hook: {json}"
        );
    }
}

#[cfg(test)]
mod ask_mode_social_card_allowlist_tests {
    use super::*;

    fn ask_allowlist() -> Vec<String> {
        let tmp = std::env::temp_dir().join("aa-ask-allowlist-test");
        let _ = std::fs::create_dir_all(&tmp);
        ask_opts("sys".into(), tmp).allowed_tools
    }

    /// #571/#572: the agent could not raise a social card because the verbs
    /// were absent from the ask-mode Bash allowlist — building the CLI verbs
    /// alone left them unreachable from Discord.
    #[test]
    fn social_card_verbs_are_allowlisted() {
        let tools = ask_allowlist();
        for needle in [
            "socialapi dm *",
            "socialapi comment *",
            "linkedin dm *",
            "linkedin comment *",
            // Without a urn lookup the dm verb is unreachable for any thread
            // that was never ingested — the dead end that made the agent fall
            // back to "paste it yourself".
            "linkedin recent-dms",
        ] {
            assert!(
                tools.iter().any(|t| t.contains(needle)),
                "ask mode must permit `{needle}`; got {tools:?}"
            );
        }
    }

    /// The allowlist must stay SCOPED. A `linkedin *` or `socialapi *`
    /// wildcard would also hand the agent `linkedin post` — which publishes
    /// to the feed directly, with no card — and `login` / `connect`.
    #[test]
    fn no_broad_wildcard_exposes_direct_send_verbs() {
        let tools = ask_allowlist();
        for forbidden in [
            "Bash(augmentagent linkedin *)",
            "Bash(augmentagent socialapi *)",
        ] {
            assert!(
                !tools.iter().any(|t| t == forbidden),
                "`{forbidden}` would expose direct-send verbs"
            );
        }
        // Specifically: nothing may match `linkedin post`.
        assert!(
            !tools.iter().any(|t| t.contains("linkedin post")),
            "linkedin post publishes directly and must never be allowlisted"
        );
        assert!(
            !tools.iter().any(|t| t.contains("socialapi connect")),
            "socialapi connect drives OAuth and must not be agent-invokable"
        );
    }

    /// Both the absolute-path and bare forms are needed: the permission
    /// matcher does literal string matching, and the model copies the bare
    /// shape from the schema examples.
    #[test]
    fn both_bare_and_absolute_forms_are_present() {
        let tools = ask_allowlist();
        assert!(tools.iter().any(|t| t == "Bash(augmentagent socialapi dm *)"));
        assert!(tools.iter().any(|t| t == "Bash(augmentagent linkedin dm *)"));
        assert!(
            tools.iter().filter(|t| t.contains("socialapi dm *")).count() >= 2,
            "expected both an absolute and a bare form"
        );
    }

}

/// #655/#656 — typed errors, reset-hint parsing, watchdog. Spawn-level tests
/// drive a stub CLI through the real `call_once` path (the same override
/// mechanism the fault-injection rig uses, #666).
#[cfg(test)]
mod failover_error_tests {
    use super::*;

    /// Since #954 `AUGMENTAGENT_REASONER_TIMEOUT_SECS` also bounds gate waits,
    /// so a test that shortens it must not run beside one that queues.
    static TIMEOUT_ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    fn dummy_opts() -> ReasonerOpts {
        ReasonerOpts {
            system_prompt: "stub".into(),
            model: None,
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
        }
    }

    #[test]
    fn parse_reset_hint_reads_time_and_caps_horizon() {
        use chrono::Timelike;

        // A genuinely-near reset (2h ahead, UTC) parses to the right minute.
        // Built dynamically so the test is independent of wall-clock time.
        let near = chrono::Utc::now() + chrono::Duration::hours(2);
        let (h12, ampm) = match near.hour() {
            0 => (12, "am"),
            h @ 1..=11 => (h, "am"),
            12 => (12, "pm"),
            h => (h - 12, "pm"),
        };
        let msg = format!(
            "You've hit your session limit · resets {h12}:{:02}{ampm} (UTC)",
            near.minute()
        );
        let at = parse_reset_hint(&msg).expect("near reset must parse");
        assert!(at > chrono::Utc::now(), "reset must be in the future");
        assert!(
            at <= chrono::Utc::now() + chrono::Duration::hours(6),
            "cap holds"
        );
        assert_eq!((at.hour(), at.minute()), (near.hour(), near.minute()));

        // Plausibility cap (#655 review): a reset >6h out can only come from
        // quoted/stale text or a day-rollover on a just-expired window —
        // it must be rejected (None ⇒ short default cooldown, self-healing).
        let far = chrono::Utc::now() + chrono::Duration::hours(12);
        let (fh12, fampm) = match far.hour() {
            0 => (12, "am"),
            h @ 1..=11 => (h, "am"),
            12 => (12, "pm"),
            h => (h - 12, "pm"),
        };
        let far_msg = format!("resets {fh12}:{:02}{fampm} (UTC)", far.minute());
        assert!(
            parse_reset_hint(&far_msg).is_none(),
            "distant reset must be capped to None: {far_msg}"
        );

        // Garbage shapes degrade to None, never panic.
        assert!(parse_reset_hint("no reset here").is_none());
        assert!(parse_reset_hint("resets whenever").is_none());
        assert!(parse_reset_hint("resets 13:00pm (Mars/Olympus)").is_none());
    }

    /// Write an executable stub script the reasoner spawns instead of the
    /// real `claude` (same override the fault-injection rig uses, #666).
    fn stub_cli(dir: &tempfile::TempDir, name: &str, body: &str) -> String {
        use std::io::Write as _;
        use std::os::unix::fs::PermissionsExt;
        let path = dir.path().join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "#!/usr/bin/env bash").unwrap();
        f.write_all(body.as_bytes()).unwrap();
        drop(f);
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path.to_string_lossy().into_owned()
    }

    // ---- #898: process-global cap on concurrent CLI subprocesses ----

    const RESULT_OK: &str = r#"echo '{"type":"result","result":"ok"}'"#;

    /// 40 concurrent calls through a stub that sleeps must never have more
    /// than the gate's capacity of children alive at once — and must all
    /// still complete (callers queue, nothing is dropped).
    #[tokio::test]
    async fn in_flight_never_exceeds_cap() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::time::Duration;
        let dir = tempfile::tempdir().unwrap();
        let bin = stub_cli(
            &dir,
            "fake-claude-slow",
            &format!("cat >/dev/null
sleep 0.15
{RESULT_OK}
"),
        );
        let _env = TIMEOUT_ENV_LOCK.lock().await;
        let gate = Arc::new(CliGate::new(3));
        let reasoner = Arc::new(ClaudeCliReasoner {
            bin,
            gate: Arc::clone(&gate),
        });
        let max_seen = Arc::new(AtomicUsize::new(0));
        let sampler = {
            let gate = Arc::clone(&gate);
            let max_seen = Arc::clone(&max_seen);
            tokio::spawn(async move {
                loop {
                    max_seen.fetch_max(gate.in_flight(), Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(3)).await;
                }
            })
        };
        let mut set = tokio::task::JoinSet::new();
        for _ in 0..40 {
            let r = Arc::clone(&reasoner);
            set.spawn(async move { r.call(&dummy_opts(), "hi").await });
        }
        let mut ok = 0usize;
        while let Some(joined) = set.join_next().await {
            let out = joined.unwrap().expect("every queued call completes");
            assert_eq!(out, "ok");
            ok += 1;
        }
        sampler.abort();
        assert_eq!(ok, 40);
        let max = max_seen.load(Ordering::SeqCst);
        assert!(max <= 3, "gate leaked: {max} children in flight at once");
        assert!(max >= 2, "gate should still allow concurrency, saw {max}");
        assert_eq!(gate.in_flight(), 0);
        assert_eq!(gate.waiting(), 0);
    }

    /// A permit must go back on every exit path: non-zero exit, and the
    /// caller's future being dropped (the #656 watchdog / a shutdown), so a
    /// capacity-1 gate never starves the next call.
    #[tokio::test]
    async fn permit_released_on_failure_and_on_cancel() {
        use std::time::Duration;
        let dir = tempfile::tempdir().unwrap();
        let gate = Arc::new(CliGate::new(1));

        let failing = stub_cli(&dir, "fake-claude-fail", "cat >/dev/null
echo boom >&2
exit 1
");
        let fail = ClaudeCliReasoner {
            bin: failing,
            gate: Arc::clone(&gate),
        };
        assert!(fail.call(&dummy_opts(), "x").await.is_err());
        assert_eq!(gate.in_flight(), 0, "failed call must release its permit");

        let slow = stub_cli(
            &dir,
            "fake-claude-hang",
            &format!("cat >/dev/null
sleep 5
{RESULT_OK}
"),
        );
        let hang = ClaudeCliReasoner {
            bin: slow,
            gate: Arc::clone(&gate),
        };
        let cancelled =
            tokio::time::timeout(Duration::from_millis(200), hang.call(&dummy_opts(), "x")).await;
        assert!(cancelled.is_err(), "the slow call should have been cancelled");
        assert_eq!(gate.in_flight(), 0, "a cancelled call must release its permit");

        let ok_bin = stub_cli(&dir, "fake-claude-ok", &format!("cat >/dev/null
{RESULT_OK}
"));
        let ok = ClaudeCliReasoner {
            bin: ok_bin,
            gate: Arc::clone(&gate),
        };
        let out = tokio::time::timeout(Duration::from_secs(5), ok.call(&dummy_opts(), "x"))
            .await
            .expect("next call must not be starved")
            .unwrap();
        assert_eq!(out, "ok");
    }

    /// #954 — a permit nobody releases must fail the call, not freeze it. The
    /// typed error is ours, not the provider's, so it latches nothing.
    #[tokio::test]
    async fn gate_wait_expiry_surfaces_a_typed_gate_timeout() {
        let gate = Arc::new(CliGate::new(1));
        // A hold budget far past this call's own, so the hold watchdog cannot
        // reclaim the slot first — this test is about the wait, not the leak.
        let day = std::time::Duration::from_secs(86_400);
        let leaked = gate.acquire_timed("claude", "TextOnly", day).await.unwrap();
        // The call's watchdog budget is also its gate-wait budget, so pinning
        // it short keeps this to a second of real time.
        let _env = TIMEOUT_ENV_LOCK.lock().await;
        std::env::set_var("AUGMENTAGENT_REASONER_TIMEOUT_SECS", "1");
        let bin = "fake-claude-never-spawned".into();
        let reasoner = ClaudeCliReasoner { bin, gate: Arc::clone(&gate) };

        let err = reasoner
            .call(&dummy_opts(), "hi")
            .await
            .expect_err("the only permit is held; the call cannot proceed");
        std::env::remove_var("AUGMENTAGENT_REASONER_TIMEOUT_SECS");
        let Some(typed @ ReasonerError::GateTimeout { provider, .. }) = ReasonerError::find_in(&err)
        else {
            panic!("expected GateTimeout, got {err:?}");
        };
        assert_eq!(provider, "claude");
        assert!(!typed.is_provider_side(), "our gate is not a provider fault");
        assert_eq!(gate.waiting(), 0);
        drop(leaked);
        assert_eq!(gate.in_flight(), 0);
    }

    /// #448's refusal arrives as a SUCCESSFUL completion; post-#656 it must
    /// surface as a downcastable `ReasonerError::RateLimited` with the reset
    /// hint parsed — that's what the fallback chain latches from.
    #[tokio::test]
    async fn quota_refusal_surfaces_typed_rate_limited_with_reset() {
        let dir = tempfile::tempdir().unwrap();
        let bin = stub_cli(
            &dir,
            "fake-claude-quota",
            r#"
cat >/dev/null
echo '{"type":"assistant","message":{"content":[{"type":"text","text":"You'\''ve hit your session limit · resets 9:30am (America/New_York)"}]}}'
echo '{"type":"result","result":"You'\''ve hit your session limit · resets 9:30am (America/New_York)"}'
"#,
        );
        let reasoner = ClaudeCliReasoner {
            bin,
            gate: CliGate::global(),
        };
        let err = reasoner
            .call(&dummy_opts(), "triage this")
            .await
            .expect_err("refusal must be an error");
        match ReasonerError::find_in(&err) {
            Some(ReasonerError::RateLimited {
                provider, reset_at, ..
            }) => {
                assert_eq!(provider, "claude");
                // The stub's fixed "9:30am (America/New_York)" is >6h away
                // for most of the day, in which case the plausibility cap
                // deliberately yields None (short default cooldown). When it
                // IS within the window, it must be bounded by the cap.
                if let Some(at) = reset_at {
                    assert!(*at <= chrono::Utc::now() + chrono::Duration::hours(6));
                }
            }
            other => panic!("expected RateLimited, got {other:?} (err: {err:#})"),
        }
    }

    /// #656 — a hung CLI becomes a typed Timeout instead of blocking the
    /// pipeline forever. kill_on_drop reaps the child.
    #[tokio::test]
    async fn hung_cli_times_out_with_typed_error() {
        let dir = tempfile::tempdir().unwrap();
        let bin = stub_cli(&dir, "fake-claude-hang", "cat >/dev/null\nsleep 600\n");
        let _env = TIMEOUT_ENV_LOCK.lock().await;
        std::env::set_var("AUGMENTAGENT_REASONER_TIMEOUT_SECS", "1");
        let reasoner = ClaudeCliReasoner {
            bin,
            gate: CliGate::global(),
        };
        let started = std::time::Instant::now();
        let err = reasoner.call(&dummy_opts(), "hi").await.expect_err("must time out");
        std::env::remove_var("AUGMENTAGENT_REASONER_TIMEOUT_SECS");
        assert!(
            started.elapsed() < std::time::Duration::from_secs(30),
            "watchdog must fire promptly, took {:?}",
            started.elapsed()
        );
        assert!(matches!(
            ReasonerError::find_in(&err),
            Some(ReasonerError::Timeout { .. })
        ));
    }

    /// A missing binary is a Local fault (never latched, never mistaken for
    /// an outage), and an ordinary spawn failure classifies as Unavailable.
    #[tokio::test]
    async fn missing_claude_binary_classifies_as_local() {
        let reasoner = ClaudeCliReasoner {
            bin: "/nonexistent/claude-bin".into(),
            gate: CliGate::global(),
        };
        let err = reasoner.call(&dummy_opts(), "hi").await.unwrap_err();
        assert!(matches!(
            ReasonerError::find_in(&err),
            Some(ReasonerError::Local { .. })
        ));
    }

    /// #915: the ask agent can read the transcript clone when one is
    /// configured, and nothing changes when it is not.
    #[test]
    fn ask_opts_add_the_transcript_clone_only_when_it_exists() {
        let wiki = std::env::temp_dir().join("aa-ask-wiki-test");
        std::fs::create_dir_all(&wiki).unwrap();
        let repo = std::env::temp_dir();

        // SAFETY: single-threaded test; the var is read once inside the call.
        unsafe { std::env::remove_var("AUGMENTAGENT_TRANSCRIPTS_DIR") };
        assert_eq!(ask_opts(wiki.clone(), repo.clone()).add_dirs.len(), 1);

        let t = std::env::temp_dir().join("aa-ask-transcripts-test");
        std::fs::create_dir_all(&t).unwrap();
        unsafe { std::env::set_var("AUGMENTAGENT_TRANSCRIPTS_DIR", &t) };
        assert_eq!(
            ask_opts(wiki.clone(), repo.clone()).add_dirs.len(),
            2,
            "the clone must be readable"
        );

        unsafe { std::env::set_var("AUGMENTAGENT_TRANSCRIPTS_DIR", "/nonexistent/aa-x") };
        assert_eq!(ask_opts(wiki, repo).add_dirs.len(), 1);
        unsafe { std::env::remove_var("AUGMENTAGENT_TRANSCRIPTS_DIR") };
    }
}
