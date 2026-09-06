//! `FallbackReasoner` — the multi-provider composite (#655/#659).
//!
//! One `Reasoner` seam, N providers behind it. The composite walks the
//! configured chain in order and returns the first success. Failover is
//! deliberately narrow:
//!
//! - It triggers ONLY on a typed [`ReasonerError`] in the failure chain —
//!   provider-side (`RateLimited`/`Timeout`/`Unavailable`, which also latch
//!   a cooldown) or `Local` (missing binary / corrupted config: try the next
//!   provider, but don't latch — a local fault can be fixed any second).
//! - An **untyped** error returns immediately. Untyped means the failure
//!   happened above the provider layer (test doubles, extraction) or is a
//!   content-level problem — re-asking a different model would convert
//!   refusals into plausible-but-wrong answers, the #450/#451 pathway.
//! - **One pass over the chain per call.** No per-provider retries; #448's
//!   no-retry rule survives intact inside each adapter.
//!
//! Latched providers are skipped without spawning anything — that skip is
//! what turns the 2-minute triage re-poll during an outage from a ~24k-token
//! spawn per email into a millisecond no-op (#660's worst ladder).
//!
//! `call_code_mode` / `call_code_mode_with_repair` are NOT overridden: the
//! trait defaults delegate to `call`, so code-mode rides the chain for free
//! and the fenced-block extraction stays above the failover boundary.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use tracing::{info, warn};

use crate::cooldown::CooldownLatch;
use crate::providers::{allowed_for, bin_resolves, chain_from_env, classify, ProviderKind};
use crate::reasoner::{ClaudeCliReasoner, Reasoner, ReasonerError, ReasonerOpts};

/// Default cooldown when a rate-limit refusal carries no parseable reset
/// hint. Claude session windows are 5-hourly; 30 minutes re-probes a few
/// times per window without hammering.
fn default_ratelimit_cooldown() -> chrono::Duration {
    let secs = std::env::var("AUGMENTAGENT_COOLDOWN_RATELIMIT_SECS")
        .ok()
        .and_then(|v| v.trim().parse::<i64>().ok())
        .filter(|s| *s > 0)
        .unwrap_or(1800);
    chrono::Duration::seconds(secs)
}

/// Cooldown for outage-shaped failures (timeout / unavailable). Deliberately
/// SHORT (#655 review): with the default claude-only chain a latch is a
/// hard no-service window, so one transient non-zero exit must cost ~a
/// minute of fast-fails, not five — real outages just re-latch on the next
/// probe, which still caps spawn pressure at ~1/min instead of every call.
fn default_unavailable_cooldown() -> chrono::Duration {
    let secs = std::env::var("AUGMENTAGENT_COOLDOWN_UNAVAILABLE_SECS")
        .ok()
        .and_then(|v| v.trim().parse::<i64>().ok())
        .filter(|s| *s > 0)
        .unwrap_or(60);
    chrono::Duration::seconds(secs)
}

/// Hard ceiling on any latch derived from a PARSED reset hint (#655 review).
/// Claude session windows are 5-hourly; anything longer means the hint came
/// from quoted/stale text and must not black the provider out for a day.
fn max_ratelimit_latch() -> chrono::Duration {
    chrono::Duration::hours(6)
}

struct Entry {
    kind: ProviderKind,
    reasoner: Arc<dyn Reasoner>,
}

/// The composite. Concrete (not `dyn`) so the ~28 construction sites and the
/// channel generics swap from `ClaudeCliReasoner` with a one-word change.
pub struct FallbackReasoner {
    entries: Vec<Entry>,
    latch: CooldownLatch,
    /// #803 — resource accounting for this instance: `(provider, calls
    /// attempted, calls that returned Ok)`, in chain order of first use. The
    /// auto-PR loop builds one reasoner per attempt, so this IS the attempt's
    /// spend and the record of which provider actually served it.
    usage: std::sync::Mutex<Vec<(&'static str, u32, u32)>>,
}

/// Why `kind` cannot serve calls on this box (binary absent, no resolvable
/// auth), or `None` when it is usable. Claude is always eligible — it is the
/// primary this daemon has run on since day one, and a missing `claude`
/// binary should fail loudly per call, not silently vanish. Public so
/// `doctor` (#658) reports the same verdict the chain builder reaches,
/// instead of a second opinion that can drift from it.
pub fn ineligible_reason(kind: ProviderKind) -> Option<String> {
    match kind {
        ProviderKind::Claude => None,
        ProviderKind::Codex => {
            let bin = crate::codex::codex_bin();
            if !bin_resolves(&bin) {
                return Some(format!("{bin:?} not installed"));
            }
            if !crate::codex::codex_auth_available() {
                return Some(
                    "no CODEX_API_KEY and no auth.json — run `codex login` or seed the key"
                        .to_string(),
                );
            }
            None
        }
        ProviderKind::Gemini => {
            let bin = crate::gemini::gemini_bin();
            if !bin_resolves(&bin) {
                return Some(format!("{bin:?} not installed"));
            }
            if crate::secret_loader::load_provider_key("GEMINI_API_KEY").is_none() {
                return Some("no GEMINI_API_KEY in keyring/env".to_string());
            }
            None
        }
        ProviderKind::Cerebras => {
            if crate::secret_loader::load_provider_key("CEREBRAS_API_KEY").is_none() {
                return Some("no CEREBRAS_API_KEY in keyring/env".to_string());
            }
            None
        }
    }
}

/// Construct the entry for one provider, or `None` when it is not eligible.
fn entry_for(kind: ProviderKind) -> Option<Entry> {
    if let Some(reason) = ineligible_reason(kind) {
        info!("reasoner chain: {} skipped ({reason})", kind.name());
        return None;
    }
    let reasoner: Arc<dyn Reasoner> = match kind {
        ProviderKind::Claude => Arc::new(ClaudeCliReasoner::new()),
        ProviderKind::Codex => Arc::new(crate::codex::CodexCliReasoner::openai()),
        ProviderKind::Gemini => Arc::new(crate::gemini::GeminiCliReasoner::new()),
        // Thin chat-completions client (#663 plan B — codex ≥0.148 removed
        // wire_api=chat and Cerebras has no Responses API).
        ProviderKind::Cerebras => Arc::new(crate::cerebras::CerebrasHttpReasoner::new()),
    };
    Some(Entry { kind, reasoner })
}

/// Build the production reasoner from `AUGMENTAGENT_REASONER_CHAIN`.
///
/// Eligibility is checked once per construction: a fallback CLI that is not
/// installed (or has no resolvable auth) is dropped from the chain with one
/// log line instead of erroring on every call.
pub fn build_reasoner() -> Arc<FallbackReasoner> {
    let mut entries: Vec<Entry> = chain_from_env().into_iter().filter_map(entry_for).collect();
    if entries.is_empty() {
        // Unreachable via chain_from_env (it always yields claude), but keep
        // the invariant explicit: the composite always has a primary.
        entries.push(Entry {
            kind: ProviderKind::Claude,
            reasoner: Arc::new(ClaudeCliReasoner::new()),
        });
    }
    if entries.len() > 1 {
        let names: Vec<_> = entries.iter().map(|e| e.kind.name()).collect();
        info!("reasoner chain active: {}", names.join(" → "));
    }
    Arc::new(FallbackReasoner {
        entries,
        latch: CooldownLatch::system(),
        usage: std::sync::Mutex::new(Vec::new()),
    })
}

/// #828 — a reasoner pinned to exactly ONE provider, or `None` when that
/// provider is not eligible.
///
/// This exists for the independent-review stage, where the whole value is
/// that the reviewer is NOT the author. `build_reasoner` would hand back
/// Claude (the head of the chain), so an independent review built on it
/// would be Claude reviewing Claude while the PR claimed otherwise.
///
/// Deliberately returns `None` rather than falling back: a caller that
/// cannot get the provider it asked for must be able to tell, and must fail
/// closed. There is no chain here, so failover cannot silently occur either.
pub fn build_pinned(kind: ProviderKind) -> Option<Arc<FallbackReasoner>> {
    entry_for(kind).map(|entry| {
        Arc::new(FallbackReasoner {
            entries: vec![entry],
            latch: CooldownLatch::system(),
            usage: std::sync::Mutex::new(Vec::new()),
        })
    })
}

impl FallbackReasoner {
    /// Claude-only composite — behaviorally identical to the pre-#655
    /// `ClaudeCliReasoner` construction it replaces.
    pub fn claude_only() -> Self {
        FallbackReasoner {
            entries: vec![Entry {
                kind: ProviderKind::Claude,
                reasoner: Arc::new(ClaudeCliReasoner::new()),
            }],
            latch: CooldownLatch::system(),
            usage: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Test constructor: explicit (kind, reasoner) chain + latch path.
    pub fn for_tests(
        chain: Vec<(ProviderKind, Arc<dyn Reasoner>)>,
        latch: CooldownLatch,
    ) -> Self {
        FallbackReasoner {
            entries: chain
                .into_iter()
                .map(|(kind, reasoner)| Entry { kind, reasoner })
                .collect(),
            latch,
            usage: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Providers currently configured (for status surfaces).
    pub fn provider_names(&self) -> Vec<&'static str> {
        self.entries.iter().map(|e| e.kind.name()).collect()
    }

    /// One more call dispatched to `name` (before the provider runs).
    fn note_call(&self, name: &'static str) {
        let mut u = self.usage.lock().unwrap_or_else(|e| e.into_inner());
        match u.iter_mut().find(|(n, _, _)| *n == name) {
            Some(row) => row.1 += 1,
            None => u.push((name, 1, 0)),
        }
    }

    /// The call just dispatched to `name` returned Ok.
    fn note_ok(&self, name: &'static str) {
        let mut u = self.usage.lock().unwrap_or_else(|e| e.into_inner());
        match u.iter_mut().find(|(n, _, _)| *n == name) {
            Some(row) => row.2 += 1,
            None => u.push((name, 1, 1)),
        }
    }

    /// #803 — `(provider, calls attempted, calls served)` for this instance.
    pub fn usage(&self) -> Vec<(&'static str, u32, u32)> {
        self.usage.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Calls attempted across every provider on this instance.
    pub fn calls(&self) -> u32 {
        self.usage().iter().map(|(_, c, _)| c).sum()
    }

    /// `"claude 3/3, codex 1/2"` — served/attempted per provider; empty when
    /// nothing was called.
    pub fn usage_summary(&self) -> String {
        self.usage()
            .iter()
            .map(|(n, c, ok)| format!("{n} {ok}/{c}"))
            .collect::<Vec<_>>()
            .join(", ")
    }

    async fn dispatch(
        &self,
        opts: &ReasonerOpts,
        user_message: &str,
        transcript: bool,
    ) -> anyhow::Result<String> {
        let class = classify(opts);
        let primary = self.entries.first().map(|e| e.kind);
        // The PRIMARY's provider-side error is what callers must see when
        // the whole chain fails (#655 review): a trailing Local fault from a
        // misconfigured fallback would otherwise mask the rate limit and
        // defeat the #660 provider-side downcast in code-mode.
        let mut first_provider_err: Option<anyhow::Error> = None;
        let mut last_err: Option<anyhow::Error> = None;
        let mut skipped_latched = 0usize;

        for entry in &self.entries {
            let name = entry.kind.name();
            if !allowed_for(entry.kind, class) {
                continue;
            }
            if let Some(until) = self.latch.latched_until(name) {
                skipped_latched += 1;
                tracing::debug!(provider = name, %until, "provider latched; skipping");
                continue;
            }
            self.note_call(name);
            let res = if transcript {
                entry.reasoner.call_transcript(opts, user_message).await
            } else {
                entry.reasoner.call(opts, user_message).await
            };
            match res {
                Ok(text) => {
                    self.note_ok(name);
                    if Some(entry.kind) != primary {
                        info!(
                            provider = name,
                            "reasoner call served by FALLBACK provider ({} unavailable)",
                            primary.map(|p| p.name()).unwrap_or("primary")
                        );
                    }
                    self.latch.clear(name);
                    return Ok(text);
                }
                Err(err) => match ReasonerError::find_in(&err) {
                    Some(re) if re.is_provider_side() => {
                        // Timeout on an agentic/write run is "this one call
                        // ran long", not "provider down" (#655 review) —
                        // don't take unrelated triage/draft calls down with
                        // a latch; still try the rest of the chain.
                        let latchworthy = !matches!(re, ReasonerError::Timeout { .. })
                            || matches!(
                                class,
                                crate::providers::CapabilityClass::TextOnly
                                    | crate::providers::CapabilityClass::ReadTools
                            );
                        if latchworthy {
                            let until = match re {
                                ReasonerError::RateLimited {
                                    reset_at: Some(at), ..
                                } => {
                                    // Clamp even a parsed reset (#655 review
                                    // — belt to parse_reset_hint's suspenders).
                                    (*at).min(Utc::now() + max_ratelimit_latch())
                                }
                                ReasonerError::RateLimited { .. } => {
                                    Utc::now() + default_ratelimit_cooldown()
                                }
                                _ => Utc::now() + default_unavailable_cooldown(),
                            };
                            warn!(
                                provider = name,
                                %until,
                                "provider failed provider-side ({re}); latched, trying next in chain"
                            );
                            self.latch.latch(name, until, &re.to_string());
                        } else {
                            warn!(
                                provider = name,
                                "provider timed out on a long-running {class:?} call; \
                                 NOT latching (one slow call is not an outage)"
                            );
                        }
                        if first_provider_err.is_none() {
                            first_provider_err = Some(err);
                        } else {
                            last_err = Some(err);
                        }
                        continue;
                    }
                    Some(ReasonerError::Local { .. } | ReasonerError::GateTimeout { .. }) => {
                        // Our fault, not the provider's — a bad local config
                        // or (#954) a jammed CLI gate. Eligible to try the
                        // next provider, which may not even be gated
                        // (cerebras is HTTP), but never latched: a fixed
                        // config should work on the very next call.
                        warn!(provider = name, "local fault ({err}); trying next");
                        if last_err.is_none() {
                            last_err = Some(err);
                        }
                        continue;
                    }
                    _ => return Err(err),
                },
            }
        }

        Err(first_provider_err.or(last_err).unwrap_or_else(|| {
            anyhow::Error::new(ReasonerError::Unavailable {
                provider: "chain".into(),
                message: format!(
                    "no provider available for {class:?} call ({} latched; chain: {})",
                    skipped_latched,
                    self.entries
                        .iter()
                        .map(|e| e.kind.name())
                        .collect::<Vec<_>>()
                        .join(",")
                ),
            })
        }))
    }
}

#[async_trait]
impl Reasoner for FallbackReasoner {
    async fn call(&self, opts: &ReasonerOpts, user_message: &str) -> anyhow::Result<String> {
        self.dispatch(opts, user_message, false).await
    }

    /// Forwarded explicitly so the LastBlock/AllBlocks capture semantics
    /// (#446) survive the composite — the trait default would collapse
    /// transcript calls onto `call`.
    async fn call_transcript(
        &self,
        opts: &ReasonerOpts,
        user_message: &str,
    ) -> anyhow::Result<String> {
        self.dispatch(opts, user_message, true).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // ---- #828: single-provider pinning for the independent review ----

    // #803 — the loop reads this per attempt for its resource accounting.
    #[tokio::test]
    async fn usage_counts_calls_and_successes_per_provider() {
        let dir = tempfile::tempdir().unwrap();
        let a = Scripted::ok("answer");
        let fb = FallbackReasoner::for_tests(
            vec![(ProviderKind::Claude, a.clone() as Arc<dyn Reasoner>)],
            latch_in(&dir),
        );
        assert_eq!(fb.calls(), 0);
        assert_eq!(fb.usage_summary(), "");
        let opts = text_only_opts();
        fb.call(&opts, "hi").await.unwrap();
        fb.call(&opts, "again").await.unwrap();
        assert_eq!(fb.calls(), 2, "two calls attempted");
        assert_eq!(fb.usage(), vec![("claude", 2, 2)], "(provider, attempted, served)");
        assert_eq!(fb.usage_summary(), "claude 2/2");
        // A provider that errors counts an attempt and no serve.
        let dir2 = tempfile::tempdir().unwrap();
        let bad = Scripted::err(|| anyhow::anyhow!("boom"));
        let fb2 = FallbackReasoner::for_tests(
            vec![(ProviderKind::Claude, bad.clone() as Arc<dyn Reasoner>)],
            latch_in(&dir2),
        );
        let _ = fb2.call(&opts, "hi").await;
        assert_eq!(fb2.usage(), vec![("claude", 1, 0)]);
        assert_eq!(fb2.usage_summary(), "claude 0/1");
    }

    #[test]
    fn build_pinned_yields_exactly_that_provider() {
        let r = build_pinned(ProviderKind::Claude).expect("claude is always eligible");
        assert_eq!(
            r.provider_names(),
            vec!["claude"],
            "a pinned reasoner must carry one provider and no chain behind it"
        );
    }

    #[test]
    fn build_pinned_fails_closed_when_the_provider_is_ineligible() {
        // The whole point of the independent review is that the reviewer is
        // not the author. If codex cannot be built, the caller must be able
        // to SEE that — a silent fall back to claude would mean Claude
        // reviewing Claude while the PR claims an independent approval.
        let _g = env_guard();
        let prev = std::env::var("CODEX_CLI").ok();
        std::env::set_var("CODEX_CLI", "/nonexistent/codex-binary-for-test");
        let got = build_pinned(ProviderKind::Codex);
        match prev {
            Some(v) => std::env::set_var("CODEX_CLI", v),
            None => std::env::remove_var("CODEX_CLI"),
        }
        assert!(
            got.is_none(),
            "an uninstalled provider must yield None, never a substitute"
        );
    }

    /// Serialises the env-mutating tests in this module (#709).
    fn env_guard() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Scripted provider double: returns a canned result and counts calls.
    struct Scripted {
        result: Box<dyn Fn() -> anyhow::Result<String> + Send + Sync>,
        calls: AtomicUsize,
        transcript_calls: AtomicUsize,
    }

    impl Scripted {
        fn ok(text: &'static str) -> Arc<Self> {
            Arc::new(Self {
                result: Box::new(move || Ok(text.to_string())),
                calls: AtomicUsize::new(0),
                transcript_calls: AtomicUsize::new(0),
            })
        }
        fn err(mk: impl Fn() -> anyhow::Error + Send + Sync + 'static) -> Arc<Self> {
            Arc::new(Self {
                result: Box::new(move || Err(mk())),
                calls: AtomicUsize::new(0),
                transcript_calls: AtomicUsize::new(0),
            })
        }
        fn count(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl Reasoner for Scripted {
        async fn call(&self, _opts: &ReasonerOpts, _msg: &str) -> anyhow::Result<String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            (self.result)()
        }
        async fn call_transcript(
            &self,
            _opts: &ReasonerOpts,
            _msg: &str,
        ) -> anyhow::Result<String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.transcript_calls.fetch_add(1, Ordering::SeqCst);
            (self.result)()
        }
    }

    fn text_only_opts() -> ReasonerOpts {
        ReasonerOpts {
            system_prompt: "s".into(),
            model: Some("claude-opus-4-8".into()),
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

    fn rate_limited() -> anyhow::Error {
        anyhow::Error::new(ReasonerError::RateLimited {
            provider: "claude".into(),
            message: "You've hit your session limit · resets 9:30am".into(),
            reset_at: None,
        })
    }

    fn latch_in(dir: &tempfile::TempDir) -> CooldownLatch {
        CooldownLatch::at(dir.path().join("cooldowns.json"))
    }

    #[tokio::test]
    async fn primary_success_never_touches_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let a = Scripted::ok("primary answer");
        let b = Scripted::ok("fallback answer");
        let fb = FallbackReasoner::for_tests(
            vec![
                (ProviderKind::Claude, a.clone() as Arc<dyn Reasoner>),
                (ProviderKind::Codex, b.clone() as Arc<dyn Reasoner>),
            ],
            latch_in(&dir),
        );
        let got = fb.call(&text_only_opts(), "hi").await.unwrap();
        assert_eq!(got, "primary answer");
        assert_eq!(a.count(), 1);
        assert_eq!(b.count(), 0, "fallback must not be probed on success");
    }

    #[tokio::test]
    async fn rate_limited_primary_fails_over_and_latches() {
        let dir = tempfile::tempdir().unwrap();
        let latch = latch_in(&dir);
        let a = Scripted::err(rate_limited);
        let b = Scripted::ok("codex answer");
        let fb = FallbackReasoner::for_tests(
            vec![
                (ProviderKind::Claude, a.clone() as Arc<dyn Reasoner>),
                (ProviderKind::Codex, b.clone() as Arc<dyn Reasoner>),
            ],
            latch.clone(),
        );
        let got = fb.call(&text_only_opts(), "hi").await.unwrap();
        assert_eq!(got, "codex answer");
        assert_eq!(a.count(), 1);
        assert_eq!(b.count(), 1);
        assert!(
            latch.latched_until("claude").is_some(),
            "quota refusal must latch the provider"
        );

        // Second call: latched primary is SKIPPED — zero spawns against the
        // wall (the #448 anti-amplification property).
        let got2 = fb.call(&text_only_opts(), "hi again").await.unwrap();
        assert_eq!(got2, "codex answer");
        assert_eq!(a.count(), 1, "latched provider must not be called again");
        assert_eq!(b.count(), 2);
    }

    #[tokio::test]
    async fn parsed_reset_hint_is_used_for_the_latch() {
        let dir = tempfile::tempdir().unwrap();
        let latch = latch_in(&dir);
        let reset = Utc::now() + chrono::Duration::hours(3);
        let a = Scripted::err(move || {
            anyhow::Error::new(ReasonerError::RateLimited {
                provider: "claude".into(),
                message: "limit".into(),
                reset_at: Some(reset),
            })
        });
        let b = Scripted::ok("ok");
        let fb = FallbackReasoner::for_tests(
            vec![
                (ProviderKind::Claude, a as Arc<dyn Reasoner>),
                (ProviderKind::Codex, b as Arc<dyn Reasoner>),
            ],
            latch.clone(),
        );
        fb.call(&text_only_opts(), "hi").await.unwrap();
        assert_eq!(latch.latched_until("claude"), Some(reset));
    }

    #[tokio::test]
    async fn capability_class_gates_the_chain() {
        let dir = tempfile::tempdir().unwrap();
        let a = Scripted::err(rate_limited);
        let b = Scripted::ok("should never serve write-tools");
        let fb = FallbackReasoner::for_tests(
            vec![
                (ProviderKind::Claude, a.clone() as Arc<dyn Reasoner>),
                (ProviderKind::Cerebras, b.clone() as Arc<dyn Reasoner>),
            ],
            latch_in(&dir),
        );
        // Write-tools preset (ingest-shaped): cerebras is text-only-eligible,
        // so the chain has no fallback and the typed error surfaces.
        let mut opts = text_only_opts();
        opts.allowed_tools = vec!["Read".into(), "Write".into(), "Edit".into()];
        let err = fb.call(&opts, "hi").await.unwrap_err();
        assert!(ReasonerError::find_in(&err).is_some());
        assert_eq!(b.count(), 0, "ineligible provider must not be tried");
    }

    #[tokio::test]
    async fn untyped_error_returns_immediately_without_failover() {
        let dir = tempfile::tempdir().unwrap();
        let a = Scripted::err(|| anyhow::anyhow!("triage parse failed: no JSON object"));
        let b = Scripted::ok("must not serve");
        let fb = FallbackReasoner::for_tests(
            vec![
                (ProviderKind::Claude, a.clone() as Arc<dyn Reasoner>),
                (ProviderKind::Codex, b.clone() as Arc<dyn Reasoner>),
            ],
            latch_in(&dir),
        );
        let err = fb.call(&text_only_opts(), "hi").await.unwrap_err();
        assert!(ReasonerError::find_in(&err).is_none());
        assert_eq!(b.count(), 0, "untyped errors must not trigger failover");
    }

    #[tokio::test]
    async fn local_fault_tries_next_without_latching() {
        let dir = tempfile::tempdir().unwrap();
        let latch = latch_in(&dir);
        let a = Scripted::err(|| {
            anyhow::Error::new(ReasonerError::Local {
                message: "claude config corrupted".into(),
            })
        });
        let b = Scripted::ok("served");
        let fb = FallbackReasoner::for_tests(
            vec![
                (ProviderKind::Claude, a as Arc<dyn Reasoner>),
                (ProviderKind::Codex, b as Arc<dyn Reasoner>),
            ],
            latch.clone(),
        );
        assert_eq!(fb.call(&text_only_opts(), "hi").await.unwrap(), "served");
        assert!(
            latch.latched_until("claude").is_none(),
            "local faults are not latched"
        );
    }

    #[tokio::test]
    async fn whole_chain_latched_fails_fast_with_typed_error() {
        let dir = tempfile::tempdir().unwrap();
        let latch = latch_in(&dir);
        let far = Utc::now() + chrono::Duration::hours(1);
        latch.latch("claude", far, "quota");
        latch.latch("codex", far, "quota");
        let a = Scripted::ok("unreachable");
        let b = Scripted::ok("unreachable");
        let fb = FallbackReasoner::for_tests(
            vec![
                (ProviderKind::Claude, a.clone() as Arc<dyn Reasoner>),
                (ProviderKind::Codex, b.clone() as Arc<dyn Reasoner>),
            ],
            latch,
        );
        let err = fb.call(&text_only_opts(), "hi").await.unwrap_err();
        let re = ReasonerError::find_in(&err).expect("typed chain-exhausted error");
        assert!(matches!(re, ReasonerError::Unavailable { .. }));
        assert_eq!(a.count() + b.count(), 0, "no spawns while fully latched");
    }

    #[tokio::test]
    async fn transcript_capture_is_forwarded_not_collapsed() {
        let dir = tempfile::tempdir().unwrap();
        let a = Scripted::ok("full transcript");
        let fb = FallbackReasoner::for_tests(
            vec![(ProviderKind::Claude, a.clone() as Arc<dyn Reasoner>)],
            latch_in(&dir),
        );
        fb.call_transcript(&text_only_opts(), "hi").await.unwrap();
        assert_eq!(
            a.transcript_calls.load(Ordering::SeqCst),
            1,
            "call_transcript must reach the provider's transcript path (#446)"
        );
    }

    #[tokio::test]
    async fn success_clears_a_stale_latch() {
        let dir = tempfile::tempdir().unwrap();
        let latch = latch_in(&dir);
        // Simulate a recovered provider whose latch just expired.
        latch.latch("claude", Utc::now() - chrono::Duration::seconds(1), "old");
        let a = Scripted::ok("back");
        let fb = FallbackReasoner::for_tests(
            vec![(ProviderKind::Claude, a as Arc<dyn Reasoner>)],
            latch.clone(),
        );
        fb.call(&text_only_opts(), "hi").await.unwrap();
        assert!(latch.active().is_empty(), "success clears the stale entry");
    }
}
