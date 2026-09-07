//! Gemini CLI reasoner adapter (#655/#662) — the owner-requested "API key
//! plugin for Google".
//!
//! Uses gemini-cli headless (NOT raw REST — REST has no agent harness, and
//! re-implementing one would be exactly the dual code path #655 forbids),
//! authenticated by `GEMINI_API_KEY` (JIT keyring load, injected per spawn):
//!
//! ```text
//! GEMINI_API_KEY=<jit> GEMINI_SYSTEM_MD=<tmp>/system.md \
//! GEMINI_CLI_SYSTEM_SETTINGS_PATH=<tmp>/settings.json \
//!     gemini --output-format json -m <model>   (user message on stdin)
//! ```
//!
//! Design notes (researched 2026-08-19, google-gemini/gemini-cli docs):
//!
//! - **`GEMINI_SYSTEM_MD`** is a FULL system-prompt replacement ("not a
//!   merge") — written to a temp file per spawn since there is no flag form.
//! - **`GEMINI_CLI_SYSTEM_SETTINGS_PATH`** is the highest-precedence settings
//!   tier: we use it to pin `tools.core` per capability class — `[]` for
//!   text-only presets (strips every built-in tool declaration, taking the
//!   ~25k default token floor to near zero) and a read-only file-tool
//!   allowlist for read-tools presets. Because it outranks user/project
//!   settings, ambient `~/.gemini/settings.json` config cannot leak in.
//! - **`-m` is always pinned** — gemini's default model is `auto`, which
//!   floats across tiers: the exact bug class the
//!   `no_daemon_preset_inherits_the_owners_interactive_model` test (#448)
//!   exists to prevent.
//! - **Headless is fail-closed for approvals** (any tool that would prompt
//!   is auto-denied), and file tools are workspace-confined to the spawn cwd
//!   (+ `--include-directories`), which we pin to the wiki root for
//!   read-tools presets.
//! - **Failure mapping**: HTTP 429 / RESOURCE_EXHAUSTED → `RateLimited`
//!   (the CLI retries transient 429s internally; a terminal failure exits
//!   non-zero with a JSON error object). Everything else provider-shaped →
//!   `Unavailable`; missing binary → `Local`. #656 watchdog + kill_on_drop.
//!
//! This adapter only ever serves text-only and read-tools presets (#658
//! policy) — full-agentic parity is gated on the BeforeTool fail-open trap
//! being probed fail-closed (#664).

use std::process::Stdio;

use async_trait::async_trait;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;
use tracing::warn;

use crate::providers::{classify, model_for, tier_of, CapabilityClass, ProviderKind};
use crate::reasoner::{caller_tag, reasoner_timeout, Reasoner, ReasonerError, ReasonerOpts};

/// Gemini binary override (`GEMINI_CLI`, mirroring `CLAUDE_CLI`) — also the
/// fault-injection rig's hook (#666).
pub fn gemini_bin() -> String {
    std::env::var("GEMINI_CLI").unwrap_or_else(|_| "gemini".into())
}

pub struct GeminiCliReasoner {
    bin: String,
    /// #898 — shared cap on concurrent CLI children.
    gate: std::sync::Arc<crate::cli_gate::CliGate>,
}

impl Default for GeminiCliReasoner {
    fn default() -> Self {
        Self {
            bin: gemini_bin(),
            gate: crate::cli_gate::CliGate::global(),
        }
    }
}

impl GeminiCliReasoner {
    pub fn new() -> Self {
        Self::default()
    }

    async fn call_watchdogged(
        &self,
        opts: &ReasonerOpts,
        user_message: &str,
    ) -> anyhow::Result<String> {
        let dur = reasoner_timeout();
        // #898 — CLI slot before the watchdog starts; #954 — bounded wait.
        let caller = caller_tag(opts);
        let acquire = self.gate.acquire_timed("gemini", &caller, dur);
        let _permit = acquire.await.map_err(ReasonerError::from)?;
        match tokio::time::timeout(dur, self.call_once(opts, user_message)).await {
            // Post-classify untyped failures (stdin EPIPE, read/wait IO) as
            // provider-side Unavailable (#655 review) so they fail over
            // instead of aborting the whole chain.
            Ok(Err(e)) if ReasonerError::find_in(&e).is_none() => {
                Err(crate::reasoner::classify_other("gemini", e))
            }
            Ok(r) => r,
            Err(_) => {
                warn!("gemini call exceeded the {}s watchdog; child killed", dur.as_secs());
                Err(anyhow::Error::new(ReasonerError::Timeout {
                    provider: "gemini".into(),
                    secs: dur.as_secs(),
                }))
            }
        }
    }

    async fn call_once(&self, opts: &ReasonerOpts, user_message: &str) -> anyhow::Result<String> {
        let model = model_for(ProviderKind::Gemini, tier_of(opts));

        // `IMAGE:` markers can't be viewed here (json headless mode, file
        // tools stripped/workspace-confined) — replace them with an honest
        // note instead of leaving a dead /tmp path the model might
        // hallucinate having opened (see crate::images).
        let user_message = &crate::images::strip_markers_with_note(user_message);

        let tmp = tempfile::tempdir().map_err(|e| {
            anyhow::Error::new(ReasonerError::Local {
                message: format!("gemini: tempdir failed: {e}"),
            })
        })?;
        let effective_system = if opts.system_prompt.trim().is_empty() {
            "You are a concise assistant."
        } else {
            &opts.system_prompt
        };
        let system_md = tmp.path().join("system.md");
        std::fs::write(&system_md, effective_system).map_err(|e| {
            anyhow::Error::new(ReasonerError::Local {
                message: format!("gemini: writing system.md failed: {e}"),
            })
        })?;

        // tools.core per capability class. Tool names are gemini-cli's, not
        // Claude's — the preset's Read/Grep/Glob allowlist translates to the
        // read-only file tools; Write/Edit/Bash/MCP presets never route here.
        let core_tools: Vec<&str> = match classify(opts) {
            CapabilityClass::TextOnly => vec![],
            _ => vec![
                "read_file",
                "read_many_files",
                "list_directory",
                "glob",
                "search_file_content",
            ],
        };
        let settings = serde_json::json!({ "tools": { "core": core_tools } }).to_string();
        let settings_path = tmp.path().join("settings.json");
        std::fs::write(&settings_path, settings).map_err(|e| {
            anyhow::Error::new(ReasonerError::Local {
                message: format!("gemini: writing settings.json failed: {e}"),
            })
        })?;

        let mut args: Vec<String> = vec![
            "--output-format".into(),
            "json".into(),
            "-m".into(),
            model.clone(),
        ];
        for dir in &opts.add_dirs {
            args.push("--include-directories".into());
            args.push(dir.to_string_lossy().into_owned());
        }

        // cwd pin: preset cwd, else the first add_dir (wiki root — gemini's
        // file tools are hard-confined to the workspace root, so this is
        // what makes Read-on-the-wiki actually work), else the per-spawn
        // TEMPDIR. Never the daemon cwd (#655 review): gemini-cli auto-loads
        // `./.gemini/.env` from its cwd chain, and the daemon runs with
        // cwd = repo root, whose `.env` holds every daemon secret — a
        // text-only spawn rooted there would import all of them.
        let cwd = opts
            .cwd
            .clone()
            .or_else(|| opts.add_dirs.first().cloned())
            .unwrap_or_else(|| tmp.path().to_path_buf());

        let mut cmd = Command::new(&self.bin);
        cmd.args(&args)
            .current_dir(&cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        // Clean env + JIT key (#128 posture). HOME stays: the CLI keeps
        // caches under ~/.gemini. NB the CLI auto-loads .env files from the
        // cwd chain — pinned cwds are audited for strays by doctor (#667).
        cmd.env_clear();
        for var in ["HOME", "PATH", "USER", "LOGNAME", "TERM", "LANG", "SHELL"] {
            if let Ok(v) = std::env::var(var) {
                cmd.env(var, v);
            }
        }
        cmd.env("GEMINI_SYSTEM_MD", &system_md);
        cmd.env("GEMINI_CLI_SYSTEM_SETTINGS_PATH", &settings_path);
        if let Some(key) = crate::secret_loader::load_provider_key("GEMINI_API_KEY") {
            cmd.env("GEMINI_API_KEY", key);
        }
        // opts.env is deliberately NOT forwarded (#655 review): it carries
        // the full-agentic ask preset's sub-CLI secrets, and no preset
        // eligible to route here needs any of it.

        let mut child = cmd.spawn().map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                anyhow::Error::new(ReasonerError::Local {
                    message: format!("gemini: binary {:?} not found on PATH", self.bin),
                })
            } else {
                anyhow::Error::new(ReasonerError::Unavailable {
                    provider: "gemini".into(),
                    message: format!("spawn failed: {e}"),
                })
            }
        })?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(user_message.as_bytes()).await?;
            stdin.shutdown().await?;
        }

        // Drain stderr CONCURRENTLY (#655 review): the CLI logs retry
        // chatter to stderr mid-run; >64KB of it with stdout still open
        // would deadlock the pipe pair until the watchdog.
        let stderr_task = child.stderr.take().map(|mut err| {
            tokio::spawn(async move {
                let mut buf = String::new();
                let _ = err.read_to_string(&mut buf).await;
                buf
            })
        });
        let mut stdout_buf = String::new();
        if let Some(mut out) = child.stdout.take() {
            out.read_to_string(&mut stdout_buf).await?;
        }
        let stderr_buf = match stderr_task {
            Some(t) => t.await.unwrap_or_default(),
            None => String::new(),
        };
        let status = child.wait().await?;

        // json mode: one object with .response (+ .error on failure). The
        // CLI logs retry chatter to stderr during SUCCESSFUL runs
        // (gemini-cli#17906) — stderr alone is never treated as failure.
        let parsed: Option<serde_json::Value> = serde_json::from_str(stdout_buf.trim()).ok();
        let response = parsed
            .as_ref()
            .and_then(|v| v.get("response"))
            .and_then(|r| r.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        let error_obj = parsed.as_ref().and_then(|v| v.get("error")).cloned();

        if status.success() && !response.is_empty() && error_obj.is_none() {
            if crate::reasoner::is_rate_limited(&response) {
                return Err(anyhow::Error::new(ReasonerError::RateLimited {
                    provider: "gemini".into(),
                    message: response,
                    reset_at: None,
                }));
            }
            return Ok(response);
        }

        let code = error_obj
            .as_ref()
            .and_then(|e| e.get("code"))
            .and_then(|c| c.as_i64());
        let detail = error_obj
            .as_ref()
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
            .map(str::to_string)
            .or_else(|| {
                stderr_buf
                    .lines()
                    .rev()
                    .find(|l| !l.trim().is_empty())
                    .map(str::to_string)
            })
            .unwrap_or_else(|| format!("gemini exited {status:?} with no response"));
        warn!("gemini call failed (code {code:?}): {detail}");

        let quota_shaped = code == Some(429)
            || detail.to_ascii_lowercase().contains("resource_exhausted")
            || detail.to_ascii_lowercase().contains("ratelimitexceeded")
            || detail.to_ascii_lowercase().contains("rate limit")
            || detail.to_ascii_lowercase().contains("quota");
        if quota_shaped {
            return Err(anyhow::Error::new(ReasonerError::RateLimited {
                provider: "gemini".into(),
                message: detail,
                reset_at: None,
            }));
        }
        Err(anyhow::Error::new(ReasonerError::Unavailable {
            provider: "gemini".into(),
            message: detail.chars().take(300).collect(),
        }))
    }
}

#[async_trait]
impl Reasoner for GeminiCliReasoner {
    async fn call(&self, opts: &ReasonerOpts, user_message: &str) -> anyhow::Result<String> {
        self.call_watchdogged(opts, user_message).await
    }

    /// json mode returns the final response either way; gemini never serves
    /// the full-agentic ask preset where LastBlock-vs-AllBlocks (#446)
    /// actually diverges, so both entrypoints share one path.
    async fn call_transcript(
        &self,
        opts: &ReasonerOpts,
        user_message: &str,
    ) -> anyhow::Result<String> {
        self.call_watchdogged(opts, user_message).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::ModelTier;
    use std::io::Write as _;
    use std::os::unix::fs::PermissionsExt;

    fn stub(dir: &tempfile::TempDir, name: &str, body: &str) -> String {
        let path = dir.path().join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "#!/usr/bin/env bash").unwrap();
        f.write_all(body.as_bytes()).unwrap();
        drop(f);
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path.to_string_lossy().into_owned()
    }

    fn opts(tools: Vec<&str>) -> ReasonerOpts {
        ReasonerOpts {
            system_prompt: "You summarize.".into(),
            model: Some("claude-haiku-4-5-20251001".into()),
            allowed_tools: tools.into_iter().map(str::to_string).collect(),
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
    async fn parses_response_and_pins_fast_tier_model() {
        let dir = tempfile::tempdir().unwrap();
        let record = dir.path().join("argv.txt");
        let bin = stub(
            &dir,
            "fake-gemini",
            &format!(
                r#"
cat >/dev/null
printf '%s\n' "$@" > {record}
echo "SYSTEM_MD=${{GEMINI_SYSTEM_MD:+set}}" >> {record}
echo "SETTINGS=${{GEMINI_CLI_SYSTEM_SETTINGS_PATH:+set}}" >> {record}
echo '{{"response":"a fine summary","stats":{{}}}}'
"#,
                record = record.display()
            ),
        );
        let r = GeminiCliReasoner {
            bin,
            gate: crate::cli_gate::CliGate::global(),
        };
        let got = r.call(&opts(vec![]), "summarize this").await.unwrap();
        assert_eq!(got, "a fine summary");
        let rec = std::fs::read_to_string(&record).unwrap();
        assert!(rec.contains("gemini-3.7-flash"), "haiku-tier preset → fast tier: {rec}");
        assert!(rec.contains("SYSTEM_MD=set"), "system prompt must travel via GEMINI_SYSTEM_MD");
        assert!(rec.contains("SETTINGS=set"), "per-spawn settings must be pinned");
        assert!(!rec.contains("\nauto\n"), "-m must never be left on gemini's floating default");
    }

    /// #658 — the argv end of the tier map: `-m` carries exactly what
    /// `model_for` resolved, on BOTH tiers, never the floating `auto` default.
    #[tokio::test]
    async fn spawn_pins_the_resolved_model_on_both_tiers() {
        for (preset_model, tier) in [
            ("claude-opus-4-8", ModelTier::Quality),
            ("claude-haiku-4-5-20251001", ModelTier::Fast),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let record = dir.path().join("argv.txt");
            let bin = stub(
                &dir,
                "fake-gemini",
                &format!(
                    "cat >/dev/null\nprintf '%s\\n' \"$@\" >{}\necho '{{\"response\":\"ok\"}}'\n",
                    record.display()
                ),
            );
            let mut o = opts(vec![]);
            o.model = Some(preset_model.into());
            GeminiCliReasoner {
            bin,
            gate: crate::cli_gate::CliGate::global(),
        }.call(&o, "hi").await.unwrap();
            let argv: Vec<String> = std::fs::read_to_string(&record)
                .unwrap()
                .lines()
                .map(str::to_string)
                .collect();
            let want = model_for(ProviderKind::Gemini, tier);
            assert!(
                argv.windows(2).any(|w| w[0] == "-m" && w[1] == want),
                "gemini must spawn with `-m {want}`, got {argv:?}"
            );
        }
    }

    #[tokio::test]
    async fn quota_error_object_maps_to_rate_limited() {
        let dir = tempfile::tempdir().unwrap();
        let bin = stub(
            &dir,
            "fake-gemini-429",
            r#"
cat >/dev/null
echo '{"error":{"type":"ApiError","message":"429 RESOURCE_EXHAUSTED: rateLimitExceeded","code":429}}'
exit 1
"#,
        );
        let r = GeminiCliReasoner {
            bin,
            gate: crate::cli_gate::CliGate::global(),
        };
        let err = r.call(&opts(vec![]), "hi").await.unwrap_err();
        match ReasonerError::find_in(&err) {
            Some(ReasonerError::RateLimited { provider, .. }) => assert_eq!(provider, "gemini"),
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stderr_chatter_alone_is_not_failure() {
        // gemini-cli#17906: the CLI logs "quota exhausted" retry noise to
        // stderr during runs that ultimately SUCCEED. Only the exit code +
        // json error object decide.
        let dir = tempfile::tempdir().unwrap();
        let bin = stub(
            &dir,
            "fake-gemini-noisy",
            r#"
cat >/dev/null
echo "Attempt 1 failed: quota exhausted, retrying..." >&2
echo '{"response":"recovered fine"}'
"#,
        );
        let r = GeminiCliReasoner {
            bin,
            gate: crate::cli_gate::CliGate::global(),
        };
        assert_eq!(r.call(&opts(vec![]), "hi").await.unwrap(), "recovered fine");
    }
}
