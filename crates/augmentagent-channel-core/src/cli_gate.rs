//! #898 — process-global cap on concurrently running CLI reasoner
//! subprocesses (`claude -p`, `codex`, `gemini`).
//!
//! Every reasoner call is a fresh CLI process costing 60–100 MB before it
//! does any work. Nothing bounded how many could be alive at once: on
//! 2026-08-31 an ingest burst spawned 297 of them (~16 GB on a 15 GB box)
//! and the kernel OOM killer took the desktop session with it (#897). The
//! only thing that stopped it at 297 was the unit's 1024 soft fd limit.
//!
//! The gate is a semaphore shared by every reasoner instance in the
//! process — each channel builds its own reasoner, so a per-instance limit
//! would not help. Callers beyond the capacity *wait*: ingest is
//! best-effort and already async, so queueing is the correct backpressure
//! (bounded since #954, below). A permit is taken immediately before
//! `Command::spawn` and lives in the same scope as the `Child`, so it is
//! released when the child is reaped, on every early-return error, and when
//! the caller's future is dropped (the #656 watchdog, shutdown).
//!
//! Capacity: `AUGMENTAGENT_REASONER_MAX_INFLIGHT` (default 4, minimum 1),
//! read once when the global gate is first used.
//!
//! #954 — on 2026-09-04 four permits were held by futures that were neither
//! running a child nor timing out, and every reasoner call in the daemon
//! queued behind them for 15 h: no email triage, no auto-PR loop, no log
//! line (the saturation warning below only fires *above* capacity, and one
//! waiter per channel never reaches it). So: the wait is bounded by the
//! caller's own #656 budget; every permit records who took it and when, so
//! a hold past that budget is logged and the slot reclaimed (a leak must not
//! narrow the gate for the life of the process); and the state is published
//! to [`snapshot_path`], since the gate lives in the daemon's process while
//! `augmentagent doctor` runs in another one.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::time::Instant;
use tracing::{info, warn};

pub const DEFAULT_MAX_INFLIGHT: usize = 4;
pub const ENV_MAX_INFLIGHT: &str = "AUGMENTAGENT_REASONER_MAX_INFLIGHT";
/// A saturated gate logs at most once per this interval — a 1,700-item
/// burst must not become 1,700 log lines.
const SATURATION_WARN_EVERY: Duration = Duration::from_secs(60);
/// How often a still-queued caller says so. An "is anything stuck" signal,
/// not a latency measurement.
const QUEUED_WARN_EVERY: Duration = Duration::from_secs(60);
/// A permit held this many times its caller's own watchdog budget has no live
/// child behind it — #656 kills at 1× and drops the permit with the future.
const STALE_HOLD_FACTOR: u32 = 2;

/// A caller gave up waiting for a permit. Ours, not the provider's — the
/// adapters map it to `ReasonerError::GateTimeout`, which latches nothing.
#[derive(Debug, thiserror::Error)]
#[error("{provider} waited {waited_secs}s for a CLI gate permit")]
pub struct GateWaitTimeout {
    pub provider: String,
    pub waited_secs: u64,
}

/// One permit currently out, tracked so a hold outliving its caller's budget
/// can be named and reclaimed (#954).
struct Held {
    provider: String,
    since: Instant,
    budget: Duration,
    reclaimed: Arc<AtomicBool>,
}

pub struct CliGate {
    sem: Arc<Semaphore>,
    capacity: usize,
    held: Mutex<HashMap<u64, Held>>,
    next_id: AtomicU64,
    waiting: AtomicUsize,
    last_warn_ms: AtomicU64,
    queued_warns: AtomicU64,
    reclaimed: AtomicU64,
    /// Only the global gate publishes for `doctor`; test gates stay silent.
    snapshot_to: Option<PathBuf>,
}

impl CliGate {
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            sem: Arc::new(Semaphore::new(capacity)),
            capacity,
            held: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(0),
            waiting: AtomicUsize::new(0),
            last_warn_ms: AtomicU64::new(0),
            queued_warns: AtomicU64::new(0),
            reclaimed: AtomicU64::new(0),
            snapshot_to: None,
        }
    }

    /// The process-wide gate every production reasoner shares.
    pub fn global() -> Arc<CliGate> {
        static GLOBAL: OnceLock<Arc<CliGate>> = OnceLock::new();
        Arc::clone(GLOBAL.get_or_init(|| {
            let capacity = parse_capacity(std::env::var(ENV_MAX_INFLIGHT).ok().as_deref());
            info!(capacity, "reasoner CLI gate armed ({ENV_MAX_INFLIGHT})");
            let mut gate = CliGate::new(capacity);
            gate.snapshot_to = Some(snapshot_path());
            Arc::new(gate)
        }))
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    fn held(&self) -> MutexGuard<'_, HashMap<u64, Held>> {
        self.held.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Permits currently held — CLI children alive (or about to spawn).
    pub fn in_flight(&self) -> usize {
        self.held().len()
    }

    /// The permit held longest, and for how long. `None` when idle. Without
    /// it a wedged gate is invisible: the 15 h freeze showed up nowhere except
    /// as channels quietly not producing work (#954).
    pub fn oldest_held(&self) -> Option<(String, Duration)> {
        self.held()
            .values()
            .min_by_key(|h| h.since)
            .map(|h| (h.provider.clone(), h.since.elapsed()))
    }

    /// Permits force-released by the hold watchdog so far.
    pub fn reclaimed(&self) -> u64 {
        self.reclaimed.load(Ordering::Relaxed)
    }

    /// Callers currently queued for a permit.
    pub fn waiting(&self) -> usize {
        self.waiting.load(Ordering::SeqCst)
    }

    /// Queued-wait warnings emitted so far (test visibility for the
    /// once-a-minute cadence, which is otherwise only in the log).
    pub fn queued_warns(&self) -> u64 {
        self.queued_warns.load(Ordering::Relaxed)
    }

    /// Wait for a slot for at most `max_wait` — the caller's own watchdog
    /// budget, so a queued call fails the same way a hung child does instead
    /// of blocking forever (#954); there is deliberately no unbounded
    /// acquire, since that is precisely the freeze. `max_wait` doubles as
    /// the hold budget the granted permit promises to finish inside. While
    /// queued, warns once a minute naming the oldest holder; the saturation
    /// threshold deliberately does not gate that warning, because one silent
    /// waiter per channel is the failure shape we are looking for.
    pub async fn acquire_timed(
        self: &Arc<Self>,
        provider: &str,
        max_wait: Duration,
    ) -> Result<CliPermit, GateWaitTimeout> {
        let started = Instant::now();
        let _queued = WaitGuard::enter(self);
        if self.waiting() > self.capacity {
            self.warn_saturated(provider);
        }
        self.reclaim_stale();
        // Held across the loop so the caller keeps its place in the
        // semaphore's FIFO queue; cancel-safe, and dropped on the deadline
        // path without ever having taken a permit.
        let mut acquire = std::pin::pin!(Arc::clone(&self.sem).acquire_owned());
        // `sleep` (not `sleep_until`) so an absurd env-set budget saturates
        // instead of overflowing the deadline arithmetic.
        let mut expired = std::pin::pin!(tokio::time::sleep(max_wait));
        // `interval_at` so the first tick lands a minute in — a permit
        // granted immediately (the normal case) must not log anything.
        let mut ticker = tokio::time::interval_at(started + QUEUED_WARN_EVERY, QUEUED_WARN_EVERY);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let permit = loop {
            tokio::select! {
                granted = &mut acquire => {
                    break granted.expect("CLI gate semaphore is never closed");
                }
                _ = &mut expired => {
                    let waited_secs = started.elapsed().as_secs();
                    let (holder, held_secs) = self.oldest_held_fields();
                    warn!(
                        provider,
                        waited_secs,
                        in_flight = self.in_flight(),
                        waiting = self.waiting(),
                        capacity = self.capacity,
                        oldest_holder = holder,
                        oldest_held_secs = held_secs,
                        "reasoner CLI gate wait timed out; giving up on this call (#954)"
                    );
                    return Err(GateWaitTimeout {
                        provider: provider.to_string(),
                        waited_secs,
                    });
                }
                _ = ticker.tick() => {
                    self.reclaim_stale();
                    self.queued_warns.fetch_add(1, Ordering::Relaxed);
                    let (holder, held_secs) = self.oldest_held_fields();
                    warn!(
                        provider,
                        waited_secs = started.elapsed().as_secs(),
                        in_flight = self.in_flight(),
                        waiting = self.waiting(),
                        capacity = self.capacity,
                        oldest_holder = holder,
                        oldest_held_secs = held_secs,
                        "reasoner CLI gate: still queued for a permit"
                    );
                }
            }
        };
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let reclaimed = Arc::new(AtomicBool::new(false));
        let held = Held {
            provider: provider.to_string(),
            since: Instant::now(),
            budget: max_wait,
            reclaimed: Arc::clone(&reclaimed),
        };
        self.held().insert(id, held);
        self.write_snapshot();
        Ok(CliPermit {
            permit: Some(permit),
            gate: Arc::clone(self),
            id,
            reclaimed,
        })
    }

    fn oldest_held_fields(&self) -> (String, u64) {
        self.oldest_held()
            .map(|(p, age)| (p, age.as_secs()))
            .unwrap_or_else(|| ("none".to_string(), 0))
    }

    /// Hand back slots whose holder is never coming (#954). A permit held
    /// past [`STALE_HOLD_FACTOR`]× the budget its caller promised cannot have
    /// a live child behind it, and without this the leak narrows the gate for
    /// the life of the process — on 2026-09-04 all four slots went that way.
    /// The holder's eventual `Drop` forgets its semaphore permit rather than
    /// returning it, so the total never exceeds `capacity`.
    fn reclaim_stale(&self) {
        let stale: Vec<(String, u64)> = {
            let mut held = self.held();
            let ids: Vec<u64> = held
                .iter()
                .filter(|(_, h)| h.since.elapsed() > h.budget.saturating_mul(STALE_HOLD_FACTOR))
                .map(|(id, _)| *id)
                .collect();
            ids.iter()
                .filter_map(|id| held.remove(id))
                .map(|h| {
                    h.reclaimed.store(true, Ordering::SeqCst);
                    (h.provider, h.since.elapsed().as_secs())
                })
                .collect()
        };
        if stale.is_empty() {
            return;
        }
        self.sem.add_permits(stale.len());
        self.reclaimed
            .fetch_add(stale.len() as u64, Ordering::Relaxed);
        for (provider, held_secs) in stale {
            warn!(
                provider,
                held_secs,
                capacity = self.capacity,
                "reasoner CLI gate: permit held far past its budget; slot reclaimed (#954)"
            );
        }
        self.write_snapshot();
    }

    /// Publish the gate's state for `augmentagent doctor`, which runs in a
    /// different process and can otherwise see nothing. One short tmpfs line,
    /// best-effort: a diagnostic must never fail or slow a reasoner call.
    fn write_snapshot(&self) {
        let Some(path) = self.snapshot_to.as_ref() else {
            return;
        };
        let now = now_ms() / 1000;
        let oldest = self.oldest_held();
        let snap = GateSnapshot {
            pid: std::process::id(),
            capacity: self.capacity,
            in_flight: self.in_flight(),
            waiting: self.waiting(),
            oldest_provider: oldest.as_ref().map(|(p, _)| p.clone()),
            oldest_since_unix: oldest.map(|(_, age)| now.saturating_sub(age.as_secs())),
            updated_unix: now,
        };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(json) = serde_json::to_string(&snap) {
            let _ = std::fs::write(path, json);
        }
    }

    fn warn_saturated(&self, provider: &str) {
        let now = now_ms();
        let last = self.last_warn_ms.load(Ordering::Relaxed);
        let due = now.saturating_sub(last) >= SATURATION_WARN_EVERY.as_millis() as u64;
        if due
            && self
                .last_warn_ms
                .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
        {
            warn!(
                provider,
                waiting = self.waiting(),
                in_flight = self.in_flight(),
                capacity = self.capacity,
                "reasoner CLI gate saturated; calls are queueing (see {ENV_MAX_INFLIGHT})"
            );
        }
    }
}

/// Held for the lifetime of one CLI child. Dropping it frees the slot.
pub struct CliPermit {
    permit: Option<OwnedSemaphorePermit>,
    gate: Arc<CliGate>,
    id: u64,
    reclaimed: Arc<AtomicBool>,
}

impl Drop for CliPermit {
    fn drop(&mut self) {
        self.gate.held().remove(&self.id);
        if self.reclaimed.load(Ordering::SeqCst) {
            // The hold watchdog already handed this slot back; returning it
            // a second time would widen the gate past capacity, which is the
            // #897 OOM this gate exists to prevent.
            if let Some(p) = self.permit.take() {
                p.forget();
            }
        }
        self.gate.write_snapshot();
    }
}

/// The gate state `augmentagent doctor` reads out of another process (#954).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateSnapshot {
    pub pid: u32,
    pub capacity: usize,
    pub in_flight: usize,
    pub waiting: usize,
    pub oldest_provider: Option<String>,
    /// Unix seconds at which the oldest still-held permit was granted.
    pub oldest_since_unix: Option<u64>,
    pub updated_unix: u64,
}

/// `${XDG_RUNTIME_DIR}/augmentagent/reasoner-gate.json` — tmpfs, so a
/// snapshot cannot outlive the boot that wrote it.
pub fn snapshot_path() -> PathBuf {
    let dir = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(dir).join("augmentagent/reasoner-gate.json")
}

/// `None` when no daemon has ever armed the gate on this boot.
pub fn read_snapshot() -> Option<GateSnapshot> {
    serde_json::from_str(&std::fs::read_to_string(snapshot_path()).ok()?).ok()
}

/// Counts a queued caller; undone on drop so a cancelled wait cannot leak.
struct WaitGuard<'a> {
    gate: &'a CliGate,
}

impl<'a> WaitGuard<'a> {
    fn enter(gate: &'a CliGate) -> Self {
        gate.waiting.fetch_add(1, Ordering::SeqCst);
        Self { gate }
    }
}

impl Drop for WaitGuard<'_> {
    fn drop(&mut self) {
        self.gate.waiting.fetch_sub(1, Ordering::SeqCst);
    }
}

fn parse_capacity(raw: Option<&str>) -> usize {
    raw.and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_INFLIGHT)
        .max(1)
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default budget for tests that only care about slot bookkeeping.
    const BUDGET: Duration = Duration::from_secs(90);

    #[tokio::test]
    async fn permits_are_bounded_and_released() {
        let gate = Arc::new(CliGate::new(2));
        let a = gate.acquire_timed("t", BUDGET).await.unwrap();
        let b = gate.acquire_timed("t", BUDGET).await.unwrap();
        assert_eq!(gate.in_flight(), 2);

        let third = gate.acquire_timed("t", Duration::from_millis(50)).await;
        assert!(third.is_err(), "third permit must wait for a free slot");
        assert_eq!(gate.in_flight(), 2);

        drop(a);
        let c = gate
            .acquire_timed("t", BUDGET)
            .await
            .expect("a released slot is handed to the next caller");
        assert_eq!(gate.in_flight(), 2);
        drop(b);
        drop(c);
        assert_eq!(gate.in_flight(), 0);
        assert_eq!(gate.waiting(), 0);
    }

    #[tokio::test]
    async fn cancelled_wait_does_not_leak_waiting_count() {
        let gate = Arc::new(CliGate::new(1));
        let held = gate.acquire_timed("t", BUDGET).await.unwrap();
        let cancelled =
            tokio::time::timeout(Duration::from_millis(30), gate.acquire_timed("t", BUDGET)).await;
        assert!(cancelled.is_err());
        assert_eq!(gate.waiting(), 0, "a dropped waiter must not stay counted");
        drop(held);
        assert_eq!(gate.in_flight(), 0);
    }

    /// #954 verbatim: capacity 4, all four permits held by futures that will
    /// never drop them, so every reasoner call in the daemon queues behind
    /// them. Before this the queue was unbounded and silent — 15 h with no
    /// email triage, no auto-PR loop and no log line. Now the wait is bounded,
    /// the holder is nameable, and the leaked slots come back.
    #[tokio::test(start_paused = true)]
    async fn leaked_permits_are_reported_reclaimed_and_never_freeze_the_gate() {
        let gate = Arc::new(CliGate::new(DEFAULT_MAX_INFLIGHT));
        let budget = Duration::from_secs(900);
        let mut leaked = Vec::new();
        for _ in 0..DEFAULT_MAX_INFLIGHT {
            leaked.push(gate.acquire_timed("claude", budget).await.unwrap());
        }
        assert_eq!(gate.in_flight(), DEFAULT_MAX_INFLIGHT);
        assert_eq!(gate.queued_warns(), 0, "an uncontended grant logs nothing");

        let Err(err) = gate.acquire_timed("claude", budget).await else {
            panic!("every slot is held; nothing can be granted");
        };
        assert_eq!(err.provider, "claude");
        assert!(err.waited_secs >= 900, "waited {}s", err.waited_secs);
        assert_eq!(gate.waiting(), 0, "timed-out waiter must not stay counted");
        // Warned once a minute although `waiting()` never exceeded capacity —
        // exactly the case the saturation warning cannot see.
        assert!(gate.queued_warns() >= 14, "{}", gate.queued_warns());
        let (holder, age) = gate.oldest_held().expect("four permits are held");
        assert_eq!(holder, "claude");
        assert!(age >= budget, "oldest permit age {age:?}");

        // Past the hold budget the slots are reclaimed, so the leak does not
        // cost capacity for the life of the process.
        tokio::time::advance(Duration::from_secs(1_000)).await;
        let permit = gate
            .acquire_timed("claude", budget)
            .await
            .expect("stale permits are reclaimed");
        assert_eq!(gate.reclaimed(), DEFAULT_MAX_INFLIGHT as u64);
        assert_eq!(gate.in_flight(), 1);

        // The reclaimed holders forget their permits on drop, so capacity is
        // restored exactly — never widened.
        drop(leaked);
        drop(permit);
        assert!(gate.oldest_held().is_none());
        assert_eq!(gate.sem.available_permits(), DEFAULT_MAX_INFLIGHT);
    }

    #[test]
    fn capacity_parsing_is_defensive() {
        assert_eq!(parse_capacity(None), DEFAULT_MAX_INFLIGHT);
        assert_eq!(parse_capacity(Some(" 8 ")), 8);
        assert_eq!(parse_capacity(Some("0")), 1, "never a zero-capacity gate");
        assert_eq!(parse_capacity(Some("lots")), DEFAULT_MAX_INFLIGHT);
        assert_eq!(CliGate::new(0).capacity(), 1);
    }
}
