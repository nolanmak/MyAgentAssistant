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
//! best-effort and already async, so queueing is the correct backpressure;
//! bounded since #954 (below). A permit is taken immediately before
//! `Command::spawn` and lives in the same scope as the `Child`, so it is
//! released when the child is reaped, on every early-return error, and when
//! the caller's future is dropped (the #656 watchdog, shutdown).
//!
//! Capacity: `AUGMENTAGENT_REASONER_MAX_INFLIGHT` (default 4, minimum 1),
//! read once when the global gate is first used.
//!
//! #954 — on 2026-09-04 four permits were held by futures that were neither
//! running a child nor timing out, and every reasoner call queued behind them
//! for 15 h: no email triage, no auto-PR loop, no log line. So the wait is
//! bounded, it says so while it lasts, and a watchdog cancels overdue holds.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore};
use tokio::time::Instant;
use tracing::{info, warn};

pub const DEFAULT_MAX_INFLIGHT: usize = 4;
pub const ENV_MAX_INFLIGHT: &str = "AUGMENTAGENT_REASONER_MAX_INFLIGHT";
/// How long one call may sit queued in silence, and the floor between lines.
const QUEUED_WARN_EVERY: Duration = Duration::from_secs(60);
/// A permit held this many times its caller's budget is not going to finish —
/// #656 kills a live child at 1× — so its owner is told to give the slot up.
const STALE_HOLD_FACTOR: u32 = 2;
/// Sweep-period ceiling, and so the snapshot's heartbeat: an older snapshot
/// with holds outstanding means the sweep itself stopped.
pub const WATCHDOG_EVERY: Duration = Duration::from_secs(30);

/// A caller gave up waiting for a permit — ours, not the provider's fault.
#[derive(Debug, thiserror::Error)]
#[error("{provider} waited {waited_secs}s for a CLI gate permit")]
pub struct GateWaitTimeout {
    pub provider: String,
    pub waited_secs: u64,
}

/// One permit out, so a hold outliving its budget can be named (#954).
struct Held {
    provider: String,
    /// Which preset asked: "claude" cannot say *which* call leaked.
    caller: String,
    since: Instant,
    budget: Duration,
    /// Asks the owner to stop — the watchdog's only lever ([`CliGate::sweep_holds`]).
    revoke: Arc<Notify>,
    revoked: bool,
}

pub struct CliGate {
    sem: Arc<Semaphore>,
    capacity: usize,
    held: Mutex<HashMap<u64, Held>>,
    next_id: AtomicU64,
    waiting: AtomicUsize,
    last_warn_ms: AtomicU64,
    /// A field, not the const, so tests can compress a minute to milliseconds.
    warn_every: Duration,
    queued_warnings: AtomicU64,
    /// Whether a hold-watchdog task is live for this gate.
    sweeping: AtomicBool,
    revoked: AtomicU64,
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
            warn_every: QUEUED_WARN_EVERY,
            queued_warnings: AtomicU64::new(0),
            sweeping: AtomicBool::new(false),
            revoked: AtomicU64::new(0),
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

    /// The permit held longest, and for how long; `None` when idle (#954).
    pub fn oldest_held(&self) -> Option<(String, Duration)> {
        self.held()
            .values()
            .min_by_key(|h| h.since)
            .map(|h| (h.provider.clone(), h.since.elapsed()))
    }

    /// Holds the watchdog has asked their owners to give up so far.
    pub fn revocations(&self) -> u64 {
        self.revoked.load(Ordering::Relaxed)
    }

    /// Callers currently queued for a permit.
    pub fn waiting(&self) -> usize {
        self.waiting.load(Ordering::SeqCst)
    }

    /// Test visibility for what otherwise only reaches the log.
    pub fn queued_warnings(&self) -> u64 {
        self.queued_warnings.load(Ordering::Relaxed)
    }

    /// Wait for a slot for at most `max_wait` — the caller's own watchdog
    /// budget, so a queued call fails the way a hung child does instead of
    /// blocking forever (#954). It doubles as the budget the permit promises
    /// to finish inside, policed by the watchdog below.
    pub async fn acquire_timed(
        self: &Arc<Self>,
        provider: &str,
        caller: &str,
        max_wait: Duration,
    ) -> Result<CliPermit, GateWaitTimeout> {
        let started = Instant::now();
        let deadline = started + max_wait;
        let queued = WaitGuard::enter(self);
        // Waited in slices, not one `timeout`: a caller stuck behind a leaked
        // permit must say so every `warn_every`, not only when it gives up —
        // the freeze never queued more callers than the gate is wide, so a
        // depth-gated warning stayed silent for 15 h. Carrying (never
        // re-creating) the future keeps the caller's place in the FIFO queue.
        let acquire = Arc::clone(&self.sem).acquire_owned();
        tokio::pin!(acquire);
        let granted = loop {
            let left = deadline.saturating_duration_since(Instant::now());
            let slice = left.min(self.warn_every);
            match tokio::time::timeout(slice, &mut acquire).await {
                Ok(granted) => break granted.expect("CLI gate semaphore is never closed"),
                Err(_) if slice < left => self.warn_queued(provider, caller, started, false),
                Err(_) => {
                    self.warn_queued(provider, caller, started, true);
                    let (provider, waited_secs) =
                        (provider.to_string(), started.elapsed().as_secs());
                    return Err(GateWaitTimeout { provider, waited_secs });
                }
            }
        };
        // Uncount first, or the published `waiting` includes this caller.
        drop(queued);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (provider, caller) = (provider.to_string(), caller.to_string());
        let revoke = Arc::new(Notify::new());
        let (r, since, budget) = (Arc::clone(&revoke), Instant::now(), max_wait);
        self.held()
            .insert(id, Held { provider, caller, since, budget, revoke: r, revoked: false });
        self.arm_watchdog();
        self.write_snapshot();
        Ok(CliPermit { _permit: granted, gate: Arc::clone(self), id, revoke })
    }

    /// One sweeper task per gate, armed by the first grant. Without it a leak
    /// is only noticed by the *next* acquire — and "there is no next call" is
    /// exactly #954: every caller left, and a human found the leak 15 h later.
    fn arm_watchdog(self: &Arc<Self>) {
        if self.sweeping.swap(true, Ordering::SeqCst) {
            return;
        }
        let gate = Arc::clone(self);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(gate.sweep_period()).await;
                gate.sweep_holds();
                gate.write_snapshot();
            }
        });
    }

    /// Sweep at the shortest budget policed: an overdue hold must be caught
    /// near its own deadline, not at the next half-minute.
    fn sweep_period(&self) -> Duration {
        let soonest = self.held().values().map(|h| h.budget).min();
        soonest.unwrap_or(WATCHDOG_EVERY).clamp(Duration::from_millis(10), WATCHDOG_EVERY)
    }

    /// One watchdog pass. A hold past [`STALE_HOLD_FACTOR`]× its caller's
    /// budget is named, and its owner asked to give the slot up.
    ///
    /// Asked, not taken: the owner may still have a CLI child attached, and
    /// `add_permits` would let a fifth child start beside it — the overcommit
    /// #897 OOM-killed the box for. Cancelling the owner drops its call future
    /// instead, which kills the child (`kill_on_drop`) and drops the permit, so
    /// a slot only ever comes back through [`CliPermit::drop`].
    fn sweep_holds(&self) {
        for h in self.held().values_mut() {
            let age = h.since.elapsed();
            if age <= h.budget.saturating_mul(STALE_HOLD_FACTOR) {
                continue;
            }
            // Idempotent, and re-sent every sweep until the owner lets go.
            h.revoke.notify_one();
            if std::mem::replace(&mut h.revoked, true) {
                continue; // Already named; `doctor` carries it from here.
            }
            self.revoked.fetch_add(1, Ordering::Relaxed);
            warn!(
                provider = h.provider,
                caller = h.caller,
                held_secs = age.as_secs(),
                budget_secs = h.budget.as_secs(),
                "reasoner CLI gate: permit held far past its caller's budget; \
                 owner asked to cancel and free the slot (#954)"
            );
        }
    }

    /// Publish the gate's state for `augmentagent doctor`, which runs in its
    /// own process. Best-effort: a diagnostic must never fail a reasoner call.
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

    /// Name a caller that is still queued — every `warn_every` it waits, and
    /// once more if it gives up. Never gated on queue depth, and it names the
    /// hold it is stuck behind: the diagnosis #954 never printed.
    fn warn_queued(&self, provider: &str, caller: &str, started: Instant, gave_up: bool) {
        if !gave_up {
            // One line per interval per gate: a burst must not become 1,700.
            let (now, every) = (now_ms(), self.warn_every.as_millis() as u64);
            let claimed = self.last_warn_ms.fetch_update(
                Ordering::Relaxed,
                Ordering::Relaxed,
                |last| (now.saturating_sub(last) >= every).then_some(now),
            );
            if claimed.is_err() {
                return;
            }
            self.queued_warnings.fetch_add(1, Ordering::Relaxed);
        }
        let (holder, held_secs) = self
            .oldest_held()
            .map_or_else(|| ("none".to_string(), 0), |(p, age)| (p, age.as_secs()));
        warn!(
            provider,
            caller,
            gave_up,
            waited_secs = started.elapsed().as_secs(),
            waiting = self.waiting(),
            in_flight = self.in_flight(),
            oldest_holder = holder,
            oldest_held_secs = held_secs,
            "reasoner CLI gate: call queued behind held permits (#954, {ENV_MAX_INFLIGHT})"
        );
    }
}

/// Held for the lifetime of one CLI child. Dropping it frees the slot — and
/// dropping it is the *only* thing that frees a slot, so the gate can never
/// admit a caller while an earlier child is still resident.
pub struct CliPermit {
    _permit: OwnedSemaphorePermit,
    gate: Arc<CliGate>,
    id: u64,
    revoke: Arc<Notify>,
}

impl CliPermit {
    /// Resolves when the hold watchdog gives up on this permit (#954). The
    /// owner must then stop: dropping its call future kills the CLI child and
    /// drops this permit, which is what actually returns the slot.
    pub async fn revoked(&self) {
        self.revoke.notified().await;
    }
}

impl Drop for CliPermit {
    fn drop(&mut self) {
        self.gate.held().remove(&self.id);
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
    pub oldest_since_unix: Option<u64>,
    pub updated_unix: u64,
}

/// `${XDG_RUNTIME_DIR}/augmentagent/reasoner-gate.json` — tmpfs, so per boot.
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

    /// Budget for tests that only care about slot bookkeeping.
    const BUDGET: Duration = Duration::from_secs(90);

    #[tokio::test]
    async fn permits_are_bounded_and_released() {
        let gate = Arc::new(CliGate::new(2));
        let a = gate.acquire_timed("t", "triage", BUDGET).await.unwrap();
        let b = gate.acquire_timed("t", "triage", BUDGET).await.unwrap();
        assert_eq!(gate.in_flight(), 2);

        let third = gate.acquire_timed("t", "triage", Duration::from_millis(50)).await;
        assert!(third.is_err(), "third permit must wait for a free slot");
        assert_eq!(gate.in_flight(), 2);

        drop(a);
        let c = gate.acquire_timed("t", "triage", BUDGET).await;
        assert!(c.is_ok(), "a released slot is handed to the next caller");
        assert_eq!(gate.in_flight(), 2);
        drop(b);
        drop(c);
        assert_eq!(gate.in_flight(), 0);
        assert_eq!(gate.waiting(), 0);
    }

    #[tokio::test]
    async fn cancelled_wait_does_not_leak_waiting_count() {
        let gate = Arc::new(CliGate::new(1));
        let held = gate.acquire_timed("t", "triage", BUDGET).await.unwrap();
        let waiter = gate.acquire_timed("t", "triage", BUDGET);
        let cancelled = tokio::time::timeout(Duration::from_millis(30), waiter).await;
        assert!(cancelled.is_err());
        assert_eq!(gate.waiting(), 0, "a dropped waiter must not stay counted");
        drop(held);
        assert_eq!(gate.in_flight(), 0);
    }

    /// #954 verbatim: capacity 4, all four permits held by futures that are
    /// neither running a child nor timing out, so every reasoner call queues
    /// behind them — 15 h with no email triage, no auto-PR loop and no log
    /// line. The one caller that does arrive gives up and leaves, so **nothing
    /// calls the gate again**: neither the warning nor the recovery may wait
    /// for a later acquire.
    #[tokio::test]
    async fn leaked_permits_are_reported_and_revoked_without_a_later_acquire() {
        let budget = Duration::from_millis(300);
        let mut gate = CliGate::new(DEFAULT_MAX_INFLIGHT);
        gate.warn_every = budget / 5;
        let gate = Arc::new(gate);
        let mut owners = Vec::new();
        for _ in 0..DEFAULT_MAX_INFLIGHT {
            let gate = Arc::clone(&gate);
            owners.push(tokio::spawn(async move {
                let permit = gate.acquire_timed("claude", "triage", budget).await.unwrap();
                permit.revoked().await; // A parked call's only way out.
            }));
        }
        while gate.in_flight() < DEFAULT_MAX_INFLIGHT {
            tokio::task::yield_now().await;
        }
        let Err(err) = gate.acquire_timed("claude", "triage", budget).await else {
            panic!("every slot is held; nothing can be granted");
        };
        assert_eq!(err.provider, "claude");
        assert_eq!(gate.waiting(), 0, "timed-out waiter must not stay counted");
        assert_eq!(gate.oldest_held().expect("four are held").0, "claude");
        // One call behind four stuck holds never makes the queue deeper than
        // the gate, so it has to name itself *while waiting*: waiting for it
        // to give up is the 15 h of silence, at scale.
        let warned = gate.queued_warnings();
        let want = "a queued caller must be named every interval it waits, not only at the end";
        assert!(warned >= 2, "{want} (saw {warned})");

        // Not one more acquire from here: the watchdog task alone must name
        // the leak and unstick it, or the gate stays narrow for good.
        for owner in owners {
            tokio::time::timeout(budget * 10, owner).await.expect("revoked").unwrap();
        }
        // Each leaked hold was named in the log, exactly once.
        assert_eq!(gate.revocations(), DEFAULT_MAX_INFLIGHT as u64);
        assert!(gate.oldest_held().is_none());
        assert_eq!(gate.sem.available_permits(), DEFAULT_MAX_INFLIGHT);
        gate.acquire_timed("claude", "triage", budget).await.expect("gate is usable");
    }

    /// #954 review — the watchdog may *ask* for a slot back but must never
    /// take it: an owner that ignores revocation can still have a CLI child
    /// resident, and admitting a second one beside it is the #897 overcommit.
    #[tokio::test]
    async fn a_revoked_hold_that_never_lets_go_keeps_its_slot() {
        let budget = Duration::from_millis(100);
        let gate = Arc::new(CliGate::new(1));
        let deaf = gate.acquire_timed("claude", "triage", budget).await.unwrap();
        let deadline = Instant::now() + budget * 20;
        while gate.revocations() == 0 && Instant::now() < deadline {
            tokio::time::sleep(budget / 4).await;
        }
        assert_eq!(gate.revocations(), 1, "the watchdog must ask");
        assert_eq!(gate.sem.available_permits(), 0, "but never widen the gate");
        assert!(gate.acquire_timed("claude", "triage", budget).await.is_err());
        assert_eq!(gate.in_flight(), 1);
        drop(deaf);
        assert_eq!(gate.sem.available_permits(), 1, "the owner's Drop frees it");
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
