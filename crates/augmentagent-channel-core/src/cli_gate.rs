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
//! line. So: the wait is bounded by the caller's own #656 budget; an
//! independent watchdog task — not the next caller, who may never come —
//! names holds past that budget, reclaims their slots, and publishes
//! [`snapshot_path`] for `augmentagent doctor`, which runs in its own process.

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
/// A permit held this many times its caller's budget has no live child behind
/// it — #656 kills at 1× and drops the permit with the future.
const STALE_HOLD_FACTOR: u32 = 2;
/// Upper bound on the hold watchdog's sweep period (it also sweeps at the
/// shortest budget it polices). Doubles as the snapshot's heartbeat: while
/// anything is held, an older snapshot means the sweep itself stopped.
pub const WATCHDOG_EVERY: Duration = Duration::from_secs(30);

/// A caller gave up waiting for a permit — ours, not the provider's fault.
#[derive(Debug, thiserror::Error)]
#[error("{provider} waited {waited_secs}s for a CLI gate permit")]
pub struct GateWaitTimeout {
    pub provider: String,
    pub waited_secs: u64,
}

/// One permit out, tracked so a hold outliving its caller's budget can be
/// named and reclaimed (#954).
struct Held {
    provider: String,
    /// Which preset asked — "claude" alone cannot say *which* call leaked.
    caller: String,
    since: Instant,
    budget: Duration,
    warned: bool,
}

pub struct CliGate {
    sem: Arc<Semaphore>,
    capacity: usize,
    held: Mutex<HashMap<u64, Held>>,
    next_id: AtomicU64,
    waiting: AtomicUsize,
    last_warn_ms: AtomicU64,
    /// Whether a hold-watchdog task is live for this gate.
    sweeping: AtomicBool,
    overdue: AtomicU64,
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
            sweeping: AtomicBool::new(false),
            overdue: AtomicU64::new(0),
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

    /// The permit held longest, and for how long — without it a wedged gate is
    /// invisible, as the 15 h freeze was (#954). `None` when idle.
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

    /// Holds named past their caller's budget — test visibility for a watchdog
    /// that otherwise only speaks to the log.
    pub fn overdue_reports(&self) -> u64 {
        self.overdue.load(Ordering::Relaxed)
    }

    /// Wait for a slot for at most `max_wait` — the caller's own watchdog
    /// budget, so a queued call fails the same way a hung child does instead
    /// of blocking forever (#954); there is deliberately no unbounded acquire,
    /// since that is precisely the freeze. `max_wait` doubles as the budget
    /// the permit promises to finish inside, policed by the watchdog armed
    /// below, which names the holder by its `caller` preset.
    pub async fn acquire_timed(
        self: &Arc<Self>,
        provider: &str,
        caller: &str,
        max_wait: Duration,
    ) -> Result<CliPermit, GateWaitTimeout> {
        let started = Instant::now();
        let queued = WaitGuard::enter(self);
        if self.waiting() > self.capacity {
            self.warn_saturated(provider);
        }
        // Cancel-safe: on the deadline the acquire future is dropped, leaving
        // the semaphore's FIFO queue without ever having taken a permit.
        let Ok(granted) =
            tokio::time::timeout(max_wait, Arc::clone(&self.sem).acquire_owned()).await
        else {
            let waited_secs = started.elapsed().as_secs();
            let (holder, held_secs) = self
                .oldest_held()
                .map_or_else(|| ("none".to_string(), 0), |(p, age)| (p, age.as_secs()));
            warn!(
                provider,
                caller,
                waited_secs,
                in_flight = self.in_flight(),
                waiting = self.waiting(),
                oldest_holder = holder,
                oldest_held_secs = held_secs,
                "reasoner CLI gate wait timed out; giving up on this call (#954)"
            );
            let provider = provider.to_string();
            return Err(GateWaitTimeout { provider, waited_secs });
        };
        // Stop counting this caller as queued *before* publishing, or `doctor`
        // reads a `waiting` that includes the caller who just got in; from
        // here the watchdog's heartbeat keeps that count fresh, and a later
        // waiter can only exist while permits are held, i.e. while it sweeps.
        drop(queued);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let held = Held {
            provider: provider.to_string(),
            caller: caller.to_string(),
            since: Instant::now(),
            budget: max_wait,
            warned: false,
        };
        self.held().insert(id, held);
        self.arm_watchdog();
        self.write_snapshot();
        Ok(CliPermit {
            permit: Some(granted.expect("CLI gate semaphore is never closed")),
            gate: Arc::clone(self),
            id,
        })
    }

    /// One sweeper task per gate, armed by the first grant and living as long
    /// as the gate. Without it a leak is only ever noticed by the *next*
    /// acquire — and "there is no next call" is exactly the #954 freeze: every
    /// caller timed out and left, and the four leaked permits were found 15 h
    /// later, by a human. Its snapshot write is the heartbeat that lets
    /// `doctor` tell a swept gate from a stopped one.
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

    /// Sweep at least as often as the shortest budget being policed, so a
    /// leaked slot returns near its own deadline, not the next half-minute.
    fn sweep_period(&self) -> Duration {
        self.held()
            .values()
            .map(|h| h.budget)
            .min()
            .unwrap_or(WATCHDOG_EVERY)
            .clamp(Duration::from_millis(10), WATCHDOG_EVERY)
    }

    /// One watchdog pass. A permit past its caller's own budget is named once
    /// — #656 kills a live child at 1×, so a longer hold has nothing behind it
    /// — and past [`STALE_HOLD_FACTOR`]× its slot is handed back, since a leak
    /// must not narrow the gate for the life of the process. The holder's
    /// `Drop` then forgets its permit rather than returning it, so the total
    /// never exceeds `capacity`.
    fn sweep_holds(&self) {
        let mut freed = 0usize;
        self.held().retain(|_, h| {
            let age = h.since.elapsed();
            if age <= h.budget {
                return true;
            }
            let stale = age > h.budget.saturating_mul(STALE_HOLD_FACTOR);
            let first = !std::mem::replace(&mut h.warned, true);
            if first {
                self.overdue.fetch_add(1, Ordering::Relaxed);
            }
            if stale {
                freed += 1;
            } else if !first {
                // Already named; nothing new to say until it goes stale.
                return true;
            }
            warn!(
                provider = h.provider,
                caller = h.caller,
                held_secs = age.as_secs(),
                budget_secs = h.budget.as_secs(),
                stale,
                "reasoner CLI gate: permit held past its caller's budget; \
                 slot reclaimed once stale (#954)"
            );
            !stale
        });
        self.sem.add_permits(freed);
        self.reclaimed.fetch_add(freed as u64, Ordering::Relaxed);
    }

    /// Publish the gate's state for `augmentagent doctor`, which runs in its
    /// own process and can otherwise see nothing. One short tmpfs line,
    /// best-effort — a diagnostic must never fail or slow a reasoner call.
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
}

impl Drop for CliPermit {
    fn drop(&mut self) {
        if self.gate.held().remove(&self.id).is_none() {
            // Gone from the map means the watchdog already handed this slot
            // back; returning it twice would widen the gate past capacity —
            // the #897 OOM it exists to prevent.
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

/// `${XDG_RUNTIME_DIR}/augmentagent/reasoner-gate.json` — tmpfs, so it cannot
/// outlive its boot.
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

    /// #954 verbatim: capacity 4, all four permits held by futures that will
    /// never drop them, so every reasoner call in the daemon queues behind
    /// them — 15 h with no email triage, no auto-PR loop and no log line. The
    /// one caller that does arrive then gives up and leaves, so from there
    /// **nothing calls the gate again**: recovery cannot depend on a later
    /// acquire. Real clock at ms scale — tokio's paused clock needs
    /// `test-util`, which forks the tokio-dependent build graph in two.
    #[tokio::test]
    async fn leaked_permits_are_reported_and_reclaimed_without_a_later_acquire() {
        let gate = Arc::new(CliGate::new(DEFAULT_MAX_INFLIGHT));
        let budget = Duration::from_millis(300);
        let mut leaked = Vec::new();
        for _ in 0..DEFAULT_MAX_INFLIGHT {
            leaked.push(gate.acquire_timed("claude", "triage", budget).await.unwrap());
        }
        let Err(err) = gate.acquire_timed("claude", "triage", budget).await else {
            panic!("every slot is held; nothing can be granted");
        };
        assert_eq!(err.provider, "claude");
        assert_eq!(gate.waiting(), 0, "timed-out waiter must not stay counted");
        assert_eq!(gate.oldest_held().expect("four are held").0, "claude");

        // Not one more acquire from here: the watchdog task alone must name the
        // leak and hand the slots back, or the gate stays narrowed for good.
        let deadline = Instant::now() + budget * 10;
        while gate.reclaimed() < DEFAULT_MAX_INFLIGHT as u64 && Instant::now() < deadline {
            tokio::time::sleep(budget / 4).await;
        }
        assert_eq!(gate.reclaimed(), DEFAULT_MAX_INFLIGHT as u64);
        // Each leaked hold was also named in the log, exactly once.
        assert_eq!(gate.overdue_reports(), DEFAULT_MAX_INFLIGHT as u64);
        assert!(gate.oldest_held().is_none());

        // The reclaimed holders forget their permits on drop, so capacity is
        // restored exactly — never widened — and the gate takes calls again.
        drop(leaked);
        assert_eq!(gate.sem.available_permits(), DEFAULT_MAX_INFLIGHT);
        gate.acquire_timed("claude", "triage", budget).await.expect("gate is usable");
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
