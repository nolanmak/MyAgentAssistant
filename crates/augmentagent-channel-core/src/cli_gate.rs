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
//! #954 — waiting for a permit is *bounded*. On 2026-09-04 four permits were
//! held by futures that were neither running a child nor timing out, and
//! every reasoner call in the daemon queued behind them for 15 h: no email
//! triage, no auto-PR loop, no log line (the saturation warning below only
//! fires *above* capacity, and one waiter per channel never reaches it).
//! [`CliGate::acquire_timed`] therefore carries the caller's own #656
//! watchdog budget — a queued call now fails the way a hung child does, so
//! the chain fails over to an ungated provider — and warns once a minute
//! while it waits, saturated or not.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

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

/// A caller gave up waiting for a permit. Ours, not the provider's — the
/// adapters map it to `ReasonerError::GateTimeout`, which latches nothing.
#[derive(Debug, thiserror::Error)]
#[error("{provider} waited {waited_secs}s for a CLI gate permit")]
pub struct GateWaitTimeout {
    pub provider: String,
    pub waited_secs: u64,
}

pub struct CliGate {
    sem: Arc<Semaphore>,
    capacity: usize,
    in_flight: AtomicUsize,
    waiting: AtomicUsize,
    last_warn_ms: AtomicU64,
    queued_warns: AtomicU64,
}

impl CliGate {
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            sem: Arc::new(Semaphore::new(capacity)),
            capacity,
            in_flight: AtomicUsize::new(0),
            waiting: AtomicUsize::new(0),
            last_warn_ms: AtomicU64::new(0),
            queued_warns: AtomicU64::new(0),
        }
    }

    /// The process-wide gate every production reasoner shares.
    pub fn global() -> Arc<CliGate> {
        static GLOBAL: OnceLock<Arc<CliGate>> = OnceLock::new();
        Arc::clone(GLOBAL.get_or_init(|| {
            let capacity = parse_capacity(std::env::var(ENV_MAX_INFLIGHT).ok().as_deref());
            info!(capacity, "reasoner CLI gate armed ({ENV_MAX_INFLIGHT})");
            Arc::new(CliGate::new(capacity))
        }))
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Permits currently held — CLI children alive (or about to spawn).
    pub fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::SeqCst)
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

    /// Wait for a slot, unbounded. Never fails (the semaphore is never
    /// closed). Cancellation-safe: a caller dropped while queued leaves no
    /// trace. Production reasoner calls use
    /// [`acquire_timed`](Self::acquire_timed) instead — an unbounded wait is
    /// exactly the #954 freeze.
    pub async fn acquire(self: &Arc<Self>, provider: &str) -> CliPermit {
        let _queued = WaitGuard::enter(self);
        if self.waiting() > self.capacity {
            self.warn_saturated(provider);
        }
        let permit = Arc::clone(&self.sem)
            .acquire_owned()
            .await
            .expect("CLI gate semaphore is never closed");
        self.in_flight.fetch_add(1, Ordering::SeqCst);
        CliPermit {
            _permit: permit,
            gate: Arc::clone(self),
        }
    }

    /// Wait for a slot for at most `max_wait` — the caller's own watchdog
    /// budget, so a queued call fails the same way a hung child does instead
    /// of blocking forever (#954). While queued, warns once a minute with
    /// the gate's state; the saturation threshold deliberately does not gate
    /// that warning, because one silent waiter per channel is the failure
    /// shape we are looking for.
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
                    warn!(
                        provider,
                        waited_secs,
                        in_flight = self.in_flight(),
                        waiting = self.waiting(),
                        capacity = self.capacity,
                        "reasoner CLI gate wait timed out; giving up on this call (#954)"
                    );
                    return Err(GateWaitTimeout {
                        provider: provider.to_string(),
                        waited_secs,
                    });
                }
                _ = ticker.tick() => {
                    self.queued_warns.fetch_add(1, Ordering::Relaxed);
                    warn!(
                        provider,
                        waited_secs = started.elapsed().as_secs(),
                        in_flight = self.in_flight(),
                        waiting = self.waiting(),
                        capacity = self.capacity,
                        "reasoner CLI gate: still queued for a permit"
                    );
                }
            }
        };
        self.in_flight.fetch_add(1, Ordering::SeqCst);
        Ok(CliPermit {
            _permit: permit,
            gate: Arc::clone(self),
        })
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
    _permit: OwnedSemaphorePermit,
    gate: Arc<CliGate>,
}

impl Drop for CliPermit {
    fn drop(&mut self) {
        self.gate.in_flight.fetch_sub(1, Ordering::SeqCst);
    }
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

    #[tokio::test]
    async fn permits_are_bounded_and_released() {
        let gate = Arc::new(CliGate::new(2));
        let a = gate.acquire("t").await;
        let b = gate.acquire("t").await;
        assert_eq!(gate.in_flight(), 2);

        let third = tokio::time::timeout(Duration::from_millis(50), gate.acquire("t")).await;
        assert!(third.is_err(), "third permit must wait for a free slot");
        assert_eq!(gate.in_flight(), 2);

        drop(a);
        let c = tokio::time::timeout(Duration::from_millis(500), gate.acquire("t"))
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
        let held = gate.acquire("t").await;
        let cancelled = tokio::time::timeout(Duration::from_millis(30), gate.acquire("t")).await;
        assert!(cancelled.is_err());
        assert_eq!(gate.waiting(), 0, "a dropped waiter must not stay counted");
        drop(held);
        assert_eq!(gate.in_flight(), 0);
    }

    /// #954 — the freeze shape: a permit nobody will ever release. Before
    /// this, a queued caller waited forever with no log line and no error.
    #[tokio::test(start_paused = true)]
    async fn queued_acquire_times_out_instead_of_hanging() {
        let gate = Arc::new(CliGate::new(1));
        let leaked = gate.acquire("t").await;

        let Err(err) = gate.acquire_timed("t", Duration::from_secs(90)).await else {
            panic!("the only permit is held; nothing can be granted");
        };
        assert_eq!(err.provider, "t");
        assert!(err.waited_secs >= 90, "waited {}s", err.waited_secs);
        assert_eq!(gate.waiting(), 0, "timed-out waiter must not stay counted");
        // The queued warning fires once a minute even though `waiting()`
        // never exceeded capacity — the exact silence seen on 2026-09-04.
        assert_eq!(gate.queued_warns(), 1);

        drop(leaked);
        assert_eq!(gate.in_flight(), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn a_fast_acquire_never_warns() {
        let gate = Arc::new(CliGate::new(1));
        let permit = gate
            .acquire_timed("t", Duration::from_secs(90))
            .await
            .expect("a free gate hands out a permit immediately");
        assert_eq!(gate.queued_warns(), 0);
        assert_eq!(gate.in_flight(), 1);
        drop(permit);
        assert_eq!(gate.in_flight(), 0);
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
