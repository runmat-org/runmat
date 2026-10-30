//! Shared monotone probe timestamps without a mutex on every geometric predicate.
//!
//! Checkpoint latency bounds the algorithm work a thread does between cancellation checks. Wall
//! time also counts time the operating system did not run the thread, so on a loaded machine it
//! would fail a mesh for the scheduler's choices. A latency that reaches
//! [`CPU_CHARGE_THRESHOLD_NS`] of wall time is therefore charged as the CPU time the observing
//! thread consumed since its previous probe of this clock (sampled at most every
//! [`CPU_SAMPLE_INTERVAL_NS`]), never more than the wall time. Short latencies stay on the
//! wall-clock fast path.
use runmat_time::Instant;
use std::{
    cell::Cell,
    sync::atomic::{AtomicU64, Ordering},
};

const CPU_CHARGE_THRESHOLD_NS: u64 = 100_000_000;
const CPU_SAMPLE_INTERVAL_NS: u64 = 50_000_000;

thread_local! {
    /// (clock identity, clock-relative wall ns, thread CPU ns) at this thread's last CPU sample.
    static CPU_SAMPLE: Cell<(usize, u64, u64)> = const { Cell::new((0, 0, 0)) };
}

#[cfg(unix)]
fn thread_cpu_ns() -> Option<u64> {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `time` is a valid, writable timespec for the duration of the call.
    let status = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut time) };
    (status == 0).then(|| {
        u64::try_from(time.tv_sec)
            .unwrap_or(0)
            .saturating_mul(1_000_000_000)
            .saturating_add(u64::try_from(time.tv_nsec).unwrap_or(0))
    })
}

#[cfg(not(unix))]
fn thread_cpu_ns() -> Option<u64> {
    None
}

pub(crate) struct CheckpointClock {
    /// Process-unique: a thread's CPU sample belongs to one clock. An address would be reused
    /// by a later clock allocated where a dropped one lived, which then inherited its sample.
    identity: usize,
    origin: Instant,
    latest_elapsed_ns: AtomicU64,
}

static NEXT_CLOCK_IDENTITY: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);

impl CheckpointClock {
    pub(crate) fn new(origin: Instant) -> Self {
        Self {
            identity: NEXT_CLOCK_IDENTITY.fetch_add(1, Ordering::Relaxed),
            origin,
            latest_elapsed_ns: AtomicU64::new(0),
        }
    }

    fn identity(&self) -> usize {
        self.identity
    }

    fn elapsed_ns(&self, now: Instant) -> u64 {
        u64::try_from(now.saturating_duration_since(self.origin).as_nanos()).unwrap_or(u64::MAX)
    }

    /// Wall latency, charged as thread CPU time once it reaches the charging threshold.
    fn charged_ns(&self, wall_ns: u64) -> u64 {
        if wall_ns < CPU_CHARGE_THRESHOLD_NS {
            return wall_ns;
        }
        let (clock, _, sampled_cpu) = CPU_SAMPLE.with(Cell::get);
        if clock != self.identity() {
            return wall_ns;
        }
        thread_cpu_ns()
            .map(|cpu| cpu.saturating_sub(sampled_cpu).min(wall_ns))
            .unwrap_or(wall_ns)
    }

    fn sample(&self, elapsed_ns: u64) {
        let (clock, sampled_wall, _) = CPU_SAMPLE.with(Cell::get);
        if clock == self.identity()
            && elapsed_ns.saturating_sub(sampled_wall) < CPU_SAMPLE_INTERVAL_NS
        {
            return;
        }
        if let Some(cpu) = thread_cpu_ns() {
            CPU_SAMPLE.with(|sample| sample.set((self.identity(), elapsed_ns, cpu)));
        }
    }

    pub(crate) fn latency_ms(&self, now: Instant) -> u64 {
        let wall = self
            .elapsed_ns(now)
            .saturating_sub(self.latest_elapsed_ns.load(Ordering::Relaxed));
        self.charged_ns(wall) / 1_000_000
    }

    pub(crate) fn observe(&self, now: Instant) -> u64 {
        let elapsed = self.elapsed_ns(now);
        // A preempted earlier probe cannot overwrite a newer worker's timestamp.
        let previous = self.latest_elapsed_ns.fetch_max(elapsed, Ordering::Relaxed);
        let latency = self.charged_ns(elapsed.saturating_sub(previous));
        self.sample(elapsed);
        latency / 1_000_000
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn out_of_order_worker_probes_preserve_monotone_latency() {
        let origin = Instant::now();
        let clock = CheckpointClock::new(origin);
        assert_eq!(clock.observe(origin + Duration::from_millis(20)), 20);
        assert_eq!(clock.observe(origin + Duration::from_millis(5)), 0);
        assert_eq!(clock.latency_ms(origin + Duration::from_millis(25)), 5);
        assert_eq!(clock.observe(origin + Duration::from_millis(30)), 10);
    }

    #[test]
    fn concurrent_workers_cannot_regress_the_retained_probe() {
        let origin = Instant::now();
        let clock = CheckpointClock::new(origin);
        std::thread::scope(|scope| {
            for worker in 0..8 {
                let clock = &clock;
                scope.spawn(move || {
                    for step in (0..1000_usize).rev() {
                        clock.observe(
                            origin + Duration::from_millis((step * 8 + worker + 1) as u64),
                        );
                    }
                });
            }
        });
        assert_eq!(clock.latency_ms(origin + Duration::from_millis(8000)), 0);
        assert_eq!(clock.latency_ms(origin + Duration::from_millis(8007)), 7);
    }

    #[cfg(unix)]
    #[test]
    fn descheduled_wall_time_is_not_charged_as_checkpoint_latency() {
        let origin = Instant::now();
        let clock = CheckpointClock::new(origin);
        clock.observe(origin + Duration::from_millis(1));
        // Two seconds of wall time pass while this thread does almost no work.
        let latency = clock.latency_ms(origin + Duration::from_millis(2_001));
        assert!(latency < 1_000, "charged {latency} ms of idle wall time");
        // Without a sample on this thread the wall time is charged conservatively.
        let other = CheckpointClock::new(origin);
        assert_eq!(
            other.latency_ms(origin + Duration::from_millis(2_000)),
            2_000
        );
    }
}
