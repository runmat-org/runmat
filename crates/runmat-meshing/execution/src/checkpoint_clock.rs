//! Shared monotone probe timestamps without a mutex on every geometric predicate.
//! Every probe still reads the clock; deadline and latency limits are unchanged.
use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

pub(crate) struct CheckpointClock {
    origin: Instant,
    latest_elapsed_ns: AtomicU64,
}

impl CheckpointClock {
    pub(crate) fn new(origin: Instant) -> Self {
        Self {
            origin,
            latest_elapsed_ns: AtomicU64::new(0),
        }
    }

    fn elapsed_ns(&self, now: Instant) -> u64 {
        u64::try_from(now.saturating_duration_since(self.origin).as_nanos()).unwrap_or(u64::MAX)
    }

    pub(crate) fn latency_ms(&self, now: Instant) -> u64 {
        self.elapsed_ns(now)
            .saturating_sub(self.latest_elapsed_ns.load(Ordering::Relaxed))
            / 1_000_000
    }

    pub(crate) fn observe(&self, now: Instant) -> u64 {
        let elapsed = self.elapsed_ns(now);
        // A preempted earlier probe cannot overwrite a newer worker's timestamp.
        let previous = self.latest_elapsed_ns.fetch_max(elapsed, Ordering::Relaxed);
        elapsed.saturating_sub(previous) / 1_000_000
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
}
