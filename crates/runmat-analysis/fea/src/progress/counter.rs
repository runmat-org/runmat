//! Live `i of n` counts for loops that run on the CPU pool.
//!
//! Pool threads do not carry the FEA progress context, so a [`CountReporter`] captures the
//! handler on the thread that creates it. Workers call [`CountReporter::add`] once per chunk of
//! items: an atomic add, then a `try_lock` on a small gate, so a busy gate never blocks a worker.
//! A count shorter than [`QUIET_START`] stays silent, which keeps repeated small loops (for
//! example re-assembly inside each nonlinear iteration) from flooding the host.
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use runmat_time::Instant;

use super::rate::{MINIMUM_EVENT_INTERVAL, UNIT_METADATA};
use super::{FeaProgressEvent, FeaProgressHandler, FeaProgressPhase, FeaProgressStatus};

/// A count reports nothing until it has run this long.
pub const QUIET_START: Duration = Duration::from_millis(500);

pub struct CountReporter {
    handler: Option<FeaProgressHandler>,
    operation: &'static str,
    phase: FeaProgressPhase,
    unit: &'static str,
    noun: &'static str,
    total: u64,
    started: Instant,
    done: AtomicU64,
    reported: AtomicBool,
    gate: Mutex<Option<Instant>>,
}

impl CountReporter {
    /// Counts `total` items of `noun` (plural, e.g. "elements") for the installed handler.
    pub fn new(
        operation: &'static str,
        phase: FeaProgressPhase,
        unit: &'static str,
        noun: &'static str,
        total: u64,
    ) -> Self {
        let context = super::capture_fea_progress_context();
        Self::with_handler(context.handler, operation, phase, unit, noun, total)
    }

    /// Counts assembled elements (`ModelAssembly`, unit `elements`).
    pub fn assembly(total: u64) -> Self {
        Self::new(
            "fea.assembly",
            FeaProgressPhase::ModelAssembly,
            "elements",
            "elements",
            total,
        )
    }

    pub(crate) fn with_handler(
        handler: Option<FeaProgressHandler>,
        operation: &'static str,
        phase: FeaProgressPhase,
        unit: &'static str,
        noun: &'static str,
        total: u64,
    ) -> Self {
        Self {
            handler,
            operation,
            phase,
            unit,
            noun,
            total,
            started: Instant::now(),
            done: AtomicU64::new(0),
            reported: AtomicBool::new(false),
            gate: Mutex::new(None),
        }
    }

    /// Records `count` more finished items.
    pub fn add(&self, count: u64) {
        if self.handler.is_none() {
            return;
        }
        self.done.fetch_add(count, Ordering::Relaxed);
        let Ok(mut last) = self.gate.try_lock() else {
            return;
        };
        let now = Instant::now();
        if now.saturating_duration_since(self.started) < QUIET_START
            || last.is_some_and(|at| now.saturating_duration_since(at) < MINIMUM_EVENT_INTERVAL)
        {
            return;
        }
        *last = Some(now);
        self.reported.store(true, Ordering::Relaxed);
        // Read under the gate so reported counts never go backwards.
        self.send(self.done.load(Ordering::Relaxed).min(self.total));
    }

    /// Reports the final count when any count was reported.
    pub fn finish(&self) {
        let _gate = self.gate.lock().unwrap_or_else(|error| error.into_inner());
        if self.reported.load(Ordering::Relaxed) {
            self.send(self.total);
        }
    }

    fn send(&self, current: u64) {
        let Some(handler) = &self.handler else {
            return;
        };
        let fraction = (self.total > 0).then(|| (current as f64 / self.total as f64).min(1.0));
        handler(FeaProgressEvent {
            operation: self.operation.into(),
            phase: self.phase,
            status: FeaProgressStatus::Advanced,
            message: format!("{current} of {} {}", self.total, self.noun),
            current: Some(current),
            total: Some(self.total),
            fraction,
            metadata: BTreeMap::from([(UNIT_METADATA.to_string(), self.unit.to_string())]),
        });
    }
}

#[cfg(test)]
#[path = "counter_tests.rs"]
mod tests;
