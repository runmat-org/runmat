use crate::checkpoint_clock::CheckpointClock;
use runmat_time::Instant;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use runmat_geometry_core::{
    GeometryEvaluationControl, GeometryEvaluationError, GeometryEvaluationErrorKind,
};
use runmat_meshing_core::{CancellationPolicy, MeshingCancellationSignal, MeshingResourceBudget};

mod allocation_phase;
#[cfg(test)]
pub(crate) mod tests;
mod usage;
pub(crate) use allocation_phase::{AllocationPhase, RecordStorage};
pub(crate) use usage::GeometryUsageCounters;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GeometryEvaluationUsage {
    /// High water mark of admitted storage, including temporary reservations.
    pub allocation_bytes: u64,
    pub live_allocation_bytes: u64,
    pub search_work: u64,
    pub iterations: u64,
}

/// Exact-evaluator projection of the execution-owned cancellation and resource envelope.
pub struct MeshingGeometryEvaluationControl<'a> {
    cancellation: &'a dyn MeshingCancellationSignal,
    started: Instant,
    last_checkpoint_at: Arc<CheckpointClock>,
    maximum_observed_checkpoint_latency_ms: Arc<AtomicU64>,
    last_checkpoint_search_work: Arc<AtomicU64>,
    maximum_wall_time_ms: u64,
    maximum_checkpoint_latency_ms: u64,
    maximum_work_units_between_checks: u64,
    maximum_allocation_bytes: u64,
    maximum_search_work: u64,
    maximum_iterations: u64,
    usage: Arc<GeometryUsageCounters>,
    phase: Option<Arc<AllocationPhase>>,
}

impl<'a> MeshingGeometryEvaluationControl<'a> {
    pub(crate) fn new(
        cancellation: &'a dyn MeshingCancellationSignal,
        started: Instant,
        last_checkpoint_at: Arc<CheckpointClock>,
        maximum_observed_checkpoint_latency_ms: Arc<AtomicU64>,
        last_checkpoint_search_work: Arc<AtomicU64>,
        resources: &MeshingResourceBudget,
        policy: &CancellationPolicy,
    ) -> Self {
        Self {
            cancellation,
            started,
            last_checkpoint_at,
            maximum_observed_checkpoint_latency_ms,
            last_checkpoint_search_work,
            maximum_wall_time_ms: resources.maximum_wall_time_ms,
            maximum_checkpoint_latency_ms: policy.maximum_checkpoint_latency_ms,
            maximum_work_units_between_checks: policy.maximum_work_units_between_checks,
            maximum_allocation_bytes: resources.maximum_memory_bytes,
            maximum_search_work: resources.maximum_search_work,
            maximum_iterations: resources.maximum_iterations,
            usage: Arc::new(GeometryUsageCounters::default()),
            phase: None,
        }
    }

    pub(crate) fn with_usage(mut self, usage: Arc<GeometryUsageCounters>) -> Self {
        self.usage = usage;
        self
    }

    pub(crate) fn with_allocation_phase(mut self, phase: Option<Arc<AllocationPhase>>) -> Self {
        self.phase = phase;
        self
    }

    pub fn remaining_allocation_bytes(&self) -> u64 {
        self.maximum_allocation_bytes
            .saturating_sub(self.usage.allocation_bytes.load(Ordering::Relaxed))
    }

    pub fn usage(&self) -> GeometryEvaluationUsage {
        GeometryEvaluationUsage {
            allocation_bytes: self.usage.peak_allocation_bytes.load(Ordering::Relaxed),
            live_allocation_bytes: self.usage.allocation_bytes.load(Ordering::Relaxed),
            search_work: self.usage.search_work.load(Ordering::Relaxed),
            iterations: self.usage.iterations.load(Ordering::Relaxed),
        }
    }

    /// Search work done by every view of this stage since the previous exact-geometry probe.
    fn observe_probe_search_work(&self) -> u64 {
        let current = self.usage.search_work.load(Ordering::Relaxed);
        let previous = self
            .usage
            .search_work_at_last_probe
            .fetch_max(current, Ordering::Relaxed);
        current.saturating_sub(previous)
    }

    fn consume(
        counter: &AtomicU64,
        count: u64,
        maximum: u64,
        name: &str,
        kind: GeometryEvaluationErrorKind,
    ) -> Result<(), GeometryEvaluationError> {
        counter
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(count).filter(|next| *next <= maximum)
            })
            .map(|_| ())
            .map_err(|_| {
                GeometryEvaluationError::new(
                    kind,
                    format!("exact geometry {name} exceeds the execution resource budget"),
                )
            })
    }
}

impl GeometryEvaluationControl for MeshingGeometryEvaluationControl<'_> {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        if self.cancellation.is_cancelled() {
            return Err(GeometryEvaluationError::new(
                GeometryEvaluationErrorKind::Cancelled,
                "execution cancelled exact geometry evaluation",
            ));
        }
        let now = Instant::now();
        let checkpoint_latency_ms = self.last_checkpoint_at.observe(now);
        let prior_maximum = self
            .maximum_observed_checkpoint_latency_ms
            .fetch_max(checkpoint_latency_ms, Ordering::Relaxed);
        crate::checkpoint_diagnostic::report_first_miss(
            prior_maximum,
            checkpoint_latency_ms,
            self.maximum_checkpoint_latency_ms,
        );
        let wall_time_ms = elapsed_millis(self.started, now);
        if wall_time_ms > self.maximum_wall_time_ms {
            return Err(GeometryEvaluationError::new(
                GeometryEvaluationErrorKind::TimeBudgetExceeded,
                format!(
                    "exact geometry evaluation exceeded its execution wall-time budget: {wall_time_ms}/{} ms",
                    self.maximum_wall_time_ms,
                ),
            ));
        }
        // Responsiveness is bounded by the work done between probes, which is a property of the
        // algorithm. Checkpoint latency depends on machine load and stays a diagnostic above.
        let work_since_probe = self.observe_probe_search_work();
        if work_since_probe > self.maximum_work_units_between_checks {
            return Err(GeometryEvaluationError::new(
                GeometryEvaluationErrorKind::SearchWorkBudgetExceeded,
                format!(
                    "exact geometry evaluation did {work_since_probe} search work units between cancellation checkpoints; the request admits {}",
                    self.maximum_work_units_between_checks,
                ),
            ));
        }
        self.last_checkpoint_search_work.store(
            self.usage.search_work.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
        Ok(())
    }

    fn consume_iterations(&self, count: u64) -> Result<(), GeometryEvaluationError> {
        Self::consume(
            &self.usage.iterations,
            count,
            self.maximum_iterations,
            "iterations",
            GeometryEvaluationErrorKind::IterationBudgetExceeded,
        )
    }

    fn consume_search_work(&self, count: u64) -> Result<(), GeometryEvaluationError> {
        Self::consume(
            &self.usage.search_work,
            count,
            self.maximum_search_work,
            "search work",
            GeometryEvaluationErrorKind::SearchWorkBudgetExceeded,
        )
    }

    fn consume_allocation_bytes(&self, count: u64) -> Result<(), GeometryEvaluationError> {
        let previous = self
            .usage
            .allocation_bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current
                    .checked_add(count)
                    .filter(|next| *next <= self.maximum_allocation_bytes)
            })
            .map_err(|live| {
                GeometryEvaluationError::new(
                    GeometryEvaluationErrorKind::AllocationBudgetExceeded,
                    format!("exact geometry allocation exceeds the execution resource budget: requested {count} bytes, live {live} bytes, maximum {} bytes", self.maximum_allocation_bytes),
                )
            })?;
        self.usage
            .peak_allocation_bytes
            .fetch_max(previous + count, Ordering::Relaxed);
        if let Some(phase) = &self.phase {
            phase.record(count);
        }
        Ok(())
    }

    fn release_allocation_bytes(&self, count: u64) {
        if let Some(phase) = &self.phase {
            phase.release(count);
        }
        self.usage
            .allocation_bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_sub(count)
            })
            .expect("released an unowned geometry allocation charge");
    }
}

impl MeshingCancellationSignal for MeshingGeometryEvaluationControl<'_> {
    fn is_cancelled(&self) -> bool {
        if self.cancellation.is_cancelled() {
            return true;
        }
        let now = Instant::now();
        let checkpoint_latency_ms = self.last_checkpoint_at.observe(now);
        let prior_maximum = self
            .maximum_observed_checkpoint_latency_ms
            .fetch_max(checkpoint_latency_ms, Ordering::Relaxed);
        self.observe_probe_search_work();
        self.last_checkpoint_search_work.store(
            self.usage.search_work.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
        crate::checkpoint_diagnostic::report_first_miss(
            prior_maximum,
            checkpoint_latency_ms,
            self.maximum_checkpoint_latency_ms,
        );
        false
    }
}

fn elapsed_millis(start: Instant, end: Instant) -> u64 {
    u64::try_from(end.duration_since(start).as_millis()).unwrap_or(u64::MAX)
}
