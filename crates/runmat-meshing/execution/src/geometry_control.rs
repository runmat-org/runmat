use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use runmat_geometry_core::{
    GeometryEvaluationControl, GeometryEvaluationError, GeometryEvaluationErrorKind,
};
use runmat_meshing_core::{CancellationPolicy, MeshingCancellationSignal, MeshingResourceBudget};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GeometryEvaluationUsage {
    pub allocation_bytes: u64,
    pub search_work: u64,
    pub iterations: u64,
}

/// Exact-evaluator projection of the execution-owned cancellation and resource envelope.
pub struct MeshingGeometryEvaluationControl<'a> {
    cancellation: &'a dyn MeshingCancellationSignal,
    started: Instant,
    last_checkpoint_at: Mutex<Instant>,
    maximum_wall_time_ms: u64,
    maximum_checkpoint_latency_ms: u64,
    maximum_allocation_bytes: u64,
    maximum_search_work: u64,
    maximum_iterations: u64,
    allocation_bytes: AtomicU64,
    search_work: AtomicU64,
    iterations: AtomicU64,
}

impl<'a> MeshingGeometryEvaluationControl<'a> {
    pub(crate) fn new(
        cancellation: &'a dyn MeshingCancellationSignal,
        started: Instant,
        resources: &MeshingResourceBudget,
        policy: &CancellationPolicy,
    ) -> Self {
        Self {
            cancellation,
            started,
            last_checkpoint_at: Mutex::new(Instant::now()),
            maximum_wall_time_ms: resources.maximum_wall_time_ms,
            maximum_checkpoint_latency_ms: policy.maximum_checkpoint_latency_ms,
            maximum_allocation_bytes: resources.maximum_memory_bytes,
            maximum_search_work: resources.maximum_search_work,
            maximum_iterations: resources.maximum_iterations,
            allocation_bytes: AtomicU64::new(0),
            search_work: AtomicU64::new(0),
            iterations: AtomicU64::new(0),
        }
    }

    pub fn usage(&self) -> GeometryEvaluationUsage {
        GeometryEvaluationUsage {
            allocation_bytes: self.allocation_bytes.load(Ordering::Relaxed),
            search_work: self.search_work.load(Ordering::Relaxed),
            iterations: self.iterations.load(Ordering::Relaxed),
        }
    }

    fn consume(
        counter: &AtomicU64,
        count: u64,
        maximum: u64,
        name: &str,
    ) -> Result<(), GeometryEvaluationError> {
        counter
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(count).filter(|next| *next <= maximum)
            })
            .map(|_| ())
            .map_err(|_| {
                GeometryEvaluationError::new(
                    GeometryEvaluationErrorKind::BudgetExceeded,
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
        let mut previous = self.last_checkpoint_at.lock().map_err(|_| {
            GeometryEvaluationError::new(
                GeometryEvaluationErrorKind::InvalidResult,
                "geometry checkpoint state is unavailable",
            )
        })?;
        if elapsed_millis(self.started, now) > self.maximum_wall_time_ms
            || elapsed_millis(*previous, now) > self.maximum_checkpoint_latency_ms
        {
            return Err(GeometryEvaluationError::new(
                GeometryEvaluationErrorKind::BudgetExceeded,
                "exact geometry evaluation exceeded its execution time envelope",
            ));
        }
        *previous = now;
        Ok(())
    }

    fn consume_iterations(&self, count: u64) -> Result<(), GeometryEvaluationError> {
        Self::consume(
            &self.iterations,
            count,
            self.maximum_iterations,
            "iterations",
        )
    }

    fn consume_search_work(&self, count: u64) -> Result<(), GeometryEvaluationError> {
        Self::consume(
            &self.search_work,
            count,
            self.maximum_search_work,
            "search work",
        )
    }

    fn consume_allocation_bytes(&self, count: u64) -> Result<(), GeometryEvaluationError> {
        Self::consume(
            &self.allocation_bytes,
            count,
            self.maximum_allocation_bytes,
            "allocation",
        )
    }
}

fn elapsed_millis(start: Instant, end: Instant) -> u64 {
    u64::try_from(end.duration_since(start).as_millis()).unwrap_or(u64::MAX)
}
