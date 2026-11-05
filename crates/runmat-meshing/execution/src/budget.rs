use crate::checkpoint_clock::CheckpointClock;
mod allocation_phase;
#[cfg(test)]
mod tests;
use runmat_time::Instant;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use runmat_meshing_core::{
    InvariantEvidence, MeshingCancellationSignal, MeshingDiagnosticEntry, MeshingDiagnosticValue,
    MeshingFailure, MeshingFailureCategory, MeshingPartitionDescriptor, MeshingProgress,
    MeshingRequest, MeshingStageEvidence, MeshingStageKind, StableDigest,
    MESHING_FAILURE_SCHEMA_VERSION, MESHING_PROGRESS_SCHEMA_VERSION,
};

use crate::geometry_control::{GeometryUsageCounters, MeshingGeometryEvaluationControl};

pub trait MeshingProgressSink {
    fn record(&mut self, progress: &MeshingProgress);
}

#[derive(Debug, Default)]
pub struct NoopMeshingProgress;

impl MeshingProgressSink for NoopMeshingProgress {
    fn record(&mut self, _progress: &MeshingProgress) {}
}

/// Absolute algorithm usage observed at one cancellation/budget checkpoint.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MeshingStageCheckpoint {
    pub completed_work: u64,
    pub estimated_work: u64,
    pub node_count: u64,
    pub element_count: u64,
    pub peak_memory_bytes: u64,
    pub peak_scratch_bytes: u64,
    pub search_work: u64,
    pub recursion_depth: u32,
    pub iterations: u64,
    pub entity_counts: BTreeMap<String, u64>,
}

pub struct MeshingStageControl<'a> {
    stage: MeshingStageKind,
    partition_index: u32,
    request: &'a MeshingRequest,
    budget: runmat_meshing_core::MeshingResourceBudget,
    cancellation: &'a dyn MeshingCancellationSignal,
    progress: &'a mut dyn MeshingProgressSink,
    started: Instant,
    last_checkpoint_at: Arc<CheckpointClock>,
    maximum_observed_checkpoint_latency_ms: Arc<AtomicU64>,
    last_checkpoint_search_work: Arc<AtomicU64>,
    last: MeshingStageCheckpoint,
    last_progress: Option<MeshingProgress>,
    sequence: u64,
    geometry_usage: Arc<GeometryUsageCounters>,
    allocation_phase: Option<Arc<crate::geometry_control::AllocationPhase>>,
}

impl<'a> MeshingStageControl<'a> {
    pub fn new(
        stage: MeshingStageKind,
        partition_index: u32,
        request: &'a MeshingRequest,
        ceilings: runmat_meshing_core::MeshingExecutionCeilings,
        cancellation: &'a dyn MeshingCancellationSignal,
        progress: &'a mut dyn MeshingProgressSink,
    ) -> Result<Self, Box<MeshingFailure>> {
        request.validate().map_err(|_| {
            failure(
                stage,
                MeshingFailureCategory::InternalInvariantViolation,
                "resolved meshing request must validate before stage execution",
                None,
            )
        })?;
        let now = Instant::now();
        Ok(Self {
            stage,
            partition_index,
            request,
            budget: runmat_meshing_core::MeshingResourceBudget::new(request.limits, ceilings),
            cancellation,
            progress,
            started: now,
            last_checkpoint_at: Arc::new(CheckpointClock::new(now)),
            maximum_observed_checkpoint_latency_ms: Arc::new(AtomicU64::new(0)),
            last_checkpoint_search_work: Arc::new(AtomicU64::new(0)),
            last: MeshingStageCheckpoint::default(),
            last_progress: None,
            sequence: 0,
            geometry_usage: Arc::new(GeometryUsageCounters::default()),
            allocation_phase: None,
        })
    }

    pub const fn request(&self) -> &MeshingRequest {
        self.request
    }

    pub fn geometry_evaluation_control(&self) -> MeshingGeometryEvaluationControl<'a> {
        MeshingGeometryEvaluationControl::new(
            self.cancellation,
            self.started,
            Arc::clone(&self.last_checkpoint_at),
            Arc::clone(&self.maximum_observed_checkpoint_latency_ms),
            Arc::clone(&self.last_checkpoint_search_work),
            &self.budget,
            &self.request.cancellation,
        )
        .with_usage(Arc::clone(&self.geometry_usage))
        .with_allocation_phase(self.allocation_phase.clone())
    }

    #[track_caller]
    pub fn guard(&self) -> Result<(), Box<MeshingFailure>> {
        let now = Instant::now();
        if self.cancellation.is_cancelled() {
            return Err(failure(
                self.stage,
                MeshingFailureCategory::Cancelled,
                "retry after the execution cancellation authority permits new work",
                None,
            ));
        }
        if elapsed_millis(self.started, now) > self.budget.maximum_wall_time_ms {
            return Err(failure(
                self.stage,
                MeshingFailureCategory::TimeBudgetExceeded,
                "increase the wall-time budget or relax the meshing request",
                Some((
                    "wall_time_ms",
                    elapsed_millis(self.started, now),
                    self.budget.maximum_wall_time_ms,
                )),
            ));
        }
        // Checkpoint latency is load-dependent telemetry; the enforced responsiveness contract is
        // the work done between checkpoints (`checkpoint`).
        crate::checkpoint_diagnostic::report_first_miss(
            self.maximum_observed_checkpoint_latency_ms
                .load(Ordering::Relaxed),
            self.last_checkpoint_at.latency_ms(now),
            self.request.cancellation.maximum_checkpoint_latency_ms,
        );
        Ok(())
    }

    pub(crate) fn stage_evidence(
        &mut self,
        partition: MeshingPartitionDescriptor,
        stage_result_digest: StableDigest,
    ) -> MeshingStageEvidence {
        // Geometry, codec and publication views share cumulative usage. They can
        // do work after the last algorithm checkpoint; retain those costs too.
        // Both inventories are absolute observations, so merge by max, not sum.
        let geometry = self.geometry_evaluation_control().usage();
        MeshingStageEvidence {
            reused_from: None,
            stage: self.stage,
            partition,
            stage_result_digest,
            entity_counts: std::mem::take(&mut self.last.entity_counts),
            invariants: vec![InvariantEvidence {
                invariant_id: "canonical_stage_validation".into(),
                passed: true,
                measured: None,
                required: None,
                unit: None,
            }],
            achieved_error_distributions: BTreeMap::new(),
            completed_work: self.last.completed_work,
            estimated_work: self.last.estimated_work,
            peak_memory_bytes: self.last.peak_memory_bytes.max(geometry.allocation_bytes),
            peak_scratch_bytes: self.last.peak_scratch_bytes,
            search_work: self.last.search_work.max(geometry.search_work),
            maximum_recursion_depth: self.last.recursion_depth,
            iterations: self.last.iterations.max(geometry.iterations),
            elapsed_time_ms: elapsed_millis(self.started, Instant::now()),
            cancellation_checkpoints: self.sequence,
        }
    }

    #[track_caller]
    pub fn checkpoint(
        &mut self,
        checkpoint: MeshingStageCheckpoint,
    ) -> Result<(), Box<MeshingFailure>> {
        let now = Instant::now();
        let elapsed_ms = elapsed_millis(self.started, now);
        let current_checkpoint_latency_ms = self.last_checkpoint_at.latency_ms(now);
        let prior_maximum = self
            .maximum_observed_checkpoint_latency_ms
            .fetch_max(current_checkpoint_latency_ms, Ordering::Relaxed);
        if self.cancellation.is_cancelled() {
            return Err(failure(
                self.stage,
                MeshingFailureCategory::Cancelled,
                "retry after the execution cancellation authority permits new work",
                None,
            ));
        }
        self.validate_monotone(&checkpoint)?;
        let work_since_check = checkpoint.search_work.saturating_sub(
            self.last_checkpoint_search_work
                .load(Ordering::Relaxed)
                .max(self.last.search_work),
        );
        crate::checkpoint_diagnostic::report_first_miss(
            prior_maximum,
            current_checkpoint_latency_ms,
            self.request.cancellation.maximum_checkpoint_latency_ms,
        );
        if work_since_check > self.request.cancellation.maximum_work_units_between_checks {
            return Err(failure(
                self.stage,
                MeshingFailureCategory::InternalInvariantViolation,
                "reduce algorithm work between mandatory cancellation checkpoints",
                Some((
                    "work_units_between_checks",
                    work_since_check,
                    self.request.cancellation.maximum_work_units_between_checks,
                )),
            ));
        }
        self.enforce_budget(&checkpoint, elapsed_ms)?;

        self.sequence = self.sequence.checked_add(1).ok_or_else(|| {
            failure(
                self.stage,
                MeshingFailureCategory::InternalInvariantViolation,
                "restart the stage because its progress sequence overflowed",
                None,
            )
        })?;
        let progress = MeshingProgress {
            schema_version: MESHING_PROGRESS_SCHEMA_VERSION,
            stage: self.stage,
            partition_index: self.partition_index,
            sequence: self.sequence,
            completed_work: checkpoint.completed_work,
            estimated_work: checkpoint.estimated_work,
            entity_counts: checkpoint.entity_counts.clone(),
            peak_memory_bytes: checkpoint.peak_memory_bytes,
            elapsed_time_ms: elapsed_ms,
            consumed_search_work: checkpoint.search_work,
            cancellation_checkpoint: self.sequence,
        };
        progress.validate().map_err(|_| {
            failure(
                self.stage,
                MeshingFailureCategory::InternalInvariantViolation,
                "emit bounded monotone meshing progress",
                None,
            )
        })?;
        if let Some(previous) = &self.last_progress {
            progress.validate_after(previous).map_err(|_| {
                failure(
                    self.stage,
                    MeshingFailureCategory::InternalInvariantViolation,
                    "emit bounded monotone meshing progress",
                    None,
                )
            })?;
        }
        self.progress.record(&progress);
        self.last = checkpoint;
        self.last_checkpoint_search_work
            .store(self.last.search_work, Ordering::Relaxed);
        self.last_progress = Some(progress);
        self.last_checkpoint_at.observe(now);
        Ok(())
    }

    fn validate_monotone(
        &self,
        checkpoint: &MeshingStageCheckpoint,
    ) -> Result<(), Box<MeshingFailure>> {
        let regressed = checkpoint.completed_work < self.last.completed_work
            || checkpoint.estimated_work < self.last.estimated_work
            || checkpoint.node_count < self.last.node_count
            || checkpoint.element_count < self.last.element_count
            || checkpoint.peak_memory_bytes < self.last.peak_memory_bytes
            || checkpoint.peak_scratch_bytes < self.last.peak_scratch_bytes
            || checkpoint.search_work < self.last.search_work
            || checkpoint.iterations < self.last.iterations
            || self.last.entity_counts.iter().any(|(name, count)| {
                checkpoint
                    .entity_counts
                    .get(name)
                    .is_none_or(|next| next < count)
            });
        if regressed || checkpoint.completed_work > checkpoint.estimated_work {
            return Err(failure(
                self.stage,
                MeshingFailureCategory::InternalInvariantViolation,
                "report absolute monotone usage at every stage checkpoint",
                None,
            ));
        }
        Ok(())
    }

    fn enforce_budget(
        &self,
        checkpoint: &MeshingStageCheckpoint,
        elapsed_ms: u64,
    ) -> Result<(), Box<MeshingFailure>> {
        let budget = &self.budget;
        for (category, name, achieved, maximum, remediation) in [
            (
                MeshingFailureCategory::NodeBudgetExceeded,
                "nodes",
                checkpoint.node_count,
                budget.maximum_nodes,
                "increase the node budget or relax the sizing request",
            ),
            (
                MeshingFailureCategory::ElementBudgetExceeded,
                "elements",
                checkpoint.element_count,
                budget.maximum_elements,
                "increase the element budget or relax the sizing request",
            ),
            (
                MeshingFailureCategory::MemoryBudgetExceeded,
                "memory_bytes",
                checkpoint.peak_memory_bytes,
                budget.maximum_memory_bytes,
                "increase the memory budget or reduce model complexity",
            ),
            (
                MeshingFailureCategory::ScratchBudgetExceeded,
                "scratch_bytes",
                checkpoint.peak_scratch_bytes,
                budget.maximum_scratch_bytes,
                "increase the scratch budget or reduce model complexity",
            ),
            (
                MeshingFailureCategory::TimeBudgetExceeded,
                "wall_time_ms",
                elapsed_ms,
                budget.maximum_wall_time_ms,
                "increase the wall-time budget or relax the meshing request",
            ),
            (
                MeshingFailureCategory::SearchWorkBudgetExceeded,
                "search_work",
                checkpoint.search_work,
                budget.maximum_search_work,
                "increase the search-work budget or simplify protected constraints",
            ),
            (
                MeshingFailureCategory::IterationBudgetExceeded,
                "iterations",
                checkpoint.iterations,
                budget.maximum_iterations,
                "increase the iteration budget or relax quality targets",
            ),
        ] {
            if achieved > maximum {
                return Err(failure(
                    self.stage,
                    category,
                    remediation,
                    Some((name, achieved, maximum)),
                ));
            }
        }
        if checkpoint.recursion_depth > budget.maximum_recursion_depth {
            return Err(failure(
                self.stage,
                MeshingFailureCategory::RecursionBudgetExceeded,
                "increase the recursion budget or simplify the geometric subdivision",
                Some((
                    "recursion_depth",
                    u64::from(checkpoint.recursion_depth),
                    u64::from(budget.maximum_recursion_depth),
                )),
            ));
        }
        Ok(())
    }
}

fn elapsed_millis(start: Instant, end: Instant) -> u64 {
    u64::try_from(end.duration_since(start).as_millis()).unwrap_or(u64::MAX)
}

pub(super) fn failure(
    stage: MeshingStageKind,
    category: MeshingFailureCategory,
    remediation: &str,
    budget: Option<(&str, u64, u64)>,
) -> Box<MeshingFailure> {
    let (request_values, achieved_values) = budget.map_or_else(
        || (Vec::new(), Vec::new()),
        |(name, achieved, maximum)| {
            (
                vec![diagnostic(name, maximum)],
                vec![diagnostic(name, achieved)],
            )
        },
    );
    Box::new(MeshingFailure {
        schema_version: MESHING_FAILURE_SCHEMA_VERSION,
        category,
        stage,
        operation: stage.operation(),
        entity_ids: Vec::new(),
        witnesses: Vec::new(),
        request_values,
        achieved_values,
        accounting_error: None,
        remediation: remediation.into(),
    })
}

fn diagnostic(name: &str, value: u64) -> MeshingDiagnosticEntry {
    MeshingDiagnosticEntry {
        name: name.into(),
        value: MeshingDiagnosticValue::Count(value),
        unit: None,
    }
}
