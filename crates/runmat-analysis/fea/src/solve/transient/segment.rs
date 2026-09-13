use runmat_analysis_core::SignalSampleApproach;

use crate::{
    contracts::SolverMethod,
    execution::{
        StructuralTransientBlockRequest, StructuralTransientBlockSolverEvidence,
        StructuralTransientCheckpoint, StructuralTransientCheckpointInput,
        StructuralTransientContinuationApproach, StructuralTransientContinuationPoint,
        StructuralTransientContinuationState, StructuralTransientObservationCursor,
        StructuralTransientSegment, StructuralTransientSegmentSample,
        StructuralTransientSegmentSampleInput, StructuralTransientSolverCounters,
    },
};

use super::{
    advance::{TransientAcceptedSample, TransientBlockResult, TransientSolverAccounting},
    state::SecondOrderState,
    TransientSolveError,
};

pub(super) struct CanonicalTransientBlockParts {
    pub(super) terminal_schedule_ordinal: u64,
    pub(super) continuation_approach: SignalSampleApproach,
    pub(super) physical_continuation_state: SecondOrderState,
    pub(super) balance: crate::execution::StructuralTransientBalanceAccumulator,
    pub(super) accounting: TransientSolverAccounting,
    pub(super) samples: Vec<TransientAcceptedSample>,
}

pub(super) fn into_canonical_segment<E>(
    request: StructuralTransientBlockRequest,
    predecessor: Option<&StructuralTransientCheckpoint>,
    block: TransientBlockResult,
    observation_cursors: Vec<StructuralTransientObservationCursor>,
) -> Result<StructuralTransientSegment, TransientSolveError<E>> {
    into_canonical_segment_parts(
        request,
        predecessor,
        CanonicalTransientBlockParts {
            terminal_schedule_ordinal: block.terminal.schedule_ordinal,
            continuation_approach: block.terminal.continuation_approach,
            physical_continuation_state: block.terminal.physical_continuation_state,
            balance: block.terminal.balance,
            accounting: block.terminal.accounting,
            samples: block.samples,
        },
        observation_cursors,
    )
}

pub(super) fn into_canonical_segment_parts<E>(
    request: StructuralTransientBlockRequest,
    predecessor: Option<&StructuralTransientCheckpoint>,
    block: CanonicalTransientBlockParts,
    observation_cursors: Vec<StructuralTransientObservationCursor>,
) -> Result<StructuralTransientSegment, TransientSolveError<E>> {
    request
        .validate_against_predecessor(predecessor)
        .map_err(contract_error)?;
    if block.terminal_schedule_ordinal != request.terminal_schedule_ordinal {
        return Err(TransientSolveError::ExecutionContract {
            detail: "transient solver block stopped at a different schedule ordinal".into(),
        });
    }
    let continuation_approach = match block.continuation_approach {
        SignalSampleApproach::At => StructuralTransientContinuationApproach::At,
        SignalSampleApproach::RightLimit => StructuralTransientContinuationApproach::RightLimit,
        SignalSampleApproach::LeftLimit => return Err(TransientSolveError::InvalidContinuation),
    };
    let terminal_counters = block.accounting.counters.clone();
    let block_counters = subtract_counters(&terminal_counters, &request.start_counters)?;
    let checkpoint = StructuralTransientCheckpoint::new(
        StructuralTransientCheckpointInput {
            scope: request.scope.clone(),
            block_ordinal: request.block_ordinal,
            continuation: StructuralTransientContinuationPoint {
                schedule_ordinal: block.terminal_schedule_ordinal,
                time_s: request.scope.schedule.points[usize::try_from(
                    block.terminal_schedule_ordinal,
                )
                .map_err(|_| TransientSolveError::ScheduleOrdinalLimit)?]
                .time_s,
                approach: continuation_approach,
            },
            state: StructuralTransientContinuationState {
                displacement: block.physical_continuation_state.displacement().to_vec(),
                velocity: block.physical_continuation_state.velocity().to_vec(),
                acceleration: block.physical_continuation_state.acceleration().to_vec(),
            },
            balance: block.balance,
            counters: terminal_counters,
            observation_cursors,
        },
        predecessor,
    )
    .map_err(contract_error)?;
    let solver_evidence = StructuralTransientBlockSolverEvidence {
        solver_method: SolverMethod::NewmarkAverageAccelerationPcg,
        solver_backend: block.accounting.solver_backend,
        preconditioner: block.accounting.preconditioner,
        counters: block_counters,
        accelerator_evidence: block.accounting.accelerator_evidence,
    };
    let samples = block
        .samples
        .into_iter()
        .map(|sample| {
            StructuralTransientSegmentSample::new(
                &request,
                StructuralTransientSegmentSampleInput {
                    schedule_ordinal: sample.schedule_ordinal,
                    time_s: sample.time_s,
                    accepted_time_step_s: sample.accepted_time_step_s,
                    state: StructuralTransientContinuationState {
                        displacement: sample.state.displacement().to_vec(),
                        velocity: sample.state.velocity().to_vec(),
                        acceleration: sample.state.acceleration().to_vec(),
                    },
                    projected_residual_norm: sample.projected_residual_norm,
                    reaction: sample.reaction,
                    strain_energy_j: sample.strain_energy_j,
                    kinetic_energy_j: sample.kinetic_energy_j,
                    applied_load_work_j: sample.applied_load_work_j,
                    support_work_j: sample.support_work_j,
                    damping_dissipation_j: sample.damping_dissipation_j,
                    energy_balance_residual_j: sample.energy_balance_residual_j,
                    relative_energy_balance_error: sample.relative_energy_balance_error,
                    dynamic_balance: sample.dynamic_balance,
                },
            )
            .map_err(contract_error)
        })
        .collect::<Result<Vec<_>, _>>()?;
    StructuralTransientSegment::new(request, checkpoint, solver_evidence, samples)
        .map_err(contract_error)
}

fn subtract_counters<E>(
    terminal: &StructuralTransientSolverCounters,
    start: &StructuralTransientSolverCounters,
) -> Result<StructuralTransientSolverCounters, TransientSolveError<E>> {
    Ok(StructuralTransientSolverCounters {
        host_sync_count: terminal
            .host_sync_count
            .checked_sub(start.host_sync_count)
            .ok_or(TransientSolveError::ExecutionContract {
                detail: "transient host synchronization counter regressed".into(),
            })?,
        device_apply_k_count: terminal
            .device_apply_k_count
            .checked_sub(start.device_apply_k_count)
            .ok_or(TransientSolveError::ExecutionContract {
                detail: "transient device apply counter regressed".into(),
            })?,
        device_apply_k_attempt_count: terminal
            .device_apply_k_attempt_count
            .checked_sub(start.device_apply_k_attempt_count)
            .ok_or(TransientSolveError::ExecutionContract {
                detail: "transient device apply-attempt counter regressed".into(),
            })?,
    })
}

fn contract_error<E>(error: impl std::fmt::Display) -> TransientSolveError<E> {
    TransientSolveError::ExecutionContract {
        detail: error.to_string(),
    }
}
