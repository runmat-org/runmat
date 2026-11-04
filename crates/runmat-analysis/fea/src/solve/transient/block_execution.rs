use runmat_analysis_core::SignalSampleApproach;

use crate::{
    assembly::AssemblySummary,
    contracts::{SolverMethod, SolverPreconditioner},
    execution::{
        StructuralTransientBlockRequest, StructuralTransientCheckpoint,
        StructuralTransientContinuationApproach, StructuralTransientObservationCursor,
        StructuralTransientSegment,
    },
    solve::backend::kind::LinearAlgebraBackendKind,
    ComputeBackend,
};

use super::state::SecondOrderState;
use super::{
    advance::{
        self, DirectTransientStepStrategy, TransientAcceptedSample, TransientInitialSample,
        TransientResumeInput, TransientSolverAccounting, TransientStepStrategy,
    },
    balance::{TransientBalanceHistory, TransientBalanceStep},
    empty::{self, EmptyTransientAdvanceState},
    require_stable_system,
    segment::{self, CanonicalTransientBlockParts},
    validate_options, TransientSolveError, TransientSolveOptions,
};

#[derive(Clone, Debug, PartialEq)]
pub struct StructuralTransientInitialConditions {
    pub displacement: Vec<f64>,
    pub velocity: Vec<f64>,
}

pub struct StructuralTransientBlockSolveInput<'a> {
    pub request: &'a StructuralTransientBlockRequest,
    pub predecessor: Option<&'a StructuralTransientCheckpoint>,
    pub initial_summary: &'a AssemblySummary,
    pub options: &'a TransientSolveOptions,
    pub backend: ComputeBackend,
    pub initial_conditions: Option<StructuralTransientInitialConditions>,
    pub observation_cursors: Vec<StructuralTransientObservationCursor>,
}

pub fn solve_structural_transient_block_with_assembly<E: std::fmt::Display>(
    input: StructuralTransientBlockSolveInput<'_>,
    mut assembly_at: impl FnMut(f64, SignalSampleApproach) -> Result<AssemblySummary, E>,
) -> Result<StructuralTransientSegment, TransientSolveError<E>> {
    let physical_dof_count = input.initial_summary.dof_count;
    input
        .request
        .validate_against_predecessor(input.predecessor)
        .map_err(contract_error)?;
    let mut identity_physical_state =
        |_: f64, _: SignalSampleApproach, summary: &AssemblySummary, state: &SecondOrderState| {
            Ok::<_, E>((summary.clone(), state.clone()))
        };
    let mut identity_numerical_state = |checkpoint: &StructuralTransientCheckpoint| {
        SecondOrderState::try_new(
            checkpoint.state.displacement.clone(),
            checkpoint.state.velocity.clone(),
            checkpoint.state.acceleration.clone(),
        )
        .map_err(|error| TransientSolveError::InvalidContinuationState {
            detail: error.to_string(),
        })
    };
    solve_structural_transient_block_with_physical_mapping(
        input,
        physical_dof_count,
        &mut assembly_at,
        &mut identity_physical_state,
        &mut identity_numerical_state,
    )
}

pub(crate) fn solve_structural_transient_block_with_physical_mapping<E: std::fmt::Display>(
    input: StructuralTransientBlockSolveInput<'_>,
    physical_dof_count: usize,
    assembly_at: &mut impl FnMut(f64, SignalSampleApproach) -> Result<AssemblySummary, E>,
    map_physical_state: &mut impl FnMut(
        f64,
        SignalSampleApproach,
        &AssemblySummary,
        &SecondOrderState,
    ) -> Result<(AssemblySummary, SecondOrderState), E>,
    reduce_checkpoint_state: &mut impl FnMut(
        &StructuralTransientCheckpoint,
    ) -> Result<SecondOrderState, TransientSolveError<E>>,
) -> Result<StructuralTransientSegment, TransientSolveError<E>> {
    let backend_kind = match input.backend {
        ComputeBackend::Cpu => LinearAlgebraBackendKind::CpuReference,
        ComputeBackend::Gpu => LinearAlgebraBackendKind::Accelerator,
    };
    let mut strategy = DirectTransientStepStrategy::new(
        input.initial_summary,
        backend_kind,
        input.options.residual_target,
    );
    let mut typed_physical_state =
        |time_s, approach, summary: &AssemblySummary, state: &SecondOrderState| {
            map_physical_state(time_s, approach, summary, state).map_err(|error| {
                TransientSolveError::PhysicalMapping {
                    detail: error.to_string(),
                }
            })
        };
    solve_structural_transient_block_with_strategy(
        input,
        physical_dof_count,
        assembly_at,
        &mut typed_physical_state,
        reduce_checkpoint_state,
        &mut strategy,
    )
}

pub(super) fn solve_structural_transient_block_with_strategy<E>(
    input: StructuralTransientBlockSolveInput<'_>,
    physical_dof_count: usize,
    assembly_at: &mut impl FnMut(f64, SignalSampleApproach) -> Result<AssemblySummary, E>,
    map_physical_state: &mut impl FnMut(
        f64,
        SignalSampleApproach,
        &AssemblySummary,
        &SecondOrderState,
    ) -> Result<
        (AssemblySummary, SecondOrderState),
        TransientSolveError<E>,
    >,
    restore_checkpoint_state: &mut impl FnMut(
        &StructuralTransientCheckpoint,
    ) -> Result<SecondOrderState, TransientSolveError<E>>,
    strategy: &mut impl TransientStepStrategy,
) -> Result<StructuralTransientSegment, TransientSolveError<E>> {
    input
        .request
        .validate_against_predecessor(input.predecessor)
        .map_err(contract_error)?;
    let schedule = &input.request.scope.schedule;
    validate_options(
        input.options,
        schedule.points[0].time_s.canonical_value(),
        schedule
            .points
            .last()
            .expect("a validated transient schedule has a terminal point")
            .time_s
            .canonical_value(),
    )?;
    if input.request.scope.dof_count != physical_dof_count || input.initial_summary.dof_count == 0 {
        return Err(TransientSolveError::DofCountChanged);
    }
    let backend_kind = match input.backend {
        ComputeBackend::Cpu => LinearAlgebraBackendKind::CpuReference,
        ComputeBackend::Gpu => LinearAlgebraBackendKind::Accelerator,
    };
    let (state, initial_sample) = match (input.predecessor, input.initial_conditions) {
        (None, Some(initial)) => {
            let (state, sample) = advance::initialize(
                input.initial_summary,
                input.options,
                schedule,
                initial.displacement,
                initial.velocity,
                strategy,
                map_physical_state,
            )?;
            (state, Some(sample))
        }
        (Some(predecessor), None) => {
            let approach = match predecessor.continuation.approach {
                StructuralTransientContinuationApproach::At => SignalSampleApproach::At,
                StructuralTransientContinuationApproach::RightLimit => {
                    SignalSampleApproach::RightLimit
                }
            };
            let continuation_summary = assembly_at(predecessor.continuation.time_s.canonical_value(), approach)
                .map_err(TransientSolveError::Assembly)?;
            require_stable_system(input.initial_summary, &continuation_summary)?;
            let numerical_state = restore_checkpoint_state(predecessor)?;
            let (physical_summary, mapped_physical_state) = map_physical_state(
                predecessor.continuation.time_s.canonical_value(),
                approach,
                &continuation_summary,
                &numerical_state,
            )
            ?;
            if mapped_physical_state.displacement() != predecessor.state.displacement.as_slice()
                || mapped_physical_state.velocity() != predecessor.state.velocity.as_slice()
                || mapped_physical_state.acceleration()
                    != predecessor.state.acceleration.as_slice()
            {
                return Err(TransientSolveError::ExecutionContract {
                    detail: "transient checkpoint physical state differs from the formulation-owned numerical reconstruction"
                        .into(),
                });
            }
            let state = advance::resume::<E>(
                schedule,
                TransientResumeInput {
                    schedule_ordinal: predecessor.continuation.schedule_ordinal,
                    numerical_state,
                    physical_state: predecessor.state.clone(),
                    continuation_approach: predecessor.continuation.approach,
                    integration_summary: continuation_summary.clone(),
                    physical_summary,
                    balance: predecessor.balance.clone(),
                    accounting: strategy.resumed_accounting(predecessor.counters.clone()),
                },
                strategy.coordinate_count(),
            )?;
            (state, None)
        }
        _ => {
            return Err(TransientSolveError::ExecutionContract {
                detail: "a first transient block requires initial conditions and a resumed block requires only its predecessor checkpoint"
                    .into(),
            })
        }
    };
    let block = advance::advance(
        advance::TransientAdvanceContext {
            initial_summary: input.initial_summary,
            options: input.options,
            backend_kind,
            schedule,
            interval_retention: super::TransientIntervalRetention::DiscardAfterBalance,
        },
        state,
        input.request.terminal_schedule_ordinal,
        assembly_at,
        map_physical_state,
        strategy,
    )?;
    segment::into_canonical_segment(
        input.request.clone(),
        input.predecessor,
        block,
        initial_sample,
        input.observation_cursors,
        strategy.solver_method(),
    )
}

pub(crate) fn solve_fully_prescribed_transient_block_with_physical_mapping<E>(
    input: StructuralTransientBlockSolveInput<'_>,
    physical_dof_count: usize,
    assembly_at: &mut impl FnMut(f64, SignalSampleApproach) -> Result<AssemblySummary, E>,
    map_physical_state: &mut impl FnMut(
        f64,
        SignalSampleApproach,
        &AssemblySummary,
    ) -> Result<(AssemblySummary, SecondOrderState), E>,
) -> Result<StructuralTransientSegment, TransientSolveError<E>> {
    input
        .request
        .validate_against_predecessor(input.predecessor)
        .map_err(contract_error)?;
    let schedule = &input.request.scope.schedule;
    validate_options(
        input.options,
        schedule.points[0].time_s.canonical_value(),
        schedule
            .points
            .last()
            .expect("a validated transient schedule has a terminal point")
            .time_s
            .canonical_value(),
    )?;
    if input.request.scope.dof_count != physical_dof_count || input.initial_summary.dof_count != 0 {
        return Err(TransientSolveError::DofCountChanged);
    }
    let (
        mut continuation_summary,
        mut continuation_state,
        continuation_approach,
        accumulator,
        initial_sample,
    ) = match (input.predecessor, input.initial_conditions) {
            (None, Some(initial_conditions)) => {
                let (summary, state) = map_physical_state(
                    schedule.points[0].time_s.canonical_value(),
                    SignalSampleApproach::At,
                    input.initial_summary,
                )
                .map_err(TransientSolveError::Assembly)?;
                if state.displacement() != initial_conditions.displacement.as_slice()
                    || state.velocity() != initial_conditions.velocity.as_slice()
                {
                    return Err(TransientSolveError::InvalidContinuationState {
                        detail: "fully prescribed initial conditions differ from the authored support motion"
                            .into(),
                    });
                }
                let history = TransientBalanceHistory::start(&summary, &state)
                    .map_err(super::balance::DynamicBalanceError::into_transient_solve)?;
                let initial_sample = TransientInitialSample {
                    time_s: runmat_types::physical::Time::new(schedule.points[0].time_s.canonical_value()),
                    state: state.clone(),
                    summary: summary.clone(),
                    projected_residual_norm: 0.0,
                    reaction: history.reaction_snapshots()[0].clone(),
                    strain_energy_j: runmat_types::physical::Energy::new(history.strain_energy_j()[0]),
                    kinetic_energy_j: runmat_types::physical::Energy::new(history.kinetic_energy_j()[0]),
                    energy_balance_residual_j: runmat_types::physical::Energy::new(history.energy_balance_residual_j()[0]),
                    relative_energy_balance_error: history.relative_energy_balance_error()[0],
                    dynamic_balance: history.balances()[0].clone(),
                };
                (
                    summary,
                    state,
                    SignalSampleApproach::At,
                    history.accumulator(),
                    Some(initial_sample),
                )
            }
            (Some(predecessor), None) => {
                let approach = continuation_approach(predecessor.continuation.approach);
                let reduced_summary = assembly_at(predecessor.continuation.time_s.canonical_value(), approach)
                    .map_err(TransientSolveError::Assembly)?;
                let (summary, state) = map_physical_state(
                    predecessor.continuation.time_s.canonical_value(),
                    approach,
                    &reduced_summary,
                )
                .map_err(TransientSolveError::Assembly)?;
                if state.displacement() != predecessor.state.displacement.as_slice()
                    || state.velocity() != predecessor.state.velocity.as_slice()
                    || state.acceleration() != predecessor.state.acceleration.as_slice()
                {
                    return Err(TransientSolveError::InvalidContinuationState {
                        detail: "fully prescribed checkpoint differs from the authored support motion"
                            .into(),
                    });
                }
                TransientBalanceHistory::resume(&summary, &state, predecessor.balance.clone())
                    .map_err(super::balance::DynamicBalanceError::into_transient_solve)?;
                (summary, state, approach, predecessor.balance.clone(), None)
            }
            _ => {
                return Err(TransientSolveError::ExecutionContract {
                    detail: "a first transient block requires initial conditions and a resumed block requires only its predecessor checkpoint"
                        .into(),
                })
            }
        };
    let empty_block = empty::advance(
        input.initial_summary,
        schedule,
        EmptyTransientAdvanceState {
            schedule_ordinal: input.request.start_schedule_ordinal,
            continuation_summary: input.initial_summary.clone(),
            continuation_approach,
        },
        input.request.terminal_schedule_ordinal,
        assembly_at,
    )?;
    let mut history =
        TransientBalanceHistory::resume(&continuation_summary, &continuation_state, accumulator)
            .map_err(super::balance::DynamicBalanceError::into_transient_solve)?;
    let mut samples = Vec::with_capacity(empty_block.samples.len());
    for sample in empty_block.samples {
        let interval = sample.integration_interval;
        let (integration_end_summary, integration_end_state) = map_physical_state(
            interval.end_time_s.canonical_value(),
            interval.end_approach,
            &interval.end_summary,
        )
        .map_err(TransientSolveError::Assembly)?;
        let (public_summary, public_state) =
            if interval.end_approach == SignalSampleApproach::LeftLimit {
                let point_summary = assembly_at(
                    interval.end_time_s.canonical_value(),
                    SignalSampleApproach::At,
                )
                .map_err(TransientSolveError::Assembly)?;
                map_physical_state(
                    interval.end_time_s.canonical_value(),
                    SignalSampleApproach::At,
                    &point_summary,
                )
                .map_err(TransientSolveError::Assembly)?
            } else {
                (
                    integration_end_summary.clone(),
                    integration_end_state.clone(),
                )
            };
        history
            .push_step(TransientBalanceStep {
                integration_start_summary: &continuation_summary,
                integration_start_state: &continuation_state,
                integration_end_summary: &integration_end_summary,
                integration_end_state: &integration_end_state,
                published_summary: &public_summary,
                published_state: &public_state,
                time_step_s: sample.accepted_time_step_s.canonical_value(),
            })
            .map_err(super::balance::DynamicBalanceError::into_transient_solve)?;
        if interval.end_approach == SignalSampleApproach::LeftLimit {
            let right_summary = assembly_at(
                interval.end_time_s.canonical_value(),
                SignalSampleApproach::RightLimit,
            )
            .map_err(TransientSolveError::Assembly)?;
            (continuation_summary, continuation_state) = map_physical_state(
                interval.end_time_s.canonical_value(),
                SignalSampleApproach::RightLimit,
                &right_summary,
            )
            .map_err(TransientSolveError::Assembly)?;
        } else {
            continuation_summary = integration_end_summary;
            continuation_state = integration_end_state;
        }
        let history_index = history.balances().len() - 1;
        samples.push(TransientAcceptedSample {
            schedule_ordinal: sample.schedule_ordinal,
            time_s: runmat_types::physical::Time::new(sample.time_s.canonical_value()),
            accepted_time_step_s: runmat_types::physical::Time::new(
                sample.accepted_time_step_s.canonical_value(),
            ),
            state: public_state,
            summary: public_summary,
            projected_residual_norm: 0.0,
            reaction: history.reaction_snapshots()[history_index].clone(),
            strain_energy_j: runmat_types::physical::Energy::new(
                history.strain_energy_j()[history_index],
            ),
            kinetic_energy_j: runmat_types::physical::Energy::new(
                history.kinetic_energy_j()[history_index],
            ),
            applied_load_work_j: runmat_types::physical::Energy::new(
                history.external_work_j()[history_index],
            ),
            support_work_j: runmat_types::physical::Energy::new(
                history.support_work_j()[history_index],
            ),
            damping_dissipation_j: runmat_types::physical::Energy::new(
                history.damping_dissipation_j()[history_index],
            ),
            energy_balance_residual_j: runmat_types::physical::Energy::new(
                history.energy_balance_residual_j()[history_index],
            ),
            relative_energy_balance_error: history.relative_energy_balance_error()[history_index],
            dynamic_balance: history.balances()[history_index].clone(),
            integration_interval: None,
        });
    }
    segment::into_canonical_segment_parts(
        input.request.clone(),
        input.predecessor,
        CanonicalTransientBlockParts {
            terminal_schedule_ordinal: empty_block.terminal.schedule_ordinal,
            continuation_approach: empty_block.terminal.continuation_approach,
            physical_continuation_state: continuation_state.clone(),
            numerical_continuation_state: continuation_state,
            balance: history.accumulator(),
            accounting: TransientSolverAccounting {
                solver_backend: LinearAlgebraBackendKind::CpuReference,
                counters: input
                    .predecessor
                    .map(|value| value.counters.clone())
                    .unwrap_or_default(),
                preconditioner: SolverPreconditioner::None,
                accelerator_evidence: Vec::new(),
            },
            solver_method: SolverMethod::NewmarkAverageAccelerationPcg,
            initial_sample,
            samples,
        },
        input.observation_cursors,
    )
}

fn continuation_approach(
    approach: StructuralTransientContinuationApproach,
) -> SignalSampleApproach {
    match approach {
        StructuralTransientContinuationApproach::At => SignalSampleApproach::At,
        StructuralTransientContinuationApproach::RightLimit => SignalSampleApproach::RightLimit,
    }
}

fn contract_error<E>(error: impl std::fmt::Display) -> TransientSolveError<E> {
    TransientSolveError::ExecutionContract {
        detail: error.to_string(),
    }
}
