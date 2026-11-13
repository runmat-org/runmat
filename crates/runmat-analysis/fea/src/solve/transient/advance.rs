use runmat_analysis_core::SignalSampleApproach;

mod step_strategy;
pub(super) use step_strategy::{
    DirectTransientStepStrategy, TransientInitialVectors, TransientNumericalStepOutcome,
    TransientSolverAccounting, TransientStepFailure, TransientStepStrategy,
};

use crate::{
    assembly::AssemblySummary,
    contracts::{SolverMethod, SolverPreconditioner, StructuralDynamicBalanceSnapshot},
    execution::{
        StructuralTransientBalanceAccumulator, StructuralTransientContinuationApproach,
        StructuralTransientContinuationState, StructuralTransientSchedule,
        StructuralTransientScheduleArrival, StructuralTransientSolverCounters,
    },
    progress::{emit_phase, is_cancelled, FeaProgressPhase, FeaProgressStatus},
    solve::backend::kind::LinearAlgebraBackendKind,
};

use super::{
    balance::{TransientBalanceHistory, TransientBalanceStep},
    effective_solve::{
        solve_average_acceleration_step_reusing, solve_initial_acceleration_reusing,
        EffectiveDynamicsSolveError,
    },
    require_stable_system,
    state::SecondOrderState,
    EffectiveDynamicsSolveResult, TransientIntegrationInterval, TransientIntervalRetention,
    TransientSolveError, TransientSolveOptions,
};

const LOCAL_TRANSIENT_BLOCK_INTERVALS: u64 = 256;

#[derive(Clone, Debug)]
pub(super) struct TransientInitialSample {
    pub(super) time_s: runmat_types::physical::Time,
    pub(super) state: SecondOrderState,
    pub(super) summary: AssemblySummary,
    pub(super) projected_residual_norm: f64,
    pub(super) reaction: Vec<f64>,
    pub(super) strain_energy_j: runmat_types::physical::Energy,
    pub(super) kinetic_energy_j: runmat_types::physical::Energy,
    pub(super) energy_balance_residual_j: runmat_types::physical::Energy,
    pub(super) relative_energy_balance_error: f64,
    pub(super) dynamic_balance: StructuralDynamicBalanceSnapshot,
}

#[derive(Clone, Debug)]
pub(super) struct TransientAcceptedSample {
    pub(super) schedule_ordinal: u64,
    pub(super) time_s: runmat_types::physical::Time,
    pub(super) accepted_time_step_s: runmat_types::physical::Time,
    pub(super) state: SecondOrderState,
    pub(super) summary: AssemblySummary,
    pub(super) projected_residual_norm: f64,
    pub(super) reaction: Vec<f64>,
    pub(super) strain_energy_j: runmat_types::physical::Energy,
    pub(super) kinetic_energy_j: runmat_types::physical::Energy,
    pub(super) applied_load_work_j: runmat_types::physical::Energy,
    pub(super) support_work_j: runmat_types::physical::Energy,
    pub(super) damping_dissipation_j: runmat_types::physical::Energy,
    pub(super) energy_balance_residual_j: runmat_types::physical::Energy,
    pub(super) relative_energy_balance_error: f64,
    pub(super) dynamic_balance: StructuralDynamicBalanceSnapshot,
    pub(super) integration_interval: Option<TransientIntegrationInterval>,
}

#[derive(Clone, Debug)]
pub(super) struct TransientAdvanceState {
    pub(super) schedule_ordinal: u64,
    pub(super) numerical_continuation_state: SecondOrderState,
    pub(super) integration_continuation_summary: AssemblySummary,
    pub(super) physical_continuation_state: SecondOrderState,
    pub(super) physical_continuation_summary: AssemblySummary,
    pub(super) continuation_approach: SignalSampleApproach,
    pub(super) balance: StructuralTransientBalanceAccumulator,
    pub(super) accounting: TransientSolverAccounting,
}

#[derive(Clone, Debug)]
pub(super) struct TransientBlockResult {
    pub(super) terminal: TransientAdvanceState,
    pub(super) samples: Vec<TransientAcceptedSample>,
}

pub(super) struct TransientResumeInput {
    pub(super) schedule_ordinal: u64,
    pub(super) numerical_state: SecondOrderState,
    pub(super) physical_state: StructuralTransientContinuationState,
    pub(super) continuation_approach: StructuralTransientContinuationApproach,
    pub(super) integration_summary: AssemblySummary,
    pub(super) physical_summary: AssemblySummary,
    pub(super) balance: StructuralTransientBalanceAccumulator,
    pub(super) accounting: TransientSolverAccounting,
}

#[derive(Clone, Copy)]
pub(super) struct TransientAdvanceContext<'a> {
    pub(super) initial_summary: &'a AssemblySummary,
    pub(super) options: &'a TransientSolveOptions,
    pub(super) backend_kind: LinearAlgebraBackendKind,
    pub(super) schedule: &'a StructuralTransientSchedule,
    pub(super) interval_retention: TransientIntervalRetention,
}

pub(super) fn initialize<E>(
    summary: &AssemblySummary,
    options: &TransientSolveOptions,
    schedule: &StructuralTransientSchedule,
    initial_displacement: Vec<f64>,
    initial_velocity: Vec<f64>,
    strategy: &mut impl TransientStepStrategy,
    map_physical_state: &mut impl FnMut(
        f64,
        SignalSampleApproach,
        &AssemblySummary,
        &SecondOrderState,
    ) -> Result<
        (AssemblySummary, SecondOrderState),
        TransientSolveError<E>,
    >,
) -> Result<(TransientAdvanceState, TransientInitialSample), TransientSolveError<E>> {
    let initial = strategy
        .initialize(
            summary,
            TransientInitialVectors {
                displacement: initial_displacement,
                velocity: initial_velocity,
            },
        )
        .map_err(|error| match error {
            TransientStepFailure::Usage(error) => TransientSolveError::Usage(error),
            TransientStepFailure::Detail(detail) => {
                TransientSolveError::InitialAcceleration { detail }
            }
        })?;
    let initial_time_s = schedule.points[0].time_s.canonical_value();
    require_numerical_residual(
        initial.projected_relative_residual,
        initial_time_s,
        options.residual_target,
    )?;
    let TransientNumericalStepOutcome {
        state: numerical_state,
        projected_relative_residual,
        accounting,
    } = initial;
    let (physical_summary, physical_state) = map_physical_state(
        initial_time_s,
        SignalSampleApproach::At,
        summary,
        &numerical_state,
    )?;
    let history = TransientBalanceHistory::start(&physical_summary, &physical_state)
        .map_err(super::balance::DynamicBalanceError::into_transient_solve)?;
    let sample = TransientInitialSample {
        time_s: runmat_types::physical::Time::new(initial_time_s),
        state: physical_state.clone(),
        summary: physical_summary.clone(),
        projected_residual_norm: projected_relative_residual,
        reaction: history.reaction_snapshots()[0].clone(),
        strain_energy_j: runmat_types::physical::Energy::new(history.strain_energy_j()[0]),
        kinetic_energy_j: runmat_types::physical::Energy::new(history.kinetic_energy_j()[0]),
        energy_balance_residual_j: runmat_types::physical::Energy::new(
            history.energy_balance_residual_j()[0],
        ),
        relative_energy_balance_error: history.relative_energy_balance_error()[0],
        dynamic_balance: history.balances()[0].clone(),
    };
    let advance = TransientAdvanceState {
        schedule_ordinal: 0,
        numerical_continuation_state: numerical_state,
        integration_continuation_summary: summary.clone(),
        physical_continuation_state: physical_state,
        physical_continuation_summary: physical_summary,
        continuation_approach: SignalSampleApproach::At,
        balance: history.accumulator(),
        accounting,
    };
    Ok((advance, sample))
}

pub(super) fn resume<E>(
    schedule: &StructuralTransientSchedule,
    input: TransientResumeInput,
    coordinate_count: usize,
) -> Result<TransientAdvanceState, TransientSolveError<E>> {
    schedule
        .validate()
        .map_err(|error| TransientSolveError::Schedule {
            detail: error.to_string(),
        })?;
    let ordinal = usize::try_from(input.schedule_ordinal)
        .map_err(|_| TransientSolveError::ScheduleOrdinalLimit)?;
    let schedule_point =
        schedule
            .points
            .get(ordinal)
            .ok_or(TransientSolveError::InvalidBlockBounds {
                start: input.schedule_ordinal,
                terminal: input.schedule_ordinal,
            })?;
    if input.schedule_ordinal == 0 || schedule_point.continuation != input.continuation_approach {
        return Err(TransientSolveError::InvalidContinuation);
    }
    let continuation_approach = match input.continuation_approach {
        StructuralTransientContinuationApproach::At => SignalSampleApproach::At,
        StructuralTransientContinuationApproach::RightLimit => SignalSampleApproach::RightLimit,
    };
    let physical_state = SecondOrderState::try_new(
        input.physical_state.displacement,
        input.physical_state.velocity,
        input.physical_state.acceleration,
    )
    .map_err(|error| TransientSolveError::InvalidContinuationState {
        detail: error.to_string(),
    })?;
    if input.numerical_state.dof_count() != coordinate_count
        || physical_state.dof_count() != input.physical_summary.dof_count
    {
        return Err(TransientSolveError::DofCountChanged);
    }
    TransientBalanceHistory::resume(
        &input.physical_summary,
        &physical_state,
        input.balance.clone(),
    )
    .map_err(super::balance::DynamicBalanceError::into_transient_solve)?;
    Ok(TransientAdvanceState {
        schedule_ordinal: input.schedule_ordinal,
        numerical_continuation_state: input.numerical_state,
        integration_continuation_summary: input.integration_summary,
        physical_continuation_state: physical_state,
        physical_continuation_summary: input.physical_summary,
        continuation_approach,
        balance: input.balance,
        accounting: input.accounting,
    })
}

pub(super) fn solve_complete<E: std::fmt::Display>(
    context: TransientAdvanceContext<'_>,
    initial_vectors: TransientInitialVectors,
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
) -> Result<(TransientInitialSample, TransientBlockResult), TransientSolveError<E>> {
    let TransientAdvanceContext {
        initial_summary: summary,
        options,
        backend_kind,
        schedule,
        interval_retention: _,
    } = context;
    let mut strategy =
        DirectTransientStepStrategy::new(summary, backend_kind, options.residual_target);
    let (mut state, initial) = initialize(
        summary,
        options,
        schedule,
        initial_vectors.displacement,
        initial_vectors.velocity,
        &mut strategy,
        map_physical_state,
    )?;
    let final_ordinal = u64::try_from(schedule.points.len() - 1)
        .map_err(|_| TransientSolveError::ScheduleOrdinalLimit)?;
    let mut samples = Vec::with_capacity(schedule.points.len() - 1);
    let mut accelerator_evidence = Vec::new();
    loop {
        let terminal_ordinal = state
            .schedule_ordinal
            .checked_add(LOCAL_TRANSIENT_BLOCK_INTERVALS)
            .ok_or(TransientSolveError::ScheduleOrdinalLimit)?
            .min(final_ordinal);
        let block = advance(
            context,
            state,
            terminal_ordinal,
            assembly_at,
            map_physical_state,
            &mut strategy,
        )?;
        samples.extend(block.samples);
        let mut terminal = block.terminal;
        accelerator_evidence.append(&mut terminal.accounting.accelerator_evidence);
        if terminal_ordinal == final_ordinal {
            terminal.accounting.accelerator_evidence = accelerator_evidence;
            return Ok((initial, TransientBlockResult { terminal, samples }));
        }
        let continuation_approach = match terminal.continuation_approach {
            SignalSampleApproach::At => StructuralTransientContinuationApproach::At,
            SignalSampleApproach::RightLimit => StructuralTransientContinuationApproach::RightLimit,
            SignalSampleApproach::LeftLimit => {
                return Err(TransientSolveError::InvalidContinuation)
            }
        };
        state = resume(
            schedule,
            TransientResumeInput {
                schedule_ordinal: terminal.schedule_ordinal,
                numerical_state: terminal.numerical_continuation_state,
                physical_state: StructuralTransientContinuationState {
                    displacement: terminal.physical_continuation_state.displacement().to_vec(),
                    velocity: terminal.physical_continuation_state.velocity().to_vec(),
                    acceleration: terminal.physical_continuation_state.acceleration().to_vec(),
                },
                continuation_approach,
                integration_summary: terminal.integration_continuation_summary,
                physical_summary: terminal.physical_continuation_summary,
                balance: terminal.balance,
                accounting: strategy.resumed_accounting(terminal.accounting.counters),
            },
            strategy.coordinate_count(),
        )?;
    }
}

pub(super) fn advance<E>(
    context: TransientAdvanceContext<'_>,
    mut state: TransientAdvanceState,
    terminal_schedule_ordinal: u64,
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
    strategy: &mut impl TransientStepStrategy,
) -> Result<TransientBlockResult, TransientSolveError<E>> {
    let TransientAdvanceContext {
        initial_summary,
        options,
        backend_kind: _,
        schedule,
        interval_retention,
    } = context;
    let start = usize::try_from(state.schedule_ordinal)
        .map_err(|_| TransientSolveError::ScheduleOrdinalLimit)?;
    let terminal = usize::try_from(terminal_schedule_ordinal)
        .map_err(|_| TransientSolveError::ScheduleOrdinalLimit)?;
    if terminal <= start || terminal >= schedule.points.len() {
        return Err(TransientSolveError::InvalidBlockBounds {
            start: state.schedule_ordinal,
            terminal: terminal_schedule_ordinal,
        });
    }
    let mut history = TransientBalanceHistory::resume(
        &state.physical_continuation_summary,
        &state.physical_continuation_state,
        state.balance.clone(),
    )
    .map_err(super::balance::DynamicBalanceError::into_transient_solve)?;
    let mut samples = Vec::with_capacity(terminal - start);
    let progress_operation = options.progress_operation.as_str();
    let final_ordinal = u64::try_from(schedule.points.len() - 1)
        .map_err(|_| TransientSolveError::ScheduleOrdinalLimit)?;

    for (ordinal, schedule_point) in schedule
        .points
        .iter()
        .enumerate()
        .take(terminal + 1)
        .skip(start + 1)
    {
        if is_cancelled() {
            emit_phase(
                progress_operation,
                FeaProgressPhase::Solve,
                FeaProgressStatus::Cancelled,
                "transient solve cancelled",
                Some(state.schedule_ordinal),
                Some(final_ordinal),
            );
            return Err(TransientSolveError::Cancelled);
        }
        let current_time_s = schedule.points[ordinal - 1].time_s.canonical_value();
        let next_time_s = schedule_point.time_s.canonical_value();
        let approach = match schedule_point.arrival {
            StructuralTransientScheduleArrival::LeftLimit => SignalSampleApproach::LeftLimit,
            StructuralTransientScheduleArrival::Continuous => SignalSampleApproach::At,
            StructuralTransientScheduleArrival::Initial => {
                return Err(TransientSolveError::Schedule {
                    detail: "a noninitial transient schedule point is marked initial".into(),
                });
            }
        };
        let time_step_s = next_time_s - current_time_s;
        emit_phase(
            progress_operation,
            FeaProgressPhase::Solve,
            FeaProgressStatus::Advanced,
            format!("solving transient step ending at {next_time_s} s"),
            Some(u64::try_from(ordinal).map_err(|_| TransientSolveError::ScheduleOrdinalLimit)?),
            Some(final_ordinal),
        );
        let step_summary =
            assembly_at(next_time_s, approach).map_err(TransientSolveError::Assembly)?;
        require_stable_system(initial_summary, &step_summary)?;
        let previous_physical_state = state.physical_continuation_state.clone();
        let previous_physical_summary = state.physical_continuation_summary.clone();
        let solved = strategy
            .advance(
                &step_summary,
                &state.numerical_continuation_state,
                time_step_s,
            )
            .map_err(|error| match error {
                TransientStepFailure::Usage(error) => TransientSolveError::Usage(error),
                TransientStepFailure::Detail(detail) => TransientSolveError::Step {
                    time_s: runmat_types::physical::Time::new(next_time_s),
                    detail,
                },
            })?;
        require_numerical_residual(
            solved.projected_relative_residual,
            next_time_s,
            options.residual_target,
        )?;
        let (physical_step_summary, physical_step_state) =
            map_physical_state(next_time_s, approach, &step_summary, &solved.state)?;
        history
            .push_step(TransientBalanceStep {
                integration_start_summary: &previous_physical_summary,
                integration_start_state: &previous_physical_state,
                integration_end_summary: &physical_step_summary,
                integration_end_state: &physical_step_state,
                published_summary: &physical_step_summary,
                published_state: &physical_step_state,
                time_step_s,
            })
            .map_err(super::balance::DynamicBalanceError::into_transient_solve)?;
        let interval =
            (interval_retention == TransientIntervalRetention::RetainForMotion).then(|| {
                TransientIntegrationInterval {
                    start_time_s: runmat_types::physical::Time::new(current_time_s),
                    start_approach: state.continuation_approach,
                    start_summary: previous_physical_summary,
                    start_displacement: previous_physical_state.displacement().to_vec(),
                    start_velocity: previous_physical_state.velocity().to_vec(),
                    start_acceleration: previous_physical_state.acceleration().to_vec(),
                    end_time_s: runmat_types::physical::Time::new(next_time_s),
                    end_approach: approach,
                    end_summary: physical_step_summary.clone(),
                    end_displacement: physical_step_state.displacement().to_vec(),
                    end_velocity: physical_step_state.velocity().to_vec(),
                    end_acceleration: physical_step_state.acceleration().to_vec(),
                }
            });
        let mut public_state = physical_step_state.clone();
        let mut public_summary = physical_step_summary.clone();
        let mut projected_residual_norm = solved.projected_relative_residual;
        state.accounting.merge(solved.accounting)?;
        state.numerical_continuation_state = solved.state;
        state.integration_continuation_summary = step_summary;
        state.physical_continuation_state = physical_step_state;
        state.physical_continuation_summary = physical_step_summary;
        state.continuation_approach = approach;

        if approach == SignalSampleApproach::LeftLimit {
            let point_summary = assembly_at(next_time_s, SignalSampleApproach::At)
                .map_err(TransientSolveError::Assembly)?;
            require_stable_system(initial_summary, &point_summary)?;
            let point_state = strategy
                .refresh_acceleration(&point_summary, &state.numerical_continuation_state)
                .map_err(|error| match error {
                    TransientStepFailure::Usage(error) => TransientSolveError::Usage(error),
                    TransientStepFailure::Detail(detail) => {
                        TransientSolveError::AccelerationRefresh {
                            time_s: runmat_types::physical::Time::new(next_time_s),
                            detail,
                        }
                    }
                })?;
            require_numerical_residual(
                point_state.projected_relative_residual,
                next_time_s,
                options.residual_target,
            )?;
            state.accounting.merge(point_state.accounting)?;
            (public_summary, public_state) = map_physical_state(
                next_time_s,
                SignalSampleApproach::At,
                &point_summary,
                &point_state.state,
            )?;
            projected_residual_norm = point_state.projected_relative_residual;
            history
                .refresh_reaction(&public_summary, &public_state)
                .map_err(super::balance::DynamicBalanceError::into_transient_solve)?;

            let right_limit_summary = assembly_at(next_time_s, SignalSampleApproach::RightLimit)
                .map_err(TransientSolveError::Assembly)?;
            require_stable_system(initial_summary, &right_limit_summary)?;
            let right_limit_state = strategy
                .refresh_acceleration(&right_limit_summary, &state.numerical_continuation_state)
                .map_err(|error| match error {
                    TransientStepFailure::Usage(error) => TransientSolveError::Usage(error),
                    TransientStepFailure::Detail(detail) => {
                        TransientSolveError::AccelerationRefresh {
                            time_s: runmat_types::physical::Time::new(next_time_s),
                            detail,
                        }
                    }
                })?;
            require_numerical_residual(
                right_limit_state.projected_relative_residual,
                next_time_s,
                options.residual_target,
            )?;
            state.accounting.merge(right_limit_state.accounting)?;
            let (physical_right_summary, physical_right_state) = map_physical_state(
                next_time_s,
                SignalSampleApproach::RightLimit,
                &right_limit_summary,
                &right_limit_state.state,
            )?;
            state.numerical_continuation_state = right_limit_state.state;
            state.integration_continuation_summary = right_limit_summary;
            state.physical_continuation_state = physical_right_state;
            state.physical_continuation_summary = physical_right_summary;
            state.continuation_approach = SignalSampleApproach::RightLimit;
        }

        let history_index = history.balances().len() - 1;
        samples.push(TransientAcceptedSample {
            schedule_ordinal: u64::try_from(ordinal)
                .map_err(|_| TransientSolveError::ScheduleOrdinalLimit)?,
            time_s: runmat_types::physical::Time::new(next_time_s),
            accepted_time_step_s: runmat_types::physical::Time::new(time_step_s),
            state: public_state,
            summary: public_summary,
            projected_residual_norm,
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
            integration_interval: interval,
        });
        state.schedule_ordinal =
            u64::try_from(ordinal).map_err(|_| TransientSolveError::ScheduleOrdinalLimit)?;
    }
    state.balance = history.accumulator();
    Ok(TransientBlockResult {
        terminal: state,
        samples,
    })
}

fn require_numerical_residual<E>(
    residual_norm: f64,
    time_s: f64,
    target: f64,
) -> Result<(), TransientSolveError<E>> {
    if !residual_norm.is_finite() || residual_norm > target {
        return Err(TransientSolveError::ResidualTargetExceeded {
            time_s: runmat_types::physical::Time::new(time_s),
            residual_norm,
            target,
        });
    }
    Ok(())
}
