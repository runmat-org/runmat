use runmat_analysis_core::SignalSampleApproach;

use crate::{
    assembly::AssemblySummary,
    contracts::{SolverMethod, SolverPreconditioner, StructuralDynamicBalanceSnapshot},
    execution::{StructuralTransientSchedule, StructuralTransientScheduleArrival},
    solve::backend::kind::LinearAlgebraBackendKind,
};

use super::{
    canonical_transient_schedule, is_cancelled, require_stable_system, transient_diagnostics,
    TransientIntegrationInterval, TransientSolveError, TransientSolveOptions, TransientSolveResult,
    TransientStepBoundary,
};

const LOCAL_EMPTY_BLOCK_INTERVALS: u64 = 256;

#[derive(Clone, Debug)]
pub(super) struct EmptyTransientAdvanceState {
    pub(super) schedule_ordinal: u64,
    pub(super) continuation_summary: AssemblySummary,
    pub(super) continuation_approach: SignalSampleApproach,
}

#[derive(Clone, Debug)]
pub(super) struct EmptyTransientAcceptedSample {
    pub(super) schedule_ordinal: u64,
    pub(super) time_s: runmat_types::physical::Time,
    pub(super) accepted_time_step_s: runmat_types::physical::Time,
    pub(super) public_summary: AssemblySummary,
    pub(super) integration_interval: TransientIntegrationInterval,
}

#[derive(Clone, Debug)]
pub(super) struct EmptyTransientBlockResult {
    pub(super) terminal: EmptyTransientAdvanceState,
    pub(super) samples: Vec<EmptyTransientAcceptedSample>,
}

pub(super) fn solve<E>(
    summary: &AssemblySummary,
    options: &TransientSolveOptions,
    start_time_s: f64,
    end_time_s: f64,
    step_boundaries: &[TransientStepBoundary],
    assembly_at: &mut impl FnMut(f64, SignalSampleApproach) -> Result<AssemblySummary, E>,
) -> Result<TransientSolveResult, TransientSolveError<E>> {
    let interval_count = options.time_step_policy.interval_count();
    let schedule =
        canonical_transient_schedule(interval_count, start_time_s, end_time_s, step_boundaries)
            .map_err(|detail| TransientSolveError::Schedule { detail })?;
    let mut result = initial_result(summary, options, schedule.clone());
    let final_ordinal = u64::try_from(schedule.points.len() - 1)
        .map_err(|_| TransientSolveError::ScheduleOrdinalLimit)?;
    let mut state = EmptyTransientAdvanceState {
        schedule_ordinal: 0,
        continuation_summary: summary.clone(),
        continuation_approach: SignalSampleApproach::At,
    };
    while state.schedule_ordinal < final_ordinal {
        let terminal = state
            .schedule_ordinal
            .checked_add(LOCAL_EMPTY_BLOCK_INTERVALS)
            .ok_or(TransientSolveError::ScheduleOrdinalLimit)?
            .min(final_ordinal);
        let block = advance(summary, &schedule, state, terminal, assembly_at)?;
        for sample in block.samples {
            debug_assert_eq!(sample.schedule_ordinal as usize, result.time_points_s.len());
            result.time_points_s.push(sample.time_s.canonical_value());
            result.displacement_snapshots.push(Vec::new());
            result.velocity_snapshots.push(Vec::new());
            result.acceleration_snapshots.push(Vec::new());
            result.residual_norms.push(0.0);
            result
                .accepted_time_steps_s
                .push(sample.accepted_time_step_s.canonical_value());
            result.assembly_snapshots.push(sample.public_summary);
            result
                .integration_intervals
                .push(sample.integration_interval);
        }
        state = block.terminal;
    }
    complete_histories(&mut result, options);
    Ok(result)
}

pub(super) fn advance<E>(
    initial_summary: &AssemblySummary,
    schedule: &StructuralTransientSchedule,
    mut state: EmptyTransientAdvanceState,
    terminal_schedule_ordinal: u64,
    assembly_at: &mut impl FnMut(f64, SignalSampleApproach) -> Result<AssemblySummary, E>,
) -> Result<EmptyTransientBlockResult, TransientSolveError<E>> {
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
    let mut samples = Vec::with_capacity(terminal - start);
    for (ordinal, schedule_point) in schedule
        .points
        .iter()
        .enumerate()
        .take(terminal + 1)
        .skip(start + 1)
    {
        if is_cancelled() {
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
                })
            }
        };
        let time_step_s = next_time_s - current_time_s;
        let step_summary =
            assembly_at(next_time_s, approach).map_err(TransientSolveError::Assembly)?;
        require_stable_system(initial_summary, &step_summary)?;
        let integration_interval = TransientIntegrationInterval::empty(
            current_time_s,
            state.continuation_approach,
            state.continuation_summary.clone(),
            next_time_s,
            approach,
            step_summary.clone(),
        );

        let public_summary = if approach == SignalSampleApproach::LeftLimit {
            let point_summary = assembly_at(next_time_s, SignalSampleApproach::At)
                .map_err(TransientSolveError::Assembly)?;
            require_stable_system(initial_summary, &point_summary)?;
            let right_summary = assembly_at(next_time_s, SignalSampleApproach::RightLimit)
                .map_err(TransientSolveError::Assembly)?;
            require_stable_system(initial_summary, &right_summary)?;
            state.continuation_summary = right_summary;
            state.continuation_approach = SignalSampleApproach::RightLimit;
            point_summary
        } else {
            state.continuation_summary = step_summary.clone();
            state.continuation_approach = approach;
            step_summary
        };
        state.schedule_ordinal =
            u64::try_from(ordinal).map_err(|_| TransientSolveError::ScheduleOrdinalLimit)?;
        samples.push(EmptyTransientAcceptedSample {
            schedule_ordinal: state.schedule_ordinal,
            time_s: runmat_types::physical::Time::new(next_time_s),
            accepted_time_step_s: runmat_types::physical::Time::new(time_step_s),
            public_summary,
            integration_interval,
        });
    }
    Ok(EmptyTransientBlockResult {
        terminal: state,
        samples,
    })
}

fn initial_result(
    summary: &AssemblySummary,
    options: &TransientSolveOptions,
    schedule: crate::execution::StructuralTransientSchedule,
) -> TransientSolveResult {
    TransientSolveResult {
        time_points_s: vec![schedule.points[0].time_s.canonical_value()],
        schedule,
        converged_steps: 0,
        total_steps: 0,
        displacement_snapshots: vec![Vec::new()],
        velocity_snapshots: vec![Vec::new()],
        acceleration_snapshots: vec![Vec::new()],
        residual_norms: vec![0.0],
        accepted_time_steps_s: Vec::new(),
        diagnostics: transient_diagnostics(
            options,
            &[],
            &[],
            &[0.0],
            &[0.0],
            &[0.0],
            LinearAlgebraBackendKind::CpuReference,
        ),
        solver_method: SolverMethod::NewmarkAverageAccelerationPcg,
        solver_backend: LinearAlgebraBackendKind::CpuReference,
        solver_host_sync_count: 0,
        device_apply_k_count: 0,
        device_apply_k_attempt_count: 0,
        preconditioner: SolverPreconditioner::None,
        accelerator_evidence: Vec::new(),
        assembly_snapshots: vec![summary.clone()],
        strain_energy_snapshots: vec![0.0],
        kinetic_energy_snapshots: vec![0.0],
        reaction_snapshots: vec![Vec::new()],
        integration_intervals: Vec::new(),
        external_work_j: runmat_types::physical::Quantity::new(vec![0.0]),
        support_work_j: runmat_types::physical::Quantity::new(vec![0.0]),
        damping_dissipation_j: runmat_types::physical::Quantity::new(vec![0.0]),
        energy_balance_residual_j: runmat_types::physical::Quantity::new(vec![0.0]),
        relative_energy_balance_error: vec![0.0],
        dynamic_balance_snapshots: vec![empty_balance()],
    }
}

fn complete_histories(result: &mut TransientSolveResult, options: &TransientSolveOptions) {
    result.total_steps = result.accepted_time_steps_s.len();
    result.converged_steps = result.total_steps;
    let snapshot_count = result.time_points_s.len();
    result.reaction_snapshots.resize(snapshot_count, Vec::new());
    result.strain_energy_snapshots.resize(snapshot_count, 0.0);
    result.kinetic_energy_snapshots.resize(snapshot_count, 0.0);
    result
        .external_work_j
        .try_resize(snapshot_count, runmat_types::physical::Energy::new(0.0))
        .expect("finite zero energy");
    result
        .support_work_j
        .try_resize(snapshot_count, runmat_types::physical::Energy::new(0.0))
        .expect("finite zero energy");
    result
        .damping_dissipation_j
        .try_resize(snapshot_count, runmat_types::physical::Energy::new(0.0))
        .expect("finite zero energy");
    result
        .energy_balance_residual_j
        .try_resize(snapshot_count, runmat_types::physical::Energy::new(0.0))
        .expect("finite zero energy");
    result
        .relative_energy_balance_error
        .resize(snapshot_count, 0.0);
    result
        .dynamic_balance_snapshots
        .resize_with(snapshot_count, empty_balance);
    result.diagnostics = transient_diagnostics(
        options,
        &result.accepted_time_steps_s,
        &result.residual_norms,
        &result.strain_energy_snapshots,
        &result.kinetic_energy_snapshots,
        &result.relative_energy_balance_error,
        LinearAlgebraBackendKind::CpuReference,
    );
}

fn empty_balance() -> StructuralDynamicBalanceSnapshot {
    StructuralDynamicBalanceSnapshot {
        applied_load: Vec::new(),
        inertia_force: Vec::new(),
        damping_force: Vec::new(),
        elastic_force: Vec::new(),
        equilibrium_residual: Vec::new(),
    }
}
