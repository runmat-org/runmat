use runmat_analysis_core::SignalSampleApproach;

use crate::{
    assembly::AssemblySummary,
    contracts::{SolverMethod, SolverPreconditioner, StructuralDynamicBalanceSnapshot},
    solve::backend::kind::LinearAlgebraBackendKind,
};

use super::{
    is_cancelled, next_step_boundary, require_stable_system, transient_diagnostics,
    TransientIntegrationInterval, TransientSolveError, TransientSolveOptions, TransientSolveResult,
    TransientStepBoundary,
};

pub(super) fn solve<E>(
    summary: &AssemblySummary,
    options: &TransientSolveOptions,
    start_time_s: f64,
    end_time_s: f64,
    step_boundaries: &[TransientStepBoundary],
    assembly_at: &mut impl FnMut(f64, SignalSampleApproach) -> Result<AssemblySummary, E>,
) -> Result<TransientSolveResult, TransientSolveError<E>> {
    let mut result = initial_result(summary, options, start_time_s);
    let interval_count = options.time_step_policy.interval_count();
    let nominal_time_step_s = (end_time_s - start_time_s) / interval_count as f64;
    let mut integration_summary = summary.clone();
    let mut integration_approach = SignalSampleApproach::At;

    while result
        .time_points_s
        .last()
        .is_some_and(|time| *time < end_time_s)
    {
        if is_cancelled() {
            return Err(TransientSolveError::Cancelled);
        }
        let current_time_s = result.time_points_s.last().copied().unwrap_or(start_time_s);
        let proposed_dt_s = nominal_time_step_s.min(end_time_s - current_time_s);
        let (next_time_s, approach) =
            next_step_boundary(current_time_s, proposed_dt_s, end_time_s, step_boundaries);
        let time_step_s = next_time_s - current_time_s;
        let step_summary =
            assembly_at(next_time_s, approach).map_err(TransientSolveError::Assembly)?;
        require_stable_system(summary, &step_summary)?;
        result
            .integration_intervals
            .push(TransientIntegrationInterval::empty(
                current_time_s,
                integration_approach,
                integration_summary.clone(),
                next_time_s,
                approach,
                step_summary.clone(),
            ));
        result.time_points_s.push(next_time_s);
        result.displacement_snapshots.push(Vec::new());
        result.velocity_snapshots.push(Vec::new());
        result.acceleration_snapshots.push(Vec::new());
        result.residual_norms.push(0.0);
        result.accepted_time_steps_s.push(time_step_s);

        let public_summary = if approach == SignalSampleApproach::LeftLimit {
            let point_summary = assembly_at(next_time_s, SignalSampleApproach::At)
                .map_err(TransientSolveError::Assembly)?;
            require_stable_system(summary, &point_summary)?;
            let right_summary = assembly_at(next_time_s, SignalSampleApproach::RightLimit)
                .map_err(TransientSolveError::Assembly)?;
            require_stable_system(summary, &right_summary)?;
            integration_summary = right_summary;
            integration_approach = SignalSampleApproach::RightLimit;
            point_summary
        } else {
            integration_summary = step_summary.clone();
            integration_approach = approach;
            step_summary
        };
        result.assembly_snapshots.push(public_summary);
    }

    complete_histories(&mut result, options);
    Ok(result)
}

fn initial_result(
    summary: &AssemblySummary,
    options: &TransientSolveOptions,
    start_time_s: f64,
) -> TransientSolveResult {
    TransientSolveResult {
        converged_steps: 0,
        total_steps: 0,
        time_points_s: vec![start_time_s],
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
        external_work_j: vec![0.0],
        support_work_j: vec![0.0],
        damping_dissipation_j: vec![0.0],
        energy_balance_residual_j: vec![0.0],
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
    result.external_work_j.resize(snapshot_count, 0.0);
    result.support_work_j.resize(snapshot_count, 0.0);
    result.damping_dissipation_j.resize(snapshot_count, 0.0);
    result.energy_balance_residual_j.resize(snapshot_count, 0.0);
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
