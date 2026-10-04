use super::super::{
    solid::{AcceptedSolidIncrement, AcceptedSolidState},
    NonlinearSolveResult,
};
use crate::{
    contracts::{SolverMethod, SolverPreconditioner},
    solve::backend::kind::LinearAlgebraBackendKind,
    FeaRunError,
};

pub(super) fn initial_result(
    state: &AcceptedSolidState,
    evaluation: &crate::assembly::nonlinear_solid::NonlinearSolidEvaluation,
) -> Result<NonlinearSolveResult, FeaRunError> {
    let fields = super::recovery::recover(evaluation, &state.constitutive)?;
    Ok(NonlinearSolveResult {
        accepted_state: state.clone(),
        converged_increments: 0,
        total_increments: 0,
        initial_load_factor: state.load_factor,
        load_factors: Vec::new(),
        initial_displacement: state.displacement.clone(),
        initial_equivalent_plastic_strain: fields.alpha,
        initial_contact_pressure: Vec::new(),
        initial_contact_gap: Vec::new(),
        displacement_snapshots: Vec::new(),
        internal_force_snapshots: Vec::new(),
        stress_snapshots: Vec::new(),
        total_strain_snapshots: Vec::new(),
        von_mises_snapshots: Vec::new(),
        plastic_strain_snapshots: Vec::new(),
        equivalent_plastic_strain_snapshots: Vec::new(),
        contact_pressure_snapshots: Vec::new(),
        contact_gap_snapshots: Vec::new(),
        residual_norms: Vec::new(),
        increment_norms: Vec::new(),
        relative_increment_norms: Vec::new(),
        iteration_counts: Vec::new(),
        failed_increments: 0,
        line_search_backtracks: 0,
        max_line_search_backtracks_per_increment: 0,
        tangent_rebuild_count: 0,
        iteration_spike_count: 0,
        convergence_stall_count: 0,
        backtrack_burst_count: 0,
        diagnostics: Vec::new(),
        solver_method: SolverMethod::IncrementalNewtonRaphson,
        solver_backend: LinearAlgebraBackendKind::CpuReference,
        solver_host_sync_count: 0,
        device_apply_k_count: 0,
        device_apply_k_attempt_count: 0,
        preconditioner: SolverPreconditioner::Jacobi,
        accelerator_evidence: Vec::new(),
    })
}

pub(super) fn append_increment(
    result: &mut NonlinearSolveResult,
    increment: &AcceptedSolidIncrement,
) -> Result<(), FeaRunError> {
    let evaluation = &increment.evaluation;
    let super::recovery::RecoveredSolidFields {
        stress,
        total_strain,
        plastic,
        alpha,
        von_mises,
    } = super::recovery::recover(evaluation, &increment.state.constitutive)?;
    if increment.attempted_work.iterations < increment.iterations
        || increment.attempted_work.backtracks < increment.backtracks
    {
        return Err(FeaRunError::Solve(
            "attempted nonlinear work omits accepted trial work".into(),
        ));
    }
    result.load_factors.push(increment.state.load_factor);
    result
        .displacement_snapshots
        .push(increment.state.displacement.clone());
    result
        .internal_force_snapshots
        .push(evaluation.internal_force.clone());
    result.stress_snapshots.push(stress);
    result.total_strain_snapshots.push(total_strain);
    result.plastic_strain_snapshots.push(plastic);
    result.equivalent_plastic_strain_snapshots.push(alpha);
    result.von_mises_snapshots.push(von_mises);
    result.contact_pressure_snapshots.push(Vec::new());
    result.contact_gap_snapshots.push(Vec::new());
    result.residual_norms.push(increment.relative_residual);
    result.increment_norms.push(increment.correction_norm);
    result
        .relative_increment_norms
        .push(increment.relative_increment);
    result
        .iteration_counts
        .push(increment.attempted_work.iterations);
    result.failed_increments += increment.cutbacks;
    result.line_search_backtracks += increment.attempted_work.backtracks;
    result.max_line_search_backtracks_per_increment = result
        .max_line_search_backtracks_per_increment
        .max(increment.attempted_work.backtracks);
    result.tangent_rebuild_count += increment.attempted_work.tangent_evaluations;
    Ok(())
}
