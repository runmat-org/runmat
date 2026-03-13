use runmat_analysis_core::{validate_model, AnalysisField, AnalysisModel};

use crate::{
    assembly::assemble_linear_system,
    contracts::{ComputeBackend, FeaNonlinearRunResult, FeaRunError, FeaRunResult},
    diagnostics::builders::extend_common_run_diagnostics,
    solve::nonlinear::{solve_nonlinear_system, NonlinearSolveOptions},
};

pub fn run_nonlinear(
    model: &AnalysisModel,
    backend: ComputeBackend,
) -> Result<FeaNonlinearRunResult, FeaRunError> {
    run_nonlinear_with_options(model, backend, NonlinearSolveOptions::default())
}

pub fn run_nonlinear_with_options(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: NonlinearSolveOptions,
) -> Result<FeaNonlinearRunResult, FeaRunError> {
    validate_model(model).map_err(|err| FeaRunError::InvalidModel(err.to_string()))?;
    let prep_context = options.prep_context;
    let thermo_context = options.thermo_mechanical_context.clone();
    let electro_context = options.electro_thermal_context.clone();

    let summary = assemble_linear_system(model, prep_context, thermo_context, electro_context);
    let nonlinear = solve_nonlinear_system(&summary, options.clone(), backend);
    let mut diagnostics = nonlinear.diagnostics.clone();
    extend_common_run_diagnostics(
        &mut diagnostics,
        model,
        &summary,
        prep_context,
        nonlinear
            .iteration_counts
            .iter()
            .copied()
            .max()
            .unwrap_or(0) as f64,
        nonlinear
            .residual_norms
            .iter()
            .copied()
            .fold(0.0_f64, f64::max),
        "auto",
        &nonlinear.preconditioner,
    );

    let displacement = nonlinear
        .displacement_snapshots
        .last()
        .cloned()
        .unwrap_or_else(|| vec![0.0; summary.dof_count.max(3)]);
    let von_mises = displacement
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
        * 1.0e11;

    let run = FeaRunResult {
        backend,
        solver_backend: nonlinear.solver_backend,
        solver_device_apply_k_ratio: if nonlinear.device_apply_k_attempt_count == 0 {
            0.0
        } else {
            nonlinear.device_apply_k_count as f64 / nonlinear.device_apply_k_attempt_count as f64
        },
        solver_method: nonlinear.solver_method,
        preconditioner: nonlinear.preconditioner,
        solver_host_sync_count: nonlinear.solver_host_sync_count,
        diagnostics,
        displacement_field: AnalysisField::host_f64(
            "displacement",
            vec![displacement.len()],
            displacement,
        ),
        von_mises_field: AnalysisField::host_f64("von_mises", vec![1], vec![von_mises]),
    };

    let displacement_snapshots = nonlinear
        .displacement_snapshots
        .into_iter()
        .enumerate()
        .map(|(index, snapshot)| {
            AnalysisField::host_f64(
                format!("nonlinear_displacement_inc{}", index),
                vec![snapshot.len()],
                snapshot,
            )
        })
        .collect();

    Ok(FeaNonlinearRunResult {
        run,
        load_factors: nonlinear.load_factors,
        displacement_snapshots,
        residual_norms: nonlinear.residual_norms,
        increment_norms: nonlinear.increment_norms,
        iteration_counts: nonlinear.iteration_counts,
        failed_increments: nonlinear.failed_increments,
        line_search_backtracks: nonlinear.line_search_backtracks,
        max_line_search_backtracks_per_increment: nonlinear
            .max_line_search_backtracks_per_increment,
        tangent_rebuild_count: nonlinear.tangent_rebuild_count,
        iteration_spike_count: nonlinear.iteration_spike_count,
        convergence_stall_count: nonlinear.convergence_stall_count,
        backtrack_burst_count: nonlinear.backtrack_burst_count,
    })
}
