use runmat_analysis_core::{validate_model, AnalysisField, AnalysisModel};

use crate::{
    assembly::assemble_linear_system,
    contracts::{ComputeBackend, FeaRunError, FeaRunResult, FeaTransientRunResult},
    diagnostics::builders::extend_common_run_diagnostics,
    solve::transient::{solve_transient_system, TransientSolveOptions},
};

pub fn run_transient(
    model: &AnalysisModel,
    backend: ComputeBackend,
) -> Result<FeaTransientRunResult, FeaRunError> {
    run_transient_with_options(model, backend, TransientSolveOptions::default())
}

pub fn run_transient_with_options(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: TransientSolveOptions,
) -> Result<FeaTransientRunResult, FeaRunError> {
    validate_model(model).map_err(|err| FeaRunError::InvalidModel(err.to_string()))?;
    let prep_context = options.prep_context;
    let thermo_context = options.thermo_mechanical_context.clone();
    let electro_context = options.electro_thermal_context.clone();

    let summary = assemble_linear_system(model, prep_context, thermo_context, electro_context);
    let transient = solve_transient_system(&summary, options.clone(), backend);
    let mut diagnostics = transient.diagnostics.clone();
    extend_common_run_diagnostics(
        &mut diagnostics,
        model,
        &summary,
        prep_context,
        transient.converged_steps as f64,
        transient
            .residual_norms
            .iter()
            .copied()
            .fold(0.0_f64, f64::max),
        "auto",
        &transient.preconditioner,
    );

    let displacement = transient
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
        solver_backend: transient.solver_backend,
        solver_device_apply_k_ratio: if transient.device_apply_k_attempt_count == 0 {
            0.0
        } else {
            transient.device_apply_k_count as f64 / transient.device_apply_k_attempt_count as f64
        },
        solver_method: transient.solver_method,
        preconditioner: transient.preconditioner,
        solver_host_sync_count: transient.solver_host_sync_count,
        diagnostics,
        displacement_field: AnalysisField::host_f64(
            "displacement",
            vec![displacement.len()],
            displacement,
        ),
        von_mises_field: AnalysisField::host_f64("von_mises", vec![1], vec![von_mises]),
    };

    let displacement_snapshots = transient
        .displacement_snapshots
        .into_iter()
        .enumerate()
        .map(|(index, snapshot)| {
            AnalysisField::host_f64(
                format!("displacement_t{}", index),
                vec![snapshot.len()],
                snapshot,
            )
        })
        .collect();

    Ok(FeaTransientRunResult {
        run,
        time_points_s: transient.time_points_s,
        displacement_snapshots,
        residual_norms: transient.residual_norms,
    })
}
