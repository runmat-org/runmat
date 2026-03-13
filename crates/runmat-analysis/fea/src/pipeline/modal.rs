use runmat_analysis_core::{validate_model, AnalysisField, AnalysisModel};

use crate::{
    assembly::assemble_linear_system,
    contracts::{ComputeBackend, FeaModalRunResult, FeaRunError, FeaRunResult, ModalSolveOptions},
    diagnostics::builders::extend_common_run_diagnostics,
    solve::modal::solve_modal_system,
};

pub fn run_modal(
    model: &AnalysisModel,
    backend: ComputeBackend,
) -> Result<FeaModalRunResult, FeaRunError> {
    run_modal_with_options(model, backend, ModalSolveOptions::default())
}

pub fn run_modal_with_options(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: ModalSolveOptions,
) -> Result<FeaModalRunResult, FeaRunError> {
    validate_model(model).map_err(|err| FeaRunError::InvalidModel(err.to_string()))?;

    let summary = assemble_linear_system(
        model,
        options.prep_context,
        options.thermo_mechanical_context,
        options.electro_thermal_context,
    );
    let modal = solve_modal_system(&summary, options.mode_count, backend);
    let mut diagnostics = modal.diagnostics.clone();
    extend_common_run_diagnostics(
        &mut diagnostics,
        model,
        &summary,
        options.prep_context,
        mode_shapes_iteration_proxy(&modal.residual_norms),
        modal.residual_norms.iter().copied().fold(0.0_f64, f64::max),
        "auto",
        if backend == ComputeBackend::Gpu {
            "jacobi"
        } else {
            "none"
        },
    );

    let displacement = modal
        .mode_shapes
        .first()
        .cloned()
        .unwrap_or_else(|| vec![0.0; summary.dof_count.max(3)]);
    let von_mises = displacement
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
        * 1.0e11;

    let run = FeaRunResult {
        backend,
        solver_backend: modal.solver_backend,
        solver_device_apply_k_ratio: if modal.device_apply_k_attempt_count == 0 {
            0.0
        } else {
            modal.device_apply_k_count as f64 / modal.device_apply_k_attempt_count as f64
        },
        solver_method: modal.solver_method,
        preconditioner: if backend == ComputeBackend::Gpu {
            "jacobi".to_string()
        } else {
            "none".to_string()
        },
        solver_host_sync_count: modal.solver_host_sync_count,
        diagnostics,
        displacement_field: AnalysisField::host_f64(
            "displacement",
            vec![displacement.len()],
            displacement,
        ),
        von_mises_field: AnalysisField::host_f64("von_mises", vec![1], vec![von_mises]),
    };

    let mode_shapes = modal
        .mode_shapes
        .into_iter()
        .enumerate()
        .map(|(index, shape)| {
            AnalysisField::host_f64(
                format!("mode_shape_{}", index + 1),
                vec![shape.len()],
                shape,
            )
        })
        .collect();

    Ok(FeaModalRunResult {
        run,
        eigenvalues_hz: modal.eigenvalues_hz,
        mode_shapes,
        residual_norms: modal.residual_norms,
    })
}

fn mode_shapes_iteration_proxy(residual_norms: &[f64]) -> f64 {
    residual_norms.len() as f64
}
