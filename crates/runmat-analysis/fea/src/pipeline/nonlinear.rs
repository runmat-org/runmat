use runmat_analysis_core::{validate_model, AnalysisField, AnalysisModel};

use crate::{
    assembly::assemble_linear_system,
    contracts::{
        fea_nonlinear_displacement_field_id, ComputeBackend, FeaNonlinearRunResult, FeaRunError,
        FeaRunResult, FEA_FIELD_STRUCTURAL_DISPLACEMENT, FEA_FIELD_STRUCTURAL_VON_MISES,
    },
    diagnostics::builders::{extend_common_run_diagnostics, CommonRunDiagnosticInputs},
    progress::{check_cancelled, emit_phase, FeaProgressPhase, FeaProgressStatus},
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
    emit_phase(
        "fea.run_nonlinear",
        FeaProgressPhase::RegionResolution,
        FeaProgressStatus::Started,
        "validating nonlinear FEA model",
        Some(0),
        Some(5),
    );
    check_cancelled("fea.run_nonlinear")?;
    validate_model(model).map_err(|err| FeaRunError::InvalidModel(err.to_string()))?;
    emit_phase(
        "fea.run_nonlinear",
        FeaProgressPhase::RegionResolution,
        FeaProgressStatus::Completed,
        "nonlinear model validation complete",
        Some(1),
        Some(5),
    );
    check_cancelled("fea.run_nonlinear")?;
    let prep_context = options.prep_context;
    let thermo_context = options.thermo_mechanical_context.clone();
    let electro_context = options.electro_thermal_context.clone();

    emit_phase(
        "fea.run_nonlinear",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Started,
        "assembling nonlinear system",
        Some(1),
        Some(5),
    );
    let summary = assemble_linear_system(model, prep_context, thermo_context, electro_context);
    emit_phase(
        "fea.run_nonlinear",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Completed,
        "nonlinear system assembly complete",
        Some(2),
        Some(5),
    );
    check_cancelled("fea.run_nonlinear")?;

    emit_phase(
        "fea.run_nonlinear",
        FeaProgressPhase::Solve,
        FeaProgressStatus::Started,
        "solving nonlinear increments",
        Some(2),
        Some(5),
    );
    let nonlinear = solve_nonlinear_system(&summary, options.clone(), backend);
    emit_phase(
        "fea.run_nonlinear",
        FeaProgressPhase::Solve,
        FeaProgressStatus::Completed,
        "nonlinear solve complete",
        Some(3),
        Some(5),
    );
    check_cancelled("fea.run_nonlinear")?;

    emit_phase(
        "fea.run_nonlinear",
        FeaProgressPhase::Postprocess,
        FeaProgressStatus::Started,
        "recovering nonlinear fields",
        Some(3),
        Some(5),
    );
    let mut diagnostics = nonlinear.diagnostics.clone();
    extend_common_run_diagnostics(
        &mut diagnostics,
        CommonRunDiagnosticInputs {
            model,
            summary: &summary,
            prep_context,
            iteration_metric: nonlinear
                .iteration_counts
                .iter()
                .copied()
                .max()
                .unwrap_or(0) as f64,
            residual_metric: nonlinear
                .residual_norms
                .iter()
                .copied()
                .fold(0.0_f64, f64::max),
            requested_preconditioner: "auto",
            effective_preconditioner: &nonlinear.preconditioner,
        },
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
        fields: vec![
            AnalysisField::host_f64(
                FEA_FIELD_STRUCTURAL_DISPLACEMENT,
                vec![displacement.len()],
                displacement,
            ),
            AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_VON_MISES, vec![1], vec![von_mises]),
        ],
    };

    let displacement_snapshots = nonlinear
        .displacement_snapshots
        .into_iter()
        .enumerate()
        .map(|(index, snapshot)| {
            AnalysisField::host_f64(
                fea_nonlinear_displacement_field_id(index),
                vec![snapshot.len()],
                snapshot,
            )
        })
        .collect();

    emit_phase(
        "fea.run_nonlinear",
        FeaProgressPhase::Postprocess,
        FeaProgressStatus::Completed,
        "nonlinear result field recovery complete",
        Some(4),
        Some(5),
    );
    check_cancelled("fea.run_nonlinear")?;
    emit_phase(
        "fea.run_nonlinear",
        FeaProgressPhase::Complete,
        FeaProgressStatus::Completed,
        "FEA nonlinear run complete",
        Some(5),
        Some(5),
    );

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
