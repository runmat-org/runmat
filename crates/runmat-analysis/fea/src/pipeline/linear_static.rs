use runmat_analysis_core::{validate_model, AnalysisModel};

use crate::{
    assembly::assemble_linear_system,
    contracts::{ComputeBackend, FeaRunError, FeaRunResult, LinearStaticSolveOptions},
    diagnostics::{
        builders::{extend_common_run_diagnostics, CommonRunDiagnosticInputs},
        FeaDiagnostic, FeaDiagnosticSeverity,
    },
    post::fields::recover_result_fields,
    progress::{check_cancelled, emit_phase, FeaProgressPhase, FeaProgressStatus},
    solve::{backend::build_backend, linear::solve_linear_system},
};

pub fn run_linear_static(
    model: &AnalysisModel,
    backend: ComputeBackend,
) -> Result<FeaRunResult, FeaRunError> {
    run_linear_static_with_options(model, backend, LinearStaticSolveOptions::default())
}

pub fn run_linear_static_with_options(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: LinearStaticSolveOptions,
) -> Result<FeaRunResult, FeaRunError> {
    emit_phase(
        "fea.run_linear_static",
        FeaProgressPhase::RegionResolution,
        FeaProgressStatus::Started,
        "validating FEA model",
        Some(0),
        Some(5),
    );
    check_cancelled("fea.run_linear_static")?;
    validate_model(model).map_err(|err| FeaRunError::InvalidModel(err.to_string()))?;
    emit_phase(
        "fea.run_linear_static",
        FeaProgressPhase::RegionResolution,
        FeaProgressStatus::Completed,
        "FEA model validation complete",
        Some(1),
        Some(5),
    );
    check_cancelled("fea.run_linear_static")?;

    emit_phase(
        "fea.run_linear_static",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Started,
        "assembling linear system",
        Some(1),
        Some(5),
    );
    let summary = assemble_linear_system(
        model,
        options.prep_context,
        options.thermo_mechanical_context,
        options.electro_thermal_context,
    );
    emit_phase(
        "fea.run_linear_static",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Completed,
        "linear system assembly complete",
        Some(2),
        Some(5),
    );
    check_cancelled("fea.run_linear_static")?;

    emit_phase(
        "fea.run_linear_static",
        FeaProgressPhase::Solve,
        FeaProgressStatus::Started,
        "solving linear system",
        Some(2),
        Some(5),
    );
    let algebra_backend = build_backend(options.algebra_backend_kind);
    let solve_result = solve_linear_system(
        &summary,
        options.preconditioner_kind,
        options.algebra_backend_kind,
        algebra_backend.as_ref(),
    );
    emit_phase(
        "fea.run_linear_static",
        FeaProgressPhase::Solve,
        FeaProgressStatus::Completed,
        "linear solve complete",
        Some(3),
        Some(5),
    );
    check_cancelled("fea.run_linear_static")?;

    emit_phase(
        "fea.run_linear_static",
        FeaProgressPhase::Postprocess,
        FeaProgressStatus::Started,
        "recovering result fields",
        Some(3),
        Some(5),
    );
    let fields = recover_result_fields(&summary, &solve_result);
    emit_phase(
        "fea.run_linear_static",
        FeaProgressPhase::Postprocess,
        FeaProgressStatus::Completed,
        "result field recovery complete",
        Some(4),
        Some(5),
    );
    check_cancelled("fea.run_linear_static")?;
    let solver_device_apply_k_ratio = if solve_result.device_apply_k_attempt_count == 0 {
        0.0
    } else {
        solve_result.device_apply_k_count as f64 / solve_result.device_apply_k_attempt_count as f64
    };

    let mut diagnostics = vec![FeaDiagnostic {
        code: "FEA_CONVERGENCE".to_string(),
        severity: if solve_result.converged {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "iterations={} residual_norm={} converged={} solver_method={} preconditioner={}",
            solve_result.iterations,
            solve_result.residual_norm,
            solve_result.converged,
            solve_result.solver_method,
            solve_result.preconditioner,
        ),
    }];
    extend_common_run_diagnostics(
        &mut diagnostics,
        CommonRunDiagnosticInputs {
            model,
            summary: &summary,
            prep_context: options.prep_context,
            iteration_metric: solve_result.iterations as f64,
            residual_metric: solve_result.residual_norm,
            requested_preconditioner: options.preconditioner_kind.as_str(),
            effective_preconditioner: &solve_result.preconditioner,
        },
    );
    diagnostics.extend(solve_result.diagnostics);

    emit_phase(
        "fea.run_linear_static",
        FeaProgressPhase::Complete,
        FeaProgressStatus::Completed,
        "FEA linear static run complete",
        Some(5),
        Some(5),
    );

    Ok(FeaRunResult {
        backend,
        solver_backend: solve_result.solver_backend,
        solver_device_apply_k_ratio,
        solver_method: solve_result.solver_method,
        preconditioner: solve_result.preconditioner,
        solver_host_sync_count: solve_result.host_sync_count,
        diagnostics,
        fields,
    })
}
