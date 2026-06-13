use runmat_analysis_core::{validate_model, AnalysisField, AnalysisModel};

use crate::{
    assembly::assemble_linear_system,
    contracts::{
        fea_transient_displacement_field_id, ComputeBackend, FeaRunError, FeaRunResult,
        FeaTransientRunResult, FEA_FIELD_STRUCTURAL_DISPLACEMENT, FEA_FIELD_STRUCTURAL_VON_MISES,
    },
    diagnostics::builders::{extend_common_run_diagnostics, CommonRunDiagnosticInputs},
    progress::{check_cancelled, emit_phase, FeaProgressPhase, FeaProgressStatus},
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
    emit_phase(
        "fea.run_transient",
        FeaProgressPhase::RegionResolution,
        FeaProgressStatus::Started,
        "validating transient FEA model",
        Some(0),
        Some(5),
    );
    check_cancelled("fea.run_transient")?;
    validate_model(model).map_err(|err| FeaRunError::InvalidModel(err.to_string()))?;
    emit_phase(
        "fea.run_transient",
        FeaProgressPhase::RegionResolution,
        FeaProgressStatus::Completed,
        "transient model validation complete",
        Some(1),
        Some(5),
    );
    check_cancelled("fea.run_transient")?;
    let prep_context = options.prep_context;
    let thermo_context = options.thermo_mechanical_context.clone();
    let electro_context = options.electro_thermal_context.clone();

    emit_phase(
        "fea.run_transient",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Started,
        "assembling transient system",
        Some(1),
        Some(5),
    );
    let summary = assemble_linear_system(model, prep_context, thermo_context, electro_context);
    emit_phase(
        "fea.run_transient",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Completed,
        "transient system assembly complete",
        Some(2),
        Some(5),
    );
    check_cancelled("fea.run_transient")?;

    emit_phase(
        "fea.run_transient",
        FeaProgressPhase::Solve,
        FeaProgressStatus::Started,
        "solving transient steps",
        Some(2),
        Some(5),
    );
    let transient = solve_transient_system(&summary, options.clone(), backend);
    emit_phase(
        "fea.run_transient",
        FeaProgressPhase::Solve,
        FeaProgressStatus::Completed,
        "transient solve complete",
        Some(3),
        Some(5),
    );
    check_cancelled("fea.run_transient")?;

    emit_phase(
        "fea.run_transient",
        FeaProgressPhase::Postprocess,
        FeaProgressStatus::Started,
        "recovering transient fields",
        Some(3),
        Some(5),
    );
    let mut diagnostics = transient.diagnostics.clone();
    extend_common_run_diagnostics(
        &mut diagnostics,
        CommonRunDiagnosticInputs {
            model,
            summary: &summary,
            prep_context,
            iteration_metric: transient.converged_steps as f64,
            residual_metric: transient
                .residual_norms
                .iter()
                .copied()
                .fold(0.0_f64, f64::max),
            requested_preconditioner: "auto",
            effective_preconditioner: &transient.preconditioner,
        },
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
        fields: vec![
            AnalysisField::host_f64(
                FEA_FIELD_STRUCTURAL_DISPLACEMENT,
                vec![displacement.len()],
                displacement,
            ),
            AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_VON_MISES, vec![1], vec![von_mises]),
        ],
    };

    let displacement_snapshots = transient
        .displacement_snapshots
        .into_iter()
        .enumerate()
        .map(|(index, snapshot)| {
            AnalysisField::host_f64(
                fea_transient_displacement_field_id(index),
                vec![snapshot.len()],
                snapshot,
            )
        })
        .collect();

    emit_phase(
        "fea.run_transient",
        FeaProgressPhase::Postprocess,
        FeaProgressStatus::Completed,
        "transient result field recovery complete",
        Some(4),
        Some(5),
    );
    check_cancelled("fea.run_transient")?;
    emit_phase(
        "fea.run_transient",
        FeaProgressPhase::Complete,
        FeaProgressStatus::Completed,
        "FEA transient run complete",
        Some(5),
        Some(5),
    );

    Ok(FeaTransientRunResult {
        run,
        time_points_s: transient.time_points_s,
        displacement_snapshots,
        residual_norms: transient.residual_norms,
    })
}
