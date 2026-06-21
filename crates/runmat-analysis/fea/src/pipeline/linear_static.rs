use runmat_analysis_core::{validate_model, AnalysisField, AnalysisModel};

use crate::{
    assembly::assemble_linear_system,
    contracts::{
        ComputeBackend, FeaRunError, FeaRunResult, LinearStaticSolveOptions,
        FEA_FIELD_STRUCTURAL_DISPLACEMENT, FEA_FIELD_STRUCTURAL_EQUATION_SCALE,
        FEA_FIELD_STRUCTURAL_RESIDUAL_NORM, FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY,
        FEA_FIELD_STRUCTURAL_VON_MISES,
    },
    diagnostics::{
        builders::{extend_common_run_diagnostics, CommonRunDiagnosticInputs},
        FeaDiagnostic, FeaDiagnosticSeverity,
    },
    pipeline::electro_thermal::recover_electro_thermal_fields,
    pipeline::thermo_mechanical::recover_thermo_mechanical_snapshots,
    post::fields::{recover_result_fields, structural_field_recovery_metrics},
    progress::{check_cancelled, emit_phase, FeaProgressPhase, FeaProgressStatus},
    solve::{backend::build_backend, linear::solve_linear_system},
};

const VECTOR_COMPONENT_COUNT: usize = 3;

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
    let mut fields = recover_result_fields(&summary, &solve_result);
    let thermo_mechanical_fields = recover_thermo_mechanical_snapshots(
        summary.thermo_mechanical.as_ref(),
        &[1.0],
        &field_snapshot(&fields, FEA_FIELD_STRUCTURAL_DISPLACEMENT),
        &field_snapshot(&fields, FEA_FIELD_STRUCTURAL_VON_MISES),
        &[solve_result.residual_norm],
        element_count_for_dofs(summary.dof_count),
    );
    fields.extend(thermo_mechanical_fields.temperature_snapshots);
    fields.extend(thermo_mechanical_fields.thermal_strain_snapshots);
    fields.extend(thermo_mechanical_fields.thermal_stress_snapshots);
    fields.extend(thermo_mechanical_fields.displacement_snapshots);
    fields.extend(thermo_mechanical_fields.von_mises_snapshots);
    fields.extend(thermo_mechanical_fields.coupling_residual_snapshots);
    let thermo_mechanical_diagnostics = thermo_mechanical_fields.diagnostics;
    let electro_thermal_fields = recover_electro_thermal_fields(
        summary.electro_thermal.as_ref(),
        &[1.0],
        &[solve_result.residual_norm],
        summary.dof_count,
    );
    let electro_thermal_diagnostics = electro_thermal_fields.diagnostics;
    fields.extend(electro_thermal_fields.static_fields);
    fields.extend(electro_thermal_fields.temperature_snapshots);
    fields.extend(electro_thermal_fields.thermal_residual_snapshots);
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
    if let (Some(residual_norm), Some(equation_scale)) = (
        scalar_field_value(&fields, FEA_FIELD_STRUCTURAL_RESIDUAL_NORM),
        scalar_field_value(&fields, FEA_FIELD_STRUCTURAL_EQUATION_SCALE),
    ) {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_STRUCTURAL_RESIDUAL".to_string(),
            severity: if residual_norm <= 1.0e-6 {
                FeaDiagnosticSeverity::Info
            } else {
                FeaDiagnosticSeverity::Warning
            },
            message: format!(
                "normalized_residual_norm={} equation_scale={}",
                residual_norm, equation_scale
            ),
        });
    }
    if let Some(total_strain_energy) =
        scalar_field_value(&fields, FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)
    {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_STRUCTURAL_ENERGY".to_string(),
            severity: if total_strain_energy.is_finite() && total_strain_energy >= 0.0 {
                FeaDiagnosticSeverity::Info
            } else {
                FeaDiagnosticSeverity::Warning
            },
            message: format!("total_strain_energy={total_strain_energy}"),
        });
        diagnostics.push(structural_linear_known_answer_diagnostic(
            total_strain_energy,
            &summary.operator.rhs,
            &solve_result.solution,
        ));
    }
    let recovery_metrics = structural_field_recovery_metrics(&summary, &solve_result.solution);
    diagnostics.push(FeaDiagnostic {
        code: "FEA_STRUCTURAL_FIELD_RECOVERY".to_string(),
        severity: if recovery_metrics.active_stiffness_edge_count > 0
            && recovery_metrics.recovery_element_count > 0
        {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "basis=operator_connectivity active_stiffness_edge_count={} constrained_edge_count={} recovery_element_count={} max_edge_displacement_jump={} mean_edge_stiffness_ratio={}",
            recovery_metrics.active_stiffness_edge_count,
            recovery_metrics.constrained_edge_count,
            recovery_metrics.recovery_element_count,
            recovery_metrics.max_edge_displacement_jump,
            recovery_metrics.mean_edge_stiffness_ratio
        ),
    });
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
    diagnostics.extend(thermo_mechanical_diagnostics);
    diagnostics.extend(electro_thermal_diagnostics);
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

fn structural_linear_known_answer_diagnostic(
    total_strain_energy: f64,
    rhs: &[f64],
    displacement: &[f64],
) -> FeaDiagnostic {
    let external_work = rhs
        .iter()
        .zip(displacement.iter())
        .map(|(force, displacement)| force * displacement)
        .sum::<f64>();
    let work_energy_ratio = if external_work.abs() > 1.0e-12 {
        (2.0 * total_strain_energy) / external_work
    } else if total_strain_energy.abs() <= 1.0e-12 {
        1.0
    } else {
        0.0
    };
    let work_energy_residual_ratio = (work_energy_ratio - 1.0).abs();
    let known_answer_coverage_ratio = if rhs.len() == displacement.len()
        && !rhs.is_empty()
        && total_strain_energy.is_finite()
        && external_work.is_finite()
    {
        1.0
    } else {
        0.0
    };
    let severity = if known_answer_coverage_ratio >= 1.0
        && total_strain_energy >= 0.0
        && work_energy_residual_ratio <= 1.0e-8
    {
        FeaDiagnosticSeverity::Info
    } else {
        FeaDiagnosticSeverity::Warning
    };

    FeaDiagnostic {
        code: "FEA_STRUCTURAL_LINEAR_KNOWN_ANSWER".to_string(),
        severity,
        message: format!(
            "identity=linear_work_energy external_work={} total_strain_energy={} work_energy_ratio={} work_energy_residual_ratio={} known_answer_coverage_ratio={}",
            external_work,
            total_strain_energy,
            work_energy_ratio,
            work_energy_residual_ratio,
            known_answer_coverage_ratio
        ),
    }
}

fn scalar_field_value(
    fields: &[runmat_analysis_core::AnalysisField],
    field_id: &str,
) -> Option<f64> {
    fields
        .iter()
        .find(|field| field.field_id == field_id)
        .and_then(runmat_analysis_core::AnalysisField::as_host_f64)
        .and_then(|values| values.first().copied())
}

fn field_snapshot(fields: &[AnalysisField], field_id: &str) -> Vec<AnalysisField> {
    fields
        .iter()
        .find(|field| field.field_id == field_id)
        .cloned()
        .into_iter()
        .collect()
}

fn element_count_for_dofs(dof_count: usize) -> usize {
    dof_count.div_ceil(VECTOR_COMPONENT_COUNT).max(1)
}
