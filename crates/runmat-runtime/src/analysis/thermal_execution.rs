//! Thermal numerical execution and runtime result admission.
use std::collections::BTreeMap;

use runmat_analysis_core::{AnalysisModel, AnalysisProcedure, PhysicsFamily};
use runmat_analysis_fea::{
    run_thermal_with_options, ComputeBackend, ExecutionBackend, ThermalProcedureControl,
    ThermalSolveOptions,
};

use crate::operations::{
    operation_error, OperationContext, OperationEnvelope, OperationErrorEnvelope,
    OperationErrorSeverity, OperationErrorSpec, OperationErrorType,
};

use super::{
    classify_run_status, contracts, diagnostic_metric, install_fea_solver_context,
    map_fea_run_error, publish_standalone_run, render_topology_from_solver_mesh,
    resolve_solver_mesh_artifact, storage, validate_thermal_run_model, AnalysisOperation,
    AnalysisRunResult, AnalysisThermalRunOptions, FallbackEvent, QualityGate, QualityPolicy,
    QualityReason, QualityReasonCode, RunProvenance, ThermalResultsData,
};

pub fn analysis_run_thermal_with_options_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: AnalysisThermalRunOptions,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    publish_standalone_run(
        execute_thermal_with_options(model, backend, options, context)?,
        "RM.FEA.RUN_THERMAL.ARTIFACT_STORE_FAILED",
    )
}

pub(super) fn execute_thermal_with_options(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: AnalysisThermalRunOptions,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    let _solver_context = install_fea_solver_context();
    if !options.residual_warn_threshold.is_finite() || options.residual_warn_threshold <= 0.0 {
        return Err(operation_error(
            AnalysisOperation::Thermal.name(),
            AnalysisOperation::Thermal.version(),
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_THERMAL.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_thermal requires a finite positive residual control",
            BTreeMap::from([
                ("snapshot_count".into(), options.snapshot_count.to_string()),
                (
                    "residual_warn_threshold".into(),
                    options.residual_warn_threshold.to_string(),
                ),
            ]),
        ));
    }
    let thermal_steps = model
        .sequence
        .steps
        .iter()
        .filter(|step| {
            step.family == PhysicsFamily::Thermal
                && matches!(
                    step.procedure,
                    AnalysisProcedure::Static | AnalysisProcedure::TransientResponse
                )
        })
        .collect::<Vec<_>>();
    let [thermal_step] = thermal_steps.as_slice() else {
        return Err(operation_error(
            AnalysisOperation::Thermal.name(),
            AnalysisOperation::Thermal.version(),
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_THERMAL.INVALID_MODEL",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "FEA model must include exactly one supported thermal step for fea.run_thermal",
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                ("geometry_id".to_string(), model.geometry_id.clone()),
            ]),
        ));
    };
    let thermal_procedure = thermal_step.procedure;
    if thermal_procedure == AnalysisProcedure::TransientResponse && options.snapshot_count < 2 {
        return Err(operation_error(
            AnalysisOperation::Thermal.name(),
            AnalysisOperation::Thermal.version(),
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_THERMAL.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "transient fea.run_thermal requires at least two snapshots",
            BTreeMap::from([("snapshot_count".into(), options.snapshot_count.to_string())]),
        ));
    }
    validate_thermal_run_model(model, &context)?;

    if model.thermal.is_none() {
        return Err(operation_error(
            AnalysisOperation::Thermal.name(),
            AnalysisOperation::Thermal.version(),
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_THERMAL.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_thermal requires model.thermal to be configured",
            BTreeMap::new(),
        ));
    }

    let solver_mesh = resolve_solver_mesh_artifact(
        model,
        options.solver_mesh_artifact_path.as_deref(),
        AnalysisOperation::Thermal.name(),
        AnalysisOperation::Thermal.version(),
        "RM.FEA.RUN_THERMAL.SOLVER_MESH_INVALID",
        &context,
    )?;
    let Some(solver_mesh) = solver_mesh else {
        return Err(operation_error(
            AnalysisOperation::Thermal.name(),
            AnalysisOperation::Thermal.version(),
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_THERMAL.SOLVER_MESH_REQUIRED",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_thermal requires a canonical solver mesh",
            BTreeMap::new(),
        ));
    };
    let thermal_run = run_thermal_with_options(
        model,
        backend,
        ThermalSolveOptions {
            procedure: match thermal_procedure {
                AnalysisProcedure::Static => ThermalProcedureControl::Steady,
                AnalysisProcedure::TransientResponse => ThermalProcedureControl::BackwardEuler {
                    snapshot_count: options.snapshot_count,
                },
                _ => unreachable!("thermal procedure was validated before execution"),
            },
            residual_target: options.residual_warn_threshold,
            solver_mesh: Some(solver_mesh.clone()),
            initial_fields: Vec::new(),
        },
    )
    .map_err(|err| map_fea_run_error(AnalysisOperation::Thermal, model, &context, err))?;

    let run = thermal_run.run;
    let mut fallback_events = Vec::new();
    if backend == ComputeBackend::Gpu && run.execution_backend != ExecutionBackend::Accelerator {
        fallback_events.push(FallbackEvent::solver_backend());
    }
    let solver_convergence = if diagnostic_metric(
        &run.diagnostics,
        "FEA_THERMAL_STABILITY",
        "max_relative_residual_norm",
    )
    .unwrap_or(0.0)
        <= options.residual_warn_threshold
    {
        QualityGate::Pass
    } else {
        QualityGate::Warn
    };
    let result_quality = if thermal_run.temperature_snapshots.is_empty() {
        QualityGate::Fail
    } else {
        solver_convergence
    };
    let mut quality_reasons = Vec::new();
    if result_quality == QualityGate::Warn {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ThermalResidualExceeded,
            detail: format!(
                "thermal residual exceeds threshold {}",
                options.residual_warn_threshold
            ),
        });
    }
    let thermal_conductivity_spread_ratio = diagnostic_metric(
        &run.diagnostics,
        "FEA_THERMAL_CONSTITUTIVE",
        "conductivity_spread_ratio",
    );
    let thermal_heat_capacity_spread_ratio = diagnostic_metric(
        &run.diagnostics,
        "FEA_THERMAL_CONSTITUTIVE",
        "heat_capacity_spread_ratio",
    );
    if thermal_conductivity_spread_ratio.unwrap_or(1.0) > 2.5
        || thermal_heat_capacity_spread_ratio.unwrap_or(1.0) > 2.5
    {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ThermalConstitutiveSpreadHigh,
            detail: format!(
                "thermal constitutive spread exceeds limit: conductivity_spread_ratio={} heat_capacity_spread_ratio={}",
                thermal_conductivity_spread_ratio.unwrap_or(1.0),
                thermal_heat_capacity_spread_ratio.unwrap_or(1.0)
            ),
        });
    }

    let numerically_acceptable = match options.quality_policy {
        QualityPolicy::Strict => {
            solver_convergence == QualityGate::Pass
                && result_quality == QualityGate::Pass
                && quality_reasons.is_empty()
        }
        QualityPolicy::Balanced => {
            result_quality != QualityGate::Fail
                && !quality_reasons
                    .iter()
                    .any(|reason| reason.code == QualityReasonCode::ThermalConstitutiveSpreadHigh)
        }
        QualityPolicy::Exploratory => true,
    };
    let run_status = classify_run_status(result_quality, numerically_acceptable);

    let solver_backend = run.solver_backend;
    let execution_backend = run.execution_backend;
    let solver_device_apply_k_ratio = run.solver_device_apply_k_ratio;
    let solver_host_sync_count = run.solver_host_sync_count;
    let solver_method = run.solver_method;
    let selected_preconditioner = run.preconditioner;

    let result = AnalysisRunResult {
        observation_scalars: Vec::new(),
        run_id: storage::next_run_id(),
        run,
        render_topology: render_topology_from_solver_mesh(Some(&solver_mesh)),
        modal_results: None,
        buckling_results: None,
        thermal_results: Some(ThermalResultsData {
            thermal_payload_version: "thermal_results/v1".to_string(),
            time_points_s: thermal_run.time_points_s,
            temperature_snapshots: thermal_run.temperature_snapshots,
            temperature_gradient_snapshots: thermal_run.temperature_gradient_snapshots,
            heat_flux_snapshots: thermal_run.heat_flux_snapshots,
            energy_storage_rate_snapshots: thermal_run.energy_storage_rate_snapshots,
            interface_node_ids: thermal_run.interface_node_ids,
            interface_temperature_snapshots: thermal_run.interface_temperature_snapshots,
            residual_norms: thermal_run.residual_norms,
            reference_temperature_k: runmat_types::physical::Temperature::new(
                thermal_run.reference_temperature_k.canonical_value(),
            ),
            initial_state: thermal_run.initial_state,
        }),
        transient_results: None,
        nonlinear_results: None,
        model_validity: QualityGate::Pass,
        solver_convergence,
        result_quality,
        run_status,
        numerically_acceptable,
        quality_reasons,
        provenance: RunProvenance {
            requested_backend: backend,
            execution_backend,
            solver_backend,
            solver_device_apply_k_ratio,
            solver_host_sync_count,
            precision_mode: contracts::format_precision_mode(options.precision_mode),
            deterministic_mode: options.deterministic_mode,
            solver_method,
            preconditioner: selected_preconditioner,
            quality_policy: contracts::format_quality_policy(options.quality_policy),
            fallback_events,
            effective_memory_ceiling_bytes: None,
            static_solve_path: None,
        },
    };

    Ok(OperationEnvelope::new(
        AnalysisOperation::Thermal.name(),
        AnalysisOperation::Thermal.version(),
        &context,
        result,
    ))
}
