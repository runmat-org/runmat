use std::collections::BTreeMap;

use runmat_analysis_core::{
    validate_model_against_geometry, AnalysisModel, AnalysisModelId, AnalysisStep,
    AnalysisStepKind, AnalysisValidationError, BoundaryCondition, BoundaryConditionKind,
    EvidenceConfidence, LoadCase, LoadKind, MaterialAssignment, MaterialModel, ReferenceFrame,
};
use runmat_analysis_fea::{
    run_linear_static_with_options, run_modal_with_options, run_nonlinear_with_options,
    run_transient_with_options, ComputeBackend, LinearStaticSolveOptions, ModalSolveOptions,
};
use runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind;
use runmat_analysis_fea::solve::preconditioner::SpdPreconditionerKind;
use runmat_geometry_core::{GeometryAsset, MaterialEvidenceConfidence, UnitSystem};

use crate::operations::{
    operation_error, OperationContext, OperationEnvelope, OperationErrorEnvelope,
    OperationErrorSeverity, OperationErrorSpec, OperationErrorType,
};

mod contracts;
mod promotion;
pub mod storage;

pub use contracts::{
    AnalysisCreateModelIntentSpec, AnalysisCreateModelProfile, AnalysisResultsData,
    AnalysisModalRunOptions, AnalysisNonlinearRunOptions, AnalysisResultsQuery,
    AnalysisResultsSummary, AnalysisRunOptions, AnalysisRunResult, AnalysisTransientRunOptions,
    AnalysisValidateResult, ModalFrequencyBasis, ModalFrequencyUnits, ModalResultsData,
    NonlinearMethod, NonlinearResultsData, PrecisionMode, PreconditionerMode, QualityGate,
    QualityPolicy, QualityReason, QualityReasonCode, RunProvenance, RunStatus,
    TransientIntegrationMethod, TransientResultsData,
};

const ANALYSIS_CREATE_MODEL_OPERATION: &str = "analysis.create_model";
const ANALYSIS_CREATE_MODEL_OP_VERSION: &str = "analysis.create_model/v1";
const ANALYSIS_VALIDATE_OPERATION: &str = "analysis.validate";
const ANALYSIS_VALIDATE_OP_VERSION: &str = "analysis.validate/v1";
const ANALYSIS_RUN_OPERATION: &str = "analysis.run_linear_static";
const ANALYSIS_RUN_OP_VERSION: &str = "analysis.run_linear_static/v1";
const ANALYSIS_RUN_MODAL_OPERATION: &str = "analysis.run_modal";
const ANALYSIS_RUN_MODAL_OP_VERSION: &str = "analysis.run_modal/v1";
const ANALYSIS_RUN_TRANSIENT_OPERATION: &str = "analysis.run_transient";
const ANALYSIS_RUN_TRANSIENT_OP_VERSION: &str = "analysis.run_transient/v1";
const ANALYSIS_RUN_NONLINEAR_OPERATION: &str = "analysis.run_nonlinear";
const ANALYSIS_RUN_NONLINEAR_OP_VERSION: &str = "analysis.run_nonlinear/v1";
const ANALYSIS_RESULTS_OPERATION: &str = "analysis.results";
const ANALYSIS_RESULTS_OP_VERSION: &str = "analysis.results/v1";
const TRANSIENT_RESIDUAL_WARN_THRESHOLD: f64 = 1.0e-4;

pub fn analysis_create_model_op(
    geometry: &GeometryAsset,
    intent: AnalysisCreateModelIntentSpec,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisModel>, OperationErrorEnvelope> {
    if intent.model_id.trim().is_empty() {
        return Err(operation_error(
            ANALYSIS_CREATE_MODEL_OPERATION,
            ANALYSIS_CREATE_MODEL_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "ANALYSIS_CREATE_MODEL_INVALID_INTENT",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "analysis model intent requires a non-empty model_id",
            BTreeMap::from([("geometry_id".to_string(), geometry.geometry_id.clone())]),
        ));
    }

    if geometry.meshes.is_empty() {
        return Err(operation_error(
            ANALYSIS_CREATE_MODEL_OPERATION,
            ANALYSIS_CREATE_MODEL_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "ANALYSIS_CREATE_MODEL_GEOMETRY_EMPTY",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "geometry must contain at least one mesh to create an analysis model",
            BTreeMap::from([("geometry_id".to_string(), geometry.geometry_id.clone())]),
        ));
    }

    if geometry.units == UnitSystem::Unspecified {
        return Err(operation_error(
            ANALYSIS_CREATE_MODEL_OPERATION,
            ANALYSIS_CREATE_MODEL_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "ANALYSIS_CREATE_MODEL_UNIT_UNSPECIFIED",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "geometry units must be specified before creating an analysis model",
            BTreeMap::from([("geometry_id".to_string(), geometry.geometry_id.clone())]),
        ));
    }

    let fixed_region_id = select_fixed_region_id(geometry)
        .or_else(|| geometry.regions.first().map(|region| region.region_id.clone()))
        .unwrap_or_else(|| "region_default".to_string());
    let load_region_id = select_load_region_id(geometry)
        .or_else(|| geometry.regions.last().map(|region| region.region_id.clone()))
        .unwrap_or_else(|| fixed_region_id.clone());

    let inferred_materials = infer_material_models(geometry);
    let inferred_assignments = infer_material_assignments(geometry, &inferred_materials);

    let (default_bc, default_load, default_step) = match intent.profile {
        AnalysisCreateModelProfile::LinearStaticStructural => (
            BoundaryCondition {
                bc_id: "bc_default_fixed".to_string(),
                region_id: fixed_region_id,
                kind: BoundaryConditionKind::Fixed,
            },
            LoadCase {
                load_id: "load_default_force".to_string(),
                region_id: load_region_id,
                kind: LoadKind::Force {
                    fx: 0.0,
                    fy: -1000.0,
                    fz: 0.0,
                },
            },
            AnalysisStep {
                step_id: "step_default_static".to_string(),
                kind: AnalysisStepKind::Static,
            },
        ),
        AnalysisCreateModelProfile::ModalStructural => (
            BoundaryCondition {
                bc_id: "bc_default_fixed".to_string(),
                region_id: fixed_region_id,
                kind: BoundaryConditionKind::Fixed,
            },
            LoadCase {
                load_id: "load_default_modal_seed".to_string(),
                region_id: load_region_id,
                kind: LoadKind::BodyForce {
                    gx: 0.0,
                    gy: 0.0,
                    gz: 0.0,
                },
            },
            AnalysisStep {
                step_id: "step_default_modal".to_string(),
                kind: AnalysisStepKind::Modal,
            },
        ),
        AnalysisCreateModelProfile::TransientStructural => (
            BoundaryCondition {
                bc_id: "bc_default_fixed".to_string(),
                region_id: fixed_region_id,
                kind: BoundaryConditionKind::Fixed,
            },
            LoadCase {
                load_id: "load_default_transient_force".to_string(),
                region_id: load_region_id,
                kind: LoadKind::Force {
                    fx: 0.0,
                    fy: -500.0,
                    fz: 0.0,
                },
            },
            AnalysisStep {
                step_id: "step_default_transient".to_string(),
                kind: AnalysisStepKind::Transient,
            },
        ),
        AnalysisCreateModelProfile::NonlinearStructural => (
            BoundaryCondition {
                bc_id: "bc_default_fixed".to_string(),
                region_id: fixed_region_id,
                kind: BoundaryConditionKind::Fixed,
            },
            LoadCase {
                load_id: "load_default_nonlinear_force".to_string(),
                region_id: load_region_id,
                kind: LoadKind::Force {
                    fx: 0.0,
                    fy: -750.0,
                    fz: 0.0,
                },
            },
            AnalysisStep {
                step_id: "step_default_nonlinear".to_string(),
                kind: AnalysisStepKind::Nonlinear,
            },
        ),
    };

    let model = AnalysisModel {
        model_id: AnalysisModelId(intent.model_id),
        geometry_id: geometry.geometry_id.clone(),
        geometry_revision: geometry.revision,
        units: geometry.units,
        frame: ReferenceFrame::Global,
        materials: inferred_materials,
        material_assignments: inferred_assignments,
        boundary_conditions: vec![default_bc],
        loads: vec![default_load],
        steps: vec![default_step],
    };

    validate_model_against_geometry(&model, geometry.units, &ReferenceFrame::Global).map_err(
        |error| {
            operation_error(
                ANALYSIS_CREATE_MODEL_OPERATION,
                ANALYSIS_CREATE_MODEL_OP_VERSION,
                &context,
                OperationErrorSpec {
                    error_code: "ANALYSIS_CREATE_MODEL_INVALID",
                    error_type: OperationErrorType::Validation,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                format!("created analysis model failed validation: {error:?}"),
                BTreeMap::from([
                    ("analysis_model_id".to_string(), model.model_id.0.clone()),
                    ("geometry_id".to_string(), geometry.geometry_id.clone()),
                ]),
            )
        },
    )?;

    Ok(OperationEnvelope::new(
        ANALYSIS_CREATE_MODEL_OPERATION,
        ANALYSIS_CREATE_MODEL_OP_VERSION,
        &context,
        model,
    ))
}

pub fn analysis_validate(
    model: &AnalysisModel,
    geometry_units: UnitSystem,
    geometry_frame: &ReferenceFrame,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisValidateResult>, OperationErrorEnvelope> {
    validate_model_against_geometry(model, geometry_units, geometry_frame)
        .map_err(|err| map_validate_error(err, model, &context))?;

    Ok(OperationEnvelope::new(
        ANALYSIS_VALIDATE_OPERATION,
        ANALYSIS_VALIDATE_OP_VERSION,
        &context,
        AnalysisValidateResult { valid: true },
    ))
}

pub fn analysis_run_linear_static_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    analysis_run_linear_static_with_options(model, backend, AnalysisRunOptions::default(), context)
}

pub fn analysis_run_modal_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    analysis_run_modal_with_options_op(model, backend, AnalysisModalRunOptions::default(), context)
}

pub fn analysis_run_modal_with_options_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: AnalysisModalRunOptions,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    let has_modal_step = model
        .steps
        .iter()
        .any(|step| step.kind == AnalysisStepKind::Modal);
    if !has_modal_step {
        return Err(operation_error(
            ANALYSIS_RUN_MODAL_OPERATION,
            ANALYSIS_RUN_MODAL_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "ANALYSIS_RUN_MODAL_INVALID_MODEL",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "analysis model must include at least one modal step for analysis.run_modal",
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                ("geometry_id".to_string(), model.geometry_id.clone()),
            ]),
        ));
    }

    if options.mode_count == 0 {
        return Err(operation_error(
            ANALYSIS_RUN_MODAL_OPERATION,
            ANALYSIS_RUN_MODAL_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "ANALYSIS_RUN_MODAL_INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "analysis.run_modal options require mode_count greater than zero",
            BTreeMap::from([("mode_count".to_string(), options.mode_count.to_string())]),
        ));
    }

    let modal_run = run_modal_with_options(
        model,
        backend,
        ModalSolveOptions {
            mode_count: options.mode_count,
        },
    )
    .map_err(
        |err| {
            operation_error(
                ANALYSIS_RUN_MODAL_OPERATION,
                ANALYSIS_RUN_MODAL_OP_VERSION,
                &context,
                OperationErrorSpec {
                    error_code: "SOLVER_MODEL_INVALID",
                    error_type: OperationErrorType::Validation,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                err.to_string(),
                BTreeMap::from([
                    ("analysis_model_id".to_string(), model.model_id.0.clone()),
                    ("geometry_id".to_string(), model.geometry_id.clone()),
                ]),
            )
        },
    )?;

    let mut run = modal_run.run;
    let mut fallback_events = Vec::new();
    promotion::promote_run_fields_to_device_refs(&mut run, &mut fallback_events);
    if backend == ComputeBackend::Gpu && run.solver_backend != "runtime_tensor" {
        fallback_events.push(
            "SOLVER_BACKEND_FALLBACK:requested=runtime_tensor:using=cpu_reference".to_string(),
        );
    }
    let solver_convergence = if run.diagnostics.iter().any(|item| {
        item.code == "FEA_MODAL_CONVERGENCE"
            && item.severity == runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info
    }) {
        QualityGate::Pass
    } else {
        QualityGate::Warn
    };
    let result_quality = if modal_run.eigenvalues_hz.is_empty() || modal_run.mode_shapes.is_empty()
    {
        QualityGate::Fail
    } else if modal_run
        .residual_norms
        .iter()
        .copied()
        .fold(0.0_f64, f64::max)
        > options.residual_warn_threshold
    {
        QualityGate::Warn
    } else {
        QualityGate::Pass
    };
    let modal_orthogonality_warn = run.diagnostics.iter().any(|item| {
        item.code == "FEA_MODAL_ORTHOGONALITY"
            && item.severity == runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
    });
    let modal_separation_warn = run.diagnostics.iter().any(|item| {
        item.code == "FEA_MODAL_SEPARATION"
            && item.severity == runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
    });

    let mut quality_reasons = Vec::new();
    if solver_convergence == QualityGate::Warn {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::SolverNotConverged,
            detail: "modal solver convergence gate is warning".to_string(),
        });
    }
    if result_quality == QualityGate::Warn {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ModalResidualExceeded,
            detail: format!(
                "modal residual exceeds threshold {}",
                options.residual_warn_threshold
            ),
        });
    }
    if modal_orthogonality_warn {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ModalOrthogonalityExceeded,
            detail: "modal M-orthogonality off-diagonal threshold exceeded".to_string(),
        });
    }
    if modal_separation_warn {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ModalSeparationLow,
            detail: "modal frequency separation threshold is low".to_string(),
        });
    }
    if fallback_events
        .iter()
        .any(|event| event.starts_with("SOLVER_BACKEND_FALLBACK"))
    {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::SolverBackendFallback,
            detail: "solver backend fell back from runtime_tensor to cpu_reference".to_string(),
        });
    }
    if fallback_events.iter().any(|event| {
        event.starts_with("BACKEND_NO_PROVIDER") || event.starts_with("BACKEND_UPLOAD_FAILED")
    }) {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::FieldPromotionFallback,
            detail: "field promotion fell back to host-backed values".to_string(),
        });
    }

    let frequency_basis = if run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_MODAL_PLACEHOLDER")
    {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ModalPlaceholder,
            detail: "modal run path currently uses linear-static placeholder backend"
                .to_string(),
        });
        ModalFrequencyBasis::PlaceholderLinearStatic
    } else {
        ModalFrequencyBasis::NativeEigenSolve
    };

    let publishable = match options.quality_policy {
        QualityPolicy::Strict => {
            solver_convergence == QualityGate::Pass
                && result_quality == QualityGate::Pass
                && quality_reasons.is_empty()
        }
        QualityPolicy::Balanced => {
            solver_convergence == QualityGate::Pass
                && result_quality == QualityGate::Pass
                && !quality_reasons
                    .iter()
                    .any(|r| {
                        matches!(
                            r.code,
                            QualityReasonCode::ModalOrthogonalityExceeded
                                | QualityReasonCode::ModalSeparationLow
                        )
                    })
        }
        QualityPolicy::Exploratory => {
            solver_convergence != QualityGate::Fail && result_quality != QualityGate::Fail
        }
    };
    let run_status = if publishable {
        RunStatus::Publishable
    } else if result_quality == QualityGate::Fail {
        RunStatus::Rejected
    } else {
        RunStatus::Degraded
    };

    let solver_backend = run.solver_backend.clone();
    let solver_device_apply_k_ratio = run.solver_device_apply_k_ratio;
    let solver_host_sync_count = run.solver_host_sync_count;
    let solver_method = run.solver_method.clone();
    let selected_preconditioner = run.preconditioner.clone();

    let result = AnalysisRunResult {
        run_id: storage::next_run_id(),
        run,
        modal_results: Some(ModalResultsData {
            modal_payload_version: "modal_results/v1".to_string(),
            eigenvalues_hz: modal_run.eigenvalues_hz,
            mode_shapes: modal_run.mode_shapes,
            residual_norms: modal_run.residual_norms,
            mode_units: ModalFrequencyUnits::Hz,
            frequency_basis,
        }),
        transient_results: None,
        nonlinear_results: None,
        model_validity: QualityGate::Pass,
        solver_convergence,
        result_quality,
        run_status,
        publishable,
        quality_reasons,
        provenance: RunProvenance {
            backend,
            solver_backend,
            solver_device_apply_k_ratio,
            solver_host_sync_count,
            precision_mode: contracts::format_precision_mode(options.precision_mode),
            deterministic_mode: options.deterministic_mode,
            solver_method,
            preconditioner: selected_preconditioner,
            quality_policy: contracts::format_quality_policy(options.quality_policy),
            fallback_events,
        },
    };

    storage::persist_run_result(&result).map_err(|err| {
        operation_error(
            ANALYSIS_RUN_MODAL_OPERATION,
            ANALYSIS_RUN_MODAL_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "ANALYSIS_ARTIFACT_STORE_FAILED",
                error_type: OperationErrorType::Internal,
                retryable: true,
                severity: OperationErrorSeverity::Error,
            },
            format!("failed to persist analysis run artifact: {err}"),
            BTreeMap::from([("run_id".to_string(), result.run_id.clone())]),
        )
    })?;

    Ok(OperationEnvelope::new(
        ANALYSIS_RUN_MODAL_OPERATION,
        ANALYSIS_RUN_MODAL_OP_VERSION,
        &context,
        result,
    ))
}

pub fn analysis_run_transient_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    analysis_run_transient_with_options_op(
        model,
        backend,
        AnalysisTransientRunOptions::default(),
        context,
    )
}

pub fn analysis_run_transient_with_options_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: AnalysisTransientRunOptions,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    let has_transient_step = model
        .steps
        .iter()
        .any(|step| step.kind == AnalysisStepKind::Transient);
    if !has_transient_step {
        return Err(operation_error(
            ANALYSIS_RUN_TRANSIENT_OPERATION,
            ANALYSIS_RUN_TRANSIENT_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "ANALYSIS_RUN_TRANSIENT_INVALID_MODEL",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "analysis model must include at least one transient step for analysis.run_transient",
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                ("geometry_id".to_string(), model.geometry_id.clone()),
            ]),
        ));
    }

    let transient_run = run_transient_with_options(
        model,
        backend,
        runmat_analysis_fea::solve::transient::TransientSolveOptions {
            time_step_s: options.time_step_s,
            min_time_step_s: options.min_time_step_s,
            max_time_step_s: options.max_time_step_s,
            step_count: options.step_count,
            max_linear_iters: options.max_linear_iters,
            tolerance: options.tolerance,
            residual_target: options.residual_target,
            adaptive_time_step: options.adaptive_time_step,
            max_step_retries: options.max_step_retries,
            adapt_min_scale: options.adapt_min_scale,
            adapt_max_scale: options.adapt_max_scale,
            adapt_growth_exponent: options.adapt_growth_exponent,
            adapt_retry_growth_cap: options.adapt_retry_growth_cap,
            adapt_nonconverged_shrink: options.adapt_nonconverged_shrink,
            dt_bucket_rel_tolerance: options.dt_bucket_rel_tolerance,
        },
    )
    .map_err(|err| {
        operation_error(
            ANALYSIS_RUN_TRANSIENT_OPERATION,
            ANALYSIS_RUN_TRANSIENT_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "SOLVER_MODEL_INVALID",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            err.to_string(),
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                ("geometry_id".to_string(), model.geometry_id.clone()),
            ]),
        )
    })?;

    let mut run = transient_run.run;
    let mut fallback_events = Vec::new();
    promotion::promote_run_fields_to_device_refs(&mut run, &mut fallback_events);
    if backend == ComputeBackend::Gpu && run.solver_backend != "runtime_tensor" {
        fallback_events.push(
            "SOLVER_BACKEND_FALLBACK:requested=runtime_tensor:using=cpu_reference".to_string(),
        );
    }
    let solver_convergence = if run.diagnostics.iter().any(|item| {
        item.code == "FEA_TRANSIENT_CONVERGENCE"
            && item.severity == runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info
    }) {
        QualityGate::Pass
    } else {
        QualityGate::Warn
    };
    let result_quality = if transient_run.displacement_snapshots.is_empty()
        || transient_run.time_points_s.is_empty()
        || transient_run
            .residual_norms
            .iter()
            .any(|residual| !residual.is_finite())
    {
        QualityGate::Fail
    } else if transient_run
        .residual_norms
        .iter()
        .copied()
        .fold(0.0_f64, f64::max)
        > TRANSIENT_RESIDUAL_WARN_THRESHOLD
    {
        QualityGate::Warn
    } else {
        QualityGate::Pass
    };
    let transient_stability_warn = run.diagnostics.iter().any(|item| {
        item.code == "FEA_TRANSIENT_STABILITY"
            && item.severity == runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
    }) || run.diagnostics.iter().any(|item| {
        item.code == "FEA_TRANSIENT_ENERGY"
            && item.severity == runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
    });
    let transient_step_failure_warn = run.diagnostics.iter().any(|item| {
        item.code == "FEA_TRANSIENT_STEP_FAILURE"
            && item.severity == runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
    });

    let mut quality_reasons = Vec::new();
    if solver_convergence == QualityGate::Warn {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::SolverNotConverged,
            detail: "transient solver convergence gate is warning".to_string(),
        });
    }
    if result_quality == QualityGate::Warn {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::TransientResidualExceeded,
            detail: format!(
                "transient residual exceeds threshold {}",
                TRANSIENT_RESIDUAL_WARN_THRESHOLD
            ),
        });
    }
    if transient_stability_warn {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::TransientStabilityExceeded,
            detail: "transient stability diagnostic exceeded threshold".to_string(),
        });
    }
    if transient_step_failure_warn {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::TransientStepFailure,
            detail: "transient step retry budget was exhausted".to_string(),
        });
    }
    if fallback_events
        .iter()
        .any(|event| event.starts_with("SOLVER_BACKEND_FALLBACK"))
    {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::SolverBackendFallback,
            detail: "solver backend fell back from runtime_tensor to cpu_reference".to_string(),
        });
    }
    if fallback_events.iter().any(|event| {
        event.starts_with("BACKEND_NO_PROVIDER") || event.starts_with("BACKEND_UPLOAD_FAILED")
    }) {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::FieldPromotionFallback,
            detail: "field promotion fell back to host-backed values".to_string(),
        });
    }
    if run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_TRANSIENT_PLACEHOLDER")
    {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::TransientPlaceholder,
            detail: "transient run path currently uses linear-static placeholder backend"
                .to_string(),
        });
    }

    let publishable = match options.quality_policy {
        QualityPolicy::Strict => {
            solver_convergence == QualityGate::Pass
                && result_quality == QualityGate::Pass
                && quality_reasons.is_empty()
        }
        QualityPolicy::Balanced => {
            solver_convergence == QualityGate::Pass
                && result_quality == QualityGate::Pass
                && !quality_reasons.iter().any(|r| {
                    matches!(
                        r.code,
                        QualityReasonCode::TransientStabilityExceeded
                            | QualityReasonCode::TransientStepFailure
                    )
                })
        }
        QualityPolicy::Exploratory => {
            solver_convergence != QualityGate::Fail && result_quality != QualityGate::Fail
        }
    };
    let run_status = if publishable {
        RunStatus::Publishable
    } else if result_quality == QualityGate::Fail {
        RunStatus::Rejected
    } else {
        RunStatus::Degraded
    };

    let solver_backend = run.solver_backend.clone();
    let solver_device_apply_k_ratio = run.solver_device_apply_k_ratio;
    let solver_host_sync_count = run.solver_host_sync_count;
    let solver_method = run.solver_method.clone();
    let selected_preconditioner = run.preconditioner.clone();

    let result = AnalysisRunResult {
        run_id: storage::next_run_id(),
        run,
        modal_results: None,
        transient_results: Some(TransientResultsData {
            transient_payload_version: "transient_results/v1".to_string(),
            time_points_s: transient_run.time_points_s,
            displacement_snapshots: transient_run.displacement_snapshots,
            residual_norms: transient_run.residual_norms,
            integration_method: TransientIntegrationMethod::ImplicitEuler,
        }),
        nonlinear_results: None,
        model_validity: QualityGate::Pass,
        solver_convergence,
        result_quality,
        run_status,
        publishable,
        quality_reasons,
        provenance: RunProvenance {
            backend,
            solver_backend,
            solver_device_apply_k_ratio,
            solver_host_sync_count,
            precision_mode: contracts::format_precision_mode(options.precision_mode),
            deterministic_mode: options.deterministic_mode,
            solver_method,
            preconditioner: selected_preconditioner,
            quality_policy: contracts::format_quality_policy(options.quality_policy),
            fallback_events,
        },
    };

    storage::persist_run_result(&result).map_err(|err| {
        operation_error(
            ANALYSIS_RUN_TRANSIENT_OPERATION,
            ANALYSIS_RUN_TRANSIENT_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "ANALYSIS_ARTIFACT_STORE_FAILED",
                error_type: OperationErrorType::Internal,
                retryable: true,
                severity: OperationErrorSeverity::Error,
            },
            format!("failed to persist analysis run artifact: {err}"),
            BTreeMap::from([("run_id".to_string(), result.run_id.clone())]),
        )
    })?;

    Ok(OperationEnvelope::new(
        ANALYSIS_RUN_TRANSIENT_OPERATION,
        ANALYSIS_RUN_TRANSIENT_OP_VERSION,
        &context,
        result,
    ))
}

pub fn analysis_run_nonlinear_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    analysis_run_nonlinear_with_options_op(
        model,
        backend,
        AnalysisNonlinearRunOptions::default(),
        context,
    )
}

pub fn analysis_run_nonlinear_with_options_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: AnalysisNonlinearRunOptions,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    let has_nonlinear_step = model
        .steps
        .iter()
        .any(|step| step.kind == AnalysisStepKind::Nonlinear);
    if !has_nonlinear_step {
        return Err(operation_error(
            ANALYSIS_RUN_NONLINEAR_OPERATION,
            ANALYSIS_RUN_NONLINEAR_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "ANALYSIS_RUN_NONLINEAR_INVALID_MODEL",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "analysis model must include at least one nonlinear step for analysis.run_nonlinear",
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                ("geometry_id".to_string(), model.geometry_id.clone()),
            ]),
        ));
    }

    if options.increment_count == 0 {
        return Err(operation_error(
            ANALYSIS_RUN_NONLINEAR_OPERATION,
            ANALYSIS_RUN_NONLINEAR_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "ANALYSIS_RUN_NONLINEAR_INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "analysis.run_nonlinear options require increment_count greater than zero",
            BTreeMap::from([(
                "increment_count".to_string(),
                options.increment_count.to_string(),
            )]),
        ));
    }

    let nonlinear_run = run_nonlinear_with_options(
        model,
        backend,
        runmat_analysis_fea::solve::nonlinear::NonlinearSolveOptions {
            increment_count: options.increment_count,
            max_newton_iters: options.max_newton_iters,
            tolerance: options.tolerance,
            line_search: options.line_search,
        },
    )
    .map_err(|err| {
        operation_error(
            ANALYSIS_RUN_NONLINEAR_OPERATION,
            ANALYSIS_RUN_NONLINEAR_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "SOLVER_MODEL_INVALID",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            err.to_string(),
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                ("geometry_id".to_string(), model.geometry_id.clone()),
            ]),
        )
    })?;

    let mut run = nonlinear_run.run;
    let mut fallback_events = Vec::new();
    promotion::promote_run_fields_to_device_refs(&mut run, &mut fallback_events);
    if backend == ComputeBackend::Gpu && run.solver_backend != "runtime_tensor" {
        fallback_events.push(
            "SOLVER_BACKEND_FALLBACK:requested=runtime_tensor:using=cpu_reference".to_string(),
        );
    }

    let solver_convergence = if run.diagnostics.iter().any(|item| {
        item.code == "FEA_NONLINEAR_CONVERGENCE"
            && item.severity == runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info
    }) {
        QualityGate::Pass
    } else {
        QualityGate::Warn
    };
    let max_nonlinear_residual = nonlinear_run
        .residual_norms
        .iter()
        .copied()
        .reduce(f64::max)
        .unwrap_or(0.0);
    let result_quality = if nonlinear_run.load_factors.is_empty()
        || nonlinear_run.displacement_snapshots.is_empty()
        || nonlinear_run.residual_norms.iter().any(|r| !r.is_finite())
    {
        QualityGate::Fail
    } else if max_nonlinear_residual > options.tolerance * 10.0 {
        QualityGate::Warn
    } else {
        QualityGate::Pass
    };
    let nonlinear_increment_warn = run.diagnostics.iter().any(|item| {
        item.code == "FEA_NONLINEAR_CONVERGENCE"
            && item.severity == runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
    });

    let mut quality_reasons = Vec::new();
    if solver_convergence == QualityGate::Warn {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::SolverNotConverged,
            detail: "nonlinear solver convergence gate is warning".to_string(),
        });
    }
    if result_quality == QualityGate::Warn {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::NonlinearResidualExceeded,
            detail: format!(
                "nonlinear residual exceeds threshold {}",
                options.tolerance * 10.0
            ),
        });
    }
    if nonlinear_increment_warn {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::NonlinearIncrementFailure,
            detail: "nonlinear increment convergence reported warnings".to_string(),
        });
    }
    if fallback_events
        .iter()
        .any(|event| event.starts_with("SOLVER_BACKEND_FALLBACK"))
    {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::SolverBackendFallback,
            detail: "solver backend fell back from runtime_tensor to cpu_reference".to_string(),
        });
    }
    if fallback_events.iter().any(|event| {
        event.starts_with("BACKEND_NO_PROVIDER") || event.starts_with("BACKEND_UPLOAD_FAILED")
    }) {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::FieldPromotionFallback,
            detail: "field promotion fell back to host-backed values".to_string(),
        });
    }

    let publishable = match options.quality_policy {
        QualityPolicy::Strict => {
            solver_convergence == QualityGate::Pass
                && result_quality == QualityGate::Pass
                && quality_reasons.is_empty()
        }
        QualityPolicy::Balanced => {
            solver_convergence == QualityGate::Pass
                && result_quality == QualityGate::Pass
                && !quality_reasons.iter().any(|r| {
                    matches!(
                        r.code,
                        QualityReasonCode::NonlinearResidualExceeded
                            | QualityReasonCode::NonlinearIncrementFailure
                    )
                })
        }
        QualityPolicy::Exploratory => {
            solver_convergence != QualityGate::Fail && result_quality != QualityGate::Fail
        }
    };
    let run_status = if publishable {
        RunStatus::Publishable
    } else if result_quality == QualityGate::Fail {
        RunStatus::Rejected
    } else {
        RunStatus::Degraded
    };

    let solver_backend = run.solver_backend.clone();
    let solver_device_apply_k_ratio = run.solver_device_apply_k_ratio;
    let solver_host_sync_count = run.solver_host_sync_count;
    let solver_method = run.solver_method.clone();
    let selected_preconditioner = run.preconditioner.clone();

    let result = AnalysisRunResult {
        run_id: storage::next_run_id(),
        run,
        modal_results: None,
        transient_results: None,
        nonlinear_results: Some(NonlinearResultsData {
            nonlinear_payload_version: "nonlinear_results/v1".to_string(),
            load_factors: nonlinear_run.load_factors,
            displacement_snapshots: nonlinear_run.displacement_snapshots,
            residual_norms: nonlinear_run.residual_norms,
            iteration_counts: nonlinear_run.iteration_counts,
            method: NonlinearMethod::IncrementalNewtonRaphson,
        }),
        model_validity: QualityGate::Pass,
        solver_convergence,
        result_quality,
        run_status,
        publishable,
        quality_reasons,
        provenance: RunProvenance {
            backend,
            solver_backend,
            solver_device_apply_k_ratio,
            solver_host_sync_count,
            precision_mode: contracts::format_precision_mode(options.precision_mode),
            deterministic_mode: options.deterministic_mode,
            solver_method,
            preconditioner: selected_preconditioner,
            quality_policy: contracts::format_quality_policy(options.quality_policy),
            fallback_events,
        },
    };

    storage::persist_run_result(&result).map_err(|err| {
        operation_error(
            ANALYSIS_RUN_NONLINEAR_OPERATION,
            ANALYSIS_RUN_NONLINEAR_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "ANALYSIS_ARTIFACT_STORE_FAILED",
                error_type: OperationErrorType::Internal,
                retryable: true,
                severity: OperationErrorSeverity::Error,
            },
            format!("failed to persist analysis run artifact: {err}"),
            BTreeMap::from([("run_id".to_string(), result.run_id.clone())]),
        )
    })?;

    Ok(OperationEnvelope::new(
        ANALYSIS_RUN_NONLINEAR_OPERATION,
        ANALYSIS_RUN_NONLINEAR_OP_VERSION,
        &context,
        result,
    ))
}

pub fn analysis_run_linear_static_with_options(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: AnalysisRunOptions,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    let requested_preconditioner = match options.preconditioner_mode {
        PreconditionerMode::Auto | PreconditionerMode::Jacobi => SpdPreconditionerKind::Jacobi,
        PreconditionerMode::Ilu => SpdPreconditionerKind::Ilu0,
        PreconditionerMode::Amg => SpdPreconditionerKind::Jacobi,
    };
    let runtime_tensor_available = runmat_accelerate_api::provider().is_some();
    let requested_solver_backend = match backend {
        ComputeBackend::Cpu => LinearAlgebraBackendKind::CpuReference,
        ComputeBackend::Gpu => {
            if runtime_tensor_available {
                LinearAlgebraBackendKind::RuntimeTensor
            } else {
                LinearAlgebraBackendKind::CpuReference
            }
        }
    };
    let run = run_linear_static_with_options(
        model,
        backend,
        LinearStaticSolveOptions {
            preconditioner_kind: requested_preconditioner,
            algebra_backend_kind: requested_solver_backend,
        },
    )
    .map_err(|err| {
        operation_error(
            ANALYSIS_RUN_OPERATION,
            ANALYSIS_RUN_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "SOLVER_MODEL_INVALID",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            err.to_string(),
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                ("geometry_id".to_string(), model.geometry_id.clone()),
            ]),
        )
    })?;

    let mut run = run;
    let mut fallback_events = Vec::new();
    promotion::promote_run_fields_to_device_refs(&mut run, &mut fallback_events);

    match options.preconditioner_mode {
        PreconditionerMode::Auto | PreconditionerMode::Jacobi | PreconditionerMode::Ilu => {}
        PreconditionerMode::Amg => {
            fallback_events.push(
                "SOLVER_PRECONDITIONER_FALLBACK:requested=amg:using=jacobi".to_string(),
            );
        }
    }

    if backend == ComputeBackend::Gpu && run.solver_backend != "runtime_tensor" {
        fallback_events.push(
            "SOLVER_BACKEND_FALLBACK:requested=runtime_tensor:using=cpu_reference".to_string(),
        );
    }

    let solver_convergence = if run.diagnostics.iter().any(|item| {
        item.code == "FEA_CONVERGENCE"
            && item.severity == runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info
    }) {
        QualityGate::Pass
    } else {
        QualityGate::Warn
    };

    let has_material_assignment_conflict = run
        .diagnostics
        .iter()
        .any(|diag| diag.code.starts_with("ANALYSIS_MATERIAL_ASSIGNMENT_CONFLICT_"));
    let result_quality = if run.displacement_field.is_empty() || run.von_mises_field.is_empty() {
        QualityGate::Fail
    } else if has_material_assignment_conflict {
        QualityGate::Warn
    } else {
        QualityGate::Pass
    };

    let mut quality_reasons = Vec::new();
    if has_material_assignment_conflict {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::MaterialAssignmentConflict,
            detail: "material assignment confidence conflict detected".to_string(),
        });
    }
    if solver_convergence == QualityGate::Warn {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::SolverNotConverged,
            detail: "solver convergence gate is warning".to_string(),
        });
    }
    if fallback_events
        .iter()
        .any(|event| event.starts_with("SOLVER_BACKEND_FALLBACK"))
    {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::SolverBackendFallback,
            detail: "solver backend fell back from runtime_tensor to cpu_reference".to_string(),
        });
    }
    if fallback_events
        .iter()
        .any(|event| event.starts_with("BACKEND_NO_PROVIDER") || event.starts_with("BACKEND_UPLOAD_FAILED"))
    {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::FieldPromotionFallback,
            detail: "field promotion fell back to host-backed values".to_string(),
        });
    }
    let solver_backend = run.solver_backend.clone();
    let solver_device_apply_k_ratio = run.solver_device_apply_k_ratio;
    let solver_host_sync_count = run.solver_host_sync_count;
    let solver_method = run.solver_method.clone();
    let selected_preconditioner = run.preconditioner.clone();

    let publishable = match options.quality_policy {
        QualityPolicy::Strict => {
            solver_convergence == QualityGate::Pass
                && result_quality == QualityGate::Pass
                && quality_reasons.is_empty()
        }
        QualityPolicy::Balanced => {
            solver_convergence == QualityGate::Pass && result_quality == QualityGate::Pass
        }
        QualityPolicy::Exploratory => {
            solver_convergence != QualityGate::Fail && result_quality != QualityGate::Fail
        }
    };
    let run_status = if publishable {
        RunStatus::Publishable
    } else if result_quality == QualityGate::Fail {
        RunStatus::Rejected
    } else {
        RunStatus::Degraded
    };

    let result = AnalysisRunResult {
        run_id: storage::next_run_id(),
        run,
        modal_results: None,
        transient_results: None,
        nonlinear_results: None,
        model_validity: QualityGate::Pass,
        solver_convergence,
        result_quality,
        run_status,
        publishable,
        quality_reasons,
        provenance: RunProvenance {
            backend,
            solver_backend,
            solver_device_apply_k_ratio,
            solver_host_sync_count,
            precision_mode: contracts::format_precision_mode(options.precision_mode),
            deterministic_mode: options.deterministic_mode,
            solver_method,
            preconditioner: selected_preconditioner,
            quality_policy: contracts::format_quality_policy(options.quality_policy),
            fallback_events,
        },
    };

    storage::persist_run_result(&result).map_err(|err| {
        operation_error(
            ANALYSIS_RUN_OPERATION,
            ANALYSIS_RUN_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "ANALYSIS_ARTIFACT_STORE_FAILED",
                error_type: OperationErrorType::Internal,
                retryable: true,
                severity: OperationErrorSeverity::Error,
            },
            format!("failed to persist analysis run artifact: {err}"),
            BTreeMap::from([("run_id".to_string(), result.run_id.clone())]),
        )
    })?;

    Ok(OperationEnvelope::new(
        ANALYSIS_RUN_OPERATION,
        ANALYSIS_RUN_OP_VERSION,
        &context,
        result,
    ))
}

pub fn analysis_results_op(
    run_result: &AnalysisRunResult,
    query: AnalysisResultsQuery,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisResultsData>, OperationErrorEnvelope> {
    let mut fields = vec![
        run_result.run.displacement_field.clone(),
        run_result.run.von_mises_field.clone(),
    ];

    if !query.include_fields.is_empty() {
        let mut filtered = Vec::new();
        for requested in &query.include_fields {
            let Some(field) = fields.iter().find(|field| &field.field_id == requested) else {
                return Err(operation_error(
                    ANALYSIS_RESULTS_OPERATION,
                    ANALYSIS_RESULTS_OP_VERSION,
                    &context,
                    OperationErrorSpec {
                        error_code: "ANALYSIS_RESULTS_FIELD_NOT_FOUND",
                        error_type: OperationErrorType::Input,
                        retryable: false,
                        severity: OperationErrorSeverity::Error,
                    },
                    format!("requested analysis field '{requested}' was not produced by run"),
                    BTreeMap::from([
                        ("requested_field".to_string(), requested.clone()),
                        (
                            "available_fields".to_string(),
                            fields
                                .iter()
                                .map(|field| field.field_id.clone())
                                .collect::<Vec<_>>()
                                .join(","),
                        ),
                    ]),
                ));
            };
            filtered.push(field.clone());
        }
        fields = filtered;
    }

    let (
        mode_count,
        available_mode_indices,
        min_frequency_hz,
        max_frequency_hz,
        max_modal_residual_norm,
        first_mode_converged,
    ) =
        if let Some(modal) = run_result.modal_results.as_ref() {
            let count = modal.eigenvalues_hz.len().min(modal.mode_shapes.len());
            let max_modal_residual_norm = modal
                .residual_norms
                .iter()
                .copied()
                .reduce(f64::max);
            let first_mode_converged = modal.residual_norms.first().copied().map(|v| v <= 1.0e-6);
            let (min_frequency_hz, max_frequency_hz) = if count == 0 {
                (None, None)
            } else {
                let mut min_value = f64::INFINITY;
                let mut max_value = f64::NEG_INFINITY;
                for value in modal.eigenvalues_hz.iter().copied().take(count) {
                    min_value = min_value.min(value);
                    max_value = max_value.max(value);
                }
                (Some(min_value), Some(max_value))
            };
            (
                count,
                (0..count).collect(),
                min_frequency_hz,
                max_frequency_hz,
                max_modal_residual_norm,
                first_mode_converged,
            )
        } else {
            (0, Vec::new(), None, None, None, None)
        };

    let (
        snapshot_count,
        time_start_s,
        time_end_s,
        max_transient_residual_norm,
        final_step_converged,
    ) =
        if let Some(transient) = run_result.transient_results.as_ref() {
            let count = transient.time_points_s.len().min(transient.displacement_snapshots.len());
            let max_residual = transient
                .residual_norms
                .iter()
                .copied()
                .reduce(f64::max);
            let final_step_converged = max_residual.map(|value| value <= 1.0e-6);
            if count == 0 {
                (0, None, None, max_residual, final_step_converged)
            } else {
                (
                    count,
                    transient.time_points_s.first().copied(),
                    transient.time_points_s.get(count - 1).copied(),
                    max_residual,
                    final_step_converged,
                )
            }
        } else {
            (0, None, None, None, None)
        };

    let (increment_count, max_nonlinear_residual_norm, final_increment_converged) =
        if let Some(nonlinear) = run_result.nonlinear_results.as_ref() {
            let count = nonlinear.load_factors.len();
            let max_residual = nonlinear.residual_norms.iter().copied().reduce(f64::max);
            let final_converged = max_residual.map(|value| value <= 1.0e-6);
            (count, max_residual, final_converged)
        } else {
            (0, None, None)
        };

    let summary = AnalysisResultsSummary {
        field_count: fields.len(),
        total_elements: fields.iter().map(|field| field.element_count()).sum(),
        mode_count,
        available_mode_indices,
        min_frequency_hz,
        max_frequency_hz,
        max_modal_residual_norm,
        first_mode_converged,
        snapshot_count,
        time_start_s,
        time_end_s,
        max_transient_residual_norm,
        final_step_converged,
        increment_count,
        max_nonlinear_residual_norm,
        final_increment_converged,
    };

    let modal_results = if query.include_modal_results {
        if let Some(modal) = run_result.modal_results.as_ref() {
            if query.mode_indices.is_empty() {
                Some(modal.clone())
            } else {
                let mut eigenvalues_hz = Vec::with_capacity(query.mode_indices.len());
                let mut mode_shapes = Vec::with_capacity(query.mode_indices.len());
                let mut residual_norms = Vec::with_capacity(query.mode_indices.len());
                for &index in &query.mode_indices {
                    let eigenvalue = modal.eigenvalues_hz.get(index).copied().ok_or_else(|| {
                        operation_error(
                            ANALYSIS_RESULTS_OPERATION,
                            ANALYSIS_RESULTS_OP_VERSION,
                            &context,
                            OperationErrorSpec {
                                error_code: "ANALYSIS_RESULTS_MODE_NOT_FOUND",
                                error_type: OperationErrorType::Input,
                                retryable: false,
                                severity: OperationErrorSeverity::Error,
                            },
                            format!("requested modal mode index '{index}' was not produced by run"),
                            BTreeMap::from([
                                ("requested_mode_index".to_string(), index.to_string()),
                                (
                                    "available_mode_count".to_string(),
                                    modal.eigenvalues_hz.len().to_string(),
                                ),
                            ]),
                        )
                    })?;
                    let mode_shape = modal.mode_shapes.get(index).cloned().ok_or_else(|| {
                        operation_error(
                            ANALYSIS_RESULTS_OPERATION,
                            ANALYSIS_RESULTS_OP_VERSION,
                            &context,
                            OperationErrorSpec {
                                error_code: "ANALYSIS_RESULTS_MODE_NOT_FOUND",
                                error_type: OperationErrorType::Input,
                                retryable: false,
                                severity: OperationErrorSeverity::Error,
                            },
                            format!("requested modal mode index '{index}' is missing mode shape data"),
                            BTreeMap::from([
                                ("requested_mode_index".to_string(), index.to_string()),
                                (
                                    "available_shape_count".to_string(),
                                    modal.mode_shapes.len().to_string(),
                                ),
                            ]),
                        )
                    })?;
                    let residual_norm = modal.residual_norms.get(index).copied().ok_or_else(|| {
                        operation_error(
                            ANALYSIS_RESULTS_OPERATION,
                            ANALYSIS_RESULTS_OP_VERSION,
                            &context,
                            OperationErrorSpec {
                                error_code: "ANALYSIS_RESULTS_MODE_NOT_FOUND",
                                error_type: OperationErrorType::Input,
                                retryable: false,
                                severity: OperationErrorSeverity::Error,
                            },
                            format!(
                                "requested modal mode index '{index}' is missing residual data"
                            ),
                            BTreeMap::from([
                                ("requested_mode_index".to_string(), index.to_string()),
                                (
                                    "available_residual_count".to_string(),
                                    modal.residual_norms.len().to_string(),
                                ),
                            ]),
                        )
                    })?;
                    eigenvalues_hz.push(eigenvalue);
                    mode_shapes.push(mode_shape);
                    residual_norms.push(residual_norm);
                }
                Some(ModalResultsData {
                    modal_payload_version: modal.modal_payload_version.clone(),
                    eigenvalues_hz,
                    mode_shapes,
                    residual_norms,
                    mode_units: modal.mode_units.clone(),
                    frequency_basis: modal.frequency_basis.clone(),
                })
            }
        } else {
            None
        }
    } else {
        None
    };

    let transient_results = if query.include_transient_results {
        if let Some(transient) = run_result.transient_results.as_ref() {
            if query.transient_snapshot_indices.is_empty() {
                Some(transient.clone())
            } else {
                let mut time_points_s = Vec::with_capacity(query.transient_snapshot_indices.len());
                let mut displacement_snapshots =
                    Vec::with_capacity(query.transient_snapshot_indices.len());
                let mut residual_norms = Vec::with_capacity(query.transient_snapshot_indices.len());

                for &index in &query.transient_snapshot_indices {
                    let time_point = transient.time_points_s.get(index).copied().ok_or_else(|| {
                        operation_error(
                            ANALYSIS_RESULTS_OPERATION,
                            ANALYSIS_RESULTS_OP_VERSION,
                            &context,
                            OperationErrorSpec {
                                error_code: "ANALYSIS_RESULTS_TRANSIENT_SNAPSHOT_NOT_FOUND",
                                error_type: OperationErrorType::Input,
                                retryable: false,
                                severity: OperationErrorSeverity::Error,
                            },
                            format!(
                                "requested transient snapshot index '{index}' was not produced by run"
                            ),
                            BTreeMap::from([
                                ("requested_snapshot_index".to_string(), index.to_string()),
                                (
                                    "available_snapshot_count".to_string(),
                                    transient.time_points_s.len().to_string(),
                                ),
                            ]),
                        )
                    })?;
                    let snapshot = transient
                        .displacement_snapshots
                        .get(index)
                        .cloned()
                        .ok_or_else(|| {
                            operation_error(
                                ANALYSIS_RESULTS_OPERATION,
                                ANALYSIS_RESULTS_OP_VERSION,
                                &context,
                                OperationErrorSpec {
                                    error_code: "ANALYSIS_RESULTS_TRANSIENT_SNAPSHOT_NOT_FOUND",
                                    error_type: OperationErrorType::Input,
                                    retryable: false,
                                    severity: OperationErrorSeverity::Error,
                                },
                                format!(
                                    "requested transient snapshot index '{index}' is missing displacement data"
                                ),
                                BTreeMap::from([
                                    ("requested_snapshot_index".to_string(), index.to_string()),
                                    (
                                        "available_displacement_snapshot_count".to_string(),
                                        transient.displacement_snapshots.len().to_string(),
                                    ),
                                ]),
                            )
                        })?;

                    if index > 0 {
                        let residual = transient.residual_norms.get(index - 1).copied().ok_or_else(|| {
                            operation_error(
                                ANALYSIS_RESULTS_OPERATION,
                                ANALYSIS_RESULTS_OP_VERSION,
                                &context,
                                OperationErrorSpec {
                                    error_code: "ANALYSIS_RESULTS_TRANSIENT_SNAPSHOT_NOT_FOUND",
                                    error_type: OperationErrorType::Input,
                                    retryable: false,
                                    severity: OperationErrorSeverity::Error,
                                },
                                format!(
                                    "requested transient snapshot index '{index}' is missing residual data"
                                ),
                                BTreeMap::from([
                                    ("requested_snapshot_index".to_string(), index.to_string()),
                                    (
                                        "available_residual_count".to_string(),
                                        transient.residual_norms.len().to_string(),
                                    ),
                                ]),
                            )
                        })?;
                        residual_norms.push(residual);
                    }

                    time_points_s.push(time_point);
                    displacement_snapshots.push(snapshot);
                }

                Some(TransientResultsData {
                    transient_payload_version: transient.transient_payload_version.clone(),
                    time_points_s,
                    displacement_snapshots,
                    residual_norms,
                    integration_method: transient.integration_method,
                })
            }
        } else {
            None
        }
    } else {
        None
    };

    let nonlinear_results = if query.include_nonlinear_results {
        run_result.nonlinear_results.clone()
    } else {
        None
    };

    let data = AnalysisResultsData {
        fields,
        modal_results,
        transient_results,
        nonlinear_results,
        diagnostics: if query.include_diagnostics {
            Some(run_result.run.diagnostics.clone())
        } else {
            None
        },
        run_status: run_result.run_status,
        publishable: run_result.publishable,
        quality_reasons: run_result.quality_reasons.clone(),
        provenance: run_result.provenance.clone(),
        summary,
    };

    Ok(OperationEnvelope::new(
        ANALYSIS_RESULTS_OPERATION,
        ANALYSIS_RESULTS_OP_VERSION,
        &context,
        data,
    ))
}

pub fn analysis_results_by_run_id_op(
    run_id: &str,
    query: AnalysisResultsQuery,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisResultsData>, OperationErrorEnvelope> {
    let run_result = storage::load_run_result(run_id).map_err(|err| {
        operation_error(
            ANALYSIS_RESULTS_OPERATION,
            ANALYSIS_RESULTS_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "ANALYSIS_ARTIFACT_STORE_FAILED",
                error_type: OperationErrorType::Internal,
                retryable: true,
                severity: OperationErrorSeverity::Error,
            },
            format!("failed to load analysis run artifact: {err}"),
            BTreeMap::from([("run_id".to_string(), run_id.to_string())]),
        )
    })?;

    let Some(run_result) = run_result else {
        return Err(operation_error(
            ANALYSIS_RESULTS_OPERATION,
            ANALYSIS_RESULTS_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "ANALYSIS_RESULTS_RUN_NOT_FOUND",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            format!("analysis run_id '{run_id}' was not found"),
            BTreeMap::from([("run_id".to_string(), run_id.to_string())]),
        ));
    };

    analysis_results_op(&run_result, query, context)
}

fn infer_material_models(geometry: &GeometryAsset) -> Vec<MaterialModel> {
    let mut materials = Vec::new();
    for evidence in &geometry.source_geometry.material_evidence {
        let value = evidence.value.to_ascii_lowercase();
        let (material_id, name, youngs_modulus_pa, poisson_ratio) = if value.contains("aluminum") {
            ("mat_aluminum", "Aluminum", 69e9, 0.33)
        } else if value.contains("steel") {
            ("mat_steel", "Steel", 200e9, 0.30)
        } else if value.contains("polymer") || value.contains("plastic") {
            ("mat_polymer", "Polymer", 3.2e9, 0.37)
        } else {
            ("mat_inferred", "Inferred Material", 100e9, 0.32)
        };

        if materials.iter().any(|m: &MaterialModel| m.material_id == material_id) {
            continue;
        }
        materials.push(MaterialModel {
            material_id: material_id.to_string(),
            name: name.to_string(),
            youngs_modulus_pa,
            poisson_ratio,
        });
    }

    if materials.is_empty() {
        materials.push(MaterialModel {
            material_id: "mat_default_steel".to_string(),
            name: "Steel (Default)".to_string(),
            youngs_modulus_pa: 200e9,
            poisson_ratio: 0.3,
        });
    }

    materials
}

fn select_fixed_region_id(geometry: &GeometryAsset) -> Option<String> {
    geometry
        .regions
        .iter()
        .find(|region| {
            let key = format!(
                "{} {}",
                region.name.to_ascii_lowercase(),
                region
                    .tag
                    .as_deref()
                    .unwrap_or_default()
                    .to_ascii_lowercase()
            );
            key.contains("root")
                || key.contains("base")
                || key.contains("fixed")
                || key.contains("mount")
        })
        .map(|region| region.region_id.clone())
}

fn select_load_region_id(geometry: &GeometryAsset) -> Option<String> {
    geometry
        .regions
        .iter()
        .find(|region| {
            let key = format!(
                "{} {}",
                region.name.to_ascii_lowercase(),
                region
                    .tag
                    .as_deref()
                    .unwrap_or_default()
                    .to_ascii_lowercase()
            );
            key.contains("tip")
                || key.contains("load")
                || key.contains("force")
                || key.contains("free")
        })
        .map(|region| region.region_id.clone())
}

fn infer_material_assignments(
    geometry: &GeometryAsset,
    materials: &[MaterialModel],
) -> Vec<MaterialAssignment> {
    let default_material = materials
        .first()
        .map(|m| m.material_id.clone())
        .unwrap_or_else(|| "mat_default_steel".to_string());
    let mut assignments = Vec::new();

    for region in &geometry.regions {
        let key = format!(
            "{} {}",
            region.name.to_ascii_lowercase(),
            region
                .tag
                .as_deref()
                .unwrap_or_default()
                .to_ascii_lowercase()
        );
        let assigned_material = if key.contains("aluminum") {
            materials
                .iter()
                .find(|m| m.material_id.contains("aluminum"))
                .map(|m| m.material_id.clone())
                .unwrap_or_else(|| default_material.clone())
        } else if key.contains("steel") {
            materials
                .iter()
                .find(|m| m.material_id.contains("steel"))
                .map(|m| m.material_id.clone())
                .unwrap_or_else(|| default_material.clone())
        } else if key.contains("polymer") || key.contains("plastic") {
            materials
                .iter()
                .find(|m| m.material_id.contains("polymer"))
                .map(|m| m.material_id.clone())
                .unwrap_or_else(|| default_material.clone())
        } else {
            default_material.clone()
        };

        let confidence = if geometry
            .source_geometry
            .material_evidence
            .iter()
            .any(|e| e.confidence == MaterialEvidenceConfidence::High)
        {
            EvidenceConfidence::Verified
        } else if geometry
            .source_geometry
            .material_evidence
            .iter()
            .any(|e| e.confidence == MaterialEvidenceConfidence::Medium)
        {
            EvidenceConfidence::Probable
        } else {
            EvidenceConfidence::Inferred
        };

        assignments.push(MaterialAssignment {
            region_id: region.region_id.clone(),
            expected_material_id: assigned_material.clone(),
            assigned_material_id: assigned_material,
            confidence,
        });
    }

    assignments
}

fn map_validate_error(
    error: AnalysisValidationError,
    model: &AnalysisModel,
    context: &OperationContext,
) -> OperationErrorEnvelope {
    let (error_code, message, mut error_context) = match error {
        AnalysisValidationError::MissingMaterials => (
            "ANALYSIS_VALIDATION_MISSING_MATERIALS",
            "analysis model must include at least one material".to_string(),
            BTreeMap::new(),
        ),
        AnalysisValidationError::MissingBoundaryConditions => (
            "ANALYSIS_VALIDATION_MISSING_BCS",
            "analysis model must include at least one boundary condition".to_string(),
            BTreeMap::new(),
        ),
        AnalysisValidationError::MissingLoads => (
            "ANALYSIS_VALIDATION_MISSING_LOADS",
            "analysis model must include at least one load".to_string(),
            BTreeMap::new(),
        ),
        AnalysisValidationError::UnitMismatch { model, geometry } => (
            "ANALYSIS_VALIDATION_UNIT_MISMATCH",
            format!("model units {model:?} do not match geometry units {geometry:?}"),
            BTreeMap::from([
                ("model_units".to_string(), format!("{model:?}")),
                ("geometry_units".to_string(), format!("{geometry:?}")),
            ]),
        ),
        AnalysisValidationError::FrameMismatch { model, geometry } => (
            "ANALYSIS_VALIDATION_FRAME_MISMATCH",
            format!("model frame {model:?} does not match geometry frame {geometry:?}"),
            BTreeMap::from([
                ("model_frame".to_string(), format!("{model:?}")),
                ("geometry_frame".to_string(), format!("{geometry:?}")),
            ]),
        ),
    };

    error_context.insert("analysis_model_id".to_string(), model.model_id.0.clone());
    error_context.insert("geometry_id".to_string(), model.geometry_id.clone());

    operation_error(
        ANALYSIS_VALIDATE_OPERATION,
        ANALYSIS_VALIDATE_OP_VERSION,
        context,
        OperationErrorSpec {
            error_code,
            error_type: OperationErrorType::Validation,
            retryable: false,
            severity: OperationErrorSeverity::Error,
        },
        message,
        error_context,
    )
}

#[cfg(test)]
mod tests;
