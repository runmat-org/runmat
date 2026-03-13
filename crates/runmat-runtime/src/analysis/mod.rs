use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::PathBuf;

use runmat_analysis_core::{
    validate_model_against_geometry, AnalysisModel, AnalysisModelId, AnalysisStep,
    AnalysisStepKind, AnalysisValidationError, BoundaryCondition, BoundaryConditionKind,
    EvidenceConfidence, LoadCase, LoadKind, MaterialAssignment, MaterialMechanicalModel,
    MaterialModel, MaterialThermalModel, ReferenceFrame,
};
use runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind;
use runmat_analysis_fea::solve::preconditioner::SpdPreconditionerKind;
use runmat_analysis_fea::{
    run_linear_static_with_options, run_modal_with_options, run_nonlinear_with_options,
    run_transient_with_options, ComputeBackend, LinearStaticSolveOptions, ModalSolveOptions,
};
use runmat_geometry_core::{GeometryAsset, MaterialEvidenceConfidence, UnitSystem};
use runmat_meshing_core::{ElementFamilyHint, MeshConnectivityClass};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::operations::{
    operation_error, OperationContext, OperationEnvelope, OperationErrorEnvelope,
    OperationErrorSeverity, OperationErrorSpec, OperationErrorType,
};

mod contracts;
mod promotion;
pub mod storage;

pub use contracts::{
    AnalysisCreateModelIntentSpec, AnalysisCreateModelPrepContext, AnalysisCreateModelProfile,
    AnalysisModalRunOptions, AnalysisNonlinearRunOptions, AnalysisResultsCompareData,
    AnalysisResultsCompareQuery, AnalysisResultsData, AnalysisResultsQuery, AnalysisResultsSummary,
    AnalysisRunKind, AnalysisRunOptions, AnalysisRunPrepContext, AnalysisRunResult,
    AnalysisTransientRunOptions, AnalysisTrendKindSummary, AnalysisTrendsData, AnalysisTrendsQuery,
    AnalysisValidateResult, ContactProxyOptions, ElectroRegionConductivityScale,
    ElectroThermalCouplingOptions, ElectroTimeProfilePoint, ModalFrequencyBasis,
    ModalFrequencyUnits, ModalResultsData, NonlinearMethod, NonlinearResultsData,
    PlasticityProxyOptions, PrecisionMode, PreconditionerMode,
    PrepCalibrationProfile, QualityGate, QualityPolicy, QualityReason, QualityReasonCode,
    RunProvenance, RunStatus, ThermoFieldInterpolationMode, ThermoFieldSource,
    ThermoMechanicalCouplingOptions, ThermoRegionTemperatureDelta, ThermoTimeProfilePoint,
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
const ANALYSIS_RESULTS_COMPARE_OPERATION: &str = "analysis.results_compare";
const ANALYSIS_RESULTS_COMPARE_OP_VERSION: &str = "analysis.results_compare/v1";
const ANALYSIS_TRENDS_OPERATION: &str = "analysis.trends";
const ANALYSIS_TRENDS_OP_VERSION: &str = "analysis.trends/v1";
const TRANSIENT_RESIDUAL_WARN_THRESHOLD: f64 = 1.0e-4;
const THERMO_SPREAD_THRESHOLD_BALANCED: f64 = 1.25;
const THERMO_HETEROGENEITY_THRESHOLD_BALANCED: f64 = 0.2;

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

    let prep_mapped_region_ids = if let Some(prep) = intent.prep_context.as_ref() {
        if prep.source_geometry_id != geometry.geometry_id
            || prep.source_geometry_revision != geometry.revision
        {
            return Err(operation_error(
                ANALYSIS_CREATE_MODEL_OPERATION,
                ANALYSIS_CREATE_MODEL_OP_VERSION,
                &context,
                OperationErrorSpec {
                    error_code: "ANALYSIS_CREATE_MODEL_PREP_MISMATCH",
                    error_type: OperationErrorType::Input,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                "analysis model prep context does not match geometry id/revision",
                BTreeMap::from([
                    ("geometry_id".to_string(), geometry.geometry_id.clone()),
                    (
                        "geometry_revision".to_string(),
                        geometry.revision.to_string(),
                    ),
                    (
                        "prep_geometry_id".to_string(),
                        prep.source_geometry_id.clone(),
                    ),
                    (
                        "prep_geometry_revision".to_string(),
                        prep.source_geometry_revision.to_string(),
                    ),
                ]),
            ));
        }

        let mesh_id_set = geometry
            .meshes
            .iter()
            .map(|mesh| mesh.mesh_id.as_str())
            .collect::<HashSet<_>>();
        let region_id_set = geometry
            .regions
            .iter()
            .map(|region| region.region_id.as_str())
            .collect::<HashSet<_>>();
        for mapping in &prep.region_mappings {
            if !region_id_set.is_empty() && !region_id_set.contains(mapping.region_id.as_str()) {
                return Err(operation_error(
                    ANALYSIS_CREATE_MODEL_OPERATION,
                    ANALYSIS_CREATE_MODEL_OP_VERSION,
                    &context,
                    OperationErrorSpec {
                        error_code: "ANALYSIS_CREATE_MODEL_PREP_REGION_NOT_FOUND",
                        error_type: OperationErrorType::Validation,
                        retryable: false,
                        severity: OperationErrorSeverity::Error,
                    },
                    format!(
                        "prep context region '{}' is not present in geometry regions",
                        mapping.region_id
                    ),
                    BTreeMap::from([("region_id".to_string(), mapping.region_id.clone())]),
                ));
            }
            if mapping.source_mesh_ids.is_empty() || mapping.prepared_mesh_ids.is_empty() {
                return Err(operation_error(
                    ANALYSIS_CREATE_MODEL_OPERATION,
                    ANALYSIS_CREATE_MODEL_OP_VERSION,
                    &context,
                    OperationErrorSpec {
                        error_code: "ANALYSIS_CREATE_MODEL_PREP_INVALID_MAPPING",
                        error_type: OperationErrorType::Input,
                        retryable: false,
                        severity: OperationErrorSeverity::Error,
                    },
                    "prep context mapping requires non-empty source/prepared mesh ids",
                    BTreeMap::from([("region_id".to_string(), mapping.region_id.clone())]),
                ));
            }
            for source_mesh_id in &mapping.source_mesh_ids {
                if !mesh_id_set.contains(source_mesh_id.as_str()) {
                    return Err(operation_error(
                        ANALYSIS_CREATE_MODEL_OPERATION,
                        ANALYSIS_CREATE_MODEL_OP_VERSION,
                        &context,
                        OperationErrorSpec {
                            error_code: "ANALYSIS_CREATE_MODEL_PREP_MESH_NOT_FOUND",
                            error_type: OperationErrorType::Validation,
                            retryable: false,
                            severity: OperationErrorSeverity::Error,
                        },
                        format!(
                            "prep context source mesh '{}' is not present in geometry",
                            source_mesh_id
                        ),
                        BTreeMap::from([("source_mesh_id".to_string(), source_mesh_id.clone())]),
                    ));
                }
            }
        }

        Some(
            prep.region_mappings
                .iter()
                .map(|mapping| mapping.region_id.clone())
                .collect::<HashSet<_>>(),
        )
    } else {
        None
    };

    let fixed_region_id = select_fixed_region_id(geometry, prep_mapped_region_ids.as_ref())
        .or_else(|| {
            geometry
                .regions
                .first()
                .map(|region| region.region_id.clone())
        })
        .unwrap_or_else(|| "region_default".to_string());
    let load_region_id = select_load_region_id(geometry, prep_mapped_region_ids.as_ref())
        .or_else(|| {
            geometry
                .regions
                .last()
                .map(|region| region.region_id.clone())
        })
        .unwrap_or_else(|| fixed_region_id.clone());

    let inferred_materials = infer_material_models(geometry);
    let inferred_assignments = infer_material_assignments(
        geometry,
        &inferred_materials,
        prep_mapped_region_ids.as_ref(),
    );

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
        AnalysisCreateModelProfile::ThermoMechanicalCoupled => (
            BoundaryCondition {
                bc_id: "bc_default_fixed".to_string(),
                region_id: fixed_region_id,
                kind: BoundaryConditionKind::Fixed,
            },
            LoadCase {
                load_id: "load_default_thermal_mech_force".to_string(),
                region_id: load_region_id,
                kind: LoadKind::Force {
                    fx: 0.0,
                    fy: -650.0,
                    fz: 0.0,
                },
            },
            AnalysisStep {
                step_id: "step_default_thermo_mech".to_string(),
                kind: AnalysisStepKind::Transient,
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
        interfaces: Vec::new(),
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

    let thermo_options = resolve_thermo_coupling_options(
        model,
        options.thermo_mechanical_coupling.clone(),
        ANALYSIS_RUN_MODAL_OPERATION,
        ANALYSIS_RUN_MODAL_OP_VERSION,
        &context,
    )?;
    if let Some(thermo_options) = thermo_options.as_ref() {
        if let Err((detail, metadata)) = validate_thermo_coupling_options(model, thermo_options) {
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
                detail,
                metadata,
            ));
        }
    }
    if let Some(electro_options) = options.electro_thermal_coupling.as_ref() {
        if let Err((detail, metadata)) = validate_electro_coupling_options(model, electro_options) {
            return Err(operation_error(
                ANALYSIS_RUN_OPERATION,
                ANALYSIS_RUN_OP_VERSION,
                &context,
                OperationErrorSpec {
                    error_code: "ANALYSIS_RUN_INVALID_OPTIONS",
                    error_type: OperationErrorType::Input,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                detail,
                metadata,
            ));
        }
    }
    if let Some(electro_options) = options.electro_thermal_coupling.as_ref() {
        if let Err((detail, metadata)) = validate_electro_coupling_options(model, electro_options) {
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
                detail,
                metadata,
            ));
        }
    }
    if let Some(electro_options) = options.electro_thermal_coupling.as_ref() {
        if let Err((detail, metadata)) = validate_electro_coupling_options(model, electro_options) {
            return Err(operation_error(
                ANALYSIS_RUN_TRANSIENT_OPERATION,
                ANALYSIS_RUN_TRANSIENT_OP_VERSION,
                &context,
                OperationErrorSpec {
                    error_code: "ANALYSIS_RUN_TRANSIENT_INVALID_OPTIONS",
                    error_type: OperationErrorType::Input,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                detail,
                metadata,
            ));
        }
    }
    if let Some(electro_options) = options.electro_thermal_coupling.as_ref() {
        if let Err((detail, metadata)) = validate_electro_coupling_options(model, electro_options) {
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
                detail,
                metadata,
            ));
        }
    }

    let prep_context = resolve_run_prep_context(
        model,
        options.prep_artifact_id.as_deref(),
        options.prep_context,
        ANALYSIS_RUN_MODAL_OPERATION,
        ANALYSIS_RUN_MODAL_OP_VERSION,
        &context,
    )?;

    let modal_run = run_modal_with_options(
        model,
        backend,
        ModalSolveOptions {
            mode_count: options.mode_count,
            prep_context: to_fea_prep_context(prep_context, options.prep_calibration_profile),
            thermo_mechanical_context: to_fea_thermo_mechanical_context(thermo_options),
            electro_thermal_context: to_fea_electro_thermal_context(
                options.electro_thermal_coupling,
            ),
        },
    )
    .map_err(|err| {
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
    })?;

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
            detail: "modal run path currently uses linear-static placeholder backend".to_string(),
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
                && !quality_reasons.iter().any(|r| {
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

    if let Some(nonlinear) = result.nonlinear_results.as_ref() {
        let event = format!(
            "analysis.run_nonlinear outcome run_id={} model_id={} backend={:?} run_status={:?} publishable={} failed_increments={} max_iteration_count={} line_search_backtracks={} tangent_rebuild_count={} max_residual_norm={} max_increment_norm={} max_backtracks_per_increment={} quality_reason_count={}",
            result.run_id,
            model.model_id.0,
            backend,
            result.run_status,
            result.publishable,
            nonlinear.failed_increments,
            nonlinear.iteration_counts.iter().copied().max().unwrap_or(0),
            nonlinear.line_search_backtracks,
            nonlinear.tangent_rebuild_count,
            nonlinear
                .residual_norms
                .iter()
                .copied()
                .reduce(f64::max)
                .unwrap_or(0.0),
            nonlinear
                .increment_norms
                .iter()
                .copied()
                .reduce(f64::max)
                .unwrap_or(0.0),
            nonlinear.max_line_search_backtracks_per_increment,
            result.quality_reasons.len()
        );
        if matches!(result.run_status, RunStatus::Degraded | RunStatus::Rejected) {
            tracing::warn!(target: "runmat_analysis", "{event}");
        } else {
            tracing::info!(target: "runmat_analysis", "{event}");
        }
    }

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

    let thermo_options = resolve_thermo_coupling_options(
        model,
        options.thermo_mechanical_coupling.clone(),
        ANALYSIS_RUN_TRANSIENT_OPERATION,
        ANALYSIS_RUN_TRANSIENT_OP_VERSION,
        &context,
    )?;
    if let Some(thermo_options) = thermo_options.as_ref() {
        if let Err((detail, metadata)) = validate_thermo_coupling_options(model, thermo_options) {
            return Err(operation_error(
                ANALYSIS_RUN_TRANSIENT_OPERATION,
                ANALYSIS_RUN_TRANSIENT_OP_VERSION,
                &context,
                OperationErrorSpec {
                    error_code: "ANALYSIS_RUN_TRANSIENT_INVALID_OPTIONS",
                    error_type: OperationErrorType::Input,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                detail,
                metadata,
            ));
        }
    }

    let transient_run = run_transient_with_options(model, backend, {
        let prep_context = resolve_run_prep_context(
            model,
            options.prep_artifact_id.as_deref(),
            options.prep_context,
            ANALYSIS_RUN_TRANSIENT_OPERATION,
            ANALYSIS_RUN_TRANSIENT_OP_VERSION,
            &context,
        )?;
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
            prep_context: to_fea_prep_context(prep_context, options.prep_calibration_profile),
            thermo_mechanical_context: to_fea_thermo_mechanical_context(thermo_options),
            electro_thermal_context: to_fea_electro_thermal_context(
                options.electro_thermal_coupling,
            ),
        }
    })
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
    let thermo_transient_warn = run.diagnostics.iter().any(|item| {
        item.code == "FEA_TM_TRANSIENT"
            && item.severity == runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
    });
    let electro_transient_warn = run.diagnostics.iter().any(|item| {
        item.code == "FEA_ET_TRANSIENT"
            && item.severity == runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
    });
    let thermo_spatial_gradient_index = diagnostic_metric(
        &run.diagnostics,
        "FEA_TM_COUPLING",
        "spatial_gradient_index",
    );
    let thermo_spatial_coverage_ratio = diagnostic_metric(
        &run.diagnostics,
        "FEA_TM_COUPLING",
        "spatial_coverage_ratio",
    );
    let thermo_temporal_variation =
        diagnostic_metric(&run.diagnostics, "FEA_TM_TRANSIENT", "temporal_variation");
    let thermo_field_extrapolation_ratio = diagnostic_metric(
        &run.diagnostics,
        "FEA_TM_TRANSIENT",
        "field_extrapolation_ratio",
    );
    let (thermo_gradient_spatial_threshold, thermo_gradient_temporal_threshold) =
        thermo_gradient_thresholds_for_policy(options.quality_policy);
    let thermo_gradient_instability = thermo_spatial_gradient_index
        .map(|value| value > thermo_gradient_spatial_threshold)
        .unwrap_or(false)
        || thermo_temporal_variation
            .map(|value| value > thermo_gradient_temporal_threshold)
            .unwrap_or(false);
    let thermo_spread_ratio = diagnostic_metric(
        &run.diagnostics,
        "FEA_TM_COUPLING",
        "constitutive_material_spread_ratio",
    );
    let thermo_heterogeneity_index = diagnostic_metric(
        &run.diagnostics,
        "FEA_TM_COUPLING",
        "assignment_heterogeneity_index",
    );
    let (thermo_spread_threshold, thermo_heterogeneity_threshold) =
        thermo_thresholds_for_policy(options.quality_policy);
    let (thermo_field_coverage_min, thermo_field_extrapolation_max) =
        thermo_field_quality_thresholds_for_policy(options.quality_policy);
    let thermo_spread_breach = thermo_spread_ratio
        .map(|value| value > thermo_spread_threshold)
        .unwrap_or(false);
    let thermo_heterogeneity_breach = thermo_heterogeneity_index
        .map(|value| value > thermo_heterogeneity_threshold)
        .unwrap_or(false);
    let thermo_field_coverage_breach = thermo_spatial_coverage_ratio
        .map(|value| value < thermo_field_coverage_min)
        .unwrap_or(false);
    let thermo_field_extrapolation_breach = thermo_field_extrapolation_ratio
        .map(|value| value > thermo_field_extrapolation_max)
        .unwrap_or(false);

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
    if thermo_transient_warn {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ThermoMechanicalTransientStress,
            detail: "thermo-mechanical transient coupling severity exceeded balanced threshold"
                .to_string(),
        });
    }
    if electro_transient_warn {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ElectroThermalTransientStress,
            detail: "electro-thermal transient coupling severity exceeded balanced threshold"
                .to_string(),
        });
    }
    if thermo_spread_breach {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ThermoMechanicalConstitutiveSpreadHigh,
            detail: format!(
                "thermo constitutive material spread ratio {} exceeds threshold {}",
                thermo_spread_ratio.unwrap_or(0.0),
                thermo_spread_threshold
            ),
        });
    }
    if thermo_heterogeneity_breach {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ThermoMechanicalAssignmentHeterogeneityHigh,
            detail: format!(
                "thermo assignment heterogeneity index {} exceeds threshold {}",
                thermo_heterogeneity_index.unwrap_or(0.0),
                thermo_heterogeneity_threshold
            ),
        });
    }
    if thermo_gradient_instability {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ThermoMechanicalGradientInstability,
            detail: format!(
                "thermo gradient instability spatial_gradient_index={} temporal_variation={} thresholds=({}, {})",
                thermo_spatial_gradient_index.unwrap_or(0.0),
                thermo_temporal_variation.unwrap_or(0.0),
                thermo_gradient_spatial_threshold,
                thermo_gradient_temporal_threshold,
            ),
        });
    }
    if thermo_field_coverage_breach {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ThermoMechanicalFieldCoverageLow,
            detail: format!(
                "thermo field spatial coverage ratio {} is below minimum {}",
                thermo_spatial_coverage_ratio.unwrap_or(0.0),
                thermo_field_coverage_min
            ),
        });
    }
    if thermo_field_extrapolation_breach {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ThermoMechanicalFieldExtrapolationHigh,
            detail: format!(
                "thermo field extrapolation ratio {} exceeds maximum {}",
                thermo_field_extrapolation_ratio.unwrap_or(0.0),
                thermo_field_extrapolation_max
            ),
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
                            | QualityReasonCode::ThermoMechanicalTransientStress
                            | QualityReasonCode::ThermoMechanicalConstitutiveSpreadHigh
                            | QualityReasonCode::ThermoMechanicalAssignmentHeterogeneityHigh
                            | QualityReasonCode::ThermoMechanicalGradientInstability
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
    if options.max_newton_iters == 0 {
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
            "analysis.run_nonlinear options require max_newton_iters greater than zero",
            BTreeMap::from([(
                "max_newton_iters".to_string(),
                options.max_newton_iters.to_string(),
            )]),
        ));
    }
    if options.tolerance <= 0.0 || !options.tolerance.is_finite() {
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
            "analysis.run_nonlinear options require finite positive tolerance",
            BTreeMap::from([("tolerance".to_string(), options.tolerance.to_string())]),
        ));
    }
    if options.increment_norm_tolerance <= 0.0 || !options.increment_norm_tolerance.is_finite() {
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
            "analysis.run_nonlinear options require finite positive increment_norm_tolerance",
            BTreeMap::from([(
                "increment_norm_tolerance".to_string(),
                options.increment_norm_tolerance.to_string(),
            )]),
        ));
    }
    if options.residual_convergence_factor < 1.0 || !options.residual_convergence_factor.is_finite()
    {
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
            "analysis.run_nonlinear options require residual_convergence_factor >= 1.0",
            BTreeMap::from([(
                "residual_convergence_factor".to_string(),
                options.residual_convergence_factor.to_string(),
            )]),
        ));
    }
    if options.line_search_reduction <= 0.0
        || options.line_search_reduction >= 1.0
        || !options.line_search_reduction.is_finite()
    {
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
            "analysis.run_nonlinear options require line_search_reduction in (0, 1)",
            BTreeMap::from([(
                "line_search_reduction".to_string(),
                options.line_search_reduction.to_string(),
            )]),
        ));
    }
    if options.tangent_refresh_interval == 0 {
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
            "analysis.run_nonlinear options require tangent_refresh_interval greater than zero",
            BTreeMap::from([(
                "tangent_refresh_interval".to_string(),
                options.tangent_refresh_interval.to_string(),
            )]),
        ));
    }

    let thermo_options = resolve_thermo_coupling_options(
        model,
        options.thermo_mechanical_coupling.clone(),
        ANALYSIS_RUN_NONLINEAR_OPERATION,
        ANALYSIS_RUN_NONLINEAR_OP_VERSION,
        &context,
    )?;
    if let Some(thermo_options) = thermo_options.as_ref() {
        if let Err((detail, metadata)) = validate_thermo_coupling_options(model, thermo_options) {
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
                detail,
                metadata,
            ));
        }
    }
    if let Some(plasticity_options) = options.plasticity_proxy.as_ref() {
        if let Err((detail, metadata)) = validate_plasticity_proxy_options(plasticity_options) {
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
                detail,
                metadata,
            ));
        }
    }
    if let Some(contact_options) = options.contact_proxy.as_ref() {
        if let Err((detail, metadata)) = validate_contact_proxy_options(contact_options) {
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
                detail,
                metadata,
            ));
        }
    }

    let nonlinear_run = run_nonlinear_with_options(model, backend, {
        let prep_context = resolve_run_prep_context(
            model,
            options.prep_artifact_id.as_deref(),
            options.prep_context,
            ANALYSIS_RUN_NONLINEAR_OPERATION,
            ANALYSIS_RUN_NONLINEAR_OP_VERSION,
            &context,
        )?;
        runmat_analysis_fea::solve::nonlinear::NonlinearSolveOptions {
            increment_count: options.increment_count,
            max_newton_iters: options.max_newton_iters,
            tolerance: options.tolerance,
            residual_convergence_factor: options.residual_convergence_factor,
            increment_norm_tolerance: options.increment_norm_tolerance,
            line_search: options.line_search,
            max_line_search_backtracks: options.max_line_search_backtracks,
            line_search_reduction: options.line_search_reduction,
            tangent_refresh_interval: options.tangent_refresh_interval,
            prep_context: to_fea_prep_context(prep_context, options.prep_calibration_profile),
            thermo_mechanical_context: to_fea_thermo_mechanical_context(thermo_options),
            electro_thermal_context: to_fea_electro_thermal_context(
                options.electro_thermal_coupling,
            ),
            plasticity_proxy_context: to_fea_plasticity_proxy_context(options.plasticity_proxy),
            contact_proxy_context: to_fea_contact_proxy_context(options.contact_proxy),
        }
    })
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
    let max_nonlinear_increment_norm = nonlinear_run
        .increment_norms
        .iter()
        .copied()
        .reduce(f64::max)
        .unwrap_or(0.0);
    let result_quality = if nonlinear_run.load_factors.is_empty()
        || nonlinear_run.displacement_snapshots.is_empty()
        || nonlinear_run.residual_norms.iter().any(|r| !r.is_finite())
        || nonlinear_run.increment_norms.iter().any(|v| !v.is_finite())
    {
        QualityGate::Fail
    } else if max_nonlinear_residual > options.tolerance * options.residual_convergence_factor * 2.0
        || max_nonlinear_increment_norm > options.increment_norm_tolerance * 4.0
    {
        QualityGate::Warn
    } else {
        QualityGate::Pass
    };
    let nonlinear_increment_warn = run.diagnostics.iter().any(|item| {
        item.code == "FEA_NONLINEAR_CONVERGENCE"
            && item.severity == runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
    });
    let max_nonlinear_iteration_count = nonlinear_run
        .iteration_counts
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let iteration_cap_hits = nonlinear_run
        .iteration_counts
        .iter()
        .filter(|&&count| count >= options.max_newton_iters.max(1))
        .count();
    let strict_increment_failure = nonlinear_run.failed_increments > 0;
    let strict_iteration_cap_exhausted = iteration_cap_hits > 0;
    let thermo_nonlinear_warn = run.diagnostics.iter().any(|item| {
        item.code == "FEA_TM_NONLINEAR"
            && item.severity == runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
    });
    let electro_nonlinear_warn = run.diagnostics.iter().any(|item| {
        item.code == "FEA_ET_NONLINEAR"
            && item.severity == runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
    });
    let plastic_nonlinear_warn = run.diagnostics.iter().any(|item| {
        item.code == "FEA_PLASTIC_NONLINEAR"
            && item.severity == runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
    });
    let contact_nonlinear_warn = run.diagnostics.iter().any(|item| {
        item.code == "FEA_CONTACT_NONLINEAR"
            && item.severity == runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
    });
    let thermo_spatial_gradient_index = diagnostic_metric(
        &run.diagnostics,
        "FEA_TM_COUPLING",
        "spatial_gradient_index",
    );
    let thermo_spatial_coverage_ratio = diagnostic_metric(
        &run.diagnostics,
        "FEA_TM_COUPLING",
        "spatial_coverage_ratio",
    );
    let thermo_temporal_variation =
        diagnostic_metric(&run.diagnostics, "FEA_TM_NONLINEAR", "temporal_variation");
    let thermo_field_extrapolation_ratio = diagnostic_metric(
        &run.diagnostics,
        "FEA_TM_NONLINEAR",
        "field_extrapolation_ratio",
    );
    let (thermo_gradient_spatial_threshold, thermo_gradient_temporal_threshold) =
        thermo_gradient_thresholds_for_policy(options.quality_policy);
    let thermo_gradient_instability = thermo_spatial_gradient_index
        .map(|value| value > thermo_gradient_spatial_threshold)
        .unwrap_or(false)
        || thermo_temporal_variation
            .map(|value| value > thermo_gradient_temporal_threshold)
            .unwrap_or(false);
    let thermo_spread_ratio = diagnostic_metric(
        &run.diagnostics,
        "FEA_TM_COUPLING",
        "constitutive_material_spread_ratio",
    );
    let thermo_heterogeneity_index = diagnostic_metric(
        &run.diagnostics,
        "FEA_TM_COUPLING",
        "assignment_heterogeneity_index",
    );
    let (thermo_spread_threshold, thermo_heterogeneity_threshold) =
        thermo_thresholds_for_policy(options.quality_policy);
    let (thermo_field_coverage_min, thermo_field_extrapolation_max) =
        thermo_field_quality_thresholds_for_policy(options.quality_policy);
    let thermo_spread_breach = thermo_spread_ratio
        .map(|value| value > thermo_spread_threshold)
        .unwrap_or(false);
    let thermo_heterogeneity_breach = thermo_heterogeneity_index
        .map(|value| value > thermo_heterogeneity_threshold)
        .unwrap_or(false);
    let thermo_field_coverage_breach = thermo_spatial_coverage_ratio
        .map(|value| value < thermo_field_coverage_min)
        .unwrap_or(false);
    let thermo_field_extrapolation_breach = thermo_field_extrapolation_ratio
        .map(|value| value > thermo_field_extrapolation_max)
        .unwrap_or(false);

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
                "nonlinear residual/increment norm exceeds thresholds residual={} increment_norm={}",
                options.tolerance * options.residual_convergence_factor * 2.0,
                options.increment_norm_tolerance * 4.0
            ),
        });
    }
    if nonlinear_increment_warn || strict_increment_failure || strict_iteration_cap_exhausted {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::NonlinearIncrementFailure,
            detail: format!(
                "nonlinear increment convergence warnings failed_increments={} iteration_cap_hits={} max_iteration_count={}",
                nonlinear_run.failed_increments,
                iteration_cap_hits,
                max_nonlinear_iteration_count
            ),
        });
    }
    if thermo_nonlinear_warn {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ThermoMechanicalNonlinearStress,
            detail: "thermo-mechanical nonlinear coupling severity exceeded balanced threshold"
                .to_string(),
        });
    }
    if electro_nonlinear_warn {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ElectroThermalNonlinearStress,
            detail: "electro-thermal nonlinear coupling severity exceeded balanced threshold"
                .to_string(),
        });
    }
    if plastic_nonlinear_warn {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::PlasticityNonlinearStress,
            detail: "plasticity nonlinear severity exceeded balanced threshold".to_string(),
        });
    }
    if contact_nonlinear_warn {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ContactNonlinearStress,
            detail: "contact nonlinear severity exceeded balanced threshold".to_string(),
        });
    }
    if thermo_spread_breach {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ThermoMechanicalConstitutiveSpreadHigh,
            detail: format!(
                "thermo constitutive material spread ratio {} exceeds threshold {}",
                thermo_spread_ratio.unwrap_or(0.0),
                thermo_spread_threshold
            ),
        });
    }
    if thermo_heterogeneity_breach {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ThermoMechanicalAssignmentHeterogeneityHigh,
            detail: format!(
                "thermo assignment heterogeneity index {} exceeds threshold {}",
                thermo_heterogeneity_index.unwrap_or(0.0),
                thermo_heterogeneity_threshold
            ),
        });
    }
    if thermo_gradient_instability {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ThermoMechanicalGradientInstability,
            detail: format!(
                "thermo gradient instability spatial_gradient_index={} temporal_variation={} thresholds=({}, {})",
                thermo_spatial_gradient_index.unwrap_or(0.0),
                thermo_temporal_variation.unwrap_or(0.0),
                thermo_gradient_spatial_threshold,
                thermo_gradient_temporal_threshold,
            ),
        });
    }
    if thermo_field_coverage_breach {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ThermoMechanicalFieldCoverageLow,
            detail: format!(
                "thermo field spatial coverage ratio {} is below minimum {}",
                thermo_spatial_coverage_ratio.unwrap_or(0.0),
                thermo_field_coverage_min
            ),
        });
    }
    if thermo_field_extrapolation_breach {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ThermoMechanicalFieldExtrapolationHigh,
            detail: format!(
                "thermo field extrapolation ratio {} exceeds maximum {}",
                thermo_field_extrapolation_ratio.unwrap_or(0.0),
                thermo_field_extrapolation_max
            ),
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
                && !strict_increment_failure
                && !strict_iteration_cap_exhausted
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
                            | QualityReasonCode::ThermoMechanicalNonlinearStress
                            | QualityReasonCode::PlasticityNonlinearStress
                            | QualityReasonCode::ContactNonlinearStress
                            | QualityReasonCode::ThermoMechanicalConstitutiveSpreadHigh
                            | QualityReasonCode::ThermoMechanicalAssignmentHeterogeneityHigh
                            | QualityReasonCode::ThermoMechanicalGradientInstability
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
            increment_norms: nonlinear_run.increment_norms,
            iteration_counts: nonlinear_run.iteration_counts,
            failed_increments: nonlinear_run.failed_increments,
            line_search_backtracks: nonlinear_run.line_search_backtracks,
            max_line_search_backtracks_per_increment: nonlinear_run
                .max_line_search_backtracks_per_increment,
            tangent_rebuild_count: nonlinear_run.tangent_rebuild_count,
            iteration_spike_count: nonlinear_run.iteration_spike_count,
            convergence_stall_count: nonlinear_run.convergence_stall_count,
            backtrack_burst_count: nonlinear_run.backtrack_burst_count,
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
    let thermo_options = resolve_thermo_coupling_options(
        model,
        options.thermo_mechanical_coupling.clone(),
        ANALYSIS_RUN_OPERATION,
        ANALYSIS_RUN_OP_VERSION,
        &context,
    )?;
    if let Some(thermo_options) = thermo_options.as_ref() {
        if let Err((detail, metadata)) = validate_thermo_coupling_options(model, thermo_options) {
            return Err(operation_error(
                ANALYSIS_RUN_OPERATION,
                ANALYSIS_RUN_OP_VERSION,
                &context,
                OperationErrorSpec {
                    error_code: "ANALYSIS_RUN_INVALID_OPTIONS",
                    error_type: OperationErrorType::Input,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                detail,
                metadata,
            ));
        }
    }

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
    let run = run_linear_static_with_options(model, backend, {
        let prep_context = resolve_run_prep_context(
            model,
            options.prep_artifact_id.as_deref(),
            options.prep_context,
            ANALYSIS_RUN_OPERATION,
            ANALYSIS_RUN_OP_VERSION,
            &context,
        )?;
        LinearStaticSolveOptions {
            preconditioner_kind: requested_preconditioner,
            algebra_backend_kind: requested_solver_backend,
            prep_context: to_fea_prep_context(prep_context, options.prep_calibration_profile),
            thermo_mechanical_context: to_fea_thermo_mechanical_context(thermo_options),
            electro_thermal_context: to_fea_electro_thermal_context(
                options.electro_thermal_coupling,
            ),
        }
    })
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
            fallback_events
                .push("SOLVER_PRECONDITIONER_FALLBACK:requested=amg:using=jacobi".to_string());
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

    let has_material_assignment_conflict = run.diagnostics.iter().any(|diag| {
        diag.code
            .starts_with("ANALYSIS_MATERIAL_ASSIGNMENT_CONFLICT_")
    });
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
    if fallback_events.iter().any(|event| {
        event.starts_with("BACKEND_NO_PROVIDER") || event.starts_with("BACKEND_UPLOAD_FAILED")
    }) {
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
    ) = if let Some(modal) = run_result.modal_results.as_ref() {
        let count = modal.eigenvalues_hz.len().min(modal.mode_shapes.len());
        let max_modal_residual_norm = modal.residual_norms.iter().copied().reduce(f64::max);
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
    ) = if let Some(transient) = run_result.transient_results.as_ref() {
        let count = transient
            .time_points_s
            .len()
            .min(transient.displacement_snapshots.len());
        let max_residual = transient.residual_norms.iter().copied().reduce(f64::max);
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

    let (
        increment_count,
        failed_increment_count,
        max_nonlinear_residual_norm,
        max_nonlinear_increment_norm,
        max_nonlinear_iteration_count,
        final_increment_converged,
        nonlinear_line_search_backtracks,
        nonlinear_max_backtracks_per_increment,
        nonlinear_tangent_rebuild_count,
        nonlinear_iteration_spike_count,
        nonlinear_convergence_stall_count,
        nonlinear_backtrack_burst_count,
    ) = if let Some(nonlinear) = run_result.nonlinear_results.as_ref() {
        let count = nonlinear.load_factors.len();
        let max_residual = nonlinear.residual_norms.iter().copied().reduce(f64::max);
        let max_increment_norm = nonlinear.increment_norms.iter().copied().reduce(f64::max);
        let max_iteration_count = nonlinear.iteration_counts.iter().copied().max();
        let final_converged =
            max_residual.map(|value| value <= 1.0e-6 && nonlinear.failed_increments == 0);
        (
            count,
            Some(nonlinear.failed_increments),
            max_residual,
            max_increment_norm,
            max_iteration_count,
            final_converged,
            Some(nonlinear.line_search_backtracks),
            Some(nonlinear.max_line_search_backtracks_per_increment),
            Some(nonlinear.tangent_rebuild_count),
            Some(nonlinear.iteration_spike_count),
            Some(nonlinear.convergence_stall_count),
            Some(nonlinear.backtrack_burst_count),
        )
    } else {
        (
            0, None, None, None, None, None, None, None, None, None, None, None,
        )
    };

    let prep_calibration_profile = diagnostic_metric_string(
        &run_result.run.diagnostics,
        "FEA_PREP_CALIBRATION",
        "profile",
    );
    let prep_calibration_fingerprint = diagnostic_metric_u64(
        &run_result.run.diagnostics,
        "FEA_PREP_CALIBRATION",
        "calibration_fingerprint",
    );
    let prep_acceptance_score = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_PREP_ACCEPTANCE",
        "acceptance_score",
    );
    let prep_acceptance_passed = diagnostic_metric_bool(
        &run_result.run.diagnostics,
        "FEA_PREP_ACCEPTANCE",
        "accepted",
    );
    let prep_acceptance_fingerprint = diagnostic_metric_u64(
        &run_result.run.diagnostics,
        "FEA_PREP_ACCEPTANCE",
        "acceptance_fingerprint",
    );
    let thermo_coupling_enabled =
        diagnostic_metric_bool(&run_result.run.diagnostics, "FEA_TM_COUPLING", "enabled");
    let thermo_coupling_fingerprint = diagnostic_metric_u64(
        &run_result.run.diagnostics,
        "FEA_TM_COUPLING",
        "coupling_fingerprint",
    );
    let thermo_constitutive_temperature_factor = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_TM_COUPLING",
        "constitutive_temperature_factor",
    );
    let thermo_effective_modulus_scale = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_TM_COUPLING",
        "effective_modulus_scale",
    );
    let thermo_constitutive_material_spread_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_TM_COUPLING",
        "constitutive_material_spread_ratio",
    );
    let thermo_assignment_heterogeneity_index = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_TM_COUPLING",
        "assignment_heterogeneity_index",
    );
    let thermo_region_delta_count = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_TM_COUPLING",
        "region_delta_count",
    );
    let thermo_spatial_coverage_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_TM_COUPLING",
        "spatial_coverage_ratio",
    );
    let thermo_field_extrapolation_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_TM_TRANSIENT",
        "field_extrapolation_ratio",
    )
    .or_else(|| {
        diagnostic_metric(
            &run_result.run.diagnostics,
            "FEA_TM_NONLINEAR",
            "field_extrapolation_ratio",
        )
    });
    let thermo_transient_severity = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_TM_TRANSIENT",
        "severity_peak",
    )
    .or_else(|| diagnostic_metric(&run_result.run.diagnostics, "FEA_TM_TRANSIENT", "severity"));
    let thermo_nonlinear_severity = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_TM_NONLINEAR",
        "severity_peak",
    )
    .or_else(|| diagnostic_metric(&run_result.run.diagnostics, "FEA_TM_NONLINEAR", "severity"));
    let electro_thermal_coupling_enabled =
        diagnostic_metric_bool(&run_result.run.diagnostics, "FEA_ET_COUPLING", "enabled");
    let electro_thermal_coupling_fingerprint = diagnostic_metric_u64(
        &run_result.run.diagnostics,
        "FEA_ET_COUPLING",
        "coupling_fingerprint",
    );
    let electro_joule_heating_scale = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_ET_COUPLING",
        "joule_heating_scale",
    );
    let electro_conductivity_spread_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_ET_COUPLING",
        "conductivity_spread_ratio",
    );
    let electro_transient_severity = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_ET_TRANSIENT",
        "severity_peak",
    )
    .or_else(|| diagnostic_metric(&run_result.run.diagnostics, "FEA_ET_TRANSIENT", "severity"));
    let electro_nonlinear_severity = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_ET_NONLINEAR",
        "severity_peak",
    )
    .or_else(|| diagnostic_metric(&run_result.run.diagnostics, "FEA_ET_NONLINEAR", "severity"));
    let plastic_nonlinear_severity = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_PLASTIC_NONLINEAR",
        "severity_peak",
    )
    .or_else(|| {
        diagnostic_metric(
            &run_result.run.diagnostics,
            "FEA_PLASTIC_NONLINEAR",
            "severity",
        )
    });
    let contact_nonlinear_severity = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_CONTACT_NONLINEAR",
        "severity_peak",
    )
    .or_else(|| {
        diagnostic_metric(
            &run_result.run.diagnostics,
            "FEA_CONTACT_NONLINEAR",
            "severity",
        )
    });

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
        failed_increment_count,
        max_nonlinear_residual_norm,
        max_nonlinear_increment_norm,
        max_nonlinear_iteration_count,
        final_increment_converged,
        nonlinear_line_search_backtracks,
        nonlinear_max_backtracks_per_increment,
        nonlinear_tangent_rebuild_count,
        nonlinear_iteration_spike_count,
        nonlinear_convergence_stall_count,
        nonlinear_backtrack_burst_count,
        prep_calibration_profile,
        prep_calibration_fingerprint,
        prep_acceptance_score,
        prep_acceptance_passed,
        prep_acceptance_fingerprint,
        thermo_coupling_enabled,
        thermo_coupling_fingerprint,
        thermo_constitutive_temperature_factor,
        thermo_effective_modulus_scale,
        thermo_constitutive_material_spread_ratio,
        thermo_assignment_heterogeneity_index,
        thermo_region_delta_count,
        thermo_spatial_coverage_ratio,
        thermo_field_extrapolation_ratio,
        thermo_transient_severity,
        thermo_nonlinear_severity,
        electro_thermal_coupling_enabled,
        electro_thermal_coupling_fingerprint,
        electro_joule_heating_scale,
        electro_conductivity_spread_ratio,
        electro_transient_severity,
        electro_nonlinear_severity,
        plastic_nonlinear_severity,
        contact_nonlinear_severity,
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
                            format!(
                                "requested modal mode index '{index}' is missing mode shape data"
                            ),
                            BTreeMap::from([
                                ("requested_mode_index".to_string(), index.to_string()),
                                (
                                    "available_shape_count".to_string(),
                                    modal.mode_shapes.len().to_string(),
                                ),
                            ]),
                        )
                    })?;
                    let residual_norm =
                        modal.residual_norms.get(index).copied().ok_or_else(|| {
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
            if query.diagnostic_codes.is_empty() {
                Some(run_result.run.diagnostics.clone())
            } else {
                Some(
                    run_result
                        .run
                        .diagnostics
                        .iter()
                        .filter(|diag| query.diagnostic_codes.iter().any(|code| code == &diag.code))
                        .cloned()
                        .collect(),
                )
            }
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

pub fn analysis_results_compare_op(
    query: AnalysisResultsCompareQuery,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisResultsCompareData>, OperationErrorEnvelope> {
    let baseline = storage::load_run_result(&query.baseline_run_id).map_err(|err| {
        operation_error(
            ANALYSIS_RESULTS_COMPARE_OPERATION,
            ANALYSIS_RESULTS_COMPARE_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "ANALYSIS_ARTIFACT_STORE_FAILED",
                error_type: OperationErrorType::Internal,
                retryable: true,
                severity: OperationErrorSeverity::Error,
            },
            format!("failed to load baseline analysis run artifact: {err}"),
            BTreeMap::from([("run_id".to_string(), query.baseline_run_id.clone())]),
        )
    })?;
    let Some(baseline) = baseline else {
        return Err(operation_error(
            ANALYSIS_RESULTS_COMPARE_OPERATION,
            ANALYSIS_RESULTS_COMPARE_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "ANALYSIS_RESULTS_RUN_NOT_FOUND",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            format!(
                "analysis baseline run_id '{}' was not found",
                query.baseline_run_id
            ),
            BTreeMap::from([("run_id".to_string(), query.baseline_run_id.clone())]),
        ));
    };

    let candidate = storage::load_run_result(&query.candidate_run_id).map_err(|err| {
        operation_error(
            ANALYSIS_RESULTS_COMPARE_OPERATION,
            ANALYSIS_RESULTS_COMPARE_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "ANALYSIS_ARTIFACT_STORE_FAILED",
                error_type: OperationErrorType::Internal,
                retryable: true,
                severity: OperationErrorSeverity::Error,
            },
            format!("failed to load candidate analysis run artifact: {err}"),
            BTreeMap::from([("run_id".to_string(), query.candidate_run_id.clone())]),
        )
    })?;
    let Some(candidate) = candidate else {
        return Err(operation_error(
            ANALYSIS_RESULTS_COMPARE_OPERATION,
            ANALYSIS_RESULTS_COMPARE_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "ANALYSIS_RESULTS_RUN_NOT_FOUND",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            format!(
                "analysis candidate run_id '{}' was not found",
                query.candidate_run_id
            ),
            BTreeMap::from([("run_id".to_string(), query.candidate_run_id.clone())]),
        ));
    };

    let baseline_solve_ms = run_solve_ms(&baseline);
    let candidate_solve_ms = run_solve_ms(&candidate);
    let failed_increment_delta = match (
        baseline.nonlinear_results.as_ref(),
        candidate.nonlinear_results.as_ref(),
    ) {
        (Some(a), Some(b)) => Some(b.failed_increments as i64 - a.failed_increments as i64),
        _ => None,
    };
    let max_iteration_delta = match (
        baseline.nonlinear_results.as_ref(),
        candidate.nonlinear_results.as_ref(),
    ) {
        (Some(a), Some(b)) => Some(
            b.iteration_counts.iter().copied().max().unwrap_or(0) as i64
                - a.iteration_counts.iter().copied().max().unwrap_or(0) as i64,
        ),
        _ => None,
    };
    let nonlinear_spike_count_delta = match (
        baseline.nonlinear_results.as_ref(),
        candidate.nonlinear_results.as_ref(),
    ) {
        (Some(a), Some(b)) => Some(b.iteration_spike_count as i64 - a.iteration_spike_count as i64),
        _ => None,
    };
    let nonlinear_stall_count_delta = match (
        baseline.nonlinear_results.as_ref(),
        candidate.nonlinear_results.as_ref(),
    ) {
        (Some(a), Some(b)) => {
            Some(b.convergence_stall_count as i64 - a.convergence_stall_count as i64)
        }
        _ => None,
    };

    let data = AnalysisResultsCompareData {
        baseline_run_id: baseline.run_id,
        candidate_run_id: candidate.run_id,
        publishable_changed: baseline.publishable != candidate.publishable,
        run_status_changed: baseline.run_status != candidate.run_status,
        quality_reason_count_delta: candidate.quality_reasons.len() as i64
            - baseline.quality_reasons.len() as i64,
        failed_increment_delta,
        max_iteration_delta,
        nonlinear_spike_count_delta,
        nonlinear_stall_count_delta,
        solve_ms_delta: match (baseline_solve_ms, candidate_solve_ms) {
            (Some(a), Some(b)) => Some(b - a),
            _ => None,
        },
    };

    Ok(OperationEnvelope::new(
        ANALYSIS_RESULTS_COMPARE_OPERATION,
        ANALYSIS_RESULTS_COMPARE_OP_VERSION,
        &context,
        data,
    ))
}

pub fn analysis_trends_op(
    query: AnalysisTrendsQuery,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisTrendsData>, OperationErrorEnvelope> {
    let runs = storage::list_run_results().map_err(|err| {
        operation_error(
            ANALYSIS_TRENDS_OPERATION,
            ANALYSIS_TRENDS_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "ANALYSIS_ARTIFACT_STORE_FAILED",
                error_type: OperationErrorType::Internal,
                retryable: true,
                severity: OperationErrorSeverity::Error,
            },
            format!("failed to list analysis run artifacts: {err}"),
            BTreeMap::new(),
        )
    })?;

    let mut grouped: HashMap<AnalysisRunKind, Vec<AnalysisRunResult>> = HashMap::new();
    for run in runs {
        grouped.entry(run_kind(&run)).or_default().push(run);
    }

    let window = query.window_size.max(1);
    let mut summaries = Vec::new();
    for kind in [
        AnalysisRunKind::LinearStatic,
        AnalysisRunKind::Modal,
        AnalysisRunKind::Transient,
        AnalysisRunKind::Nonlinear,
    ] {
        let Some(mut entries) = grouped.remove(&kind) else {
            continue;
        };
        entries.sort_by(|a, b| b.run_id.cmp(&a.run_id));
        if entries.len() > window {
            entries.truncate(window);
        }
        let sample_count = entries.len();
        if sample_count == 0 {
            continue;
        }

        let mut solve_samples = entries
            .iter()
            .filter_map(run_solve_ms)
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        solve_samples.sort_by(|a, b| a.total_cmp(b));
        let median_solve_ms = percentile(&solve_samples, 0.5);
        let p95_solve_ms = percentile(&solve_samples, 0.95);
        let publishable_rate =
            entries.iter().filter(|run| run.publishable).count() as f64 / sample_count as f64;

        let failed_increment_rate = if kind == AnalysisRunKind::Nonlinear {
            let failed = entries
                .iter()
                .filter_map(|run| run.nonlinear_results.as_ref())
                .filter(|nonlinear| nonlinear.failed_increments > 0)
                .count();
            Some(failed as f64 / sample_count as f64)
        } else {
            None
        };
        let mean_spike_count = if kind == AnalysisRunKind::Nonlinear {
            let values = entries
                .iter()
                .filter_map(|run| run.nonlinear_results.as_ref())
                .map(|nonlinear| nonlinear.iteration_spike_count as f64)
                .collect::<Vec<_>>();
            Some(mean(&values))
        } else {
            None
        };
        let mean_stall_count = if kind == AnalysisRunKind::Nonlinear {
            let values = entries
                .iter()
                .filter_map(|run| run.nonlinear_results.as_ref())
                .map(|nonlinear| nonlinear.convergence_stall_count as f64)
                .collect::<Vec<_>>();
            Some(mean(&values))
        } else {
            None
        };
        let prep_acceptance_rate = {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric_bool(&run.run.diagnostics, "FEA_PREP_ACCEPTANCE", "accepted")
                })
                .collect::<Vec<_>>();
            if values.is_empty() {
                None
            } else {
                Some(values.iter().filter(|value| **value).count() as f64 / values.len() as f64)
            }
        };
        let prep_calibration_fast_rate = calibration_profile_rate(&entries, "fast");
        let prep_calibration_balanced_rate = calibration_profile_rate(&entries, "balanced");
        let prep_calibration_conservative_rate = calibration_profile_rate(&entries, "conservative");
        let thermo_coupling_enabled_rate = {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric_bool(&run.run.diagnostics, "FEA_TM_COUPLING", "enabled")
                })
                .collect::<Vec<_>>();
            if values.is_empty() {
                None
            } else {
                Some(values.iter().filter(|value| **value).count() as f64 / values.len() as f64)
            }
        };
        let thermo_transient_warn_rate = if kind == AnalysisRunKind::Transient {
            diagnostic_warning_rate(&entries, "FEA_TM_TRANSIENT")
        } else {
            None
        };
        let thermo_nonlinear_warn_rate = if kind == AnalysisRunKind::Nonlinear {
            diagnostic_warning_rate(&entries, "FEA_TM_NONLINEAR")
        } else {
            None
        };
        let thermo_spread_breach_rate = {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(
                        &run.run.diagnostics,
                        "FEA_TM_COUPLING",
                        "constitutive_material_spread_ratio",
                    )
                })
                .collect::<Vec<_>>();
            if values.is_empty() {
                None
            } else {
                Some(
                    values
                        .iter()
                        .filter(|value| **value > THERMO_SPREAD_THRESHOLD_BALANCED)
                        .count() as f64
                        / values.len() as f64,
                )
            }
        };
        let thermo_heterogeneity_breach_rate = {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(
                        &run.run.diagnostics,
                        "FEA_TM_COUPLING",
                        "assignment_heterogeneity_index",
                    )
                })
                .collect::<Vec<_>>();
            if values.is_empty() {
                None
            } else {
                Some(
                    values
                        .iter()
                        .filter(|value| **value > THERMO_HETEROGENEITY_THRESHOLD_BALANCED)
                        .count() as f64
                        / values.len() as f64,
                )
            }
        };
        let electro_thermal_coupling_enabled_rate = {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric_bool(&run.run.diagnostics, "FEA_ET_COUPLING", "enabled")
                })
                .collect::<Vec<_>>();
            if values.is_empty() {
                None
            } else {
                Some(values.iter().filter(|value| **value).count() as f64 / values.len() as f64)
            }
        };
        let electro_transient_warn_rate = if kind == AnalysisRunKind::Transient {
            diagnostic_warning_rate(&entries, "FEA_ET_TRANSIENT")
        } else {
            None
        };
        let electro_nonlinear_warn_rate = if kind == AnalysisRunKind::Nonlinear {
            diagnostic_warning_rate(&entries, "FEA_ET_NONLINEAR")
        } else {
            None
        };
        let plastic_nonlinear_warn_rate = if kind == AnalysisRunKind::Nonlinear {
            diagnostic_warning_rate(&entries, "FEA_PLASTIC_NONLINEAR")
        } else {
            None
        };
        let contact_nonlinear_warn_rate = if kind == AnalysisRunKind::Nonlinear {
            diagnostic_warning_rate(&entries, "FEA_CONTACT_NONLINEAR")
        } else {
            None
        };

        summaries.push(AnalysisTrendKindSummary {
            run_kind: kind,
            sample_count,
            median_solve_ms,
            p95_solve_ms,
            publishable_rate,
            failed_increment_rate,
            mean_spike_count,
            mean_stall_count,
            prep_acceptance_rate,
            prep_calibration_fast_rate,
            prep_calibration_balanced_rate,
            prep_calibration_conservative_rate,
            thermo_coupling_enabled_rate,
            thermo_transient_warn_rate,
            thermo_nonlinear_warn_rate,
            thermo_spread_breach_rate,
            thermo_heterogeneity_breach_rate,
            electro_thermal_coupling_enabled_rate,
            electro_transient_warn_rate,
            electro_nonlinear_warn_rate,
            plastic_nonlinear_warn_rate,
            contact_nonlinear_warn_rate,
        });
    }

    Ok(OperationEnvelope::new(
        ANALYSIS_TRENDS_OPERATION,
        ANALYSIS_TRENDS_OP_VERSION,
        &context,
        AnalysisTrendsData {
            window_size: window,
            summaries,
        },
    ))
}

fn run_kind(run: &AnalysisRunResult) -> AnalysisRunKind {
    if run.nonlinear_results.is_some() {
        AnalysisRunKind::Nonlinear
    } else if run.transient_results.is_some() {
        AnalysisRunKind::Transient
    } else if run.modal_results.is_some() {
        AnalysisRunKind::Modal
    } else {
        AnalysisRunKind::LinearStatic
    }
}

fn to_fea_prep_context(
    context: Option<AnalysisRunPrepContext>,
    calibration_profile: Option<PrepCalibrationProfile>,
) -> Option<runmat_analysis_fea::FeaPrepContext> {
    context.map(|prep| runmat_analysis_fea::FeaPrepContext {
        prepared_mesh_count: prep.prepared_mesh_count,
        prepared_node_count: prep.prepared_node_count,
        prepared_element_count: prep.prepared_element_count,
        mapped_region_count: prep.mapped_region_count,
        min_scaled_jacobian: prep.min_scaled_jacobian,
        mean_aspect_ratio: prep.mean_aspect_ratio,
        inverted_element_count: prep.inverted_element_count,
        mapped_load_count: prep.mapped_load_count,
        mapped_bc_count: prep.mapped_bc_count,
        layout_seed: prep.layout_seed,
        topology_dof_multiplier: prep.topology_dof_multiplier,
        topology_bandwidth_proxy: prep.topology_bandwidth_proxy,
        mapped_region_participation_ratio: prep.mapped_region_participation_ratio,
        topology_surface_patch_ratio: prep.topology_surface_patch_ratio,
        topology_volume_core_ratio: prep.topology_volume_core_ratio,
        topology_mixed_family_ratio: prep.topology_mixed_family_ratio,
        topology_region_span_mean: prep.topology_region_span_mean,
        topology_region_block_count: prep.topology_region_block_count,
        topology_region_mesh_mean: prep.topology_region_mesh_mean,
        topology_region_mesh_variance: prep.topology_region_mesh_variance,
        topology_triangle_family_ratio: prep.topology_triangle_family_ratio,
        topology_quad_family_ratio: prep.topology_quad_family_ratio,
        topology_tet_family_ratio: prep.topology_tet_family_ratio,
        topology_hex_family_ratio: prep.topology_hex_family_ratio,
        calibration_profile_override: calibration_profile.and_then(map_calibration_profile),
    })
}

fn map_calibration_profile(
    profile: PrepCalibrationProfile,
) -> Option<runmat_analysis_fea::FeaPrepCalibrationProfile> {
    match profile {
        PrepCalibrationProfile::Auto => None,
        PrepCalibrationProfile::Fast => Some(runmat_analysis_fea::FeaPrepCalibrationProfile::Fast),
        PrepCalibrationProfile::Balanced => {
            Some(runmat_analysis_fea::FeaPrepCalibrationProfile::Balanced)
        }
        PrepCalibrationProfile::Conservative => {
            Some(runmat_analysis_fea::FeaPrepCalibrationProfile::Conservative)
        }
    }
}

fn to_fea_thermo_mechanical_context(
    options: Option<ThermoMechanicalCouplingOptions>,
) -> Option<runmat_analysis_fea::FeaThermoMechanicalContext> {
    options.map(|tm| runmat_analysis_fea::FeaThermoMechanicalContext {
        enabled: tm.enabled,
        reference_temperature_k: tm.reference_temperature_k,
        applied_temperature_delta_k: tm.applied_temperature_delta_k,
        thermal_expansion_coefficient: tm.thermal_expansion_coefficient,
        field_source: tm
            .field_source
            .map(|source| runmat_analysis_fea::FeaThermoFieldSource {
                source_id: source.source_id,
                revision: source.revision,
                interpolation_mode: source.interpolation_mode.map(|mode| match mode {
                    contracts::ThermoFieldInterpolationMode::Linear => {
                        runmat_analysis_fea::FeaThermoFieldInterpolationMode::Linear
                    }
                    contracts::ThermoFieldInterpolationMode::Step => {
                        runmat_analysis_fea::FeaThermoFieldInterpolationMode::Step
                    }
                }),
                expected_region_ids: source.expected_region_ids,
            }),
        region_temperature_deltas: tm
            .region_temperature_deltas
            .into_iter()
            .map(
                |ThermoRegionTemperatureDelta {
                     region_id,
                     temperature_delta_k,
                 }| runmat_analysis_fea::FeaThermoRegionTemperatureDelta {
                    region_id,
                    temperature_delta_k,
                },
            )
            .collect(),
        time_profile: tm
            .time_profile
            .into_iter()
            .map(
                |ThermoTimeProfilePoint {
                     normalized_time,
                     scale,
                 }| runmat_analysis_fea::FeaThermoTimeProfilePoint {
                    normalized_time,
                    scale,
                },
            )
            .collect(),
    })
}

fn to_fea_electro_thermal_context(
    options: Option<ElectroThermalCouplingOptions>,
) -> Option<runmat_analysis_fea::FeaElectroThermalContext> {
    options.map(|et| runmat_analysis_fea::FeaElectroThermalContext {
        enabled: et.enabled,
        reference_temperature_k: et.reference_temperature_k,
        applied_voltage_v: et.applied_voltage_v,
        base_electrical_conductivity_s_per_m: et.base_electrical_conductivity_s_per_m,
        resistive_heating_coefficient: et.resistive_heating_coefficient,
        region_conductivity_scales: et
            .region_conductivity_scales
            .into_iter()
            .map(
                |ElectroRegionConductivityScale {
                     region_id,
                     conductivity_scale,
                 }| runmat_analysis_fea::FeaElectroRegionConductivityScale {
                    region_id,
                    conductivity_scale,
                },
            )
            .collect(),
        time_profile: et
            .time_profile
            .into_iter()
            .map(
                |ElectroTimeProfilePoint {
                     normalized_time,
                     current_scale,
                 }| runmat_analysis_fea::FeaElectroTimeProfilePoint {
                    normalized_time,
                    current_scale,
                },
            )
            .collect(),
    })
}

fn to_fea_plasticity_proxy_context(
    options: Option<PlasticityProxyOptions>,
) -> Option<runmat_analysis_fea::FeaPlasticityProxyContext> {
    options.map(|plasticity| runmat_analysis_fea::FeaPlasticityProxyContext {
        enabled: plasticity.enabled,
        yield_strain: plasticity.yield_strain,
        hardening_modulus_ratio: plasticity.hardening_modulus_ratio,
        saturation_exponent: plasticity.saturation_exponent,
    })
}

fn to_fea_contact_proxy_context(
    options: Option<ContactProxyOptions>,
) -> Option<runmat_analysis_fea::FeaContactProxyContext> {
    options.map(|contact| runmat_analysis_fea::FeaContactProxyContext {
        enabled: contact.enabled,
        penalty_stiffness_scale: contact.penalty_stiffness_scale,
        max_penetration_ratio: contact.max_penetration_ratio,
        friction_coefficient: contact.friction_coefficient,
    })
}

fn validate_thermo_coupling_options(
    model: &AnalysisModel,
    options: &ThermoMechanicalCouplingOptions,
) -> Result<(), (String, BTreeMap<String, String>)> {
    if !options.enabled {
        return Ok(());
    }
    if !options.reference_temperature_k.is_finite() || options.reference_temperature_k <= 0.0 {
        return Err((
            "thermo coupling requires finite positive reference_temperature_k".to_string(),
            BTreeMap::from([(
                "reference_temperature_k".to_string(),
                options.reference_temperature_k.to_string(),
            )]),
        ));
    }
    if !options.applied_temperature_delta_k.is_finite() {
        return Err((
            "thermo coupling requires finite applied_temperature_delta_k".to_string(),
            BTreeMap::from([(
                "applied_temperature_delta_k".to_string(),
                options.applied_temperature_delta_k.to_string(),
            )]),
        ));
    }
    if !options.thermal_expansion_coefficient.is_finite()
        || options.thermal_expansion_coefficient < 0.0
    {
        return Err((
            "thermo coupling requires finite non-negative thermal_expansion_coefficient"
                .to_string(),
            BTreeMap::from([(
                "thermal_expansion_coefficient".to_string(),
                options.thermal_expansion_coefficient.to_string(),
            )]),
        ));
    }

    let mut last_t = -1.0_f64;
    for (idx, point) in options.time_profile.iter().enumerate() {
        if !point.normalized_time.is_finite()
            || point.normalized_time < 0.0
            || point.normalized_time > 1.0
        {
            return Err((
                "thermo time_profile normalized_time must be finite and within [0, 1]".to_string(),
                BTreeMap::from([
                    ("time_profile_index".to_string(), idx.to_string()),
                    (
                        "normalized_time".to_string(),
                        point.normalized_time.to_string(),
                    ),
                ]),
            ));
        }
        if !point.scale.is_finite() {
            return Err((
                "thermo time_profile scale must be finite".to_string(),
                BTreeMap::from([
                    ("time_profile_index".to_string(), idx.to_string()),
                    ("scale".to_string(), point.scale.to_string()),
                ]),
            ));
        }
        if point.normalized_time + 1.0e-12 < last_t {
            return Err((
                "thermo time_profile normalized_time must be monotonic non-decreasing".to_string(),
                BTreeMap::from([
                    ("time_profile_index".to_string(), idx.to_string()),
                    (
                        "normalized_time".to_string(),
                        point.normalized_time.to_string(),
                    ),
                ]),
            ));
        }
        last_t = point.normalized_time;
    }

    let model_region_ids = model
        .material_assignments
        .iter()
        .map(|assignment| assignment.region_id.as_str())
        .collect::<HashSet<_>>();

    for delta in &options.region_temperature_deltas {
        if !delta.temperature_delta_k.is_finite() {
            return Err((
                "thermo region_temperature_deltas must use finite temperature_delta_k".to_string(),
                BTreeMap::from([
                    ("region_id".to_string(), delta.region_id.clone()),
                    (
                        "temperature_delta_k".to_string(),
                        delta.temperature_delta_k.to_string(),
                    ),
                ]),
            ));
        }
    }

    if let Some(source) = options.field_source.as_ref() {
        if source.source_id.trim().is_empty() {
            return Err((
                "thermo field_source requires a non-empty source_id".to_string(),
                BTreeMap::new(),
            ));
        }
        for expected_region in &source.expected_region_ids {
            if !model_region_ids.contains(expected_region.as_str()) {
                return Err((
                    "thermo field_source expected_region_ids must exist in model material assignments"
                        .to_string(),
                    BTreeMap::from([("region_id".to_string(), expected_region.clone())]),
                ));
            }
        }
    }

    Ok(())
}

fn validate_electro_coupling_options(
    model: &AnalysisModel,
    options: &ElectroThermalCouplingOptions,
) -> Result<(), (String, BTreeMap<String, String>)> {
    if !options.enabled {
        return Ok(());
    }
    if !options.reference_temperature_k.is_finite() || options.reference_temperature_k <= 0.0 {
        return Err((
            "electro-thermal coupling requires finite positive reference_temperature_k".to_string(),
            BTreeMap::from([(
                "reference_temperature_k".to_string(),
                options.reference_temperature_k.to_string(),
            )]),
        ));
    }
    if !options.applied_voltage_v.is_finite() {
        return Err((
            "electro-thermal coupling requires finite applied_voltage_v".to_string(),
            BTreeMap::from([(
                "applied_voltage_v".to_string(),
                options.applied_voltage_v.to_string(),
            )]),
        ));
    }
    if !options.base_electrical_conductivity_s_per_m.is_finite()
        || options.base_electrical_conductivity_s_per_m <= 0.0
    {
        return Err((
            "electro-thermal coupling requires finite positive base_electrical_conductivity_s_per_m"
                .to_string(),
            BTreeMap::from([(
                "base_electrical_conductivity_s_per_m".to_string(),
                options.base_electrical_conductivity_s_per_m.to_string(),
            )]),
        ));
    }
    if !options.resistive_heating_coefficient.is_finite()
        || options.resistive_heating_coefficient < 0.0
    {
        return Err((
            "electro-thermal coupling requires finite non-negative resistive_heating_coefficient"
                .to_string(),
            BTreeMap::from([(
                "resistive_heating_coefficient".to_string(),
                options.resistive_heating_coefficient.to_string(),
            )]),
        ));
    }
    let mut last_t = -1.0_f64;
    for (idx, point) in options.time_profile.iter().enumerate() {
        if !point.normalized_time.is_finite()
            || point.normalized_time < 0.0
            || point.normalized_time > 1.0
        {
            return Err((
                "electro time_profile normalized_time must be finite and within [0, 1]".to_string(),
                BTreeMap::from([
                    ("time_profile_index".to_string(), idx.to_string()),
                    (
                        "normalized_time".to_string(),
                        point.normalized_time.to_string(),
                    ),
                ]),
            ));
        }
        if !point.current_scale.is_finite() {
            return Err((
                "electro time_profile current_scale must be finite".to_string(),
                BTreeMap::from([
                    ("time_profile_index".to_string(), idx.to_string()),
                    ("current_scale".to_string(), point.current_scale.to_string()),
                ]),
            ));
        }
        if point.normalized_time + 1.0e-12 < last_t {
            return Err((
                "electro time_profile normalized_time must be monotonic non-decreasing".to_string(),
                BTreeMap::from([
                    ("time_profile_index".to_string(), idx.to_string()),
                    (
                        "normalized_time".to_string(),
                        point.normalized_time.to_string(),
                    ),
                ]),
            ));
        }
        last_t = point.normalized_time;
    }
    let model_region_ids = model
        .material_assignments
        .iter()
        .map(|assignment| assignment.region_id.as_str())
        .collect::<HashSet<_>>();
    for scale in &options.region_conductivity_scales {
        if !scale.conductivity_scale.is_finite() || scale.conductivity_scale <= 0.0 {
            return Err((
                "electro region_conductivity_scales must use finite positive conductivity_scale"
                    .to_string(),
                BTreeMap::from([
                    ("region_id".to_string(), scale.region_id.clone()),
                    (
                        "conductivity_scale".to_string(),
                        scale.conductivity_scale.to_string(),
                    ),
                ]),
            ));
        }
        if !model_region_ids.is_empty() && !model_region_ids.contains(scale.region_id.as_str()) {
            return Err((
                "electro region_conductivity_scales region_id must exist in model material assignments"
                    .to_string(),
                BTreeMap::from([("region_id".to_string(), scale.region_id.clone())]),
            ));
        }
    }
    Ok(())
}

fn validate_plasticity_proxy_options(
    options: &PlasticityProxyOptions,
) -> Result<(), (String, BTreeMap<String, String>)> {
    if !options.enabled {
        return Ok(());
    }
    if !options.yield_strain.is_finite() || options.yield_strain <= 0.0 {
        return Err((
            "plasticity proxy requires finite positive yield_strain".to_string(),
            BTreeMap::from([("yield_strain".to_string(), options.yield_strain.to_string())]),
        ));
    }
    if !options.hardening_modulus_ratio.is_finite() || options.hardening_modulus_ratio < 0.0 {
        return Err((
            "plasticity proxy requires finite non-negative hardening_modulus_ratio".to_string(),
            BTreeMap::from([(
                "hardening_modulus_ratio".to_string(),
                options.hardening_modulus_ratio.to_string(),
            )]),
        ));
    }
    if !options.saturation_exponent.is_finite() || options.saturation_exponent < 0.0 {
        return Err((
            "plasticity proxy requires finite non-negative saturation_exponent".to_string(),
            BTreeMap::from([(
                "saturation_exponent".to_string(),
                options.saturation_exponent.to_string(),
            )]),
        ));
    }
    Ok(())
}

fn validate_contact_proxy_options(
    options: &ContactProxyOptions,
) -> Result<(), (String, BTreeMap<String, String>)> {
    if !options.enabled {
        return Ok(());
    }
    if !options.penalty_stiffness_scale.is_finite() || options.penalty_stiffness_scale <= 0.0 {
        return Err((
            "contact proxy requires finite positive penalty_stiffness_scale".to_string(),
            BTreeMap::from([(
                "penalty_stiffness_scale".to_string(),
                options.penalty_stiffness_scale.to_string(),
            )]),
        ));
    }
    if !options.max_penetration_ratio.is_finite() || options.max_penetration_ratio < 0.0 {
        return Err((
            "contact proxy requires finite non-negative max_penetration_ratio".to_string(),
            BTreeMap::from([(
                "max_penetration_ratio".to_string(),
                options.max_penetration_ratio.to_string(),
            )]),
        ));
    }
    if !options.friction_coefficient.is_finite() || options.friction_coefficient < 0.0 {
        return Err((
            "contact proxy requires finite non-negative friction_coefficient".to_string(),
            BTreeMap::from([(
                "friction_coefficient".to_string(),
                options.friction_coefficient.to_string(),
            )]),
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Deserialize)]
struct ThermoFieldArtifact {
    schema_version: String,
    source_geometry_id: String,
    source_geometry_revision: u32,
    #[serde(default)]
    artifact_status: Option<String>,
    #[serde(default)]
    approved_by: Option<String>,
    #[serde(default)]
    payload_hash: Option<String>,
    #[serde(default)]
    signature: Option<String>,
    #[serde(default)]
    field_source: Option<ThermoFieldSource>,
    #[serde(default)]
    region_temperature_deltas: Vec<ThermoRegionTemperatureDelta>,
    #[serde(default)]
    time_profile: Vec<ThermoTimeProfilePoint>,
}

fn thermo_field_payload_hash(artifact: &ThermoFieldArtifact) -> String {
    let source = artifact.field_source.as_ref();
    let source_id = source.map(|s| s.source_id.as_str()).unwrap_or("");
    let source_revision = source.map(|s| s.revision).unwrap_or(0);
    let interpolation = source
        .and_then(|s| s.interpolation_mode)
        .map(|mode| match mode {
            ThermoFieldInterpolationMode::Linear => "linear",
            ThermoFieldInterpolationMode::Step => "step",
        })
        .unwrap_or("");
    let expected_regions = source
        .map(|s| s.expected_region_ids.join(","))
        .unwrap_or_default();
    let region_terms = artifact
        .region_temperature_deltas
        .iter()
        .map(|delta| {
            format!(
                "{}:{:016x}",
                delta.region_id,
                delta.temperature_delta_k.to_bits()
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let time_terms = artifact
        .time_profile
        .iter()
        .map(|point| {
            format!(
                "{:016x}:{:016x}",
                point.normalized_time.to_bits(),
                point.scale.to_bits()
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let canonical = format!(
        "{}|{}|{}|{}|{}|{}|{}|{}|{}",
        artifact.schema_version,
        artifact.source_geometry_id,
        artifact.source_geometry_revision,
        source_id,
        source_revision,
        interpolation,
        expected_regions,
        region_terms,
        time_terms
    );
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    format!("sha256:{:x}", hasher.finalize())
}

fn thermo_field_signature(payload_hash: &str, approved_by: &str, signing_key: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(format!("{payload_hash}:{approved_by}:{signing_key}").as_bytes());
    format!("sigv1:sha256:{:x}", hasher.finalize())
}

fn resolve_thermo_coupling_options(
    model: &AnalysisModel,
    options: Option<ThermoMechanicalCouplingOptions>,
    operation: &'static str,
    op_version: &'static str,
    context: &OperationContext,
) -> Result<Option<ThermoMechanicalCouplingOptions>, OperationErrorEnvelope> {
    let Some(mut options) = options else {
        return Ok(None);
    };
    let Some(field_artifact_id) = options.field_artifact_id.as_deref() else {
        return Ok(Some(options));
    };

    let root = PathBuf::from(
        std::env::var("RUNMAT_THERMO_FIELD_ARTIFACT_ROOT")
            .unwrap_or_else(|_| "target/runmat-analysis-artifacts/thermo-fields".to_string()),
    );
    let path = root.join(format!("{field_artifact_id}.json"));
    if !path.exists() {
        return Err(operation_error(
            operation,
            op_version,
            context,
            OperationErrorSpec {
                error_code: "ANALYSIS_RUN_THERMO_FIELD_NOT_FOUND",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            format!(
                "thermo field artifact '{}' was not found",
                field_artifact_id
            ),
            BTreeMap::from([
                (
                    "thermo_field_artifact_id".to_string(),
                    field_artifact_id.to_string(),
                ),
                (
                    "thermo_field_artifact_path".to_string(),
                    path.display().to_string(),
                ),
            ]),
        ));
    }

    let raw = fs::read_to_string(&path).map_err(|err| {
        operation_error(
            operation,
            op_version,
            context,
            OperationErrorSpec {
                error_code: "ANALYSIS_RUN_THERMO_FIELD_STORE_FAILED",
                error_type: OperationErrorType::Internal,
                retryable: true,
                severity: OperationErrorSeverity::Error,
            },
            format!("failed to read thermo field artifact: {err}"),
            BTreeMap::from([(
                "thermo_field_artifact_path".to_string(),
                path.display().to_string(),
            )]),
        )
    })?;
    let artifact: ThermoFieldArtifact = serde_json::from_str(&raw).map_err(|err| {
        operation_error(
            operation,
            op_version,
            context,
            OperationErrorSpec {
                error_code: "ANALYSIS_RUN_THERMO_FIELD_INVALID",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            format!("invalid thermo field artifact payload: {err}"),
            BTreeMap::from([(
                "thermo_field_artifact_path".to_string(),
                path.display().to_string(),
            )]),
        )
    })?;

    if artifact.schema_version != "analysis_thermo_field_artifact/v1" {
        return Err(operation_error(
            operation,
            op_version,
            context,
            OperationErrorSpec {
                error_code: "ANALYSIS_RUN_THERMO_FIELD_SCHEMA_UNSUPPORTED",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            format!(
                "thermo field artifact schema '{}' is not supported",
                artifact.schema_version
            ),
            BTreeMap::from([
                (
                    "thermo_field_artifact_id".to_string(),
                    field_artifact_id.to_string(),
                ),
                (
                    "thermo_field_artifact_schema".to_string(),
                    artifact.schema_version.clone(),
                ),
            ]),
        ));
    }

    if artifact.source_geometry_id != model.geometry_id
        || artifact.source_geometry_revision != model.geometry_revision
    {
        return Err(operation_error(
            operation,
            op_version,
            context,
            OperationErrorSpec {
                error_code: "ANALYSIS_RUN_THERMO_FIELD_MISMATCH",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "thermo field artifact geometry lineage does not match analysis model",
            BTreeMap::from([
                (
                    "thermo_field_artifact_id".to_string(),
                    field_artifact_id.to_string(),
                ),
                ("model_geometry_id".to_string(), model.geometry_id.clone()),
                (
                    "model_geometry_revision".to_string(),
                    model.geometry_revision.to_string(),
                ),
                (
                    "artifact_geometry_id".to_string(),
                    artifact.source_geometry_id.clone(),
                ),
                (
                    "artifact_geometry_revision".to_string(),
                    artifact.source_geometry_revision.to_string(),
                ),
            ]),
        ));
    }

    let expected_hash = thermo_field_payload_hash(&artifact);
    if artifact.payload_hash.as_deref() != Some(expected_hash.as_str()) {
        return Err(operation_error(
            operation,
            op_version,
            context,
            OperationErrorSpec {
                error_code: "ANALYSIS_RUN_THERMO_FIELD_DIGEST_MISMATCH",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "thermo field artifact payload hash does not match payload contents",
            BTreeMap::from([
                (
                    "thermo_field_artifact_id".to_string(),
                    field_artifact_id.to_string(),
                ),
                ("expected_payload_hash".to_string(), expected_hash),
                (
                    "artifact_payload_hash".to_string(),
                    artifact.payload_hash.clone().unwrap_or_default(),
                ),
            ]),
        ));
    }

    if matches!(artifact.artifact_status.as_deref(), Some("approved")) {
        let Some(approved_by) = artifact.approved_by.as_deref() else {
            return Err(operation_error(
                operation,
                op_version,
                context,
                OperationErrorSpec {
                    error_code: "ANALYSIS_RUN_THERMO_FIELD_APPROVER_MISSING",
                    error_type: OperationErrorType::Validation,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                "approved thermo field artifact is missing approved_by",
                BTreeMap::from([(
                    "thermo_field_artifact_id".to_string(),
                    field_artifact_id.to_string(),
                )]),
            ));
        };

        let allowed = std::env::var("RUNMAT_THERMO_FIELD_ALLOWED_APPROVERS")
            .ok()
            .map(|value| {
                value
                    .split(',')
                    .map(|entry| entry.trim().to_string())
                    .filter(|entry| !entry.is_empty())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if !allowed.is_empty() && !allowed.iter().any(|entry| entry == approved_by) {
            return Err(operation_error(
                operation,
                op_version,
                context,
                OperationErrorSpec {
                    error_code: "ANALYSIS_RUN_THERMO_FIELD_APPROVER_UNAUTHORIZED",
                    error_type: OperationErrorType::Validation,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                "thermo field artifact approver is not authorized",
                BTreeMap::from([
                    (
                        "thermo_field_artifact_id".to_string(),
                        field_artifact_id.to_string(),
                    ),
                    ("approved_by".to_string(), approved_by.to_string()),
                ]),
            ));
        }

        let signing_key = std::env::var("RUNMAT_THERMO_FIELD_SIGNING_KEY")
            .unwrap_or_else(|_| "runmat-dev-thermo-signing-key".to_string());
        let expected_signature = thermo_field_signature(&expected_hash, approved_by, &signing_key);
        if artifact.signature.as_deref() != Some(expected_signature.as_str()) {
            return Err(operation_error(
                operation,
                op_version,
                context,
                OperationErrorSpec {
                    error_code: "ANALYSIS_RUN_THERMO_FIELD_SIGNATURE_INVALID",
                    error_type: OperationErrorType::Validation,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                "thermo field artifact signature validation failed",
                BTreeMap::from([
                    (
                        "thermo_field_artifact_id".to_string(),
                        field_artifact_id.to_string(),
                    ),
                    ("expected_signature".to_string(), expected_signature),
                    (
                        "artifact_signature".to_string(),
                        artifact.signature.clone().unwrap_or_default(),
                    ),
                ]),
            ));
        }
    }

    options.field_source = artifact.field_source;
    options.region_temperature_deltas = artifact.region_temperature_deltas;
    options.time_profile = artifact.time_profile;

    Ok(Some(options))
}

fn resolve_run_prep_context(
    model: &AnalysisModel,
    prep_artifact_id: Option<&str>,
    legacy_prep_context: Option<AnalysisRunPrepContext>,
    operation: &'static str,
    op_version: &'static str,
    context: &OperationContext,
) -> Result<Option<AnalysisRunPrepContext>, OperationErrorEnvelope> {
    if prep_artifact_id.is_none() {
        if legacy_prep_context.is_some() {
            return Err(operation_error(
                operation,
                op_version,
                context,
                OperationErrorSpec {
                    error_code: "ANALYSIS_RUN_PREP_UNTRUSTED_CONTEXT",
                    error_type: OperationErrorType::Input,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                "analysis run prep_context must be referenced by prep_artifact_id",
                BTreeMap::from([("analysis_model_id".to_string(), model.model_id.0.clone())]),
            ));
        }
        return Ok(None);
    }

    let prep_artifact_id = prep_artifact_id.expect("checked is_some");
    let Some(artifact) = crate::geometry::load_prep_artifact(prep_artifact_id).map_err(|err| {
        operation_error(
            operation,
            op_version,
            context,
            OperationErrorSpec {
                error_code: "ANALYSIS_RUN_PREP_STORE_FAILED",
                error_type: OperationErrorType::Internal,
                retryable: true,
                severity: OperationErrorSeverity::Error,
            },
            format!("failed to load prep artifact: {err}"),
            BTreeMap::from([("prep_artifact_id".to_string(), prep_artifact_id.to_string())]),
        )
    })?
    else {
        return Err(operation_error(
            operation,
            op_version,
            context,
            OperationErrorSpec {
                error_code: "ANALYSIS_RUN_PREP_NOT_FOUND",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            format!("prep artifact '{}' was not found", prep_artifact_id),
            BTreeMap::from([("prep_artifact_id".to_string(), prep_artifact_id.to_string())]),
        ));
    };

    if artifact.schema_version != "geometry_prep_artifact/v1" {
        return Err(operation_error(
            operation,
            op_version,
            context,
            OperationErrorSpec {
                error_code: "ANALYSIS_RUN_PREP_SCHEMA_UNSUPPORTED",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            format!(
                "prep artifact schema '{}' is not supported",
                artifact.schema_version
            ),
            BTreeMap::from([("prep_artifact_id".to_string(), prep_artifact_id.to_string())]),
        ));
    }

    if artifact.source_geometry_id != model.geometry_id
        || artifact.source_geometry_revision != model.geometry_revision
    {
        crate::geometry::record_prep_mismatch_reject();
        return Err(operation_error(
            operation,
            op_version,
            context,
            OperationErrorSpec {
                error_code: "ANALYSIS_RUN_PREP_MISMATCH",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "prep artifact geometry lineage does not match analysis model",
            BTreeMap::from([
                ("prep_artifact_id".to_string(), prep_artifact_id.to_string()),
                ("model_geometry_id".to_string(), model.geometry_id.clone()),
                (
                    "model_geometry_revision".to_string(),
                    model.geometry_revision.to_string(),
                ),
                (
                    "prep_geometry_id".to_string(),
                    artifact.source_geometry_id.clone(),
                ),
                (
                    "prep_geometry_revision".to_string(),
                    artifact.source_geometry_revision.to_string(),
                ),
            ]),
        ));
    }

    let require_latest_revision = std::env::var("RUNMAT_GEOMETRY_PREP_REQUIRE_LATEST_REVISION")
        .ok()
        .map(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(true);
    if require_latest_revision {
        if let Some(latest_revision) = crate::geometry::latest_prep_revision_for_geometry(
            &model.geometry_id,
        )
        .map_err(|err| {
            operation_error(
                operation,
                op_version,
                context,
                OperationErrorSpec {
                    error_code: "ANALYSIS_RUN_PREP_STORE_FAILED",
                    error_type: OperationErrorType::Internal,
                    retryable: true,
                    severity: OperationErrorSeverity::Error,
                },
                format!("failed to evaluate prep artifact freshness: {err}"),
                BTreeMap::from([("prep_artifact_id".to_string(), prep_artifact_id.to_string())]),
            )
        })? {
            if artifact.source_geometry_revision < latest_revision {
                crate::geometry::record_prep_stale_reject();
                return Err(operation_error(
                    operation,
                    op_version,
                    context,
                    OperationErrorSpec {
                        error_code: "ANALYSIS_RUN_PREP_STALE",
                        error_type: OperationErrorType::Validation,
                        retryable: false,
                        severity: OperationErrorSeverity::Error,
                    },
                    "prep artifact is stale; a newer geometry revision prep artifact exists",
                    BTreeMap::from([
                        ("prep_artifact_id".to_string(), prep_artifact_id.to_string()),
                        (
                            "prep_geometry_revision".to_string(),
                            artifact.source_geometry_revision.to_string(),
                        ),
                        (
                            "latest_geometry_revision".to_string(),
                            latest_revision.to_string(),
                        ),
                    ]),
                ));
            }
        }
    }

    let prepared_mesh_count = artifact.prep.prepared_meshes.len();
    let prepared_node_count = artifact
        .prep
        .prepared_meshes
        .iter()
        .map(|mesh| mesh.node_count as usize)
        .sum::<usize>();
    let prepared_element_count = artifact
        .prep
        .prepared_meshes
        .iter()
        .map(|mesh| mesh.element_count as usize)
        .sum::<usize>();
    let mesh_count = prepared_mesh_count.max(1) as f64;
    let topology_surface_patch_ratio = artifact
        .prep
        .prepared_meshes
        .iter()
        .filter(|mesh| mesh.connectivity_class == MeshConnectivityClass::SurfacePatch)
        .count() as f64
        / mesh_count;
    let topology_volume_core_ratio = artifact
        .prep
        .prepared_meshes
        .iter()
        .filter(|mesh| mesh.connectivity_class == MeshConnectivityClass::VolumeCore)
        .count() as f64
        / mesh_count;
    let topology_mixed_family_ratio = artifact
        .prep
        .prepared_meshes
        .iter()
        .filter(|mesh| mesh.element_family_hint == ElementFamilyHint::Mixed)
        .count() as f64
        / mesh_count;
    let topology_triangle_family_ratio = artifact
        .prep
        .prepared_meshes
        .iter()
        .filter(|mesh| mesh.element_family_hint == ElementFamilyHint::Triangle)
        .count() as f64
        / mesh_count;
    let topology_quad_family_ratio = artifact
        .prep
        .prepared_meshes
        .iter()
        .filter(|mesh| mesh.element_family_hint == ElementFamilyHint::Quad)
        .count() as f64
        / mesh_count;
    let topology_tet_family_ratio = artifact
        .prep
        .prepared_meshes
        .iter()
        .filter(|mesh| mesh.element_family_hint == ElementFamilyHint::Tet)
        .count() as f64
        / mesh_count;
    let topology_hex_family_ratio = artifact
        .prep
        .prepared_meshes
        .iter()
        .filter(|mesh| mesh.element_family_hint == ElementFamilyHint::Hex)
        .count() as f64
        / mesh_count;
    let topology_region_span_mean = artifact
        .prep
        .prepared_meshes
        .iter()
        .map(|mesh| mesh.region_span_hint as f64)
        .sum::<f64>()
        / mesh_count;
    let region_block_count = artifact.prep.region_mappings.len().max(1);
    let region_mesh_counts = artifact
        .prep
        .region_mappings
        .iter()
        .map(|mapping| mapping.prepared_mesh_ids.len().max(1) as f64)
        .collect::<Vec<_>>();
    let topology_region_mesh_mean = if region_mesh_counts.is_empty() {
        1.0
    } else {
        region_mesh_counts.iter().sum::<f64>() / region_mesh_counts.len() as f64
    };
    let topology_region_mesh_variance = if region_mesh_counts.len() <= 1 {
        0.0
    } else {
        region_mesh_counts
            .iter()
            .map(|count| {
                let delta = *count - topology_region_mesh_mean;
                delta * delta
            })
            .sum::<f64>()
            / region_mesh_counts.len() as f64
    };
    let topology_dof_multiplier = if model.loads.is_empty() {
        1.0
    } else {
        ((prepared_node_count as f64 / (model.loads.len() as f64 * 3.0)).clamp(1.0, 4.0) * 0.35
            + 1.0)
            .min(4.0)
    };
    let topology_bandwidth_proxy = artifact
        .prep
        .prepared_meshes
        .iter()
        .map(|mesh| mesh.region_span_hint)
        .sum::<u32>()
        .max(1)
        .min(128);
    let mapped_region_participation_ratio = if artifact.prep.region_mappings.is_empty() {
        0.0
    } else {
        (artifact
            .prep
            .region_mappings
            .iter()
            .filter(|mapping| {
                model
                    .loads
                    .iter()
                    .any(|load| load.region_id == mapping.region_id)
                    || model
                        .boundary_conditions
                        .iter()
                        .any(|bc| bc.region_id == mapping.region_id)
            })
            .count() as f64
            / artifact.prep.region_mappings.len() as f64)
            .clamp(0.0, 1.0)
    };

    Ok(Some(AnalysisRunPrepContext {
        prepared_mesh_count,
        prepared_node_count,
        prepared_element_count,
        mapped_region_count: artifact.prep.region_mappings.len(),
        min_scaled_jacobian: artifact.prep.quality.min_scaled_jacobian,
        mean_aspect_ratio: artifact.prep.quality.mean_aspect_ratio,
        inverted_element_count: artifact.prep.quality.inverted_element_count as usize,
        mapped_load_count: model
            .loads
            .iter()
            .filter(|load| {
                artifact
                    .prep
                    .region_mappings
                    .iter()
                    .any(|mapping| mapping.region_id == load.region_id)
            })
            .count(),
        mapped_bc_count: model
            .boundary_conditions
            .iter()
            .filter(|bc| {
                artifact
                    .prep
                    .region_mappings
                    .iter()
                    .any(|mapping| mapping.region_id == bc.region_id)
            })
            .count(),
        layout_seed: {
            let mut seed = 1469598103934665603_u64;
            for mapping in &artifact.prep.region_mappings {
                for byte in mapping.region_id.as_bytes() {
                    seed ^= *byte as u64;
                    seed = seed.wrapping_mul(1099511628211_u64);
                }
            }
            seed
        },
        topology_dof_multiplier,
        topology_bandwidth_proxy,
        mapped_region_participation_ratio,
        topology_surface_patch_ratio,
        topology_volume_core_ratio,
        topology_mixed_family_ratio,
        topology_region_span_mean,
        topology_region_block_count: region_block_count,
        topology_region_mesh_mean,
        topology_region_mesh_variance,
        topology_triangle_family_ratio,
        topology_quad_family_ratio,
        topology_tet_family_ratio,
        topology_hex_family_ratio,
    }))
}

fn run_solve_ms(run: &AnalysisRunResult) -> Option<f64> {
    for code in ["FEA_NONLINEAR_COST", "FEA_TRANSIENT_COST", "FEA_MODAL_COST"] {
        if let Some(value) = diagnostic_metric(&run.run.diagnostics, code, "solve_ms") {
            return Some(value);
        }
    }
    None
}

fn diagnostic_metric(
    diagnostics: &[runmat_analysis_fea::diagnostics::FeaDiagnostic],
    code: &str,
    key: &str,
) -> Option<f64> {
    diagnostics
        .iter()
        .find(|diag| diag.code == code)
        .and_then(|diag| {
            diag.message
                .split_whitespace()
                .find_map(|token| token.strip_prefix(&format!("{key}=")))
        })
        .and_then(|value| value.parse::<f64>().ok())
}

fn diagnostic_metric_u64(
    diagnostics: &[runmat_analysis_fea::diagnostics::FeaDiagnostic],
    code: &str,
    key: &str,
) -> Option<u64> {
    diagnostics
        .iter()
        .find(|diag| diag.code == code)
        .and_then(|diag| {
            diag.message
                .split_whitespace()
                .find_map(|token| token.strip_prefix(&format!("{key}=")))
        })
        .and_then(|value| value.parse::<u64>().ok())
}

fn diagnostic_metric_bool(
    diagnostics: &[runmat_analysis_fea::diagnostics::FeaDiagnostic],
    code: &str,
    key: &str,
) -> Option<bool> {
    diagnostics
        .iter()
        .find(|diag| diag.code == code)
        .and_then(|diag| {
            diag.message
                .split_whitespace()
                .find_map(|token| token.strip_prefix(&format!("{key}=")))
        })
        .and_then(|value| value.parse::<bool>().ok())
}

fn diagnostic_metric_string(
    diagnostics: &[runmat_analysis_fea::diagnostics::FeaDiagnostic],
    code: &str,
    key: &str,
) -> Option<String> {
    diagnostics
        .iter()
        .find(|diag| diag.code == code)
        .and_then(|diag| {
            diag.message
                .split_whitespace()
                .find_map(|token| token.strip_prefix(&format!("{key}=")))
        })
        .map(|value| value.to_string())
}

fn percentile(sorted_samples: &[f64], ratio: f64) -> Option<f64> {
    if sorted_samples.is_empty() {
        return None;
    }
    let index = ((sorted_samples.len() - 1) as f64 * ratio.clamp(0.0, 1.0)).round() as usize;
    sorted_samples.get(index).copied()
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn calibration_profile_rate(entries: &[AnalysisRunResult], profile: &str) -> Option<f64> {
    let values = entries
        .iter()
        .filter_map(|run| {
            diagnostic_metric_string(&run.run.diagnostics, "FEA_PREP_CALIBRATION", "profile")
        })
        .collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }
    Some(
        values
            .iter()
            .filter(|value| value.as_str() == profile)
            .count() as f64
            / values.len() as f64,
    )
}

fn diagnostic_warning_rate(entries: &[AnalysisRunResult], code: &str) -> Option<f64> {
    let values = entries
        .iter()
        .filter_map(|run| {
            run.run
                .diagnostics
                .iter()
                .find(|diag| diag.code == code)
                .map(|diag| {
                    diag.severity
                        == runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
                })
        })
        .collect::<Vec<_>>();
    if values.is_empty() {
        None
    } else {
        Some(values.iter().filter(|value| **value).count() as f64 / values.len() as f64)
    }
}

fn thermo_thresholds_for_policy(policy: QualityPolicy) -> (f64, f64) {
    match policy {
        QualityPolicy::Strict => (1.15, 0.12),
        QualityPolicy::Balanced => (
            THERMO_SPREAD_THRESHOLD_BALANCED,
            THERMO_HETEROGENEITY_THRESHOLD_BALANCED,
        ),
        QualityPolicy::Exploratory => (1.4, 0.35),
    }
}

fn thermo_gradient_thresholds_for_policy(policy: QualityPolicy) -> (f64, f64) {
    match policy {
        QualityPolicy::Strict => (0.22, 0.25),
        QualityPolicy::Balanced => (0.30, 0.35),
        QualityPolicy::Exploratory => (0.45, 0.55),
    }
}

fn thermo_field_quality_thresholds_for_policy(policy: QualityPolicy) -> (f64, f64) {
    match policy {
        QualityPolicy::Strict => (0.55, 0.02),
        QualityPolicy::Balanced => (0.45, 0.08),
        QualityPolicy::Exploratory => (0.30, 0.18),
    }
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

        if materials
            .iter()
            .any(|m: &MaterialModel| m.material_id == material_id)
        {
            continue;
        }
        materials.push(MaterialModel {
            material_id: material_id.to_string(),
            name: name.to_string(),
            mechanical: MaterialMechanicalModel {
                youngs_modulus_pa,
                poisson_ratio,
            },
            thermal: MaterialThermalModel {
                reference_temperature_k: 293.15,
                modulus_temp_coeff_per_k: -2.5e-4,
                ..MaterialThermalModel::default()
            },
            electrical: None,
            plastic: None,
        });
    }

    if materials.is_empty() {
        materials.push(MaterialModel {
            material_id: "mat_default_steel".to_string(),
            name: "Steel (Default)".to_string(),
            mechanical: MaterialMechanicalModel {
                youngs_modulus_pa: 200e9,
                poisson_ratio: 0.3,
            },
            thermal: MaterialThermalModel {
                reference_temperature_k: 293.15,
                modulus_temp_coeff_per_k: -2.5e-4,
                ..MaterialThermalModel::default()
            },
            electrical: None,
            plastic: None,
        });
    }

    materials
}

fn select_fixed_region_id(
    geometry: &GeometryAsset,
    prep_regions: Option<&HashSet<String>>,
) -> Option<String> {
    geometry
        .regions
        .iter()
        .filter(|region| {
            prep_regions
                .map(|mapped| mapped.contains(&region.region_id))
                .unwrap_or(true)
        })
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

fn select_load_region_id(
    geometry: &GeometryAsset,
    prep_regions: Option<&HashSet<String>>,
) -> Option<String> {
    geometry
        .regions
        .iter()
        .filter(|region| {
            prep_regions
                .map(|mapped| mapped.contains(&region.region_id))
                .unwrap_or(true)
        })
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
    prep_regions: Option<&HashSet<String>>,
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

        let evidence_confidence = if geometry
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
        let confidence = if prep_regions
            .map(|mapped| mapped.contains(&region.region_id))
            .unwrap_or(false)
        {
            EvidenceConfidence::Verified
        } else {
            evidence_confidence
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
