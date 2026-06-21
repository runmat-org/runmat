use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::ErrorKind;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock, RwLock};

use chrono::Utc;
use runmat_analysis_core::{
    validate_model_against_geometry, AnalysisField, AnalysisInterfaceKind, AnalysisModel,
    AnalysisModelId, AnalysisStep, AnalysisStepKind, AnalysisValidationError, BoundaryCondition,
    BoundaryConditionKind, EvidenceConfidence, LoadCase, LoadKind, MaterialAssignment,
    MaterialMechanicalModel, MaterialModel, MaterialThermalModel, ReferenceFrame,
};
use runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind;
use runmat_analysis_fea::solve::preconditioner::SpdPreconditionerKind;
use runmat_analysis_fea::{
    fea_cht_energy_residual_field_id, fea_cht_fluid_temperature_field_id,
    fea_cht_interface_heat_flux_field_id, fea_cht_interface_temperature_jump_field_id,
    fea_cht_solid_temperature_field_id, fea_fsi_coupling_iteration_count_field_id,
    fea_fsi_fluid_pressure_field_id, fea_fsi_fluid_velocity_field_id,
    fea_fsi_interface_displacement_field_id, fea_fsi_interface_pressure_field_id,
    fea_fsi_interface_residual_field_id, fea_fsi_interface_traction_field_id,
    fea_fsi_structural_displacement_field_id, run_electromagnetic_with_options,
    run_linear_static_with_options, run_modal_with_options, run_nonlinear_with_options,
    run_thermal_with_options, run_transient_with_options, ComputeBackend,
    ElectromagneticSolveOptions, FeaProgressEvent, FeaProgressHandler, FeaProgressPhase,
    FeaProgressStatus, FeaRunError, FeaRunResult, LinearStaticSolveOptions, ModalSolveOptions,
    ThermalSolveOptions, FEA_FIELD_ACOUSTIC_PARTICLE_VELOCITY, FEA_FIELD_ACOUSTIC_PHASE,
    FEA_FIELD_ACOUSTIC_PRESSURE_IMAG, FEA_FIELD_ACOUSTIC_PRESSURE_MAGNITUDE,
    FEA_FIELD_ACOUSTIC_PRESSURE_REAL, FEA_FIELD_ACOUSTIC_SOUND_PRESSURE_LEVEL_DB,
    FEA_FIELD_CFD_PRESSURE, FEA_FIELD_CFD_RESIDUAL_CONTINUITY, FEA_FIELD_CFD_RESIDUAL_MOMENTUM,
    FEA_FIELD_CFD_REYNOLDS_NUMBER, FEA_FIELD_CFD_VELOCITY, FEA_FIELD_CFD_VORTICITY,
    FEA_FIELD_CFD_WALL_SHEAR_STRESS, FEA_FIELD_CHT_FLUID_PRESSURE, FEA_FIELD_CHT_FLUID_VELOCITY,
};
use runmat_geometry_core::{GeometryAsset, MaterialEvidenceConfidence, UnitSystem};
use runmat_meshing_core::{ElementFamilyHint, MeshConnectivityClass};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::operations::{
    operation_error, OperationContext, OperationEnvelope, OperationErrorEnvelope,
    OperationErrorSeverity, OperationErrorSpec, OperationErrorType,
};
use policy::{
    breach_rate_greater_than, breach_rate_less_than, electromagnetic_sweep_thresholds_for_policy,
    electromagnetic_thresholds_for_policy, thermo_field_quality_thresholds_for_policy,
    thermo_gradient_thresholds_for_policy, thermo_thresholds_for_policy,
    ElectromagneticQualityThresholds, EM_ASSIGNMENT_COVERAGE_MIN_BALANCED,
    EM_BOUNDARY_ANCHOR_MIN_BALANCED, EM_BOUNDARY_ENERGY_MIN_BALANCED,
    EM_BOUNDARY_LOCALIZATION_MIN_BALANCED, EM_BOUNDARY_PENALTY_CONTRIBUTION_MAX_BALANCED,
    EM_CONDITIONING_MAX_BALANCED, EM_CONDUCTIVITY_SPREAD_THRESHOLD_BALANCED,
    EM_ENERGY_IMBALANCE_MAX_BALANCED, EM_FALLBACK_COEFFICIENT_MAX_BALANCED,
    EM_FLUX_DIVERGENCE_MAX_BALANCED, EM_GROUND_EFFECTIVENESS_MIN_BALANCED,
    EM_HETEROGENEITY_THRESHOLD_BALANCED, EM_IMAG_RESIDUAL_MAX_BALANCED,
    EM_INSULATION_LEAKAGE_MAX_BALANCED, EM_REAL_RESIDUAL_MAX_BALANCED,
    EM_REGION_CONTRAST_MAX_BALANCED, EM_RESONANCE_Q_MIN_BALANCED,
    EM_SOURCE_INTERFERENCE_MAX_BALANCED, EM_SOURCE_MATERIAL_ALIGNMENT_MIN_BALANCED,
    EM_SOURCE_OVERLAP_MAX_BALANCED, EM_SOURCE_REALIZATION_MIN_BALANCED,
    EM_SOURCE_REGION_COVERAGE_MIN_BALANCED, EM_SOURCE_REGION_ENERGY_CONSISTENCY_MIN_BALANCED,
    EM_SWEEP_COUNT_MIN_BALANCED, THERMO_HETEROGENEITY_THRESHOLD_BALANCED,
    THERMO_SPREAD_THRESHOLD_BALANCED,
};

mod contracts;
mod fea_document;
#[cfg(feature = "plot-core")]
mod figures;
mod policy;
mod promotion;
pub mod storage;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FeaRuntimeConfig {
    pub artifact_root: Option<PathBuf>,
    pub study_artifact_root: Option<PathBuf>,
    pub thermo_field_artifact_root: Option<PathBuf>,
}

fn fea_runtime_config() -> &'static RwLock<FeaRuntimeConfig> {
    static CONFIG: OnceLock<RwLock<FeaRuntimeConfig>> = OnceLock::new();
    CONFIG.get_or_init(|| RwLock::new(FeaRuntimeConfig::default()))
}

fn current_fea_runtime_config() -> FeaRuntimeConfig {
    fea_runtime_config()
        .read()
        .map(|guard| guard.clone())
        .unwrap_or_default()
}

pub fn default_fea_artifact_root() -> PathBuf {
    PathBuf::from("artifacts")
}

pub fn configure_fea_runtime(config: FeaRuntimeConfig) -> Result<(), String> {
    let mut guard = fea_runtime_config()
        .write()
        .map_err(|_| "FEA runtime config lock poisoned".to_string())?;
    *guard = config;
    Ok(())
}

thread_local! {
    static FEA_PROGRESS_HANDLER: RefCell<Option<FeaProgressHandler>> = const { RefCell::new(None) };
}

pub struct FeaProgressHandlerGuard {
    previous: Option<FeaProgressHandler>,
}

impl Drop for FeaProgressHandlerGuard {
    fn drop(&mut self) {
        FEA_PROGRESS_HANDLER.with(|slot| {
            slot.replace(self.previous.take());
        });
    }
}

pub fn replace_fea_progress_handler(
    handler: Option<FeaProgressHandler>,
) -> FeaProgressHandlerGuard {
    let previous = FEA_PROGRESS_HANDLER.with(|slot| slot.replace(handler));
    FeaProgressHandlerGuard { previous }
}

fn install_fea_solver_context() -> runmat_analysis_fea::FeaProgressContextGuard {
    let host_handler = FEA_PROGRESS_HANDLER.with(|slot| slot.borrow().clone());
    let handler = Some(Arc::new(move |event: FeaProgressEvent| {
        tracing::info!(
            target: "runmat_analysis",
            operation = %event.operation,
            phase = ?event.phase,
            status = ?event.status,
            current = event.current,
            total = event.total,
            fraction = event.fraction,
            "{}", event.message
        );
        if let Some(host_handler) = host_handler.as_ref() {
            host_handler(event);
        }
    }) as FeaProgressHandler);
    runmat_analysis_fea::replace_fea_progress_context(
        handler,
        Some(Arc::new(crate::interrupt::is_cancelled)),
    )
}

pub use contracts::{
    AnalysisAcousticRunOptions, AnalysisCfdRunOptions, AnalysisChtRunOptions,
    AnalysisCreateModelIntentSpec, AnalysisCreateModelPrepContext, AnalysisCreateModelProfile,
    AnalysisElectromagneticRunOptions, AnalysisFieldDescriptor, AnalysisFieldKind,
    AnalysisFieldStorage, AnalysisFsiRunOptions, AnalysisModalRunOptions,
    AnalysisNonlinearRunOptions, AnalysisResultsCompareData, AnalysisResultsCompareQuery,
    AnalysisResultsData, AnalysisResultsQuery, AnalysisResultsSummary, AnalysisRunKind,
    AnalysisRunOptions, AnalysisRunPrepContext, AnalysisRunResult, AnalysisStudyIssue,
    AnalysisStudyPlanData, AnalysisStudyRunData, AnalysisStudySpec, AnalysisStudySweepData,
    AnalysisStudySweepFailureEntry, AnalysisStudySweepPlanData, AnalysisStudySweepPlanEntry,
    AnalysisStudySweepRunEntry, AnalysisStudySweepSpec, AnalysisStudySweepValidateData,
    AnalysisStudySweepValidateEntry, AnalysisStudyValidateResult, AnalysisThermalRunOptions,
    AnalysisTransientRunOptions, AnalysisTrendKindSummary, AnalysisTrendsData, AnalysisTrendsQuery,
    AnalysisValidateResult, ContactInterfaceOptions, ElectroRegionConductivityScale,
    ElectroThermalCouplingOptions, ElectroTimeProfilePoint, ElectromagneticResultsData,
    ModalFrequencyBasis, ModalFrequencyUnits, ModalResultsData, NonlinearMethod,
    NonlinearResultsData, PlasticityConstitutiveOptions, PrecisionMode, PreconditionerMode,
    PrepCalibrationProfile, QualityGate, QualityPolicy, QualityReason, QualityReasonCode,
    RunProvenance, RunStatus, ThermalResultsData, ThermoFieldInterpolationMode, ThermoFieldSource,
    ThermoMechanicalCouplingOptions, ThermoRegionTemperatureDelta, ThermoTimeProfilePoint,
    TransientIntegrationMethod, TransientResultsData,
};
pub use fea_document::{
    is_fea_file_path, load_fea_document_from_path_async, parse_and_resolve_fea_document,
    FeaResolvedDocument,
};
#[cfg(feature = "plot-core")]
pub use figures::{
    analysis_generate_study_run_figures, AnalysisFigureGenerationOptions, AnalysisGeneratedFigure,
    AnalysisGeneratedFigureKind,
};

const ANALYSIS_CREATE_MODEL_OPERATION: &str = "fea.create_model";
const ANALYSIS_CREATE_MODEL_OP_VERSION: &str = "fea.create_model/v1";
const ANALYSIS_VALIDATE_STUDY_OPERATION: &str = "fea.validate_study";
const ANALYSIS_VALIDATE_STUDY_OP_VERSION: &str = "fea.validate_study/v1";
const ANALYSIS_PLAN_STUDY_OPERATION: &str = "fea.plan_study";
const ANALYSIS_PLAN_STUDY_OP_VERSION: &str = "fea.plan_study/v1";
const ANALYSIS_PLAN_STUDY_SWEEP_OPERATION: &str = "fea.plan_study_sweep";
const ANALYSIS_PLAN_STUDY_SWEEP_OP_VERSION: &str = "fea.plan_study_sweep/v1";
const ANALYSIS_RUN_STUDY_OPERATION: &str = "fea.run_study";
const ANALYSIS_RUN_STUDY_OP_VERSION: &str = "fea.run_study/v1";
const ANALYSIS_VALIDATE_STUDY_SWEEP_OPERATION: &str = "fea.validate_study_sweep";
const ANALYSIS_VALIDATE_STUDY_SWEEP_OP_VERSION: &str = "fea.validate_study_sweep/v1";
const ANALYSIS_RUN_STUDY_SWEEP_OPERATION: &str = "fea.run_study_sweep";
const ANALYSIS_RUN_STUDY_SWEEP_OP_VERSION: &str = "fea.run_study_sweep/v1";
const ANALYSIS_VALIDATE_OPERATION: &str = "fea.validate";
const ANALYSIS_VALIDATE_OP_VERSION: &str = "fea.validate/v1";
const ANALYSIS_RUN_OPERATION: &str = "fea.run_linear_static";
const ANALYSIS_RUN_OP_VERSION: &str = "fea.run_linear_static/v1";
const ANALYSIS_RUN_MODAL_OPERATION: &str = "fea.run_modal";
const ANALYSIS_RUN_MODAL_OP_VERSION: &str = "fea.run_modal/v1";
const ANALYSIS_RUN_ACOUSTIC_OPERATION: &str = "fea.run_acoustic";
const ANALYSIS_RUN_ACOUSTIC_OP_VERSION: &str = "fea.run_acoustic/v1";
const ANALYSIS_RUN_TRANSIENT_OPERATION: &str = "fea.run_transient";
const ANALYSIS_RUN_TRANSIENT_OP_VERSION: &str = "fea.run_transient/v1";
const ANALYSIS_RUN_THERMAL_OPERATION: &str = "fea.run_thermal";
const ANALYSIS_RUN_THERMAL_OP_VERSION: &str = "fea.run_thermal/v1";
const ANALYSIS_RUN_NONLINEAR_OPERATION: &str = "fea.run_nonlinear";
const ANALYSIS_RUN_NONLINEAR_OP_VERSION: &str = "fea.run_nonlinear/v1";
const ANALYSIS_RUN_ELECTROMAGNETIC_OPERATION: &str = "fea.run_electromagnetic";
const ANALYSIS_RUN_ELECTROMAGNETIC_OP_VERSION: &str = "fea.run_electromagnetic/v1";
const ANALYSIS_RUN_CFD_OPERATION: &str = "fea.run_cfd";
const ANALYSIS_RUN_CFD_OP_VERSION: &str = "fea.run_cfd/v1";
const ANALYSIS_RUN_CHT_OPERATION: &str = "fea.run_cht";
const ANALYSIS_RUN_CHT_OP_VERSION: &str = "fea.run_cht/v1";
const ANALYSIS_RUN_FSI_OPERATION: &str = "fea.run_fsi";
const ANALYSIS_RUN_FSI_OP_VERSION: &str = "fea.run_fsi/v1";
const ANALYSIS_RESULTS_OPERATION: &str = "fea.results";
const ANALYSIS_RESULTS_OP_VERSION: &str = "fea.results/v1";
const ANALYSIS_RESULTS_COMPARE_OPERATION: &str = "fea.results_compare";
const ANALYSIS_RESULTS_COMPARE_OP_VERSION: &str = "fea.results_compare/v1";
const ANALYSIS_TRENDS_OPERATION: &str = "fea.trends";
const ANALYSIS_TRENDS_OP_VERSION: &str = "fea.trends/v1";
const TRANSIENT_RESIDUAL_WARN_THRESHOLD: f64 = 1.0e-4;

fn map_fea_run_error(
    operation: &str,
    op_version: &str,
    default_error_code: &'static str,
    cancel_error_code: &'static str,
    model: &AnalysisModel,
    context: &OperationContext,
    err: FeaRunError,
) -> OperationErrorEnvelope {
    match err {
        FeaRunError::Cancelled => operation_error(
            operation,
            op_version,
            context,
            OperationErrorSpec {
                error_code: cancel_error_code,
                error_type: OperationErrorType::Cancelled,
                retryable: false,
                severity: OperationErrorSeverity::Warning,
            },
            "FEA run cancelled by user",
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                ("geometry_id".to_string(), model.geometry_id.clone()),
            ]),
        ),
        FeaRunError::InvalidModel(message) => operation_error(
            operation,
            op_version,
            context,
            OperationErrorSpec {
                error_code: default_error_code,
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            message,
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                ("geometry_id".to_string(), model.geometry_id.clone()),
            ]),
        ),
    }
}

fn persist_fea_run_result_with_progress(
    operation: &str,
    op_version: &str,
    artifact_error_code: &'static str,
    context: &OperationContext,
    result: &AnalysisRunResult,
) -> Result<(), OperationErrorEnvelope> {
    runmat_analysis_fea::emit_fea_progress_phase(
        operation,
        FeaProgressPhase::ArtifactPersistence,
        FeaProgressStatus::Started,
        "persisting FEA run artifact",
        None,
        None,
    );
    match storage::persist_run_result(result) {
        Ok(_record) => {
            runmat_analysis_fea::emit_fea_progress_phase(
                operation,
                FeaProgressPhase::ArtifactPersistence,
                FeaProgressStatus::Completed,
                "FEA run artifact persisted",
                None,
                None,
            );
            Ok(())
        }
        Err(err) => {
            let message = format!("failed to persist FEA run artifact: {err}");
            runmat_analysis_fea::emit_fea_progress_phase(
                operation,
                FeaProgressPhase::ArtifactPersistence,
                FeaProgressStatus::Failed,
                &message,
                None,
                None,
            );
            Err(operation_error(
                operation,
                op_version,
                context,
                OperationErrorSpec {
                    error_code: artifact_error_code,
                    error_type: OperationErrorType::Internal,
                    retryable: true,
                    severity: OperationErrorSeverity::Error,
                },
                message,
                BTreeMap::from([("run_id".to_string(), result.run_id.clone())]),
            ))
        }
    }
}

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
                error_code: "RM.FEA.CREATE_MODEL.INVALID_INTENT",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "FEA model intent requires a non-empty model_id",
            BTreeMap::from([("geometry_id".to_string(), geometry.geometry_id.clone())]),
        ));
    }

    if geometry.meshes.is_empty() {
        return Err(operation_error(
            ANALYSIS_CREATE_MODEL_OPERATION,
            ANALYSIS_CREATE_MODEL_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.CREATE_MODEL.GEOMETRY_EMPTY",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "geometry must contain at least one mesh to create an FEA model",
            BTreeMap::from([("geometry_id".to_string(), geometry.geometry_id.clone())]),
        ));
    }

    if geometry.units == UnitSystem::Unspecified {
        return Err(operation_error(
            ANALYSIS_CREATE_MODEL_OPERATION,
            ANALYSIS_CREATE_MODEL_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.CREATE_MODEL.UNIT_UNSPECIFIED",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "geometry units must be specified before creating an FEA model",
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
                    error_code: "RM.FEA.CREATE_MODEL.PREP_MISMATCH",
                    error_type: OperationErrorType::Input,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                "FEA model prep context does not match geometry id/revision",
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
                        error_code: "RM.FEA.CREATE_MODEL.PREP_REGION_NOT_FOUND",
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
                        error_code: "RM.FEA.CREATE_MODEL.PREP_INVALID_MAPPING",
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
                            error_code: "RM.FEA.CREATE_MODEL.PREP_MESH_NOT_FOUND",
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

    let (default_bc, default_load, default_steps) = match intent.profile {
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
            vec![AnalysisStep {
                step_id: "step_default_static".to_string(),
                kind: AnalysisStepKind::Static,
            }],
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
            vec![AnalysisStep {
                step_id: "step_default_thermo_mech".to_string(),
                kind: AnalysisStepKind::Transient,
            }],
        ),
        AnalysisCreateModelProfile::ThermalStandalone => (
            BoundaryCondition {
                bc_id: "bc_default_fixed".to_string(),
                region_id: fixed_region_id,
                kind: BoundaryConditionKind::Fixed,
            },
            LoadCase {
                load_id: "load_default_thermal_seed".to_string(),
                region_id: load_region_id,
                kind: LoadKind::BodyForce {
                    gx: 0.0,
                    gy: 0.0,
                    gz: 0.0,
                },
            },
            vec![AnalysisStep {
                step_id: "step_default_thermal".to_string(),
                kind: AnalysisStepKind::Thermal,
            }],
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
            vec![AnalysisStep {
                step_id: "step_default_modal".to_string(),
                kind: AnalysisStepKind::Modal,
            }],
        ),
        AnalysisCreateModelProfile::AcousticHarmonic => (
            BoundaryCondition {
                bc_id: "bc_default_fixed".to_string(),
                region_id: fixed_region_id,
                kind: BoundaryConditionKind::Fixed,
            },
            LoadCase {
                load_id: "load_default_acoustic_harmonic_seed".to_string(),
                region_id: load_region_id,
                kind: LoadKind::BodyForce {
                    gx: 0.0,
                    gy: 0.0,
                    gz: 0.0,
                },
            },
            vec![AnalysisStep {
                step_id: "step_default_acoustic_harmonic".to_string(),
                kind: AnalysisStepKind::Modal,
            }],
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
            vec![AnalysisStep {
                step_id: "step_default_transient".to_string(),
                kind: AnalysisStepKind::Transient,
            }],
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
            vec![AnalysisStep {
                step_id: "step_default_nonlinear".to_string(),
                kind: AnalysisStepKind::Nonlinear,
            }],
        ),
        AnalysisCreateModelProfile::ElectromagneticStatic => (
            BoundaryCondition {
                bc_id: "bc_default_fixed".to_string(),
                region_id: fixed_region_id,
                kind: BoundaryConditionKind::Fixed,
            },
            LoadCase {
                load_id: "load_default_em_seed".to_string(),
                region_id: load_region_id,
                kind: LoadKind::BodyForce {
                    gx: 0.0,
                    gy: 0.0,
                    gz: 0.0,
                },
            },
            vec![AnalysisStep {
                step_id: "step_default_electromagnetic".to_string(),
                kind: AnalysisStepKind::Electromagnetic,
            }],
        ),
        AnalysisCreateModelProfile::CfdSteadyState => (
            BoundaryCondition {
                bc_id: "bc_default_fixed".to_string(),
                region_id: fixed_region_id,
                kind: BoundaryConditionKind::Fixed,
            },
            LoadCase {
                load_id: "load_default_cfd_seed".to_string(),
                region_id: load_region_id,
                kind: LoadKind::BodyForce {
                    gx: 0.0,
                    gy: 0.0,
                    gz: 0.0,
                },
            },
            vec![AnalysisStep {
                step_id: "step_default_cfd".to_string(),
                kind: AnalysisStepKind::Cfd,
            }],
        ),
        AnalysisCreateModelProfile::CfdTransient => (
            BoundaryCondition {
                bc_id: "bc_default_fixed".to_string(),
                region_id: fixed_region_id,
                kind: BoundaryConditionKind::Fixed,
            },
            LoadCase {
                load_id: "load_default_cfd_transient_seed".to_string(),
                region_id: load_region_id,
                kind: LoadKind::BodyForce {
                    gx: 0.0,
                    gy: 0.0,
                    gz: 0.0,
                },
            },
            vec![AnalysisStep {
                step_id: "step_default_cfd_transient".to_string(),
                kind: AnalysisStepKind::Cfd,
            }],
        ),
        AnalysisCreateModelProfile::ChtCoupled => (
            BoundaryCondition {
                bc_id: "bc_default_fixed".to_string(),
                region_id: fixed_region_id,
                kind: BoundaryConditionKind::Fixed,
            },
            LoadCase {
                load_id: "load_default_cht_seed".to_string(),
                region_id: load_region_id,
                kind: LoadKind::BodyForce {
                    gx: 0.0,
                    gy: 0.0,
                    gz: 0.0,
                },
            },
            vec![
                AnalysisStep {
                    step_id: "step_default_cht_flow".to_string(),
                    kind: AnalysisStepKind::Cfd,
                },
                AnalysisStep {
                    step_id: "step_default_cht_thermal".to_string(),
                    kind: AnalysisStepKind::Thermal,
                },
            ],
        ),
        AnalysisCreateModelProfile::FsiCoupled => (
            BoundaryCondition {
                bc_id: "bc_default_fixed".to_string(),
                region_id: fixed_region_id,
                kind: BoundaryConditionKind::Fixed,
            },
            LoadCase {
                load_id: "load_default_fsi_seed".to_string(),
                region_id: load_region_id,
                kind: LoadKind::Force {
                    fx: 0.0,
                    fy: -450.0,
                    fz: 0.0,
                },
            },
            vec![
                AnalysisStep {
                    step_id: "step_default_fsi_structure".to_string(),
                    kind: AnalysisStepKind::Transient,
                },
                AnalysisStep {
                    step_id: "step_default_fsi_flow".to_string(),
                    kind: AnalysisStepKind::Cfd,
                },
            ],
        ),
    };

    let cfd = match intent.profile {
        AnalysisCreateModelProfile::CfdSteadyState => Some(runmat_analysis_core::CfdDomain {
            enabled: true,
            solve_family: runmat_analysis_core::CfdSolveFamily::SteadyState,
            reference_density_kg_per_m3: 1.225,
            dynamic_viscosity_pa_s: 1.81e-5,
            inlet_velocity_m_per_s: 5.0,
            turbulence_intensity: 0.05,
            time_profile: Vec::new(),
        }),
        AnalysisCreateModelProfile::CfdTransient => Some(runmat_analysis_core::CfdDomain {
            enabled: true,
            solve_family: runmat_analysis_core::CfdSolveFamily::Transient,
            reference_density_kg_per_m3: 1.225,
            dynamic_viscosity_pa_s: 1.81e-5,
            inlet_velocity_m_per_s: 5.0,
            turbulence_intensity: 0.08,
            time_profile: vec![
                runmat_analysis_core::CfdTimeProfilePoint {
                    normalized_time: 0.0,
                    inlet_scale: 0.5,
                },
                runmat_analysis_core::CfdTimeProfilePoint {
                    normalized_time: 1.0,
                    inlet_scale: 1.0,
                },
            ],
        }),
        AnalysisCreateModelProfile::ChtCoupled => Some(runmat_analysis_core::CfdDomain {
            enabled: true,
            solve_family: runmat_analysis_core::CfdSolveFamily::Transient,
            reference_density_kg_per_m3: 1.225,
            dynamic_viscosity_pa_s: 1.81e-5,
            inlet_velocity_m_per_s: 4.5,
            turbulence_intensity: 0.07,
            time_profile: vec![
                runmat_analysis_core::CfdTimeProfilePoint {
                    normalized_time: 0.0,
                    inlet_scale: 0.7,
                },
                runmat_analysis_core::CfdTimeProfilePoint {
                    normalized_time: 1.0,
                    inlet_scale: 1.0,
                },
            ],
        }),
        AnalysisCreateModelProfile::FsiCoupled => Some(runmat_analysis_core::CfdDomain {
            enabled: true,
            solve_family: runmat_analysis_core::CfdSolveFamily::Transient,
            reference_density_kg_per_m3: 1.225,
            dynamic_viscosity_pa_s: 1.81e-5,
            inlet_velocity_m_per_s: 4.0,
            turbulence_intensity: 0.06,
            time_profile: vec![
                runmat_analysis_core::CfdTimeProfilePoint {
                    normalized_time: 0.0,
                    inlet_scale: 0.6,
                },
                runmat_analysis_core::CfdTimeProfilePoint {
                    normalized_time: 1.0,
                    inlet_scale: 1.0,
                },
            ],
        }),
        _ => None,
    };
    let electromagnetic = match intent.profile {
        AnalysisCreateModelProfile::ElectromagneticStatic => {
            Some(runmat_analysis_core::ElectromagneticDomain {
                enabled: true,
                reference_frequency_hz: 60.0,
                applied_current_a: 100.0,
            })
        }
        _ => None,
    };
    let thermo_mechanical = match intent.profile {
        AnalysisCreateModelProfile::ChtCoupled => {
            Some(runmat_analysis_core::ThermoMechanicalDomain {
                enabled: true,
                reference_temperature_k: 293.15,
                applied_temperature_delta_k: 35.0,
                field_artifact_id: None,
                field_source: None,
                region_temperature_deltas: Vec::new(),
                time_profile: vec![
                    runmat_analysis_core::ThermoTimeProfilePoint {
                        normalized_time: 0.0,
                        scale: 0.6,
                    },
                    runmat_analysis_core::ThermoTimeProfilePoint {
                        normalized_time: 1.0,
                        scale: 1.0,
                    },
                ],
            })
        }
        _ => None,
    };

    let model = AnalysisModel {
        model_id: AnalysisModelId(intent.model_id),
        geometry_id: geometry.geometry_id.clone(),
        geometry_revision: geometry.revision,
        units: geometry.units,
        frame: ReferenceFrame::Global,
        materials: inferred_materials,
        material_assignments: inferred_assignments,
        thermo_mechanical,
        electro_thermal: None,
        electromagnetic,
        cfd,
        interfaces: Vec::new(),
        boundary_conditions: vec![default_bc],
        loads: vec![default_load],
        steps: default_steps,
    };

    validate_model_against_geometry(&model, geometry.units, &ReferenceFrame::Global).map_err(
        |error| {
            operation_error(
                ANALYSIS_CREATE_MODEL_OPERATION,
                ANALYSIS_CREATE_MODEL_OP_VERSION,
                &context,
                OperationErrorSpec {
                    error_code: "RM.FEA.CREATE_MODEL.INVALID",
                    error_type: OperationErrorType::Validation,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                format!("created FEA model failed validation: {error:?}"),
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

pub fn analysis_validate_study_op(
    spec: &AnalysisStudySpec,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisStudyValidateResult>, OperationErrorEnvelope> {
    let issue_codes = validate_study_issue_codes(spec);
    let issues: Vec<AnalysisStudyIssue> = issue_codes
        .iter()
        .map(|code| AnalysisStudyIssue {
            code: code.clone(),
            message: study_issue_message(code).to_string(),
        })
        .collect();
    let study_fingerprint = study_fingerprint(spec);
    let evidence_artifact_path = persist_study_evidence(
        &study_fingerprint,
        "validate",
        serde_json::json!({
            "schema_version": "fea_study_validate_artifact/v1",
            "study_id": spec.study_id.clone(),
            "study_fingerprint": study_fingerprint.clone(),
            "valid": issue_codes.is_empty(),
            "issue_codes": issue_codes.clone(),
            "issues": issues.clone(),
            "electromagnetic_run_options": spec.electromagnetic_run_options.clone(),
        }),
    )
    .map_err(|err| {
        operation_error(
            ANALYSIS_VALIDATE_STUDY_OPERATION,
            ANALYSIS_VALIDATE_STUDY_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.VALIDATE_STUDY.ARTIFACT_STORE_FAILED",
                error_type: OperationErrorType::Internal,
                retryable: true,
                severity: OperationErrorSeverity::Error,
            },
            format!("failed to persist study validation evidence artifact: {err}"),
            BTreeMap::from([("study_id".to_string(), spec.study_id.clone())]),
        )
    })?;
    Ok(OperationEnvelope::new(
        ANALYSIS_VALIDATE_STUDY_OPERATION,
        ANALYSIS_VALIDATE_STUDY_OP_VERSION,
        &context,
        AnalysisStudyValidateResult {
            valid: issue_codes.is_empty(),
            issue_codes,
            issues,
            evidence_artifact_path,
        },
    ))
}

pub fn analysis_plan_study_op(
    spec: &AnalysisStudySpec,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisStudyPlanData>, OperationErrorEnvelope> {
    let issue_codes = validate_study_issue_codes(spec);
    if !issue_codes.is_empty() {
        return Err(operation_error(
            ANALYSIS_PLAN_STUDY_OPERATION,
            ANALYSIS_PLAN_STUDY_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.PLAN_STUDY.INVALID_SPEC",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "study spec is invalid; run fea.validate for issue details",
            BTreeMap::from([("issue_codes".to_string(), issue_codes.join(","))]),
        ));
    }

    let study_fingerprint = study_fingerprint(spec);
    let run_operation = run_operation_for_kind(spec.run_kind).to_string();
    let run_op_version = run_operation_version_for_kind(spec.run_kind).to_string();
    let operation_sequence = study_operation_sequence(spec, &run_op_version);
    let evidence_artifact_path = persist_study_evidence(
        &study_fingerprint,
        "plan",
        serde_json::json!({
            "schema_version": "fea_study_plan_artifact/v1",
            "study_id": spec.study_id.clone(),
            "model_id": spec.create_model_intent.model_id.clone(),
            "run_kind": spec.run_kind,
            "backend": spec.backend,
            "run_options": study_run_options_json(spec),
            "study_fingerprint": study_fingerprint.clone(),
            "operation_sequence": operation_sequence.clone(),
            "run_operation": run_operation.clone(),
            "run_op_version": run_op_version.clone(),
        }),
    )
    .map_err(|err| {
        operation_error(
            ANALYSIS_PLAN_STUDY_OPERATION,
            ANALYSIS_PLAN_STUDY_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.PLAN_STUDY.ARTIFACT_STORE_FAILED",
                error_type: OperationErrorType::Internal,
                retryable: true,
                severity: OperationErrorSeverity::Error,
            },
            format!("failed to persist study plan evidence artifact: {err}"),
            BTreeMap::from([("study_id".to_string(), spec.study_id.clone())]),
        )
    })?;
    Ok(OperationEnvelope::new(
        ANALYSIS_PLAN_STUDY_OPERATION,
        ANALYSIS_PLAN_STUDY_OP_VERSION,
        &context,
        AnalysisStudyPlanData {
            study_id: spec.study_id.clone(),
            model_id: spec.create_model_intent.model_id.clone(),
            run_kind: spec.run_kind,
            backend: spec.backend,
            electromagnetic_run_options: spec.electromagnetic_run_options.clone(),
            run_options: study_run_options_json(spec),
            operation_sequence,
            run_operation,
            run_op_version,
            study_fingerprint,
            evidence_artifact_path,
        },
    ))
}

pub fn analysis_run_study_op(
    spec: &AnalysisStudySpec,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisStudyRunData>, OperationErrorEnvelope> {
    let issue_codes = validate_study_issue_codes(spec);
    if !issue_codes.is_empty() {
        return Err(operation_error(
            ANALYSIS_RUN_STUDY_OPERATION,
            ANALYSIS_RUN_STUDY_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_STUDY.INVALID_SPEC",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "study spec is invalid; run fea.validate for issue details",
            BTreeMap::from([("issue_codes".to_string(), issue_codes.join(","))]),
        ));
    }

    let study_fingerprint = study_fingerprint(spec);
    let run_operation = run_operation_for_kind(spec.run_kind).to_string();
    let run_op_version = run_operation_version_for_kind(spec.run_kind).to_string();
    let operation_sequence = study_operation_sequence(spec, &run_op_version);

    let model = match &spec.model {
        Some(model) => model.clone(),
        None => {
            analysis_create_model_op(
                &spec.geometry,
                spec.create_model_intent.clone(),
                context.clone(),
            )?
            .data
        }
    };
    analysis_validate(
        &model,
        spec.geometry.units,
        &ReferenceFrame::Global,
        context.clone(),
    )?;
    let resolved_electromagnetic_run_options = if spec.run_kind == AnalysisRunKind::Electromagnetic
    {
        Some(spec.electromagnetic_run_options.clone().unwrap_or_default())
    } else {
        None
    };

    let run_envelope = match spec.run_kind {
        AnalysisRunKind::LinearStatic => match spec.linear_static_run_options.clone() {
            Some(options) => analysis_run_linear_static_with_options(
                &model,
                spec.backend,
                options,
                context.clone(),
            ),
            None => analysis_run_linear_static_op(&model, spec.backend, context.clone()),
        },
        AnalysisRunKind::Modal => match spec.modal_run_options.clone() {
            Some(options) => {
                analysis_run_modal_with_options_op(&model, spec.backend, options, context.clone())
            }
            None => analysis_run_modal_op(&model, spec.backend, context.clone()),
        },
        AnalysisRunKind::Acoustic => match spec.acoustic_run_options.clone() {
            Some(options) => analysis_run_acoustic_with_options_op(
                &model,
                spec.backend,
                options,
                context.clone(),
            ),
            None => analysis_run_acoustic_op(&model, spec.backend, context.clone()),
        },
        AnalysisRunKind::Thermal => match spec.thermal_run_options.clone() {
            Some(options) => {
                analysis_run_thermal_with_options_op(&model, spec.backend, options, context.clone())
            }
            None => analysis_run_thermal_op(&model, spec.backend, context.clone()),
        },
        AnalysisRunKind::Transient => match spec.transient_run_options.clone() {
            Some(options) => analysis_run_transient_with_options_op(
                &model,
                spec.backend,
                options,
                context.clone(),
            ),
            None => analysis_run_transient_op(&model, spec.backend, context.clone()),
        },
        AnalysisRunKind::Cfd => match spec.cfd_run_options.clone() {
            Some(options) => {
                analysis_run_cfd_with_options_op(&model, spec.backend, options, context.clone())
            }
            None => analysis_run_cfd_op(&model, spec.backend, context.clone()),
        },
        AnalysisRunKind::Cht => match spec.cht_run_options.clone() {
            Some(options) => {
                analysis_run_cht_with_options_op(&model, spec.backend, options, context.clone())
            }
            None => analysis_run_cht_op(&model, spec.backend, context.clone()),
        },
        AnalysisRunKind::Fsi => match spec.fsi_run_options.clone() {
            Some(options) => {
                analysis_run_fsi_with_options_op(&model, spec.backend, options, context.clone())
            }
            None => analysis_run_fsi_op(&model, spec.backend, context.clone()),
        },
        AnalysisRunKind::Nonlinear => match spec.nonlinear_run_options.clone() {
            Some(options) => analysis_run_nonlinear_with_options_op(
                &model,
                spec.backend,
                options,
                context.clone(),
            ),
            None => analysis_run_nonlinear_op(&model, spec.backend, context.clone()),
        },
        AnalysisRunKind::Electromagnetic => analysis_run_electromagnetic_with_options_op(
            &model,
            spec.backend,
            resolved_electromagnetic_run_options
                .clone()
                .unwrap_or_default(),
            context.clone(),
        ),
    }?;

    let evidence_artifact_path = persist_study_evidence(
        &study_fingerprint,
        "run",
        serde_json::json!({
            "schema_version": "fea_study_run_artifact/v1",
            "study_id": spec.study_id.clone(),
            "model_id": model.model_id.0.clone(),
            "run_kind": spec.run_kind,
            "backend": spec.backend,
            "run_options": study_run_options_json(spec),
            "resolved_electromagnetic_run_options": resolved_electromagnetic_run_options.clone(),
            "study_fingerprint": study_fingerprint.clone(),
            "operation_sequence": operation_sequence.clone(),
            "run_operation": run_operation.clone(),
            "run_op_version": run_op_version.clone(),
            "run_id": run_envelope.data.run_id.clone(),
            "run_status": run_envelope.data.run_status,
            "publishable": run_envelope.data.publishable,
            "solver_convergence": run_envelope.data.solver_convergence,
            "result_quality": run_envelope.data.result_quality,
            "quality_reasons": run_envelope.data.quality_reasons.clone(),
            "provenance": run_envelope.data.provenance.clone(),
        }),
    )
    .map_err(|err| {
        operation_error(
            ANALYSIS_RUN_STUDY_OPERATION,
            ANALYSIS_RUN_STUDY_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_STUDY.ARTIFACT_STORE_FAILED",
                error_type: OperationErrorType::Internal,
                retryable: true,
                severity: OperationErrorSeverity::Error,
            },
            format!("failed to persist study run evidence artifact: {err}"),
            BTreeMap::from([
                ("study_id".to_string(), spec.study_id.clone()),
                ("run_id".to_string(), run_envelope.data.run_id.clone()),
            ]),
        )
    })?;

    Ok(OperationEnvelope::new(
        ANALYSIS_RUN_STUDY_OPERATION,
        ANALYSIS_RUN_STUDY_OP_VERSION,
        &context,
        AnalysisStudyRunData {
            study_id: spec.study_id.clone(),
            model_id: model.model_id.0.clone(),
            run_kind: spec.run_kind,
            backend: spec.backend,
            electromagnetic_run_options: resolved_electromagnetic_run_options,
            run_options: study_run_options_json(spec),
            study_fingerprint,
            operation_sequence,
            run_operation,
            run_op_version,
            run_id: run_envelope.data.run_id,
            run_status: run_envelope.data.run_status,
            publishable: run_envelope.data.publishable,
            solver_convergence: run_envelope.data.solver_convergence,
            result_quality: run_envelope.data.result_quality,
            quality_reasons: run_envelope.data.quality_reasons,
            provenance: run_envelope.data.provenance,
            evidence_artifact_path,
        },
    ))
}

pub fn analysis_plan_study_sweep_op(
    spec: &AnalysisStudySweepSpec,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisStudySweepPlanData>, OperationErrorEnvelope> {
    let mut issue_codes = Vec::new();
    if spec.sweep_id.trim().is_empty() {
        issue_codes.push("RM.FEA.STUDY_SWEEP.ID_EMPTY".to_string());
    }
    if spec.studies.is_empty() {
        issue_codes.push("RM.FEA.STUDY_SWEEP.STUDIES_EMPTY".to_string());
    }
    if !issue_codes.is_empty() {
        return Err(operation_error(
            ANALYSIS_PLAN_STUDY_SWEEP_OPERATION,
            ANALYSIS_PLAN_STUDY_SWEEP_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.PLAN_STUDY_SWEEP.INVALID_SPEC",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "study sweep spec is invalid",
            BTreeMap::from([("issue_codes".to_string(), issue_codes.join(","))]),
        ));
    }

    let mut plan_entries = Vec::with_capacity(spec.studies.len());
    let mut failure_entries = Vec::new();
    for (index, study) in spec.studies.iter().enumerate() {
        let planned = match analysis_plan_study_op(study, context.clone()) {
            Ok(plan) => plan,
            Err(err) => {
                if spec.fail_fast {
                    return Err(operation_error(
                        ANALYSIS_PLAN_STUDY_SWEEP_OPERATION,
                        ANALYSIS_PLAN_STUDY_SWEEP_OP_VERSION,
                        &context,
                        OperationErrorSpec {
                            error_code: "RM.FEA.PLAN_STUDY_SWEEP.STUDY_FAILED",
                            error_type: OperationErrorType::Validation,
                            retryable: false,
                            severity: OperationErrorSeverity::Error,
                        },
                        format!(
                            "study sweep planning failed at index {} for study_id {}: {}",
                            index, study.study_id, err.error_code
                        ),
                        BTreeMap::from([
                            ("sweep_id".to_string(), spec.sweep_id.clone()),
                            ("study_id".to_string(), study.study_id.clone()),
                            ("study_index".to_string(), index.to_string()),
                            ("cause_error_code".to_string(), err.error_code),
                        ]),
                    ));
                }
                failure_entries.push(AnalysisStudySweepFailureEntry {
                    study_id: study.study_id.clone(),
                    study_index: index,
                    error_code: err.error_code,
                    message: err.message,
                });
                continue;
            }
        };
        plan_entries.push(AnalysisStudySweepPlanEntry {
            study_id: planned.data.study_id,
            model_id: planned.data.model_id,
            run_kind: planned.data.run_kind,
            backend: planned.data.backend,
            electromagnetic_run_options: planned.data.electromagnetic_run_options,
            run_options: planned.data.run_options,
            operation_sequence: planned.data.operation_sequence,
            run_operation: planned.data.run_operation,
            run_op_version: planned.data.run_op_version,
            study_fingerprint: planned.data.study_fingerprint,
        });
    }

    let sanitized_sweep_id = sanitize_study_sweep_id(&spec.sweep_id);
    let evidence_path = study_evidence_root()
        .join("sweeps")
        .join(sanitized_sweep_id)
        .join("plan.json");
    if let Some(parent) = evidence_path.parent() {
        fs_create_dir_all(parent).map_err(|err| {
            operation_error(
                ANALYSIS_PLAN_STUDY_SWEEP_OPERATION,
                ANALYSIS_PLAN_STUDY_SWEEP_OP_VERSION,
                &context,
                OperationErrorSpec {
                    error_code: "RM.FEA.PLAN_STUDY_SWEEP.ARTIFACT_STORE_FAILED",
                    error_type: OperationErrorType::Internal,
                    retryable: true,
                    severity: OperationErrorSeverity::Error,
                },
                format!("failed to create study sweep planning evidence directory: {err}"),
                BTreeMap::from([("sweep_id".to_string(), spec.sweep_id.clone())]),
            )
        })?;
    }
    let payload = serde_json::json!({
        "schema_version": "fea_study_sweep_plan_artifact/v1",
        "sweep_id": spec.sweep_id.clone(),
        "study_count": spec.studies.len(),
        "planned_count": plan_entries.len(),
        "failed_count": failure_entries.len(),
        "failure_entries": failure_entries.clone(),
        "plan_entries": plan_entries.clone(),
    });
    let payload_bytes = serde_json::to_vec_pretty(&payload).map_err(|err| {
        operation_error(
            ANALYSIS_PLAN_STUDY_SWEEP_OPERATION,
            ANALYSIS_PLAN_STUDY_SWEEP_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.PLAN_STUDY_SWEEP.ARTIFACT_STORE_FAILED",
                error_type: OperationErrorType::Internal,
                retryable: true,
                severity: OperationErrorSeverity::Error,
            },
            format!("failed to encode study sweep planning evidence payload: {err}"),
            BTreeMap::from([("sweep_id".to_string(), spec.sweep_id.clone())]),
        )
    })?;
    atomic_write_bytes(&evidence_path, &payload_bytes).map_err(|err| {
        operation_error(
            ANALYSIS_PLAN_STUDY_SWEEP_OPERATION,
            ANALYSIS_PLAN_STUDY_SWEEP_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.PLAN_STUDY_SWEEP.ARTIFACT_STORE_FAILED",
                error_type: OperationErrorType::Internal,
                retryable: true,
                severity: OperationErrorSeverity::Error,
            },
            err,
            BTreeMap::from([("sweep_id".to_string(), spec.sweep_id.clone())]),
        )
    })?;

    Ok(OperationEnvelope::new(
        ANALYSIS_PLAN_STUDY_SWEEP_OPERATION,
        ANALYSIS_PLAN_STUDY_SWEEP_OP_VERSION,
        &context,
        AnalysisStudySweepPlanData {
            sweep_id: spec.sweep_id.clone(),
            study_count: spec.studies.len(),
            planned_count: plan_entries.len(),
            failed_count: failure_entries.len(),
            failure_entries,
            plan_entries,
            evidence_artifact_path: evidence_path.display().to_string(),
        },
    ))
}

pub fn analysis_validate_study_sweep_op(
    spec: &AnalysisStudySweepSpec,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisStudySweepValidateData>, OperationErrorEnvelope> {
    let mut issue_codes = Vec::new();
    if spec.sweep_id.trim().is_empty() {
        issue_codes.push("RM.FEA.STUDY_SWEEP.ID_EMPTY".to_string());
    }
    if spec.studies.is_empty() {
        issue_codes.push("RM.FEA.STUDY_SWEEP.STUDIES_EMPTY".to_string());
    }

    let study_entries: Vec<AnalysisStudySweepValidateEntry> = spec
        .studies
        .iter()
        .map(|study| {
            let study_issue_codes = validate_study_issue_codes(study);
            let issues = study_issue_codes
                .iter()
                .map(|code| AnalysisStudyIssue {
                    code: code.clone(),
                    message: study_issue_message(code).to_string(),
                })
                .collect::<Vec<_>>();
            AnalysisStudySweepValidateEntry {
                study_id: study.study_id.clone(),
                valid: study_issue_codes.is_empty(),
                issue_codes: study_issue_codes,
                issues,
            }
        })
        .collect();

    let valid = issue_codes.is_empty() && study_entries.iter().all(|entry| entry.valid);
    let sanitized_sweep_id = sanitize_study_sweep_id(&spec.sweep_id);
    let evidence_path = study_evidence_root()
        .join("sweeps")
        .join(sanitized_sweep_id)
        .join("validate.json");
    if let Some(parent) = evidence_path.parent() {
        fs_create_dir_all(parent).map_err(|err| {
            operation_error(
                ANALYSIS_VALIDATE_STUDY_SWEEP_OPERATION,
                ANALYSIS_VALIDATE_STUDY_SWEEP_OP_VERSION,
                &context,
                OperationErrorSpec {
                    error_code: "RM.FEA.VALIDATE_STUDY_SWEEP.ARTIFACT_STORE_FAILED",
                    error_type: OperationErrorType::Internal,
                    retryable: true,
                    severity: OperationErrorSeverity::Error,
                },
                format!("failed to create study sweep validation evidence directory: {err}"),
                BTreeMap::from([("sweep_id".to_string(), spec.sweep_id.clone())]),
            )
        })?;
    }
    let payload = serde_json::json!({
        "schema_version": "fea_study_sweep_validate_artifact/v1",
        "sweep_id": spec.sweep_id.clone(),
        "valid": valid,
        "issue_codes": issue_codes.clone(),
        "study_entries": study_entries,
    });
    let payload_bytes = serde_json::to_vec_pretty(&payload).map_err(|err| {
        operation_error(
            ANALYSIS_VALIDATE_STUDY_SWEEP_OPERATION,
            ANALYSIS_VALIDATE_STUDY_SWEEP_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.VALIDATE_STUDY_SWEEP.ARTIFACT_STORE_FAILED",
                error_type: OperationErrorType::Internal,
                retryable: true,
                severity: OperationErrorSeverity::Error,
            },
            format!("failed to encode study sweep validation evidence payload: {err}"),
            BTreeMap::from([("sweep_id".to_string(), spec.sweep_id.clone())]),
        )
    })?;
    atomic_write_bytes(&evidence_path, &payload_bytes).map_err(|err| {
        operation_error(
            ANALYSIS_VALIDATE_STUDY_SWEEP_OPERATION,
            ANALYSIS_VALIDATE_STUDY_SWEEP_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.VALIDATE_STUDY_SWEEP.ARTIFACT_STORE_FAILED",
                error_type: OperationErrorType::Internal,
                retryable: true,
                severity: OperationErrorSeverity::Error,
            },
            err,
            BTreeMap::from([("sweep_id".to_string(), spec.sweep_id.clone())]),
        )
    })?;

    Ok(OperationEnvelope::new(
        ANALYSIS_VALIDATE_STUDY_SWEEP_OPERATION,
        ANALYSIS_VALIDATE_STUDY_SWEEP_OP_VERSION,
        &context,
        AnalysisStudySweepValidateData {
            sweep_id: spec.sweep_id.clone(),
            valid,
            issue_codes,
            study_entries,
            evidence_artifact_path: evidence_path.display().to_string(),
        },
    ))
}

pub fn analysis_run_study_sweep_op(
    spec: &AnalysisStudySweepSpec,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisStudySweepData>, OperationErrorEnvelope> {
    let mut issue_codes = Vec::new();
    if spec.sweep_id.trim().is_empty() {
        issue_codes.push("RM.FEA.STUDY_SWEEP.ID_EMPTY".to_string());
    }
    if spec.studies.is_empty() {
        issue_codes.push("RM.FEA.STUDY_SWEEP.STUDIES_EMPTY".to_string());
    }
    if !issue_codes.is_empty() {
        return Err(operation_error(
            ANALYSIS_RUN_STUDY_SWEEP_OPERATION,
            ANALYSIS_RUN_STUDY_SWEEP_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_STUDY_SWEEP.INVALID_SPEC",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "study sweep spec is invalid",
            BTreeMap::from([("issue_codes".to_string(), issue_codes.join(","))]),
        ));
    }

    let mut run_entries = Vec::with_capacity(spec.studies.len());
    let mut failure_entries = Vec::new();
    for (index, study) in spec.studies.iter().enumerate() {
        let run = match analysis_run_study_op(study, context.clone()) {
            Ok(run) => run,
            Err(err) => {
                if spec.fail_fast {
                    return Err(operation_error(
                        ANALYSIS_RUN_STUDY_SWEEP_OPERATION,
                        ANALYSIS_RUN_STUDY_SWEEP_OP_VERSION,
                        &context,
                        OperationErrorSpec {
                            error_code: "RM.FEA.RUN_STUDY_SWEEP.STUDY_FAILED",
                            error_type: OperationErrorType::Validation,
                            retryable: false,
                            severity: OperationErrorSeverity::Error,
                        },
                        format!(
                            "study sweep failed at index {} for study_id {}: {}",
                            index, study.study_id, err.error_code
                        ),
                        BTreeMap::from([
                            ("sweep_id".to_string(), spec.sweep_id.clone()),
                            ("study_id".to_string(), study.study_id.clone()),
                            ("study_index".to_string(), index.to_string()),
                            ("cause_error_code".to_string(), err.error_code),
                        ]),
                    ));
                }
                failure_entries.push(AnalysisStudySweepFailureEntry {
                    study_id: study.study_id.clone(),
                    study_index: index,
                    error_code: err.error_code,
                    message: err.message,
                });
                continue;
            }
        };
        run_entries.push(AnalysisStudySweepRunEntry {
            study_id: run.data.study_id,
            run_kind: run.data.run_kind,
            run_id: run.data.run_id,
            run_status: run.data.run_status,
            publishable: run.data.publishable,
            run_operation: run.data.run_operation,
            run_op_version: run.data.run_op_version,
        });
    }

    let sanitized_sweep_id = sanitize_study_sweep_id(&spec.sweep_id);
    let evidence_root = study_evidence_root()
        .join("sweeps")
        .join(sanitized_sweep_id)
        .join("run.json");
    if let Some(parent) = evidence_root.parent() {
        fs_create_dir_all(parent).map_err(|err| {
            operation_error(
                ANALYSIS_RUN_STUDY_SWEEP_OPERATION,
                ANALYSIS_RUN_STUDY_SWEEP_OP_VERSION,
                &context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RUN_STUDY_SWEEP.ARTIFACT_STORE_FAILED",
                    error_type: OperationErrorType::Internal,
                    retryable: true,
                    severity: OperationErrorSeverity::Error,
                },
                format!("failed to create study sweep evidence directory: {err}"),
                BTreeMap::from([("sweep_id".to_string(), spec.sweep_id.clone())]),
            )
        })?;
    }
    let payload = serde_json::json!({
        "schema_version": "fea_study_sweep_run_artifact/v1",
        "sweep_id": spec.sweep_id.clone(),
        "fail_fast": spec.fail_fast,
        "study_count": spec.studies.len(),
        "success_count": run_entries.len(),
        "failed_count": failure_entries.len(),
        "failure_entries": failure_entries.clone(),
        "run_entries": run_entries.clone(),
    });
    let payload_bytes = serde_json::to_vec_pretty(&payload).map_err(|err| {
        operation_error(
            ANALYSIS_RUN_STUDY_SWEEP_OPERATION,
            ANALYSIS_RUN_STUDY_SWEEP_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_STUDY_SWEEP.ARTIFACT_STORE_FAILED",
                error_type: OperationErrorType::Internal,
                retryable: true,
                severity: OperationErrorSeverity::Error,
            },
            format!("failed to encode study sweep evidence payload: {err}"),
            BTreeMap::from([("sweep_id".to_string(), spec.sweep_id.clone())]),
        )
    })?;
    atomic_write_bytes(&evidence_root, &payload_bytes).map_err(|err| {
        operation_error(
            ANALYSIS_RUN_STUDY_SWEEP_OPERATION,
            ANALYSIS_RUN_STUDY_SWEEP_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_STUDY_SWEEP.ARTIFACT_STORE_FAILED",
                error_type: OperationErrorType::Internal,
                retryable: true,
                severity: OperationErrorSeverity::Error,
            },
            err,
            BTreeMap::from([("sweep_id".to_string(), spec.sweep_id.clone())]),
        )
    })?;

    let evidence_artifact_path = evidence_root.display().to_string();
    Ok(OperationEnvelope::new(
        ANALYSIS_RUN_STUDY_SWEEP_OPERATION,
        ANALYSIS_RUN_STUDY_SWEEP_OP_VERSION,
        &context,
        AnalysisStudySweepData {
            sweep_id: spec.sweep_id.clone(),
            study_count: spec.studies.len(),
            success_count: run_entries.len(),
            failed_count: failure_entries.len(),
            failure_entries,
            run_entries,
            evidence_artifact_path,
        },
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

pub fn analysis_run_acoustic_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    analysis_run_acoustic_with_options_op(
        model,
        backend,
        AnalysisAcousticRunOptions::default(),
        context,
    )
}

pub fn analysis_run_modal_with_options_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: AnalysisModalRunOptions,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    let _solver_context = install_fea_solver_context();
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
                error_code: "RM.FEA.RUN_MODAL.INVALID_MODEL",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "FEA model must include at least one modal step for fea.run_modal",
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
                error_code: "RM.FEA.RUN_MODAL.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_modal options require mode_count greater than zero",
            BTreeMap::from([("mode_count".to_string(), options.mode_count.to_string())]),
        ));
    }

    let thermo_options = resolve_thermo_coupling_options(
        model,
        model_thermo_coupling_options(model),
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
                    error_code: "RM.FEA.RUN_MODAL.INVALID_OPTIONS",
                    error_type: OperationErrorType::Input,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                detail,
                metadata,
            ));
        }
    }
    let electro_options = model_electro_coupling_options(model);
    if let Some(electro_options) = electro_options.as_ref() {
        if let Err((detail, metadata)) = validate_electro_coupling_options(model, electro_options) {
            return Err(operation_error(
                ANALYSIS_RUN_MODAL_OPERATION,
                ANALYSIS_RUN_MODAL_OP_VERSION,
                &context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RUN_MODAL.INVALID_OPTIONS",
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
            electro_thermal_context: to_fea_electro_thermal_context(electro_options),
        },
    )
    .map_err(|err| {
        map_fea_run_error(
            ANALYSIS_RUN_MODAL_OPERATION,
            ANALYSIS_RUN_MODAL_OP_VERSION,
            "RM.FEA.RUN_MODAL.SOLVER_MODEL_INVALID",
            "RM.FEA.RUN_MODAL.CANCELLED",
            model,
            &context,
            err,
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

    let frequency_basis = ModalFrequencyBasis::NativeEigenSolve;

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
        thermal_results: None,
        transient_results: None,
        nonlinear_results: None,
        electromagnetic_results: None,
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
            "fea.run_nonlinear outcome run_id={} model_id={} backend={:?} run_status={:?} publishable={} failed_increments={} max_iteration_count={} line_search_backtracks={} tangent_rebuild_count={} max_residual_norm={} max_increment_norm={} max_backtracks_per_increment={} quality_reason_count={}",
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

    persist_fea_run_result_with_progress(
        ANALYSIS_RUN_MODAL_OPERATION,
        ANALYSIS_RUN_MODAL_OP_VERSION,
        "RM.FEA.RUN_MODAL.ARTIFACT_STORE_FAILED",
        &context,
        &result,
    )?;

    Ok(OperationEnvelope::new(
        ANALYSIS_RUN_MODAL_OPERATION,
        ANALYSIS_RUN_MODAL_OP_VERSION,
        &context,
        result,
    ))
}

pub fn analysis_run_acoustic_with_options_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: AnalysisAcousticRunOptions,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    let _solver_context = install_fea_solver_context();
    let has_modal_step = model
        .steps
        .iter()
        .any(|step| step.kind == AnalysisStepKind::Modal);
    if !has_modal_step {
        return Err(operation_error(
            ANALYSIS_RUN_ACOUSTIC_OPERATION,
            ANALYSIS_RUN_ACOUSTIC_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_ACOUSTIC.INVALID_MODEL",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "FEA model must include an acoustic harmonic step marker for fea.run_acoustic",
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                ("geometry_id".to_string(), model.geometry_id.clone()),
            ]),
        ));
    }

    if options.mode_count == 0 {
        return Err(operation_error(
            ANALYSIS_RUN_ACOUSTIC_OPERATION,
            ANALYSIS_RUN_ACOUSTIC_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_ACOUSTIC.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_acoustic options require mode_count greater than zero",
            BTreeMap::from([("mode_count".to_string(), options.mode_count.to_string())]),
        ));
    }

    let thermo_options = resolve_thermo_coupling_options(
        model,
        model_thermo_coupling_options(model),
        ANALYSIS_RUN_ACOUSTIC_OPERATION,
        ANALYSIS_RUN_ACOUSTIC_OP_VERSION,
        &context,
    )?;
    if let Some(thermo_options) = thermo_options.as_ref() {
        if let Err((detail, metadata)) = validate_thermo_coupling_options(model, thermo_options) {
            return Err(operation_error(
                ANALYSIS_RUN_ACOUSTIC_OPERATION,
                ANALYSIS_RUN_ACOUSTIC_OP_VERSION,
                &context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RUN_ACOUSTIC.INVALID_OPTIONS",
                    error_type: OperationErrorType::Input,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                detail,
                metadata,
            ));
        }
    }
    let electro_options = model_electro_coupling_options(model);
    if let Some(electro_options) = electro_options.as_ref() {
        if let Err((detail, metadata)) = validate_electro_coupling_options(model, electro_options) {
            return Err(operation_error(
                ANALYSIS_RUN_ACOUSTIC_OPERATION,
                ANALYSIS_RUN_ACOUSTIC_OP_VERSION,
                &context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RUN_ACOUSTIC.INVALID_OPTIONS",
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
        ANALYSIS_RUN_ACOUSTIC_OPERATION,
        ANALYSIS_RUN_ACOUSTIC_OP_VERSION,
        &context,
    )?;

    let mut run = solve_acoustic_harmonic(
        model,
        backend,
        options.mode_count,
        prep_context,
        options.residual_warn_threshold,
    );
    let acoustic_residual_norm = diagnostic_metric(
        &run.diagnostics,
        "FEA_ACOUSTIC_HELMHOLTZ_RESIDUAL",
        "normalized_residual_norm",
    )
    .unwrap_or(f64::INFINITY);
    let mut fallback_events = Vec::new();
    promotion::promote_run_fields_to_device_refs(&mut run, &mut fallback_events);
    if backend == ComputeBackend::Gpu && run.solver_backend != "runtime_tensor" {
        fallback_events.push(
            "SOLVER_BACKEND_FALLBACK:requested=runtime_tensor:using=cpu_reference".to_string(),
        );
    }
    let solver_convergence = if acoustic_residual_norm <= options.residual_warn_threshold {
        QualityGate::Pass
    } else {
        QualityGate::Warn
    };
    let result_quality = if run.fields_are_empty() {
        QualityGate::Fail
    } else if acoustic_residual_norm > options.residual_warn_threshold {
        QualityGate::Warn
    } else {
        QualityGate::Pass
    };

    let mut quality_reasons = Vec::new();
    if solver_convergence == QualityGate::Warn {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::SolverNotConverged,
            detail: "acoustic solver convergence gate is warning".to_string(),
        });
    }
    if result_quality == QualityGate::Warn {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ModalResidualExceeded,
            detail: format!(
                "acoustic residual exceeds threshold {}",
                options.residual_warn_threshold
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
                && quality_reasons.is_empty()
        }
        QualityPolicy::Balanced => {
            solver_convergence == QualityGate::Pass
                && result_quality == QualityGate::Pass
                && quality_reasons.is_empty()
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
        thermal_results: None,
        transient_results: None,
        nonlinear_results: None,
        electromagnetic_results: None,
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

    persist_fea_run_result_with_progress(
        ANALYSIS_RUN_ACOUSTIC_OPERATION,
        ANALYSIS_RUN_ACOUSTIC_OP_VERSION,
        "RM.FEA.RUN_ACOUSTIC.ARTIFACT_STORE_FAILED",
        &context,
        &result,
    )?;

    Ok(OperationEnvelope::new(
        ANALYSIS_RUN_ACOUSTIC_OPERATION,
        ANALYSIS_RUN_ACOUSTIC_OP_VERSION,
        &context,
        result,
    ))
}

fn solve_acoustic_harmonic(
    model: &AnalysisModel,
    backend: ComputeBackend,
    mode_count: usize,
    prep_context: Option<AnalysisRunPrepContext>,
    residual_warn_threshold: f64,
) -> FeaRunResult {
    let node_count = acoustic_node_count(model, prep_context);
    let reference_temperature_k = acoustic_reference_temperature_k(model);
    let speed_of_sound_m_per_s = 331.3 * (reference_temperature_k / 273.15).sqrt();
    let density_kg_per_m3 = 1.225 * (293.15 / reference_temperature_k.max(1.0));
    let drive_frequency_hz = acoustic_drive_frequency_hz(mode_count, node_count);
    let damping_ratio = 0.02 + 0.002 * mode_count.saturating_sub(1).min(12) as f64;
    let source_real = acoustic_source_vector(model, node_count);
    let source_imag = vec![0.0; node_count];
    let (diag_real, diag_imag, offdiag) = acoustic_helmholtz_operator(
        node_count,
        drive_frequency_hz,
        speed_of_sound_m_per_s,
        damping_ratio,
    );
    let (pressure_real, pressure_imag) =
        solve_complex_tridiagonal(&diag_real, &diag_imag, offdiag, &source_real, &source_imag);
    let normalized_residual_norm = acoustic_residual_norm(
        &diag_real,
        &diag_imag,
        offdiag,
        &pressure_real,
        &pressure_imag,
        &source_real,
        &source_imag,
    );
    let pressure_magnitude = pressure_real
        .iter()
        .zip(pressure_imag.iter())
        .map(|(real, imag)| real.hypot(*imag))
        .collect::<Vec<_>>();
    let phase = pressure_real
        .iter()
        .zip(pressure_imag.iter())
        .map(|(real, imag)| imag.atan2(*real))
        .collect::<Vec<_>>();
    let sound_pressure_level_db = pressure_magnitude
        .iter()
        .map(|pressure| 20.0 * (pressure.max(2.0e-5) / 2.0e-5).log10())
        .collect::<Vec<_>>();
    let particle_velocity =
        recover_acoustic_particle_velocity(&pressure_real, drive_frequency_hz, density_kg_per_m3);
    let peak_pressure_pa = pressure_magnitude.iter().copied().fold(0.0_f64, f64::max);
    let fields = vec![
        AnalysisField::host_f64(
            FEA_FIELD_ACOUSTIC_PRESSURE_REAL,
            vec![node_count],
            pressure_real,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_ACOUSTIC_PRESSURE_IMAG,
            vec![node_count],
            pressure_imag,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_ACOUSTIC_PRESSURE_MAGNITUDE,
            vec![node_count],
            pressure_magnitude,
        ),
        AnalysisField::host_f64(FEA_FIELD_ACOUSTIC_PHASE, vec![node_count], phase),
        AnalysisField::host_f64(
            FEA_FIELD_ACOUSTIC_SOUND_PRESSURE_LEVEL_DB,
            vec![node_count],
            sound_pressure_level_db,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_ACOUSTIC_PARTICLE_VELOCITY,
            vec![node_count, 3],
            particle_velocity,
        ),
    ];
    let severity = if normalized_residual_norm <= residual_warn_threshold {
        runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info
    } else {
        runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
    };
    FeaRunResult {
        backend,
        solver_backend: "cpu_reference".to_string(),
        solver_device_apply_k_ratio: 0.0,
        solver_method: "acoustic_helmholtz_harmonic".to_string(),
        preconditioner: "none".to_string(),
        solver_host_sync_count: 0,
        diagnostics: vec![
            runmat_analysis_fea::diagnostics::FeaDiagnostic {
                code: "FEA_ACOUSTIC_HELMHOLTZ_RESIDUAL".to_string(),
                severity,
                message: format!(
                    "normalized_residual_norm={} equation_scale={} residual_warn_threshold={}",
                    normalized_residual_norm,
                    source_norm(&source_real, &source_imag).max(1.0),
                    residual_warn_threshold,
                ),
            },
            runmat_analysis_fea::diagnostics::FeaDiagnostic {
                code: "FEA_ACOUSTIC_HARMONIC_RESPONSE".to_string(),
                severity: runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info,
                message: format!(
                    "drive_frequency_hz={} speed_of_sound_m_per_s={} density_kg_per_m3={} damping_ratio={} acoustic_node_count={} acoustic_field_count={} peak_pressure_pa={}",
                    drive_frequency_hz,
                    speed_of_sound_m_per_s,
                    density_kg_per_m3,
                    damping_ratio,
                    node_count,
                    fields.len(),
                    peak_pressure_pa,
                ),
            },
        ],
        fields,
    }
}

fn acoustic_node_count(
    model: &AnalysisModel,
    prep_context: Option<AnalysisRunPrepContext>,
) -> usize {
    prep_context
        .map(|prep| prep.prepared_node_count.max(3))
        .unwrap_or_else(|| model.loads.len().saturating_mul(3).max(3))
        .min(512)
}

fn acoustic_reference_temperature_k(model: &AnalysisModel) -> f64 {
    if model.materials.is_empty() {
        293.15
    } else {
        model
            .materials
            .iter()
            .map(|material| material.thermal.reference_temperature_k.max(1.0))
            .sum::<f64>()
            / model.materials.len() as f64
    }
}

fn acoustic_drive_frequency_hz(mode_count: usize, node_count: usize) -> f64 {
    (125.0 * mode_count.max(1) as f64 * (node_count as f64).sqrt()).clamp(50.0, 20_000.0)
}

fn acoustic_source_vector(model: &AnalysisModel, node_count: usize) -> Vec<f64> {
    let mut source = vec![0.0; node_count.max(1)];
    for (index, load) in model.loads.iter().enumerate() {
        let node = (index * 3 + load.region_id.len()) % source.len();
        let amplitude = match &load.kind {
            LoadKind::Pressure { magnitude_pa } => *magnitude_pa,
            LoadKind::Force { fx, fy, fz } => fx.hypot(*fy).hypot(*fz) * 1.0e-3,
            LoadKind::BodyForce { gx, gy, gz } => gx.hypot(*gy).hypot(*gz),
            LoadKind::CurrentDensity { jx, jy, jz, .. } => jx.hypot(*jy).hypot(*jz) * 1.0e-4,
            LoadKind::CoilCurrent { current_a, .. } => current_a.abs() * 1.0e-2,
        };
        source[node] += amplitude;
    }
    if source.iter().all(|value| value.abs() <= 1.0e-12) {
        let center = source.len() / 2;
        source[center] = 1.0;
    }
    source
}

fn acoustic_helmholtz_operator(
    node_count: usize,
    drive_frequency_hz: f64,
    speed_of_sound_m_per_s: f64,
    damping_ratio: f64,
) -> (Vec<f64>, Vec<f64>, f64) {
    let n = node_count.max(1);
    let length_m = 1.0_f64;
    let h = length_m / n.saturating_sub(1).max(1) as f64;
    let omega = 2.0 * std::f64::consts::PI * drive_frequency_hz.max(1.0);
    let wave_number = omega / speed_of_sound_m_per_s.max(1.0);
    let kh_sq = (wave_number * h).powi(2);
    let mut diag_real = vec![2.0 - kh_sq; n];
    let diag_imag = vec![2.0 * damping_ratio.max(0.0) * kh_sq.max(1.0e-9); n];
    if n == 1 {
        diag_real[0] = 1.0 - kh_sq;
    } else {
        diag_real[0] = 1.0 - 0.5 * kh_sq;
        diag_real[n - 1] = 1.0 - 0.5 * kh_sq;
    }
    (diag_real, diag_imag, -1.0)
}

fn solve_complex_tridiagonal(
    diag_real: &[f64],
    diag_imag: &[f64],
    offdiag: f64,
    source_real: &[f64],
    source_imag: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let n = diag_real.len().max(1);
    let mut c_real = vec![0.0; n];
    let mut c_imag = vec![0.0; n];
    let mut d_real = vec![0.0; n];
    let mut d_imag = vec![0.0; n];
    let (inv_real, inv_imag) = complex_recip(diag_real[0], diag_imag[0]);
    c_real[0] = offdiag * inv_real;
    c_imag[0] = offdiag * inv_imag;
    let rhs0_real = source_real.first().copied().unwrap_or(0.0);
    let rhs0_imag = source_imag.first().copied().unwrap_or(0.0);
    (d_real[0], d_imag[0]) = complex_mul(rhs0_real, rhs0_imag, inv_real, inv_imag);
    for i in 1..n {
        let denom_real = diag_real[i] - offdiag * c_real[i - 1];
        let denom_imag = diag_imag[i] - offdiag * c_imag[i - 1];
        let (inv_real, inv_imag) = complex_recip(denom_real, denom_imag);
        if i + 1 < n {
            c_real[i] = offdiag * inv_real;
            c_imag[i] = offdiag * inv_imag;
        }
        let rhs_real = source_real.get(i).copied().unwrap_or(0.0) - offdiag * d_real[i - 1];
        let rhs_imag = source_imag.get(i).copied().unwrap_or(0.0) - offdiag * d_imag[i - 1];
        (d_real[i], d_imag[i]) = complex_mul(rhs_real, rhs_imag, inv_real, inv_imag);
    }
    let mut x_real = vec![0.0; n];
    let mut x_imag = vec![0.0; n];
    x_real[n - 1] = d_real[n - 1];
    x_imag[n - 1] = d_imag[n - 1];
    for i in (0..n.saturating_sub(1)).rev() {
        let (cx_real, cx_imag) = complex_mul(c_real[i], c_imag[i], x_real[i + 1], x_imag[i + 1]);
        x_real[i] = d_real[i] - cx_real;
        x_imag[i] = d_imag[i] - cx_imag;
    }
    (x_real, x_imag)
}

fn acoustic_residual_norm(
    diag_real: &[f64],
    diag_imag: &[f64],
    offdiag: f64,
    pressure_real: &[f64],
    pressure_imag: &[f64],
    source_real: &[f64],
    source_imag: &[f64],
) -> f64 {
    let mut residual_sq = 0.0_f64;
    for i in 0..diag_real.len() {
        let mut applied_real = diag_real[i] * pressure_real[i] - diag_imag[i] * pressure_imag[i];
        let mut applied_imag = diag_real[i] * pressure_imag[i] + diag_imag[i] * pressure_real[i];
        if i > 0 {
            applied_real += offdiag * pressure_real[i - 1];
            applied_imag += offdiag * pressure_imag[i - 1];
        }
        if i + 1 < diag_real.len() {
            applied_real += offdiag * pressure_real[i + 1];
            applied_imag += offdiag * pressure_imag[i + 1];
        }
        let real = applied_real - source_real.get(i).copied().unwrap_or(0.0);
        let imag = applied_imag - source_imag.get(i).copied().unwrap_or(0.0);
        residual_sq += real * real + imag * imag;
    }
    residual_sq.sqrt() / source_norm(source_real, source_imag).max(1.0)
}

fn source_norm(source_real: &[f64], source_imag: &[f64]) -> f64 {
    source_real
        .iter()
        .zip(source_imag.iter().chain(std::iter::repeat(&0.0)))
        .map(|(real, imag)| real * real + imag * imag)
        .sum::<f64>()
        .sqrt()
}

fn complex_recip(real: f64, imag: f64) -> (f64, f64) {
    let denom = (real * real + imag * imag).max(1.0e-24);
    (real / denom, -imag / denom)
}

fn complex_mul(a_real: f64, a_imag: f64, b_real: f64, b_imag: f64) -> (f64, f64) {
    (
        a_real * b_real - a_imag * b_imag,
        a_real * b_imag + a_imag * b_real,
    )
}

fn recover_acoustic_particle_velocity(
    pressure_real: &[f64],
    drive_frequency_hz: f64,
    density_kg_per_m3: f64,
) -> Vec<f64> {
    let node_count = pressure_real.len().max(1);
    let omega = (2.0 * std::f64::consts::PI * drive_frequency_hz).max(1.0e-12);
    let impedance_scale = (density_kg_per_m3.max(1.0e-12) * omega).max(1.0e-12);
    let mut velocity = vec![0.0; node_count * 3];
    for node in 0..pressure_real.len() {
        let left = if node == 0 {
            pressure_real[node]
        } else {
            pressure_real[node - 1]
        };
        let right = if node + 1 >= pressure_real.len() {
            pressure_real[node]
        } else {
            pressure_real[node + 1]
        };
        let spacing = if node == 0 || node + 1 >= pressure_real.len() {
            1.0
        } else {
            2.0
        };
        velocity[node * 3] = -((right - left) / spacing) / impedance_scale;
    }
    velocity
}

fn cfd_reynolds_number(domain: &runmat_analysis_core::CfdDomain) -> f64 {
    domain.reference_density_kg_per_m3 * domain.inlet_velocity_m_per_s
        / domain.dynamic_viscosity_pa_s
}

fn cfd_node_count_from_transient(
    transient_run: &runmat_analysis_fea::FeaTransientRunResult,
) -> usize {
    transient_run
        .velocity_snapshots
        .last()
        .or_else(|| transient_run.displacement_snapshots.last())
        .and_then(|field| {
            field
                .shape
                .first()
                .copied()
                .or_else(|| field.as_host_f64().map(|values| values.len().div_ceil(3)))
        })
        .unwrap_or(1)
        .max(1)
}

fn cfd_profile_scale(domain: &runmat_analysis_core::CfdDomain, step_index: usize) -> f64 {
    domain
        .time_profile
        .get(step_index)
        .map(|point| point.inlet_scale)
        .filter(|scale| scale.is_finite() && *scale >= 0.0)
        .unwrap_or(1.0)
}

fn recover_cfd_velocity_pressure(
    domain: &runmat_analysis_core::CfdDomain,
    node_count: usize,
    step_index: usize,
) -> (Vec<f64>, Vec<f64>) {
    let profile_scale = cfd_profile_scale(domain, step_index);
    let inlet_velocity = domain.inlet_velocity_m_per_s * profile_scale;
    let dynamic_pressure =
        0.5 * domain.reference_density_kg_per_m3 * inlet_velocity * inlet_velocity;
    let turbulence_scale = 1.0 + domain.turbulence_intensity.clamp(0.0, 1.0);
    let denom = node_count.saturating_sub(1).max(1) as f64;
    let mut velocity = Vec::with_capacity(node_count * 3);
    let mut pressure = Vec::with_capacity(node_count);

    for node in 0..node_count {
        let xi = node as f64 / denom;
        let centered = 2.0 * xi - 1.0;
        let axial_shape = (1.0 - 0.25 * centered * centered).max(0.0);
        let axial = inlet_velocity * axial_shape;
        let transverse = inlet_velocity * domain.turbulence_intensity * (xi - 0.5) * 0.05;
        velocity.extend_from_slice(&[axial, transverse, 0.0]);
        pressure.push(dynamic_pressure * turbulence_scale * (1.0 - 0.15 * xi));
    }

    (velocity, pressure)
}

fn recover_cfd_vorticity(velocity: &[f64], node_count: usize) -> Vec<f64> {
    let mut vorticity = vec![0.0; node_count * 3];
    if node_count < 2 {
        return vorticity;
    }

    for node in 0..node_count {
        let prev = node.saturating_sub(1);
        let next = (node + 1).min(node_count - 1);
        let prev_base = prev * 3;
        let next_base = next * 3;
        let dvx = velocity.get(next_base).copied().unwrap_or(0.0)
            - velocity.get(prev_base).copied().unwrap_or(0.0);
        let dvy = velocity.get(next_base + 1).copied().unwrap_or(0.0)
            - velocity.get(prev_base + 1).copied().unwrap_or(0.0);
        let base = node * 3;
        vorticity[base] = 0.0;
        vorticity[base + 1] = -dvx * 0.5;
        vorticity[base + 2] = dvy * 0.5;
    }

    vorticity
}

fn recover_cfd_wall_shear_stress(
    domain: &runmat_analysis_core::CfdDomain,
    velocity: &[f64],
    node_count: usize,
) -> Vec<f64> {
    let mut shear = vec![0.0; node_count * 3];
    let viscosity = domain.dynamic_viscosity_pa_s;
    for node in 0..node_count {
        let base = node * 3;
        shear[base] = viscosity * velocity.get(base).copied().unwrap_or(0.0);
        shear[base + 1] = viscosity * velocity.get(base + 1).copied().unwrap_or(0.0);
    }
    shear
}

fn recover_cfd_residual_fields(
    residual_norms: &[f64],
    fallback_len: usize,
) -> (Vec<f64>, Vec<f64>) {
    let residual_count = residual_norms.len().max(fallback_len).max(1);
    let mut momentum = Vec::with_capacity(residual_count);
    let mut continuity = Vec::with_capacity(residual_count);

    for index in 0..residual_count {
        let residual = residual_norms
            .get(index)
            .copied()
            .filter(|value| value.is_finite())
            .unwrap_or(0.0);
        momentum.push(residual);
        continuity.push(residual * 0.5);
    }

    (momentum, continuity)
}

fn build_cfd_run_fields(
    domain: &runmat_analysis_core::CfdDomain,
    transient_run: &runmat_analysis_fea::FeaTransientRunResult,
) -> Vec<AnalysisField> {
    let node_count = cfd_node_count_from_transient(transient_run);
    let (velocity, pressure) = recover_cfd_velocity_pressure(domain, node_count, 0);
    let vorticity = recover_cfd_vorticity(&velocity, node_count);
    let wall_shear_stress = recover_cfd_wall_shear_stress(domain, &velocity, node_count);
    let (residual_momentum, residual_continuity) =
        recover_cfd_residual_fields(&transient_run.residual_norms, 1);
    let residual_count = residual_momentum.len();

    vec![
        AnalysisField::host_f64(FEA_FIELD_CFD_VELOCITY, vec![node_count, 3], velocity),
        AnalysisField::host_f64(FEA_FIELD_CFD_PRESSURE, vec![node_count], pressure),
        AnalysisField::host_f64(FEA_FIELD_CFD_VORTICITY, vec![node_count, 3], vorticity),
        AnalysisField::host_f64(
            FEA_FIELD_CFD_WALL_SHEAR_STRESS,
            vec![node_count, 3],
            wall_shear_stress,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_CFD_RESIDUAL_MOMENTUM,
            vec![residual_count],
            residual_momentum,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_CFD_RESIDUAL_CONTINUITY,
            vec![residual_count],
            residual_continuity,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_CFD_REYNOLDS_NUMBER,
            vec![1],
            vec![cfd_reynolds_number(domain)],
        ),
    ]
}

fn field_scalar_magnitudes(field: &AnalysisField, fallback_len: usize) -> Vec<f64> {
    let Some(values) = field.as_host_f64() else {
        return vec![0.0; fallback_len.max(1)];
    };
    match field.shape.as_slice() {
        [count, components] if *components > 1 => {
            let mut magnitudes = Vec::with_capacity(*count);
            for index in 0..*count {
                let start = index * *components;
                let magnitude = values
                    .get(start..start + *components)
                    .unwrap_or(&[])
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt();
                magnitudes.push(magnitude);
            }
            magnitudes
        }
        _ => values.to_vec(),
    }
}

fn clone_host_field_as(
    field_id: String,
    source: &AnalysisField,
    fallback_len: usize,
) -> AnalysisField {
    let values = source
        .as_host_f64()
        .map(|values| values.to_vec())
        .unwrap_or_else(|| vec![0.0; fallback_len.max(1)]);
    let shape = if source.shape.is_empty() {
        vec![values.len()]
    } else {
        source.shape.clone()
    };
    AnalysisField::host_f64(field_id, shape, values)
}

fn build_cht_run_fields(
    domain: &runmat_analysis_core::CfdDomain,
    transient_run: &runmat_analysis_fea::FeaTransientRunResult,
    thermal_run: &runmat_analysis_fea::FeaThermalRunResult,
) -> Vec<AnalysisField> {
    let node_count = cfd_node_count_from_transient(transient_run);
    let (fluid_velocity, fluid_pressure) = recover_cfd_velocity_pressure(domain, node_count, 0);
    let mut fields = vec![
        AnalysisField::host_f64(
            FEA_FIELD_CHT_FLUID_VELOCITY,
            vec![node_count, 3],
            fluid_velocity,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_CHT_FLUID_PRESSURE,
            vec![node_count],
            fluid_pressure,
        ),
    ];

    for (step_index, temperature) in thermal_run.temperature_snapshots.iter().enumerate() {
        let fallback_len = temperature.element_count().max(1);
        fields.push(clone_host_field_as(
            fea_cht_fluid_temperature_field_id(step_index),
            temperature,
            fallback_len,
        ));
        fields.push(clone_host_field_as(
            fea_cht_solid_temperature_field_id(step_index),
            temperature,
            fallback_len,
        ));

        let heat_flux = thermal_run
            .heat_flux_snapshots
            .get(step_index)
            .map(|field| field_scalar_magnitudes(field, fallback_len))
            .unwrap_or_else(|| vec![0.0; fallback_len]);
        let interface_count = heat_flux.len().max(1);
        fields.push(AnalysisField::host_f64(
            fea_cht_interface_heat_flux_field_id(step_index),
            vec![interface_count],
            heat_flux,
        ));

        let residual = thermal_run
            .residual_norms
            .get(step_index)
            .copied()
            .filter(|value| value.is_finite())
            .unwrap_or(0.0);
        fields.push(AnalysisField::host_f64(
            fea_cht_interface_temperature_jump_field_id(step_index),
            vec![interface_count],
            vec![residual; interface_count],
        ));
        fields.push(AnalysisField::host_f64(
            fea_cht_energy_residual_field_id(step_index),
            vec![1],
            vec![residual],
        ));
    }

    fields
}

fn build_fsi_run_fields(
    domain: &runmat_analysis_core::CfdDomain,
    transient_run: &runmat_analysis_fea::FeaTransientRunResult,
) -> Vec<AnalysisField> {
    let node_count = cfd_node_count_from_transient(transient_run);
    let mut fields = Vec::new();
    let step_count = transient_run.time_points_s.len().max(1);

    for step_index in 0..step_count {
        let (fluid_velocity, fluid_pressure) =
            recover_cfd_velocity_pressure(domain, node_count, step_index);
        fields.push(AnalysisField::host_f64(
            fea_fsi_fluid_velocity_field_id(step_index),
            vec![node_count, 3],
            fluid_velocity,
        ));
        fields.push(AnalysisField::host_f64(
            fea_fsi_fluid_pressure_field_id(step_index),
            vec![node_count],
            fluid_pressure.clone(),
        ));

        let displacement = transient_run
            .displacement_snapshots
            .get(step_index)
            .or_else(|| transient_run.displacement_snapshots.last());
        let fallback_displacement_len = node_count * 3;
        if let Some(displacement) = displacement {
            fields.push(clone_host_field_as(
                fea_fsi_structural_displacement_field_id(step_index),
                displacement,
                fallback_displacement_len,
            ));
            fields.push(clone_host_field_as(
                fea_fsi_interface_displacement_field_id(step_index),
                displacement,
                fallback_displacement_len,
            ));
        } else {
            let zeros = vec![0.0; fallback_displacement_len];
            fields.push(AnalysisField::host_f64(
                fea_fsi_structural_displacement_field_id(step_index),
                vec![node_count, 3],
                zeros.clone(),
            ));
            fields.push(AnalysisField::host_f64(
                fea_fsi_interface_displacement_field_id(step_index),
                vec![node_count, 3],
                zeros,
            ));
        }

        fields.push(AnalysisField::host_f64(
            fea_fsi_interface_pressure_field_id(step_index),
            vec![node_count],
            fluid_pressure.clone(),
        ));
        let mut traction = Vec::with_capacity(node_count * 3);
        for pressure in &fluid_pressure {
            traction.extend_from_slice(&[-*pressure, 0.0, 0.0]);
        }
        fields.push(AnalysisField::host_f64(
            fea_fsi_interface_traction_field_id(step_index),
            vec![node_count, 3],
            traction,
        ));

        let residual = transient_run
            .residual_norms
            .get(step_index)
            .copied()
            .filter(|value| value.is_finite())
            .unwrap_or(0.0);
        fields.push(AnalysisField::host_f64(
            fea_fsi_interface_residual_field_id(step_index),
            vec![1],
            vec![residual],
        ));
        fields.push(AnalysisField::host_f64(
            fea_fsi_coupling_iteration_count_field_id(step_index),
            vec![1],
            vec![1.0],
        ));
    }

    fields
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

pub fn analysis_run_cfd_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    analysis_run_cfd_with_options_op(model, backend, AnalysisCfdRunOptions::default(), context)
}

pub fn analysis_run_cfd_with_options_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: AnalysisCfdRunOptions,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    let _solver_context = install_fea_solver_context();
    let has_cfd_step = model
        .steps
        .iter()
        .any(|step| step.kind == AnalysisStepKind::Cfd);
    if !has_cfd_step {
        return Err(operation_error(
            ANALYSIS_RUN_CFD_OPERATION,
            ANALYSIS_RUN_CFD_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_CFD.INVALID_MODEL",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "FEA model must include at least one cfd step for fea.run_cfd",
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                ("geometry_id".to_string(), model.geometry_id.clone()),
            ]),
        ));
    }

    let Some(cfd_domain) = model.cfd.as_ref() else {
        return Err(operation_error(
            ANALYSIS_RUN_CFD_OPERATION,
            ANALYSIS_RUN_CFD_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_CFD.INVALID_MODEL",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_cfd requires model.cfd to be configured",
            BTreeMap::from([("analysis_model_id".to_string(), model.model_id.0.clone())]),
        ));
    };

    if !cfd_domain.enabled {
        return Err(operation_error(
            ANALYSIS_RUN_CFD_OPERATION,
            ANALYSIS_RUN_CFD_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_CFD.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_cfd requires cfd domain enabled=true",
            BTreeMap::from([("analysis_model_id".to_string(), model.model_id.0.clone())]),
        ));
    }
    if !cfd_domain.reference_density_kg_per_m3.is_finite()
        || cfd_domain.reference_density_kg_per_m3 <= 0.0
    {
        return Err(operation_error(
            ANALYSIS_RUN_CFD_OPERATION,
            ANALYSIS_RUN_CFD_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_CFD.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_cfd requires finite positive reference_density_kg_per_m3",
            BTreeMap::from([(
                "reference_density_kg_per_m3".to_string(),
                cfd_domain.reference_density_kg_per_m3.to_string(),
            )]),
        ));
    }
    if !cfd_domain.dynamic_viscosity_pa_s.is_finite() || cfd_domain.dynamic_viscosity_pa_s <= 0.0 {
        return Err(operation_error(
            ANALYSIS_RUN_CFD_OPERATION,
            ANALYSIS_RUN_CFD_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_CFD.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_cfd requires finite positive dynamic_viscosity_pa_s",
            BTreeMap::from([(
                "dynamic_viscosity_pa_s".to_string(),
                cfd_domain.dynamic_viscosity_pa_s.to_string(),
            )]),
        ));
    }
    if !cfd_domain.inlet_velocity_m_per_s.is_finite() || cfd_domain.inlet_velocity_m_per_s < 0.0 {
        return Err(operation_error(
            ANALYSIS_RUN_CFD_OPERATION,
            ANALYSIS_RUN_CFD_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_CFD.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_cfd requires finite non-negative inlet_velocity_m_per_s",
            BTreeMap::from([(
                "inlet_velocity_m_per_s".to_string(),
                cfd_domain.inlet_velocity_m_per_s.to_string(),
            )]),
        ));
    }
    if !cfd_domain.turbulence_intensity.is_finite()
        || cfd_domain.turbulence_intensity < 0.0
        || cfd_domain.turbulence_intensity > 1.0
    {
        return Err(operation_error(
            ANALYSIS_RUN_CFD_OPERATION,
            ANALYSIS_RUN_CFD_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_CFD.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_cfd requires turbulence_intensity in [0, 1]",
            BTreeMap::from([(
                "turbulence_intensity".to_string(),
                cfd_domain.turbulence_intensity.to_string(),
            )]),
        ));
    }

    if !options.time_step_s.is_finite() || options.time_step_s <= 0.0 {
        return Err(operation_error(
            ANALYSIS_RUN_CFD_OPERATION,
            ANALYSIS_RUN_CFD_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_CFD.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_cfd options require finite positive time_step_s",
            BTreeMap::from([("time_step_s".to_string(), options.time_step_s.to_string())]),
        ));
    }
    if options.step_count == 0 {
        return Err(operation_error(
            ANALYSIS_RUN_CFD_OPERATION,
            ANALYSIS_RUN_CFD_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_CFD.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_cfd options require step_count greater than zero",
            BTreeMap::from([("step_count".to_string(), options.step_count.to_string())]),
        ));
    }
    if options.max_linear_iters == 0 {
        return Err(operation_error(
            ANALYSIS_RUN_CFD_OPERATION,
            ANALYSIS_RUN_CFD_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_CFD.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_cfd options require max_linear_iters greater than zero",
            BTreeMap::from([(
                "max_linear_iters".to_string(),
                options.max_linear_iters.to_string(),
            )]),
        ));
    }
    if !options.tolerance.is_finite() || options.tolerance <= 0.0 {
        return Err(operation_error(
            ANALYSIS_RUN_CFD_OPERATION,
            ANALYSIS_RUN_CFD_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_CFD.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_cfd options require finite positive tolerance",
            BTreeMap::from([("tolerance".to_string(), options.tolerance.to_string())]),
        ));
    }
    if !options.residual_warn_threshold.is_finite() || options.residual_warn_threshold <= 0.0 {
        return Err(operation_error(
            ANALYSIS_RUN_CFD_OPERATION,
            ANALYSIS_RUN_CFD_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_CFD.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_cfd options require finite positive residual_warn_threshold",
            BTreeMap::from([(
                "residual_warn_threshold".to_string(),
                options.residual_warn_threshold.to_string(),
            )]),
        ));
    }

    let prep_context = resolve_run_prep_context(
        model,
        options.prep_artifact_id.as_deref(),
        options.prep_context,
        ANALYSIS_RUN_CFD_OPERATION,
        ANALYSIS_RUN_CFD_OP_VERSION,
        &context,
    )?;

    let transient_run = run_transient_with_options(
        model,
        backend,
        runmat_analysis_fea::solve::transient::TransientSolveOptions {
            time_step_s: options.time_step_s,
            min_time_step_s: options.time_step_s,
            max_time_step_s: options.time_step_s,
            step_count: options.step_count,
            max_linear_iters: options.max_linear_iters,
            tolerance: options.tolerance,
            residual_target: options.tolerance,
            adaptive_time_step: false,
            max_step_retries: 0,
            adapt_min_scale: 0.8,
            adapt_max_scale: 1.2,
            adapt_growth_exponent: 0.35,
            adapt_retry_growth_cap: 1.05,
            adapt_nonconverged_shrink: 0.75,
            dt_bucket_rel_tolerance: 0.0,
            progress_operation: ANALYSIS_RUN_CFD_OPERATION.to_string(),
            prep_context: to_fea_prep_context(prep_context, options.prep_calibration_profile),
            thermo_mechanical_context: None,
            electro_thermal_context: None,
        },
    )
    .map_err(|err| {
        map_fea_run_error(
            ANALYSIS_RUN_CFD_OPERATION,
            ANALYSIS_RUN_CFD_OP_VERSION,
            "RM.FEA.RUN_CFD.SOLVER_MODEL_INVALID",
            "RM.FEA.RUN_CFD.CANCELLED",
            model,
            &context,
            err,
        )
    })?;

    let cfd_fields = build_cfd_run_fields(cfd_domain, &transient_run);
    let mut run = transient_run.run;
    run.fields.extend(cfd_fields);
    let mut fallback_events = Vec::new();
    promotion::promote_run_fields_to_device_refs(&mut run, &mut fallback_events);
    if backend == ComputeBackend::Gpu && run.solver_backend != "runtime_tensor" {
        fallback_events.push(
            "SOLVER_BACKEND_FALLBACK:requested=runtime_tensor:using=cpu_reference".to_string(),
        );
    }

    let reynolds_number = cfd_reynolds_number(cfd_domain);
    let solve_family = match cfd_domain.solve_family {
        runmat_analysis_core::CfdSolveFamily::SteadyState => "steady_state",
        runmat_analysis_core::CfdSolveFamily::Transient => "transient",
    };
    run.diagnostics.push(runmat_analysis_fea::diagnostics::FeaDiagnostic {
        code: "FEA_CFD_FLOW".to_string(),
        severity: runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info,
        message: format!(
            "density={} viscosity={} inlet_velocity={} turbulence_intensity={} reynolds_number={} solve_family={} profile_point_count={}",
            cfd_domain.reference_density_kg_per_m3,
            cfd_domain.dynamic_viscosity_pa_s,
            cfd_domain.inlet_velocity_m_per_s,
            cfd_domain.turbulence_intensity,
            reynolds_number,
            solve_family,
            cfd_domain.time_profile.len(),
        ),
    });

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
        > options.residual_warn_threshold
    {
        QualityGate::Warn
    } else {
        QualityGate::Pass
    };

    let mut quality_reasons = Vec::new();
    if solver_convergence == QualityGate::Warn {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::SolverNotConverged,
            detail: "cfd solver convergence gate is warning".to_string(),
        });
    }
    if result_quality == QualityGate::Warn {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::TransientResidualExceeded,
            detail: format!(
                "cfd residual exceeds threshold {}",
                options.residual_warn_threshold
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
    let solver_backend = run.solver_backend.clone();
    let solver_device_apply_k_ratio = run.solver_device_apply_k_ratio;
    let solver_host_sync_count = run.solver_host_sync_count;
    let solver_method = run.solver_method.clone();
    let selected_preconditioner = run.preconditioner.clone();

    let result = AnalysisRunResult {
        run_id: storage::next_run_id(),
        run,
        modal_results: None,
        thermal_results: None,
        transient_results: Some(TransientResultsData {
            transient_payload_version: "transient_results/v1".to_string(),
            time_points_s: transient_run.time_points_s,
            displacement_snapshots: transient_run.displacement_snapshots,
            velocity_snapshots: transient_run.velocity_snapshots,
            acceleration_snapshots: transient_run.acceleration_snapshots,
            von_mises_snapshots: transient_run.von_mises_snapshots,
            kinetic_energy_snapshots: transient_run.kinetic_energy_snapshots,
            strain_energy_snapshots: transient_run.strain_energy_snapshots,
            residual_norm_snapshots: transient_run.residual_norm_snapshots,
            thermo_mechanical_temperature_snapshots: transient_run
                .thermo_mechanical_temperature_snapshots,
            thermo_mechanical_thermal_strain_snapshots: transient_run
                .thermo_mechanical_thermal_strain_snapshots,
            thermo_mechanical_thermal_stress_snapshots: transient_run
                .thermo_mechanical_thermal_stress_snapshots,
            thermo_mechanical_displacement_snapshots: transient_run
                .thermo_mechanical_displacement_snapshots,
            thermo_mechanical_von_mises_snapshots: transient_run
                .thermo_mechanical_von_mises_snapshots,
            thermo_mechanical_coupling_residual_snapshots: transient_run
                .thermo_mechanical_coupling_residual_snapshots,
            electro_thermal_temperature_snapshots: transient_run
                .electro_thermal_temperature_snapshots,
            electro_thermal_thermal_residual_snapshots: transient_run
                .electro_thermal_thermal_residual_snapshots,
            residual_norms: transient_run.residual_norms,
            integration_method: TransientIntegrationMethod::ImplicitEuler,
        }),
        nonlinear_results: None,
        electromagnetic_results: None,
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

    persist_fea_run_result_with_progress(
        ANALYSIS_RUN_CFD_OPERATION,
        ANALYSIS_RUN_CFD_OP_VERSION,
        "RM.FEA.RUN_CFD.ARTIFACT_STORE_FAILED",
        &context,
        &result,
    )?;

    Ok(OperationEnvelope::new(
        ANALYSIS_RUN_CFD_OPERATION,
        ANALYSIS_RUN_CFD_OP_VERSION,
        &context,
        result,
    ))
}

pub fn analysis_run_thermal_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    analysis_run_thermal_with_options_op(
        model,
        backend,
        AnalysisThermalRunOptions::default(),
        context,
    )
}

pub fn analysis_run_cht_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    analysis_run_cht_with_options_op(model, backend, AnalysisChtRunOptions::default(), context)
}

pub fn analysis_run_cht_with_options_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: AnalysisChtRunOptions,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    let _solver_context = install_fea_solver_context();
    let has_cfd_step = model
        .steps
        .iter()
        .any(|step| step.kind == AnalysisStepKind::Cfd);
    if !has_cfd_step {
        return Err(operation_error(
            ANALYSIS_RUN_CHT_OPERATION,
            ANALYSIS_RUN_CHT_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_CHT.INVALID_MODEL",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "FEA model must include at least one cfd step for fea.run_cht",
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                ("geometry_id".to_string(), model.geometry_id.clone()),
            ]),
        ));
    }
    let has_thermal_step = model
        .steps
        .iter()
        .any(|step| step.kind == AnalysisStepKind::Thermal);
    if !has_thermal_step {
        return Err(operation_error(
            ANALYSIS_RUN_CHT_OPERATION,
            ANALYSIS_RUN_CHT_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_CHT.INVALID_MODEL",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "FEA model must include at least one thermal step for fea.run_cht",
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                ("geometry_id".to_string(), model.geometry_id.clone()),
            ]),
        ));
    }
    let Some(cfd_domain) = model.cfd.as_ref() else {
        return Err(operation_error(
            ANALYSIS_RUN_CHT_OPERATION,
            ANALYSIS_RUN_CHT_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_CHT.INVALID_MODEL",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_cht requires model.cfd to be configured",
            BTreeMap::from([("analysis_model_id".to_string(), model.model_id.0.clone())]),
        ));
    };
    if !cfd_domain.enabled {
        return Err(operation_error(
            ANALYSIS_RUN_CHT_OPERATION,
            ANALYSIS_RUN_CHT_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_CHT.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_cht requires cfd domain enabled=true",
            BTreeMap::from([("analysis_model_id".to_string(), model.model_id.0.clone())]),
        ));
    }
    if !cfd_domain.reference_density_kg_per_m3.is_finite()
        || cfd_domain.reference_density_kg_per_m3 <= 0.0
    {
        return Err(operation_error(
            ANALYSIS_RUN_CHT_OPERATION,
            ANALYSIS_RUN_CHT_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_CHT.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_cht requires finite positive reference_density_kg_per_m3",
            BTreeMap::from([(
                "reference_density_kg_per_m3".to_string(),
                cfd_domain.reference_density_kg_per_m3.to_string(),
            )]),
        ));
    }
    if !cfd_domain.dynamic_viscosity_pa_s.is_finite() || cfd_domain.dynamic_viscosity_pa_s <= 0.0 {
        return Err(operation_error(
            ANALYSIS_RUN_CHT_OPERATION,
            ANALYSIS_RUN_CHT_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_CHT.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_cht requires finite positive dynamic_viscosity_pa_s",
            BTreeMap::from([(
                "dynamic_viscosity_pa_s".to_string(),
                cfd_domain.dynamic_viscosity_pa_s.to_string(),
            )]),
        ));
    }
    if !cfd_domain.inlet_velocity_m_per_s.is_finite() || cfd_domain.inlet_velocity_m_per_s < 0.0 {
        return Err(operation_error(
            ANALYSIS_RUN_CHT_OPERATION,
            ANALYSIS_RUN_CHT_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_CHT.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_cht requires finite non-negative inlet_velocity_m_per_s",
            BTreeMap::from([(
                "inlet_velocity_m_per_s".to_string(),
                cfd_domain.inlet_velocity_m_per_s.to_string(),
            )]),
        ));
    }
    if !cfd_domain.turbulence_intensity.is_finite()
        || cfd_domain.turbulence_intensity < 0.0
        || cfd_domain.turbulence_intensity > 1.0
    {
        return Err(operation_error(
            ANALYSIS_RUN_CHT_OPERATION,
            ANALYSIS_RUN_CHT_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_CHT.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_cht requires turbulence_intensity in [0, 1]",
            BTreeMap::from([(
                "turbulence_intensity".to_string(),
                cfd_domain.turbulence_intensity.to_string(),
            )]),
        ));
    }
    if !options.time_step_s.is_finite() || options.time_step_s <= 0.0 {
        return Err(operation_error(
            ANALYSIS_RUN_CHT_OPERATION,
            ANALYSIS_RUN_CHT_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_CHT.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_cht options require finite positive time_step_s",
            BTreeMap::from([("time_step_s".to_string(), options.time_step_s.to_string())]),
        ));
    }
    if options.step_count == 0 || options.max_linear_iters == 0 {
        return Err(operation_error(
            ANALYSIS_RUN_CHT_OPERATION,
            ANALYSIS_RUN_CHT_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_CHT.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_cht options require step_count/max_linear_iters greater than zero",
            BTreeMap::new(),
        ));
    }
    if !options.tolerance.is_finite() || options.tolerance <= 0.0 {
        return Err(operation_error(
            ANALYSIS_RUN_CHT_OPERATION,
            ANALYSIS_RUN_CHT_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_CHT.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_cht options require finite positive tolerance",
            BTreeMap::from([("tolerance".to_string(), options.tolerance.to_string())]),
        ));
    }
    if !options.residual_warn_threshold.is_finite() || options.residual_warn_threshold <= 0.0 {
        return Err(operation_error(
            ANALYSIS_RUN_CHT_OPERATION,
            ANALYSIS_RUN_CHT_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_CHT.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_cht options require finite positive residual_warn_threshold",
            BTreeMap::from([(
                "residual_warn_threshold".to_string(),
                options.residual_warn_threshold.to_string(),
            )]),
        ));
    }

    let thermo_options = resolve_thermo_coupling_options(
        model,
        model_thermo_coupling_options(model),
        ANALYSIS_RUN_CHT_OPERATION,
        ANALYSIS_RUN_CHT_OP_VERSION,
        &context,
    )?;
    let Some(thermo_options) = thermo_options else {
        return Err(operation_error(
            ANALYSIS_RUN_CHT_OPERATION,
            ANALYSIS_RUN_CHT_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_CHT.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_cht requires model.thermo_mechanical to be configured",
            BTreeMap::new(),
        ));
    };
    if let Err((detail, metadata)) = validate_thermo_coupling_options(model, &thermo_options) {
        return Err(operation_error(
            ANALYSIS_RUN_CHT_OPERATION,
            ANALYSIS_RUN_CHT_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_CHT.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            detail,
            metadata,
        ));
    }
    let applied_temperature_delta_k = thermo_options.applied_temperature_delta_k;

    let prep_context = resolve_run_prep_context(
        model,
        options.prep_artifact_id.as_deref(),
        options.prep_context,
        ANALYSIS_RUN_CHT_OPERATION,
        ANALYSIS_RUN_CHT_OP_VERSION,
        &context,
    )?;

    let thermal_run = run_thermal_with_options(
        model,
        backend,
        ThermalSolveOptions {
            step_count: options.step_count,
            time_step_s: options.time_step_s,
            residual_target: options.residual_warn_threshold,
            prep_context: to_fea_prep_context(prep_context, options.prep_calibration_profile),
            thermo_mechanical_context: to_fea_thermo_mechanical_context(Some(
                thermo_options.clone(),
            )),
        },
    )
    .map_err(|err| {
        map_fea_run_error(
            ANALYSIS_RUN_CHT_OPERATION,
            ANALYSIS_RUN_CHT_OP_VERSION,
            "RM.FEA.RUN_CHT.SOLVER_MODEL_INVALID",
            "RM.FEA.RUN_CHT.CANCELLED",
            model,
            &context,
            err,
        )
    })?;

    let transient_run = run_transient_with_options(
        model,
        backend,
        runmat_analysis_fea::solve::transient::TransientSolveOptions {
            time_step_s: options.time_step_s,
            min_time_step_s: options.time_step_s,
            max_time_step_s: options.time_step_s,
            step_count: options.step_count,
            max_linear_iters: options.max_linear_iters,
            tolerance: options.tolerance,
            residual_target: options.tolerance,
            adaptive_time_step: false,
            max_step_retries: 0,
            adapt_min_scale: 0.8,
            adapt_max_scale: 1.2,
            adapt_growth_exponent: 0.35,
            adapt_retry_growth_cap: 1.05,
            adapt_nonconverged_shrink: 0.75,
            dt_bucket_rel_tolerance: 0.0,
            progress_operation: ANALYSIS_RUN_CHT_OPERATION.to_string(),
            prep_context: to_fea_prep_context(prep_context, options.prep_calibration_profile),
            thermo_mechanical_context: to_fea_thermo_mechanical_context(Some(thermo_options)),
            electro_thermal_context: None,
        },
    )
    .map_err(|err| {
        map_fea_run_error(
            ANALYSIS_RUN_CHT_OPERATION,
            ANALYSIS_RUN_CHT_OP_VERSION,
            "RM.FEA.RUN_CHT.SOLVER_MODEL_INVALID",
            "RM.FEA.RUN_CHT.CANCELLED",
            model,
            &context,
            err,
        )
    })?;

    let cht_fields = build_cht_run_fields(cfd_domain, &transient_run, &thermal_run);
    let mut run = transient_run.run;
    run.fields.extend(cht_fields);
    run.diagnostics.extend(thermal_run.run.diagnostics.clone());
    let reynolds_number = cfd_reynolds_number(cfd_domain);
    run.diagnostics.push(runmat_analysis_fea::diagnostics::FeaDiagnostic {
        code: "FEA_CFD_FLOW".to_string(),
        severity: runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info,
        message: format!(
            "density={} viscosity={} inlet_velocity={} turbulence_intensity={} reynolds_number={} solve_family={} profile_point_count={}",
            cfd_domain.reference_density_kg_per_m3,
            cfd_domain.dynamic_viscosity_pa_s,
            cfd_domain.inlet_velocity_m_per_s,
            cfd_domain.turbulence_intensity,
            reynolds_number,
            match cfd_domain.solve_family {
                runmat_analysis_core::CfdSolveFamily::SteadyState => "steady_state",
                runmat_analysis_core::CfdSolveFamily::Transient => "transient",
            },
            cfd_domain.time_profile.len(),
        ),
    });
    run.diagnostics.push(runmat_analysis_fea::diagnostics::FeaDiagnostic {
        code: "FEA_CHT_COUPLING".to_string(),
        severity: runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info,
        message: format!(
            "reference_temperature_k={} applied_temperature_delta_k={} step_count={} time_step_s={}",
            thermal_run.reference_temperature_k,
            applied_temperature_delta_k,
            options.step_count,
            options.time_step_s,
        ),
    });

    let mut fallback_events = Vec::new();
    promotion::promote_run_fields_to_device_refs(&mut run, &mut fallback_events);
    if backend == ComputeBackend::Gpu && run.solver_backend != "runtime_tensor" {
        fallback_events.push(
            "SOLVER_BACKEND_FALLBACK:requested=runtime_tensor:using=cpu_reference".to_string(),
        );
    }

    let transient_converged = run.diagnostics.iter().any(|item| {
        item.code == "FEA_TRANSIENT_CONVERGENCE"
            && item.severity == runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info
    });
    let max_transient_residual = transient_run
        .residual_norms
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let max_thermal_residual = thermal_run
        .residual_norms
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let solver_convergence =
        if transient_converged && max_thermal_residual <= options.residual_warn_threshold {
            QualityGate::Pass
        } else {
            QualityGate::Warn
        };
    let result_quality = if transient_run.displacement_snapshots.is_empty()
        || transient_run.time_points_s.is_empty()
        || thermal_run.temperature_snapshots.is_empty()
        || thermal_run.time_points_s.is_empty()
        || transient_run.residual_norms.iter().any(|r| !r.is_finite())
        || thermal_run.residual_norms.iter().any(|r| !r.is_finite())
    {
        QualityGate::Fail
    } else if max_transient_residual > options.residual_warn_threshold
        || max_thermal_residual > options.residual_warn_threshold
    {
        QualityGate::Warn
    } else {
        QualityGate::Pass
    };

    let mut quality_reasons = Vec::new();
    if solver_convergence == QualityGate::Warn {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::SolverNotConverged,
            detail: "cht solver convergence gate is warning".to_string(),
        });
    }
    if max_transient_residual > options.residual_warn_threshold {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::TransientResidualExceeded,
            detail: format!(
                "cht transient residual exceeds threshold {}",
                options.residual_warn_threshold
            ),
        });
    }
    if max_thermal_residual > options.residual_warn_threshold {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ThermalResidualExceeded,
            detail: format!(
                "cht thermal residual exceeds threshold {}",
                options.residual_warn_threshold
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
    let solver_backend = run.solver_backend.clone();
    let solver_device_apply_k_ratio = run.solver_device_apply_k_ratio;
    let solver_host_sync_count = run.solver_host_sync_count;
    let solver_method = run.solver_method.clone();
    let selected_preconditioner = run.preconditioner.clone();

    let result = AnalysisRunResult {
        run_id: storage::next_run_id(),
        run,
        modal_results: None,
        thermal_results: Some(ThermalResultsData {
            thermal_payload_version: "thermal_results/v1".to_string(),
            time_points_s: thermal_run.time_points_s,
            temperature_snapshots: thermal_run.temperature_snapshots,
            temperature_gradient_snapshots: thermal_run.temperature_gradient_snapshots,
            heat_flux_snapshots: thermal_run.heat_flux_snapshots,
            heat_source_snapshots: thermal_run.heat_source_snapshots,
            boundary_heat_flux_snapshots: thermal_run.boundary_heat_flux_snapshots,
            residual_norms: thermal_run.residual_norms,
            reference_temperature_k: thermal_run.reference_temperature_k,
        }),
        transient_results: Some(TransientResultsData {
            transient_payload_version: "transient_results/v1".to_string(),
            time_points_s: transient_run.time_points_s,
            displacement_snapshots: transient_run.displacement_snapshots,
            velocity_snapshots: transient_run.velocity_snapshots,
            acceleration_snapshots: transient_run.acceleration_snapshots,
            von_mises_snapshots: transient_run.von_mises_snapshots,
            kinetic_energy_snapshots: transient_run.kinetic_energy_snapshots,
            strain_energy_snapshots: transient_run.strain_energy_snapshots,
            residual_norm_snapshots: transient_run.residual_norm_snapshots,
            thermo_mechanical_temperature_snapshots: transient_run
                .thermo_mechanical_temperature_snapshots,
            thermo_mechanical_thermal_strain_snapshots: transient_run
                .thermo_mechanical_thermal_strain_snapshots,
            thermo_mechanical_thermal_stress_snapshots: transient_run
                .thermo_mechanical_thermal_stress_snapshots,
            thermo_mechanical_displacement_snapshots: transient_run
                .thermo_mechanical_displacement_snapshots,
            thermo_mechanical_von_mises_snapshots: transient_run
                .thermo_mechanical_von_mises_snapshots,
            thermo_mechanical_coupling_residual_snapshots: transient_run
                .thermo_mechanical_coupling_residual_snapshots,
            electro_thermal_temperature_snapshots: transient_run
                .electro_thermal_temperature_snapshots,
            electro_thermal_thermal_residual_snapshots: transient_run
                .electro_thermal_thermal_residual_snapshots,
            residual_norms: transient_run.residual_norms,
            integration_method: TransientIntegrationMethod::ImplicitEuler,
        }),
        nonlinear_results: None,
        electromagnetic_results: None,
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

    persist_fea_run_result_with_progress(
        ANALYSIS_RUN_CHT_OPERATION,
        ANALYSIS_RUN_CHT_OP_VERSION,
        "RM.FEA.RUN_CHT.ARTIFACT_STORE_FAILED",
        &context,
        &result,
    )?;

    Ok(OperationEnvelope::new(
        ANALYSIS_RUN_CHT_OPERATION,
        ANALYSIS_RUN_CHT_OP_VERSION,
        &context,
        result,
    ))
}

pub fn analysis_run_fsi_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    analysis_run_fsi_with_options_op(model, backend, AnalysisFsiRunOptions::default(), context)
}

pub fn analysis_run_fsi_with_options_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: AnalysisFsiRunOptions,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    let _solver_context = install_fea_solver_context();
    let has_cfd_step = model
        .steps
        .iter()
        .any(|step| step.kind == AnalysisStepKind::Cfd);
    if !has_cfd_step {
        return Err(operation_error(
            ANALYSIS_RUN_FSI_OPERATION,
            ANALYSIS_RUN_FSI_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_FSI.INVALID_MODEL",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "FEA model must include at least one cfd step for fea.run_fsi",
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                ("geometry_id".to_string(), model.geometry_id.clone()),
            ]),
        ));
    }
    let has_transient_step = model
        .steps
        .iter()
        .any(|step| step.kind == AnalysisStepKind::Transient);
    if !has_transient_step {
        return Err(operation_error(
            ANALYSIS_RUN_FSI_OPERATION,
            ANALYSIS_RUN_FSI_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_FSI.INVALID_MODEL",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "FEA model must include at least one transient step for fea.run_fsi",
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                ("geometry_id".to_string(), model.geometry_id.clone()),
            ]),
        ));
    }
    let Some(cfd_domain) = model.cfd.as_ref() else {
        return Err(operation_error(
            ANALYSIS_RUN_FSI_OPERATION,
            ANALYSIS_RUN_FSI_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_FSI.INVALID_MODEL",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_fsi requires model.cfd to be configured",
            BTreeMap::from([("analysis_model_id".to_string(), model.model_id.0.clone())]),
        ));
    };
    if !cfd_domain.enabled {
        return Err(operation_error(
            ANALYSIS_RUN_FSI_OPERATION,
            ANALYSIS_RUN_FSI_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_FSI.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_fsi requires cfd domain enabled=true",
            BTreeMap::from([("analysis_model_id".to_string(), model.model_id.0.clone())]),
        ));
    }
    if !cfd_domain.reference_density_kg_per_m3.is_finite()
        || cfd_domain.reference_density_kg_per_m3 <= 0.0
    {
        return Err(operation_error(
            ANALYSIS_RUN_FSI_OPERATION,
            ANALYSIS_RUN_FSI_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_FSI.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_fsi requires finite positive reference_density_kg_per_m3",
            BTreeMap::from([(
                "reference_density_kg_per_m3".to_string(),
                cfd_domain.reference_density_kg_per_m3.to_string(),
            )]),
        ));
    }
    if !cfd_domain.dynamic_viscosity_pa_s.is_finite() || cfd_domain.dynamic_viscosity_pa_s <= 0.0 {
        return Err(operation_error(
            ANALYSIS_RUN_FSI_OPERATION,
            ANALYSIS_RUN_FSI_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_FSI.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_fsi requires finite positive dynamic_viscosity_pa_s",
            BTreeMap::from([(
                "dynamic_viscosity_pa_s".to_string(),
                cfd_domain.dynamic_viscosity_pa_s.to_string(),
            )]),
        ));
    }
    if !cfd_domain.inlet_velocity_m_per_s.is_finite() || cfd_domain.inlet_velocity_m_per_s < 0.0 {
        return Err(operation_error(
            ANALYSIS_RUN_FSI_OPERATION,
            ANALYSIS_RUN_FSI_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_FSI.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_fsi requires finite non-negative inlet_velocity_m_per_s",
            BTreeMap::from([(
                "inlet_velocity_m_per_s".to_string(),
                cfd_domain.inlet_velocity_m_per_s.to_string(),
            )]),
        ));
    }
    if !cfd_domain.turbulence_intensity.is_finite()
        || cfd_domain.turbulence_intensity < 0.0
        || cfd_domain.turbulence_intensity > 1.0
    {
        return Err(operation_error(
            ANALYSIS_RUN_FSI_OPERATION,
            ANALYSIS_RUN_FSI_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_FSI.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_fsi requires turbulence_intensity in [0, 1]",
            BTreeMap::from([(
                "turbulence_intensity".to_string(),
                cfd_domain.turbulence_intensity.to_string(),
            )]),
        ));
    }
    if !options.time_step_s.is_finite() || options.time_step_s <= 0.0 {
        return Err(operation_error(
            ANALYSIS_RUN_FSI_OPERATION,
            ANALYSIS_RUN_FSI_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_FSI.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_fsi options require finite positive time_step_s",
            BTreeMap::from([("time_step_s".to_string(), options.time_step_s.to_string())]),
        ));
    }
    if options.step_count == 0 || options.max_linear_iters == 0 {
        return Err(operation_error(
            ANALYSIS_RUN_FSI_OPERATION,
            ANALYSIS_RUN_FSI_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_FSI.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_fsi options require step_count/max_linear_iters greater than zero",
            BTreeMap::new(),
        ));
    }
    if !options.tolerance.is_finite() || options.tolerance <= 0.0 {
        return Err(operation_error(
            ANALYSIS_RUN_FSI_OPERATION,
            ANALYSIS_RUN_FSI_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_FSI.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_fsi options require finite positive tolerance",
            BTreeMap::from([("tolerance".to_string(), options.tolerance.to_string())]),
        ));
    }
    if !options.residual_warn_threshold.is_finite() || options.residual_warn_threshold <= 0.0 {
        return Err(operation_error(
            ANALYSIS_RUN_FSI_OPERATION,
            ANALYSIS_RUN_FSI_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_FSI.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_fsi options require finite positive residual_warn_threshold",
            BTreeMap::from([(
                "residual_warn_threshold".to_string(),
                options.residual_warn_threshold.to_string(),
            )]),
        ));
    }

    let prep_context = resolve_run_prep_context(
        model,
        options.prep_artifact_id.as_deref(),
        options.prep_context,
        ANALYSIS_RUN_FSI_OPERATION,
        ANALYSIS_RUN_FSI_OP_VERSION,
        &context,
    )?;

    let transient_run = run_transient_with_options(
        model,
        backend,
        runmat_analysis_fea::solve::transient::TransientSolveOptions {
            time_step_s: options.time_step_s,
            min_time_step_s: options.time_step_s,
            max_time_step_s: options.time_step_s,
            step_count: options.step_count,
            max_linear_iters: options.max_linear_iters,
            tolerance: options.tolerance,
            residual_target: options.tolerance,
            adaptive_time_step: false,
            max_step_retries: 0,
            adapt_min_scale: 0.8,
            adapt_max_scale: 1.2,
            adapt_growth_exponent: 0.35,
            adapt_retry_growth_cap: 1.05,
            adapt_nonconverged_shrink: 0.75,
            dt_bucket_rel_tolerance: 0.0,
            progress_operation: ANALYSIS_RUN_FSI_OPERATION.to_string(),
            prep_context: to_fea_prep_context(prep_context, options.prep_calibration_profile),
            thermo_mechanical_context: None,
            electro_thermal_context: None,
        },
    )
    .map_err(|err| {
        map_fea_run_error(
            ANALYSIS_RUN_FSI_OPERATION,
            ANALYSIS_RUN_FSI_OP_VERSION,
            "RM.FEA.RUN_FSI.SOLVER_MODEL_INVALID",
            "RM.FEA.RUN_FSI.CANCELLED",
            model,
            &context,
            err,
        )
    })?;

    let fsi_fields = build_fsi_run_fields(cfd_domain, &transient_run);
    let mut run = transient_run.run;
    run.fields.extend(fsi_fields);
    let reynolds_number = cfd_reynolds_number(cfd_domain);
    run.diagnostics.push(runmat_analysis_fea::diagnostics::FeaDiagnostic {
        code: "FEA_CFD_FLOW".to_string(),
        severity: runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info,
        message: format!(
            "density={} viscosity={} inlet_velocity={} turbulence_intensity={} reynolds_number={} solve_family={} profile_point_count={}",
            cfd_domain.reference_density_kg_per_m3,
            cfd_domain.dynamic_viscosity_pa_s,
            cfd_domain.inlet_velocity_m_per_s,
            cfd_domain.turbulence_intensity,
            reynolds_number,
            match cfd_domain.solve_family {
                runmat_analysis_core::CfdSolveFamily::SteadyState => "steady_state",
                runmat_analysis_core::CfdSolveFamily::Transient => "transient",
            },
            cfd_domain.time_profile.len(),
        ),
    });
    run.diagnostics.push(runmat_analysis_fea::diagnostics::FeaDiagnostic {
        code: "FEA_FSI_COUPLING".to_string(),
        severity: runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info,
        message: format!(
            "step_count={} time_step_s={} structural_step_count={} cfd_profile_point_count={} interface_count={}",
            options.step_count,
            options.time_step_s,
            model
                .steps
                .iter()
                .filter(|step| step.kind == AnalysisStepKind::Transient)
                .count(),
            cfd_domain.time_profile.len(),
            model.interfaces.len(),
        ),
    });

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
    let max_transient_residual = transient_run
        .residual_norms
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let result_quality = if transient_run.displacement_snapshots.is_empty()
        || transient_run.time_points_s.is_empty()
        || transient_run
            .residual_norms
            .iter()
            .any(|residual| !residual.is_finite())
    {
        QualityGate::Fail
    } else if max_transient_residual > options.residual_warn_threshold {
        QualityGate::Warn
    } else {
        QualityGate::Pass
    };

    let mut quality_reasons = Vec::new();
    if solver_convergence == QualityGate::Warn {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::SolverNotConverged,
            detail: "fsi solver convergence gate is warning".to_string(),
        });
    }
    if max_transient_residual > options.residual_warn_threshold {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::TransientResidualExceeded,
            detail: format!(
                "fsi transient residual exceeds threshold {}",
                options.residual_warn_threshold
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
    let solver_backend = run.solver_backend.clone();
    let solver_device_apply_k_ratio = run.solver_device_apply_k_ratio;
    let solver_host_sync_count = run.solver_host_sync_count;
    let solver_method = run.solver_method.clone();
    let selected_preconditioner = run.preconditioner.clone();

    let result = AnalysisRunResult {
        run_id: storage::next_run_id(),
        run,
        modal_results: None,
        thermal_results: None,
        transient_results: Some(TransientResultsData {
            transient_payload_version: "transient_results/v1".to_string(),
            time_points_s: transient_run.time_points_s,
            displacement_snapshots: transient_run.displacement_snapshots,
            velocity_snapshots: transient_run.velocity_snapshots,
            acceleration_snapshots: transient_run.acceleration_snapshots,
            von_mises_snapshots: transient_run.von_mises_snapshots,
            kinetic_energy_snapshots: transient_run.kinetic_energy_snapshots,
            strain_energy_snapshots: transient_run.strain_energy_snapshots,
            residual_norm_snapshots: transient_run.residual_norm_snapshots,
            thermo_mechanical_temperature_snapshots: transient_run
                .thermo_mechanical_temperature_snapshots,
            thermo_mechanical_thermal_strain_snapshots: transient_run
                .thermo_mechanical_thermal_strain_snapshots,
            thermo_mechanical_thermal_stress_snapshots: transient_run
                .thermo_mechanical_thermal_stress_snapshots,
            thermo_mechanical_displacement_snapshots: transient_run
                .thermo_mechanical_displacement_snapshots,
            thermo_mechanical_von_mises_snapshots: transient_run
                .thermo_mechanical_von_mises_snapshots,
            thermo_mechanical_coupling_residual_snapshots: transient_run
                .thermo_mechanical_coupling_residual_snapshots,
            electro_thermal_temperature_snapshots: transient_run
                .electro_thermal_temperature_snapshots,
            electro_thermal_thermal_residual_snapshots: transient_run
                .electro_thermal_thermal_residual_snapshots,
            residual_norms: transient_run.residual_norms,
            integration_method: TransientIntegrationMethod::ImplicitEuler,
        }),
        nonlinear_results: None,
        electromagnetic_results: None,
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

    persist_fea_run_result_with_progress(
        ANALYSIS_RUN_FSI_OPERATION,
        ANALYSIS_RUN_FSI_OP_VERSION,
        "RM.FEA.RUN_FSI.ARTIFACT_STORE_FAILED",
        &context,
        &result,
    )?;

    Ok(OperationEnvelope::new(
        ANALYSIS_RUN_FSI_OPERATION,
        ANALYSIS_RUN_FSI_OP_VERSION,
        &context,
        result,
    ))
}

pub fn analysis_run_thermal_with_options_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: AnalysisThermalRunOptions,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    let _solver_context = install_fea_solver_context();
    let has_thermal_step = model
        .steps
        .iter()
        .any(|step| step.kind == AnalysisStepKind::Thermal);
    if !has_thermal_step {
        return Err(operation_error(
            ANALYSIS_RUN_THERMAL_OPERATION,
            ANALYSIS_RUN_THERMAL_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_THERMAL.INVALID_MODEL",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "FEA model must include at least one thermal step for fea.run_thermal",
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                ("geometry_id".to_string(), model.geometry_id.clone()),
            ]),
        ));
    }

    let thermo_options = resolve_thermo_coupling_options(
        model,
        model_thermo_coupling_options(model),
        ANALYSIS_RUN_THERMAL_OPERATION,
        ANALYSIS_RUN_THERMAL_OP_VERSION,
        &context,
    )?;
    let Some(thermo_options) = thermo_options else {
        return Err(operation_error(
            ANALYSIS_RUN_THERMAL_OPERATION,
            ANALYSIS_RUN_THERMAL_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_THERMAL.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_thermal requires model.thermo_mechanical to be configured",
            BTreeMap::new(),
        ));
    };
    if let Err((detail, metadata)) = validate_thermo_coupling_options(model, &thermo_options) {
        return Err(operation_error(
            ANALYSIS_RUN_THERMAL_OPERATION,
            ANALYSIS_RUN_THERMAL_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_THERMAL.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            detail,
            metadata,
        ));
    }

    let prep_context = resolve_run_prep_context(
        model,
        options.prep_artifact_id.as_deref(),
        options.prep_context,
        ANALYSIS_RUN_THERMAL_OPERATION,
        ANALYSIS_RUN_THERMAL_OP_VERSION,
        &context,
    )?;

    let thermal_run = run_thermal_with_options(
        model,
        backend,
        ThermalSolveOptions {
            step_count: options.step_count,
            time_step_s: options.time_step_s,
            residual_target: options.residual_warn_threshold,
            prep_context: to_fea_prep_context(prep_context, options.prep_calibration_profile),
            thermo_mechanical_context: to_fea_thermo_mechanical_context(Some(thermo_options)),
        },
    )
    .map_err(|err| {
        map_fea_run_error(
            ANALYSIS_RUN_THERMAL_OPERATION,
            ANALYSIS_RUN_THERMAL_OP_VERSION,
            "RM.FEA.RUN_THERMAL.SOLVER_MODEL_INVALID",
            "RM.FEA.RUN_THERMAL.CANCELLED",
            model,
            &context,
            err,
        )
    })?;

    let mut run = thermal_run.run;
    let mut fallback_events = Vec::new();
    promotion::promote_run_fields_to_device_refs(&mut run, &mut fallback_events);
    let solver_convergence = if diagnostic_metric(
        &run.diagnostics,
        "FEA_THERMAL_STABILITY",
        "max_residual_norm",
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

    let publishable = match options.quality_policy {
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
        thermal_results: Some(ThermalResultsData {
            thermal_payload_version: "thermal_results/v1".to_string(),
            time_points_s: thermal_run.time_points_s,
            temperature_snapshots: thermal_run.temperature_snapshots,
            temperature_gradient_snapshots: thermal_run.temperature_gradient_snapshots,
            heat_flux_snapshots: thermal_run.heat_flux_snapshots,
            heat_source_snapshots: thermal_run.heat_source_snapshots,
            boundary_heat_flux_snapshots: thermal_run.boundary_heat_flux_snapshots,
            residual_norms: thermal_run.residual_norms,
            reference_temperature_k: thermal_run.reference_temperature_k,
        }),
        transient_results: None,
        nonlinear_results: None,
        electromagnetic_results: None,
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

    persist_fea_run_result_with_progress(
        ANALYSIS_RUN_THERMAL_OPERATION,
        ANALYSIS_RUN_THERMAL_OP_VERSION,
        "RM.FEA.RUN_THERMAL.ARTIFACT_STORE_FAILED",
        &context,
        &result,
    )?;

    Ok(OperationEnvelope::new(
        ANALYSIS_RUN_THERMAL_OPERATION,
        ANALYSIS_RUN_THERMAL_OP_VERSION,
        &context,
        result,
    ))
}

pub fn analysis_run_transient_with_options_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: AnalysisTransientRunOptions,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    let _solver_context = install_fea_solver_context();
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
                error_code: "RM.FEA.RUN_TRANSIENT.INVALID_MODEL",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "FEA model must include at least one transient step for fea.run_transient",
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                ("geometry_id".to_string(), model.geometry_id.clone()),
            ]),
        ));
    }

    let thermo_options = resolve_thermo_coupling_options(
        model,
        model_thermo_coupling_options(model),
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
                    error_code: "RM.FEA.RUN_TRANSIENT.INVALID_OPTIONS",
                    error_type: OperationErrorType::Input,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                detail,
                metadata,
            ));
        }
    }
    let electro_options = model_electro_coupling_options(model);
    if let Some(electro_options) = electro_options.as_ref() {
        if let Err((detail, metadata)) = validate_electro_coupling_options(model, electro_options) {
            return Err(operation_error(
                ANALYSIS_RUN_TRANSIENT_OPERATION,
                ANALYSIS_RUN_TRANSIENT_OP_VERSION,
                &context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RUN_TRANSIENT.INVALID_OPTIONS",
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
            progress_operation: ANALYSIS_RUN_TRANSIENT_OPERATION.to_string(),
            prep_context: to_fea_prep_context(prep_context, options.prep_calibration_profile),
            thermo_mechanical_context: to_fea_thermo_mechanical_context(thermo_options),
            electro_thermal_context: to_fea_electro_thermal_context(electro_options),
        }
    })
    .map_err(|err| {
        map_fea_run_error(
            ANALYSIS_RUN_TRANSIENT_OPERATION,
            ANALYSIS_RUN_TRANSIENT_OP_VERSION,
            "RM.FEA.RUN_TRANSIENT.SOLVER_MODEL_INVALID",
            "RM.FEA.RUN_TRANSIENT.CANCELLED",
            model,
            &context,
            err,
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
        thermal_results: None,
        transient_results: Some(TransientResultsData {
            transient_payload_version: "transient_results/v1".to_string(),
            time_points_s: transient_run.time_points_s,
            displacement_snapshots: transient_run.displacement_snapshots,
            velocity_snapshots: transient_run.velocity_snapshots,
            acceleration_snapshots: transient_run.acceleration_snapshots,
            von_mises_snapshots: transient_run.von_mises_snapshots,
            kinetic_energy_snapshots: transient_run.kinetic_energy_snapshots,
            strain_energy_snapshots: transient_run.strain_energy_snapshots,
            residual_norm_snapshots: transient_run.residual_norm_snapshots,
            thermo_mechanical_temperature_snapshots: transient_run
                .thermo_mechanical_temperature_snapshots,
            thermo_mechanical_thermal_strain_snapshots: transient_run
                .thermo_mechanical_thermal_strain_snapshots,
            thermo_mechanical_thermal_stress_snapshots: transient_run
                .thermo_mechanical_thermal_stress_snapshots,
            thermo_mechanical_displacement_snapshots: transient_run
                .thermo_mechanical_displacement_snapshots,
            thermo_mechanical_von_mises_snapshots: transient_run
                .thermo_mechanical_von_mises_snapshots,
            thermo_mechanical_coupling_residual_snapshots: transient_run
                .thermo_mechanical_coupling_residual_snapshots,
            electro_thermal_temperature_snapshots: transient_run
                .electro_thermal_temperature_snapshots,
            electro_thermal_thermal_residual_snapshots: transient_run
                .electro_thermal_thermal_residual_snapshots,
            residual_norms: transient_run.residual_norms,
            integration_method: TransientIntegrationMethod::ImplicitEuler,
        }),
        nonlinear_results: None,
        electromagnetic_results: None,
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

    persist_fea_run_result_with_progress(
        ANALYSIS_RUN_TRANSIENT_OPERATION,
        ANALYSIS_RUN_TRANSIENT_OP_VERSION,
        "RM.FEA.RUN_TRANSIENT.ARTIFACT_STORE_FAILED",
        &context,
        &result,
    )?;

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
    let _solver_context = install_fea_solver_context();
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
                error_code: "RM.FEA.RUN_NONLINEAR.INVALID_MODEL",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "FEA model must include at least one nonlinear step for fea.run_nonlinear",
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
                error_code: "RM.FEA.RUN_NONLINEAR.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_nonlinear options require increment_count greater than zero",
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
                error_code: "RM.FEA.RUN_NONLINEAR.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_nonlinear options require max_newton_iters greater than zero",
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
                error_code: "RM.FEA.RUN_NONLINEAR.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_nonlinear options require finite positive tolerance",
            BTreeMap::from([("tolerance".to_string(), options.tolerance.to_string())]),
        ));
    }
    if options.increment_norm_tolerance <= 0.0 || !options.increment_norm_tolerance.is_finite() {
        return Err(operation_error(
            ANALYSIS_RUN_NONLINEAR_OPERATION,
            ANALYSIS_RUN_NONLINEAR_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_NONLINEAR.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_nonlinear options require finite positive increment_norm_tolerance",
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
                error_code: "RM.FEA.RUN_NONLINEAR.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_nonlinear options require residual_convergence_factor >= 1.0",
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
                error_code: "RM.FEA.RUN_NONLINEAR.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_nonlinear options require line_search_reduction in (0, 1)",
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
                error_code: "RM.FEA.RUN_NONLINEAR.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_nonlinear options require tangent_refresh_interval greater than zero",
            BTreeMap::from([(
                "tangent_refresh_interval".to_string(),
                options.tangent_refresh_interval.to_string(),
            )]),
        ));
    }

    let thermo_options = resolve_thermo_coupling_options(
        model,
        model_thermo_coupling_options(model),
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
                    error_code: "RM.FEA.RUN_NONLINEAR.INVALID_OPTIONS",
                    error_type: OperationErrorType::Input,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                detail,
                metadata,
            ));
        }
    }
    let electro_options = model_electro_coupling_options(model);
    if let Some(electro_options) = electro_options.as_ref() {
        if let Err((detail, metadata)) = validate_electro_coupling_options(model, electro_options) {
            return Err(operation_error(
                ANALYSIS_RUN_NONLINEAR_OPERATION,
                ANALYSIS_RUN_NONLINEAR_OP_VERSION,
                &context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RUN_NONLINEAR.INVALID_OPTIONS",
                    error_type: OperationErrorType::Input,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                detail,
                metadata,
            ));
        }
    }
    let plasticity_options = model_plasticity_constitutive_options(model);
    if let Some(plasticity_options) = plasticity_options.as_ref() {
        if let Err((detail, metadata)) =
            validate_plasticity_constitutive_options(plasticity_options)
        {
            return Err(operation_error(
                ANALYSIS_RUN_NONLINEAR_OPERATION,
                ANALYSIS_RUN_NONLINEAR_OP_VERSION,
                &context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RUN_NONLINEAR.INVALID_OPTIONS",
                    error_type: OperationErrorType::Input,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                detail,
                metadata,
            ));
        }
    }
    let contact_options = model_contact_interface_options(model);
    if let Some(contact_options) = contact_options.as_ref() {
        if let Err((detail, metadata)) = validate_contact_interface_options(contact_options) {
            return Err(operation_error(
                ANALYSIS_RUN_NONLINEAR_OPERATION,
                ANALYSIS_RUN_NONLINEAR_OP_VERSION,
                &context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RUN_NONLINEAR.INVALID_OPTIONS",
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
            electro_thermal_context: to_fea_electro_thermal_context(electro_options),
            plasticity_context: to_fea_plasticity_constitutive_context(plasticity_options),
            contact_context: to_fea_contact_interface_context(contact_options),
        }
    })
    .map_err(|err| {
        map_fea_run_error(
            ANALYSIS_RUN_NONLINEAR_OPERATION,
            ANALYSIS_RUN_NONLINEAR_OP_VERSION,
            "RM.FEA.RUN_NONLINEAR.SOLVER_MODEL_INVALID",
            "RM.FEA.RUN_NONLINEAR.CANCELLED",
            model,
            &context,
            err,
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
        thermal_results: None,
        transient_results: None,
        nonlinear_results: Some(NonlinearResultsData {
            nonlinear_payload_version: "nonlinear_results/v1".to_string(),
            load_factors: nonlinear_run.load_factors,
            displacement_snapshots: nonlinear_run.displacement_snapshots,
            von_mises_snapshots: nonlinear_run.von_mises_snapshots,
            plastic_strain_snapshots: nonlinear_run.plastic_strain_snapshots,
            equivalent_plastic_strain_snapshots: nonlinear_run.equivalent_plastic_strain_snapshots,
            contact_pressure_snapshots: nonlinear_run.contact_pressure_snapshots,
            contact_gap_snapshots: nonlinear_run.contact_gap_snapshots,
            load_factor_snapshots: nonlinear_run.load_factor_snapshots,
            residual_norm_snapshots: nonlinear_run.residual_norm_snapshots,
            thermo_mechanical_temperature_snapshots: nonlinear_run
                .thermo_mechanical_temperature_snapshots,
            thermo_mechanical_thermal_strain_snapshots: nonlinear_run
                .thermo_mechanical_thermal_strain_snapshots,
            thermo_mechanical_thermal_stress_snapshots: nonlinear_run
                .thermo_mechanical_thermal_stress_snapshots,
            thermo_mechanical_displacement_snapshots: nonlinear_run
                .thermo_mechanical_displacement_snapshots,
            thermo_mechanical_von_mises_snapshots: nonlinear_run
                .thermo_mechanical_von_mises_snapshots,
            thermo_mechanical_coupling_residual_snapshots: nonlinear_run
                .thermo_mechanical_coupling_residual_snapshots,
            electro_thermal_temperature_snapshots: nonlinear_run
                .electro_thermal_temperature_snapshots,
            electro_thermal_thermal_residual_snapshots: nonlinear_run
                .electro_thermal_thermal_residual_snapshots,
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
        electromagnetic_results: None,
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

    persist_fea_run_result_with_progress(
        ANALYSIS_RUN_NONLINEAR_OPERATION,
        ANALYSIS_RUN_NONLINEAR_OP_VERSION,
        "RM.FEA.RUN_NONLINEAR.ARTIFACT_STORE_FAILED",
        &context,
        &result,
    )?;

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
    let _solver_context = install_fea_solver_context();
    let thermo_options = resolve_thermo_coupling_options(
        model,
        model_thermo_coupling_options(model),
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
                    error_code: "RM.FEA.RUN_LINEAR_STATIC.INVALID_OPTIONS",
                    error_type: OperationErrorType::Input,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                detail,
                metadata,
            ));
        }
    }
    let electro_options = model_electro_coupling_options(model);
    if let Some(electro_options) = electro_options.as_ref() {
        if let Err((detail, metadata)) = validate_electro_coupling_options(model, electro_options) {
            return Err(operation_error(
                ANALYSIS_RUN_OPERATION,
                ANALYSIS_RUN_OP_VERSION,
                &context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RUN_LINEAR_STATIC.INVALID_OPTIONS",
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
            electro_thermal_context: to_fea_electro_thermal_context(electro_options),
        }
    })
    .map_err(|err| {
        map_fea_run_error(
            ANALYSIS_RUN_OPERATION,
            ANALYSIS_RUN_OP_VERSION,
            "RM.FEA.RUN_LINEAR_STATIC.SOLVER_MODEL_INVALID",
            "RM.FEA.RUN_LINEAR_STATIC.CANCELLED",
            model,
            &context,
            err,
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
    let result_quality = if run.fields_are_empty() {
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
    let solver_backend = run.solver_backend.clone();
    let solver_device_apply_k_ratio = run.solver_device_apply_k_ratio;
    let solver_host_sync_count = run.solver_host_sync_count;
    let solver_method = run.solver_method.clone();
    let preconditioner = run.preconditioner.clone();

    let result = AnalysisRunResult {
        run_id: storage::next_run_id(),
        run,
        modal_results: None,
        thermal_results: None,
        transient_results: None,
        nonlinear_results: None,
        electromagnetic_results: None,
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
            preconditioner,
            quality_policy: contracts::format_quality_policy(options.quality_policy),
            fallback_events,
        },
    };

    persist_fea_run_result_with_progress(
        ANALYSIS_RUN_OPERATION,
        ANALYSIS_RUN_OP_VERSION,
        "RM.FEA.RUN_LINEAR_STATIC.ARTIFACT_STORE_FAILED",
        &context,
        &result,
    )?;

    Ok(OperationEnvelope::new(
        ANALYSIS_RUN_OPERATION,
        ANALYSIS_RUN_OP_VERSION,
        &context,
        result,
    ))
}

pub fn analysis_run_electromagnetic_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    analysis_run_electromagnetic_with_options_op(
        model,
        backend,
        AnalysisElectromagneticRunOptions::default(),
        context,
    )
}

pub fn analysis_run_electromagnetic_with_options_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: AnalysisElectromagneticRunOptions,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    let _solver_context = install_fea_solver_context();
    let has_electromagnetic_step = model
        .steps
        .iter()
        .any(|step| step.kind == AnalysisStepKind::Electromagnetic);
    if !has_electromagnetic_step {
        return Err(operation_error(
            ANALYSIS_RUN_ELECTROMAGNETIC_OPERATION,
            ANALYSIS_RUN_ELECTROMAGNETIC_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_ELECTROMAGNETIC.REQUIRES_STEP",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "FEA model must include at least one electromagnetic step for fea.run_electromagnetic",
            BTreeMap::from([("analysis_model_id".to_string(), model.model_id.0.clone())]),
        ));
    }

    let Some(em_domain) = model.electromagnetic.as_ref() else {
        return Err(operation_error(
            ANALYSIS_RUN_ELECTROMAGNETIC_OPERATION,
            ANALYSIS_RUN_ELECTROMAGNETIC_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_ELECTROMAGNETIC.INVALID_MODEL",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_electromagnetic requires model.electromagnetic to be configured",
            BTreeMap::from([("analysis_model_id".to_string(), model.model_id.0.clone())]),
        ));
    };
    if !em_domain.enabled {
        return Err(operation_error(
            ANALYSIS_RUN_ELECTROMAGNETIC_OPERATION,
            ANALYSIS_RUN_ELECTROMAGNETIC_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_ELECTROMAGNETIC.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_electromagnetic requires electromagnetic domain enabled=true",
            BTreeMap::from([("analysis_model_id".to_string(), model.model_id.0.clone())]),
        ));
    }
    if !em_domain.reference_frequency_hz.is_finite() || em_domain.reference_frequency_hz <= 0.0 {
        return Err(operation_error(
            ANALYSIS_RUN_ELECTROMAGNETIC_OPERATION,
            ANALYSIS_RUN_ELECTROMAGNETIC_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_ELECTROMAGNETIC.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_electromagnetic requires finite positive reference_frequency_hz",
            BTreeMap::from([(
                "reference_frequency_hz".to_string(),
                em_domain.reference_frequency_hz.to_string(),
            )]),
        ));
    }
    if !em_domain.applied_current_a.is_finite() || em_domain.applied_current_a <= 0.0 {
        return Err(operation_error(
            ANALYSIS_RUN_ELECTROMAGNETIC_OPERATION,
            ANALYSIS_RUN_ELECTROMAGNETIC_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_ELECTROMAGNETIC.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_electromagnetic requires finite positive applied_current_a",
            BTreeMap::from([(
                "applied_current_a".to_string(),
                em_domain.applied_current_a.to_string(),
            )]),
        ));
    }
    if !options.residual_target.is_finite() || options.residual_target <= 0.0 {
        return Err(operation_error(
            ANALYSIS_RUN_ELECTROMAGNETIC_OPERATION,
            ANALYSIS_RUN_ELECTROMAGNETIC_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_ELECTROMAGNETIC.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_electromagnetic requires residual_target to be finite and positive",
            BTreeMap::from([(
                "residual_target".to_string(),
                options.residual_target.to_string(),
            )]),
        ));
    }
    if !options.harmonic_tolerance.is_finite() || options.harmonic_tolerance <= 0.0 {
        return Err(operation_error(
            ANALYSIS_RUN_ELECTROMAGNETIC_OPERATION,
            ANALYSIS_RUN_ELECTROMAGNETIC_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_ELECTROMAGNETIC.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_electromagnetic requires harmonic_tolerance to be finite and positive",
            BTreeMap::from([(
                "harmonic_tolerance".to_string(),
                options.harmonic_tolerance.to_string(),
            )]),
        ));
    }
    if options.harmonic_max_iterations == 0 {
        return Err(operation_error(
            ANALYSIS_RUN_ELECTROMAGNETIC_OPERATION,
            ANALYSIS_RUN_ELECTROMAGNETIC_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_ELECTROMAGNETIC.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_electromagnetic requires harmonic_max_iterations greater than zero",
            BTreeMap::from([(
                "harmonic_max_iterations".to_string(),
                options.harmonic_max_iterations.to_string(),
            )]),
        ));
    }

    let prep_context = resolve_run_prep_context(
        model,
        options.prep_artifact_id.as_deref(),
        options.prep_context,
        ANALYSIS_RUN_ELECTROMAGNETIC_OPERATION,
        ANALYSIS_RUN_ELECTROMAGNETIC_OP_VERSION,
        &context,
    )?;

    let sweep_frequency_hz = normalize_em_sweep_frequency_hz(
        em_domain.reference_frequency_hz,
        options.sweep_enabled,
        &options.sweep_frequency_hz,
    )
    .ok_or_else(|| {
        operation_error(
            ANALYSIS_RUN_ELECTROMAGNETIC_OPERATION,
            ANALYSIS_RUN_ELECTROMAGNETIC_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_ELECTROMAGNETIC.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_electromagnetic sweep_frequency_hz must contain finite positive values",
            BTreeMap::new(),
        )
    })?;
    let solve_options = ElectromagneticSolveOptions {
        prep_context: to_fea_prep_context(prep_context, options.prep_calibration_profile),
        residual_target: options.residual_target,
        harmonic_tolerance: options.harmonic_tolerance,
        harmonic_max_iterations: options.harmonic_max_iterations,
    };
    let mut sweep_runs = Vec::with_capacity(sweep_frequency_hz.len());
    let mut sweep_peak_flux_density = Vec::with_capacity(sweep_frequency_hz.len());
    let mut sweep_solve_quality = Vec::with_capacity(sweep_frequency_hz.len());
    for frequency_hz in &sweep_frequency_hz {
        let mut sweep_model = model.clone();
        if let Some(domain) = sweep_model.electromagnetic.as_mut() {
            domain.reference_frequency_hz = *frequency_hz;
        }
        let sweep_run =
            run_electromagnetic_with_options(&sweep_model, backend, solve_options.clone())
                .map_err(|err| {
                    map_fea_run_error(
                        ANALYSIS_RUN_ELECTROMAGNETIC_OPERATION,
                        ANALYSIS_RUN_ELECTROMAGNETIC_OP_VERSION,
                        "RM.FEA.RUN_ELECTROMAGNETIC.SOLVER_MODEL_INVALID",
                        "RM.FEA.RUN_ELECTROMAGNETIC.CANCELLED",
                        model,
                        &context,
                        err,
                    )
                })?;
        sweep_peak_flux_density.push(peak_abs_field_value(
            &sweep_run.magnetic_flux_density_magnitude_field,
        ));
        sweep_solve_quality.push(sweep_run.solve_quality);
        sweep_runs.push(sweep_run);
    }
    let sweep_metrics = summarize_em_sweep(&sweep_frequency_hz, &sweep_peak_flux_density);
    let primary_index =
        nearest_frequency_index(&sweep_frequency_hz, em_domain.reference_frequency_hz).unwrap_or(0);
    let em_run = sweep_runs[primary_index].clone();
    let mut run = em_run.run.clone();
    run.diagnostics.push(runmat_analysis_fea::diagnostics::FeaDiagnostic {
        code: "FEA_EM_SWEEP".to_string(),
        severity: if sweep_metrics.sweep_count > 1 {
            runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info
        } else {
            runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "sweep_count={} resonance_peak_frequency_hz={} resonance_peak_flux_density={} resonance_bandwidth_hz={} resonance_quality_factor={} resonance_flux_gain={}",
            sweep_metrics.sweep_count,
            sweep_metrics.resonance_peak_frequency_hz.unwrap_or(0.0),
            sweep_metrics.resonance_peak_flux_density.unwrap_or(0.0),
            sweep_metrics.resonance_bandwidth_hz.unwrap_or(0.0),
            sweep_metrics.resonance_quality_factor.unwrap_or(0.0),
            sweep_metrics.resonance_flux_gain.unwrap_or(0.0),
        ),
    });
    let solver_convergence = if run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_EM_STATIC"
            && diag.severity == runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info
    }) {
        QualityGate::Pass
    } else {
        QualityGate::Warn
    };
    let mut result_quality = if em_run.solve_quality >= 0.85 {
        QualityGate::Pass
    } else if em_run.solve_quality >= 0.6 {
        QualityGate::Warn
    } else {
        QualityGate::Fail
    };
    let mut quality_reasons = Vec::new();
    let em_conductivity_spread_ratio = diagnostic_metric(
        &run.diagnostics,
        "FEA_EM_STATIC",
        "conductivity_spread_ratio",
    );
    let em_assignment_heterogeneity_index = diagnostic_metric(
        &run.diagnostics,
        "FEA_EM_STATIC",
        "electromagnetic_material_heterogeneity_index",
    );
    let em_assignment_coverage_ratio = diagnostic_metric(
        &run.diagnostics,
        "FEA_EM_STATIC",
        "assignment_coverage_ratio",
    );
    let em_fallback_coefficient_ratio = diagnostic_metric(
        &run.diagnostics,
        "FEA_EM_STATIC",
        "fallback_coefficient_ratio",
    );
    let em_region_contrast_index = diagnostic_metric(
        &run.diagnostics,
        "FEA_EM_STATIC",
        "region_coefficient_contrast_index",
    );
    let em_condition_number_estimate = diagnostic_metric(
        &run.diagnostics,
        "FEA_EM_STATIC",
        "condition_number_estimate",
    );
    let em_source_realization_ratio = diagnostic_metric(
        &run.diagnostics,
        "FEA_EM_STATIC",
        "source_realization_ratio",
    );
    let em_source_region_coverage_ratio = diagnostic_metric(
        &run.diagnostics,
        "FEA_EM_STATIC",
        "source_region_coverage_ratio",
    );
    let em_source_material_alignment_ratio = diagnostic_metric(
        &run.diagnostics,
        "FEA_EM_STATIC",
        "source_material_alignment_ratio",
    );
    let em_source_overlap_ratio =
        diagnostic_metric(&run.diagnostics, "FEA_EM_STATIC", "source_overlap_ratio");
    let em_source_interference_index = diagnostic_metric(
        &run.diagnostics,
        "FEA_EM_STATIC",
        "source_interference_index",
    );
    let em_boundary_anchor_ratio =
        diagnostic_metric(&run.diagnostics, "FEA_EM_STATIC", "boundary_anchor_ratio");
    let em_boundary_condition_localization_ratio = diagnostic_metric(
        &run.diagnostics,
        "FEA_EM_STATIC",
        "boundary_condition_localization_ratio",
    );
    let em_ground_anchor_effectiveness_ratio = diagnostic_metric(
        &run.diagnostics,
        "FEA_EM_STATIC",
        "ground_anchor_effectiveness_ratio",
    );
    let em_insulation_leakage_ratio = diagnostic_metric(
        &run.diagnostics,
        "FEA_EM_STATIC",
        "insulation_leakage_ratio",
    );
    let em_flux_divergence_ratio =
        diagnostic_metric(&run.diagnostics, "FEA_EM_STATIC", "flux_divergence_ratio");
    let em_energy_imbalance_ratio =
        diagnostic_metric(&run.diagnostics, "FEA_EM_STATIC", "energy_imbalance_ratio");
    let em_boundary_energy_ratio =
        diagnostic_metric(&run.diagnostics, "FEA_EM_STATIC", "boundary_energy_ratio");
    let em_boundary_penalty_conditioning_contribution = diagnostic_metric(
        &run.diagnostics,
        "FEA_EM_STATIC",
        "boundary_penalty_conditioning_contribution",
    );
    let em_source_region_energy_consistency_ratio = diagnostic_metric(
        &run.diagnostics,
        "FEA_EM_STATIC",
        "source_region_energy_consistency_ratio",
    );
    let em_real_residual_norm =
        diagnostic_metric(&run.diagnostics, "FEA_EM_STATIC", "real_residual_norm");
    let em_imag_residual_norm =
        diagnostic_metric(&run.diagnostics, "FEA_EM_STATIC", "imag_residual_norm");
    let em_sweep_count = diagnostic_metric(&run.diagnostics, "FEA_EM_SWEEP", "sweep_count");
    let em_resonance_quality_factor =
        diagnostic_metric(&run.diagnostics, "FEA_EM_SWEEP", "resonance_quality_factor");
    let ElectromagneticQualityThresholds {
        em_spread_threshold,
        em_heterogeneity_threshold,
        em_coverage_min_threshold,
        em_fallback_max_threshold,
        em_contrast_max_threshold,
        em_conditioning_max_threshold,
        em_source_realization_min_threshold,
        em_source_region_coverage_min_threshold,
        em_source_material_alignment_min_threshold,
        em_source_overlap_max_threshold,
        em_source_interference_max_threshold,
        em_boundary_anchor_min_threshold,
        em_boundary_localization_min_threshold,
        em_ground_effectiveness_min_threshold,
        em_insulation_leakage_max_threshold,
        em_divergence_max_threshold,
        em_energy_imbalance_max_threshold,
        em_boundary_energy_min_threshold,
        em_boundary_penalty_contribution_max_threshold,
        em_source_region_energy_consistency_min_threshold,
        em_real_residual_max_threshold,
        em_imag_residual_max_threshold,
    } = electromagnetic_thresholds_for_policy(options.quality_policy);
    let (em_sweep_count_min_threshold, em_resonance_q_min_threshold) =
        electromagnetic_sweep_thresholds_for_policy(options.quality_policy);
    let em_spread_breach = em_conductivity_spread_ratio
        .map(|value| value > em_spread_threshold)
        .unwrap_or(false);
    let em_heterogeneity_breach = em_assignment_heterogeneity_index
        .map(|value| value > em_heterogeneity_threshold)
        .unwrap_or(false);
    let em_coverage_breach = em_assignment_coverage_ratio
        .map(|value| value < em_coverage_min_threshold)
        .unwrap_or(false);
    let em_fallback_breach = em_fallback_coefficient_ratio
        .map(|value| value > em_fallback_max_threshold)
        .unwrap_or(false);
    let em_contrast_breach = em_region_contrast_index
        .map(|value| value > em_contrast_max_threshold)
        .unwrap_or(false);
    let em_conditioning_breach = em_condition_number_estimate
        .map(|value| value > em_conditioning_max_threshold)
        .unwrap_or(false);
    let em_source_realization_breach = em_source_realization_ratio
        .map(|value| value < em_source_realization_min_threshold)
        .unwrap_or(false);
    let em_source_region_coverage_breach = em_source_region_coverage_ratio
        .map(|value| value < em_source_region_coverage_min_threshold)
        .unwrap_or(false);
    let em_source_material_alignment_breach = em_source_material_alignment_ratio
        .map(|value| value < em_source_material_alignment_min_threshold)
        .unwrap_or(false);
    let em_source_overlap_breach = em_source_overlap_ratio
        .map(|value| value > em_source_overlap_max_threshold)
        .unwrap_or(false);
    let em_source_interference_breach = em_source_interference_index
        .map(|value| value > em_source_interference_max_threshold)
        .unwrap_or(false);
    let em_boundary_anchor_breach = em_boundary_anchor_ratio
        .map(|value| value < em_boundary_anchor_min_threshold)
        .unwrap_or(false);
    let em_boundary_localization_breach = em_boundary_condition_localization_ratio
        .map(|value| value < em_boundary_localization_min_threshold)
        .unwrap_or(false);
    let em_ground_effectiveness_breach = em_ground_anchor_effectiveness_ratio
        .map(|value| value < em_ground_effectiveness_min_threshold)
        .unwrap_or(false);
    let em_insulation_leakage_breach = em_insulation_leakage_ratio
        .map(|value| value > em_insulation_leakage_max_threshold)
        .unwrap_or(false);
    let em_divergence_breach = em_flux_divergence_ratio
        .map(|value| value > em_divergence_max_threshold)
        .unwrap_or(false);
    let em_energy_imbalance_breach = em_energy_imbalance_ratio
        .map(|value| value > em_energy_imbalance_max_threshold)
        .unwrap_or(false);
    let em_boundary_energy_breach = em_boundary_energy_ratio
        .map(|value| value < em_boundary_energy_min_threshold)
        .unwrap_or(false);
    let em_boundary_penalty_contribution_breach = em_boundary_penalty_conditioning_contribution
        .map(|value| value > em_boundary_penalty_contribution_max_threshold)
        .unwrap_or(false);
    let em_source_region_energy_consistency_breach = em_source_region_energy_consistency_ratio
        .map(|value| value < em_source_region_energy_consistency_min_threshold)
        .unwrap_or(false);
    let em_real_residual_breach = em_real_residual_norm
        .map(|value| value > em_real_residual_max_threshold)
        .unwrap_or(false);
    let em_imag_residual_breach = em_imag_residual_norm
        .map(|value| value > em_imag_residual_max_threshold)
        .unwrap_or(false);
    let sweep_governance_active = options.sweep_enabled || !options.sweep_frequency_hz.is_empty();
    let em_sweep_coverage_breach = sweep_governance_active
        && em_sweep_count
            .map(|value| value < em_sweep_count_min_threshold)
            .unwrap_or(false);
    let em_resonance_sharpness_breach = sweep_governance_active
        && em_resonance_quality_factor
            .map(|value| value < em_resonance_q_min_threshold)
            .unwrap_or(false);
    if (em_spread_breach
        || em_heterogeneity_breach
        || em_coverage_breach
        || em_fallback_breach
        || em_contrast_breach
        || em_conditioning_breach
        || em_source_realization_breach
        || em_source_region_coverage_breach
        || em_source_material_alignment_breach
        || em_source_overlap_breach
        || em_source_interference_breach
        || em_boundary_anchor_breach
        || em_boundary_localization_breach
        || em_ground_effectiveness_breach
        || em_insulation_leakage_breach
        || em_divergence_breach
        || em_energy_imbalance_breach
        || em_boundary_energy_breach
        || em_boundary_penalty_contribution_breach
        || em_source_region_energy_consistency_breach
        || em_real_residual_breach
        || em_imag_residual_breach
        || em_sweep_coverage_breach
        || em_resonance_sharpness_breach)
        && result_quality == QualityGate::Pass
    {
        result_quality = QualityGate::Warn;
    }
    if solver_convergence == QualityGate::Warn {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::SolverNotConverged,
            detail: "electromagnetic solver convergence gate is warning".to_string(),
        });
    }
    if result_quality != QualityGate::Pass {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ElectromagneticSolveQualityLow,
            detail: "electromagnetic static solve quality below production target".to_string(),
        });
    }
    if em_spread_breach {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ElectromagneticConductivitySpreadHigh,
            detail: format!(
                "electromagnetic conductivity spread ratio {} exceeds threshold {}",
                em_conductivity_spread_ratio.unwrap_or(0.0),
                em_spread_threshold
            ),
        });
    }
    if em_heterogeneity_breach {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ElectromagneticMaterialHeterogeneityHigh,
            detail: format!(
                "electromagnetic material heterogeneity index {} exceeds threshold {}",
                em_assignment_heterogeneity_index.unwrap_or(0.0),
                em_heterogeneity_threshold
            ),
        });
    }
    if em_coverage_breach {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ElectromagneticAssignmentCoverageLow,
            detail: format!(
                "electromagnetic assignment coverage ratio {} is below threshold {}",
                em_assignment_coverage_ratio.unwrap_or(0.0),
                em_coverage_min_threshold
            ),
        });
    }
    if em_fallback_breach {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ElectromagneticFallbackCoefficientHigh,
            detail: format!(
                "electromagnetic fallback coefficient ratio {} exceeds threshold {}",
                em_fallback_coefficient_ratio.unwrap_or(0.0),
                em_fallback_max_threshold
            ),
        });
    }
    if em_contrast_breach {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ElectromagneticRegionContrastHigh,
            detail: format!(
                "electromagnetic region coefficient contrast index {} exceeds threshold {}",
                em_region_contrast_index.unwrap_or(0.0),
                em_contrast_max_threshold
            ),
        });
    }
    if em_conditioning_breach {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ElectromagneticConditioningHigh,
            detail: format!(
                "electromagnetic condition-number estimate {} exceeds threshold {}",
                em_condition_number_estimate.unwrap_or(0.0),
                em_conditioning_max_threshold
            ),
        });
    }
    if em_source_realization_breach {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ElectromagneticSourceRealizationLow,
            detail: format!(
                "electromagnetic source realization ratio {} is below threshold {}",
                em_source_realization_ratio.unwrap_or(0.0),
                em_source_realization_min_threshold
            ),
        });
    }
    if em_source_region_coverage_breach {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ElectromagneticSourceRegionCoverageLow,
            detail: format!(
                "electromagnetic source region coverage ratio {} is below threshold {}",
                em_source_region_coverage_ratio.unwrap_or(0.0),
                em_source_region_coverage_min_threshold
            ),
        });
    }
    if em_source_material_alignment_breach {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ElectromagneticSourceMaterialAlignmentLow,
            detail: format!(
                "electromagnetic source material alignment ratio {} is below threshold {}",
                em_source_material_alignment_ratio.unwrap_or(0.0),
                em_source_material_alignment_min_threshold
            ),
        });
    }
    if em_source_overlap_breach {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ElectromagneticSourceOverlapHigh,
            detail: format!(
                "electromagnetic source overlap ratio {} exceeds threshold {}",
                em_source_overlap_ratio.unwrap_or(0.0),
                em_source_overlap_max_threshold
            ),
        });
    }
    if em_source_interference_breach {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ElectromagneticSourceInterferenceHigh,
            detail: format!(
                "electromagnetic source interference index {} exceeds threshold {}",
                em_source_interference_index.unwrap_or(0.0),
                em_source_interference_max_threshold
            ),
        });
    }
    if em_boundary_anchor_breach {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ElectromagneticBoundaryAnchoringLow,
            detail: format!(
                "electromagnetic boundary anchor ratio {} is below threshold {}",
                em_boundary_anchor_ratio.unwrap_or(0.0),
                em_boundary_anchor_min_threshold
            ),
        });
    }
    if em_boundary_localization_breach {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ElectromagneticBoundaryLocalizationLow,
            detail: format!(
                "electromagnetic boundary condition localization ratio {} is below threshold {}",
                em_boundary_condition_localization_ratio.unwrap_or(0.0),
                em_boundary_localization_min_threshold
            ),
        });
    }
    if em_ground_effectiveness_breach {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ElectromagneticGroundAnchorEffectivenessLow,
            detail: format!(
                "electromagnetic ground anchor effectiveness ratio {} is below threshold {}",
                em_ground_anchor_effectiveness_ratio.unwrap_or(0.0),
                em_ground_effectiveness_min_threshold
            ),
        });
    }
    if em_insulation_leakage_breach {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ElectromagneticInsulationLeakageHigh,
            detail: format!(
                "electromagnetic insulation leakage ratio {} exceeds threshold {}",
                em_insulation_leakage_ratio.unwrap_or(0.0),
                em_insulation_leakage_max_threshold
            ),
        });
    }
    if em_divergence_breach {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ElectromagneticFluxDivergenceHigh,
            detail: format!(
                "electromagnetic flux divergence ratio {} exceeds threshold {}",
                em_flux_divergence_ratio.unwrap_or(0.0),
                em_divergence_max_threshold
            ),
        });
    }
    if em_energy_imbalance_breach {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ElectromagneticEnergyImbalanceHigh,
            detail: format!(
                "electromagnetic energy imbalance ratio {} exceeds threshold {}",
                em_energy_imbalance_ratio.unwrap_or(0.0),
                em_energy_imbalance_max_threshold
            ),
        });
    }
    if em_boundary_energy_breach {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ElectromagneticBoundaryEnergyLow,
            detail: format!(
                "electromagnetic boundary energy ratio {} is below threshold {}",
                em_boundary_energy_ratio.unwrap_or(0.0),
                em_boundary_energy_min_threshold
            ),
        });
    }
    if em_boundary_penalty_contribution_breach {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ElectromagneticBoundaryPenaltyConditioningHigh,
            detail: format!(
                "electromagnetic boundary penalty conditioning contribution {} exceeds threshold {}",
                em_boundary_penalty_conditioning_contribution.unwrap_or(0.0),
                em_boundary_penalty_contribution_max_threshold
            ),
        });
    }
    if em_source_region_energy_consistency_breach {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ElectromagneticSourceRegionEnergyConsistencyLow,
            detail: format!(
                "electromagnetic source-region energy consistency ratio {} is below threshold {}",
                em_source_region_energy_consistency_ratio.unwrap_or(0.0),
                em_source_region_energy_consistency_min_threshold
            ),
        });
    }
    if em_real_residual_breach {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ElectromagneticRealResidualHigh,
            detail: format!(
                "electromagnetic real residual norm {} exceeds threshold {}",
                em_real_residual_norm.unwrap_or(0.0),
                em_real_residual_max_threshold
            ),
        });
    }
    if em_imag_residual_breach {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ElectromagneticImagResidualHigh,
            detail: format!(
                "electromagnetic imaginary residual norm {} exceeds threshold {}",
                em_imag_residual_norm.unwrap_or(0.0),
                em_imag_residual_max_threshold
            ),
        });
    }
    if em_sweep_coverage_breach {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ElectromagneticSweepCoverageLow,
            detail: format!(
                "electromagnetic sweep count {} is below threshold {}",
                em_sweep_count.unwrap_or(0.0),
                em_sweep_count_min_threshold
            ),
        });
    }
    if em_resonance_sharpness_breach {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ElectromagneticResonanceSharpnessLow,
            detail: format!(
                "electromagnetic resonance quality factor {} is below threshold {}",
                em_resonance_quality_factor.unwrap_or(0.0),
                em_resonance_q_min_threshold
            ),
        });
    }

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
    let solver_backend = run.solver_backend.clone();
    let solver_device_apply_k_ratio = run.solver_device_apply_k_ratio;
    let solver_host_sync_count = run.solver_host_sync_count;
    let solver_method = run.solver_method.clone();
    let preconditioner = run.preconditioner.clone();

    let result = AnalysisRunResult {
        run_id: storage::next_run_id(),
        run,
        modal_results: None,
        thermal_results: None,
        transient_results: None,
        nonlinear_results: None,
        electromagnetic_results: Some(ElectromagneticResultsData {
            electromagnetic_payload_version: "electromagnetic_results/v1".to_string(),
            reference_frequency_hz: em_run.reference_frequency_hz,
            applied_current_a: em_run.applied_current_a,
            vector_potential_real: em_run.vector_potential_real_field,
            vector_potential_imag: em_run.vector_potential_imag_field,
            magnetic_flux_density_real: em_run.magnetic_flux_density_real_field,
            magnetic_flux_density_imag: em_run.magnetic_flux_density_imag_field,
            magnetic_flux_density_magnitude: em_run.magnetic_flux_density_magnitude_field,
            magnetic_field_real: em_run.magnetic_field_real_field,
            magnetic_field_imag: em_run.magnetic_field_imag_field,
            current_density_real: em_run.current_density_real_field,
            current_density_imag: em_run.current_density_imag_field,
            electric_field_real: em_run.electric_field_real_field,
            electric_field_imag: em_run.electric_field_imag_field,
            power_loss_density: em_run.power_loss_density_field,
            energy_density: em_run.energy_density_field,
            residual_real: em_run.residual_real_field,
            residual_imag: em_run.residual_imag_field,
            electric_flux_density_real: em_run.electric_flux_density_real_field,
            electric_flux_density_imag: em_run.electric_flux_density_imag_field,
            poynting_vector_real: em_run.poynting_vector_real_field,
            poynting_vector_imag: em_run.poynting_vector_imag_field,
            sweep_frequency_hz,
            sweep_peak_flux_density,
            sweep_solve_quality,
            resonance_peak_frequency_hz: sweep_metrics.resonance_peak_frequency_hz,
            resonance_peak_flux_density: sweep_metrics.resonance_peak_flux_density,
            resonance_bandwidth_hz: sweep_metrics.resonance_bandwidth_hz,
            resonance_quality_factor: sweep_metrics.resonance_quality_factor,
            resonance_flux_gain: sweep_metrics.resonance_flux_gain,
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
            preconditioner,
            quality_policy: contracts::format_quality_policy(options.quality_policy),
            fallback_events: Vec::new(),
        },
    };

    persist_fea_run_result_with_progress(
        ANALYSIS_RUN_ELECTROMAGNETIC_OPERATION,
        ANALYSIS_RUN_ELECTROMAGNETIC_OP_VERSION,
        "RM.FEA.RUN_ELECTROMAGNETIC.ARTIFACT_STORE_FAILED",
        &context,
        &result,
    )?;

    Ok(OperationEnvelope::new(
        ANALYSIS_RUN_ELECTROMAGNETIC_OPERATION,
        ANALYSIS_RUN_ELECTROMAGNETIC_OP_VERSION,
        &context,
        result,
    ))
}

fn collect_analysis_result_fields(run_result: &AnalysisRunResult) -> Vec<AnalysisField> {
    let mut fields = Vec::new();
    let mut seen = HashSet::new();

    for field in &run_result.run.fields {
        push_analysis_result_field(&mut fields, &mut seen, field);
    }

    if let Some(modal) = run_result.modal_results.as_ref() {
        for field in &modal.mode_shapes {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
    }

    if let Some(thermal) = run_result.thermal_results.as_ref() {
        for field in &thermal.temperature_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
        for field in &thermal.temperature_gradient_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
        for field in &thermal.heat_flux_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
        for field in &thermal.heat_source_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
        for field in &thermal.boundary_heat_flux_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
    }

    if let Some(transient) = run_result.transient_results.as_ref() {
        for field in &transient.displacement_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
        for field in &transient.velocity_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
        for field in &transient.acceleration_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
        for field in &transient.von_mises_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
        for field in &transient.kinetic_energy_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
        for field in &transient.strain_energy_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
        for field in &transient.residual_norm_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
        for field in &transient.thermo_mechanical_temperature_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
        for field in &transient.thermo_mechanical_thermal_strain_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
        for field in &transient.thermo_mechanical_thermal_stress_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
        for field in &transient.thermo_mechanical_displacement_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
        for field in &transient.thermo_mechanical_von_mises_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
        for field in &transient.thermo_mechanical_coupling_residual_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
        for field in &transient.electro_thermal_temperature_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
        for field in &transient.electro_thermal_thermal_residual_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
    }

    if let Some(nonlinear) = run_result.nonlinear_results.as_ref() {
        for field in &nonlinear.displacement_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
        for field in &nonlinear.von_mises_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
        for field in &nonlinear.plastic_strain_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
        for field in &nonlinear.equivalent_plastic_strain_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
        for field in &nonlinear.contact_pressure_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
        for field in &nonlinear.contact_gap_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
        for field in &nonlinear.load_factor_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
        for field in &nonlinear.residual_norm_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
        for field in &nonlinear.thermo_mechanical_temperature_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
        for field in &nonlinear.thermo_mechanical_thermal_strain_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
        for field in &nonlinear.thermo_mechanical_thermal_stress_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
        for field in &nonlinear.thermo_mechanical_displacement_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
        for field in &nonlinear.thermo_mechanical_von_mises_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
        for field in &nonlinear.thermo_mechanical_coupling_residual_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
        for field in &nonlinear.electro_thermal_temperature_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
        for field in &nonlinear.electro_thermal_thermal_residual_snapshots {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
    }

    if let Some(electromagnetic) = run_result.electromagnetic_results.as_ref() {
        for field in [
            &electromagnetic.vector_potential_real,
            &electromagnetic.vector_potential_imag,
            &electromagnetic.magnetic_flux_density_real,
            &electromagnetic.magnetic_flux_density_imag,
            &electromagnetic.magnetic_flux_density_magnitude,
            &electromagnetic.magnetic_field_real,
            &electromagnetic.magnetic_field_imag,
            &electromagnetic.current_density_real,
            &electromagnetic.current_density_imag,
            &electromagnetic.electric_field_real,
            &electromagnetic.electric_field_imag,
            &electromagnetic.power_loss_density,
            &electromagnetic.energy_density,
            &electromagnetic.residual_real,
            &electromagnetic.residual_imag,
            &electromagnetic.electric_flux_density_real,
            &electromagnetic.electric_flux_density_imag,
            &electromagnetic.poynting_vector_real,
            &electromagnetic.poynting_vector_imag,
        ] {
            push_analysis_result_field(&mut fields, &mut seen, field);
        }
    }

    fields
}

fn push_analysis_result_field(
    fields: &mut Vec<AnalysisField>,
    seen: &mut HashSet<String>,
    field: &AnalysisField,
) {
    if !seen.insert(field.field_id.clone()) {
        return;
    }
    fields.push(field.clone());
}

fn filter_analysis_fields_by_indices(
    fields: &[AnalysisField],
    indices: &[usize],
) -> Vec<AnalysisField> {
    if fields.is_empty() {
        return Vec::new();
    }
    indices
        .iter()
        .filter_map(|index| fields.get(*index).cloned())
        .collect()
}

pub(crate) fn analysis_run_field_ids(run_result: &AnalysisRunResult) -> Vec<String> {
    collect_analysis_result_fields(run_result)
        .into_iter()
        .map(|field| field.field_id)
        .collect()
}

pub fn analysis_results_op(
    run_result: &AnalysisRunResult,
    query: AnalysisResultsQuery,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisResultsData>, OperationErrorEnvelope> {
    let mut collected_fields = collect_analysis_result_fields(run_result);

    if !query.include_fields.is_empty() {
        let mut filtered = Vec::new();
        for requested in &query.include_fields {
            let Some(field) = collected_fields
                .iter()
                .find(|field| &field.field_id == requested)
            else {
                return Err(operation_error(
                    ANALYSIS_RESULTS_OPERATION,
                    ANALYSIS_RESULTS_OP_VERSION,
                    &context,
                    OperationErrorSpec {
                        error_code: "RM.FEA.RESULTS.FIELD_NOT_FOUND",
                        error_type: OperationErrorType::Input,
                        retryable: false,
                        severity: OperationErrorSeverity::Error,
                    },
                    format!("requested FEA field '{requested}' was not produced by run"),
                    BTreeMap::from([
                        ("requested_field".to_string(), requested.clone()),
                        (
                            "available_fields".to_string(),
                            collected_fields
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
        collected_fields = filtered;
    }
    let field_descriptors = collected_fields
        .iter()
        .map(AnalysisFieldDescriptor::from_field)
        .collect::<Vec<_>>();

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
    } else if let Some(thermal) = run_result.thermal_results.as_ref() {
        let count = thermal
            .time_points_s
            .len()
            .min(thermal.temperature_snapshots.len());
        let max_residual = thermal.residual_norms.iter().copied().reduce(f64::max);
        let final_step_converged = max_residual.map(|value| value <= 1.0e-6);
        if count == 0 {
            (0, None, None, max_residual, final_step_converged)
        } else {
            (
                count,
                thermal.time_points_s.first().copied(),
                thermal.time_points_s.get(count - 1).copied(),
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
    let thermo_field_clamp_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_TM_TRANSIENT",
        "field_clamp_ratio",
    )
    .or_else(|| {
        diagnostic_metric(
            &run_result.run.diagnostics,
            "FEA_TM_NONLINEAR",
            "field_clamp_ratio",
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
    let electro_transient_time_scale_mean = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_ET_TRANSIENT",
        "time_scale_mean",
    );
    let electro_nonlinear_severity = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_ET_NONLINEAR",
        "severity_peak",
    )
    .or_else(|| diagnostic_metric(&run_result.run.diagnostics, "FEA_ET_NONLINEAR", "severity"));
    let electro_nonlinear_time_scale_mean = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_ET_NONLINEAR",
        "time_scale_mean",
    );
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
    let plastic_nonlinear_severity_mean = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_PLASTIC_NONLINEAR",
        "severity_mean",
    );
    let plastic_load_realization_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_PLASTIC_NONLINEAR",
        "load_realization_ratio",
    );
    let plastic_load_amplification_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_PLASTIC_NONLINEAR",
        "load_amplification_ratio",
    );
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
    let contact_nonlinear_severity_mean = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_CONTACT_NONLINEAR",
        "severity_mean",
    );
    let contact_load_realization_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_CONTACT_NONLINEAR",
        "load_realization_ratio",
    );
    let contact_load_amplification_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_CONTACT_NONLINEAR",
        "load_amplification_ratio",
    );
    let thermal_max_residual_norm = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_THERMAL_STABILITY",
        "max_residual_norm",
    );
    let thermal_min_temperature_k = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_THERMAL_STABILITY",
        "min_temperature_k",
    );
    let thermal_max_temperature_k = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_THERMAL_STABILITY",
        "max_temperature_k",
    );
    let thermal_conductivity_spread_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_THERMAL_CONSTITUTIVE",
        "conductivity_spread_ratio",
    );
    let thermal_heat_capacity_spread_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_THERMAL_CONSTITUTIVE",
        "heat_capacity_spread_ratio",
    );
    let thermal_spatial_gradient_index = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_THERMAL_OUTCOME",
        "spatial_gradient_index",
    );
    let thermal_monotonic_response_fraction = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_THERMAL_OUTCOME",
        "monotonic_response_fraction",
    );
    let thermal_response_realization_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_THERMAL_OUTCOME",
        "thermal_response_realization_ratio",
    );
    let electromagnetic_enabled =
        diagnostic_metric_bool(&run_result.run.diagnostics, "FEA_EM_STATIC", "enabled");
    let electromagnetic_reference_frequency_hz = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "reference_frequency_hz",
    );
    let electromagnetic_applied_current_a = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "applied_current_a",
    );
    let electromagnetic_solve_quality = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "solve_quality",
    );
    let electromagnetic_conductivity_spread_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "conductivity_spread_ratio",
    );
    let electromagnetic_relative_permittivity_spread_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "relative_permittivity_spread_ratio",
    );
    let electromagnetic_relative_permeability_spread_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "relative_permeability_spread_ratio",
    );
    let electromagnetic_material_heterogeneity_index = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "electromagnetic_material_heterogeneity_index",
    );
    let electromagnetic_assignment_coverage_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "assignment_coverage_ratio",
    );
    let electromagnetic_fallback_coefficient_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "fallback_coefficient_ratio",
    );
    let electromagnetic_region_coefficient_contrast_index = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "region_coefficient_contrast_index",
    );
    let electromagnetic_condition_number_estimate = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "condition_number_estimate",
    );
    let electromagnetic_source_realization_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "source_realization_ratio",
    );
    let electromagnetic_source_region_coverage_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "source_region_coverage_ratio",
    );
    let electromagnetic_source_material_alignment_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "source_material_alignment_ratio",
    );
    let electromagnetic_source_localization_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "source_localization_ratio",
    );
    let electromagnetic_source_overlap_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "source_overlap_ratio",
    );
    let electromagnetic_source_interference_index = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "source_interference_index",
    );
    let electromagnetic_boundary_anchor_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "boundary_anchor_ratio",
    );
    let electromagnetic_boundary_condition_localization_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "boundary_condition_localization_ratio",
    );
    let electromagnetic_ground_anchor_effectiveness_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "ground_anchor_effectiveness_ratio",
    );
    let electromagnetic_insulation_leakage_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "insulation_leakage_ratio",
    );
    let electromagnetic_flux_divergence_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "flux_divergence_ratio",
    );
    let electromagnetic_energy_imbalance_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "energy_imbalance_ratio",
    );
    let electromagnetic_boundary_energy_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "boundary_energy_ratio",
    );
    let electromagnetic_boundary_penalty_conditioning_contribution = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "boundary_penalty_conditioning_contribution",
    );
    let electromagnetic_source_region_energy_consistency_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "source_region_energy_consistency_ratio",
    );
    let electromagnetic_real_residual_norm = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "real_residual_norm",
    );
    let electromagnetic_imag_residual_norm = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "imag_residual_norm",
    );
    let electromagnetic_sweep_count =
        diagnostic_metric(&run_result.run.diagnostics, "FEA_EM_SWEEP", "sweep_count");
    let electromagnetic_resonance_peak_frequency_hz = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_SWEEP",
        "resonance_peak_frequency_hz",
    );
    let electromagnetic_resonance_peak_flux_density = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_SWEEP",
        "resonance_peak_flux_density",
    );
    let electromagnetic_resonance_bandwidth_hz = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_SWEEP",
        "resonance_bandwidth_hz",
    );
    let electromagnetic_resonance_quality_factor = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_SWEEP",
        "resonance_quality_factor",
    );
    let electromagnetic_resonance_flux_gain = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_SWEEP",
        "resonance_flux_gain",
    );

    let summary = AnalysisResultsSummary {
        field_count: field_descriptors.len(),
        total_elements: field_descriptors
            .iter()
            .map(|field| field.element_count)
            .sum(),
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
        thermo_field_clamp_ratio,
        thermo_transient_severity,
        thermo_nonlinear_severity,
        electro_thermal_coupling_enabled,
        electro_thermal_coupling_fingerprint,
        electro_joule_heating_scale,
        electro_conductivity_spread_ratio,
        electro_transient_severity,
        electro_transient_time_scale_mean,
        electro_nonlinear_severity,
        electro_nonlinear_time_scale_mean,
        plastic_nonlinear_severity,
        plastic_nonlinear_severity_mean,
        plastic_load_realization_ratio,
        plastic_load_amplification_ratio,
        contact_nonlinear_severity,
        contact_nonlinear_severity_mean,
        contact_load_realization_ratio,
        contact_load_amplification_ratio,
        thermal_max_residual_norm,
        thermal_min_temperature_k,
        thermal_max_temperature_k,
        thermal_conductivity_spread_ratio,
        thermal_heat_capacity_spread_ratio,
        thermal_spatial_gradient_index,
        thermal_monotonic_response_fraction,
        thermal_response_realization_ratio,
        electromagnetic_enabled,
        electromagnetic_reference_frequency_hz,
        electromagnetic_applied_current_a,
        electromagnetic_solve_quality,
        electromagnetic_conductivity_spread_ratio,
        electromagnetic_relative_permittivity_spread_ratio,
        electromagnetic_relative_permeability_spread_ratio,
        electromagnetic_material_heterogeneity_index,
        electromagnetic_assignment_coverage_ratio,
        electromagnetic_fallback_coefficient_ratio,
        electromagnetic_region_coefficient_contrast_index,
        electromagnetic_condition_number_estimate,
        electromagnetic_source_realization_ratio,
        electromagnetic_source_region_coverage_ratio,
        electromagnetic_source_material_alignment_ratio,
        electromagnetic_source_localization_ratio,
        electromagnetic_source_overlap_ratio,
        electromagnetic_source_interference_index,
        electromagnetic_boundary_anchor_ratio,
        electromagnetic_boundary_condition_localization_ratio,
        electromagnetic_ground_anchor_effectiveness_ratio,
        electromagnetic_insulation_leakage_ratio,
        electromagnetic_flux_divergence_ratio,
        electromagnetic_energy_imbalance_ratio,
        electromagnetic_boundary_energy_ratio,
        electromagnetic_boundary_penalty_conditioning_contribution,
        electromagnetic_source_region_energy_consistency_ratio,
        electromagnetic_real_residual_norm,
        electromagnetic_imag_residual_norm,
        electromagnetic_sweep_count,
        electromagnetic_resonance_peak_frequency_hz,
        electromagnetic_resonance_peak_flux_density,
        electromagnetic_resonance_bandwidth_hz,
        electromagnetic_resonance_quality_factor,
        electromagnetic_resonance_flux_gain,
    };

    let modal_results = if query.include_modal_results && query.include_field_values {
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
                                error_code: "RM.FEA.RESULTS.MODE_NOT_FOUND",
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
                                error_code: "RM.FEA.RESULTS.MODE_NOT_FOUND",
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
                                    error_code: "RM.FEA.RESULTS.MODE_NOT_FOUND",
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
                    mode_units: modal.mode_units,
                    frequency_basis: modal.frequency_basis,
                })
            }
        } else {
            None
        }
    } else {
        None
    };

    let transient_results = if query.include_transient_results && query.include_field_values {
        if let Some(transient) = run_result.transient_results.as_ref() {
            if query.transient_snapshot_indices.is_empty() {
                Some(transient.clone())
            } else {
                let mut time_points_s = Vec::with_capacity(query.transient_snapshot_indices.len());
                let mut displacement_snapshots =
                    Vec::with_capacity(query.transient_snapshot_indices.len());
                let mut velocity_snapshots =
                    Vec::with_capacity(query.transient_snapshot_indices.len());
                let mut acceleration_snapshots =
                    Vec::with_capacity(query.transient_snapshot_indices.len());
                let mut von_mises_snapshots =
                    Vec::with_capacity(query.transient_snapshot_indices.len());
                let mut kinetic_energy_snapshots =
                    Vec::with_capacity(query.transient_snapshot_indices.len());
                let mut strain_energy_snapshots =
                    Vec::with_capacity(query.transient_snapshot_indices.len());
                let mut residual_norm_snapshots =
                    Vec::with_capacity(query.transient_snapshot_indices.len());
                let mut residual_norms = Vec::with_capacity(query.transient_snapshot_indices.len());
                let thermo_mechanical_temperature_snapshots = filter_analysis_fields_by_indices(
                    &transient.thermo_mechanical_temperature_snapshots,
                    &query.transient_snapshot_indices,
                );
                let thermo_mechanical_thermal_strain_snapshots = filter_analysis_fields_by_indices(
                    &transient.thermo_mechanical_thermal_strain_snapshots,
                    &query.transient_snapshot_indices,
                );
                let thermo_mechanical_thermal_stress_snapshots = filter_analysis_fields_by_indices(
                    &transient.thermo_mechanical_thermal_stress_snapshots,
                    &query.transient_snapshot_indices,
                );
                let thermo_mechanical_displacement_snapshots = filter_analysis_fields_by_indices(
                    &transient.thermo_mechanical_displacement_snapshots,
                    &query.transient_snapshot_indices,
                );
                let thermo_mechanical_von_mises_snapshots = filter_analysis_fields_by_indices(
                    &transient.thermo_mechanical_von_mises_snapshots,
                    &query.transient_snapshot_indices,
                );
                let thermo_mechanical_coupling_residual_snapshots =
                    filter_analysis_fields_by_indices(
                        &transient.thermo_mechanical_coupling_residual_snapshots,
                        &query.transient_snapshot_indices,
                    );
                let electro_thermal_temperature_snapshots = filter_analysis_fields_by_indices(
                    &transient.electro_thermal_temperature_snapshots,
                    &query.transient_snapshot_indices,
                );
                let electro_thermal_thermal_residual_snapshots = filter_analysis_fields_by_indices(
                    &transient.electro_thermal_thermal_residual_snapshots,
                    &query.transient_snapshot_indices,
                );

                for &index in &query.transient_snapshot_indices {
                    let time_point = transient.time_points_s.get(index).copied().ok_or_else(|| {
                        operation_error(
                            ANALYSIS_RESULTS_OPERATION,
                            ANALYSIS_RESULTS_OP_VERSION,
                            &context,
                            OperationErrorSpec {
                                error_code: "RM.FEA.RESULTS.TRANSIENT_SNAPSHOT_NOT_FOUND",
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
                                    error_code: "RM.FEA.RESULTS.TRANSIENT_SNAPSHOT_NOT_FOUND",
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
                    let velocity = transient.velocity_snapshots.get(index).cloned().ok_or_else(|| {
                        operation_error(
                            ANALYSIS_RESULTS_OPERATION,
                            ANALYSIS_RESULTS_OP_VERSION,
                            &context,
                            OperationErrorSpec {
                                error_code: "RM.FEA.RESULTS.TRANSIENT_SNAPSHOT_NOT_FOUND",
                                error_type: OperationErrorType::Input,
                                retryable: false,
                                severity: OperationErrorSeverity::Error,
                            },
                            format!(
                                "requested transient snapshot index '{index}' is missing velocity data"
                            ),
                            BTreeMap::from([
                                ("requested_snapshot_index".to_string(), index.to_string()),
                                (
                                    "available_velocity_snapshot_count".to_string(),
                                    transient.velocity_snapshots.len().to_string(),
                                ),
                            ]),
                        )
                    })?;
                    let acceleration =
                        transient
                            .acceleration_snapshots
                            .get(index)
                            .cloned()
                            .ok_or_else(|| {
                                operation_error(
                                    ANALYSIS_RESULTS_OPERATION,
                                    ANALYSIS_RESULTS_OP_VERSION,
                                    &context,
                                    OperationErrorSpec {
                                        error_code:
                                            "RM.FEA.RESULTS.TRANSIENT_SNAPSHOT_NOT_FOUND",
                                        error_type: OperationErrorType::Input,
                                        retryable: false,
                                        severity: OperationErrorSeverity::Error,
                                    },
                                    format!(
                                        "requested transient snapshot index '{index}' is missing acceleration data"
                                    ),
                                    BTreeMap::from([
                                        ("requested_snapshot_index".to_string(), index.to_string()),
                                        (
                                            "available_acceleration_snapshot_count".to_string(),
                                            transient.acceleration_snapshots.len().to_string(),
                                        ),
                                    ]),
                                )
                            })?;
                    let von_mises =
                        transient
                            .von_mises_snapshots
                            .get(index)
                            .cloned()
                            .ok_or_else(|| {
                                operation_error(
                                    ANALYSIS_RESULTS_OPERATION,
                                    ANALYSIS_RESULTS_OP_VERSION,
                                    &context,
                                    OperationErrorSpec {
                                        error_code:
                                            "RM.FEA.RESULTS.TRANSIENT_SNAPSHOT_NOT_FOUND",
                                        error_type: OperationErrorType::Input,
                                        retryable: false,
                                        severity: OperationErrorSeverity::Error,
                                    },
                                    format!(
                                        "requested transient snapshot index '{index}' is missing von Mises data"
                                    ),
                                    BTreeMap::from([
                                        ("requested_snapshot_index".to_string(), index.to_string()),
                                        (
                                            "available_von_mises_snapshot_count".to_string(),
                                            transient.von_mises_snapshots.len().to_string(),
                                        ),
                                    ]),
                                )
                            })?;
                    let kinetic_energy =
                        transient
                            .kinetic_energy_snapshots
                            .get(index)
                            .cloned()
                            .ok_or_else(|| {
                                operation_error(
                                    ANALYSIS_RESULTS_OPERATION,
                                    ANALYSIS_RESULTS_OP_VERSION,
                                    &context,
                                    OperationErrorSpec {
                                        error_code:
                                            "RM.FEA.RESULTS.TRANSIENT_SNAPSHOT_NOT_FOUND",
                                        error_type: OperationErrorType::Input,
                                        retryable: false,
                                        severity: OperationErrorSeverity::Error,
                                    },
                                    format!(
                                        "requested transient snapshot index '{index}' is missing kinetic energy data"
                                    ),
                                    BTreeMap::from([
                                        ("requested_snapshot_index".to_string(), index.to_string()),
                                        (
                                            "available_kinetic_energy_snapshot_count".to_string(),
                                            transient.kinetic_energy_snapshots.len().to_string(),
                                        ),
                                    ]),
                                )
                            })?;
                    let strain_energy =
                        transient
                            .strain_energy_snapshots
                            .get(index)
                            .cloned()
                            .ok_or_else(|| {
                                operation_error(
                                    ANALYSIS_RESULTS_OPERATION,
                                    ANALYSIS_RESULTS_OP_VERSION,
                                    &context,
                                    OperationErrorSpec {
                                        error_code:
                                            "RM.FEA.RESULTS.TRANSIENT_SNAPSHOT_NOT_FOUND",
                                        error_type: OperationErrorType::Input,
                                        retryable: false,
                                        severity: OperationErrorSeverity::Error,
                                    },
                                    format!(
                                        "requested transient snapshot index '{index}' is missing strain energy data"
                                    ),
                                    BTreeMap::from([
                                        ("requested_snapshot_index".to_string(), index.to_string()),
                                        (
                                            "available_strain_energy_snapshot_count".to_string(),
                                            transient.strain_energy_snapshots.len().to_string(),
                                        ),
                                    ]),
                                )
                            })?;
                    let residual_norm_snapshot = transient
                        .residual_norm_snapshots
                        .get(index)
                        .cloned()
                        .ok_or_else(|| {
                            operation_error(
                                ANALYSIS_RESULTS_OPERATION,
                                ANALYSIS_RESULTS_OP_VERSION,
                                &context,
                                OperationErrorSpec {
                                    error_code: "RM.FEA.RESULTS.TRANSIENT_SNAPSHOT_NOT_FOUND",
                                    error_type: OperationErrorType::Input,
                                    retryable: false,
                                    severity: OperationErrorSeverity::Error,
                                },
                                format!(
                                    "requested transient snapshot index '{index}' is missing residual field data"
                                ),
                                BTreeMap::from([
                                    ("requested_snapshot_index".to_string(), index.to_string()),
                                    (
                                        "available_residual_snapshot_count".to_string(),
                                        transient.residual_norm_snapshots.len().to_string(),
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
                                    error_code: "RM.FEA.RESULTS.TRANSIENT_SNAPSHOT_NOT_FOUND",
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
                    velocity_snapshots.push(velocity);
                    acceleration_snapshots.push(acceleration);
                    von_mises_snapshots.push(von_mises);
                    kinetic_energy_snapshots.push(kinetic_energy);
                    strain_energy_snapshots.push(strain_energy);
                    residual_norm_snapshots.push(residual_norm_snapshot);
                }

                Some(TransientResultsData {
                    transient_payload_version: transient.transient_payload_version.clone(),
                    time_points_s,
                    displacement_snapshots,
                    velocity_snapshots,
                    acceleration_snapshots,
                    von_mises_snapshots,
                    kinetic_energy_snapshots,
                    strain_energy_snapshots,
                    residual_norm_snapshots,
                    thermo_mechanical_temperature_snapshots,
                    thermo_mechanical_thermal_strain_snapshots,
                    thermo_mechanical_thermal_stress_snapshots,
                    thermo_mechanical_displacement_snapshots,
                    thermo_mechanical_von_mises_snapshots,
                    thermo_mechanical_coupling_residual_snapshots,
                    electro_thermal_temperature_snapshots,
                    electro_thermal_thermal_residual_snapshots,
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

    let thermal_results = if query.include_field_values {
        run_result.thermal_results.clone()
    } else {
        None
    };

    let nonlinear_results = if query.include_nonlinear_results && query.include_field_values {
        run_result.nonlinear_results.clone()
    } else {
        None
    };
    let electromagnetic_results =
        if query.include_electromagnetic_results && query.include_field_values {
            run_result.electromagnetic_results.clone()
        } else {
            None
        };
    let fields = if query.include_field_values {
        collected_fields
    } else {
        Vec::new()
    };

    let data = AnalysisResultsData {
        field_descriptors,
        fields,
        modal_results,
        thermal_results,
        transient_results,
        nonlinear_results,
        electromagnetic_results,
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
                error_code: "RM.FEA.RESULTS.ARTIFACT_STORE_FAILED",
                error_type: OperationErrorType::Internal,
                retryable: true,
                severity: OperationErrorSeverity::Error,
            },
            format!("failed to load FEA run artifact: {err}"),
            BTreeMap::from([("run_id".to_string(), run_id.to_string())]),
        )
    })?;

    let Some(run_result) = run_result else {
        return Err(operation_error(
            ANALYSIS_RESULTS_OPERATION,
            ANALYSIS_RESULTS_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RESULTS.RUN_NOT_FOUND",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            format!("FEA run_id '{run_id}' was not found"),
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
                error_code: "RM.FEA.RESULTS_COMPARE.ARTIFACT_STORE_FAILED",
                error_type: OperationErrorType::Internal,
                retryable: true,
                severity: OperationErrorSeverity::Error,
            },
            format!("failed to load baseline FEA run artifact: {err}"),
            BTreeMap::from([("run_id".to_string(), query.baseline_run_id.clone())]),
        )
    })?;
    let Some(baseline) = baseline else {
        return Err(operation_error(
            ANALYSIS_RESULTS_COMPARE_OPERATION,
            ANALYSIS_RESULTS_COMPARE_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RESULTS_COMPARE.RUN_NOT_FOUND",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            format!(
                "FEA baseline run_id '{}' was not found",
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
                error_code: "RM.FEA.RESULTS_COMPARE.ARTIFACT_STORE_FAILED",
                error_type: OperationErrorType::Internal,
                retryable: true,
                severity: OperationErrorSeverity::Error,
            },
            format!("failed to load candidate FEA run artifact: {err}"),
            BTreeMap::from([("run_id".to_string(), query.candidate_run_id.clone())]),
        )
    })?;
    let Some(candidate) = candidate else {
        return Err(operation_error(
            ANALYSIS_RESULTS_COMPARE_OPERATION,
            ANALYSIS_RESULTS_COMPARE_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RESULTS_COMPARE.RUN_NOT_FOUND",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            format!(
                "FEA candidate run_id '{}' was not found",
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
                error_code: "RM.FEA.TRENDS.ARTIFACT_STORE_FAILED",
                error_type: OperationErrorType::Internal,
                retryable: true,
                severity: OperationErrorSeverity::Error,
            },
            format!("failed to list FEA run artifacts: {err}"),
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
        AnalysisRunKind::Acoustic,
        AnalysisRunKind::Thermal,
        AnalysisRunKind::Transient,
        AnalysisRunKind::Cfd,
        AnalysisRunKind::Cht,
        AnalysisRunKind::Fsi,
        AnalysisRunKind::Nonlinear,
        AnalysisRunKind::Electromagnetic,
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
            breach_rate_greater_than(&values, THERMO_SPREAD_THRESHOLD_BALANCED)
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
            breach_rate_greater_than(&values, THERMO_HETEROGENEITY_THRESHOLD_BALANCED)
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
        let thermal_stability_warn_rate = if kind == AnalysisRunKind::Thermal {
            diagnostic_warning_rate(&entries, "FEA_THERMAL_STABILITY")
        } else {
            None
        };
        let thermal_constitutive_warn_rate = if kind == AnalysisRunKind::Thermal {
            diagnostic_warning_rate(&entries, "FEA_THERMAL_CONSTITUTIVE")
        } else {
            None
        };
        let thermal_spread_breach_rate = if kind == AnalysisRunKind::Thermal {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(
                        &run.run.diagnostics,
                        "FEA_THERMAL_CONSTITUTIVE",
                        "conductivity_spread_ratio",
                    )
                })
                .collect::<Vec<_>>();
            breach_rate_greater_than(&values, 2.5)
        } else {
            None
        };
        let electromagnetic_solve_warn_rate = if kind == AnalysisRunKind::Electromagnetic {
            Some(diagnostic_warning_rate(&entries, "FEA_EM_STATIC").unwrap_or(0.0))
        } else {
            None
        };
        let electromagnetic_spread_breach_rate = if kind == AnalysisRunKind::Electromagnetic {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(
                        &run.run.diagnostics,
                        "FEA_EM_STATIC",
                        "conductivity_spread_ratio",
                    )
                })
                .collect::<Vec<_>>();
            breach_rate_greater_than(&values, EM_CONDUCTIVITY_SPREAD_THRESHOLD_BALANCED)
        } else {
            None
        };
        let electromagnetic_heterogeneity_breach_rate = if kind == AnalysisRunKind::Electromagnetic
        {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(
                        &run.run.diagnostics,
                        "FEA_EM_STATIC",
                        "electromagnetic_material_heterogeneity_index",
                    )
                })
                .collect::<Vec<_>>();
            breach_rate_greater_than(&values, EM_HETEROGENEITY_THRESHOLD_BALANCED)
        } else {
            None
        };
        let electromagnetic_coverage_breach_rate = if kind == AnalysisRunKind::Electromagnetic {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(
                        &run.run.diagnostics,
                        "FEA_EM_STATIC",
                        "assignment_coverage_ratio",
                    )
                })
                .collect::<Vec<_>>();
            breach_rate_less_than(&values, EM_ASSIGNMENT_COVERAGE_MIN_BALANCED)
        } else {
            None
        };
        let electromagnetic_fallback_breach_rate = if kind == AnalysisRunKind::Electromagnetic {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(
                        &run.run.diagnostics,
                        "FEA_EM_STATIC",
                        "fallback_coefficient_ratio",
                    )
                })
                .collect::<Vec<_>>();
            breach_rate_greater_than(&values, EM_FALLBACK_COEFFICIENT_MAX_BALANCED)
        } else {
            None
        };
        let electromagnetic_contrast_breach_rate = if kind == AnalysisRunKind::Electromagnetic {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(
                        &run.run.diagnostics,
                        "FEA_EM_STATIC",
                        "region_coefficient_contrast_index",
                    )
                })
                .collect::<Vec<_>>();
            breach_rate_greater_than(&values, EM_REGION_CONTRAST_MAX_BALANCED)
        } else {
            None
        };
        let electromagnetic_conditioning_breach_rate = if kind == AnalysisRunKind::Electromagnetic {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(
                        &run.run.diagnostics,
                        "FEA_EM_STATIC",
                        "condition_number_estimate",
                    )
                })
                .collect::<Vec<_>>();
            breach_rate_greater_than(&values, EM_CONDITIONING_MAX_BALANCED)
        } else {
            None
        };
        let electromagnetic_source_realization_breach_rate =
            if kind == AnalysisRunKind::Electromagnetic {
                let values = entries
                    .iter()
                    .filter_map(|run| {
                        diagnostic_metric(
                            &run.run.diagnostics,
                            "FEA_EM_STATIC",
                            "source_realization_ratio",
                        )
                    })
                    .collect::<Vec<_>>();
                breach_rate_less_than(&values, EM_SOURCE_REALIZATION_MIN_BALANCED)
            } else {
                None
            };
        let electromagnetic_source_region_coverage_breach_rate =
            if kind == AnalysisRunKind::Electromagnetic {
                let values = entries
                    .iter()
                    .filter_map(|run| {
                        diagnostic_metric(
                            &run.run.diagnostics,
                            "FEA_EM_STATIC",
                            "source_region_coverage_ratio",
                        )
                    })
                    .collect::<Vec<_>>();
                breach_rate_less_than(&values, EM_SOURCE_REGION_COVERAGE_MIN_BALANCED)
            } else {
                None
            };
        let electromagnetic_source_material_alignment_breach_rate =
            if kind == AnalysisRunKind::Electromagnetic {
                let values = entries
                    .iter()
                    .filter_map(|run| {
                        diagnostic_metric(
                            &run.run.diagnostics,
                            "FEA_EM_STATIC",
                            "source_material_alignment_ratio",
                        )
                    })
                    .collect::<Vec<_>>();
                breach_rate_less_than(&values, EM_SOURCE_MATERIAL_ALIGNMENT_MIN_BALANCED)
            } else {
                None
            };
        let electromagnetic_source_overlap_breach_rate = if kind == AnalysisRunKind::Electromagnetic
        {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(
                        &run.run.diagnostics,
                        "FEA_EM_STATIC",
                        "source_overlap_ratio",
                    )
                })
                .collect::<Vec<_>>();
            breach_rate_greater_than(&values, EM_SOURCE_OVERLAP_MAX_BALANCED)
        } else {
            None
        };
        let electromagnetic_source_interference_breach_rate =
            if kind == AnalysisRunKind::Electromagnetic {
                let values = entries
                    .iter()
                    .filter_map(|run| {
                        diagnostic_metric(
                            &run.run.diagnostics,
                            "FEA_EM_STATIC",
                            "source_interference_index",
                        )
                    })
                    .collect::<Vec<_>>();
                breach_rate_greater_than(&values, EM_SOURCE_INTERFERENCE_MAX_BALANCED)
            } else {
                None
            };
        let electromagnetic_boundary_anchor_breach_rate =
            if kind == AnalysisRunKind::Electromagnetic {
                let values = entries
                    .iter()
                    .filter_map(|run| {
                        diagnostic_metric(
                            &run.run.diagnostics,
                            "FEA_EM_STATIC",
                            "boundary_anchor_ratio",
                        )
                    })
                    .collect::<Vec<_>>();
                breach_rate_less_than(&values, EM_BOUNDARY_ANCHOR_MIN_BALANCED)
            } else {
                None
            };
        let electromagnetic_boundary_localization_breach_rate =
            if kind == AnalysisRunKind::Electromagnetic {
                let values = entries
                    .iter()
                    .filter_map(|run| {
                        diagnostic_metric(
                            &run.run.diagnostics,
                            "FEA_EM_STATIC",
                            "boundary_condition_localization_ratio",
                        )
                    })
                    .collect::<Vec<_>>();
                breach_rate_less_than(&values, EM_BOUNDARY_LOCALIZATION_MIN_BALANCED)
            } else {
                None
            };
        let electromagnetic_ground_effectiveness_breach_rate =
            if kind == AnalysisRunKind::Electromagnetic {
                let values = entries
                    .iter()
                    .filter_map(|run| {
                        diagnostic_metric(
                            &run.run.diagnostics,
                            "FEA_EM_STATIC",
                            "ground_anchor_effectiveness_ratio",
                        )
                    })
                    .collect::<Vec<_>>();
                breach_rate_less_than(&values, EM_GROUND_EFFECTIVENESS_MIN_BALANCED)
            } else {
                None
            };
        let electromagnetic_insulation_leakage_breach_rate =
            if kind == AnalysisRunKind::Electromagnetic {
                let values = entries
                    .iter()
                    .filter_map(|run| {
                        diagnostic_metric(
                            &run.run.diagnostics,
                            "FEA_EM_STATIC",
                            "insulation_leakage_ratio",
                        )
                    })
                    .collect::<Vec<_>>();
                breach_rate_greater_than(&values, EM_INSULATION_LEAKAGE_MAX_BALANCED)
            } else {
                None
            };
        let electromagnetic_divergence_breach_rate = if kind == AnalysisRunKind::Electromagnetic {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(
                        &run.run.diagnostics,
                        "FEA_EM_STATIC",
                        "flux_divergence_ratio",
                    )
                })
                .collect::<Vec<_>>();
            breach_rate_greater_than(&values, EM_FLUX_DIVERGENCE_MAX_BALANCED)
        } else {
            None
        };
        let electromagnetic_energy_imbalance_breach_rate =
            if kind == AnalysisRunKind::Electromagnetic {
                let values = entries
                    .iter()
                    .filter_map(|run| {
                        diagnostic_metric(
                            &run.run.diagnostics,
                            "FEA_EM_STATIC",
                            "energy_imbalance_ratio",
                        )
                    })
                    .collect::<Vec<_>>();
                breach_rate_greater_than(&values, EM_ENERGY_IMBALANCE_MAX_BALANCED)
            } else {
                None
            };
        let electromagnetic_boundary_energy_breach_rate =
            if kind == AnalysisRunKind::Electromagnetic {
                let values = entries
                    .iter()
                    .filter_map(|run| {
                        diagnostic_metric(
                            &run.run.diagnostics,
                            "FEA_EM_STATIC",
                            "boundary_energy_ratio",
                        )
                    })
                    .collect::<Vec<_>>();
                breach_rate_less_than(&values, EM_BOUNDARY_ENERGY_MIN_BALANCED)
            } else {
                None
            };
        let electromagnetic_boundary_penalty_contribution_breach_rate =
            if kind == AnalysisRunKind::Electromagnetic {
                let values = entries
                    .iter()
                    .filter_map(|run| {
                        diagnostic_metric(
                            &run.run.diagnostics,
                            "FEA_EM_STATIC",
                            "boundary_penalty_conditioning_contribution",
                        )
                    })
                    .collect::<Vec<_>>();
                breach_rate_greater_than(&values, EM_BOUNDARY_PENALTY_CONTRIBUTION_MAX_BALANCED)
            } else {
                None
            };
        let electromagnetic_source_region_energy_consistency_breach_rate =
            if kind == AnalysisRunKind::Electromagnetic {
                let values = entries
                    .iter()
                    .filter_map(|run| {
                        diagnostic_metric(
                            &run.run.diagnostics,
                            "FEA_EM_STATIC",
                            "source_region_energy_consistency_ratio",
                        )
                    })
                    .collect::<Vec<_>>();
                breach_rate_less_than(&values, EM_SOURCE_REGION_ENERGY_CONSISTENCY_MIN_BALANCED)
            } else {
                None
            };
        let electromagnetic_real_residual_breach_rate = if kind == AnalysisRunKind::Electromagnetic
        {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(&run.run.diagnostics, "FEA_EM_STATIC", "real_residual_norm")
                })
                .collect::<Vec<_>>();
            breach_rate_greater_than(&values, EM_REAL_RESIDUAL_MAX_BALANCED)
        } else {
            None
        };
        let electromagnetic_imag_residual_breach_rate = if kind == AnalysisRunKind::Electromagnetic
        {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(&run.run.diagnostics, "FEA_EM_STATIC", "imag_residual_norm")
                })
                .collect::<Vec<_>>();
            breach_rate_greater_than(&values, EM_IMAG_RESIDUAL_MAX_BALANCED)
        } else {
            None
        };
        let electromagnetic_sweep_coverage_breach_rate = if kind == AnalysisRunKind::Electromagnetic
        {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(&run.run.diagnostics, "FEA_EM_SWEEP", "sweep_count")
                })
                .collect::<Vec<_>>();
            breach_rate_less_than(&values, EM_SWEEP_COUNT_MIN_BALANCED)
        } else {
            None
        };
        let electromagnetic_resonance_sharpness_breach_rate =
            if kind == AnalysisRunKind::Electromagnetic {
                let values = entries
                    .iter()
                    .filter_map(|run| {
                        diagnostic_metric(
                            &run.run.diagnostics,
                            "FEA_EM_SWEEP",
                            "resonance_quality_factor",
                        )
                    })
                    .collect::<Vec<_>>();
                breach_rate_less_than(&values, EM_RESONANCE_Q_MIN_BALANCED)
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
            thermal_stability_warn_rate,
            thermal_constitutive_warn_rate,
            thermal_spread_breach_rate,
            electromagnetic_solve_warn_rate,
            electromagnetic_spread_breach_rate,
            electromagnetic_heterogeneity_breach_rate,
            electromagnetic_coverage_breach_rate,
            electromagnetic_fallback_breach_rate,
            electromagnetic_contrast_breach_rate,
            electromagnetic_conditioning_breach_rate,
            electromagnetic_source_realization_breach_rate,
            electromagnetic_source_region_coverage_breach_rate,
            electromagnetic_source_material_alignment_breach_rate,
            electromagnetic_source_overlap_breach_rate,
            electromagnetic_source_interference_breach_rate,
            electromagnetic_boundary_anchor_breach_rate,
            electromagnetic_boundary_localization_breach_rate,
            electromagnetic_ground_effectiveness_breach_rate,
            electromagnetic_insulation_leakage_breach_rate,
            electromagnetic_divergence_breach_rate,
            electromagnetic_energy_imbalance_breach_rate,
            electromagnetic_boundary_energy_breach_rate,
            electromagnetic_boundary_penalty_contribution_breach_rate,
            electromagnetic_source_region_energy_consistency_breach_rate,
            electromagnetic_real_residual_breach_rate,
            electromagnetic_imag_residual_breach_rate,
            electromagnetic_sweep_coverage_breach_rate,
            electromagnetic_resonance_sharpness_breach_rate,
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
    if run
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_ACOUSTIC_HARMONIC_RESPONSE")
    {
        AnalysisRunKind::Acoustic
    } else if run.electromagnetic_results.is_some()
        || run
            .run
            .diagnostics
            .iter()
            .any(|diag| diag.code == "FEA_EM_STATIC")
    {
        AnalysisRunKind::Electromagnetic
    } else if run
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_CHT_COUPLING")
    {
        AnalysisRunKind::Cht
    } else if run
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_FSI_COUPLING")
    {
        AnalysisRunKind::Fsi
    } else if run
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_CFD_FLOW")
    {
        AnalysisRunKind::Cfd
    } else if run.nonlinear_results.is_some() {
        AnalysisRunKind::Nonlinear
    } else if run.thermal_results.is_some() {
        AnalysisRunKind::Thermal
    } else if run.transient_results.is_some() {
        AnalysisRunKind::Transient
    } else if run.modal_results.is_some() {
        AnalysisRunKind::Modal
    } else {
        AnalysisRunKind::LinearStatic
    }
}

fn run_operation_version_for_kind(kind: AnalysisRunKind) -> &'static str {
    match kind {
        AnalysisRunKind::LinearStatic => ANALYSIS_RUN_OP_VERSION,
        AnalysisRunKind::Modal => ANALYSIS_RUN_MODAL_OP_VERSION,
        AnalysisRunKind::Acoustic => ANALYSIS_RUN_ACOUSTIC_OP_VERSION,
        AnalysisRunKind::Thermal => ANALYSIS_RUN_THERMAL_OP_VERSION,
        AnalysisRunKind::Transient => ANALYSIS_RUN_TRANSIENT_OP_VERSION,
        AnalysisRunKind::Cfd => ANALYSIS_RUN_CFD_OP_VERSION,
        AnalysisRunKind::Cht => ANALYSIS_RUN_CHT_OP_VERSION,
        AnalysisRunKind::Fsi => ANALYSIS_RUN_FSI_OP_VERSION,
        AnalysisRunKind::Nonlinear => ANALYSIS_RUN_NONLINEAR_OP_VERSION,
        AnalysisRunKind::Electromagnetic => ANALYSIS_RUN_ELECTROMAGNETIC_OP_VERSION,
    }
}

fn run_operation_for_kind(kind: AnalysisRunKind) -> &'static str {
    match kind {
        AnalysisRunKind::LinearStatic => ANALYSIS_RUN_OPERATION,
        AnalysisRunKind::Modal => ANALYSIS_RUN_MODAL_OPERATION,
        AnalysisRunKind::Acoustic => ANALYSIS_RUN_ACOUSTIC_OPERATION,
        AnalysisRunKind::Thermal => ANALYSIS_RUN_THERMAL_OPERATION,
        AnalysisRunKind::Transient => ANALYSIS_RUN_TRANSIENT_OPERATION,
        AnalysisRunKind::Cfd => ANALYSIS_RUN_CFD_OPERATION,
        AnalysisRunKind::Cht => ANALYSIS_RUN_CHT_OPERATION,
        AnalysisRunKind::Fsi => ANALYSIS_RUN_FSI_OPERATION,
        AnalysisRunKind::Nonlinear => ANALYSIS_RUN_NONLINEAR_OPERATION,
        AnalysisRunKind::Electromagnetic => ANALYSIS_RUN_ELECTROMAGNETIC_OPERATION,
    }
}

fn sanitize_study_sweep_id(sweep_id: &str) -> String {
    sweep_id
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn validate_study_issue_codes(spec: &AnalysisStudySpec) -> Vec<String> {
    let mut issue_codes = Vec::new();

    if spec.study_id.trim().is_empty() {
        issue_codes.push("RM.FEA.STUDY.ID_EMPTY".to_string());
    }
    if spec.create_model_intent.model_id.trim().is_empty() {
        issue_codes.push("RM.FEA.STUDY.MODEL_ID_EMPTY".to_string());
    }
    if spec.geometry.meshes.is_empty() {
        issue_codes.push("RM.FEA.STUDY.GEOMETRY_MESHES_EMPTY".to_string());
    }
    if spec.geometry.units == UnitSystem::Unspecified {
        issue_codes.push("RM.FEA.STUDY.GEOMETRY_UNITS_UNSPECIFIED".to_string());
    }
    if !profile_supports_run_kind(spec.create_model_intent.profile, spec.run_kind) {
        issue_codes.push("RM.FEA.STUDY.RUN_KIND_PROFILE_MISMATCH".to_string());
    }
    if let Some(model) = &spec.model {
        if model.geometry_id != spec.geometry.geometry_id
            || model.geometry_revision != spec.geometry.revision
        {
            issue_codes.push("RM.FEA.STUDY.MODEL_GEOMETRY_MISMATCH".to_string());
        }
        if validate_model_against_geometry(model, spec.geometry.units, &ReferenceFrame::Global)
            .is_err()
        {
            issue_codes.push("RM.FEA.STUDY.MODEL_INVALID".to_string());
        }
    }
    if spec.electromagnetic_run_options.is_some()
        && spec.run_kind != AnalysisRunKind::Electromagnetic
    {
        issue_codes.push("RM.FEA.STUDY.RUN_OPTIONS_KIND_MISMATCH".to_string());
    }
    if spec.linear_static_run_options.is_some() && spec.run_kind != AnalysisRunKind::LinearStatic
        || spec.modal_run_options.is_some() && spec.run_kind != AnalysisRunKind::Modal
        || spec.acoustic_run_options.is_some() && spec.run_kind != AnalysisRunKind::Acoustic
        || spec.thermal_run_options.is_some() && spec.run_kind != AnalysisRunKind::Thermal
        || spec.transient_run_options.is_some() && spec.run_kind != AnalysisRunKind::Transient
        || spec.cfd_run_options.is_some() && spec.run_kind != AnalysisRunKind::Cfd
        || spec.cht_run_options.is_some() && spec.run_kind != AnalysisRunKind::Cht
        || spec.fsi_run_options.is_some() && spec.run_kind != AnalysisRunKind::Fsi
        || spec.nonlinear_run_options.is_some() && spec.run_kind != AnalysisRunKind::Nonlinear
    {
        issue_codes.push("RM.FEA.STUDY.RUN_OPTIONS_KIND_MISMATCH".to_string());
    }
    if spec.run_kind == AnalysisRunKind::Electromagnetic {
        if let Some(options) = spec.electromagnetic_run_options.as_ref() {
            if !options.residual_target.is_finite() || options.residual_target <= 0.0 {
                issue_codes
                    .push("RM.FEA.STUDY.ELECTROMAGNETIC_RESIDUAL_TARGET_INVALID".to_string());
            }
            if !options.harmonic_tolerance.is_finite() || options.harmonic_tolerance <= 0.0 {
                issue_codes
                    .push("RM.FEA.STUDY.ELECTROMAGNETIC_HARMONIC_TOLERANCE_INVALID".to_string());
            }
            if options.harmonic_max_iterations == 0 {
                issue_codes.push(
                    "RM.FEA.STUDY.ELECTROMAGNETIC_HARMONIC_MAX_ITERATIONS_INVALID".to_string(),
                );
            }
            if options.sweep_enabled
                && !options
                    .sweep_frequency_hz
                    .iter()
                    .all(|frequency_hz| frequency_hz.is_finite() && *frequency_hz > 0.0)
            {
                issue_codes
                    .push("RM.FEA.STUDY.ELECTROMAGNETIC_SWEEP_FREQUENCY_INVALID".to_string());
            }
        }
    }

    issue_codes
}

fn study_issue_message(code: &str) -> &'static str {
    match code {
        "RM.FEA.STUDY.ID_EMPTY" => "study_id must be non-empty",
        "RM.FEA.STUDY.MODEL_ID_EMPTY" => "create_model_intent.model_id must be non-empty",
        "RM.FEA.STUDY.GEOMETRY_MESHES_EMPTY" => "geometry must contain at least one mesh",
        "RM.FEA.STUDY.GEOMETRY_UNITS_UNSPECIFIED" => {
            "geometry.units must be specified (not unspecified)"
        }
        "RM.FEA.STUDY.RUN_KIND_PROFILE_MISMATCH" => {
            "model.profile selects the solver; run kind must match the selected profile when supplied"
        }
        "RM.FEA.STUDY.MODEL_GEOMETRY_MISMATCH" => {
            "resolved model geometry id or revision does not match the study geometry"
        }
        "RM.FEA.STUDY.MODEL_INVALID" => "resolved model failed FEA validation",
        "RM.FEA.STUDY.RUN_OPTIONS_KIND_MISMATCH" => {
            "run options are only valid for the solver selected by model.profile"
        }
        "RM.FEA.STUDY.ELECTROMAGNETIC_RESIDUAL_TARGET_INVALID" => {
            "electromagnetic_run_options.residual_target must be finite and positive"
        }
        "RM.FEA.STUDY.ELECTROMAGNETIC_HARMONIC_TOLERANCE_INVALID" => {
            "electromagnetic_run_options.harmonic_tolerance must be finite and positive"
        }
        "RM.FEA.STUDY.ELECTROMAGNETIC_HARMONIC_MAX_ITERATIONS_INVALID" => {
            "electromagnetic_run_options.harmonic_max_iterations must be greater than zero"
        }
        "RM.FEA.STUDY.ELECTROMAGNETIC_SWEEP_FREQUENCY_INVALID" => {
            "electromagnetic_run_options.sweep_frequency_hz must contain finite positive values when sweep_enabled is true"
        }
        _ => "unrecognized study validation issue",
    }
}

fn profile_supports_run_kind(
    profile: AnalysisCreateModelProfile,
    run_kind: AnalysisRunKind,
) -> bool {
    profile.derived_run_kind() == run_kind
}

fn study_fingerprint(spec: &AnalysisStudySpec) -> String {
    let payload = serde_json::to_vec(spec).unwrap_or_else(|_| format!("{spec:?}").into_bytes());
    let mut hasher = Sha256::new();
    hasher.update(payload);
    format!("sha256:{:x}", hasher.finalize())
}

fn study_operation_sequence(spec: &AnalysisStudySpec, run_op_version: &str) -> Vec<String> {
    let mut operation_sequence = Vec::with_capacity(3);
    if spec.model.is_none() {
        operation_sequence.push(ANALYSIS_CREATE_MODEL_OP_VERSION.to_string());
    }
    operation_sequence.push(ANALYSIS_VALIDATE_OP_VERSION.to_string());
    operation_sequence.push(run_op_version.to_string());
    operation_sequence
}

fn study_run_options_json(spec: &AnalysisStudySpec) -> serde_json::Value {
    match spec.run_kind {
        AnalysisRunKind::LinearStatic => serde_json::to_value(&spec.linear_static_run_options),
        AnalysisRunKind::Modal => serde_json::to_value(&spec.modal_run_options),
        AnalysisRunKind::Acoustic => serde_json::to_value(&spec.acoustic_run_options),
        AnalysisRunKind::Thermal => serde_json::to_value(&spec.thermal_run_options),
        AnalysisRunKind::Transient => serde_json::to_value(&spec.transient_run_options),
        AnalysisRunKind::Cfd => serde_json::to_value(&spec.cfd_run_options),
        AnalysisRunKind::Cht => serde_json::to_value(&spec.cht_run_options),
        AnalysisRunKind::Fsi => serde_json::to_value(&spec.fsi_run_options),
        AnalysisRunKind::Nonlinear => serde_json::to_value(&spec.nonlinear_run_options),
        AnalysisRunKind::Electromagnetic => serde_json::to_value(&spec.electromagnetic_run_options),
    }
    .unwrap_or(serde_json::Value::Null)
}

fn study_evidence_root() -> PathBuf {
    let config = current_fea_runtime_config();
    config
        .study_artifact_root
        .or_else(|| {
            std::env::var("RUNMAT_FEA_STUDY_ARTIFACT_ROOT")
                .or_else(|_| std::env::var("RUNMAT_ANALYSIS_STUDY_ARTIFACT_ROOT"))
                .ok()
                .map(PathBuf::from)
        })
        .unwrap_or_else(|| {
            config
                .artifact_root
                .unwrap_or_else(default_fea_artifact_root)
                .join("studies")
        })
}

fn thermo_field_artifact_root() -> PathBuf {
    let config = current_fea_runtime_config();
    config
        .thermo_field_artifact_root
        .or_else(|| {
            std::env::var("RUNMAT_THERMO_FIELD_ARTIFACT_ROOT")
                .ok()
                .map(PathBuf::from)
        })
        .unwrap_or_else(|| {
            config
                .artifact_root
                .unwrap_or_else(default_fea_artifact_root)
                .join("thermo-fields")
        })
}

fn persist_study_evidence(
    study_fingerprint: &str,
    stage: &str,
    payload: serde_json::Value,
) -> Result<String, String> {
    let study_key = study_fingerprint.replace(':', "_");
    let root = study_evidence_root().join(study_key);
    fs_create_dir_all(&root)
        .map_err(|err| format!("failed to create study evidence directory: {err}"))?;
    let path = root.join(format!("{stage}.json"));
    let bytes = serde_json::to_vec_pretty(&payload)
        .map_err(|err| format!("failed to encode study evidence payload: {err}"))?;
    atomic_write_bytes(&path, &bytes)?;
    Ok(path.display().to_string())
}

fn atomic_write_bytes(path: &PathBuf, bytes: &[u8]) -> Result<(), String> {
    let tmp = path.with_extension(format!(
        "tmp-{}-{}",
        std::process::id(),
        Utc::now().timestamp_nanos_opt().unwrap_or_default()
    ));
    fs_write(&tmp, bytes)
        .map_err(|err| format!("failed to write temporary study evidence file: {err}"))?;
    fs_rename(&tmp, path).map_err(|err| {
        let _ = fs_remove_file(&tmp);
        format!("failed to atomically persist study evidence file: {err}")
    })
}

fn fs_create_dir_all(path: impl Into<PathBuf>) -> std::io::Result<()> {
    runmat_filesystem::create_dir_all(path.into())
}

fn fs_write(path: impl Into<PathBuf>, bytes: &[u8]) -> std::io::Result<()> {
    runmat_filesystem::write(path.into(), bytes)
}

fn fs_rename(from: impl Into<PathBuf>, to: impl Into<PathBuf>) -> std::io::Result<()> {
    runmat_filesystem::rename(from.into(), to.into())
}

fn fs_remove_file(path: impl Into<PathBuf>) -> std::io::Result<()> {
    match runmat_filesystem::remove_file(path.into()) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err),
    }
}

fn fs_exists(path: impl Into<PathBuf>) -> std::io::Result<bool> {
    match runmat_filesystem::metadata(path.into()) {
        Ok(_) => Ok(true),
        Err(err) if err.kind() == ErrorKind::NotFound => Ok(false),
        Err(err) => Err(err),
    }
}

fn fs_read_to_string(path: impl Into<PathBuf>) -> std::io::Result<String> {
    runmat_filesystem::read_to_string(path.into())
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
        topology_bandwidth_estimate: prep.topology_bandwidth_estimate,
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

fn model_thermo_coupling_options(model: &AnalysisModel) -> Option<ThermoMechanicalCouplingOptions> {
    let domain = model.thermo_mechanical.as_ref()?;
    let expansion = if model.materials.is_empty() {
        1.2e-5
    } else {
        model
            .materials
            .iter()
            .map(|material| material.thermal.expansion_coefficient_per_k.max(0.0))
            .sum::<f64>()
            / model.materials.len() as f64
    };

    Some(ThermoMechanicalCouplingOptions {
        enabled: domain.enabled,
        reference_temperature_k: domain.reference_temperature_k,
        applied_temperature_delta_k: domain.applied_temperature_delta_k,
        thermal_expansion_coefficient: expansion,
        field_artifact_id: domain.field_artifact_id.clone(),
        field_source: domain
            .field_source
            .as_ref()
            .map(|source| ThermoFieldSource {
                source_id: source.source_id.clone(),
                revision: source.revision,
                interpolation_mode: source.interpolation_mode.map(|mode| match mode {
                    runmat_analysis_core::ThermoFieldInterpolationMode::Linear => {
                        ThermoFieldInterpolationMode::Linear
                    }
                    runmat_analysis_core::ThermoFieldInterpolationMode::Step => {
                        ThermoFieldInterpolationMode::Step
                    }
                }),
                expected_region_ids: source.expected_region_ids.clone(),
            }),
        region_temperature_deltas: domain
            .region_temperature_deltas
            .iter()
            .map(|delta| ThermoRegionTemperatureDelta {
                region_id: delta.region_id.clone(),
                temperature_delta_k: delta.temperature_delta_k,
            })
            .collect(),
        time_profile: domain
            .time_profile
            .iter()
            .map(|point| ThermoTimeProfilePoint {
                normalized_time: point.normalized_time,
                scale: point.scale,
            })
            .collect(),
    })
}

fn model_electro_coupling_options(model: &AnalysisModel) -> Option<ElectroThermalCouplingOptions> {
    let domain = model.electro_thermal.as_ref()?;
    let electrical_materials: Vec<_> = model
        .materials
        .iter()
        .filter_map(|material| material.electrical.as_ref())
        .collect();
    let base_conductivity = if electrical_materials.is_empty() {
        1.0
    } else {
        electrical_materials
            .iter()
            .map(|e| e.conductivity_s_per_m.max(1.0e-12))
            .sum::<f64>()
            / electrical_materials.len() as f64
    };
    let resistive_coeff = if electrical_materials.is_empty() {
        0.0
    } else {
        electrical_materials
            .iter()
            .map(|e| e.resistive_heating_coefficient.max(0.0))
            .sum::<f64>()
            / electrical_materials.len() as f64
    };

    Some(ElectroThermalCouplingOptions {
        enabled: domain.enabled,
        reference_temperature_k: domain.reference_temperature_k,
        applied_voltage_v: domain.applied_voltage_v,
        base_electrical_conductivity_s_per_m: base_conductivity,
        resistive_heating_coefficient: resistive_coeff,
        region_conductivity_scales: domain
            .region_conductivity_scales
            .iter()
            .map(|scale| ElectroRegionConductivityScale {
                region_id: scale.region_id.clone(),
                conductivity_scale: scale.conductivity_scale,
            })
            .collect(),
        time_profile: domain
            .time_profile
            .iter()
            .map(|point| ElectroTimeProfilePoint {
                normalized_time: point.normalized_time,
                current_scale: point.current_scale,
            })
            .collect(),
    })
}

fn model_plasticity_constitutive_options(
    model: &AnalysisModel,
) -> Option<PlasticityConstitutiveOptions> {
    let plastic = model
        .materials
        .iter()
        .find_map(|material| material.plastic.as_ref())?;
    Some(PlasticityConstitutiveOptions {
        enabled: true,
        yield_strain: plastic.yield_strain,
        hardening_modulus_ratio: plastic.hardening_modulus_ratio,
        saturation_exponent: plastic.saturation_exponent,
    })
}

fn model_contact_interface_options(model: &AnalysisModel) -> Option<ContactInterfaceOptions> {
    model
        .interfaces
        .iter()
        .map(|interface| match &interface.kind {
            AnalysisInterfaceKind::Contact(contact) => ContactInterfaceOptions {
                enabled: true,
                penalty_stiffness_scale: contact.penalty_stiffness_scale,
                max_penetration_ratio: contact.max_penetration_ratio,
                friction_coefficient: contact.friction_coefficient,
            },
        })
        .next()
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

fn to_fea_plasticity_constitutive_context(
    options: Option<PlasticityConstitutiveOptions>,
) -> Option<runmat_analysis_fea::FeaPlasticityConstitutiveContext> {
    options.map(
        |plasticity| runmat_analysis_fea::FeaPlasticityConstitutiveContext {
            enabled: plasticity.enabled,
            yield_strain: plasticity.yield_strain,
            hardening_modulus_ratio: plasticity.hardening_modulus_ratio,
            saturation_exponent: plasticity.saturation_exponent,
        },
    )
}

fn to_fea_contact_interface_context(
    options: Option<ContactInterfaceOptions>,
) -> Option<runmat_analysis_fea::FeaContactInterfaceContext> {
    options.map(|contact| runmat_analysis_fea::FeaContactInterfaceContext {
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

fn validate_plasticity_constitutive_options(
    options: &PlasticityConstitutiveOptions,
) -> Result<(), (String, BTreeMap<String, String>)> {
    if !options.enabled {
        return Ok(());
    }
    if !options.yield_strain.is_finite() || options.yield_strain <= 0.0 {
        return Err((
            "plasticity constitutive model requires finite positive yield_strain".to_string(),
            BTreeMap::from([("yield_strain".to_string(), options.yield_strain.to_string())]),
        ));
    }
    if !options.hardening_modulus_ratio.is_finite() || options.hardening_modulus_ratio < 0.0 {
        return Err((
            "plasticity constitutive model requires finite non-negative hardening_modulus_ratio"
                .to_string(),
            BTreeMap::from([(
                "hardening_modulus_ratio".to_string(),
                options.hardening_modulus_ratio.to_string(),
            )]),
        ));
    }
    if !options.saturation_exponent.is_finite() || options.saturation_exponent < 0.0 {
        return Err((
            "plasticity constitutive model requires finite non-negative saturation_exponent"
                .to_string(),
            BTreeMap::from([(
                "saturation_exponent".to_string(),
                options.saturation_exponent.to_string(),
            )]),
        ));
    }
    Ok(())
}

fn validate_contact_interface_options(
    options: &ContactInterfaceOptions,
) -> Result<(), (String, BTreeMap<String, String>)> {
    if !options.enabled {
        return Ok(());
    }
    if !options.penalty_stiffness_scale.is_finite() || options.penalty_stiffness_scale <= 0.0 {
        return Err((
            "contact interface model requires finite positive penalty_stiffness_scale".to_string(),
            BTreeMap::from([(
                "penalty_stiffness_scale".to_string(),
                options.penalty_stiffness_scale.to_string(),
            )]),
        ));
    }
    if !options.max_penetration_ratio.is_finite() || options.max_penetration_ratio < 0.0 {
        return Err((
            "contact interface model requires finite non-negative max_penetration_ratio"
                .to_string(),
            BTreeMap::from([(
                "max_penetration_ratio".to_string(),
                options.max_penetration_ratio.to_string(),
            )]),
        ));
    }
    if !options.friction_coefficient.is_finite() || options.friction_coefficient < 0.0 {
        return Err((
            "contact interface model requires finite non-negative friction_coefficient".to_string(),
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

    let root = thermo_field_artifact_root();
    let path = root.join(format!("{field_artifact_id}.json"));
    if !fs_exists(&path).map_err(|err| {
        operation_error(
            operation,
            op_version,
            context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_THERMO_FIELD.STORE_FAILED",
                error_type: OperationErrorType::Internal,
                retryable: true,
                severity: OperationErrorSeverity::Error,
            },
            format!("failed to inspect thermo field artifact: {err}"),
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
        )
    })? {
        return Err(operation_error(
            operation,
            op_version,
            context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_THERMO_FIELD.NOT_FOUND",
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

    let raw = fs_read_to_string(&path).map_err(|err| {
        operation_error(
            operation,
            op_version,
            context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_THERMO_FIELD.STORE_FAILED",
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
                error_code: "RM.FEA.RUN_THERMO_FIELD.INVALID",
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

    if artifact.schema_version != "fea_thermo_field_artifact/v1"
        && artifact.schema_version != "analysis_thermo_field_artifact/v1"
    {
        return Err(operation_error(
            operation,
            op_version,
            context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_THERMO_FIELD.SCHEMA_UNSUPPORTED",
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
                error_code: "RM.FEA.RUN_THERMO_FIELD.MISMATCH",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "thermo field artifact geometry lineage does not match FEA model",
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
                error_code: "RM.FEA.RUN_THERMO_FIELD.DIGEST_MISMATCH",
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
                    error_code: "RM.FEA.RUN_THERMO_FIELD.APPROVER_MISSING",
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
                    error_code: "RM.FEA.RUN_THERMO_FIELD.APPROVER_UNAUTHORIZED",
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
                    error_code: "RM.FEA.RUN_THERMO_FIELD.SIGNATURE_INVALID",
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
                    error_code: "RM.FEA.RUN_PREP.UNTRUSTED_CONTEXT",
                    error_type: OperationErrorType::Input,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                "FEA run prep_context must be referenced by prep_artifact_id",
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
                error_code: "RM.FEA.RUN_PREP.STORE_FAILED",
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
                error_code: "RM.FEA.RUN_PREP.NOT_FOUND",
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
                error_code: "RM.FEA.RUN_PREP.SCHEMA_UNSUPPORTED",
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
                error_code: "RM.FEA.RUN_PREP.MISMATCH",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "prep artifact geometry lineage does not match FEA model",
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

    if crate::geometry::require_latest_prep_revision() {
        if let Some(latest_revision) = crate::geometry::latest_prep_revision_for_geometry(
            &model.geometry_id,
        )
        .map_err(|err| {
            operation_error(
                operation,
                op_version,
                context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RUN_PREP.STORE_FAILED",
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
                        error_code: "RM.FEA.RUN_PREP.STALE",
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
    let topology_bandwidth_estimate = artifact
        .prep
        .prepared_meshes
        .iter()
        .map(|mesh| mesh.region_span_hint)
        .sum::<u32>()
        .clamp(1, 128);
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
        topology_bandwidth_estimate,
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

#[derive(Debug, Clone, Default)]
struct EmSweepSummary {
    sweep_count: usize,
    resonance_peak_frequency_hz: Option<f64>,
    resonance_peak_flux_density: Option<f64>,
    resonance_bandwidth_hz: Option<f64>,
    resonance_quality_factor: Option<f64>,
    resonance_flux_gain: Option<f64>,
}

fn normalize_em_sweep_frequency_hz(
    reference_frequency_hz: f64,
    sweep_enabled: bool,
    requested: &[f64],
) -> Option<Vec<f64>> {
    let mut values = if sweep_enabled {
        requested.to_vec()
    } else {
        Vec::new()
    };
    if values.is_empty() {
        values.push(reference_frequency_hz);
    }
    if !values
        .iter()
        .all(|frequency_hz| frequency_hz.is_finite() && *frequency_hz > 0.0)
    {
        return None;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    values.dedup_by(|a, b| (*a - *b).abs() <= 1.0e-9);
    Some(values)
}

fn nearest_frequency_index(frequencies_hz: &[f64], target_hz: f64) -> Option<usize> {
    frequencies_hz
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| (*a - target_hz).abs().total_cmp(&(*b - target_hz).abs()))
        .map(|(index, _)| index)
}

fn peak_abs_field_value(field: &runmat_analysis_core::AnalysisField) -> f64 {
    field
        .as_host_f64()
        .map(|values| values.iter().copied().map(f64::abs).fold(0.0_f64, f64::max))
        .unwrap_or(0.0)
}

fn summarize_em_sweep(frequencies_hz: &[f64], peak_flux_density: &[f64]) -> EmSweepSummary {
    if frequencies_hz.is_empty() || frequencies_hz.len() != peak_flux_density.len() {
        return EmSweepSummary::default();
    }
    let sweep_count = frequencies_hz.len();
    let (peak_index, peak_flux_density_value) = peak_flux_density
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap_or((0, 0.0));
    let peak_frequency_hz = frequencies_hz[peak_index];
    let min_flux_density_value = peak_flux_density
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let resonance_flux_gain =
        (peak_flux_density_value / min_flux_density_value.max(1.0e-12)).max(1.0);

    let half_power = peak_flux_density_value * std::f64::consts::FRAC_1_SQRT_2;
    let mut left = peak_index;
    while left > 0 && peak_flux_density[left - 1] >= half_power {
        left -= 1;
    }
    let mut right = peak_index;
    while right + 1 < peak_flux_density.len() && peak_flux_density[right + 1] >= half_power {
        right += 1;
    }
    let resonance_bandwidth_hz = if right > left {
        Some((frequencies_hz[right] - frequencies_hz[left]).max(0.0))
    } else {
        None
    };
    let resonance_quality_factor = resonance_bandwidth_hz
        .filter(|bandwidth| *bandwidth > 0.0)
        .map(|bandwidth| (peak_frequency_hz / bandwidth).max(0.0));

    EmSweepSummary {
        sweep_count,
        resonance_peak_frequency_hz: Some(peak_frequency_hz),
        resonance_peak_flux_density: Some(peak_flux_density_value),
        resonance_bandwidth_hz,
        resonance_quality_factor,
        resonance_flux_gain: Some(resonance_flux_gain),
    }
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
            "RM.FEA.VALIDATE.MISSING_MATERIALS",
            "FEA model must include at least one material".to_string(),
            BTreeMap::new(),
        ),
        AnalysisValidationError::MissingBoundaryConditions => (
            "RM.FEA.VALIDATE.MISSING_BCS",
            "FEA model must include at least one boundary condition".to_string(),
            BTreeMap::new(),
        ),
        AnalysisValidationError::MissingLoads => (
            "RM.FEA.VALIDATE.MISSING_LOADS",
            "FEA model must include at least one load".to_string(),
            BTreeMap::new(),
        ),
        AnalysisValidationError::UnitMismatch { model, geometry } => (
            "RM.FEA.VALIDATE.UNIT_MISMATCH",
            format!("model units {model:?} do not match geometry units {geometry:?}"),
            BTreeMap::from([
                ("model_units".to_string(), format!("{model:?}")),
                ("geometry_units".to_string(), format!("{geometry:?}")),
            ]),
        ),
        AnalysisValidationError::FrameMismatch { model, geometry } => (
            "RM.FEA.VALIDATE.FRAME_MISMATCH",
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
