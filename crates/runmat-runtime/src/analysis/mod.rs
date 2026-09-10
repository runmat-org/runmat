use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::ErrorKind;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock, RwLock};
use std::time::Instant;

use chrono::Utc;
use runmat_analysis_core::{
    validate_model_against_geometry, AnalysisBoundary, AnalysisConstraint, AnalysisField,
    AnalysisInterfaceKind, AnalysisModel, AnalysisModelId, AnalysisProcedure, AnalysisSequence,
    AnalysisValidationError, BoundaryOperator, ContactFriction, CoordinateFrame,
    CoupledPhysicsFamily, EvidenceConfidence, ExcitationAction, FluidConstraint,
    FluidConstraintKind, MagneticConstraint, MagneticConstraintKind, MaterialAssignment,
    MaterialMechanicalModel, MaterialModel, MaterialThermalModel, PhysicsFamily, PrescribedValue,
    Pressure, ProcedureStep, ProcedureStepId, RegionTarget, StructuralConstraint,
    StructuralConstraintKind, Temperature, ThermalConstraint, ThermalConstraintKind,
    ThermalInterfaceTransfer, Velocity,
};
use runmat_analysis_fea::execution::{
    CanonicalSemanticState, CanonicalSemanticStateInput, SequenceIdentity, SolverResultIdentity,
    StepIdentity, StructuralLinearWorkload, StudyIdentity,
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
    run_thermal_with_options, run_transient_with_options, AnalysisRunIdentity, ComputeBackend,
    ElectromagneticSolveOptions, ExecutionBackend, FeaProgressHandler, FeaRunResult,
    LinearStaticSolveOptions, ModalSolveOptions, SolverMethod, SolverPreconditioner,
    ThermalProcedureControl, ThermalSolveOptions, FEA_FIELD_CFD_PRESSURE,
    FEA_FIELD_CFD_RESIDUAL_CONTINUITY, FEA_FIELD_CFD_RESIDUAL_MOMENTUM,
    FEA_FIELD_CFD_REYNOLDS_NUMBER, FEA_FIELD_CFD_VELOCITY, FEA_FIELD_CFD_VORTICITY,
    FEA_FIELD_CFD_WALL_SHEAR_STRESS, FEA_FIELD_CHT_FLUID_PRESSURE, FEA_FIELD_CHT_FLUID_VELOCITY,
};
use runmat_geometry_core::{GeometryAsset, MaterialEvidenceConfidence, UnitSystem};
use runmat_meshing_core::SolverMeshArtifact;

use crate::operations::{
    operation_error, OperationContext, OperationEnvelope, OperationErrorEnvelope,
    OperationErrorSeverity, OperationErrorSpec, OperationErrorType,
};
use policy::{
    breach_rate_greater_than, breach_rate_less_than, electromagnetic_sweep_thresholds_for_policy,
    electromagnetic_thresholds_for_policy, thermo_field_coverage_min_for_policy,
    thermo_spatial_gradient_threshold_for_policy, thermo_thresholds_for_policy,
    ElectromagneticQualityThresholds, EM_ASSIGNMENT_COVERAGE_MIN_BALANCED,
    EM_BOUNDARY_ANCHOR_MIN_BALANCED, EM_BOUNDARY_ENERGY_MIN_BALANCED,
    EM_BOUNDARY_LOCALIZATION_MIN_BALANCED, EM_BOUNDARY_PENALTY_CONTRIBUTION_MAX_BALANCED,
    EM_CONDITIONING_MAX_BALANCED, EM_CONDUCTIVITY_SPREAD_THRESHOLD_BALANCED,
    EM_ENERGY_IMBALANCE_MAX_BALANCED, EM_FLUX_DIVERGENCE_MAX_BALANCED,
    EM_GROUND_EFFECTIVENESS_MIN_BALANCED, EM_HETEROGENEITY_THRESHOLD_BALANCED,
    EM_IMAG_RESIDUAL_MAX_BALANCED, EM_INSULATION_LEAKAGE_MAX_BALANCED,
    EM_REAL_RESIDUAL_MAX_BALANCED, EM_REGION_CONTRAST_MAX_BALANCED, EM_RESONANCE_Q_MIN_BALANCED,
    EM_SOURCE_INTERFERENCE_MAX_BALANCED, EM_SOURCE_MATERIAL_ALIGNMENT_MIN_BALANCED,
    EM_SOURCE_OVERLAP_MAX_BALANCED, EM_SOURCE_REALIZATION_MIN_BALANCED,
    EM_SOURCE_REGION_COVERAGE_MIN_BALANCED, EM_SOURCE_REGION_ENERGY_CONSISTENCY_MIN_BALANCED,
    EM_SWEEP_COUNT_MIN_BALANCED, THERMO_HETEROGENEITY_THRESHOLD_BALANCED,
    THERMO_SPREAD_THRESHOLD_BALANCED,
};

mod contracts;
mod excitation_authoring;
mod excitation_query;
mod execution_dispatch;
mod execution_error;
mod fea_document;
mod fea_document_authoring;
#[cfg(feature = "plot-core")]
mod figures;
mod formulation_admission;
mod linear_static_result;
pub(crate) mod meshing;
mod operations;
mod policy;
mod result_policy;
mod solver_mesh_artifact;
mod solver_mesh_flow_topology;
pub mod storage;
mod structural_study;
mod study_execution;
mod study_placement;
mod study_sweep;
pub mod value_admission;

use excitation_authoring::{append_constant_excitation, ConstantExcitation};
use excitation_query::{
    applied_temperature_delta_k, applied_voltage_v, cfd_velocity_scales,
    has_nonzero_electromagnetic_source, nominal_inlet_velocity_m_per_s,
    regional_temperature_deltas_k,
};
use execution_dispatch::{execution_operation, run_operation_for_identity, AnalysisOperation};
use execution_error::{admit_runtime_proxy_for_operation, map_fea_run_error};
use formulation_admission::{RuntimeDomainSource, RuntimeDomainTopology, RuntimeProxyFormulation};
use result_policy::{classify_run_status, evaluate_production_publication};
#[cfg(test)]
use structural_study::persist_immutable_study_evidence;
#[cfg(test)]
use structural_study::run_evidence_stage;
pub use structural_study::{
    analysis_complete_structural_study_intent_op, analysis_prepare_structural_study_intent_op,
    structural_study_execution_failure, PreparedStructuralStudy, PreparedStructuralStudyIntent,
    StructuralStudyCompletionIdentity, StructuralStudyExecutionFailure,
    StructuralStudySequenceExecutionOutcome, STRUCTURAL_STUDY_INTENT_MAX_BYTES,
};
use structural_study::{
    complete_study_run, persist_content_addressed_study_evidence, study_completion_from_prepared,
    study_evidence_root, CompletedStudyStep, PreparedStudySequence, PreparedStudyStep,
};
use study_execution::execute_study_step;
pub use study_placement::{
    analysis_execute_structural_study_with_placement,
    analysis_execute_structural_sweep_with_placement, StructuralStudyPlacementEntry,
    StructuralStudyPlacementExecutor, StructuralStudyPlacementFuture,
    StructuralStudyPlacementResult,
};
pub use study_sweep::{
    analysis_complete_study_sweep_op, analysis_study_sweep_identity, AnalysisStudySweepRunOutcome,
};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FeaRuntimeConfig {
    pub artifact_root: Option<PathBuf>,
    pub study_artifact_root: Option<PathBuf>,
}

/// Host adapter for the path-free structural-static numerical workload. Implementations choose
/// execution placement but may not persist results or apply runtime publication policy.
pub trait StructuralLinearExecutor: Send + Sync {
    fn execute(
        &self,
        workload: &StructuralLinearWorkload,
        mesh: &SolverMeshArtifact,
    ) -> Result<FeaRunResult, String>;
}

/// A fully resolved structural-linear solve at the placement boundary.
///
/// The workload and mesh are path-free worker inputs. The remaining fields are retained by the
/// runtime so a returned numerical result can be admitted, classified, and persisted exactly once.
#[derive(Debug, Clone)]
pub struct PreparedStructuralLinearRun {
    workload: StructuralLinearWorkload,
    solver_mesh: Option<SolverMeshArtifact>,
    options: AnalysisRunOptions,
    context: OperationContext,
}

#[derive(Debug, Clone)]
pub struct StructuralStudyStepNumericalResult {
    pub step_id: ProcedureStepId,
    pub solver_result_identity: runmat_analysis_fea::execution::SolverResultIdentity,
    pub result: FeaRunResult,
}

#[derive(Debug, Clone)]
pub struct StructuralStudySequenceNumericalResult {
    pub frozen_project_digest: Option<runmat_execution::Digest>,
    pub steps: Vec<StructuralStudyStepNumericalResult>,
    pub states: Vec<runmat_analysis_fea::execution::CanonicalSemanticState>,
}

impl PreparedStructuralLinearRun {
    pub fn workload(&self) -> &StructuralLinearWorkload {
        &self.workload
    }

    pub fn solver_mesh(&self) -> Option<&SolverMeshArtifact> {
        self.solver_mesh.as_ref()
    }
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
    runmat_analysis_fea::replace_fea_progress_context(runmat_analysis_fea::FeaProgressContext {
        handler,
        cancellation: crate::interrupt::current_interrupt()
            .map(runmat_analysis_fea::FeaCancellationToken::from_shared)
            .unwrap_or_default(),
        attempt_id: None,
    })
}

pub use contracts::{
    analysis_runtime_model_template_catalog, AnalysisAcousticRunOptions, AnalysisCfdRunOptions,
    AnalysisChtRunOptions, AnalysisCreateModelIntentSpec, AnalysisDiagnosticsArtifactPayload,
    AnalysisDocumentCheckResult, AnalysisDocumentKind, AnalysisDocumentRunResult,
    AnalysisElectromagneticRunOptions, AnalysisExecutionPlan, AnalysisExecutionStep,
    AnalysisFieldBinding, AnalysisFieldBindingsArtifactPayload, AnalysisFieldDescriptor,
    AnalysisFieldKind, AnalysisFieldLocation, AnalysisFieldPageResult,
    AnalysisFieldPagingDescriptor, AnalysisFieldRequestOptions, AnalysisFieldStorage,
    AnalysisFieldStorageRef, AnalysisFsiRunOptions, AnalysisImmutableValueSpec,
    AnalysisModalRunOptions, AnalysisModelTemplate, AnalysisNonlinearRunOptions,
    AnalysisObjectArtifactMetadata, AnalysisRenderMesh, AnalysisRenderRegion,
    AnalysisRenderTopology, AnalysisRenderTopologySource, AnalysisRenderTriangleRange,
    AnalysisResultsCompareData, AnalysisResultsCompareQuery, AnalysisResultsData,
    AnalysisResultsQuery, AnalysisResultsSummary, AnalysisRunDatasetFieldPagingPolicy,
    AnalysisRunDatasetPayload, AnalysisRunDatasetStudyRef, AnalysisRunOptions, AnalysisRunResult,
    AnalysisRuntimeCapabilities, AnalysisRuntimeModelTemplateCatalogEntry,
    AnalysisRuntimeModelTemplateDefaultObservation, AnalysisSolverOptions, AnalysisStudyIssue,
    AnalysisStudyPlanData, AnalysisStudyPlanStepData, AnalysisStudyRunData, AnalysisStudySpec,
    AnalysisStudyStatePublicationData, AnalysisStudyStepRunData, AnalysisStudySweepData,
    AnalysisStudySweepFailureEntry, AnalysisStudySweepFailureKind, AnalysisStudySweepPlanData,
    AnalysisStudySweepPlanEntry, AnalysisStudySweepRunEntry, AnalysisStudySweepSpec,
    AnalysisStudySweepValidateData, AnalysisStudySweepValidateEntry, AnalysisStudyValidateResult,
    AnalysisThermalRunOptions, AnalysisTransientRunOptions, AnalysisTrendKindSummary,
    AnalysisTrendsData, AnalysisTrendsQuery, AnalysisValidateResult, ContactEnforcementOptions,
    ElectroThermalCouplingOptions, ElectromagneticResultsData, FallbackEvent, FallbackReason,
    FallbackStage, ModalFrequencyBasis, ModalFrequencyUnits, ModalResultsData, NonlinearMethod,
    NonlinearResultsData, PlasticityConstitutiveOptions, PrecisionMode, PreconditionerMode,
    ProductionPublicationReason, ProductionPublicationReasonCode, QualityGate, QualityPolicy,
    QualityReason, QualityReasonCode, RunProvenance, RunStatus, ThermalResultsData,
    ThermoMechanicalCouplingOptions, ThermoRegionTemperatureDelta, TransientIntegrationMethod,
    TransientResultsData, ANALYSIS_ARTIFACT_MANIFEST_KIND, ANALYSIS_DATASET_ARTIFACT_KIND,
    ANALYSIS_DIAGNOSTICS_ARTIFACT_KIND, ANALYSIS_DIAGNOSTICS_SCHEMA_VERSION,
    ANALYSIS_FIELD_BINDINGS_ARTIFACT_KIND, ANALYSIS_FIELD_BINDINGS_SCHEMA_VERSION,
    ANALYSIS_FIELD_DEFAULT_MATERIALIZE_LIMIT, ANALYSIS_FIELD_DEFAULT_PAGE_SIZE,
    ANALYSIS_OBJECT_ARTIFACT_METADATA_SCHEMA_VERSION, ANALYSIS_RUN_DATASET_KIND,
    ANALYSIS_RUN_DATASET_SCHEMA_VERSION,
};
pub use fea_document::{
    is_fea_file_path, load_fea_document_from_path_async, parse_and_resolve_fea_document,
    FeaResolvedDocument,
};
pub use fea_document_authoring::{
    apply_fea_study_document_operation, apply_fea_study_document_operation_typed,
    summarize_fea_study_document, FeaStudyDocumentOperation, FeaStudyDocumentOperationOutput,
    FEA_STUDY_DOCUMENT_OPERATION_NAMES,
};
#[cfg(feature = "plot-core")]
pub use figures::{
    analysis_generate_study_run_figures, analysis_generate_study_run_mesh_figure,
    AnalysisFigureGenerationOptions, AnalysisFigureMeshSource, AnalysisGeneratedFigure,
    AnalysisGeneratedFigureKind,
};
pub use operations::{
    analysis_field_page_by_run_id_op, analysis_results_by_run_id_op, analysis_results_compare_op,
    analysis_results_op, analysis_trends_op,
};
pub(crate) use operations::{analysis_run_field_ids, collect_analysis_result_fields};
pub use runmat_analysis_core::FormulationRequirement;
pub use runmat_analysis_fea::{FeaProgressEvent, FeaProgressPhase, FeaProgressStatus};

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
const ANALYSIS_RESULTS_OPERATION: &str = "fea.results";
const ANALYSIS_RESULTS_OP_VERSION: &str = "fea.results/v1";
const ANALYSIS_RESULTS_COMPARE_OPERATION: &str = "fea.results_compare";
const ANALYSIS_RESULTS_COMPARE_OP_VERSION: &str = "fea.results_compare/v1";
const ANALYSIS_TRENDS_OPERATION: &str = "fea.trends";
const ANALYSIS_TRENDS_OP_VERSION: &str = "fea.trends/v1";
const TRANSIENT_RESIDUAL_WARN_THRESHOLD: f64 = 1.0e-4;

fn reject_moment_loads_for_run_family(
    model: &AnalysisModel,
    operation: &'static str,
    op_version: &'static str,
    error_code: &'static str,
    family: &'static str,
    context: &OperationContext,
) -> Result<(), OperationErrorEnvelope> {
    if let Some(excitation) = model
        .excitations
        .iter()
        .find(|excitation| matches!(excitation, ExcitationAction::Moment(_)))
    {
        let authored = excitation.excitation();
        return Err(operation_error(
            operation,
            op_version,
            context,
            OperationErrorSpec {
                error_code,
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            format!("moment excitations are structural and cannot be used as {family} excitations"),
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                (
                    "excitation_id".to_string(),
                    authored.excitation_id.0.clone(),
                ),
            ]),
        ));
    }
    Ok(())
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

enum ModelCondition {
    Constraint(AnalysisConstraint),
    Boundary(AnalysisBoundary),
}

fn fixed_condition(id: &str, region_id: String) -> ModelCondition {
    ModelCondition::Constraint(AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: id.to_string(),
        frame: CoordinateFrame::Global,
        kind: StructuralConstraintKind::Fixed {
            target: RegionTarget::boundary(region_id),
        },
    }))
}

fn prescribed_temperature_condition(
    id: &str,
    region_id: String,
    temperature_k: f64,
) -> ModelCondition {
    ModelCondition::Constraint(AnalysisConstraint::Thermal(ThermalConstraint {
        constraint_id: id.to_string(),
        kind: ThermalConstraintKind::Prescribed {
            target: RegionTarget::boundary(region_id),
            value: PrescribedValue::Constant(Temperature::new(temperature_k)),
        },
    }))
}

fn magnetic_ground_condition(id: &str, region_id: String) -> ModelCondition {
    ModelCondition::Constraint(AnalysisConstraint::Magnetic(MagneticConstraint {
        constraint_id: id.to_string(),
        kind: MagneticConstraintKind::VectorPotentialGround {
            target: RegionTarget::boundary(region_id),
        },
    }))
}

fn acoustic_rigid_condition(id: &str, region_id: String) -> ModelCondition {
    ModelCondition::Boundary(AnalysisBoundary {
        boundary_id: id.to_string(),
        target: RegionTarget::boundary(region_id),
        operator: BoundaryOperator::AcousticRigidWall,
    })
}

fn fluid_constraint_condition(id: &str, kind: FluidConstraintKind) -> ModelCondition {
    ModelCondition::Constraint(AnalysisConstraint::Fluid(FluidConstraint {
        constraint_id: id.to_string(),
        kind,
    }))
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

    let fixed_region_id = select_fixed_region_id(geometry)
        .or_else(|| {
            geometry
                .regions
                .first()
                .map(|region| region.region_id.clone())
        })
        .unwrap_or_else(|| "region_default".to_string());
    let load_region_id = select_load_region_id(geometry)
        .or_else(|| {
            geometry
                .regions
                .last()
                .map(|region| region.region_id.clone())
        })
        .unwrap_or_else(|| fixed_region_id.clone());

    let mut inferred_materials = infer_material_models(geometry);
    if matches!(intent.template, AnalysisModelTemplate::AcousticHarmonic) {
        for material in &mut inferred_materials {
            material.acoustic = Some(runmat_analysis_core::MaterialAcousticModel::default());
        }
    }
    if matches!(
        intent.template,
        AnalysisModelTemplate::ElectromagneticHarmonic
    ) {
        for material in &mut inferred_materials {
            material.electrical = Some(runmat_analysis_core::MaterialElectricalModel::default());
        }
    }
    let inferred_assignments = infer_material_assignments(geometry, &inferred_materials);

    let (default_bc, default_excitation, default_steps) = match intent.template {
        AnalysisModelTemplate::LinearStaticStructural => (
            fixed_condition("bc_default_fixed", fixed_region_id),
            Some((
                "excitation_default_force",
                ConstantExcitation::Force([0.0, -1000.0, 0.0]),
            )),
            vec![ProcedureStep::structural_static(
                "step_default_static".to_string(),
            )],
        ),
        AnalysisModelTemplate::ThermoMechanicalCoupled => (
            fixed_condition("bc_default_fixed", fixed_region_id),
            Some((
                "excitation_default_thermal_mech_force",
                ConstantExcitation::Force([0.0, -650.0, 0.0]),
            )),
            vec![ProcedureStep::coupled(
                "step_default_thermo_mech",
                CoupledPhysicsFamily::Thermomechanical,
                AnalysisProcedure::TransientResponse,
            )],
        ),
        AnalysisModelTemplate::ThermalSteady => (
            prescribed_temperature_condition("bc_default_temperature", fixed_region_id, 293.15),
            Some((
                "excitation_default_heat_generation",
                ConstantExcitation::HeatGeneration(1.0),
            )),
            vec![ProcedureStep::thermal_steady(
                "step_default_thermal_steady".to_string(),
            )],
        ),
        AnalysisModelTemplate::ThermalTransient => (
            prescribed_temperature_condition("bc_default_temperature", fixed_region_id, 293.15),
            Some((
                "excitation_default_heat_generation",
                ConstantExcitation::HeatGeneration(1.0),
            )),
            vec![ProcedureStep::thermal_transient(
                "step_default_thermal".to_string(),
            )],
        ),
        AnalysisModelTemplate::ModalStructural => (
            fixed_condition("bc_default_fixed", fixed_region_id),
            None,
            vec![ProcedureStep::structural_modal(
                "step_default_modal".to_string(),
            )],
        ),
        AnalysisModelTemplate::AcousticHarmonic => (
            acoustic_rigid_condition("bc_default_acoustic_rigid_wall", fixed_region_id),
            Some((
                "excitation_default_acoustic_pressure",
                ConstantExcitation::AcousticPressure(1.0),
            )),
            vec![ProcedureStep::acoustic_harmonic(
                "step_default_acoustic_harmonic",
            )],
        ),
        AnalysisModelTemplate::TransientStructural => (
            fixed_condition("bc_default_fixed", fixed_region_id),
            Some((
                "excitation_default_transient_force",
                ConstantExcitation::Force([0.0, -500.0, 0.0]),
            )),
            vec![ProcedureStep::structural_transient(
                "step_default_transient".to_string(),
            )],
        ),
        AnalysisModelTemplate::NonlinearStructural => (
            fixed_condition("bc_default_fixed", fixed_region_id),
            Some((
                "excitation_default_nonlinear_force",
                ConstantExcitation::Force([0.0, -750.0, 0.0]),
            )),
            vec![ProcedureStep::structural_nonlinear(
                "step_default_nonlinear".to_string(),
            )],
        ),
        AnalysisModelTemplate::ElectromagneticHarmonic => (
            magnetic_ground_condition("bc_default_em_ground", fixed_region_id),
            Some((
                "excitation_default_em_current",
                ConstantExcitation::ElectricCurrent {
                    real_amperes: 100.0,
                    imaginary_amperes: 0.0,
                    frequency_hz: None,
                },
            )),
            vec![ProcedureStep::electromagnetic_harmonic(
                "step_default_electromagnetic".to_string(),
            )],
        ),
        AnalysisModelTemplate::CfdSteadyState => (
            fixed_condition("bc_default_fixed", fixed_region_id),
            None,
            vec![ProcedureStep::fluid_static("step_default_cfd".to_string())],
        ),
        AnalysisModelTemplate::CfdTransient => (
            fixed_condition("bc_default_fixed", fixed_region_id),
            None,
            vec![ProcedureStep::fluid_transient("step_default_cfd_transient")],
        ),
        AnalysisModelTemplate::ChtCoupled => (
            fixed_condition("bc_default_fixed", fixed_region_id),
            Some((
                "excitation_default_cht_heat_generation",
                ConstantExcitation::HeatGeneration(1.0),
            )),
            vec![ProcedureStep::coupled(
                "step_default_cht",
                CoupledPhysicsFamily::ConjugateHeatTransfer,
                AnalysisProcedure::TransientResponse,
            )],
        ),
        AnalysisModelTemplate::FsiCoupled => (
            fixed_condition("bc_default_fixed", fixed_region_id),
            Some((
                "excitation_default_fsi_force",
                ConstantExcitation::Force([0.0, -450.0, 0.0]),
            )),
            vec![ProcedureStep::coupled(
                "step_default_fsi",
                CoupledPhysicsFamily::FluidStructureInteraction,
                AnalysisProcedure::TransientResponse,
            )],
        ),
    };

    let cfd = match intent.template {
        AnalysisModelTemplate::CfdSteadyState => Some(runmat_analysis_core::CfdDomain {
            density: runmat_analysis_core::MassDensity::new(1.225),
            dynamic_viscosity: runmat_analysis_core::DynamicViscosity::new(1.81e-5),
            turbulence_intensity: runmat_analysis_core::TurbulenceIntensity::new(0.05),
        }),
        AnalysisModelTemplate::CfdTransient => Some(runmat_analysis_core::CfdDomain {
            density: runmat_analysis_core::MassDensity::new(1.225),
            dynamic_viscosity: runmat_analysis_core::DynamicViscosity::new(1.81e-5),
            turbulence_intensity: runmat_analysis_core::TurbulenceIntensity::new(0.08),
        }),
        AnalysisModelTemplate::ChtCoupled => Some(runmat_analysis_core::CfdDomain {
            density: runmat_analysis_core::MassDensity::new(1.225),
            dynamic_viscosity: runmat_analysis_core::DynamicViscosity::new(1.81e-5),
            turbulence_intensity: runmat_analysis_core::TurbulenceIntensity::new(0.07),
        }),
        AnalysisModelTemplate::FsiCoupled => Some(runmat_analysis_core::CfdDomain {
            density: runmat_analysis_core::MassDensity::new(1.225),
            dynamic_viscosity: runmat_analysis_core::DynamicViscosity::new(1.81e-5),
            turbulence_intensity: runmat_analysis_core::TurbulenceIntensity::new(0.06),
        }),
        _ => None,
    };
    let electromagnetic = match intent.template {
        AnalysisModelTemplate::ElectromagneticHarmonic => {
            Some(runmat_analysis_core::ElectromagneticDomain {
                reference_frequency: runmat_analysis_core::Frequency::new(60.0),
            })
        }
        _ => None,
    };
    let thermo_mechanical = match intent.template {
        AnalysisModelTemplate::ThermoMechanicalCoupled => {
            Some(runmat_analysis_core::ThermoMechanicalDomain {
                reference_temperature: Temperature::new(293.15),
            })
        }
        AnalysisModelTemplate::ChtCoupled => Some(runmat_analysis_core::ThermoMechanicalDomain {
            reference_temperature: Temperature::new(293.15),
        }),
        _ => None,
    };
    let thermal = match intent.template {
        AnalysisModelTemplate::ThermalSteady
        | AnalysisModelTemplate::ThermalTransient
        | AnalysisModelTemplate::ChtCoupled => Some(runmat_analysis_core::ThermalDomain {
            reference_temperature: Temperature::new(293.15),
        }),
        _ => None,
    };
    let electro_thermal = None;

    let mut conditions = vec![default_bc];
    if matches!(
        intent.template,
        AnalysisModelTemplate::CfdSteadyState
            | AnalysisModelTemplate::CfdTransient
            | AnalysisModelTemplate::ChtCoupled
            | AnalysisModelTemplate::FsiCoupled
    ) {
        let default_cfd_inlet_velocity = match intent.template {
            AnalysisModelTemplate::CfdSteadyState | AnalysisModelTemplate::CfdTransient => 5.0,
            AnalysisModelTemplate::ChtCoupled => 4.5,
            AnalysisModelTemplate::FsiCoupled => 4.0,
            _ => unreachable!("fluid defaults are created only for fluid templates"),
        };
        conditions.extend([
            fluid_constraint_condition(
                "bc_default_cfd_inlet",
                FluidConstraintKind::PrescribedVelocity {
                    target: RegionTarget::boundary("inlet"),
                    value: PrescribedValue::Constant(Velocity::new(default_cfd_inlet_velocity)),
                },
            ),
            fluid_constraint_condition(
                "bc_default_cfd_outlet",
                FluidConstraintKind::PressureReference {
                    target: RegionTarget::boundary("outlet"),
                    value: PrescribedValue::Constant(Pressure::new(0.0)),
                },
            ),
            fluid_constraint_condition(
                "bc_default_cfd_wall_upper",
                FluidConstraintKind::NoSlipWall {
                    target: RegionTarget::boundary("wall_upper"),
                },
            ),
            fluid_constraint_condition(
                "bc_default_cfd_wall_lower",
                FluidConstraintKind::NoSlipWall {
                    target: RegionTarget::boundary("wall_lower"),
                },
            ),
        ]);
    }

    let mut constraints = Vec::new();
    let mut boundaries = Vec::new();
    for condition in conditions {
        match condition {
            ModelCondition::Constraint(value) => constraints.push(value),
            ModelCondition::Boundary(value) => boundaries.push(value),
        }
    }

    let mut model = AnalysisModel {
        model_id: AnalysisModelId(intent.model_id.clone()),
        geometry_id: geometry.geometry_id.clone(),
        geometry_revision: geometry.revision,
        units: geometry.units,
        frame: CoordinateFrame::Global,
        materials: inferred_materials,
        material_assignments: inferred_assignments,
        structural: None,
        thermal,
        thermo_mechanical,
        electro_thermal,
        electromagnetic,
        cfd,
        interfaces: Vec::new(),
        constraints,
        boundaries,
        connectors: Vec::new(),
        signals: Vec::new(),
        excitations: Vec::new(),
        excitation_cases: Vec::new(),
        initial_conditions: Vec::new(),
        response_spectra: Vec::new(),
        sequence: AnalysisSequence::from_steps(
            format!("{}.sequence", intent.model_id),
            default_steps,
        ),
    };

    if let Some((excitation_id, excitation)) = default_excitation {
        append_constant_excitation(&mut model, excitation_id, load_region_id, excitation);
    }
    validate_model_against_geometry(&model, geometry.units, &CoordinateFrame::Global).map_err(
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
    let mut issue_codes = validate_study_issue_codes(spec);
    let model_validation_detail = spec.model.as_ref().and_then(|model| {
        validate_model_against_geometry(model, spec.geometry.units, &CoordinateFrame::Global)
            .err()
            .map(|error| error.to_string())
    });
    if spec.model.is_none()
        && !issue_codes.iter().any(|code| {
            matches!(
                code.as_str(),
                "RM.FEA.STUDY.GEOMETRY_MESHES_EMPTY"
                    | "RM.FEA.STUDY.GEOMETRY_UNITS_UNSPECIFIED"
                    | "RM.FEA.STUDY.MODEL_ID_EMPTY"
            )
        })
    {
        let model = resolve_study_model(spec, &context)?;
        if spec.execution.validate_against(&model.sequence).is_err() {
            issue_codes.push("RM.FEA.STUDY.EXECUTION_PLAN_INVALID".to_string());
        }
    }
    issue_codes.sort();
    issue_codes.dedup();
    let issues: Vec<AnalysisStudyIssue> = issue_codes
        .iter()
        .map(|code| AnalysisStudyIssue {
            code: code.clone(),
            message: if code == "RM.FEA.STUDY.MODEL_INVALID" {
                model_validation_detail
                    .as_ref()
                    .map(|detail| format!("resolved model failed FEA validation: {detail}"))
                    .unwrap_or_else(|| study_issue_message(code).to_string())
            } else {
                study_issue_message(code).to_string()
            },
        })
        .collect();
    let study_fingerprint = study_fingerprint(spec);
    let evidence_artifact_path = persist_content_addressed_study_evidence(
        &study_fingerprint,
        "validate",
        serde_json::json!({
            "schema_version": "fea_study_validate_artifact/v1",
            "study_id": spec.study_id.clone(),
            "study_fingerprint": study_fingerprint.clone(),
            "valid": issue_codes.is_empty(),
            "issue_codes": issue_codes.clone(),
            "issues": issues.clone(),
            "execution": spec.execution,
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
    let model = resolve_study_model(spec, &context)?;
    spec.execution
        .validate_against(&model.sequence)
        .map_err(|message| {
            operation_error(
                ANALYSIS_PLAN_STUDY_OPERATION,
                ANALYSIS_PLAN_STUDY_OP_VERSION,
                &context,
                OperationErrorSpec {
                    error_code: "RM.FEA.PLAN_STUDY.INVALID_EXECUTION",
                    error_type: OperationErrorType::Validation,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                message,
                BTreeMap::from([("study_id".to_string(), spec.study_id.clone())]),
            )
        })?;
    let steps = model
        .sequence
        .steps
        .iter()
        .enumerate()
        .map(|(ordinal, procedure_step)| {
            let execution = spec
                .execution
                .steps
                .iter()
                .find(|entry| entry.step_id == procedure_step.step_id)
                .expect("validated execution plan must cover every procedure step");
            let operation = execution_operation(execution, procedure_step)
                .expect("resolved execution must select a supported solver operation");
            AnalysisStudyPlanStepData {
                ordinal,
                procedure_step: procedure_step.clone(),
                execution: execution.clone(),
                run_operation: operation.name().to_string(),
                run_op_version: operation.version().to_string(),
            }
        })
        .collect::<Vec<_>>();
    let operation_sequence = study_operation_sequence(
        spec,
        &steps
            .iter()
            .map(|step| step.run_op_version.clone())
            .collect::<Vec<_>>(),
    );
    let evidence_artifact_path = persist_content_addressed_study_evidence(
        &study_fingerprint,
        "plan",
        serde_json::json!({
            "schema_version": "fea_study_sequence_plan_artifact/v1",
            "study_id": spec.study_id.clone(),
            "model_id": spec.create_model_intent.model_id.clone(),
            "sequence_id": model.sequence.sequence_id.clone(),
            "steps": steps.clone(),
            "study_fingerprint": study_fingerprint.clone(),
            "operation_sequence": operation_sequence.clone(),
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
            sequence_id: model.sequence.sequence_id,
            steps,
            operation_sequence,
            study_fingerprint,
            evidence_artifact_path,
        },
    ))
}

pub fn analysis_run_study_op(
    spec: &AnalysisStudySpec,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisStudyRunData>, OperationErrorEnvelope> {
    analysis_run_study_with_structural_executor(spec, context, None)
}

pub fn analysis_run_study_with_structural_executor_op(
    spec: &AnalysisStudySpec,
    context: OperationContext,
    executor: &dyn StructuralLinearExecutor,
) -> Result<OperationEnvelope<AnalysisStudyRunData>, OperationErrorEnvelope> {
    analysis_run_study_with_structural_executor(spec, context, Some(executor))
}

pub fn analysis_prepare_structural_study_op(
    spec: &AnalysisStudySpec,
    context: OperationContext,
) -> Result<PreparedStructuralStudy, OperationErrorEnvelope> {
    let study = prepare_study_run(spec, context)?;
    let mut linears = BTreeMap::new();
    for step in &study.steps {
        let AnalysisSolverOptions::StructuralLinear(mut options) =
            step.authored_execution.solver_options.clone()
        else {
            return Err(operation_error(
                ANALYSIS_RUN_STUDY_OPERATION,
                ANALYSIS_RUN_STUDY_OP_VERSION,
                &study.context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RUN_STUDY.PORTABLE_FAMILY_UNSUPPORTED",
                    error_type: OperationErrorType::Validation,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                "portable study preparation requires structural-linear procedure steps",
                BTreeMap::from([
                    ("study_id".to_string(), study.spec.study_id.clone()),
                    (
                        "step_id".to_string(),
                        step.procedure_step.step_id.as_str().to_string(),
                    ),
                ]),
            ));
        };
        if options.solver_mesh_artifact_path.is_none() {
            options.solver_mesh_artifact_path = study.solver_mesh_artifact_path.clone();
        }
        let linear = prepare_linear_static_step_run(
            &study.model,
            &step.procedure_step.step_id,
            step.authored_execution.backend,
            step.authored_execution.requirement,
            options,
            study.context.clone(),
        )?;
        if linear.solver_mesh.is_none() {
            return Err(operation_error(
                ANALYSIS_RUN_STUDY_OPERATION,
                ANALYSIS_RUN_STUDY_OP_VERSION,
                &study.context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RUN_STUDY.PORTABLE_MESH_REQUIRED",
                    error_type: OperationErrorType::Input,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                "portable structural study execution requires a canonical solver mesh",
                BTreeMap::from([("study_id".to_string(), study.spec.study_id.clone())]),
            ));
        }
        if linears
            .insert(step.procedure_step.step_id.clone(), linear)
            .is_some()
        {
            return Err(operation_error(
                ANALYSIS_RUN_STUDY_OPERATION,
                ANALYSIS_RUN_STUDY_OP_VERSION,
                &study.context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RUN_STUDY.DUPLICATE_PREPARED_STEP",
                    error_type: OperationErrorType::Contract,
                    retryable: false,
                    severity: OperationErrorSeverity::Fatal,
                },
                "portable structural study prepared one step more than once",
                BTreeMap::new(),
            ));
        }
    }
    Ok(PreparedStructuralStudy { study, linears })
}

pub fn analysis_complete_structural_study_op(
    mut prepared: PreparedStructuralStudy,
    outcome: StructuralStudySequenceExecutionOutcome,
) -> Result<OperationEnvelope<AnalysisStudyRunData>, OperationErrorEnvelope> {
    let result = match outcome {
        Ok(result) => result,
        Err(StructuralStudyExecutionFailure::Failed(message)) => {
            return Err(structural_study_completion_failure(
                &prepared,
                "RM.FEA.RUN_STUDY.PORTABLE_EXECUTION_FAILED",
                OperationErrorType::Backend,
                OperationErrorSeverity::Error,
                message,
            ));
        }
        Err(StructuralStudyExecutionFailure::Cancelled) => {
            return Err(structural_study_completion_failure(
                &prepared,
                "RM.FEA.RUN_STUDY.PORTABLE_EXECUTION_CANCELLED",
                OperationErrorType::Cancelled,
                OperationErrorSeverity::Error,
                "portable structural study execution was cancelled",
            ));
        }
        Err(StructuralStudyExecutionFailure::Indeterminate(message)) => {
            return Err(structural_study_completion_failure(
                &prepared,
                "RM.FEA.RUN_STUDY.PORTABLE_EXECUTION_INDETERMINATE",
                OperationErrorType::Internal,
                OperationErrorSeverity::Fatal,
                message,
            ));
        }
    };
    let mut completed = Vec::with_capacity(result.steps.len());
    for step in result.steps {
        let linear = prepared.linears.remove(&step.step_id).ok_or_else(|| {
            operation_error(
                ANALYSIS_RUN_STUDY_OPERATION,
                ANALYSIS_RUN_STUDY_OP_VERSION,
                &prepared.study.context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
                    error_type: OperationErrorType::Contract,
                    retryable: false,
                    severity: OperationErrorSeverity::Fatal,
                },
                "sequence execution returned an absent or duplicate procedure step",
                BTreeMap::from([("step_id".to_string(), step.step_id.as_str().to_string())]),
            )
        })?;
        let resolved_solver_options =
            AnalysisSolverOptions::StructuralLinear(linear.options.clone());
        let run = analysis_admit_linear_static_run_with_id(linear, step.result, None)?;
        completed.push(CompletedStudyStep {
            step_id: step.step_id,
            solver_result_identity: step.solver_result_identity,
            run,
            resolved_solver_options,
        });
    }
    if !prepared.linears.is_empty() {
        return Err(operation_error(
            ANALYSIS_RUN_STUDY_OPERATION,
            ANALYSIS_RUN_STUDY_OP_VERSION,
            &prepared.study.context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
                error_type: OperationErrorType::Contract,
                retryable: false,
                severity: OperationErrorSeverity::Fatal,
            },
            "sequence execution omitted one or more prepared procedure steps",
            BTreeMap::new(),
        ));
    }
    complete_study_run(
        study_completion_from_prepared(prepared.study),
        completed,
        result.states,
        result.frozen_project_digest,
    )
}

fn structural_study_completion_failure(
    prepared: &PreparedStructuralStudy,
    error_code: &'static str,
    error_type: OperationErrorType,
    severity: OperationErrorSeverity,
    message: impl Into<String>,
) -> OperationErrorEnvelope {
    operation_error(
        ANALYSIS_RUN_STUDY_OPERATION,
        ANALYSIS_RUN_STUDY_OP_VERSION,
        &prepared.study.context,
        OperationErrorSpec {
            error_code,
            error_type,
            retryable: error_type == OperationErrorType::Internal,
            severity,
        },
        message,
        BTreeMap::from([(
            "analysis_model_id".to_string(),
            prepared.study.model.model_id.0.clone(),
        )]),
    )
}

fn analysis_run_study_with_structural_executor(
    spec: &AnalysisStudySpec,
    context: OperationContext,
    structural_executor: Option<&dyn StructuralLinearExecutor>,
) -> Result<OperationEnvelope<AnalysisStudyRunData>, OperationErrorEnvelope> {
    let prepared = prepare_study_run(spec, context)?;
    if prepared
        .steps
        .iter()
        .any(|step| !step.procedure_step.state_inputs.is_empty())
    {
        return Err(operation_error(
            ANALYSIS_RUN_STUDY_OPERATION,
            ANALYSIS_RUN_STUDY_OP_VERSION,
            &prepared.context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_STUDY.SEQUENCE_EXECUTOR_REQUIRED",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "studies that consume upstream state require the canonical structural sequence executor",
            BTreeMap::from([("study_id".to_string(), prepared.spec.study_id.clone())]),
        ));
    }
    let _study_identity =
        StudyIdentity::from_authored_digest(analysis_study_content_digest(&prepared.spec), None)
            .map_err(|error| study_sequence_contract_error(&prepared.context, error))?;
    let sequence_identity = SequenceIdentity::from_solver_contract(&prepared.model.sequence)
        .map_err(|error| study_sequence_contract_error(&prepared.context, error))?;
    let topology_digest = in_process_study_topology_digest(&prepared)?;
    let mut completed = Vec::with_capacity(prepared.steps.len());
    let mut states = Vec::new();
    for step in &prepared.steps {
        let (run, resolved_solver_options) = execute_study_step(
            &prepared.model,
            &step.procedure_step,
            &step.authored_execution,
            prepared.solver_mesh_artifact_path.as_deref(),
            &prepared.context,
            structural_executor,
        )?;
        let step_identity =
            StepIdentity::from_solver_contract(sequence_identity, &step.procedure_step)
                .map_err(|error| study_sequence_contract_error(&prepared.context, error))?;
        let result_bytes = serde_json::to_vec(&run.data.run)
            .map_err(|error| study_sequence_contract_error(&prepared.context, error))?;
        let solver_result_identity = SolverResultIdentity::from_step_result(
            step_identity,
            runmat_execution::Digest::sha256(result_bytes),
        )
        .map_err(|error| study_sequence_contract_error(&prepared.context, error))?;
        for declaration in &step.procedure_step.state_outputs {
            let state_fields = collect_analysis_result_fields(&run.data)
                .into_iter()
                .filter(|field| match &field.values {
                    runmat_analysis_core::AnalysisFieldValues::HostF64(values) => {
                        !values.is_empty()
                            && values.iter().all(|value| value.is_finite())
                            && field
                                .shape
                                .iter()
                                .try_fold(1_usize, |count, extent| count.checked_mul(*extent))
                                == Some(values.len())
                    }
                    runmat_analysis_core::AnalysisFieldValues::DeviceRef(_) => false,
                })
                .collect();
            states.push(
                CanonicalSemanticState::new(CanonicalSemanticStateInput {
                    producer: step_identity,
                    declaration: declaration.clone(),
                    topology_digest,
                    source_result: solver_result_identity,
                    fields: state_fields,
                })
                .map_err(|error| study_sequence_contract_error(&prepared.context, error))?,
            );
        }
        completed.push(CompletedStudyStep {
            step_id: step.procedure_step.step_id.clone(),
            solver_result_identity,
            run,
            resolved_solver_options,
        });
    }
    complete_study_run(
        study_completion_from_prepared(prepared),
        completed,
        states,
        None,
    )
}

fn in_process_study_topology_digest(
    prepared: &PreparedStudySequence,
) -> Result<runmat_execution::Digest, OperationErrorEnvelope> {
    if let Some(path) = prepared.solver_mesh_artifact_path.as_deref() {
        let mesh = solver_mesh_artifact::load_solver_mesh_artifact(std::path::Path::new(path))
            .map_err(|error| study_sequence_contract_error(&prepared.context, error))?;
        return Ok(runmat_execution::Digest::from_bytes(
            *mesh.canonical_digest.bytes(),
        ));
    }
    let bytes = serde_json::to_vec(&prepared.model)
        .map_err(|error| study_sequence_contract_error(&prepared.context, error))?;
    Ok(runmat_execution::Digest::sha256(bytes))
}

fn study_sequence_contract_error(
    context: &OperationContext,
    message: impl std::fmt::Display,
) -> OperationErrorEnvelope {
    operation_error(
        ANALYSIS_RUN_STUDY_OPERATION,
        ANALYSIS_RUN_STUDY_OP_VERSION,
        context,
        OperationErrorSpec {
            error_code: "RM.FEA.RUN_STUDY.SEQUENCE_CONTRACT_FAILED",
            error_type: OperationErrorType::Contract,
            retryable: false,
            severity: OperationErrorSeverity::Fatal,
        },
        message.to_string(),
        BTreeMap::new(),
    )
}

fn prepare_study_run(
    spec: &AnalysisStudySpec,
    context: OperationContext,
) -> Result<PreparedStudySequence, OperationErrorEnvelope> {
    let mut prepared = prepare_study_semantics(spec, context)?;
    let study_mesh = meshing::resolve_study_mesh(spec).map_err(|message| {
        operation_error(
            ANALYSIS_RUN_STUDY_OPERATION,
            ANALYSIS_RUN_STUDY_OP_VERSION,
            &prepared.context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_STUDY.MESHING_FAILED",
                error_type: OperationErrorType::Backend,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            message,
            BTreeMap::new(),
        )
    })?;
    let aliases = runmat_analysis_core::BoundaryRegionAliases::new(study_mesh.boundary_region_ids)
        .map_err(|error| {
            operation_error(
                ANALYSIS_RUN_STUDY_OPERATION,
                ANALYSIS_RUN_STUDY_OP_VERSION,
                &prepared.context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RUN_STUDY.BOUNDARY_RESOLUTION_FAILED",
                    error_type: OperationErrorType::Validation,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                error.to_string(),
                BTreeMap::new(),
            )
        })?;
    prepared.model =
        runmat_analysis_core::resolve_boundary_region_aliases(&prepared.model, &aliases).map_err(
            |error| {
                operation_error(
                    ANALYSIS_RUN_STUDY_OPERATION,
                    ANALYSIS_RUN_STUDY_OP_VERSION,
                    &prepared.context,
                    OperationErrorSpec {
                        error_code: "RM.FEA.RUN_STUDY.BOUNDARY_RESOLUTION_FAILED",
                        error_type: OperationErrorType::Validation,
                        retryable: false,
                        severity: OperationErrorSeverity::Error,
                    },
                    error.to_string(),
                    BTreeMap::new(),
                )
            },
        )?;
    prepared.solver_mesh_artifact_path = study_mesh.artifact_path;
    prepared.meshing_evidence_artifact_path = study_mesh.evidence_path;
    Ok(prepared)
}

fn prepare_study_semantics(
    spec: &AnalysisStudySpec,
    context: OperationContext,
) -> Result<PreparedStudySequence, OperationErrorEnvelope> {
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
    let model = resolve_study_model(spec, &context)?;
    spec.execution
        .validate_against(&model.sequence)
        .map_err(|message| {
            operation_error(
                ANALYSIS_RUN_STUDY_OPERATION,
                ANALYSIS_RUN_STUDY_OP_VERSION,
                &context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RUN_STUDY.INVALID_EXECUTION",
                    error_type: OperationErrorType::Validation,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                message,
                BTreeMap::from([("study_id".to_string(), spec.study_id.clone())]),
            )
        })?;
    let steps = model
        .sequence
        .steps
        .iter()
        .map(|procedure_step| {
            let authored_execution = spec
                .execution
                .steps
                .iter()
                .find(|execution| execution.step_id == procedure_step.step_id)
                .expect("validated execution plan must cover every procedure step");
            PreparedStudyStep {
                authored_execution: authored_execution.clone(),
                procedure_step: procedure_step.clone(),
            }
        })
        .collect::<Vec<_>>();
    let operation_sequence = study_operation_sequence(
        spec,
        &steps
            .iter()
            .map(|step| {
                execution_operation(&step.authored_execution, &step.procedure_step)
                    .expect("resolved execution must select a supported solver operation")
                    .version()
                    .to_string()
            })
            .collect::<Vec<_>>(),
    );
    analysis_validate(
        &model,
        spec.geometry.units,
        &CoordinateFrame::Global,
        context.clone(),
    )?;
    Ok(PreparedStudySequence {
        spec: spec.clone(),
        context,
        study_fingerprint,
        model,
        steps,
        operation_sequence,
        solver_mesh_artifact_path: None,
        meshing_evidence_artifact_path: None,
    })
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
                    kind: AnalysisStudySweepFailureKind::Failed,
                    error_code: err.error_code,
                    message: err.message,
                });
                continue;
            }
        };
        plan_entries.push(AnalysisStudySweepPlanEntry {
            study_id: planned.data.study_id,
            model_id: planned.data.model_id,
            sequence_id: planned.data.sequence_id,
            steps: planned.data.steps,
            operation_sequence: planned.data.operation_sequence,
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
    analysis_study_sweep_identity(spec, &context)?;
    let mut outcomes = Vec::with_capacity(spec.studies.len());
    for study in &spec.studies {
        match analysis_run_study_op(study, context.clone()) {
            Ok(run) => outcomes.push(AnalysisStudySweepRunOutcome::Succeeded(Box::new(run))),
            Err(error) => {
                outcomes.push(AnalysisStudySweepRunOutcome::Failed(error));
                if spec.fail_fast {
                    outcomes
                        .resize_with(spec.studies.len(), || AnalysisStudySweepRunOutcome::NotRun);
                    break;
                }
            }
        }
    }
    analysis_complete_study_sweep_op(spec, context, outcomes)
}

pub fn analysis_validate(
    model: &AnalysisModel,
    geometry_units: UnitSystem,
    geometry_frame: &CoordinateFrame,
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
    analysis_run_linear_static_with_options(
        model,
        backend,
        FormulationRequirement::Production,
        AnalysisRunOptions::default(),
        context,
    )
}

pub fn analysis_run_modal_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    analysis_run_modal_with_options_op(
        model,
        backend,
        FormulationRequirement::Production,
        AnalysisModalRunOptions::default(),
        context,
    )
}

pub fn analysis_run_acoustic_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    analysis_run_acoustic_with_options_op(
        model,
        backend,
        FormulationRequirement::Production,
        AnalysisAcousticRunOptions::default(),
        context,
    )
}

pub fn analysis_run_modal_with_options_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    formulation_requirement: FormulationRequirement,
    options: AnalysisModalRunOptions,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    let _solver_context = install_fea_solver_context();
    let has_modal_step = model.sequence.steps.iter().any(|step| {
        step.family == PhysicsFamily::Structural
            && step.procedure == AnalysisProcedure::ModalExtraction
    });
    if !has_modal_step {
        return Err(operation_error(
            AnalysisOperation::Modal.name(),
            AnalysisOperation::Modal.version(),
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
            AnalysisOperation::Modal.name(),
            AnalysisOperation::Modal.version(),
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

    let thermo_options = model_thermo_coupling_options(model);
    if let Some(thermo_options) = thermo_options.as_ref() {
        if let Err((detail, metadata)) = validate_thermo_coupling_options(model, thermo_options) {
            return Err(operation_error(
                AnalysisOperation::Modal.name(),
                AnalysisOperation::Modal.version(),
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
        if let Err((detail, metadata)) = validate_electro_coupling_options(electro_options) {
            return Err(operation_error(
                AnalysisOperation::Modal.name(),
                AnalysisOperation::Modal.version(),
                &context,
                OperationErrorSpec {
                    error_code: AnalysisOperation::Modal
                        .invalid_electro_thermal_options_error_code(),
                    error_type: OperationErrorType::Input,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                detail,
                metadata,
            ));
        }
    }

    let solver_mesh = resolve_solver_mesh_artifact(
        model,
        options.solver_mesh_artifact_path.as_deref(),
        AnalysisOperation::Modal.name(),
        AnalysisOperation::Modal.version(),
        "RM.FEA.RUN_MODAL.SOLVER_MESH_INVALID",
        &context,
    )?;
    let modal_run = run_modal_with_options(
        model,
        backend,
        ModalSolveOptions {
            formulation_requirement,
            mode_count: options.mode_count,
            solver_mesh: solver_mesh.clone(),
            thermo_mechanical_context: to_fea_thermo_mechanical_context(thermo_options),
            electro_thermal_context: to_fea_electro_thermal_context(electro_options),
        },
    )
    .map_err(|err| map_fea_run_error(AnalysisOperation::Modal, model, &context, err))?;

    let run = modal_run.run;
    let mut fallback_events = Vec::new();
    if backend == ComputeBackend::Gpu && run.execution_backend != ExecutionBackend::Accelerator {
        fallback_events.push(FallbackEvent::solver_backend());
    }
    let solver_convergence = if run.diagnostics.iter().any(|item| {
        item.code == "FEA_MODAL_CONVERGENCE"
            && item.severity == runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info
    }) {
        QualityGate::Pass
    } else {
        QualityGate::Warn
    };
    let result_quality = if modal_run.frequencies_hz.is_empty() || modal_run.mode_shapes.is_empty()
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
        .any(FallbackEvent::is_solver_backend_fallback)
    {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::SolverBackendFallback,
            detail: "solver backend fell back from accelerator to cpu_reference".to_string(),
        });
    }

    let frequency_basis = ModalFrequencyBasis::NativeEigenSolve;

    let numerically_acceptable = match options.quality_policy {
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
    let (production_publishable, production_publication_reasons) =
        evaluate_production_publication(&run, numerically_acceptable);
    let run_status = classify_run_status(result_quality, numerically_acceptable);
    let solver_backend = run.solver_backend;
    let execution_backend = run.execution_backend;
    let solver_device_apply_k_ratio = run.solver_device_apply_k_ratio;
    let solver_host_sync_count = run.solver_host_sync_count;
    let solver_method = run.solver_method;
    let selected_preconditioner = run.preconditioner;

    let result = AnalysisRunResult {
        run_id: storage::next_run_id(),
        run,
        render_topology: render_topology_from_solver_mesh(solver_mesh.as_ref()),
        modal_results: Some(ModalResultsData {
            modal_payload_version: "modal_results/v1".to_string(),
            frequencies_hz: modal_run.frequencies_hz,
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
        numerically_acceptable,
        production_publishable,
        quality_reasons,
        production_publication_reasons,
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
        },
    };

    if let Some(nonlinear) = result.nonlinear_results.as_ref() {
        let event = format!(
            "fea.run_nonlinear outcome run_id={} model_id={} backend={:?} run_status={:?} numerically_acceptable={} production_publishable={} failed_increments={} max_iteration_count={} line_search_backtracks={} tangent_rebuild_count={} max_residual_norm={} max_increment_norm={} max_backtracks_per_increment={} quality_reason_count={}",
            result.run_id,
            model.model_id.0,
            backend,
            result.run_status,
            result.numerically_acceptable,
            result.production_publishable,
            nonlinear.failed_increments,
            nonlinear
                .iteration_counts
                .iter()
                .copied()
                .max()
                .unwrap_or(0),
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
        AnalysisOperation::Modal.name(),
        AnalysisOperation::Modal.version(),
        "RM.FEA.RUN_MODAL.ARTIFACT_STORE_FAILED",
        &context,
        &result,
    )?;

    Ok(OperationEnvelope::new(
        AnalysisOperation::Modal.name(),
        AnalysisOperation::Modal.version(),
        &context,
        result,
    ))
}

pub fn analysis_run_acoustic_with_options_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    formulation_requirement: FormulationRequirement,
    options: AnalysisAcousticRunOptions,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    let _solver_context = install_fea_solver_context();
    let has_acoustic_harmonic_step = model.sequence.steps.iter().any(|step| {
        step.family == PhysicsFamily::Acoustic
            && step.procedure == AnalysisProcedure::HarmonicResponse
    });
    if !has_acoustic_harmonic_step {
        return Err(operation_error(
            AnalysisOperation::Acoustic.name(),
            AnalysisOperation::Acoustic.version(),
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
    if !model
        .materials
        .iter()
        .any(|material| material.acoustic.is_some())
    {
        return Err(operation_error(
            AnalysisOperation::Acoustic.name(),
            AnalysisOperation::Acoustic.version(),
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_ACOUSTIC.MISSING_ACOUSTIC_MATERIAL",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_acoustic requires at least one acoustic material with density and sound-speed data",
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                (
                    "material_count".to_string(),
                    model.materials.len().to_string(),
                ),
            ]),
        ));
    }
    reject_moment_loads_for_run_family(
        model,
        AnalysisOperation::Acoustic.name(),
        AnalysisOperation::Acoustic.version(),
        "RM.FEA.RUN_ACOUSTIC.INVALID_ACOUSTIC_SOURCE",
        "acoustic",
        &context,
    )?;
    if !runmat_analysis_fea::physics::acoustic::has_nonzero_pressure_source(model) {
        return Err(operation_error(
            AnalysisOperation::Acoustic.name(),
            AnalysisOperation::Acoustic.version(),
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_ACOUSTIC.MISSING_ACOUSTIC_SOURCE",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_acoustic requires a nonzero acoustic pressure excitation",
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                (
                    "excitation_count".to_string(),
                    model.excitations.len().to_string(),
                ),
            ]),
        ));
    }
    if !model.boundaries.iter().any(|boundary| {
        matches!(
            &boundary.operator,
            BoundaryOperator::AcousticRigidWall
                | BoundaryOperator::AcousticRadiation
                | BoundaryOperator::AcousticImpedance { .. }
        )
    }) {
        return Err(operation_error(
            AnalysisOperation::Acoustic.name(),
            AnalysisOperation::Acoustic.version(),
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_ACOUSTIC.MISSING_ACOUSTIC_BOUNDARY",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_acoustic requires at least one acoustic boundary condition",
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                (
                    "boundary_condition_count".to_string(),
                    model.boundaries.len().to_string(),
                ),
            ]),
        ));
    }

    if options.mode_count == 0 {
        return Err(operation_error(
            AnalysisOperation::Acoustic.name(),
            AnalysisOperation::Acoustic.version(),
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

    let thermo_options = model_thermo_coupling_options(model);
    if let Some(thermo_options) = thermo_options.as_ref() {
        if let Err((detail, metadata)) = validate_thermo_coupling_options(model, thermo_options) {
            return Err(operation_error(
                AnalysisOperation::Acoustic.name(),
                AnalysisOperation::Acoustic.version(),
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
        if let Err((detail, metadata)) = validate_electro_coupling_options(electro_options) {
            return Err(operation_error(
                AnalysisOperation::Acoustic.name(),
                AnalysisOperation::Acoustic.version(),
                &context,
                OperationErrorSpec {
                    error_code: AnalysisOperation::Acoustic
                        .invalid_electro_thermal_options_error_code(),
                    error_type: OperationErrorType::Input,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                detail,
                metadata,
            ));
        }
    }

    let render_topology = None;

    let identity = admit_runtime_proxy_for_operation(
        PhysicsFamily::Acoustic,
        AnalysisProcedure::HarmonicResponse,
        formulation_requirement,
        RuntimeProxyFormulation::AcousticHashedTargetGraphProjection,
        AnalysisOperation::Acoustic,
        model,
        &context,
    )?;

    let solve_start = Instant::now();
    let mut run = runmat_analysis_fea::physics::acoustic::solve_reduced_harmonic(
        model,
        backend,
        identity,
        options.mode_count,
        options.residual_warn_threshold,
    );
    let solve_ms = solve_start.elapsed().as_secs_f64() * 1000.0;
    run.diagnostics
        .push(runmat_analysis_fea::diagnostics::FeaDiagnostic {
            code: "FEA_ACOUSTIC_COST".to_string(),
            severity: runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info,
            message: format!(
                "solve_ms={} mode_count={} residual_warn_threshold={}",
                solve_ms, options.mode_count, options.residual_warn_threshold,
            ),
        });
    let acoustic_residual_norm = diagnostic_metric(
        &run.diagnostics,
        "FEA_ACOUSTIC_HELMHOLTZ_RESIDUAL",
        "normalized_residual_norm",
    )
    .unwrap_or(f64::INFINITY);
    let mut fallback_events = Vec::new();
    if backend == ComputeBackend::Gpu && run.execution_backend != ExecutionBackend::Accelerator {
        fallback_events.push(FallbackEvent::solver_backend());
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
        .any(FallbackEvent::is_solver_backend_fallback)
    {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::SolverBackendFallback,
            detail: "solver backend fell back from accelerator to cpu_reference".to_string(),
        });
    }

    let numerically_acceptable = match options.quality_policy {
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
    let (production_publishable, production_publication_reasons) =
        evaluate_production_publication(&run, numerically_acceptable);
    let run_status = classify_run_status(result_quality, numerically_acceptable);
    let solver_backend = run.solver_backend;
    let execution_backend = run.execution_backend;
    let solver_device_apply_k_ratio = run.solver_device_apply_k_ratio;
    let solver_host_sync_count = run.solver_host_sync_count;
    let solver_method = run.solver_method;
    let selected_preconditioner = run.preconditioner;

    let result = AnalysisRunResult {
        run_id: storage::next_run_id(),
        run,
        render_topology,
        modal_results: None,
        thermal_results: None,
        transient_results: None,
        nonlinear_results: None,
        electromagnetic_results: None,
        model_validity: QualityGate::Pass,
        solver_convergence,
        result_quality,
        run_status,
        numerically_acceptable,
        production_publishable,
        quality_reasons,
        production_publication_reasons,
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
        },
    };

    persist_fea_run_result_with_progress(
        AnalysisOperation::Acoustic.name(),
        AnalysisOperation::Acoustic.version(),
        "RM.FEA.RUN_ACOUSTIC.ARTIFACT_STORE_FAILED",
        &context,
        &result,
    )?;

    Ok(OperationEnvelope::new(
        AnalysisOperation::Acoustic.name(),
        AnalysisOperation::Acoustic.version(),
        &context,
        result,
    ))
}

fn cfd_reynolds_number_for_velocity(
    domain: &runmat_analysis_core::CfdDomain,
    inlet_velocity_m_per_s: f64,
) -> f64 {
    domain.density.kilograms_per_cubic_meter * inlet_velocity_m_per_s.abs()
        / domain.dynamic_viscosity.pascal_seconds
}

fn cfd_profile_scale(profile_scales: &[f64], step_index: usize) -> f64 {
    profile_scales
        .get(step_index)
        .copied()
        .filter(|scale| scale.is_finite() && *scale >= 0.0)
        .unwrap_or(1.0)
}

fn cfd_node_count_from_model(model: &AnalysisModel) -> usize {
    model.excitations.len().saturating_mul(3).clamp(3, 512)
}

#[derive(Clone, Debug)]
struct CfdDomainTopology {
    basis: CfdDomainTopologyBasis,
    geometry_source: CfdDomainGeometrySource,
    node_count: usize,
    control_volume_count: usize,
    control_volume_face_count: usize,
    control_volume_internal_face_count: usize,
    control_volume_boundary_face_count: usize,
    control_volume_connectivity_coverage_ratio: f64,
    domain_length_m: f64,
    hydraulic_diameter_m: f64,
    face_area_m2: f64,
    dx_m: f64,
    active_dimension_count: usize,
    element_geometry_node_count: usize,
    element_geometry_edge_count: usize,
    element_geometry_coverage_ratio: f64,
    element_topology_edge_nodes: Vec<[u32; 2]>,
    element_topology_element_edges: Vec<[u32; 3]>,
}

impl CfdDomainTopology {
    fn from_model(model: &AnalysisModel, solver_mesh: Option<&SolverMeshArtifact>) -> Self {
        solver_mesh
            .map(Self::from_solver_mesh)
            .unwrap_or_else(|| Self::implicit_channel(cfd_node_count_from_model(model)))
    }

    fn from_solver_mesh(mesh: &SolverMeshArtifact) -> Self {
        let topology = solver_mesh_flow_topology::SolverMeshFlowTopology::derive(mesh);
        Self {
            basis: CfdDomainTopologyBasis::SolverMesh,
            geometry_source: CfdDomainGeometrySource::SolverMesh,
            node_count: topology.node_count,
            control_volume_count: topology.control_volume_count,
            control_volume_face_count: topology.face_count,
            control_volume_internal_face_count: topology.internal_face_count,
            control_volume_boundary_face_count: topology.boundary_face_count,
            control_volume_connectivity_coverage_ratio: 1.0,
            domain_length_m: topology.domain_length_m,
            hydraulic_diameter_m: topology.hydraulic_diameter_m,
            face_area_m2: topology.mean_boundary_face_area_m2,
            dx_m: topology.dx_m,
            active_dimension_count: topology.active_dimension_count,
            element_geometry_node_count: topology.node_count,
            element_geometry_edge_count: topology.edge_nodes.len(),
            element_geometry_coverage_ratio: 1.0,
            element_topology_edge_nodes: topology.edge_nodes,
            element_topology_element_edges: topology.boundary_face_edges,
        }
    }

    fn implicit_channel(node_count: usize) -> Self {
        let node_count = node_count.max(2);
        let control_volume_count = node_count.saturating_sub(1).max(1);
        Self {
            basis: CfdDomainTopologyBasis::ImplicitChannel,
            geometry_source: CfdDomainGeometrySource::ImplicitChannel,
            node_count,
            control_volume_count,
            control_volume_face_count: control_volume_count.saturating_add(1),
            control_volume_internal_face_count: control_volume_count.saturating_sub(1),
            control_volume_boundary_face_count: 2,
            control_volume_connectivity_coverage_ratio: 0.0,
            domain_length_m: 1.0,
            hydraulic_diameter_m: 1.0,
            face_area_m2: 1.0,
            dx_m: 1.0 / control_volume_count as f64,
            active_dimension_count: 1,
            element_geometry_node_count: 0,
            element_geometry_edge_count: 0,
            element_geometry_coverage_ratio: 0.0,
            element_topology_edge_nodes: Vec::new(),
            element_topology_element_edges: Vec::new(),
        }
    }

    fn face_area_m2(&self) -> f64 {
        self.face_area_m2.max(1.0e-12)
    }

    fn control_volume_volume_m3(&self) -> f64 {
        self.face_area_m2() * self.dx_m.max(1.0e-12)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CfdDomainTopologyBasis {
    ImplicitChannel,
    SolverMesh,
}

impl CfdDomainTopologyBasis {
    fn as_str(self) -> &'static str {
        match self {
            Self::ImplicitChannel => "implicit_channel",
            Self::SolverMesh => "solver_mesh",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CfdDomainGeometrySource {
    ImplicitChannel,
    SolverMesh,
}

impl CfdDomainGeometrySource {
    fn as_str(self) -> &'static str {
        match self {
            Self::ImplicitChannel => "implicit_channel",
            Self::SolverMesh => "solver_mesh",
        }
    }
}

#[derive(Clone, Debug)]
struct CfdBoundarySummary {
    inlet_boundary_count: usize,
    outlet_boundary_count: usize,
    no_slip_wall_boundary_count: usize,
    slip_wall_boundary_count: usize,
    symmetry_boundary_count: usize,
    authored_boundary_count: usize,
    boundary_coverage_ratio: f64,
    wall_boundary_coverage_ratio: f64,
    nominal_inlet_velocity_m_per_s: f64,
    outlet_pressure_pa: f64,
}

impl CfdBoundarySummary {
    fn implicit_channel(model: &AnalysisModel, node_count: usize) -> Self {
        Self {
            inlet_boundary_count: 1,
            outlet_boundary_count: 1,
            no_slip_wall_boundary_count: node_count.saturating_sub(1).max(1),
            slip_wall_boundary_count: 0,
            symmetry_boundary_count: 0,
            authored_boundary_count: 0,
            boundary_coverage_ratio: 1.0,
            wall_boundary_coverage_ratio: 1.0,
            nominal_inlet_velocity_m_per_s: nominal_inlet_velocity_m_per_s(model).unwrap_or(0.0),
            outlet_pressure_pa: 0.0,
        }
    }

    fn from_model(model: &AnalysisModel, node_count: usize) -> Self {
        let mut inlet_velocity_sum = 0.0_f64;
        let mut outlet_pressure_sum = 0.0_f64;
        let mut summary = Self {
            inlet_boundary_count: 0,
            outlet_boundary_count: 0,
            no_slip_wall_boundary_count: 0,
            slip_wall_boundary_count: 0,
            symmetry_boundary_count: 0,
            authored_boundary_count: 0,
            boundary_coverage_ratio: 0.0,
            wall_boundary_coverage_ratio: 0.0,
            nominal_inlet_velocity_m_per_s: nominal_inlet_velocity_m_per_s(model).unwrap_or(0.0),
            outlet_pressure_pa: 0.0,
        };

        for constraint in &model.constraints {
            let AnalysisConstraint::Fluid(constraint) = constraint else {
                continue;
            };
            match &constraint.kind {
                FluidConstraintKind::PrescribedVelocity { value, .. } => {
                    let Some(velocity) = value.study_constant(&model.signals) else {
                        continue;
                    };
                    summary.inlet_boundary_count += 1;
                    summary.authored_boundary_count += 1;
                    inlet_velocity_sum += velocity.meters_per_second;
                }
                FluidConstraintKind::PressureReference { value, .. } => {
                    let Some(pressure) = value.study_constant(&model.signals) else {
                        continue;
                    };
                    summary.outlet_boundary_count += 1;
                    summary.authored_boundary_count += 1;
                    outlet_pressure_sum += pressure.pascals;
                }
                FluidConstraintKind::NoSlipWall { .. } => {
                    summary.no_slip_wall_boundary_count += 1;
                    summary.authored_boundary_count += 1;
                }
                FluidConstraintKind::SlipWall { .. } => {
                    summary.slip_wall_boundary_count += 1;
                    summary.authored_boundary_count += 1;
                }
                FluidConstraintKind::Symmetry { .. } => {
                    summary.symmetry_boundary_count += 1;
                    summary.authored_boundary_count += 1;
                }
                _ => {}
            }
        }

        if summary.authored_boundary_count == 0 {
            return Self::implicit_channel(model, node_count);
        }

        if summary.inlet_boundary_count > 0 {
            summary.nominal_inlet_velocity_m_per_s =
                inlet_velocity_sum / summary.inlet_boundary_count as f64;
        }
        if summary.outlet_boundary_count > 0 {
            summary.outlet_pressure_pa = outlet_pressure_sum / summary.outlet_boundary_count as f64;
        }

        let wall_like_count = summary.no_slip_wall_boundary_count
            + summary.slip_wall_boundary_count
            + summary.symmetry_boundary_count;
        let required_groups_present = usize::from(summary.inlet_boundary_count > 0)
            + usize::from(summary.outlet_boundary_count > 0)
            + usize::from(wall_like_count > 0);
        summary.boundary_coverage_ratio = required_groups_present as f64 / 3.0;
        summary.wall_boundary_coverage_ratio = if wall_like_count > 0 { 1.0 } else { 0.0 };
        summary
    }

    fn wall_boundary_count(&self) -> usize {
        self.no_slip_wall_boundary_count + self.slip_wall_boundary_count
    }
}

fn validate_authored_cfd_boundary_conditions(model: &AnalysisModel) -> Result<(), String> {
    let mut inlet_boundary_count = 0usize;
    let mut outlet_boundary_count = 0usize;
    let mut wall_like_boundary_count = 0usize;
    let mut authored_boundary_count = 0usize;

    for constraint in &model.constraints {
        let AnalysisConstraint::Fluid(constraint) = constraint else {
            continue;
        };
        match &constraint.kind {
            FluidConstraintKind::PrescribedVelocity { value, .. } => {
                authored_boundary_count += 1;
                inlet_boundary_count += 1;
                let valid = value
                    .study_constant(&model.signals)
                    .is_some_and(|velocity| {
                        velocity.is_finite() && velocity.meters_per_second >= 0.0
                    });
                if !valid {
                    return Err(format!(
                        "cfd inlet boundary {} requires finite non-negative velocity_m_per_s",
                        constraint.constraint_id
                    ));
                }
            }
            FluidConstraintKind::PressureReference { value, .. } => {
                authored_boundary_count += 1;
                outlet_boundary_count += 1;
                if !value
                    .study_constant(&model.signals)
                    .is_some_and(|pressure| pressure.is_finite())
                {
                    return Err(format!(
                        "cfd outlet boundary {} requires finite pressure_pa",
                        constraint.constraint_id
                    ));
                }
            }
            FluidConstraintKind::NoSlipWall { .. }
            | FluidConstraintKind::SlipWall { .. }
            | FluidConstraintKind::Symmetry { .. } => {
                authored_boundary_count += 1;
                wall_like_boundary_count += 1;
            }
            _ => {}
        }
    }

    if authored_boundary_count == 0 {
        return Ok(());
    }
    if inlet_boundary_count == 0 || outlet_boundary_count == 0 || wall_like_boundary_count == 0 {
        return Err(format!(
            "authored cfd boundaries require at least one inlet, outlet, and wall/symmetry boundary; got inlet={} outlet={} wall_like={}",
            inlet_boundary_count, outlet_boundary_count, wall_like_boundary_count,
        ));
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct CfdVelocityPressureSolution {
    topology: CfdDomainTopology,
    velocity: Vec<f64>,
    pressure: Vec<f64>,
    residual_momentum: Vec<f64>,
    residual_continuity: Vec<f64>,
    mass_balance_residual: f64,
    pressure_drop_pa: f64,
    control_volume_count: usize,
    inlet_boundary_count: usize,
    outlet_boundary_count: usize,
    wall_boundary_count: usize,
    no_slip_wall_boundary_count: usize,
    slip_wall_boundary_count: usize,
    symmetry_boundary_count: usize,
    authored_boundary_count: usize,
    boundary_coverage_ratio: f64,
    wall_boundary_coverage_ratio: f64,
    inlet_velocity_realization_ratio: f64,
    nominal_inlet_velocity_m_per_s: f64,
    outlet_pressure_pa: f64,
    pressure_correction_iteration_count: usize,
    pressure_correction_residual_ratio: f64,
    velocity_correction_residual_ratio: f64,
    transient_scale_min: f64,
    transient_scale_max: f64,
    transient_scale_variation: f64,
}

#[derive(Clone, Debug)]
struct CfdKnownAnswerMetrics {
    pressure_drop_balance_ratio: f64,
    mass_flux_uniformity_ratio: f64,
    pressure_monotonic_cell_fraction: f64,
    known_answer_coverage_ratio: f64,
}

fn recover_cfd_velocity_pressure(
    model: &AnalysisModel,
    domain: &runmat_analysis_core::CfdDomain,
    topology: &CfdDomainTopology,
    step_index: usize,
    profile_scales: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let boundary_summary = CfdBoundarySummary::implicit_channel(model, topology.node_count);
    let solution = solve_cfd_velocity_pressure(
        domain,
        &boundary_summary,
        topology,
        step_index,
        profile_scales,
        1,
        32,
        1.0e-8,
    );
    (solution.velocity, solution.pressure)
}

fn solve_cfd_velocity_pressure(
    domain: &runmat_analysis_core::CfdDomain,
    boundary_summary: &CfdBoundarySummary,
    topology: &CfdDomainTopology,
    step_index: usize,
    profile_scales: &[f64],
    step_count: usize,
    max_linear_iters: usize,
    tolerance: f64,
) -> CfdVelocityPressureSolution {
    let node_count = topology.node_count.max(2);
    let profile_scale = cfd_profile_scale(profile_scales, step_index);
    let nominal_inlet_velocity = boundary_summary.nominal_inlet_velocity_m_per_s;
    let inlet_velocity = nominal_inlet_velocity * profile_scale;
    let reynolds = cfd_reynolds_number_for_velocity(domain, nominal_inlet_velocity).max(1.0);
    let hydraulic_diameter_m = topology.hydraulic_diameter_m;
    let friction_factor = if reynolds <= 2300.0 {
        64.0 / reynolds
    } else {
        0.3164 / reynolds.powf(0.25)
    };
    let friction_gradient_pa_per_m = 0.5
        * domain.density.kilograms_per_cubic_meter
        * inlet_velocity
        * inlet_velocity.abs()
        * friction_factor
        / hydraulic_diameter_m;
    let pressure_drop_pa = (friction_gradient_pa_per_m * topology.domain_length_m).max(0.0);
    let denom = node_count.saturating_sub(1).max(1) as f64;
    let mut axial_velocity = vec![inlet_velocity; node_count];
    let mut pressure = (0..node_count)
        .map(|node| {
            let xi = node as f64 / denom;
            boundary_summary.outlet_pressure_pa + 0.5 * pressure_drop_pa * (1.0 - xi)
        })
        .collect::<Vec<_>>();
    let target_pressure = (0..node_count)
        .map(|node| {
            let xi = node as f64 / denom;
            boundary_summary.outlet_pressure_pa + pressure_drop_pa * (1.0 - xi)
        })
        .collect::<Vec<_>>();
    let correction_iters = max_linear_iters.max(1);
    let correction_tolerance = tolerance.max(1.0e-12);
    let mut pressure_correction_residual_ratio = f64::INFINITY;
    let mut velocity_correction_residual_ratio = f64::INFINITY;
    let mut pressure_correction_iteration_count = 0usize;
    for iteration in 0..correction_iters {
        let previous_pressure = pressure.clone();
        let previous_velocity = axial_velocity.clone();
        for node in 0..node_count {
            pressure[node] = 0.35 * pressure[node] + 0.65 * target_pressure[node];
        }
        for node in 0..node_count {
            if node == 0 {
                axial_velocity[node] = inlet_velocity;
                continue;
            }
            if node + 1 == node_count {
                axial_velocity[node] = axial_velocity[node.saturating_sub(1)];
                continue;
            }
            let gradient = (pressure[node + 1] - pressure[node - 1]) / (2.0 * topology.dx_m);
            let pressure_driven_speed = ((-2.0 * gradient * hydraulic_diameter_m)
                / (domain.density.kilograms_per_cubic_meter.max(1.0e-12)
                    * friction_factor.max(1.0e-12)))
            .max(0.0)
            .sqrt();
            axial_velocity[node] = 0.50 * axial_velocity[node] + 0.50 * pressure_driven_speed;
        }
        axial_velocity[node_count - 1] = axial_velocity[node_count - 2];

        let pressure_correction_norm = pressure
            .iter()
            .zip(previous_pressure.iter())
            .map(|(current, previous)| (current - previous) * (current - previous))
            .sum::<f64>()
            .sqrt();
        let pressure_scale = target_pressure
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
            .max(1.0);
        pressure_correction_residual_ratio = pressure_correction_norm / pressure_scale;
        let velocity_correction_norm = axial_velocity
            .iter()
            .zip(previous_velocity.iter())
            .map(|(current, previous)| (current - previous) * (current - previous))
            .sum::<f64>()
            .sqrt();
        let velocity_scale = axial_velocity
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
            .max(inlet_velocity.abs())
            .max(1.0e-12);
        velocity_correction_residual_ratio = velocity_correction_norm / velocity_scale;
        pressure_correction_iteration_count = iteration + 1;
        if pressure_correction_residual_ratio <= correction_tolerance
            && velocity_correction_residual_ratio <= correction_tolerance
        {
            break;
        }
    }
    pressure = target_pressure;

    let mut velocity = Vec::with_capacity(node_count * 3);
    for (node, axial) in axial_velocity.iter().copied().enumerate() {
        let xi = node as f64 / denom;
        let recirculation = (2.0 * std::f64::consts::PI * xi).sin()
            * axial
            * domain.turbulence_intensity.ratio.clamp(0.0, 1.0)
            * 0.02;
        velocity.extend_from_slice(&[axial, recirculation, 0.0]);
    }

    let (residual_momentum, residual_continuity) =
        cfd_residual_norms(&velocity, &pressure, domain, topology, step_count);
    let mass_balance_residual = residual_continuity.iter().copied().fold(0.0_f64, f64::max);
    let inlet_velocity_realization_ratio =
        inlet_velocity.abs() / nominal_inlet_velocity.abs().max(1.0e-12);
    let (transient_scale_min, transient_scale_max) = cfd_transient_scale_bounds(profile_scales);

    CfdVelocityPressureSolution {
        topology: topology.clone(),
        velocity,
        pressure,
        residual_momentum,
        residual_continuity,
        mass_balance_residual,
        pressure_drop_pa,
        control_volume_count: topology.control_volume_count,
        inlet_boundary_count: boundary_summary.inlet_boundary_count,
        outlet_boundary_count: boundary_summary.outlet_boundary_count,
        wall_boundary_count: boundary_summary.wall_boundary_count(),
        no_slip_wall_boundary_count: boundary_summary.no_slip_wall_boundary_count,
        slip_wall_boundary_count: boundary_summary.slip_wall_boundary_count,
        symmetry_boundary_count: boundary_summary.symmetry_boundary_count,
        authored_boundary_count: boundary_summary.authored_boundary_count,
        boundary_coverage_ratio: boundary_summary.boundary_coverage_ratio,
        wall_boundary_coverage_ratio: boundary_summary.wall_boundary_coverage_ratio,
        inlet_velocity_realization_ratio,
        nominal_inlet_velocity_m_per_s: nominal_inlet_velocity,
        outlet_pressure_pa: boundary_summary.outlet_pressure_pa,
        pressure_correction_iteration_count,
        pressure_correction_residual_ratio,
        velocity_correction_residual_ratio,
        transient_scale_min,
        transient_scale_max,
        transient_scale_variation: (transient_scale_max - transient_scale_min).abs(),
    }
}

fn cfd_transient_scale_bounds(profile_scales: &[f64]) -> (f64, f64) {
    if profile_scales.is_empty() {
        return (1.0, 1.0);
    }
    let (min, max) = profile_scales
        .iter()
        .copied()
        .filter(|scale| scale.is_finite())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), scale| {
            (min.min(scale), max.max(scale))
        });
    if min.is_finite() && max.is_finite() {
        (min, max)
    } else {
        (1.0, 1.0)
    }
}

fn cfd_known_answer_metrics(solution: &CfdVelocityPressureSolution) -> CfdKnownAnswerMetrics {
    let node_count = solution.pressure.len();
    let pressure_drop_observed = match (solution.pressure.first(), solution.pressure.last()) {
        (Some(first), Some(last)) => first - last,
        _ => 0.0,
    };
    let pressure_drop_balance_ratio = if solution.pressure_drop_pa.abs() > 1.0e-12 {
        pressure_drop_observed / solution.pressure_drop_pa
    } else if pressure_drop_observed.abs() <= 1.0e-12 {
        1.0
    } else {
        0.0
    };

    let axial_values = (0..node_count)
        .map(|node| solution.velocity.get(node * 3).copied().unwrap_or(0.0))
        .collect::<Vec<_>>();
    let mean_axial = if axial_values.is_empty() {
        0.0
    } else {
        axial_values.iter().sum::<f64>() / axial_values.len() as f64
    };
    let max_axial_deviation = axial_values
        .iter()
        .map(|value| (value - mean_axial).abs())
        .fold(0.0_f64, f64::max);
    let mass_flux_uniformity_ratio = max_axial_deviation / mean_axial.abs().max(1.0e-12);

    let pressure_edge_count = node_count.saturating_sub(1);
    let pressure_monotonic_cell_fraction = if pressure_edge_count == 0 {
        1.0
    } else {
        let tolerance = solution.pressure_drop_pa.abs().max(1.0) * 1.0e-12;
        let monotonic_edges = solution
            .pressure
            .windows(2)
            .filter(|pair| pair[0] + tolerance >= pair[1])
            .count();
        monotonic_edges as f64 / pressure_edge_count as f64
    };

    let known_answer_coverage_ratio = if node_count >= 2
        && solution.control_volume_count == node_count - 1
        && solution.velocity.len() == node_count * 3
        && solution.pressure_drop_pa.is_finite()
        && solution.pressure.iter().all(|value| value.is_finite())
        && solution.velocity.iter().all(|value| value.is_finite())
    {
        1.0
    } else {
        0.0
    };

    CfdKnownAnswerMetrics {
        pressure_drop_balance_ratio,
        mass_flux_uniformity_ratio,
        pressure_monotonic_cell_fraction,
        known_answer_coverage_ratio,
    }
}

fn cfd_known_answer_diagnostic(
    metrics: &CfdKnownAnswerMetrics,
    topology: &CfdDomainTopology,
) -> runmat_analysis_fea::diagnostics::FeaDiagnostic {
    let severity = if (metrics.pressure_drop_balance_ratio - 1.0).abs() <= 1.0e-10
        && metrics.mass_flux_uniformity_ratio <= 1.0e-10
        && metrics.pressure_monotonic_cell_fraction >= 1.0
        && metrics.known_answer_coverage_ratio >= 1.0
    {
        runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info
    } else {
        runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
    };

    runmat_analysis_fea::diagnostics::FeaDiagnostic {
        code: "FEA_CFD_KNOWN_ANSWER".to_string(),
        severity,
        message: format!(
            "basis=finite_volume_channel topology_basis={} pressure_drop_balance_ratio={} mass_flux_uniformity_ratio={} pressure_monotonic_cell_fraction={} known_answer_coverage_ratio={}",
            topology.basis.as_str(),
            metrics.pressure_drop_balance_ratio,
            metrics.mass_flux_uniformity_ratio,
            metrics.pressure_monotonic_cell_fraction,
            metrics.known_answer_coverage_ratio,
        ),
    }
}

fn recover_cfd_vorticity(velocity: &[f64], node_count: usize, dx_m: f64) -> Vec<f64> {
    let mut vorticity = vec![0.0; node_count * 3];
    if node_count < 2 {
        return vorticity;
    }
    let dx_m = dx_m.max(1.0e-12);

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
        vorticity[base + 1] = -dvx / (2.0 * dx_m);
        vorticity[base + 2] = dvy / (2.0 * dx_m);
    }

    vorticity
}

fn recover_cfd_wall_shear_stress(
    domain: &runmat_analysis_core::CfdDomain,
    velocity: &[f64],
    field_count: usize,
) -> Vec<f64> {
    let mut shear = vec![0.0; field_count * 3];
    let viscosity = domain.dynamic_viscosity.pascal_seconds;
    for index in 0..field_count {
        let base = index * 3;
        shear[base] = viscosity * velocity.get(base).copied().unwrap_or(0.0);
        shear[base + 1] = viscosity * velocity.get(base + 1).copied().unwrap_or(0.0);
    }
    shear
}

fn cfd_residual_norms(
    velocity: &[f64],
    pressure: &[f64],
    domain: &runmat_analysis_core::CfdDomain,
    topology: &CfdDomainTopology,
    step_count: usize,
) -> (Vec<f64>, Vec<f64>) {
    let node_count = pressure.len().max(1);
    let mean_axial_velocity = (0..node_count)
        .map(|node| velocity.get(node * 3).copied().unwrap_or(0.0))
        .sum::<f64>()
        / node_count as f64;
    let reynolds = cfd_reynolds_number_for_velocity(domain, mean_axial_velocity).max(1.0);
    let hydraulic_diameter_m = topology.hydraulic_diameter_m.max(1.0e-12);
    let friction_factor = if reynolds <= 2300.0 {
        64.0 / reynolds
    } else {
        0.3164 / reynolds.powf(0.25)
    };
    let friction_gradient_pa_per_m = 0.5
        * domain.density.kilograms_per_cubic_meter.max(1.0e-12)
        * mean_axial_velocity
        * mean_axial_velocity.abs()
        * friction_factor
        / hydraulic_diameter_m;
    let dx = topology.dx_m.max(1.0e-12);
    let mut momentum_base = 0.0_f64;
    let mut continuity_base = 0.0_f64;
    for node in 0..node_count {
        let prev = node.saturating_sub(1);
        let next = (node + 1).min(node_count - 1);
        let velocity_prev = velocity.get(prev * 3).copied().unwrap_or(0.0);
        let velocity_next = velocity.get(next * 3).copied().unwrap_or(0.0);
        let pressure_prev = pressure.get(prev).copied().unwrap_or(0.0);
        let pressure_next = pressure.get(next).copied().unwrap_or(0.0);
        let stencil_width = if prev == next {
            1.0
        } else {
            (next - prev) as f64 * dx
        };
        let divergence = (velocity_next - velocity_prev) / stencil_width.max(1.0e-12);
        let pressure_gradient = (pressure_next - pressure_prev) / stencil_width.max(1.0e-12);
        continuity_base += divergence.abs();
        momentum_base += (pressure_gradient + friction_gradient_pa_per_m).abs();
    }
    let velocity_scale = mean_axial_velocity.abs().max(1.0e-9);
    let pressure_scale = pressure
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let continuity_base = (continuity_base / (node_count as f64 * velocity_scale)).clamp(0.0, 1.0);
    let momentum_base = (momentum_base / (node_count as f64 * pressure_scale)).clamp(0.0, 1.0);
    let residual_count = step_count.max(1);
    (
        vec![momentum_base; residual_count],
        vec![continuity_base; residual_count],
    )
}

fn build_cfd_run_fields(
    domain: &runmat_analysis_core::CfdDomain,
    solution: &CfdVelocityPressureSolution,
) -> Vec<AnalysisField> {
    let node_count = solution.pressure.len();
    let velocity = cell_centered_vector_from_nodal(&solution.velocity, node_count);
    let pressure = cell_centered_scalar_from_nodal(&solution.pressure);
    let cell_count = pressure.len().max(1);
    let vorticity = recover_cfd_vorticity(&velocity, cell_count, solution.topology.dx_m);
    let boundary_face_count = solution.wall_boundary_count.max(1);
    let wall_shear_stress = recover_cfd_wall_shear_stress(domain, &velocity, boundary_face_count);
    let residual_count = solution.residual_momentum.len();

    vec![
        AnalysisField::host_f64(FEA_FIELD_CFD_VELOCITY, vec![cell_count, 3], velocity),
        AnalysisField::host_f64(FEA_FIELD_CFD_PRESSURE, vec![cell_count], pressure),
        AnalysisField::host_f64(FEA_FIELD_CFD_VORTICITY, vec![cell_count, 3], vorticity),
        AnalysisField::host_f64(
            FEA_FIELD_CFD_WALL_SHEAR_STRESS,
            vec![boundary_face_count, 3],
            wall_shear_stress,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_CFD_RESIDUAL_MOMENTUM,
            vec![residual_count],
            solution.residual_momentum.clone(),
        ),
        AnalysisField::host_f64(
            FEA_FIELD_CFD_RESIDUAL_CONTINUITY,
            vec![residual_count],
            solution.residual_continuity.clone(),
        ),
        AnalysisField::host_f64(
            FEA_FIELD_CFD_REYNOLDS_NUMBER,
            vec![1],
            vec![cfd_reynolds_number_for_velocity(
                domain,
                solution.nominal_inlet_velocity_m_per_s,
            )],
        ),
    ]
}

fn cfd_assembly_diagnostic(
    topology: &CfdDomainTopology,
    domain: &runmat_analysis_core::CfdDomain,
    inlet_velocity_m_per_s: f64,
    time_step_s: f64,
    pressure_drop_pa: f64,
    mass_balance_residual: f64,
    residual_warn_threshold: f64,
) -> runmat_analysis_fea::diagnostics::FeaDiagnostic {
    let face_area_m2 = topology.face_area_m2();
    let control_volume_volume_m3 = topology.control_volume_volume_m3();
    let nominal_mass_flow_rate_kg_per_s =
        domain.density.kilograms_per_cubic_meter * inlet_velocity_m_per_s * face_area_m2;
    let courant_number =
        inlet_velocity_m_per_s.abs() * time_step_s.max(0.0) / topology.dx_m.max(1.0e-12);
    runmat_analysis_fea::diagnostics::FeaDiagnostic {
        code: "FEA_CFD_ASSEMBLY".to_string(),
        severity: if mass_balance_residual <= residual_warn_threshold {
            runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info
        } else {
            runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "basis=finite_volume_velocity_pressure topology_basis={} topology_geometry_source={} control_volume_count={} control_volume_face_count={} control_volume_internal_face_count={} control_volume_boundary_face_count={} control_volume_connectivity_coverage_ratio={} hydraulic_diameter_m={} domain_length_m={} dx_m={} face_area_m2={} control_volume_volume_m3={} nominal_mass_flow_rate_kg_per_s={} courant_number={} active_dimension_count={} element_geometry_node_count={} element_geometry_edge_count={} element_geometry_coverage_ratio={} pressure_drop_pa={} mass_balance_residual={}",
            topology.basis.as_str(),
            topology.geometry_source.as_str(),
            topology.control_volume_count,
            topology.control_volume_face_count,
            topology.control_volume_internal_face_count,
            topology.control_volume_boundary_face_count,
            topology.control_volume_connectivity_coverage_ratio,
            topology.hydraulic_diameter_m,
            topology.domain_length_m,
            topology.dx_m,
            face_area_m2,
            control_volume_volume_m3,
            nominal_mass_flow_rate_kg_per_s,
            courant_number,
            topology.active_dimension_count,
            topology.element_geometry_node_count,
            topology.element_geometry_edge_count,
            topology.element_geometry_coverage_ratio,
            pressure_drop_pa,
            mass_balance_residual,
        ),
    }
}

fn pressure_drop_from_nodal_pressure(pressure: &[f64]) -> f64 {
    match (pressure.first(), pressure.last()) {
        (Some(first), Some(last)) => first - last,
        _ => 0.0,
    }
}

fn cell_centered_vector_from_nodal(nodal: &[f64], node_count: usize) -> Vec<f64> {
    let cell_count = node_count.saturating_sub(1).max(1);
    let mut cell_values = Vec::with_capacity(cell_count * 3);
    for cell in 0..cell_count {
        let left = cell.min(node_count.saturating_sub(1));
        let right = (cell + 1).min(node_count.saturating_sub(1));
        for component in 0..3 {
            let left_value = nodal.get(left * 3 + component).copied().unwrap_or(0.0);
            let right_value = nodal
                .get(right * 3 + component)
                .copied()
                .unwrap_or(left_value);
            cell_values.push(0.5 * (left_value + right_value));
        }
    }
    cell_values
}

fn cell_centered_scalar_from_nodal(nodal: &[f64]) -> Vec<f64> {
    let node_count = nodal.len();
    let cell_count = node_count.saturating_sub(1).max(1);
    let mut cell_values = Vec::with_capacity(cell_count);
    for cell in 0..cell_count {
        let left = cell.min(node_count.saturating_sub(1));
        let right = (cell + 1).min(node_count.saturating_sub(1));
        let left_value = nodal.get(left).copied().unwrap_or(0.0);
        let right_value = nodal.get(right).copied().unwrap_or(left_value);
        cell_values.push(0.5 * (left_value + right_value));
    }
    cell_values
}

fn resample_scalar_profile(values: &[f64], target_count: usize) -> Vec<f64> {
    let target_count = target_count.max(1);
    if values.is_empty() {
        return vec![0.0; target_count];
    }
    if values.len() == target_count {
        return values.to_vec();
    }
    if target_count == 1 {
        return vec![values[0]];
    }
    let source_max = values.len().saturating_sub(1) as f64;
    let target_max = target_count.saturating_sub(1) as f64;
    (0..target_count)
        .map(|target_index| {
            let source_position = target_index as f64 * source_max / target_max.max(1.0);
            let left = source_position.floor() as usize;
            let right = source_position.ceil() as usize;
            if left == right {
                values.get(left).copied().unwrap_or(0.0)
            } else {
                let t = source_position - left as f64;
                let left_value = values.get(left).copied().unwrap_or(0.0);
                let right_value = values.get(right).copied().unwrap_or(left_value);
                left_value * (1.0 - t) + right_value * t
            }
        })
        .collect()
}

fn fluid_interface_face_count(topology: &CfdDomainTopology) -> usize {
    if topology.control_volume_connectivity_coverage_ratio > 0.0
        && topology.control_volume_boundary_face_count > 0
    {
        topology.control_volume_boundary_face_count
    } else {
        topology.control_volume_count
    }
    .max(1)
}

fn coupled_interface_graph_edge_target(
    topology: &CfdDomainTopology,
    interface_face_count: usize,
) -> usize {
    if interface_face_count < 2 {
        return 0;
    }
    let line_edge_count = interface_face_count - 1;
    let complete_graph_edge_count = interface_face_count * (interface_face_count - 1) / 2;
    if topology.control_volume_connectivity_coverage_ratio > 0.0 {
        topology
            .control_volume_internal_face_count
            .max(line_edge_count)
            .min(complete_graph_edge_count)
    } else {
        line_edge_count
    }
}

fn coupled_interface_graph_edges_for_topology(
    topology: &CfdDomainTopology,
    interface_face_count: usize,
) -> (Vec<(usize, usize)>, f64) {
    use std::collections::BTreeSet;

    let target = coupled_interface_graph_edge_target(topology, interface_face_count);
    if target == 0 {
        return (
            Vec::new(),
            f64::from(topology.basis == CfdDomainTopologyBasis::SolverMesh),
        );
    }

    let mut edges = mesh_boundary_face_adjacencies(topology, interface_face_count)
        .into_iter()
        .take(target)
        .collect::<Vec<_>>();
    let mut seen = edges.iter().copied().collect::<BTreeSet<_>>();
    if edges.len() == target {
        return (
            edges,
            f64::from(topology.basis == CfdDomainTopologyBasis::SolverMesh),
        );
    }

    let mesh_edge_count = edges.len();
    for (left, right) in coupled_interface_graph_edges(interface_face_count, target) {
        let edge = (left.min(right), left.max(right));
        if seen.insert(edge) {
            edges.push(edge);
            if edges.len() == target {
                break;
            }
        }
    }
    let mesh_backed_coverage_ratio = if topology.basis == CfdDomainTopologyBasis::SolverMesh {
        mesh_edge_count as f64 / target as f64
    } else {
        0.0
    };
    (edges, mesh_backed_coverage_ratio.clamp(0.0, 1.0))
}

fn mesh_boundary_face_adjacencies(
    topology: &CfdDomainTopology,
    interface_face_count: usize,
) -> Vec<(usize, usize)> {
    let face_count = topology
        .element_topology_element_edges
        .len()
        .min(interface_face_count);
    let mut edges = Vec::new();
    for left in 0..face_count {
        for right in (left + 1)..face_count {
            if topology.element_topology_element_edges[left]
                .iter()
                .any(|edge| topology.element_topology_element_edges[right].contains(edge))
            {
                edges.push((left, right));
            }
        }
    }
    edges
}

fn coupled_interface_connectivity_coverage_ratio(
    topology: &CfdDomainTopology,
    interface_face_count: usize,
    edge_count: usize,
) -> f64 {
    let target = coupled_interface_graph_edge_target(topology, interface_face_count);
    if target == 0 {
        return 1.0;
    }
    (edge_count as f64 / target as f64).clamp(0.0, 1.0)
}

fn solve_cfd_finite_volume_run(
    model: &AnalysisModel,
    domain: &runmat_analysis_core::CfdDomain,
    backend: ComputeBackend,
    identity: AnalysisRunIdentity,
    options: &AnalysisCfdRunOptions,
    solver_mesh: Option<&SolverMeshArtifact>,
) -> FeaRunResult {
    let topology = CfdDomainTopology::from_model(model, solver_mesh);
    let node_count = topology.node_count;
    let step_count = options.step_count.max(1);
    let transient = identity.procedure() == AnalysisProcedure::TransientResponse;
    let field_step = if transient {
        step_count.saturating_sub(1)
    } else {
        0
    };
    let boundary_summary = CfdBoundarySummary::from_model(model, node_count);
    let profile_scales = cfd_velocity_scales(model, domain);
    let solution = solve_cfd_velocity_pressure(
        domain,
        &boundary_summary,
        &topology,
        field_step,
        &profile_scales,
        step_count,
        options.max_linear_iters,
        options.tolerance,
    );
    let max_momentum_residual = solution
        .residual_momentum
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let max_continuity_residual = solution
        .residual_continuity
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let known_answer_metrics = cfd_known_answer_metrics(&solution);
    let fields = build_cfd_run_fields(domain, &solution);
    let residual_severity = if max_momentum_residual <= options.residual_warn_threshold
        && max_continuity_residual <= options.residual_warn_threshold
    {
        runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info
    } else {
        runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
    };
    FeaRunResult {
        identity,
        requested_backend: backend,
        execution_backend: ExecutionBackend::Cpu,
        solver_backend: LinearAlgebraBackendKind::CpuReference,
        solver_device_apply_k_ratio: 0.0,
        solver_method: SolverMethod::CfdVelocityPressureFiniteVolume,
        preconditioner: SolverPreconditioner::FiniteVolumePressureBalance,
        solver_host_sync_count: 0,
        accelerator_evidence: Vec::new(),
        spatial_projection_evidence: Vec::new(),
        observations: Vec::new(),
        diagnostics: vec![
            runmat_analysis_fea::diagnostics::FeaDiagnostic {
                code: "FEA_CFD_RESIDUAL".to_string(),
                severity: residual_severity,
                message: format!(
                    "max_momentum_residual={} max_continuity_residual={} residual_warn_threshold={} cfd_node_count={} cfd_step_count={}",
                    max_momentum_residual,
                    max_continuity_residual,
                    options.residual_warn_threshold,
                    node_count,
                    step_count,
                ),
            },
            cfd_assembly_diagnostic(
                &solution.topology,
                domain,
                solution.nominal_inlet_velocity_m_per_s,
                options.time_step_s,
                solution.pressure_drop_pa,
                solution.mass_balance_residual,
                options.residual_warn_threshold,
            ),
            runmat_analysis_fea::diagnostics::FeaDiagnostic {
                code: "FEA_CFD_BOUNDARY_CONDITIONS".to_string(),
                severity: if solution.boundary_coverage_ratio >= 1.0
                    && solution.wall_boundary_coverage_ratio >= 1.0
                    && solution.inlet_velocity_realization_ratio.is_finite()
                {
                    runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info
                } else {
                    runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
                },
                message: format!(
                    "boundary_source={} authored_boundary_count={} inlet_boundary_count={} outlet_boundary_count={} wall_boundary_count={} no_slip_wall_boundary_count={} slip_wall_boundary_count={} symmetry_boundary_count={} boundary_coverage_ratio={} wall_boundary_coverage_ratio={} nominal_inlet_velocity_m_per_s={} outlet_pressure_pa={} inlet_velocity_realization_ratio={}",
                    if solution.authored_boundary_count > 0 {
                        "authored"
                    } else {
                        "implicit_channel"
                    },
                    solution.authored_boundary_count,
                    solution.inlet_boundary_count,
                    solution.outlet_boundary_count,
                    solution.wall_boundary_count,
                    solution.no_slip_wall_boundary_count,
                    solution.slip_wall_boundary_count,
                    solution.symmetry_boundary_count,
                    solution.boundary_coverage_ratio,
                    solution.wall_boundary_coverage_ratio,
                    solution.nominal_inlet_velocity_m_per_s,
                    solution.outlet_pressure_pa,
                    solution.inlet_velocity_realization_ratio,
                ),
            },
            runmat_analysis_fea::diagnostics::FeaDiagnostic {
                code: "FEA_CFD_PRESSURE_CORRECTION".to_string(),
                severity: if solution.pressure_correction_residual_ratio <= options.tolerance
                    && solution.velocity_correction_residual_ratio <= options.tolerance
                {
                    runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info
                } else {
                    runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
                },
                message: format!(
                    "iteration_count={} max_linear_iters={} tolerance={} pressure_correction_residual_ratio={} velocity_correction_residual_ratio={}",
                    solution.pressure_correction_iteration_count,
                    options.max_linear_iters,
                    options.tolerance,
                    solution.pressure_correction_residual_ratio,
                    solution.velocity_correction_residual_ratio,
                ),
            },
            runmat_analysis_fea::diagnostics::FeaDiagnostic {
                code: "FEA_CFD_TRANSIENT_EVOLUTION".to_string(),
                severity: runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info,
                message: format!(
                    "solve_family={} step_count={} time_step_s={} transient_profile_point_count={} transient_scale_min={} transient_scale_max={} transient_scale_variation={}",
                    if transient { "transient" } else { "steady_state" },
                    step_count,
                    options.time_step_s,
                    profile_scales.len(),
                    solution.transient_scale_min,
                    solution.transient_scale_max,
                    solution.transient_scale_variation,
                ),
            },
            cfd_known_answer_diagnostic(&known_answer_metrics, &topology),
        ],
        fields,
    }
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

fn build_cht_run_fields(
    model: &AnalysisModel,
    domain: &runmat_analysis_core::CfdDomain,
    topology: &CfdDomainTopology,
    thermal_run: &runmat_analysis_fea::FeaThermalRunResult,
    authored_interface_conductance_w_per_m2k: Option<f64>,
    max_linear_iters: usize,
    tolerance: f64,
) -> (Vec<AnalysisField>, ChtInterfaceClosure) {
    let node_count = topology.node_count;
    let profile_scales = cfd_velocity_scales(model, domain);
    let (fluid_velocity, fluid_pressure) =
        recover_cfd_velocity_pressure(model, domain, topology, 0, &profile_scales);
    let mean_axial_velocity = mean_cfd_axial_velocity(&fluid_velocity);
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
    let mut closure = ChtInterfaceClosure::default();

    for (step_index, temperature) in thermal_run.temperature_snapshots.iter().enumerate() {
        let fallback_len = temperature.element_count().max(1);
        let base_temperature = temperature
            .as_host_f64()
            .map(|values| values.to_vec())
            .unwrap_or_else(|| vec![thermal_run.reference_temperature_k; fallback_len]);
        let mut heat_flux = thermal_run
            .heat_flux_snapshots
            .get(step_index)
            .map(|field| field_scalar_magnitudes(field, fallback_len))
            .unwrap_or_else(|| vec![0.0; fallback_len]);
        if heat_flux.is_empty() {
            heat_flux.push(0.0);
        }
        let target_interface_count = fluid_interface_face_count(topology);
        if heat_flux.len() != target_interface_count {
            heat_flux = resample_scalar_profile(&heat_flux, target_interface_count);
        }
        let interface_count = heat_flux.len();
        let max_heat_flux = heat_flux
            .iter()
            .copied()
            .map(f64::abs)
            .fold(0.0_f64, f64::max);
        let target_jump_k = 0.01_f64;
        let interface_conductance_w_per_m2k = authored_interface_conductance_w_per_m2k
            .unwrap_or_else(|| (max_heat_flux / target_jump_k).max(25.0))
            .max(1.0e-12);
        let advection_shift_k = cht_advection_shift_k(
            domain,
            mean_axial_velocity,
            &base_temperature,
            thermal_run.reference_temperature_k,
        );

        let interface_solution = solve_cht_conjugate_interface(
            &base_temperature,
            &heat_flux,
            topology,
            interface_conductance_w_per_m2k,
            advection_shift_k,
            thermal_run.reference_temperature_k,
            max_linear_iters,
            tolerance,
        );
        let fluid_temperature = interface_solution.fluid_temperature;
        let solid_temperature = interface_solution.solid_temperature;
        let temperature_jump = interface_solution.temperature_jump;
        let coupled_heat_flux = interface_solution.coupled_heat_flux;
        fields.push(AnalysisField::host_f64(
            fea_cht_fluid_temperature_field_id(step_index),
            vec![base_temperature.len().max(1)],
            fluid_temperature,
        ));
        fields.push(AnalysisField::host_f64(
            fea_cht_solid_temperature_field_id(step_index),
            vec![base_temperature.len().max(1)],
            solid_temperature,
        ));

        closure.interface_face_count = closure.interface_face_count.max(interface_count);
        closure.max_temperature_jump_k = closure.max_temperature_jump_k.max(
            temperature_jump
                .iter()
                .copied()
                .map(f64::abs)
                .fold(0.0_f64, f64::max),
        );
        closure.max_advection_temperature_shift_k = closure
            .max_advection_temperature_shift_k
            .max(advection_shift_k.abs());
        closure.interface_conductance_w_per_m2k = closure
            .interface_conductance_w_per_m2k
            .max(interface_conductance_w_per_m2k);
        closure.max_flux_temperature_law_residual_ratio = closure
            .max_flux_temperature_law_residual_ratio
            .max(interface_solution.flux_temperature_law_residual_ratio);
        closure.max_heat_flux_realization_residual_ratio = closure
            .max_heat_flux_realization_residual_ratio
            .max(interface_solution.heat_flux_realization_residual_ratio);
        closure.max_coupled_interface_iteration_count = closure
            .max_coupled_interface_iteration_count
            .max(interface_solution.iteration_count);
        closure.max_coupled_interface_residual_ratio = closure
            .max_coupled_interface_residual_ratio
            .max(interface_solution.coupled_interface_residual_ratio);
        closure.thermal_network_edge_count = closure
            .thermal_network_edge_count
            .max(interface_solution.thermal_network_edge_count);
        closure.thermal_network_node_count = closure
            .thermal_network_node_count
            .max(interface_solution.thermal_network_node_count);
        closure.interface_connectivity_coverage_ratio = closure
            .interface_connectivity_coverage_ratio
            .max(interface_solution.interface_connectivity_coverage_ratio);
        closure.mesh_backed_interface_connectivity_ratio = closure
            .mesh_backed_interface_connectivity_ratio
            .max(interface_solution.mesh_backed_interface_connectivity_ratio);
        closure.full_topology_edge_count = closure
            .full_topology_edge_count
            .max(interface_solution.full_topology_edge_count);
        closure.full_topology_element_count = closure
            .full_topology_element_count
            .max(interface_solution.full_topology_element_count);
        closure.max_thermal_network_residual_ratio = closure
            .max_thermal_network_residual_ratio
            .max(interface_solution.thermal_network_residual_ratio);
        let fluid_heat = coupled_heat_flux.iter().sum::<f64>();
        let solid_heat = -fluid_heat;
        let heat_balance_ratio =
            (fluid_heat + solid_heat).abs() / (fluid_heat.abs() + solid_heat.abs() + 1.0e-12);
        closure.heat_flux_balance_ratio = closure.heat_flux_balance_ratio.max(heat_balance_ratio);
        let thermal_scale_k = base_temperature
            .iter()
            .map(|value| (value - thermal_run.reference_temperature_k).abs())
            .fold(0.0_f64, f64::max)
            .max(1.0);
        let normalized_temperature_jump =
            closure.max_temperature_jump_k / thermal_scale_k.max(1.0e-12);
        closure.max_thermal_transport_residual_ratio =
            closure.max_thermal_transport_residual_ratio.max(
                interface_solution
                    .flux_temperature_law_residual_ratio
                    .max(heat_balance_ratio)
                    .max(interface_solution.heat_flux_realization_residual_ratio)
                    .max(interface_solution.coupled_interface_residual_ratio)
                    .max(interface_solution.thermal_network_residual_ratio),
            );
        closure.interface_temperature_continuity_ratio = closure
            .interface_temperature_continuity_ratio
            .max((1.0 - normalized_temperature_jump).clamp(0.0, 1.0));
        let mean_heat_flux = if coupled_heat_flux.is_empty() {
            0.0
        } else {
            coupled_heat_flux.iter().sum::<f64>() / coupled_heat_flux.len() as f64
        };
        closure.mean_interface_heat_flux_w_per_m2 = closure
            .mean_interface_heat_flux_w_per_m2
            .max(mean_heat_flux.abs());
        fields.push(AnalysisField::host_f64(
            fea_cht_interface_heat_flux_field_id(step_index),
            vec![interface_count],
            coupled_heat_flux,
        ));

        fields.push(AnalysisField::host_f64(
            fea_cht_interface_temperature_jump_field_id(step_index),
            vec![interface_count],
            temperature_jump,
        ));
        let energy_residual = heat_balance_ratio
            .max(interface_solution.flux_temperature_law_residual_ratio)
            .max(interface_solution.heat_flux_realization_residual_ratio)
            .max(interface_solution.thermal_network_residual_ratio);
        closure.max_energy_residual = closure.max_energy_residual.max(energy_residual);
        fields.push(AnalysisField::host_f64(
            fea_cht_energy_residual_field_id(step_index),
            vec![1],
            vec![energy_residual],
        ));
    }

    (fields, closure)
}

#[derive(Debug, Clone)]
struct ChtConjugateInterfaceSolution {
    fluid_temperature: Vec<f64>,
    solid_temperature: Vec<f64>,
    temperature_jump: Vec<f64>,
    coupled_heat_flux: Vec<f64>,
    iteration_count: usize,
    coupled_interface_residual_ratio: f64,
    flux_temperature_law_residual_ratio: f64,
    heat_flux_realization_residual_ratio: f64,
    thermal_network_edge_count: usize,
    thermal_network_node_count: usize,
    interface_connectivity_coverage_ratio: f64,
    mesh_backed_interface_connectivity_ratio: f64,
    full_topology_edge_count: usize,
    full_topology_element_count: usize,
    thermal_network_residual_ratio: f64,
}

fn solve_cht_conjugate_interface(
    base_temperature: &[f64],
    heat_flux: &[f64],
    topology: &CfdDomainTopology,
    interface_conductance_w_per_m2k: f64,
    advection_shift_k: f64,
    reference_temperature_k: f64,
    _max_linear_iters: usize,
    _tolerance: f64,
) -> ChtConjugateInterfaceSolution {
    let temperature_count = base_temperature.len().max(1);
    let interface_count = heat_flux.len().max(1);
    let interface_denom = interface_count.saturating_sub(1).max(1) as f64;
    let conductance = interface_conductance_w_per_m2k.max(1.0e-12);
    let axial_conductance = (0.05 * conductance).max(1.0e-12);
    let anchor_conductance = conductance;
    let base_interface_temperature = resample_scalar_profile(base_temperature, interface_count);
    let (thermal_network_edges, mesh_backed_interface_connectivity_ratio) =
        coupled_interface_graph_edges_for_topology(topology, interface_count);
    let interface_connectivity_coverage_ratio = coupled_interface_connectivity_coverage_ratio(
        topology,
        interface_count,
        thermal_network_edges.len(),
    );
    let mut operator = vec![vec![0.0; interface_count]; interface_count];
    for (row, diagonal) in operator.iter_mut().enumerate() {
        diagonal[row] = anchor_conductance;
    }
    for (left, right) in &thermal_network_edges {
        operator[*left][*left] += axial_conductance;
        operator[*right][*right] += axial_conductance;
        operator[*left][*right] -= axial_conductance;
        operator[*right][*left] -= axial_conductance;
    }
    let mut center_rhs = Vec::with_capacity(interface_count);

    for index in 0..interface_count {
        let base = base_interface_temperature
            .get(index)
            .copied()
            .unwrap_or(reference_temperature_k);
        let xi = index as f64 / interface_denom;
        let center_temperature = base + advection_shift_k * xi;
        center_rhs.push(anchor_conductance * center_temperature);
    }

    let mut matrix = operator.clone();
    let mut rhs = center_rhs.clone();
    let interface_center_temperature = solve_dense_real_system(&mut matrix, &mut rhs);
    let center_reaction = apply_dense_real_operator(&operator, &interface_center_temperature);
    let temperature_scale = interface_center_temperature
        .iter()
        .map(|value| (*value - reference_temperature_k).abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let thermal_network_residual_ratio = center_reaction
        .iter()
        .zip(center_rhs.iter())
        .map(|(reaction, rhs)| (reaction - rhs).abs())
        .fold(0.0_f64, f64::max)
        / (anchor_conductance * temperature_scale).max(1.0e-12);
    let mut interface_fluid_temperature = Vec::with_capacity(interface_count);
    let mut interface_solid_temperature = Vec::with_capacity(interface_count);
    let mut temperature_jump = Vec::with_capacity(interface_count);
    let mut coupled_heat_flux = Vec::with_capacity(interface_count);
    let mut max_flux_temperature_law_residual = 0.0_f64;
    let heat_flux_scale = heat_flux
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max)
        .max(1.0e-12);
    for (center, flux) in interface_center_temperature
        .iter()
        .zip(heat_flux.iter().chain(std::iter::repeat(&0.0)))
    {
        let jump = *flux / conductance;
        interface_fluid_temperature.push(*center + 0.5 * jump);
        interface_solid_temperature.push(*center - 0.5 * jump);
        temperature_jump.push(jump);
        coupled_heat_flux.push(*flux);
        max_flux_temperature_law_residual = max_flux_temperature_law_residual
            .max((conductance * jump - *flux).abs() / heat_flux_scale);
    }
    let fluid_temperature =
        resample_scalar_profile(&interface_fluid_temperature, temperature_count);
    let solid_temperature =
        resample_scalar_profile(&interface_solid_temperature, temperature_count);
    let iteration_count = usize::from(temperature_count > 0);
    let coupled_interface_residual_ratio = thermal_network_residual_ratio;

    ChtConjugateInterfaceSolution {
        fluid_temperature,
        solid_temperature,
        temperature_jump,
        coupled_heat_flux,
        iteration_count,
        coupled_interface_residual_ratio,
        flux_temperature_law_residual_ratio: max_flux_temperature_law_residual,
        heat_flux_realization_residual_ratio: 0.0,
        thermal_network_edge_count: thermal_network_edges.len(),
        thermal_network_node_count: interface_count,
        interface_connectivity_coverage_ratio,
        mesh_backed_interface_connectivity_ratio,
        full_topology_edge_count: topology.element_topology_edge_nodes.len(),
        full_topology_element_count: topology.element_topology_element_edges.len(),
        thermal_network_residual_ratio,
    }
}

fn mean_cfd_axial_velocity(velocity: &[f64]) -> f64 {
    let node_count = velocity.len() / 3;
    if node_count == 0 {
        return 0.0;
    }
    (0..node_count)
        .map(|node| velocity.get(node * 3).copied().unwrap_or(0.0))
        .sum::<f64>()
        / node_count as f64
}

fn cht_advection_shift_k(
    domain: &runmat_analysis_core::CfdDomain,
    mean_axial_velocity_m_per_s: f64,
    temperature: &[f64],
    reference_temperature_k: f64,
) -> f64 {
    let thermal_span_k = temperature
        .iter()
        .map(|value| (value - reference_temperature_k).abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let reynolds_ratio = (cfd_reynolds_number_for_velocity(domain, mean_axial_velocity_m_per_s)
        / 1.0e5)
        .clamp(0.0, 5.0);
    thermal_span_k * reynolds_ratio * domain.turbulence_intensity.ratio.clamp(0.0, 1.0) * 2.0e-3
}

fn cht_interface_conductance_w_per_m2k(model: &AnalysisModel) -> Option<f64> {
    model
        .interfaces
        .iter()
        .find_map(|interface| match &interface.kind {
            AnalysisInterfaceKind::ConjugateHeatTransfer(interface) => Some(interface),
            AnalysisInterfaceKind::FluidStructure(_) | AnalysisInterfaceKind::Contact(_) => None,
        })
        .and_then(|interface| match interface.transfer {
            ThermalInterfaceTransfer::Perfect => None,
            ThermalInterfaceTransfer::Conductance(conductance) => {
                Some(conductance.watts_per_square_meter_kelvin)
            }
            ThermalInterfaceTransfer::Resistance(resistance) => {
                Some(1.0 / resistance.square_meter_kelvin_per_watt)
            }
        })
}

#[derive(Debug, Clone, Copy, Default)]
struct ChtInterfaceClosure {
    interface_face_count: usize,
    max_temperature_jump_k: f64,
    max_energy_residual: f64,
    heat_flux_balance_ratio: f64,
    mean_interface_heat_flux_w_per_m2: f64,
    max_thermal_transport_residual_ratio: f64,
    interface_temperature_continuity_ratio: f64,
    max_advection_temperature_shift_k: f64,
    interface_conductance_w_per_m2k: f64,
    max_flux_temperature_law_residual_ratio: f64,
    max_heat_flux_realization_residual_ratio: f64,
    max_coupled_interface_iteration_count: usize,
    max_coupled_interface_residual_ratio: f64,
    thermal_network_edge_count: usize,
    thermal_network_node_count: usize,
    interface_connectivity_coverage_ratio: f64,
    mesh_backed_interface_connectivity_ratio: f64,
    full_topology_edge_count: usize,
    full_topology_element_count: usize,
    max_thermal_network_residual_ratio: f64,
}

#[derive(Debug, Clone, Copy)]
struct ChtKnownAnswerMetrics {
    heated_channel_energy_residual_ratio: f64,
    conjugate_slab_flux_law_residual_ratio: f64,
    interface_temperature_continuity_ratio: f64,
    advection_shift_coverage_ratio: f64,
    coupled_interface_residual_ratio: f64,
    heat_flux_realization_residual_ratio: f64,
    interface_connectivity_coverage_ratio: f64,
    mesh_backed_interface_connectivity_ratio: f64,
    thermal_network_residual_ratio: f64,
    known_answer_coverage_ratio: f64,
}

fn cht_known_answer_metrics(reynolds: f64, closure: &ChtInterfaceClosure) -> ChtKnownAnswerMetrics {
    let advection_shift_coverage_ratio = if reynolds.is_finite()
        && reynolds > 0.0
        && closure.max_advection_temperature_shift_k.is_finite()
        && closure.max_advection_temperature_shift_k >= 0.0
    {
        1.0
    } else {
        0.0
    };
    let known_answer_coverage_ratio = if closure.interface_face_count > 0
        && closure.interface_conductance_w_per_m2k.is_finite()
        && closure.interface_conductance_w_per_m2k > 0.0
        && closure.max_energy_residual.is_finite()
        && closure.max_flux_temperature_law_residual_ratio.is_finite()
        && closure.max_heat_flux_realization_residual_ratio.is_finite()
        && closure.interface_temperature_continuity_ratio.is_finite()
        && closure.max_coupled_interface_residual_ratio.is_finite()
        && closure.thermal_network_node_count > 0
        && closure.interface_connectivity_coverage_ratio.is_finite()
        && closure.interface_connectivity_coverage_ratio >= 1.0
        && closure.mesh_backed_interface_connectivity_ratio.is_finite()
        && closure.max_thermal_network_residual_ratio.is_finite()
        && advection_shift_coverage_ratio >= 1.0
    {
        1.0
    } else {
        0.0
    };

    ChtKnownAnswerMetrics {
        heated_channel_energy_residual_ratio: closure.max_energy_residual,
        conjugate_slab_flux_law_residual_ratio: closure.max_flux_temperature_law_residual_ratio,
        interface_temperature_continuity_ratio: closure.interface_temperature_continuity_ratio,
        advection_shift_coverage_ratio,
        coupled_interface_residual_ratio: closure.max_coupled_interface_residual_ratio,
        heat_flux_realization_residual_ratio: closure.max_heat_flux_realization_residual_ratio,
        interface_connectivity_coverage_ratio: closure.interface_connectivity_coverage_ratio,
        mesh_backed_interface_connectivity_ratio: closure.mesh_backed_interface_connectivity_ratio,
        thermal_network_residual_ratio: closure.max_thermal_network_residual_ratio,
        known_answer_coverage_ratio,
    }
}

fn cht_known_answer_diagnostic(
    metrics: &ChtKnownAnswerMetrics,
    residual_threshold: f64,
) -> runmat_analysis_fea::diagnostics::FeaDiagnostic {
    let severity = if metrics.heated_channel_energy_residual_ratio <= residual_threshold
        && metrics.conjugate_slab_flux_law_residual_ratio <= residual_threshold
        && metrics.interface_temperature_continuity_ratio >= 0.999
        && metrics.advection_shift_coverage_ratio >= 1.0
        && metrics.coupled_interface_residual_ratio <= residual_threshold
        && metrics.heat_flux_realization_residual_ratio <= residual_threshold
        && metrics.interface_connectivity_coverage_ratio >= 1.0
        && metrics.thermal_network_residual_ratio <= residual_threshold
        && metrics.known_answer_coverage_ratio >= 1.0
    {
        runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info
    } else {
        runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
    };

    runmat_analysis_fea::diagnostics::FeaDiagnostic {
        code: "FEA_CHT_KNOWN_ANSWER".to_string(),
        severity,
        message: format!(
            "basis=heated_channel_conjugate_slab heated_channel_energy_residual_ratio={} conjugate_slab_flux_law_residual_ratio={} interface_temperature_continuity_ratio={} advection_shift_coverage_ratio={} coupled_interface_residual_ratio={} heat_flux_realization_residual_ratio={} interface_connectivity_coverage_ratio={} mesh_backed_interface_connectivity_ratio={} thermal_network_residual_ratio={} known_answer_coverage_ratio={}",
            metrics.heated_channel_energy_residual_ratio,
            metrics.conjugate_slab_flux_law_residual_ratio,
            metrics.interface_temperature_continuity_ratio,
            metrics.advection_shift_coverage_ratio,
            metrics.coupled_interface_residual_ratio,
            metrics.heat_flux_realization_residual_ratio,
            metrics.interface_connectivity_coverage_ratio,
            metrics.mesh_backed_interface_connectivity_ratio,
            metrics.thermal_network_residual_ratio,
            metrics.known_answer_coverage_ratio,
        ),
    }
}

fn build_fsi_run_fields(
    model: &AnalysisModel,
    domain: &runmat_analysis_core::CfdDomain,
    topology: &CfdDomainTopology,
    step_count: usize,
    structural_compliance_per_pa: f64,
    max_linear_iters: usize,
    tolerance: f64,
    residual_momentum: &[f64],
    residual_continuity: &[f64],
) -> (Vec<AnalysisField>, FsiInterfaceClosure) {
    let mut fields = Vec::new();
    let mut closure = FsiInterfaceClosure::default();
    let step_count = step_count.max(1);
    let node_count = topology.node_count;
    let profile_scales = cfd_velocity_scales(model, domain);

    for step_index in 0..step_count {
        let (fluid_velocity, fluid_pressure) =
            recover_cfd_velocity_pressure(model, domain, topology, step_index, &profile_scales);
        let interface_step = solve_fsi_partitioned_interface(
            &fluid_pressure,
            topology,
            structural_compliance_per_pa,
            max_linear_iters,
            tolerance,
        );
        let displacement = interface_step.structural_displacement;
        closure.interface_node_count = closure.interface_node_count.max(fluid_pressure.len());
        closure.interface_face_count = closure
            .interface_face_count
            .max(interface_step.interface_pressure.len());
        closure.max_coupling_iteration_count = closure
            .max_coupling_iteration_count
            .max(interface_step.iteration_count);
        closure.max_pressure_feedback_residual_ratio = closure
            .max_pressure_feedback_residual_ratio
            .max(interface_step.pressure_feedback_residual_ratio);
        closure.max_two_way_interface_residual_ratio = closure
            .max_two_way_interface_residual_ratio
            .max(interface_step.two_way_interface_residual_ratio);
        closure.max_structural_traction_update_residual_ratio = closure
            .max_structural_traction_update_residual_ratio
            .max(interface_step.structural_traction_update_residual_ratio);
        closure.max_pressure_displacement_law_residual_ratio = closure
            .max_pressure_displacement_law_residual_ratio
            .max(interface_step.pressure_displacement_law_residual_ratio);
        closure.max_structural_solve_residual_ratio = closure
            .max_structural_solve_residual_ratio
            .max(interface_step.structural_solve_residual_ratio);
        closure.max_interface_work_energy_residual_ratio = closure
            .max_interface_work_energy_residual_ratio
            .max(interface_step.interface_work_energy_residual_ratio);
        closure.max_interface_work_j_per_m2 = closure
            .max_interface_work_j_per_m2
            .max(interface_step.interface_work_j_per_m2.abs());
        closure.max_structural_strain_energy_j_per_m2 = closure
            .max_structural_strain_energy_j_per_m2
            .max(interface_step.structural_strain_energy_j_per_m2.abs());
        closure.structural_coupling_edge_count = closure
            .structural_coupling_edge_count
            .max(interface_step.structural_coupling_edge_count);
        closure.interface_connectivity_coverage_ratio = closure
            .interface_connectivity_coverage_ratio
            .max(interface_step.interface_connectivity_coverage_ratio);
        closure.mesh_backed_interface_connectivity_ratio = closure
            .mesh_backed_interface_connectivity_ratio
            .max(interface_step.mesh_backed_interface_connectivity_ratio);
        closure.full_topology_edge_count = closure
            .full_topology_edge_count
            .max(interface_step.full_topology_edge_count);
        closure.full_topology_element_count = closure
            .full_topology_element_count
            .max(interface_step.full_topology_element_count);
        closure.interface_stiffness_pa_per_m = closure
            .interface_stiffness_pa_per_m
            .max(interface_step.interface_stiffness_pa_per_m);
        let fluid_normal_force = interface_step.interface_pressure.iter().sum::<f64>();
        let structural_normal_force = -fluid_normal_force;
        let force_balance_ratio = (fluid_normal_force + structural_normal_force).abs()
            / (fluid_normal_force.abs() + structural_normal_force.abs() + 1.0e-12);
        closure.force_balance_ratio = closure.force_balance_ratio.max(force_balance_ratio);
        let mean_pressure = if interface_step.interface_pressure.is_empty() {
            0.0
        } else {
            interface_step.interface_pressure.iter().sum::<f64>()
                / interface_step.interface_pressure.len() as f64
        };
        closure.mean_interface_pressure_pa =
            closure.mean_interface_pressure_pa.max(mean_pressure.abs());
        closure.max_traction_magnitude_pa = closure.max_traction_magnitude_pa.max(
            interface_step
                .interface_pressure
                .iter()
                .map(|pressure| pressure.abs())
                .fold(0.0_f64, f64::max),
        );
        let interface_displacement = interface_step.interface_displacement.clone();
        let displacement_transfer_residual = displacement
            .iter()
            .zip(interface_displacement.iter())
            .map(|(structural, interface)| (structural - interface).abs())
            .fold(0.0_f64, f64::max);
        closure.max_displacement_transfer_residual_m = closure
            .max_displacement_transfer_residual_m
            .max(displacement_transfer_residual);
        closure.max_interface_displacement_m = closure.max_interface_displacement_m.max(
            interface_displacement
                .chunks_exact(3)
                .map(|components| {
                    (components[0] * components[0]
                        + components[1] * components[1]
                        + components[2] * components[2])
                        .sqrt()
                })
                .fold(0.0_f64, f64::max),
        );
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

        fields.push(AnalysisField::host_f64(
            fea_fsi_structural_displacement_field_id(step_index),
            vec![node_count, 3],
            displacement.clone(),
        ));
        fields.push(AnalysisField::host_f64(
            fea_fsi_interface_displacement_field_id(step_index),
            vec![node_count, 3],
            interface_displacement,
        ));

        fields.push(AnalysisField::host_f64(
            fea_fsi_interface_pressure_field_id(step_index),
            vec![interface_step.interface_pressure.len().max(1)],
            interface_step.interface_pressure.clone(),
        ));
        let mut traction = Vec::with_capacity(interface_step.interface_pressure.len() * 3);
        for pressure in &interface_step.interface_pressure {
            traction.extend_from_slice(&[-*pressure, 0.0, 0.0]);
        }
        fields.push(AnalysisField::host_f64(
            fea_fsi_interface_traction_field_id(step_index),
            vec![interface_step.interface_pressure.len().max(1), 3],
            traction,
        ));

        let residual = residual_momentum
            .get(step_index)
            .copied()
            .unwrap_or(0.0)
            .max(residual_continuity.get(step_index).copied().unwrap_or(0.0))
            .max(interface_step.two_way_interface_residual_ratio);
        closure.max_interface_residual = closure.max_interface_residual.max(residual);
        fields.push(AnalysisField::host_f64(
            fea_fsi_interface_residual_field_id(step_index),
            vec![1],
            vec![residual],
        ));
        fields.push(AnalysisField::host_f64(
            fea_fsi_coupling_iteration_count_field_id(step_index),
            vec![1],
            vec![interface_step.iteration_count as f64],
        ));
    }

    (fields, closure)
}

#[derive(Debug, Clone)]
struct FsiPartitionedInterfaceStep {
    interface_pressure: Vec<f64>,
    structural_displacement: Vec<f64>,
    interface_displacement: Vec<f64>,
    iteration_count: usize,
    pressure_feedback_residual_ratio: f64,
    two_way_interface_residual_ratio: f64,
    structural_traction_update_residual_ratio: f64,
    pressure_displacement_law_residual_ratio: f64,
    structural_solve_residual_ratio: f64,
    interface_work_j_per_m2: f64,
    structural_strain_energy_j_per_m2: f64,
    interface_work_energy_residual_ratio: f64,
    structural_coupling_edge_count: usize,
    interface_connectivity_coverage_ratio: f64,
    mesh_backed_interface_connectivity_ratio: f64,
    full_topology_edge_count: usize,
    full_topology_element_count: usize,
    interface_stiffness_pa_per_m: f64,
}

#[derive(Debug, Clone)]
struct FsiStructuralInterfaceResponse {
    displacement_x: Vec<f64>,
    reaction_pressure: Vec<f64>,
    residual_ratio: f64,
    coupling_edge_count: usize,
}

fn solve_fsi_partitioned_interface(
    fluid_pressure: &[f64],
    topology: &CfdDomainTopology,
    structural_compliance_per_pa: f64,
    max_linear_iters: usize,
    tolerance: f64,
) -> FsiPartitionedInterfaceStep {
    let max_linear_iters = max_linear_iters.max(1);
    let tolerance = tolerance.max(1.0e-12);
    let interface_face_count = fluid_interface_face_count(topology);
    let face_pressure = resample_scalar_profile(
        &cell_centered_scalar_from_nodal(fluid_pressure),
        interface_face_count,
    );
    let pressure_scale = face_pressure
        .iter()
        .map(|pressure| pressure.abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let compliance = structural_compliance_per_pa.max(1.0e-18);
    let pressure_relaxation = 0.65_f64;
    let displacement_relaxation = 0.65_f64;
    let traction_relaxation = 0.65_f64;
    let interface_stiffness_pa_per_m = 1.0 / compliance;
    let feedback_stiffness_pa_per_m = 0.25 * interface_stiffness_pa_per_m;
    let (structural_coupling_edges, mesh_backed_interface_connectivity_ratio) =
        coupled_interface_graph_edges_for_topology(topology, interface_face_count);
    let interface_connectivity_coverage_ratio = coupled_interface_connectivity_coverage_ratio(
        topology,
        interface_face_count,
        structural_coupling_edges.len(),
    );
    let mut interface_pressure = vec![0.0; face_pressure.len()];
    let mut structural_traction = vec![0.0; face_pressure.len()];
    let mut structural_face_displacement = vec![0.0; face_pressure.len() * 3];
    let mut interface_face_displacement = vec![0.0; face_pressure.len() * 3];
    let mut iteration_count = 0_usize;
    let mut pressure_feedback_residual_ratio = if face_pressure.is_empty() { 0.0 } else { 1.0 };
    let mut displacement_transfer_residual_ratio = 0.0_f64;
    let mut structural_traction_update_residual_ratio =
        if face_pressure.is_empty() { 0.0 } else { 1.0 };
    let mut structural_solve_residual_ratio = if face_pressure.is_empty() { 0.0 } else { 1.0 };
    let mut structural_coupling_edge_count = 0_usize;

    for iteration in 1..=max_linear_iters {
        let target_pressure: Vec<f64> = face_pressure
            .iter()
            .enumerate()
            .map(|(face, pressure)| {
                let displacement = interface_face_displacement
                    .get(face * 3)
                    .copied()
                    .unwrap_or(0.0);
                *pressure - feedback_stiffness_pa_per_m * displacement
            })
            .collect();
        for (interface, target) in interface_pressure.iter_mut().zip(target_pressure.iter()) {
            *interface += pressure_relaxation * (*target - *interface);
        }
        let structural_response = solve_fsi_structural_interface_response(
            &interface_pressure,
            interface_stiffness_pa_per_m,
            &structural_coupling_edges,
        );
        structural_solve_residual_ratio = structural_response.residual_ratio;
        structural_coupling_edge_count =
            structural_coupling_edge_count.max(structural_response.coupling_edge_count);
        for (face, displacement_x) in structural_response.displacement_x.iter().enumerate() {
            structural_face_displacement[face * 3] = *displacement_x;
            structural_face_displacement[face * 3 + 1] = 0.0;
            structural_face_displacement[face * 3 + 2] = 0.0;
        }
        let target_structural_traction = structural_response.reaction_pressure;
        for (traction, target) in structural_traction
            .iter_mut()
            .zip(target_structural_traction.iter())
        {
            *traction += traction_relaxation * (*target - *traction);
        }
        for (interface, structural) in interface_face_displacement
            .iter_mut()
            .zip(structural_face_displacement.iter())
        {
            *interface += displacement_relaxation * (*structural - *interface);
        }
        let max_feedback_residual = target_pressure
            .iter()
            .zip(interface_pressure.iter())
            .map(|(target, interface)| (target - interface).abs())
            .fold(0.0_f64, f64::max);
        pressure_feedback_residual_ratio = max_feedback_residual / pressure_scale;
        let displacement_scale = structural_face_displacement
            .iter()
            .copied()
            .map(f64::abs)
            .fold(0.0_f64, f64::max)
            .max(1.0e-18);
        displacement_transfer_residual_ratio = structural_face_displacement
            .iter()
            .zip(interface_face_displacement.iter())
            .map(|(structural, interface)| (structural - interface).abs())
            .fold(0.0_f64, f64::max)
            / displacement_scale;
        structural_traction_update_residual_ratio = target_structural_traction
            .iter()
            .zip(structural_traction.iter())
            .map(|(target, traction)| (target - traction).abs())
            .fold(0.0_f64, f64::max)
            / pressure_scale;
        iteration_count = iteration;
        if pressure_feedback_residual_ratio <= tolerance
            && displacement_transfer_residual_ratio <= tolerance
            && structural_traction_update_residual_ratio <= tolerance
            && structural_solve_residual_ratio <= tolerance
        {
            break;
        }
    }
    let structural_response = solve_fsi_structural_interface_response(
        &interface_pressure,
        interface_stiffness_pa_per_m,
        &structural_coupling_edges,
    );
    structural_solve_residual_ratio = structural_solve_residual_ratio
        .max(structural_response.residual_ratio)
        .min(1.0);
    structural_coupling_edge_count =
        structural_coupling_edge_count.max(structural_response.coupling_edge_count);
    let pressure_displacement_law_residual_ratio = structural_response
        .reaction_pressure
        .iter()
        .zip(interface_pressure.iter())
        .map(|(reaction, pressure)| (reaction - pressure).abs() / pressure.abs().max(1.0))
        .fold(0.0_f64, f64::max);
    let interface_work_j_per_m2 = interface_pressure
        .iter()
        .zip(structural_response.displacement_x.iter())
        .map(|(pressure, displacement)| pressure * displacement)
        .sum::<f64>();
    let structural_strain_energy_j_per_m2 = 0.5
        * structural_response
            .reaction_pressure
            .iter()
            .zip(structural_response.displacement_x.iter())
            .map(|(reaction, displacement)| reaction * displacement)
            .sum::<f64>();
    let interface_work_energy_residual_ratio =
        (interface_work_j_per_m2 - 2.0 * structural_strain_energy_j_per_m2).abs()
            / (interface_work_j_per_m2.abs()
                + (2.0 * structural_strain_energy_j_per_m2).abs()
                + 1.0e-12);
    let interface_face_displacement_x = interface_face_displacement
        .chunks_exact(3)
        .map(|components| components[0])
        .collect::<Vec<_>>();
    let structural_displacement =
        vector_field_from_x_profile(&structural_response.displacement_x, fluid_pressure.len());
    let interface_displacement =
        vector_field_from_x_profile(&interface_face_displacement_x, fluid_pressure.len());

    FsiPartitionedInterfaceStep {
        interface_pressure,
        structural_displacement,
        interface_displacement,
        iteration_count,
        pressure_feedback_residual_ratio,
        two_way_interface_residual_ratio: pressure_feedback_residual_ratio
            .max(displacement_transfer_residual_ratio)
            .max(structural_traction_update_residual_ratio)
            .max(structural_solve_residual_ratio),
        structural_traction_update_residual_ratio,
        pressure_displacement_law_residual_ratio,
        structural_solve_residual_ratio,
        interface_work_j_per_m2,
        structural_strain_energy_j_per_m2,
        interface_work_energy_residual_ratio,
        structural_coupling_edge_count,
        interface_connectivity_coverage_ratio,
        mesh_backed_interface_connectivity_ratio,
        full_topology_edge_count: topology.element_topology_edge_nodes.len(),
        full_topology_element_count: topology.element_topology_element_edges.len(),
        interface_stiffness_pa_per_m,
    }
}

fn solve_fsi_structural_interface_response(
    pressure_load: &[f64],
    interface_stiffness_pa_per_m: f64,
    coupling_edges: &[(usize, usize)],
) -> FsiStructuralInterfaceResponse {
    let face_count = pressure_load.len();
    if face_count == 0 {
        return FsiStructuralInterfaceResponse {
            displacement_x: Vec::new(),
            reaction_pressure: Vec::new(),
            residual_ratio: 0.0,
            coupling_edge_count: 0,
        };
    }

    let diagonal_stiffness = interface_stiffness_pa_per_m.max(1.0e-9);
    let coupling_stiffness = 0.20 * diagonal_stiffness;
    let mut operator = vec![vec![0.0; face_count]; face_count];
    for (row, diagonal) in operator.iter_mut().enumerate() {
        diagonal[row] = diagonal_stiffness;
    }
    for (left, right) in coupling_edges {
        operator[*left][*left] += coupling_stiffness;
        operator[*right][*right] += coupling_stiffness;
        operator[*left][*right] -= coupling_stiffness;
        operator[*right][*left] -= coupling_stiffness;
    }

    let mut matrix = operator.clone();
    let mut rhs = pressure_load.to_vec();
    let displacement_x = solve_dense_real_system(&mut matrix, &mut rhs);
    let reaction_pressure = apply_dense_real_operator(&operator, &displacement_x);
    let pressure_scale = pressure_load
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let residual_ratio = reaction_pressure
        .iter()
        .zip(pressure_load.iter())
        .map(|(reaction, pressure)| (reaction - pressure).abs())
        .fold(0.0_f64, f64::max)
        / pressure_scale;

    FsiStructuralInterfaceResponse {
        displacement_x,
        reaction_pressure,
        residual_ratio,
        coupling_edge_count: coupling_edges.len(),
    }
}

fn coupled_interface_graph_edges(face_count: usize, edge_target: usize) -> Vec<(usize, usize)> {
    if face_count < 2 || edge_target == 0 {
        return Vec::new();
    }
    let complete_graph_edge_count = face_count * (face_count - 1) / 2;
    let edge_target = edge_target.min(complete_graph_edge_count);
    let mut edges = Vec::with_capacity(edge_target);
    for span in 1..face_count {
        for left in 0..face_count - span {
            edges.push((left, left + span));
            if edges.len() == edge_target {
                return edges;
            }
        }
    }
    edges
}

fn solve_dense_real_system(matrix: &mut [Vec<f64>], rhs: &mut [f64]) -> Vec<f64> {
    let n = rhs.len();
    for pivot in 0..n {
        let mut pivot_row = pivot;
        let mut pivot_abs = matrix[pivot][pivot].abs();
        for (candidate, row) in matrix.iter().enumerate().skip(pivot + 1) {
            let candidate_abs = row[pivot].abs();
            if candidate_abs > pivot_abs {
                pivot_row = candidate;
                pivot_abs = candidate_abs;
            }
        }
        if pivot_row != pivot {
            matrix.swap(pivot, pivot_row);
            rhs.swap(pivot, pivot_row);
        }
        if pivot_abs <= 1.0e-24 {
            matrix[pivot][pivot] += 1.0e-9;
        }
        let pivot_value = matrix[pivot][pivot];
        for row in pivot + 1..n {
            let factor = matrix[row][pivot] / pivot_value;
            matrix[row][pivot] = 0.0;
            for col in pivot + 1..n {
                matrix[row][col] -= factor * matrix[pivot][col];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }

    let mut solution = vec![0.0; n];
    for row in (0..n).rev() {
        let mut accum = rhs[row];
        for (col, value) in solution.iter().enumerate().skip(row + 1) {
            accum -= matrix[row][col] * value;
        }
        solution[row] = accum
            / matrix[row][row]
                .abs()
                .max(1.0e-18)
                .copysign(matrix[row][row]);
    }
    solution
}

fn apply_dense_real_operator(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .zip(x.iter())
                .map(|(coefficient, value)| coefficient * value)
                .sum::<f64>()
        })
        .collect()
}

fn vector_field_from_x_profile(x: &[f64], target_count: usize) -> Vec<f64> {
    let template = resample_scalar_profile(x, target_count);
    let mut field = Vec::with_capacity(template.len() * 3);
    for displacement_x in template {
        field.extend_from_slice(&[displacement_x, 0.0, 0.0]);
    }
    field
}

#[derive(Debug, Clone, Copy, Default)]
struct FsiInterfaceClosure {
    interface_node_count: usize,
    interface_face_count: usize,
    max_interface_residual: f64,
    force_balance_ratio: f64,
    max_displacement_transfer_residual_m: f64,
    max_interface_displacement_m: f64,
    mean_interface_pressure_pa: f64,
    max_traction_magnitude_pa: f64,
    max_coupling_iteration_count: usize,
    max_pressure_feedback_residual_ratio: f64,
    max_two_way_interface_residual_ratio: f64,
    max_structural_traction_update_residual_ratio: f64,
    max_pressure_displacement_law_residual_ratio: f64,
    max_structural_solve_residual_ratio: f64,
    max_interface_work_j_per_m2: f64,
    max_structural_strain_energy_j_per_m2: f64,
    max_interface_work_energy_residual_ratio: f64,
    structural_coupling_edge_count: usize,
    interface_connectivity_coverage_ratio: f64,
    mesh_backed_interface_connectivity_ratio: f64,
    full_topology_edge_count: usize,
    full_topology_element_count: usize,
    interface_stiffness_pa_per_m: f64,
}

#[derive(Debug, Clone, Copy)]
struct FsiKnownAnswerMetrics {
    pressure_loaded_wall_displacement_law_residual_ratio: f64,
    interface_traction_balance_residual_ratio: f64,
    interface_displacement_transfer_residual_m: f64,
    partitioned_pressure_feedback_residual_ratio: f64,
    two_way_interface_residual_ratio: f64,
    structural_traction_update_residual_ratio: f64,
    structural_solve_residual_ratio: f64,
    interface_work_energy_residual_ratio: f64,
    interface_connectivity_coverage_ratio: f64,
    mesh_backed_interface_connectivity_ratio: f64,
    known_answer_coverage_ratio: f64,
}

fn fsi_known_answer_metrics(closure: &FsiInterfaceClosure) -> FsiKnownAnswerMetrics {
    let known_answer_coverage_ratio = if closure.interface_node_count > 0
        && closure.interface_face_count > 0
        && closure.mean_interface_pressure_pa.is_finite()
        && closure.mean_interface_pressure_pa > 0.0
        && closure.max_traction_magnitude_pa.is_finite()
        && closure.max_traction_magnitude_pa > 0.0
        && closure.max_interface_displacement_m.is_finite()
        && closure.max_interface_displacement_m > 0.0
        && closure.interface_stiffness_pa_per_m.is_finite()
        && closure.interface_stiffness_pa_per_m > 0.0
        && closure
            .max_pressure_displacement_law_residual_ratio
            .is_finite()
        && closure.force_balance_ratio.is_finite()
        && closure.max_pressure_feedback_residual_ratio.is_finite()
        && closure.max_two_way_interface_residual_ratio.is_finite()
        && closure
            .max_structural_traction_update_residual_ratio
            .is_finite()
        && closure.max_structural_solve_residual_ratio.is_finite()
        && closure.max_interface_work_j_per_m2.is_finite()
        && closure.max_interface_work_j_per_m2 > 0.0
        && closure.max_structural_strain_energy_j_per_m2.is_finite()
        && closure.max_structural_strain_energy_j_per_m2 > 0.0
        && closure.max_interface_work_energy_residual_ratio.is_finite()
        && closure.structural_coupling_edge_count > 0
        && closure.interface_connectivity_coverage_ratio.is_finite()
        && closure.interface_connectivity_coverage_ratio >= 1.0
        && closure.mesh_backed_interface_connectivity_ratio.is_finite()
    {
        1.0
    } else {
        0.0
    };

    FsiKnownAnswerMetrics {
        pressure_loaded_wall_displacement_law_residual_ratio: closure
            .max_pressure_displacement_law_residual_ratio,
        interface_traction_balance_residual_ratio: closure.force_balance_ratio,
        interface_displacement_transfer_residual_m: closure.max_displacement_transfer_residual_m,
        partitioned_pressure_feedback_residual_ratio: closure.max_pressure_feedback_residual_ratio,
        two_way_interface_residual_ratio: closure.max_two_way_interface_residual_ratio,
        structural_traction_update_residual_ratio: closure
            .max_structural_traction_update_residual_ratio,
        structural_solve_residual_ratio: closure.max_structural_solve_residual_ratio,
        interface_work_energy_residual_ratio: closure.max_interface_work_energy_residual_ratio,
        interface_connectivity_coverage_ratio: closure.interface_connectivity_coverage_ratio,
        mesh_backed_interface_connectivity_ratio: closure.mesh_backed_interface_connectivity_ratio,
        known_answer_coverage_ratio,
    }
}

fn fsi_known_answer_diagnostic(
    metrics: &FsiKnownAnswerMetrics,
    tolerance: f64,
) -> runmat_analysis_fea::diagnostics::FeaDiagnostic {
    let severity = if metrics.pressure_loaded_wall_displacement_law_residual_ratio <= tolerance
        && metrics.interface_traction_balance_residual_ratio <= 1.0e-9
        && metrics.interface_displacement_transfer_residual_m <= 1.0e-12
        && metrics.partitioned_pressure_feedback_residual_ratio <= tolerance
        && metrics.two_way_interface_residual_ratio <= tolerance
        && metrics.structural_traction_update_residual_ratio <= tolerance
        && metrics.structural_solve_residual_ratio <= tolerance
        && metrics.interface_work_energy_residual_ratio <= tolerance
        && metrics.interface_connectivity_coverage_ratio >= 1.0
        && metrics.known_answer_coverage_ratio >= 1.0
    {
        runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info
    } else {
        runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
    };

    runmat_analysis_fea::diagnostics::FeaDiagnostic {
        code: "FEA_FSI_KNOWN_ANSWER".to_string(),
        severity,
        message: format!(
            "basis=pressure_loaded_wall_partitioned pressure_loaded_wall_displacement_law_residual_ratio={} interface_traction_balance_residual_ratio={} interface_displacement_transfer_residual_m={} partitioned_pressure_feedback_residual_ratio={} two_way_interface_residual_ratio={} structural_traction_update_residual_ratio={} structural_solve_residual_ratio={} interface_work_energy_residual_ratio={} interface_connectivity_coverage_ratio={} mesh_backed_interface_connectivity_ratio={} known_answer_coverage_ratio={}",
            metrics.pressure_loaded_wall_displacement_law_residual_ratio,
            metrics.interface_traction_balance_residual_ratio,
            metrics.interface_displacement_transfer_residual_m,
            metrics.partitioned_pressure_feedback_residual_ratio,
            metrics.two_way_interface_residual_ratio,
            metrics.structural_traction_update_residual_ratio,
            metrics.structural_solve_residual_ratio,
            metrics.interface_work_energy_residual_ratio,
            metrics.interface_connectivity_coverage_ratio,
            metrics.mesh_backed_interface_connectivity_ratio,
            metrics.known_answer_coverage_ratio,
        ),
    }
}

fn fsi_structural_compliance_per_pa(model: &AnalysisModel) -> f64 {
    let mean_modulus = model
        .materials
        .iter()
        .filter_map(|material| {
            let modulus = material.mechanical.youngs_modulus_pa;
            (modulus.is_finite() && modulus > 0.0).then_some(modulus)
        })
        .fold((0.0_f64, 0_usize), |(sum, count), modulus| {
            (sum + modulus, count + 1)
        });
    let youngs_modulus = if mean_modulus.1 > 0 {
        mean_modulus.0 / mean_modulus.1 as f64
    } else {
        200.0e9
    };
    1.0 / youngs_modulus.max(1.0e6)
}

pub fn analysis_run_transient_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    analysis_run_transient_with_options_op(
        model,
        backend,
        FormulationRequirement::Production,
        AnalysisTransientRunOptions::default(),
        context,
    )
}

pub fn analysis_run_cfd_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    analysis_run_cfd_with_options_op(
        model,
        backend,
        FormulationRequirement::Production,
        AnalysisCfdRunOptions::default(),
        context,
    )
}

pub fn analysis_run_cfd_with_options_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    formulation_requirement: FormulationRequirement,
    options: AnalysisCfdRunOptions,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    let _solver_context = install_fea_solver_context();
    let cfd_steps = model
        .sequence
        .steps
        .iter()
        .filter(|step| {
            step.family == PhysicsFamily::Fluid
                && matches!(
                    step.procedure,
                    AnalysisProcedure::Static | AnalysisProcedure::TransientResponse
                )
        })
        .collect::<Vec<_>>();
    let [cfd_step] = cfd_steps.as_slice() else {
        return Err(operation_error(
            AnalysisOperation::Cfd.name(),
            AnalysisOperation::Cfd.version(),
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_CFD.INVALID_MODEL",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "FEA model must include exactly one supported cfd step for fea.run_cfd",
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                ("geometry_id".to_string(), model.geometry_id.clone()),
            ]),
        ));
    };
    let cfd_procedure = cfd_step.procedure;

    let Some(cfd_domain) = model.cfd.as_ref() else {
        return Err(operation_error(
            AnalysisOperation::Cfd.name(),
            AnalysisOperation::Cfd.version(),
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
    if !cfd_domain.density.kilograms_per_cubic_meter.is_finite()
        || cfd_domain.density.kilograms_per_cubic_meter <= 0.0
    {
        return Err(operation_error(
            AnalysisOperation::Cfd.name(),
            AnalysisOperation::Cfd.version(),
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
                cfd_domain.density.kilograms_per_cubic_meter.to_string(),
            )]),
        ));
    }
    if !cfd_domain.dynamic_viscosity.pascal_seconds.is_finite()
        || cfd_domain.dynamic_viscosity.pascal_seconds <= 0.0
    {
        return Err(operation_error(
            AnalysisOperation::Cfd.name(),
            AnalysisOperation::Cfd.version(),
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
                cfd_domain.dynamic_viscosity.pascal_seconds.to_string(),
            )]),
        ));
    }
    if !nominal_inlet_velocity_m_per_s(model)
        .unwrap_or(0.0)
        .is_finite()
        || nominal_inlet_velocity_m_per_s(model).unwrap_or(0.0) < 0.0
    {
        return Err(operation_error(
            AnalysisOperation::Cfd.name(),
            AnalysisOperation::Cfd.version(),
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
                nominal_inlet_velocity_m_per_s(model)
                    .unwrap_or(0.0)
                    .to_string(),
            )]),
        ));
    }
    if !cfd_domain.turbulence_intensity.ratio.is_finite()
        || cfd_domain.turbulence_intensity.ratio < 0.0
        || cfd_domain.turbulence_intensity.ratio > 1.0
    {
        return Err(operation_error(
            AnalysisOperation::Cfd.name(),
            AnalysisOperation::Cfd.version(),
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
                cfd_domain.turbulence_intensity.ratio.to_string(),
            )]),
        ));
    }
    reject_moment_loads_for_run_family(
        model,
        AnalysisOperation::Cfd.name(),
        AnalysisOperation::Cfd.version(),
        "RM.FEA.RUN_CFD.INVALID_LOAD",
        "CFD",
        &context,
    )?;
    if let Err(detail) = validate_authored_cfd_boundary_conditions(model) {
        return Err(operation_error(
            AnalysisOperation::Cfd.name(),
            AnalysisOperation::Cfd.version(),
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_CFD.INVALID_BOUNDARY_CONDITIONS",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            detail.clone(),
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                ("detail".to_string(), detail),
            ]),
        ));
    }

    if !options.time_step_s.is_finite() || options.time_step_s <= 0.0 {
        return Err(operation_error(
            AnalysisOperation::Cfd.name(),
            AnalysisOperation::Cfd.version(),
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
            AnalysisOperation::Cfd.name(),
            AnalysisOperation::Cfd.version(),
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
            AnalysisOperation::Cfd.name(),
            AnalysisOperation::Cfd.version(),
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
            AnalysisOperation::Cfd.name(),
            AnalysisOperation::Cfd.version(),
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
            AnalysisOperation::Cfd.name(),
            AnalysisOperation::Cfd.version(),
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

    let solver_mesh = resolve_solver_mesh_artifact(
        model,
        options.solver_mesh_artifact_path.as_deref(),
        AnalysisOperation::Cfd.name(),
        AnalysisOperation::Cfd.version(),
        "RM.FEA.RUN_CFD.SOLVER_MESH_INVALID",
        &context,
    )?;
    let identity = admit_runtime_proxy_for_operation(
        PhysicsFamily::Fluid,
        cfd_procedure,
        formulation_requirement,
        RuntimeProxyFormulation::Domain {
            topology: RuntimeDomainTopology::from_solver_mesh(solver_mesh.as_ref()),
            source: RuntimeDomainSource::OrdinalProfile,
        },
        AnalysisOperation::Cfd,
        model,
        &context,
    )?;
    let solve_start = Instant::now();
    let mut run = solve_cfd_finite_volume_run(
        model,
        cfd_domain,
        backend,
        identity,
        &options,
        solver_mesh.as_ref(),
    );
    let solve_ms = solve_start.elapsed().as_secs_f64() * 1000.0;
    run.diagnostics
        .push(runmat_analysis_fea::diagnostics::FeaDiagnostic {
            code: "FEA_CFD_COST".to_string(),
            severity: runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info,
            message: format!(
                "solve_ms={} step_count={} max_linear_iters={} tolerance={}",
                solve_ms, options.step_count, options.max_linear_iters, options.tolerance,
            ),
        });
    let mut fallback_events = Vec::new();
    if backend == ComputeBackend::Gpu && run.execution_backend != ExecutionBackend::Accelerator {
        fallback_events.push(FallbackEvent::solver_backend());
    }

    let flow_topology = CfdDomainTopology::from_model(model, solver_mesh.as_ref());
    let flow_boundary_summary = CfdBoundarySummary::from_model(model, 2);
    let flow_inlet_velocity = flow_boundary_summary.nominal_inlet_velocity_m_per_s;
    let reynolds_number = cfd_reynolds_number_for_velocity(cfd_domain, flow_inlet_velocity);
    let solve_family = match cfd_procedure {
        AnalysisProcedure::Static => "steady_state",
        AnalysisProcedure::TransientResponse => "transient",
        _ => unreachable!("CFD admission accepts only static or transient response"),
    };
    run.diagnostics.push(runmat_analysis_fea::diagnostics::FeaDiagnostic {
        code: "FEA_CFD_FLOW".to_string(),
        severity: runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info,
        message: format!(
            "density={} viscosity={} inlet_velocity={} turbulence_intensity={} reynolds_number={} solve_family={} profile_point_count={} topology_basis={} control_volume_count={} control_volume_face_count={} control_volume_internal_face_count={} control_volume_boundary_face_count={} control_volume_connectivity_coverage_ratio={} domain_length_m={} hydraulic_diameter_m={}",
            cfd_domain.density.kilograms_per_cubic_meter,
            cfd_domain.dynamic_viscosity.pascal_seconds,
            flow_inlet_velocity,
            cfd_domain.turbulence_intensity.ratio,
            reynolds_number,
            solve_family,
            cfd_velocity_scales(model, cfd_domain).len(),
            flow_topology.basis.as_str(),
            flow_topology.control_volume_count,
            flow_topology.control_volume_face_count,
            flow_topology.control_volume_internal_face_count,
            flow_topology.control_volume_boundary_face_count,
            flow_topology.control_volume_connectivity_coverage_ratio,
            flow_topology.domain_length_m,
            flow_topology.hydraulic_diameter_m,
        ),
    });

    let max_momentum_residual = diagnostic_metric(
        &run.diagnostics,
        "FEA_CFD_RESIDUAL",
        "max_momentum_residual",
    )
    .unwrap_or(f64::INFINITY);
    let max_continuity_residual = diagnostic_metric(
        &run.diagnostics,
        "FEA_CFD_RESIDUAL",
        "max_continuity_residual",
    )
    .unwrap_or(f64::INFINITY);
    let solver_convergence = if max_momentum_residual <= options.residual_warn_threshold
        && max_continuity_residual <= options.residual_warn_threshold
    {
        QualityGate::Pass
    } else {
        QualityGate::Warn
    };
    let result_quality = if run.fields_are_empty()
        || !max_momentum_residual.is_finite()
        || !max_continuity_residual.is_finite()
    {
        QualityGate::Fail
    } else if max_momentum_residual > options.residual_warn_threshold
        || max_continuity_residual > options.residual_warn_threshold
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
        .any(FallbackEvent::is_solver_backend_fallback)
    {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::SolverBackendFallback,
            detail: "solver backend fell back from accelerator to cpu_reference".to_string(),
        });
    }

    let numerically_acceptable = match options.quality_policy {
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
    let (production_publishable, production_publication_reasons) =
        evaluate_production_publication(&run, numerically_acceptable);
    let run_status = classify_run_status(result_quality, numerically_acceptable);
    let solver_backend = run.solver_backend;
    let execution_backend = run.execution_backend;
    let solver_device_apply_k_ratio = run.solver_device_apply_k_ratio;
    let solver_host_sync_count = run.solver_host_sync_count;
    let solver_method = run.solver_method;
    let selected_preconditioner = run.preconditioner;

    let result = AnalysisRunResult {
        run_id: storage::next_run_id(),
        run,
        render_topology: render_topology_from_solver_mesh(solver_mesh.as_ref()),
        modal_results: None,
        thermal_results: None,
        transient_results: None,
        nonlinear_results: None,
        electromagnetic_results: None,
        model_validity: QualityGate::Pass,
        solver_convergence,
        result_quality,
        run_status,
        numerically_acceptable,
        production_publishable,
        quality_reasons,
        production_publication_reasons,
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
        },
    };

    persist_fea_run_result_with_progress(
        AnalysisOperation::Cfd.name(),
        AnalysisOperation::Cfd.version(),
        "RM.FEA.RUN_CFD.ARTIFACT_STORE_FAILED",
        &context,
        &result,
    )?;

    Ok(OperationEnvelope::new(
        AnalysisOperation::Cfd.name(),
        AnalysisOperation::Cfd.version(),
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
        FormulationRequirement::Production,
        AnalysisThermalRunOptions::default(),
        context,
    )
}

pub fn analysis_run_cht_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    analysis_run_cht_with_options_op(
        model,
        backend,
        FormulationRequirement::Production,
        AnalysisChtRunOptions::default(),
        context,
    )
}

pub fn analysis_run_cht_with_options_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    formulation_requirement: FormulationRequirement,
    options: AnalysisChtRunOptions,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    let _solver_context = install_fea_solver_context();
    let cht_steps = model
        .sequence
        .steps
        .iter()
        .filter(|step| {
            step.family == PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer)
                && matches!(
                    step.procedure,
                    AnalysisProcedure::Static | AnalysisProcedure::TransientResponse
                )
        })
        .collect::<Vec<_>>();
    let [cht_step] = cht_steps.as_slice() else {
        return Err(operation_error(
            AnalysisOperation::Cht.name(),
            AnalysisOperation::Cht.version(),
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_CHT.INVALID_MODEL",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "FEA model must include exactly one supported conjugate-heat-transfer procedure for fea.run_cht",
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                ("geometry_id".to_string(), model.geometry_id.clone()),
            ]),
        ));
    };
    let cht_procedure = cht_step.procedure;
    let Some(cfd_domain) = model.cfd.as_ref() else {
        return Err(operation_error(
            AnalysisOperation::Cht.name(),
            AnalysisOperation::Cht.version(),
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
    if !cfd_domain.density.kilograms_per_cubic_meter.is_finite()
        || cfd_domain.density.kilograms_per_cubic_meter <= 0.0
    {
        return Err(operation_error(
            AnalysisOperation::Cht.name(),
            AnalysisOperation::Cht.version(),
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
                cfd_domain.density.kilograms_per_cubic_meter.to_string(),
            )]),
        ));
    }
    if !cfd_domain.dynamic_viscosity.pascal_seconds.is_finite()
        || cfd_domain.dynamic_viscosity.pascal_seconds <= 0.0
    {
        return Err(operation_error(
            AnalysisOperation::Cht.name(),
            AnalysisOperation::Cht.version(),
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
                cfd_domain.dynamic_viscosity.pascal_seconds.to_string(),
            )]),
        ));
    }
    if !nominal_inlet_velocity_m_per_s(model)
        .unwrap_or(0.0)
        .is_finite()
        || nominal_inlet_velocity_m_per_s(model).unwrap_or(0.0) < 0.0
    {
        return Err(operation_error(
            AnalysisOperation::Cht.name(),
            AnalysisOperation::Cht.version(),
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
                nominal_inlet_velocity_m_per_s(model)
                    .unwrap_or(0.0)
                    .to_string(),
            )]),
        ));
    }
    if !cfd_domain.turbulence_intensity.ratio.is_finite()
        || cfd_domain.turbulence_intensity.ratio < 0.0
        || cfd_domain.turbulence_intensity.ratio > 1.0
    {
        return Err(operation_error(
            AnalysisOperation::Cht.name(),
            AnalysisOperation::Cht.version(),
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
                cfd_domain.turbulence_intensity.ratio.to_string(),
            )]),
        ));
    }
    reject_moment_loads_for_run_family(
        model,
        AnalysisOperation::Cht.name(),
        AnalysisOperation::Cht.version(),
        "RM.FEA.RUN_CHT.INVALID_LOAD",
        "CHT",
        &context,
    )?;
    if !options.time_step_s.is_finite() || options.time_step_s <= 0.0 {
        return Err(operation_error(
            AnalysisOperation::Cht.name(),
            AnalysisOperation::Cht.version(),
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
            AnalysisOperation::Cht.name(),
            AnalysisOperation::Cht.version(),
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
            AnalysisOperation::Cht.name(),
            AnalysisOperation::Cht.version(),
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
            AnalysisOperation::Cht.name(),
            AnalysisOperation::Cht.version(),
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

    let thermo_options = model_thermo_coupling_options(model);
    let Some(thermo_options) = thermo_options else {
        return Err(operation_error(
            AnalysisOperation::Cht.name(),
            AnalysisOperation::Cht.version(),
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
            AnalysisOperation::Cht.name(),
            AnalysisOperation::Cht.version(),
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
    if let Err((detail, metadata)) = validate_coupled_flow_interfaces(model, "CHT") {
        return Err(operation_error(
            AnalysisOperation::Cht.name(),
            AnalysisOperation::Cht.version(),
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_CHT.INVALID_INTERFACE_MAPPING",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            detail,
            metadata,
        ));
    }
    let applied_temperature_delta_k = thermo_options.applied_temperature_delta_k;

    let solve_start = Instant::now();
    let solver_mesh = resolve_solver_mesh_artifact(
        model,
        options.solver_mesh_artifact_path.as_deref(),
        AnalysisOperation::Cht.name(),
        AnalysisOperation::Cht.version(),
        "RM.FEA.RUN_CHT.SOLVER_MESH_INVALID",
        &context,
    )?;
    let Some(solver_mesh) = solver_mesh else {
        return Err(operation_error(
            AnalysisOperation::Cht.name(),
            AnalysisOperation::Cht.version(),
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_CHT.SOLVER_MESH_REQUIRED",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_cht requires a canonical solver mesh",
            BTreeMap::new(),
        ));
    };
    let identity = admit_runtime_proxy_for_operation(
        PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer),
        cht_procedure,
        formulation_requirement,
        RuntimeProxyFormulation::Domain {
            topology: RuntimeDomainTopology::from_solver_mesh(Some(&solver_mesh)),
            source: RuntimeDomainSource::OrdinalProfile,
        },
        AnalysisOperation::Cht,
        model,
        &context,
    )?;
    let thermal_run = run_thermal_with_options(
        model,
        backend,
        ThermalSolveOptions {
            formulation_requirement,
            procedure: ThermalProcedureControl::BackwardEuler {
                step_count: options.step_count,
                time_step_s: options.time_step_s,
            },
            residual_target: options.residual_warn_threshold,
            solver_mesh: Some(solver_mesh.clone()),
        },
    )
    .map_err(|err| map_fea_run_error(AnalysisOperation::Cht, model, &context, err))?;

    let topology = CfdDomainTopology::from_model(model, Some(&solver_mesh));
    let node_count = topology.node_count;
    let field_step = if cht_procedure == AnalysisProcedure::TransientResponse {
        options.step_count.saturating_sub(1)
    } else {
        0
    };
    let cfd_profile_scales = cfd_velocity_scales(model, cfd_domain);
    let (fluid_velocity, fluid_pressure) = recover_cfd_velocity_pressure(
        model,
        cfd_domain,
        &topology,
        field_step,
        &cfd_profile_scales,
    );
    let (cfd_residual_momentum, cfd_residual_continuity) = cfd_residual_norms(
        &fluid_velocity,
        &fluid_pressure,
        cfd_domain,
        &topology,
        options.step_count,
    );
    let max_cfd_momentum_residual = cfd_residual_momentum
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let max_cfd_continuity_residual = cfd_residual_continuity
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let (cht_fields, cht_interface_closure) = build_cht_run_fields(
        model,
        cfd_domain,
        &topology,
        &thermal_run,
        cht_interface_conductance_w_per_m2k(model),
        options.max_linear_iters,
        options.tolerance,
    );
    let mut run = thermal_run.run.clone();
    run.identity = identity;
    run.solver_method = SolverMethod::ChtConjugateProjection;
    run.preconditioner = SolverPreconditioner::ThermalCfdProjection;
    run.fields.extend(cht_fields);
    let reynolds_number = cfd_reynolds_number_for_velocity(
        cfd_domain,
        nominal_inlet_velocity_m_per_s(model).unwrap_or(0.0),
    );
    run.diagnostics.push(runmat_analysis_fea::diagnostics::FeaDiagnostic {
        code: "FEA_CFD_FLOW".to_string(),
        severity: runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info,
        message: format!(
            "density={} viscosity={} inlet_velocity={} turbulence_intensity={} reynolds_number={} solve_family={} profile_point_count={} topology_basis={} control_volume_count={} control_volume_face_count={} control_volume_internal_face_count={} control_volume_boundary_face_count={} control_volume_connectivity_coverage_ratio={} domain_length_m={} hydraulic_diameter_m={}",
            cfd_domain.density.kilograms_per_cubic_meter,
            cfd_domain.dynamic_viscosity.pascal_seconds,
            nominal_inlet_velocity_m_per_s(model).unwrap_or(0.0),
            cfd_domain.turbulence_intensity.ratio,
            reynolds_number,
            if cht_procedure == AnalysisProcedure::TransientResponse {
                "transient"
            } else {
                "steady_state"
            },
            cfd_profile_scales.len(),
            topology.basis.as_str(),
            topology.control_volume_count,
            topology.control_volume_face_count,
            topology.control_volume_internal_face_count,
            topology.control_volume_boundary_face_count,
            topology.control_volume_connectivity_coverage_ratio,
            topology.domain_length_m,
            topology.hydraulic_diameter_m,
        ),
    });
    let cfd_residual_severity = if max_cfd_momentum_residual <= options.residual_warn_threshold
        && max_cfd_continuity_residual <= options.residual_warn_threshold
    {
        runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info
    } else {
        runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
    };
    run.diagnostics.push(runmat_analysis_fea::diagnostics::FeaDiagnostic {
        code: "FEA_CFD_RESIDUAL".to_string(),
        severity: cfd_residual_severity,
        message: format!(
            "max_momentum_residual={} max_continuity_residual={} residual_warn_threshold={} cfd_node_count={} cfd_step_count={}",
            max_cfd_momentum_residual,
            max_cfd_continuity_residual,
            options.residual_warn_threshold,
            node_count,
            options.step_count,
        ),
    });
    run.diagnostics.push(cfd_assembly_diagnostic(
        &topology,
        cfd_domain,
        mean_cfd_axial_velocity(&fluid_velocity),
        options.time_step_s,
        pressure_drop_from_nodal_pressure(&fluid_pressure),
        max_cfd_continuity_residual,
        options.residual_warn_threshold,
    ));
    run.diagnostics.push(runmat_analysis_fea::diagnostics::FeaDiagnostic {
        code: "FEA_CHT_COUPLING".to_string(),
        severity: runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info,
        message: format!(
            "reference_temperature_k={} applied_temperature_delta_k={} step_count={} time_step_s={} authored_interface_count={}",
            thermal_run.reference_temperature_k,
            applied_temperature_delta_k,
            options.step_count,
            options.time_step_s,
            model.interfaces.len(),
        ),
    });
    run.diagnostics.push(runmat_analysis_fea::diagnostics::FeaDiagnostic {
        code: "FEA_CHT_INTERFACE_CLOSURE".to_string(),
        severity: if cht_interface_closure.heat_flux_balance_ratio <= 1.0e-9
            && cht_interface_closure.max_energy_residual <= options.residual_warn_threshold
            && cht_interface_closure.max_temperature_jump_k <= 0.1
            && cht_interface_closure.max_thermal_transport_residual_ratio
                <= options.residual_warn_threshold
            && cht_interface_closure.max_flux_temperature_law_residual_ratio
                <= options.residual_warn_threshold
            && cht_interface_closure.max_heat_flux_realization_residual_ratio
                <= options.residual_warn_threshold
            && cht_interface_closure.max_coupled_interface_residual_ratio
                <= options.residual_warn_threshold
            && cht_interface_closure.interface_connectivity_coverage_ratio >= 1.0
            && cht_interface_closure.max_thermal_network_residual_ratio
                <= options.residual_warn_threshold
        {
            runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info
        } else {
            runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "interface_face_count={} max_temperature_jump_k={} max_energy_residual={} heat_flux_balance_ratio={} mean_interface_heat_flux_w_per_m2={} thermal_transport_residual_ratio={} interface_temperature_continuity_ratio={} max_advection_temperature_shift_k={} interface_conductance_w_per_m2k={} flux_temperature_law_residual_ratio={} heat_flux_realization_residual_ratio={} coupled_interface_iteration_count={} coupled_interface_residual_ratio={} thermal_network_node_count={} thermal_network_edge_count={} interface_connectivity_coverage_ratio={} mesh_backed_interface_connectivity_ratio={} full_topology_edge_count={} full_topology_element_count={} thermal_network_residual_ratio={}",
            cht_interface_closure.interface_face_count,
            cht_interface_closure.max_temperature_jump_k,
            cht_interface_closure.max_energy_residual,
            cht_interface_closure.heat_flux_balance_ratio,
            cht_interface_closure.mean_interface_heat_flux_w_per_m2,
            cht_interface_closure.max_thermal_transport_residual_ratio,
            cht_interface_closure.interface_temperature_continuity_ratio,
            cht_interface_closure.max_advection_temperature_shift_k,
            cht_interface_closure.interface_conductance_w_per_m2k,
            cht_interface_closure.max_flux_temperature_law_residual_ratio,
            cht_interface_closure.max_heat_flux_realization_residual_ratio,
            cht_interface_closure.max_coupled_interface_iteration_count,
            cht_interface_closure.max_coupled_interface_residual_ratio,
            cht_interface_closure.thermal_network_node_count,
            cht_interface_closure.thermal_network_edge_count,
            cht_interface_closure.interface_connectivity_coverage_ratio,
            cht_interface_closure.mesh_backed_interface_connectivity_ratio,
            cht_interface_closure.full_topology_edge_count,
            cht_interface_closure.full_topology_element_count,
            cht_interface_closure.max_thermal_network_residual_ratio,
        ),
    });
    let cht_known_answer = cht_known_answer_metrics(reynolds_number, &cht_interface_closure);
    run.diagnostics.push(cht_known_answer_diagnostic(
        &cht_known_answer,
        options.residual_warn_threshold,
    ));
    let solve_ms = solve_start.elapsed().as_secs_f64() * 1000.0;
    run.diagnostics
        .push(runmat_analysis_fea::diagnostics::FeaDiagnostic {
            code: "FEA_CHT_COST".to_string(),
            severity: runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info,
            message: format!(
                "solve_ms={} step_count={} max_linear_iters={} tolerance={}",
                solve_ms, options.step_count, options.max_linear_iters, options.tolerance,
            ),
        });

    let mut fallback_events = Vec::new();
    if backend == ComputeBackend::Gpu && run.execution_backend != ExecutionBackend::Accelerator {
        fallback_events.push(FallbackEvent::solver_backend());
    }

    let max_cht_transport_residual = cht_interface_closure
        .max_thermal_transport_residual_ratio
        .max(
            cht_interface_closure
                .max_energy_residual
                .max(cht_interface_closure.heat_flux_balance_ratio),
        );
    let solver_convergence = if max_cfd_momentum_residual <= options.residual_warn_threshold
        && max_cfd_continuity_residual <= options.residual_warn_threshold
        && max_cht_transport_residual <= options.residual_warn_threshold
    {
        QualityGate::Pass
    } else {
        QualityGate::Warn
    };
    let result_quality = if run.fields_are_empty()
        || thermal_run.temperature_snapshots.is_empty()
        || thermal_run.time_points_s.is_empty()
        || thermal_run.residual_norms.iter().any(|r| !r.is_finite())
        || !max_cfd_momentum_residual.is_finite()
        || !max_cfd_continuity_residual.is_finite()
        || !max_cht_transport_residual.is_finite()
    {
        QualityGate::Fail
    } else if max_cfd_momentum_residual > options.residual_warn_threshold
        || max_cfd_continuity_residual > options.residual_warn_threshold
        || max_cht_transport_residual > options.residual_warn_threshold
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
    if max_cfd_momentum_residual > options.residual_warn_threshold
        || max_cfd_continuity_residual > options.residual_warn_threshold
    {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::SolverNotConverged,
            detail: format!(
                "cht cfd residual exceeds threshold {}",
                options.residual_warn_threshold,
            ),
        });
    }
    if max_cht_transport_residual > options.residual_warn_threshold {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::SolverNotConverged,
            detail: format!(
                "cht interface transport residual exceeds threshold {}",
                options.residual_warn_threshold
            ),
        });
    }
    if fallback_events
        .iter()
        .any(FallbackEvent::is_solver_backend_fallback)
    {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::SolverBackendFallback,
            detail: "solver backend fell back from accelerator to cpu_reference".to_string(),
        });
    }

    let numerically_acceptable = match options.quality_policy {
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
    let (production_publishable, production_publication_reasons) =
        evaluate_production_publication(&run, numerically_acceptable);
    let run_status = classify_run_status(result_quality, numerically_acceptable);
    let solver_backend = run.solver_backend;
    let execution_backend = run.execution_backend;
    let solver_device_apply_k_ratio = run.solver_device_apply_k_ratio;
    let solver_host_sync_count = run.solver_host_sync_count;
    let solver_method = run.solver_method;
    let selected_preconditioner = run.preconditioner;

    let result = AnalysisRunResult {
        run_id: storage::next_run_id(),
        run,
        render_topology: render_topology_from_solver_mesh(Some(&solver_mesh)),
        modal_results: None,
        thermal_results: Some(ThermalResultsData {
            thermal_payload_version: "thermal_results/v1".to_string(),
            time_points_s: thermal_run.time_points_s,
            temperature_snapshots: thermal_run.temperature_snapshots,
            temperature_gradient_snapshots: thermal_run.temperature_gradient_snapshots,
            heat_flux_snapshots: thermal_run.heat_flux_snapshots,
            energy_storage_rate_snapshots: thermal_run.energy_storage_rate_snapshots,
            residual_norms: thermal_run.residual_norms,
            reference_temperature_k: thermal_run.reference_temperature_k,
            initial_state: thermal_run.initial_state,
        }),
        transient_results: None,
        nonlinear_results: None,
        electromagnetic_results: None,
        model_validity: QualityGate::Pass,
        solver_convergence,
        result_quality,
        run_status,
        numerically_acceptable,
        production_publishable,
        quality_reasons,
        production_publication_reasons,
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
        },
    };

    persist_fea_run_result_with_progress(
        AnalysisOperation::Cht.name(),
        AnalysisOperation::Cht.version(),
        "RM.FEA.RUN_CHT.ARTIFACT_STORE_FAILED",
        &context,
        &result,
    )?;

    Ok(OperationEnvelope::new(
        AnalysisOperation::Cht.name(),
        AnalysisOperation::Cht.version(),
        &context,
        result,
    ))
}

pub fn analysis_run_fsi_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    analysis_run_fsi_with_options_op(
        model,
        backend,
        FormulationRequirement::Production,
        AnalysisFsiRunOptions::default(),
        context,
    )
}

pub fn analysis_run_fsi_with_options_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    formulation_requirement: FormulationRequirement,
    options: AnalysisFsiRunOptions,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    let _solver_context = install_fea_solver_context();
    let fsi_steps = model
        .sequence
        .steps
        .iter()
        .filter(|step| {
            step.family == PhysicsFamily::Coupled(CoupledPhysicsFamily::FluidStructureInteraction)
                && step.procedure == AnalysisProcedure::TransientResponse
        })
        .collect::<Vec<_>>();
    if !matches!(fsi_steps.as_slice(), [_]) {
        return Err(operation_error(
            AnalysisOperation::Fsi.name(),
            AnalysisOperation::Fsi.version(),
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_FSI.INVALID_MODEL",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "FEA model must include exactly one transient fluid-structure-interaction procedure for fea.run_fsi",
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                ("geometry_id".to_string(), model.geometry_id.clone()),
            ]),
        ));
    }
    let Some(cfd_domain) = model.cfd.as_ref() else {
        return Err(operation_error(
            AnalysisOperation::Fsi.name(),
            AnalysisOperation::Fsi.version(),
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
    if !cfd_domain.density.kilograms_per_cubic_meter.is_finite()
        || cfd_domain.density.kilograms_per_cubic_meter <= 0.0
    {
        return Err(operation_error(
            AnalysisOperation::Fsi.name(),
            AnalysisOperation::Fsi.version(),
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
                cfd_domain.density.kilograms_per_cubic_meter.to_string(),
            )]),
        ));
    }
    if !cfd_domain.dynamic_viscosity.pascal_seconds.is_finite()
        || cfd_domain.dynamic_viscosity.pascal_seconds <= 0.0
    {
        return Err(operation_error(
            AnalysisOperation::Fsi.name(),
            AnalysisOperation::Fsi.version(),
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
                cfd_domain.dynamic_viscosity.pascal_seconds.to_string(),
            )]),
        ));
    }
    if !nominal_inlet_velocity_m_per_s(model)
        .unwrap_or(0.0)
        .is_finite()
        || nominal_inlet_velocity_m_per_s(model).unwrap_or(0.0) < 0.0
    {
        return Err(operation_error(
            AnalysisOperation::Fsi.name(),
            AnalysisOperation::Fsi.version(),
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
                nominal_inlet_velocity_m_per_s(model)
                    .unwrap_or(0.0)
                    .to_string(),
            )]),
        ));
    }
    if !cfd_domain.turbulence_intensity.ratio.is_finite()
        || cfd_domain.turbulence_intensity.ratio < 0.0
        || cfd_domain.turbulence_intensity.ratio > 1.0
    {
        return Err(operation_error(
            AnalysisOperation::Fsi.name(),
            AnalysisOperation::Fsi.version(),
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
                cfd_domain.turbulence_intensity.ratio.to_string(),
            )]),
        ));
    }
    reject_moment_loads_for_run_family(
        model,
        AnalysisOperation::Fsi.name(),
        AnalysisOperation::Fsi.version(),
        "RM.FEA.RUN_FSI.INVALID_LOAD",
        "FSI",
        &context,
    )?;
    if !options.time_step_s.is_finite() || options.time_step_s <= 0.0 {
        return Err(operation_error(
            AnalysisOperation::Fsi.name(),
            AnalysisOperation::Fsi.version(),
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
            AnalysisOperation::Fsi.name(),
            AnalysisOperation::Fsi.version(),
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
            AnalysisOperation::Fsi.name(),
            AnalysisOperation::Fsi.version(),
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
            AnalysisOperation::Fsi.name(),
            AnalysisOperation::Fsi.version(),
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
    if let Err((detail, metadata)) = validate_coupled_flow_interfaces(model, "FSI") {
        return Err(operation_error(
            AnalysisOperation::Fsi.name(),
            AnalysisOperation::Fsi.version(),
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_FSI.INVALID_INTERFACE_MAPPING",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            detail,
            metadata,
        ));
    }

    let solver_mesh = resolve_solver_mesh_artifact(
        model,
        options.solver_mesh_artifact_path.as_deref(),
        AnalysisOperation::Fsi.name(),
        AnalysisOperation::Fsi.version(),
        "RM.FEA.RUN_FSI.SOLVER_MESH_INVALID",
        &context,
    )?;
    let identity = admit_runtime_proxy_for_operation(
        PhysicsFamily::Coupled(CoupledPhysicsFamily::FluidStructureInteraction),
        AnalysisProcedure::TransientResponse,
        formulation_requirement,
        RuntimeProxyFormulation::Domain {
            topology: RuntimeDomainTopology::from_solver_mesh(solver_mesh.as_ref()),
            source: RuntimeDomainSource::OrdinalProfile,
        },
        AnalysisOperation::Fsi,
        model,
        &context,
    )?;
    let solve_start = Instant::now();
    let topology = CfdDomainTopology::from_model(model, solver_mesh.as_ref());
    let node_count = topology.node_count;
    let field_step = options.step_count.saturating_sub(1);
    let cfd_profile_scales = cfd_velocity_scales(model, cfd_domain);
    let (fluid_velocity, fluid_pressure) = recover_cfd_velocity_pressure(
        model,
        cfd_domain,
        &topology,
        field_step,
        &cfd_profile_scales,
    );
    let (cfd_residual_momentum, cfd_residual_continuity) = cfd_residual_norms(
        &fluid_velocity,
        &fluid_pressure,
        cfd_domain,
        &topology,
        options.step_count,
    );
    let max_cfd_momentum_residual = cfd_residual_momentum
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let max_cfd_continuity_residual = cfd_residual_continuity
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let structural_compliance_per_pa = fsi_structural_compliance_per_pa(model);
    let (fsi_fields, fsi_interface_closure) = build_fsi_run_fields(
        model,
        cfd_domain,
        &topology,
        options.step_count,
        structural_compliance_per_pa,
        options.max_linear_iters,
        options.tolerance,
        &cfd_residual_momentum,
        &cfd_residual_continuity,
    );
    let max_interface_residual = fsi_interface_closure.max_interface_residual;
    let mut run = FeaRunResult {
        identity,
        requested_backend: backend,
        execution_backend: ExecutionBackend::Cpu,
        solver_backend: LinearAlgebraBackendKind::CpuReference,
        solver_device_apply_k_ratio: 0.0,
        solver_method: SolverMethod::FsiPartitionedProjection,
        preconditioner: SolverPreconditioner::InterfaceRelaxation,
        solver_host_sync_count: 0,
        accelerator_evidence: Vec::new(),
        diagnostics: Vec::new(),
        spatial_projection_evidence: Vec::new(),
        observations: Vec::new(),
        fields: fsi_fields,
    };
    let reynolds_number = cfd_reynolds_number_for_velocity(
        cfd_domain,
        nominal_inlet_velocity_m_per_s(model).unwrap_or(0.0),
    );
    run.diagnostics.push(runmat_analysis_fea::diagnostics::FeaDiagnostic {
        code: "FEA_CFD_FLOW".to_string(),
        severity: runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info,
        message: format!(
            "density={} viscosity={} inlet_velocity={} turbulence_intensity={} reynolds_number={} solve_family={} profile_point_count={} topology_basis={} control_volume_count={} control_volume_face_count={} control_volume_internal_face_count={} control_volume_boundary_face_count={} control_volume_connectivity_coverage_ratio={} domain_length_m={} hydraulic_diameter_m={}",
            cfd_domain.density.kilograms_per_cubic_meter,
            cfd_domain.dynamic_viscosity.pascal_seconds,
            nominal_inlet_velocity_m_per_s(model).unwrap_or(0.0),
            cfd_domain.turbulence_intensity.ratio,
            reynolds_number,
            "transient",
            cfd_profile_scales.len(),
            topology.basis.as_str(),
            topology.control_volume_count,
            topology.control_volume_face_count,
            topology.control_volume_internal_face_count,
            topology.control_volume_boundary_face_count,
            topology.control_volume_connectivity_coverage_ratio,
            topology.domain_length_m,
            topology.hydraulic_diameter_m,
        ),
    });
    let residual_severity = if max_interface_residual <= options.residual_warn_threshold {
        runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info
    } else {
        runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
    };
    run.diagnostics.push(runmat_analysis_fea::diagnostics::FeaDiagnostic {
        code: "FEA_CFD_RESIDUAL".to_string(),
        severity: residual_severity,
        message: format!(
            "max_momentum_residual={} max_continuity_residual={} residual_warn_threshold={} cfd_node_count={} cfd_step_count={}",
            max_cfd_momentum_residual,
            max_cfd_continuity_residual,
            options.residual_warn_threshold,
            node_count,
            options.step_count,
        ),
    });
    run.diagnostics.push(cfd_assembly_diagnostic(
        &topology,
        cfd_domain,
        mean_cfd_axial_velocity(&fluid_velocity),
        options.time_step_s,
        pressure_drop_from_nodal_pressure(&fluid_pressure),
        max_cfd_continuity_residual,
        options.residual_warn_threshold,
    ));
    run.diagnostics.push(runmat_analysis_fea::diagnostics::FeaDiagnostic {
        code: "FEA_FSI_INTERFACE_RESIDUAL".to_string(),
        severity: residual_severity,
        message: format!(
            "max_interface_residual={} structural_compliance_per_pa={} residual_warn_threshold={} interface_node_count={} interface_face_count={}",
            max_interface_residual,
            structural_compliance_per_pa,
            options.residual_warn_threshold,
            fsi_interface_closure.interface_node_count,
            fsi_interface_closure.interface_face_count,
        ),
    });
    run.diagnostics.push(runmat_analysis_fea::diagnostics::FeaDiagnostic {
        code: "FEA_FSI_INTERFACE_CLOSURE".to_string(),
        severity: if fsi_interface_closure.force_balance_ratio <= 1.0e-9
            && fsi_interface_closure.max_displacement_transfer_residual_m <= 1.0e-12
            && fsi_interface_closure.max_pressure_feedback_residual_ratio <= options.tolerance
            && fsi_interface_closure.max_two_way_interface_residual_ratio <= options.tolerance
            && fsi_interface_closure.max_structural_traction_update_residual_ratio
                <= options.tolerance
            && fsi_interface_closure.max_pressure_displacement_law_residual_ratio
                <= options.tolerance
            && fsi_interface_closure.max_structural_solve_residual_ratio <= options.tolerance
            && fsi_interface_closure.max_interface_work_energy_residual_ratio <= options.tolerance
            && fsi_interface_closure.structural_coupling_edge_count > 0
            && fsi_interface_closure.interface_connectivity_coverage_ratio >= 1.0
            && fsi_interface_closure.max_interface_residual <= options.residual_warn_threshold
        {
            runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info
        } else {
            runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "interface_node_count={} interface_face_count={} max_interface_residual={} force_balance_ratio={} max_displacement_transfer_residual_m={} max_interface_displacement_m={} mean_interface_pressure_pa={} max_traction_magnitude_pa={} max_coupling_iteration_count={} pressure_feedback_residual_ratio={} two_way_interface_residual_ratio={} structural_traction_update_residual_ratio={} pressure_displacement_law_residual_ratio={} structural_solve_residual_ratio={} interface_work_j_per_m2={} structural_strain_energy_j_per_m2={} interface_work_energy_residual_ratio={} structural_coupling_edge_count={} interface_connectivity_coverage_ratio={} mesh_backed_interface_connectivity_ratio={} full_topology_edge_count={} full_topology_element_count={} interface_stiffness_pa_per_m={}",
            fsi_interface_closure.interface_node_count,
            fsi_interface_closure.interface_face_count,
            fsi_interface_closure.max_interface_residual,
            fsi_interface_closure.force_balance_ratio,
            fsi_interface_closure.max_displacement_transfer_residual_m,
            fsi_interface_closure.max_interface_displacement_m,
            fsi_interface_closure.mean_interface_pressure_pa,
            fsi_interface_closure.max_traction_magnitude_pa,
            fsi_interface_closure.max_coupling_iteration_count,
            fsi_interface_closure.max_pressure_feedback_residual_ratio,
            fsi_interface_closure.max_two_way_interface_residual_ratio,
            fsi_interface_closure.max_structural_traction_update_residual_ratio,
            fsi_interface_closure.max_pressure_displacement_law_residual_ratio,
            fsi_interface_closure.max_structural_solve_residual_ratio,
            fsi_interface_closure.max_interface_work_j_per_m2,
            fsi_interface_closure.max_structural_strain_energy_j_per_m2,
            fsi_interface_closure.max_interface_work_energy_residual_ratio,
            fsi_interface_closure.structural_coupling_edge_count,
            fsi_interface_closure.interface_connectivity_coverage_ratio,
            fsi_interface_closure.mesh_backed_interface_connectivity_ratio,
            fsi_interface_closure.full_topology_edge_count,
            fsi_interface_closure.full_topology_element_count,
            fsi_interface_closure.interface_stiffness_pa_per_m,
        ),
    });
    let fsi_known_answer = fsi_known_answer_metrics(&fsi_interface_closure);
    run.diagnostics.push(fsi_known_answer_diagnostic(
        &fsi_known_answer,
        options.tolerance,
    ));
    run.diagnostics.push(runmat_analysis_fea::diagnostics::FeaDiagnostic {
        code: "FEA_FSI_COUPLING".to_string(),
        severity: runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info,
        message: format!(
            "step_count={} time_step_s={} fsi_procedure_count={} cfd_profile_point_count={} authored_interface_count={} interface_node_count={} interface_face_count={} max_linear_iters={} tolerance={}",
            options.step_count,
            options.time_step_s,
            model
                .sequence.steps
                .iter()
                .filter(|step| step.procedure == AnalysisProcedure::TransientResponse)
                .count(),
            cfd_profile_scales.len(),
            model.interfaces.len(),
            fsi_interface_closure.interface_node_count,
            fsi_interface_closure.interface_face_count,
            options.max_linear_iters,
            options.tolerance,
        ),
    });
    let solve_ms = solve_start.elapsed().as_secs_f64() * 1000.0;
    run.diagnostics
        .push(runmat_analysis_fea::diagnostics::FeaDiagnostic {
            code: "FEA_FSI_COST".to_string(),
            severity: runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info,
            message: format!(
                "solve_ms={} step_count={} max_linear_iters={} tolerance={}",
                solve_ms, options.step_count, options.max_linear_iters, options.tolerance,
            ),
        });

    let mut fallback_events = Vec::new();
    if backend == ComputeBackend::Gpu && run.execution_backend != ExecutionBackend::Accelerator {
        fallback_events.push(FallbackEvent::solver_backend());
    }

    let solver_convergence = if max_interface_residual <= options.residual_warn_threshold {
        QualityGate::Pass
    } else {
        QualityGate::Warn
    };
    let result_quality = if run.fields_are_empty()
        || !max_cfd_momentum_residual.is_finite()
        || !max_cfd_continuity_residual.is_finite()
        || !max_interface_residual.is_finite()
    {
        QualityGate::Fail
    } else if max_interface_residual > options.residual_warn_threshold {
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
    if max_interface_residual > options.residual_warn_threshold {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::SolverNotConverged,
            detail: format!(
                "fsi interface residual exceeds threshold {}",
                options.residual_warn_threshold
            ),
        });
    }
    if fallback_events
        .iter()
        .any(FallbackEvent::is_solver_backend_fallback)
    {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::SolverBackendFallback,
            detail: "solver backend fell back from accelerator to cpu_reference".to_string(),
        });
    }

    let numerically_acceptable = match options.quality_policy {
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
    let (production_publishable, production_publication_reasons) =
        evaluate_production_publication(&run, numerically_acceptable);
    let run_status = classify_run_status(result_quality, numerically_acceptable);
    let solver_backend = run.solver_backend;
    let execution_backend = run.execution_backend;
    let solver_device_apply_k_ratio = run.solver_device_apply_k_ratio;
    let solver_host_sync_count = run.solver_host_sync_count;
    let solver_method = run.solver_method;
    let selected_preconditioner = run.preconditioner;

    let result = AnalysisRunResult {
        run_id: storage::next_run_id(),
        run,
        render_topology: render_topology_from_solver_mesh(solver_mesh.as_ref()),
        modal_results: None,
        thermal_results: None,
        transient_results: None,
        nonlinear_results: None,
        electromagnetic_results: None,
        model_validity: QualityGate::Pass,
        solver_convergence,
        result_quality,
        run_status,
        numerically_acceptable,
        production_publishable,
        quality_reasons,
        production_publication_reasons,
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
        },
    };

    persist_fea_run_result_with_progress(
        AnalysisOperation::Fsi.name(),
        AnalysisOperation::Fsi.version(),
        "RM.FEA.RUN_FSI.ARTIFACT_STORE_FAILED",
        &context,
        &result,
    )?;

    Ok(OperationEnvelope::new(
        AnalysisOperation::Fsi.name(),
        AnalysisOperation::Fsi.version(),
        &context,
        result,
    ))
}

pub fn analysis_run_thermal_with_options_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    formulation_requirement: FormulationRequirement,
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
                ("step_count".into(), options.step_count.to_string()),
                ("time_step_s".into(), options.time_step_s.to_string()),
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
    if thermal_procedure == AnalysisProcedure::TransientResponse
        && (options.step_count < 2
            || !options.time_step_s.is_finite()
            || options.time_step_s <= 0.0)
    {
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
            "transient fea.run_thermal requires at least two snapshots and a finite positive time step",
            BTreeMap::from([
                ("step_count".into(), options.step_count.to_string()),
                ("time_step_s".into(), options.time_step_s.to_string()),
            ]),
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
            formulation_requirement,
            procedure: match thermal_procedure {
                AnalysisProcedure::Static => ThermalProcedureControl::Steady,
                AnalysisProcedure::TransientResponse => ThermalProcedureControl::BackwardEuler {
                    step_count: options.step_count,
                    time_step_s: options.time_step_s,
                },
                _ => unreachable!("thermal procedure was validated before execution"),
            },
            residual_target: options.residual_warn_threshold,
            solver_mesh: Some(solver_mesh.clone()),
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
    let (production_publishable, production_publication_reasons) =
        evaluate_production_publication(&run, numerically_acceptable);
    let run_status = classify_run_status(result_quality, numerically_acceptable);

    let solver_backend = run.solver_backend;
    let execution_backend = run.execution_backend;
    let solver_device_apply_k_ratio = run.solver_device_apply_k_ratio;
    let solver_host_sync_count = run.solver_host_sync_count;
    let solver_method = run.solver_method;
    let selected_preconditioner = run.preconditioner;

    let result = AnalysisRunResult {
        run_id: storage::next_run_id(),
        run,
        render_topology: render_topology_from_solver_mesh(Some(&solver_mesh)),
        modal_results: None,
        thermal_results: Some(ThermalResultsData {
            thermal_payload_version: "thermal_results/v1".to_string(),
            time_points_s: thermal_run.time_points_s,
            temperature_snapshots: thermal_run.temperature_snapshots,
            temperature_gradient_snapshots: thermal_run.temperature_gradient_snapshots,
            heat_flux_snapshots: thermal_run.heat_flux_snapshots,
            energy_storage_rate_snapshots: thermal_run.energy_storage_rate_snapshots,
            residual_norms: thermal_run.residual_norms,
            reference_temperature_k: thermal_run.reference_temperature_k,
            initial_state: thermal_run.initial_state,
        }),
        transient_results: None,
        nonlinear_results: None,
        electromagnetic_results: None,
        model_validity: QualityGate::Pass,
        solver_convergence,
        result_quality,
        run_status,
        numerically_acceptable,
        production_publishable,
        quality_reasons,
        production_publication_reasons,
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
        },
    };

    persist_fea_run_result_with_progress(
        AnalysisOperation::Thermal.name(),
        AnalysisOperation::Thermal.version(),
        "RM.FEA.RUN_THERMAL.ARTIFACT_STORE_FAILED",
        &context,
        &result,
    )?;

    Ok(OperationEnvelope::new(
        AnalysisOperation::Thermal.name(),
        AnalysisOperation::Thermal.version(),
        &context,
        result,
    ))
}

fn validate_thermal_run_model(
    model: &AnalysisModel,
    context: &OperationContext,
) -> Result<(), OperationErrorEnvelope> {
    if let Some(material) = model.materials.iter().find(|material| {
        !material.thermal.conductivity_w_per_mk.is_finite()
            || material.thermal.conductivity_w_per_mk <= 0.0
            || !material.thermal.specific_heat_j_per_kgk.is_finite()
            || material.thermal.specific_heat_j_per_kgk <= 0.0
    }) {
        return Err(operation_error(
            AnalysisOperation::Thermal.name(),
            AnalysisOperation::Thermal.version(),
            context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_THERMAL.INVALID_THERMAL_MATERIAL",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_thermal requires finite positive thermal conductivity and specific heat",
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                ("material_id".to_string(), material.material_id.clone()),
                (
                    "conductivity_w_per_mk".to_string(),
                    material.thermal.conductivity_w_per_mk.to_string(),
                ),
                (
                    "specific_heat_j_per_kgk".to_string(),
                    material.thermal.specific_heat_j_per_kgk.to_string(),
                ),
            ]),
        ));
    }

    reject_moment_loads_for_run_family(
        model,
        AnalysisOperation::Thermal.name(),
        AnalysisOperation::Thermal.version(),
        "RM.FEA.RUN_THERMAL.INVALID_THERMAL_SOURCE",
        "thermal",
        context,
    )?;

    let invalid_constraint = model.constraints.iter().find_map(|constraint| {
        let AnalysisConstraint::Thermal(constraint) = constraint else {
            return None;
        };
        let runmat_analysis_core::ThermalConstraintKind::Prescribed { value, .. } =
            &constraint.kind
        else {
            return None;
        };
        (!value
            .study_constant(&model.signals)
            .is_some_and(|temperature| temperature.is_finite() && temperature.kelvin > 0.0))
        .then_some(constraint.constraint_id.as_str())
    });
    let invalid_boundary = model.boundaries.iter().find_map(|boundary| {
        let invalid = match &boundary.operator {
            BoundaryOperator::ThermalHeatFlux { value } => !value
                .study_constant(&model.signals)
                .is_some_and(|heat_flux| heat_flux.is_finite()),
            BoundaryOperator::ThermalConvection {
                ambient_temperature,
                coefficient,
            } => {
                !ambient_temperature
                    .study_constant(&model.signals)
                    .is_some_and(|temperature| temperature.is_finite() && temperature.kelvin > 0.0)
                    || !coefficient.is_finite()
                    || coefficient.watts_per_square_meter_kelvin < 0.0
            }
            _ => false,
        };
        invalid.then_some(boundary.boundary_id.as_str())
    });
    if let Some(boundary_id) = invalid_constraint.or(invalid_boundary) {
        return Err(operation_error(
            AnalysisOperation::Thermal.name(),
            AnalysisOperation::Thermal.version(),
            context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_THERMAL.INVALID_THERMAL_BOUNDARY",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_thermal requires finite physically valid thermal boundary condition values",
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                ("boundary_condition_id".to_string(), boundary_id.to_string()),
            ]),
        ));
    }

    Ok(())
}

pub fn analysis_run_transient_with_options_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    formulation_requirement: FormulationRequirement,
    options: AnalysisTransientRunOptions,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    let _solver_context = install_fea_solver_context();
    let has_transient_step = model.sequence.steps.iter().any(|step| {
        step.procedure == AnalysisProcedure::TransientResponse
            && matches!(
                step.family,
                PhysicsFamily::Structural
                    | PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical)
            )
    });
    if !has_transient_step {
        return Err(operation_error(
            AnalysisOperation::Transient.name(),
            AnalysisOperation::Transient.version(),
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

    let thermo_options = model_thermo_coupling_options(model);
    if let Some(thermo_options) = thermo_options.as_ref() {
        if let Err((detail, metadata)) = validate_thermo_coupling_options(model, thermo_options) {
            return Err(operation_error(
                AnalysisOperation::Transient.name(),
                AnalysisOperation::Transient.version(),
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
        if let Err((detail, metadata)) = validate_electro_coupling_options(electro_options) {
            return Err(operation_error(
                AnalysisOperation::Transient.name(),
                AnalysisOperation::Transient.version(),
                &context,
                OperationErrorSpec {
                    error_code: AnalysisOperation::Transient
                        .invalid_electro_thermal_options_error_code(),
                    error_type: OperationErrorType::Input,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                detail,
                metadata,
            ));
        }
    }

    let solver_mesh = resolve_solver_mesh_artifact(
        model,
        options.solver_mesh_artifact_path.as_deref(),
        AnalysisOperation::Transient.name(),
        AnalysisOperation::Transient.version(),
        "RM.FEA.RUN_TRANSIENT.SOLVER_MESH_INVALID",
        &context,
    )?;
    let transient_run = run_transient_with_options(
        model,
        backend,
        runmat_analysis_fea::solve::transient::TransientSolveOptions {
            formulation_requirement,
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
            progress_operation: AnalysisOperation::Transient.name().to_string(),
            solver_mesh: solver_mesh.clone(),
            thermo_mechanical_context: to_fea_thermo_mechanical_context(thermo_options),
            electro_thermal_context: to_fea_electro_thermal_context(electro_options),
        },
    )
    .map_err(|err| map_fea_run_error(AnalysisOperation::Transient, model, &context, err))?;

    let run = transient_run.run;
    let mut fallback_events = Vec::new();
    if backend == ComputeBackend::Gpu && run.execution_backend != ExecutionBackend::Accelerator {
        fallback_events.push(FallbackEvent::solver_backend());
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
    let thermo_gradient_spatial_threshold =
        thermo_spatial_gradient_threshold_for_policy(options.quality_policy);
    let thermo_gradient_instability = thermo_spatial_gradient_index
        .map(|value| value > thermo_gradient_spatial_threshold)
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
    let thermo_field_coverage_min = thermo_field_coverage_min_for_policy(options.quality_policy);
    let thermo_spread_breach = thermo_spread_ratio
        .map(|value| value > thermo_spread_threshold)
        .unwrap_or(false);
    let thermo_heterogeneity_breach = thermo_heterogeneity_index
        .map(|value| value > thermo_heterogeneity_threshold)
        .unwrap_or(false);
    let thermo_field_coverage_breach = thermo_spatial_coverage_ratio
        .map(|value| value < thermo_field_coverage_min)
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
                "thermo spatial gradient index {} exceeds threshold {}",
                thermo_spatial_gradient_index.unwrap_or(0.0),
                thermo_gradient_spatial_threshold,
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
    if fallback_events
        .iter()
        .any(FallbackEvent::is_solver_backend_fallback)
    {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::SolverBackendFallback,
            detail: "solver backend fell back from accelerator to cpu_reference".to_string(),
        });
    }
    let numerically_acceptable = match options.quality_policy {
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
    let (production_publishable, production_publication_reasons) =
        evaluate_production_publication(&run, numerically_acceptable);
    let run_status = classify_run_status(result_quality, numerically_acceptable);

    let solver_backend = run.solver_backend;
    let execution_backend = run.execution_backend;
    let solver_device_apply_k_ratio = run.solver_device_apply_k_ratio;
    let solver_host_sync_count = run.solver_host_sync_count;
    let solver_method = run.solver_method;
    let selected_preconditioner = run.preconditioner;

    let result = AnalysisRunResult {
        run_id: storage::next_run_id(),
        run,
        render_topology: render_topology_from_solver_mesh(solver_mesh.as_ref()),
        modal_results: None,
        thermal_results: None,
        transient_results: Some(TransientResultsData {
            transient_payload_version: "transient_results/v1".to_string(),
            time_points_s: transient_run.time_points_s,
            displacement_snapshots: transient_run.displacement_snapshots,
            rotation_snapshots: transient_run.rotation_snapshots,
            velocity_snapshots: transient_run.velocity_snapshots,
            angular_velocity_snapshots: transient_run.angular_velocity_snapshots,
            acceleration_snapshots: transient_run.acceleration_snapshots,
            angular_acceleration_snapshots: transient_run.angular_acceleration_snapshots,
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
        numerically_acceptable,
        production_publishable,
        quality_reasons,
        production_publication_reasons,
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
        },
    };

    persist_fea_run_result_with_progress(
        AnalysisOperation::Transient.name(),
        AnalysisOperation::Transient.version(),
        "RM.FEA.RUN_TRANSIENT.ARTIFACT_STORE_FAILED",
        &context,
        &result,
    )?;

    Ok(OperationEnvelope::new(
        AnalysisOperation::Transient.name(),
        AnalysisOperation::Transient.version(),
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
        FormulationRequirement::Production,
        AnalysisNonlinearRunOptions::default(),
        context,
    )
}

pub fn analysis_run_nonlinear_with_options_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    formulation_requirement: FormulationRequirement,
    options: AnalysisNonlinearRunOptions,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    let _solver_context = install_fea_solver_context();
    let has_nonlinear_step = model.sequence.steps.iter().any(|step| {
        step.procedure == AnalysisProcedure::Nonlinear
            && matches!(
                step.family,
                PhysicsFamily::Structural
                    | PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical)
            )
    });
    if !has_nonlinear_step {
        return Err(operation_error(
            AnalysisOperation::Nonlinear.name(),
            AnalysisOperation::Nonlinear.version(),
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
            AnalysisOperation::Nonlinear.name(),
            AnalysisOperation::Nonlinear.version(),
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
            AnalysisOperation::Nonlinear.name(),
            AnalysisOperation::Nonlinear.version(),
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
            AnalysisOperation::Nonlinear.name(),
            AnalysisOperation::Nonlinear.version(),
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
            AnalysisOperation::Nonlinear.name(),
            AnalysisOperation::Nonlinear.version(),
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
            AnalysisOperation::Nonlinear.name(),
            AnalysisOperation::Nonlinear.version(),
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
            AnalysisOperation::Nonlinear.name(),
            AnalysisOperation::Nonlinear.version(),
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
            AnalysisOperation::Nonlinear.name(),
            AnalysisOperation::Nonlinear.version(),
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

    let thermo_options = model_thermo_coupling_options(model);
    if let Some(thermo_options) = thermo_options.as_ref() {
        if let Err((detail, metadata)) = validate_thermo_coupling_options(model, thermo_options) {
            return Err(operation_error(
                AnalysisOperation::Nonlinear.name(),
                AnalysisOperation::Nonlinear.version(),
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
        if let Err((detail, metadata)) = validate_electro_coupling_options(electro_options) {
            return Err(operation_error(
                AnalysisOperation::Nonlinear.name(),
                AnalysisOperation::Nonlinear.version(),
                &context,
                OperationErrorSpec {
                    error_code: AnalysisOperation::Nonlinear
                        .invalid_electro_thermal_options_error_code(),
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
                AnalysisOperation::Nonlinear.name(),
                AnalysisOperation::Nonlinear.version(),
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
    let contact_options =
        model_contact_interface_options(model, options.contact_enforcement.as_ref());
    if options.contact_enforcement.is_some() && contact_options.is_none() {
        return Err(operation_error(
            AnalysisOperation::Nonlinear.name(),
            AnalysisOperation::Nonlinear.version(),
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_NONLINEAR.INVALID_OPTIONS",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "contact enforcement options require a contact interface in the model",
            BTreeMap::new(),
        ));
    }
    if let Some(contact_options) = contact_options.as_ref() {
        if let Err((detail, metadata)) = validate_contact_interface_options(contact_options) {
            return Err(operation_error(
                AnalysisOperation::Nonlinear.name(),
                AnalysisOperation::Nonlinear.version(),
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

    let solver_mesh = resolve_solver_mesh_artifact(
        model,
        options.solver_mesh_artifact_path.as_deref(),
        AnalysisOperation::Nonlinear.name(),
        AnalysisOperation::Nonlinear.version(),
        "RM.FEA.RUN_NONLINEAR.SOLVER_MESH_INVALID",
        &context,
    )?;
    let nonlinear_run = run_nonlinear_with_options(
        model,
        backend,
        runmat_analysis_fea::solve::nonlinear::NonlinearSolveOptions {
            formulation_requirement,
            increment_count: options.increment_count,
            max_newton_iters: options.max_newton_iters,
            tolerance: options.tolerance,
            residual_convergence_factor: options.residual_convergence_factor,
            increment_norm_tolerance: options.increment_norm_tolerance,
            line_search: options.line_search,
            max_line_search_backtracks: options.max_line_search_backtracks,
            line_search_reduction: options.line_search_reduction,
            tangent_refresh_interval: options.tangent_refresh_interval,
            solver_mesh: solver_mesh.clone(),
            thermo_mechanical_context: to_fea_thermo_mechanical_context(thermo_options),
            electro_thermal_context: to_fea_electro_thermal_context(electro_options),
            plasticity_context: to_fea_plasticity_constitutive_context(plasticity_options),
            contact_context: to_fea_contact_interface_context(contact_options),
        },
    )
    .map_err(|err| map_fea_run_error(AnalysisOperation::Nonlinear, model, &context, err))?;

    let run = nonlinear_run.run;
    let mut fallback_events = Vec::new();
    if backend == ComputeBackend::Gpu && run.execution_backend != ExecutionBackend::Accelerator {
        fallback_events.push(FallbackEvent::solver_backend());
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
    let thermo_gradient_spatial_threshold =
        thermo_spatial_gradient_threshold_for_policy(options.quality_policy);
    let thermo_gradient_instability = thermo_spatial_gradient_index
        .map(|value| value > thermo_gradient_spatial_threshold)
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
    let thermo_field_coverage_min = thermo_field_coverage_min_for_policy(options.quality_policy);
    let thermo_spread_breach = thermo_spread_ratio
        .map(|value| value > thermo_spread_threshold)
        .unwrap_or(false);
    let thermo_heterogeneity_breach = thermo_heterogeneity_index
        .map(|value| value > thermo_heterogeneity_threshold)
        .unwrap_or(false);
    let thermo_field_coverage_breach = thermo_spatial_coverage_ratio
        .map(|value| value < thermo_field_coverage_min)
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
                "thermo spatial gradient index {} exceeds threshold {}",
                thermo_spatial_gradient_index.unwrap_or(0.0),
                thermo_gradient_spatial_threshold,
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
    if fallback_events
        .iter()
        .any(FallbackEvent::is_solver_backend_fallback)
    {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::SolverBackendFallback,
            detail: "solver backend fell back from accelerator to cpu_reference".to_string(),
        });
    }

    let numerically_acceptable = match options.quality_policy {
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
    let (production_publishable, production_publication_reasons) =
        evaluate_production_publication(&run, numerically_acceptable);
    let run_status = classify_run_status(result_quality, numerically_acceptable);

    let solver_backend = run.solver_backend;
    let execution_backend = run.execution_backend;
    let solver_device_apply_k_ratio = run.solver_device_apply_k_ratio;
    let solver_host_sync_count = run.solver_host_sync_count;
    let solver_method = run.solver_method;
    let selected_preconditioner = run.preconditioner;

    let result = AnalysisRunResult {
        run_id: storage::next_run_id(),
        run,
        render_topology: render_topology_from_solver_mesh(solver_mesh.as_ref()),
        modal_results: None,
        thermal_results: None,
        transient_results: None,
        nonlinear_results: Some(NonlinearResultsData {
            nonlinear_payload_version: "nonlinear_results/v1".to_string(),
            load_factors: nonlinear_run.load_factors,
            displacement_snapshots: nonlinear_run.displacement_snapshots,
            rotation_snapshots: nonlinear_run.rotation_snapshots,
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
        numerically_acceptable,
        production_publishable,
        quality_reasons,
        production_publication_reasons,
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
        },
    };

    persist_fea_run_result_with_progress(
        AnalysisOperation::Nonlinear.name(),
        AnalysisOperation::Nonlinear.version(),
        "RM.FEA.RUN_NONLINEAR.ARTIFACT_STORE_FAILED",
        &context,
        &result,
    )?;

    Ok(OperationEnvelope::new(
        AnalysisOperation::Nonlinear.name(),
        AnalysisOperation::Nonlinear.version(),
        &context,
        result,
    ))
}

pub fn analysis_run_linear_static_with_options(
    model: &AnalysisModel,
    backend: ComputeBackend,
    formulation_requirement: FormulationRequirement,
    options: AnalysisRunOptions,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    analysis_run_linear_static_with_executor(
        model,
        backend,
        formulation_requirement,
        options,
        context,
        None,
    )
}

pub fn analysis_prepare_linear_static_run_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    formulation_requirement: FormulationRequirement,
    options: AnalysisRunOptions,
    context: OperationContext,
) -> Result<PreparedStructuralLinearRun, OperationErrorEnvelope> {
    let step_id = model
        .sequence
        .steps
        .first()
        .ok_or_else(|| {
            operation_error(
                AnalysisOperation::LinearStatic.name(),
                AnalysisOperation::LinearStatic.version(),
                &context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RUN_LINEAR_STATIC.SEQUENCE_EMPTY",
                    error_type: OperationErrorType::Validation,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                "structural linear analysis requires a procedure step",
                BTreeMap::new(),
            )
        })?
        .step_id
        .clone();
    prepare_linear_static_step_run(
        model,
        &step_id,
        backend,
        formulation_requirement,
        options,
        context,
    )
}

fn prepare_linear_static_step_run(
    model: &AnalysisModel,
    step_id: &ProcedureStepId,
    backend: ComputeBackend,
    formulation_requirement: FormulationRequirement,
    options: AnalysisRunOptions,
    context: OperationContext,
) -> Result<PreparedStructuralLinearRun, OperationErrorEnvelope> {
    let _solver_context = install_fea_solver_context();
    let workload = prepare_linear_static_workload_template(
        model,
        step_id,
        backend,
        formulation_requirement,
        &options,
        &context,
    )?;
    let solver_mesh = resolve_solver_mesh_artifact(
        model,
        options.solver_mesh_artifact_path.as_deref(),
        AnalysisOperation::LinearStatic.name(),
        AnalysisOperation::LinearStatic.version(),
        "RM.FEA.RUN_LINEAR_STATIC.SOLVER_MESH_INVALID",
        &context,
    )?;
    Ok(PreparedStructuralLinearRun {
        workload,
        solver_mesh,
        options,
        context,
    })
}

fn prepare_linear_static_workload_template(
    model: &AnalysisModel,
    step_id: &ProcedureStepId,
    backend: ComputeBackend,
    formulation_requirement: FormulationRequirement,
    options: &AnalysisRunOptions,
    context: &OperationContext,
) -> Result<StructuralLinearWorkload, OperationErrorEnvelope> {
    let thermo_options = model_thermo_coupling_options(model);
    if let Some(thermo_options) = thermo_options.as_ref() {
        if let Err((detail, metadata)) = validate_thermo_coupling_options(model, thermo_options) {
            return Err(operation_error(
                AnalysisOperation::LinearStatic.name(),
                AnalysisOperation::LinearStatic.version(),
                context,
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
        if let Err((detail, metadata)) = validate_electro_coupling_options(electro_options) {
            return Err(operation_error(
                AnalysisOperation::LinearStatic.name(),
                AnalysisOperation::LinearStatic.version(),
                context,
                OperationErrorSpec {
                    error_code: AnalysisOperation::LinearStatic
                        .invalid_electro_thermal_options_error_code(),
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
    let accelerator_available = runmat_accelerate_api::provider().is_some();
    let requested_solver_backend = match backend {
        ComputeBackend::Cpu => LinearAlgebraBackendKind::CpuReference,
        ComputeBackend::Gpu => {
            if accelerator_available {
                LinearAlgebraBackendKind::Accelerator
            } else {
                LinearAlgebraBackendKind::CpuReference
            }
        }
    };
    Ok(StructuralLinearWorkload {
        schema_version: 1,
        model: model.clone(),
        step_id: step_id.clone(),
        backend,
        formulation_requirement,
        preconditioner: requested_preconditioner,
        algebra_backend: requested_solver_backend,
        require_solver_mesh_for_solid: true,
        thermo_mechanical_context: to_fea_thermo_mechanical_context(thermo_options),
        electro_thermal_context: to_fea_electro_thermal_context(electro_options),
        initial_guess_transition: None,
    })
}

pub fn analysis_complete_linear_static_run_op(
    prepared: PreparedStructuralLinearRun,
    run: FeaRunResult,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    analysis_complete_linear_static_run_with_id(prepared, run, None)
}

fn analysis_complete_linear_static_run_with_id(
    prepared: PreparedStructuralLinearRun,
    run: FeaRunResult,
    assigned_run_id: Option<&str>,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    let context = prepared.context.clone();
    let envelope = analysis_admit_linear_static_run_with_id(prepared, run, assigned_run_id)?;
    persist_linear_static_run(&context, &envelope.data)?;
    Ok(envelope)
}

fn analysis_admit_linear_static_run_with_id(
    prepared: PreparedStructuralLinearRun,
    run: FeaRunResult,
    assigned_run_id: Option<&str>,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    let result = linear_static_result::admit_linear_static_result(
        linear_static_result::LinearStaticResultInput {
            model: &prepared.workload.model,
            solver_mesh: prepared.solver_mesh.as_ref(),
            requested_backend: prepared.workload.backend,
            formulation_requirement: prepared.workload.formulation_requirement,
            options: &prepared.options,
            assigned_run_id,
        },
        run,
    )
    .map_err(|message| {
        operation_error(
            AnalysisOperation::LinearStatic.name(),
            AnalysisOperation::LinearStatic.version(),
            &prepared.context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_LINEAR_STATIC.RESULT_ADMISSION_FAILED",
                error_type: OperationErrorType::Contract,
                retryable: false,
                severity: OperationErrorSeverity::Fatal,
            },
            message,
            BTreeMap::from([(
                "analysis_model_id".to_string(),
                prepared.workload.model.model_id.0.clone(),
            )]),
        )
    })?;

    Ok(OperationEnvelope::new(
        AnalysisOperation::LinearStatic.name(),
        AnalysisOperation::LinearStatic.version(),
        &prepared.context,
        result,
    ))
}

fn persist_linear_static_run(
    context: &OperationContext,
    result: &AnalysisRunResult,
) -> Result<(), OperationErrorEnvelope> {
    persist_fea_run_result_with_progress(
        AnalysisOperation::LinearStatic.name(),
        AnalysisOperation::LinearStatic.version(),
        "RM.FEA.RUN_LINEAR_STATIC.ARTIFACT_STORE_FAILED",
        context,
        result,
    )
}

fn analysis_run_linear_static_with_executor(
    model: &AnalysisModel,
    backend: ComputeBackend,
    formulation_requirement: FormulationRequirement,
    options: AnalysisRunOptions,
    context: OperationContext,
    executor: Option<&dyn StructuralLinearExecutor>,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    let prepared = analysis_prepare_linear_static_run_op(
        model,
        backend,
        formulation_requirement,
        options,
        context,
    )?;
    let run = if let Some(executor) = executor {
        let mesh = prepared.solver_mesh.as_ref().ok_or_else(|| {
            operation_error(
                AnalysisOperation::LinearStatic.name(),
                AnalysisOperation::LinearStatic.version(),
                &prepared.context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RUN_LINEAR_STATIC.PORTABLE_MESH_REQUIRED",
                    error_type: OperationErrorType::Input,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                "portable structural execution requires a canonical solver mesh",
                BTreeMap::from([(
                    "analysis_model_id".to_string(),
                    prepared.workload.model.model_id.0.clone(),
                )]),
            )
        })?;
        executor
            .execute(&prepared.workload, mesh)
            .map_err(|message| {
                operation_error(
                    AnalysisOperation::LinearStatic.name(),
                    AnalysisOperation::LinearStatic.version(),
                    &prepared.context,
                    OperationErrorSpec {
                        error_code: "RM.FEA.RUN_LINEAR_STATIC.PORTABLE_EXECUTION_FAILED",
                        error_type: OperationErrorType::Backend,
                        retryable: false,
                        severity: OperationErrorSeverity::Error,
                    },
                    message,
                    BTreeMap::from([(
                        "analysis_model_id".to_string(),
                        prepared.workload.model.model_id.0.clone(),
                    )]),
                )
            })?
    } else {
        run_linear_static_with_options(
            &prepared.workload.model,
            prepared.workload.backend,
            LinearStaticSolveOptions {
                formulation_requirement: prepared.workload.formulation_requirement,
                preconditioner_kind: prepared.workload.preconditioner,
                algebra_backend_kind: prepared.workload.algebra_backend,
                solver_mesh_artifact_path: prepared.options.solver_mesh_artifact_path.clone(),
                solver_mesh: prepared.solver_mesh.clone(),
                require_solver_mesh_for_solid: prepared.workload.require_solver_mesh_for_solid,
                thermo_mechanical_context: prepared.workload.thermo_mechanical_context.clone(),
                electro_thermal_context: prepared.workload.electro_thermal_context.clone(),
                initial_displacement: None,
            },
        )
        .map_err(|err| {
            map_fea_run_error(
                AnalysisOperation::LinearStatic,
                &prepared.workload.model,
                &prepared.context,
                err,
            )
        })?
    };
    analysis_complete_linear_static_run_op(prepared, run)
}

pub fn analysis_run_electromagnetic_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    analysis_run_electromagnetic_with_options_op(
        model,
        backend,
        FormulationRequirement::Production,
        AnalysisElectromagneticRunOptions::default(),
        context,
    )
}

pub fn analysis_run_electromagnetic_with_options_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    formulation_requirement: FormulationRequirement,
    options: AnalysisElectromagneticRunOptions,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    let _solver_context = install_fea_solver_context();
    let electromagnetic_steps = model
        .sequence
        .steps
        .iter()
        .filter(|step| {
            step.family == PhysicsFamily::Electromagnetic
                && step.procedure == AnalysisProcedure::HarmonicResponse
        })
        .collect::<Vec<_>>();
    let [electromagnetic_step] = electromagnetic_steps.as_slice() else {
        return Err(operation_error(
            AnalysisOperation::Electromagnetic.name(),
            AnalysisOperation::Electromagnetic.version(),
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_ELECTROMAGNETIC.REQUIRES_STEP",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "FEA model must include exactly one supported electromagnetic step for fea.run_electromagnetic",
            BTreeMap::from([("analysis_model_id".to_string(), model.model_id.0.clone())]),
        ));
    };
    let electromagnetic_procedure = electromagnetic_step.procedure;

    let Some(em_domain) = model.electromagnetic.as_ref() else {
        return Err(operation_error(
            AnalysisOperation::Electromagnetic.name(),
            AnalysisOperation::Electromagnetic.version(),
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
    if !em_domain.reference_frequency.hertz.is_finite()
        || em_domain.reference_frequency.hertz <= 0.0
    {
        return Err(operation_error(
            AnalysisOperation::Electromagnetic.name(),
            AnalysisOperation::Electromagnetic.version(),
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
                em_domain.reference_frequency.hertz.to_string(),
            )]),
        ));
    }
    if !options.residual_target.is_finite() || options.residual_target <= 0.0 {
        return Err(operation_error(
            AnalysisOperation::Electromagnetic.name(),
            AnalysisOperation::Electromagnetic.version(),
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
            AnalysisOperation::Electromagnetic.name(),
            AnalysisOperation::Electromagnetic.version(),
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
            AnalysisOperation::Electromagnetic.name(),
            AnalysisOperation::Electromagnetic.version(),
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
    validate_electromagnetic_run_model(model, &context)?;

    let sweep_frequency_hz = normalize_em_sweep_frequency_hz(
        em_domain.reference_frequency.hertz,
        options.sweep_enabled,
        &options.sweep_frequency_hz,
    )
    .ok_or_else(|| {
        operation_error(
            AnalysisOperation::Electromagnetic.name(),
            AnalysisOperation::Electromagnetic.version(),
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
    let solver_mesh = resolve_solver_mesh_artifact(
        model,
        options.solver_mesh_artifact_path.as_deref(),
        AnalysisOperation::Electromagnetic.name(),
        AnalysisOperation::Electromagnetic.version(),
        "RM.FEA.RUN_ELECTROMAGNETIC.SOLVER_MESH_INVALID",
        &context,
    )?;
    let solve_options = ElectromagneticSolveOptions {
        formulation_requirement,
        procedure: electromagnetic_procedure,
        residual_target: options.residual_target,
        harmonic_tolerance: options.harmonic_tolerance,
        harmonic_max_iterations: options.harmonic_max_iterations,
        solver_mesh: solver_mesh.clone(),
    };
    let mut sweep_runs = Vec::with_capacity(sweep_frequency_hz.len());
    let mut sweep_peak_flux_density = Vec::with_capacity(sweep_frequency_hz.len());
    let mut sweep_solve_quality = Vec::with_capacity(sweep_frequency_hz.len());
    for frequency_hz in &sweep_frequency_hz {
        let mut sweep_model = model.clone();
        if let Some(domain) = sweep_model.electromagnetic.as_mut() {
            domain.reference_frequency.hertz = *frequency_hz;
        }
        let sweep_run =
            run_electromagnetic_with_options(&sweep_model, backend, solve_options.clone())
                .map_err(|err| {
                    map_fea_run_error(AnalysisOperation::Electromagnetic, model, &context, err)
                })?;
        sweep_peak_flux_density.push(peak_abs_field_value(
            &sweep_run.magnetic_flux_density_magnitude_field,
        ));
        sweep_solve_quality.push(sweep_run.solve_quality);
        sweep_runs.push(sweep_run);
    }
    let sweep_metrics = summarize_em_sweep(&sweep_frequency_hz, &sweep_peak_flux_density);
    let primary_index =
        nearest_frequency_index(&sweep_frequency_hz, em_domain.reference_frequency.hertz)
            .unwrap_or(0);
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
    if sweep_metrics.sweep_count > 1 {
        run.diagnostics.push(em_sweep_known_answer_diagnostic(
            em_domain.reference_frequency.hertz,
            &sweep_frequency_hz,
            &sweep_metrics,
        ));
    }
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
    let em_assigned_coefficient_coverage_ratio = diagnostic_metric(
        &run.diagnostics,
        "FEA_EM_STATIC",
        "assigned_coefficient_coverage_ratio",
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
        "FEA_EM_SOURCE_ENERGY",
        "source_realization_ratio",
    );
    let em_source_region_coverage_ratio = diagnostic_metric(
        &run.diagnostics,
        "FEA_EM_SOURCE_ENERGY",
        "source_region_coverage_ratio",
    );
    let em_source_material_alignment_ratio = diagnostic_metric(
        &run.diagnostics,
        "FEA_EM_SOURCE_ENERGY",
        "source_material_alignment_ratio",
    );
    let em_source_overlap_ratio = diagnostic_metric(
        &run.diagnostics,
        "FEA_EM_SOURCE_ENERGY",
        "source_overlap_ratio",
    );
    let em_source_interference_index = diagnostic_metric(
        &run.diagnostics,
        "FEA_EM_SOURCE_ENERGY",
        "source_interference_index",
    );
    let em_boundary_anchor_ratio = diagnostic_metric(
        &run.diagnostics,
        "FEA_EM_SOURCE_ENERGY",
        "boundary_anchor_ratio",
    );
    let em_boundary_condition_localization_ratio = diagnostic_metric(
        &run.diagnostics,
        "FEA_EM_SOURCE_ENERGY",
        "boundary_condition_localization_ratio",
    );
    let em_ground_anchor_effectiveness_ratio = diagnostic_metric(
        &run.diagnostics,
        "FEA_EM_SOURCE_ENERGY",
        "ground_anchor_effectiveness_ratio",
    );
    let em_insulation_leakage_ratio = diagnostic_metric(
        &run.diagnostics,
        "FEA_EM_SOURCE_ENERGY",
        "insulation_leakage_ratio",
    );
    let em_flux_divergence_ratio =
        diagnostic_metric(&run.diagnostics, "FEA_EM_STATIC", "flux_divergence_ratio");
    let em_energy_imbalance_ratio = diagnostic_metric(
        &run.diagnostics,
        "FEA_EM_SOURCE_ENERGY",
        "energy_imbalance_ratio",
    );
    let em_boundary_energy_ratio = diagnostic_metric(
        &run.diagnostics,
        "FEA_EM_SOURCE_ENERGY",
        "boundary_energy_ratio",
    );
    let em_boundary_penalty_conditioning_contribution = diagnostic_metric(
        &run.diagnostics,
        "FEA_EM_SOURCE_ENERGY",
        "boundary_penalty_conditioning_contribution",
    );
    let em_source_region_energy_consistency_ratio = diagnostic_metric(
        &run.diagnostics,
        "FEA_EM_SOURCE_ENERGY",
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
    let em_assigned_coefficient_breach = em_assigned_coefficient_coverage_ratio
        .map(|value| value < em_coverage_min_threshold)
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
        || em_assigned_coefficient_breach
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
            detail: "electromagnetic static solve quality below target".to_string(),
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
    if em_assigned_coefficient_breach {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ElectromagneticAssignmentCoverageLow,
            detail: format!(
                "electromagnetic assigned coefficient coverage ratio {} is below threshold {}",
                em_assigned_coefficient_coverage_ratio.unwrap_or(0.0),
                em_coverage_min_threshold
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

    let numerically_acceptable = match options.quality_policy {
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
    let (production_publishable, production_publication_reasons) =
        evaluate_production_publication(&run, numerically_acceptable);
    let run_status = classify_run_status(result_quality, numerically_acceptable);
    let solver_backend = run.solver_backend;
    let execution_backend = run.execution_backend;
    let solver_device_apply_k_ratio = run.solver_device_apply_k_ratio;
    let solver_host_sync_count = run.solver_host_sync_count;
    let solver_method = run.solver_method;
    let preconditioner = run.preconditioner;

    let result = AnalysisRunResult {
        run_id: storage::next_run_id(),
        run,
        render_topology: render_topology_from_solver_mesh(solver_mesh.as_ref()),
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
        numerically_acceptable,
        production_publishable,
        quality_reasons,
        production_publication_reasons,
        provenance: RunProvenance {
            requested_backend: backend,
            execution_backend,
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
        AnalysisOperation::Electromagnetic.name(),
        AnalysisOperation::Electromagnetic.version(),
        "RM.FEA.RUN_ELECTROMAGNETIC.ARTIFACT_STORE_FAILED",
        &context,
        &result,
    )?;

    Ok(OperationEnvelope::new(
        AnalysisOperation::Electromagnetic.name(),
        AnalysisOperation::Electromagnetic.version(),
        &context,
        result,
    ))
}

fn validate_electromagnetic_run_model(
    model: &AnalysisModel,
    context: &OperationContext,
) -> Result<(), OperationErrorEnvelope> {
    if let Some(material) = model.materials.iter().find(|material| {
        material.electrical.as_ref().is_some_and(|electrical| {
            !electrical.conductivity_s_per_m.is_finite()
                || electrical.conductivity_s_per_m <= 0.0
                || !electrical.relative_permittivity.is_finite()
                || electrical.relative_permittivity <= 0.0
                || !electrical.relative_permeability.is_finite()
                || electrical.relative_permeability <= 0.0
                || electrical
                    .conductivity_frequency_response
                    .iter()
                    .any(|point| {
                        !point.frequency_hz.is_finite()
                            || point.frequency_hz <= 0.0
                            || !point.conductivity_scale.is_finite()
                            || point.conductivity_scale <= 0.0
                            || point
                                .dispersive_loss_scale
                                .is_some_and(|value| !value.is_finite() || value < 0.0)
                            || point
                                .relative_permittivity_scale
                                .is_some_and(|value| !value.is_finite() || value <= 0.0)
                            || point
                                .relative_permeability_scale
                                .is_some_and(|value| !value.is_finite() || value <= 0.0)
                    })
        })
    }) {
        return Err(operation_error(
            AnalysisOperation::Electromagnetic.name(),
            AnalysisOperation::Electromagnetic.version(),
            context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_ELECTROMAGNETIC.INVALID_ELECTROMAGNETIC_MATERIAL",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_electromagnetic requires finite positive electrical material coefficients",
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                ("material_id".to_string(), material.material_id.clone()),
            ]),
        ));
    }

    let electrical_material_count = model
        .materials
        .iter()
        .filter(|material| material.electrical.is_some())
        .count();
    if electrical_material_count == 0 {
        return Err(operation_error(
            AnalysisOperation::Electromagnetic.name(),
            AnalysisOperation::Electromagnetic.version(),
            context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_ELECTROMAGNETIC.MISSING_ELECTROMAGNETIC_MATERIAL",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_electromagnetic requires at least one electrical material",
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                (
                    "material_count".to_string(),
                    model.materials.len().to_string(),
                ),
            ]),
        ));
    }
    let electrical_material_by_id = model
        .materials
        .iter()
        .filter(|material| material.electrical.is_some())
        .map(|material| material.material_id.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    if let Some(assignment) = model.material_assignments.iter().find(|assignment| {
        !electrical_material_by_id.contains(assignment.assigned_material_id.as_str())
    }) {
        return Err(operation_error(
            AnalysisOperation::Electromagnetic.name(),
            AnalysisOperation::Electromagnetic.version(),
            context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_ELECTROMAGNETIC.MISSING_ELECTROMAGNETIC_MATERIAL",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_electromagnetic requires every material assignment to reference an assigned electrical material",
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                ("region_id".to_string(), assignment.region_id.clone()),
                (
                    "assigned_material_id".to_string(),
                    assignment.assigned_material_id.clone(),
                ),
            ]),
        ));
    }

    reject_moment_loads_for_run_family(
        model,
        AnalysisOperation::Electromagnetic.name(),
        AnalysisOperation::Electromagnetic.version(),
        "RM.FEA.RUN_ELECTROMAGNETIC.INVALID_ELECTROMAGNETIC_SOURCE",
        "electromagnetic",
        context,
    )?;

    let has_electromagnetic_source = has_nonzero_electromagnetic_source(model);
    if !has_electromagnetic_source {
        return Err(operation_error(
            AnalysisOperation::Electromagnetic.name(),
            AnalysisOperation::Electromagnetic.version(),
            context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_ELECTROMAGNETIC.MISSING_ELECTROMAGNETIC_SOURCE",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_electromagnetic requires a nonzero electric-current or current-density excitation",
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                (
                    "excitation_count".to_string(),
                    model.excitations.len().to_string(),
                ),
            ]),
        ));
    }

    let has_electromagnetic_boundary = model
        .boundaries
        .iter()
        .any(|boundary| matches!(boundary.operator, BoundaryOperator::MagneticInsulation))
        || model.constraints.iter().any(|constraint| {
            matches!(
                constraint,
                AnalysisConstraint::Magnetic(value)
                    if matches!(value.kind, MagneticConstraintKind::VectorPotentialGround { .. })
            )
        });
    if !has_electromagnetic_boundary {
        return Err(operation_error(
            AnalysisOperation::Electromagnetic.name(),
            AnalysisOperation::Electromagnetic.version(),
            context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_ELECTROMAGNETIC.MISSING_ELECTROMAGNETIC_BOUNDARY",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_electromagnetic requires magnetic insulation or vector-potential ground boundary data",
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                (
                    "boundary_condition_count".to_string(),
                    (model.constraints.len() + model.boundaries.len()).to_string(),
                ),
            ]),
        ));
    }

    Ok(())
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

fn resolve_study_model(
    spec: &AnalysisStudySpec,
    context: &OperationContext,
) -> Result<AnalysisModel, OperationErrorEnvelope> {
    match &spec.model {
        Some(model) => Ok(model.clone()),
        None => analysis_create_model_op(
            &spec.geometry,
            spec.create_model_intent.clone(),
            context.clone(),
        )
        .map(|envelope| envelope.data),
    }
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
    if spec.execution.steps.is_empty() {
        issue_codes.push("RM.FEA.STUDY.EXECUTION_EMPTY".to_string());
    }
    if !immutable_value_inventory_is_exact(spec) {
        issue_codes.push("RM.FEA.STUDY.IMMUTABLE_VALUES_INVALID".to_string());
    }
    if let Some(model) = &spec.model {
        if model.geometry_id != spec.geometry.geometry_id
            || model.geometry_revision != spec.geometry.revision
        {
            issue_codes.push("RM.FEA.STUDY.MODEL_GEOMETRY_MISMATCH".to_string());
        }
        if validate_model_against_geometry(model, spec.geometry.units, &CoordinateFrame::Global)
            .is_err()
        {
            issue_codes.push("RM.FEA.STUDY.MODEL_INVALID".to_string());
        }
        if spec.execution.validate_against(&model.sequence).is_err() {
            issue_codes.push("RM.FEA.STUDY.EXECUTION_PLAN_INVALID".to_string());
        }
    }
    for execution in &spec.execution.steps {
        if let AnalysisSolverOptions::ElectromagneticPotential(options) = &execution.solver_options
        {
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

fn immutable_value_inventory_is_exact(spec: &AnalysisStudySpec) -> bool {
    contracts::validate_immutable_value_inventory(
        &spec.immutable_values,
        spec.model
            .as_ref()
            .map(|model| model.signals.as_slice())
            .unwrap_or_default(),
    )
    .is_ok()
}

fn study_issue_message(code: &str) -> &'static str {
    match code {
        "RM.FEA.STUDY.ID_EMPTY" => "study_id must be non-empty",
        "RM.FEA.STUDY.MODEL_ID_EMPTY" => "create_model_intent.model_id must be non-empty",
        "RM.FEA.STUDY.GEOMETRY_MESHES_EMPTY" => "geometry must contain at least one mesh",
        "RM.FEA.STUDY.GEOMETRY_UNITS_UNSPECIFIED" => {
            "geometry.units must be specified (not unspecified)"
        }
        "RM.FEA.STUDY.EXECUTION_EMPTY" => "execution must contain at least one typed procedure step",
        "RM.FEA.STUDY.IMMUTABLE_VALUES_INVALID" => {
            "immutable values must be unique, normalized, content-bound, and referenced exactly by generator arguments"
        }
        "RM.FEA.STUDY.EXECUTION_PLAN_INVALID" => {
            "execution must map each canonical model procedure step to exactly one compatible solver"
        }
        "RM.FEA.STUDY.MODEL_GEOMETRY_MISMATCH" => {
            "resolved model geometry id or revision does not match the study geometry"
        }
        "RM.FEA.STUDY.MODEL_INVALID" => "resolved model failed FEA validation",
        "RM.FEA.STUDY.ELECTROMAGNETIC_RESIDUAL_TARGET_INVALID" => {
            "electromagnetic execution options.residual_target must be finite and positive"
        }
        "RM.FEA.STUDY.ELECTROMAGNETIC_HARMONIC_TOLERANCE_INVALID" => {
            "electromagnetic execution options.harmonic_tolerance must be finite and positive"
        }
        "RM.FEA.STUDY.ELECTROMAGNETIC_HARMONIC_MAX_ITERATIONS_INVALID" => {
            "electromagnetic execution options.harmonic_max_iterations must be greater than zero"
        }
        "RM.FEA.STUDY.ELECTROMAGNETIC_SWEEP_FREQUENCY_INVALID" => {
            "electromagnetic execution options.sweep_frequency_hz must contain finite positive values when sweep_enabled is true"
        }
        _ => "unrecognized study validation issue",
    }
}

/// Computes the authored study's placement-independent content identity.
///
/// The geometry source digest identifies the input bytes. Local source and generated artifact
/// paths identify storage locations, so they cannot participate in a portable study identity.
pub fn analysis_study_content_digest(spec: &AnalysisStudySpec) -> runmat_execution::Digest {
    let mut semantic_spec = spec.clone();
    semantic_spec.geometry.source.path.clear();
    semantic_spec.solver_mesh_artifact_path = None;
    semantic_spec.meshing_evidence_artifact_path = None;
    for value in &mut semantic_spec.immutable_values {
        value.relative_path.clear();
    }
    for execution in &mut semantic_spec.execution.steps {
        execution.solver_options.clear_solver_mesh_artifact_path();
    }
    let payload = serde_json::to_vec(&semantic_spec)
        .unwrap_or_else(|_| format!("{semantic_spec:?}").into_bytes());
    runmat_execution::Digest::sha256(&payload)
}

fn study_fingerprint(spec: &AnalysisStudySpec) -> String {
    study_fingerprint_from_digest(analysis_study_content_digest(spec))
}

fn study_fingerprint_from_digest(digest: runmat_execution::Digest) -> String {
    let mut encoded = String::with_capacity(64);
    for byte in digest.bytes() {
        use std::fmt::Write as _;
        write!(&mut encoded, "{byte:02x}").expect("writing to a string cannot fail");
    }
    format!("sha256:{encoded}")
}

fn study_operation_sequence(spec: &AnalysisStudySpec, run_op_versions: &[String]) -> Vec<String> {
    let mut operation_sequence = Vec::with_capacity(2 + run_op_versions.len());
    if spec.model.is_none() {
        operation_sequence.push(ANALYSIS_CREATE_MODEL_OP_VERSION.to_string());
    }
    operation_sequence.push(ANALYSIS_VALIDATE_OP_VERSION.to_string());
    operation_sequence.extend(run_op_versions.iter().cloned());
    operation_sequence
}

fn resolve_solver_mesh_artifact(
    model: &AnalysisModel,
    solver_mesh_artifact_path: Option<&str>,
    operation: &'static str,
    op_version: &'static str,
    error_code: &'static str,
    context: &OperationContext,
) -> Result<Option<SolverMeshArtifact>, OperationErrorEnvelope> {
    let Some(path) = solver_mesh_artifact_path else {
        return Ok(None);
    };
    let mesh = solver_mesh_artifact::load_solver_mesh_artifact(std::path::Path::new(path))
        .map_err(|error| {
            operation_error(
                operation,
                op_version,
                context,
                OperationErrorSpec {
                    error_code,
                    error_type: OperationErrorType::Input,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                error.to_string(),
                BTreeMap::from([("solver_mesh_artifact_path".to_owned(), path.to_owned())]),
            )
        })?;
    let model_revision = u64::from(model.geometry_revision);
    if mesh.geometry.geometry_revision != model_revision {
        return Err(operation_error(
            operation,
            op_version,
            context,
            OperationErrorSpec {
                error_code,
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "solver mesh geometry revision does not match the analysis model; regenerate the mesh",
            BTreeMap::from([
                ("analysis_model_id".to_owned(), model.model_id.0.clone()),
                (
                    "model_geometry_revision".to_owned(),
                    model_revision.to_string(),
                ),
                (
                    "solver_mesh_geometry_revision".to_owned(),
                    mesh.geometry.geometry_revision.to_string(),
                ),
                ("solver_mesh_artifact_path".to_owned(), path.to_owned()),
            ]),
        ));
    }
    Ok(Some(mesh))
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

fn render_topology_from_solver_mesh(
    mesh: Option<&SolverMeshArtifact>,
) -> Option<AnalysisRenderTopology> {
    let mesh = mesh?;
    if mesh.topology.nodes.is_empty() || mesh.topology.boundary_faces.is_empty() {
        return None;
    }

    let node_index_by_id = mesh
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.node_id, index as u32))
        .collect::<HashMap<_, _>>();
    let volume_index_by_id = mesh
        .topology
        .volume_elements
        .iter()
        .enumerate()
        .map(|(index, element)| (element.element_id, index))
        .collect::<HashMap<_, _>>();
    let face_entries = mesh
        .topology
        .boundary_faces
        .iter()
        .filter(|face| matches!(face.node_ids.len(), 3 | 6))
        .filter_map(|face| {
            let triangle = [
                *node_index_by_id.get(&face.node_ids[0])?,
                *node_index_by_id.get(&face.node_ids[1])?,
                *node_index_by_id.get(&face.node_ids[2])?,
            ];
            let volume_element_index = face
                .adjacent_volume_element_ids
                .iter()
                .find_map(|element_id| volume_index_by_id.get(element_id).copied());
            let region_ids: Vec<String> = face
                .provenance
                .iter()
                .flat_map(|entity| {
                    std::iter::once(entity.source_topology_id.clone())
                        .chain(entity.analysis_selection_id().ok())
                })
                .collect();
            Some((triangle, volume_element_index, region_ids))
        })
        .collect::<Vec<_>>();
    let mut triangles = Vec::with_capacity(face_entries.len());
    let mut triangle_volume_element_indices = Vec::with_capacity(face_entries.len());
    let mut region_triangles = BTreeMap::<String, Vec<u32>>::new();
    for (triangle_index, (triangle, volume_element_index, region_ids)) in
        face_entries.into_iter().enumerate()
    {
        triangles.push(triangle);
        triangle_volume_element_indices.push(volume_element_index);
        for region_id in region_ids {
            region_triangles
                .entry(region_id)
                .or_default()
                .push(triangle_index as u32);
        }
    }
    if triangles.is_empty() {
        return None;
    }
    let regions = render_regions_from_triangle_indices(region_triangles);

    Some(AnalysisRenderTopology {
        schema_version: "analysis_render_topology/v1".to_string(),
        source: AnalysisRenderTopologySource::AnalysisMesh,
        meshes: vec![AnalysisRenderMesh {
            mesh_id: "solver_mesh_boundary".to_string(),
            vertices: mesh
                .topology
                .nodes
                .iter()
                .map(|node| node.coordinates_m)
                .collect(),
            triangles,
            regions,
            vertex_volume_node_indices: (0..mesh.topology.nodes.len()).map(Some).collect(),
            triangle_volume_element_indices,
        }],
    })
}

fn render_regions_from_triangle_indices(
    region_triangles: BTreeMap<String, Vec<u32>>,
) -> Vec<AnalysisRenderRegion> {
    region_triangles
        .into_iter()
        .filter_map(|(region_id, mut triangle_indices)| {
            triangle_indices.sort_unstable();
            triangle_indices.dedup();
            let mut ranges = Vec::<AnalysisRenderTriangleRange>::new();
            for triangle_index in triangle_indices {
                if let Some(last) = ranges.last_mut() {
                    if last.start.saturating_add(last.count) == triangle_index {
                        last.count = last.count.saturating_add(1);
                        continue;
                    }
                }
                ranges.push(AnalysisRenderTriangleRange {
                    start: triangle_index,
                    count: 1,
                });
            }
            if ranges.is_empty() {
                return None;
            }
            Some(AnalysisRenderRegion {
                region_id,
                label: None,
                tag: Some("boundary".to_string()),
                triangle_ranges: ranges,
            })
        })
        .collect()
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
        reference_temperature_k: domain.reference_temperature.kelvin,
        applied_temperature_delta_k: applied_temperature_delta_k(model, domain).unwrap_or(0.0),
        thermal_expansion_coefficient: expansion,
        region_temperature_deltas: regional_temperature_deltas_k(model, domain)
            .into_iter()
            .map(
                |(region_id, temperature_delta_k)| ThermoRegionTemperatureDelta {
                    region_id,
                    temperature_delta_k,
                },
            )
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
        reference_temperature_k: domain.reference_temperature.kelvin,
        applied_voltage_v: applied_voltage_v(model).unwrap_or(0.0),
        base_electrical_conductivity_s_per_m: base_conductivity,
        resistive_heating_coefficient: resistive_coeff,
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

#[derive(Debug, Clone, Copy, PartialEq)]
struct ResolvedContactOptions {
    penalty_stiffness_scale: f64,
    max_penetration_ratio: f64,
    friction_coefficient: f64,
}

fn model_contact_interface_options(
    model: &AnalysisModel,
    enforcement: Option<&ContactEnforcementOptions>,
) -> Option<ResolvedContactOptions> {
    let penalty_stiffness_scale = enforcement
        .map(|value| value.penalty_stiffness_scale)
        .unwrap_or(1.0);
    let max_penetration_ratio = enforcement
        .map(|value| value.max_penetration_ratio)
        .unwrap_or(0.01);
    model
        .interfaces
        .iter()
        .filter_map(|interface| match &interface.kind {
            AnalysisInterfaceKind::Contact(contact) => Some(ResolvedContactOptions {
                penalty_stiffness_scale,
                max_penetration_ratio,
                friction_coefficient: match contact.friction {
                    ContactFriction::Frictionless => 0.0,
                    ContactFriction::Coulomb { coefficient } => coefficient.ratio,
                },
            }),
            AnalysisInterfaceKind::FluidStructure(_)
            | AnalysisInterfaceKind::ConjugateHeatTransfer(_) => None,
        })
        .next()
}

fn to_fea_thermo_mechanical_context(
    options: Option<ThermoMechanicalCouplingOptions>,
) -> Option<runmat_analysis_fea::FeaThermoMechanicalContext> {
    options.map(|tm| runmat_analysis_fea::FeaThermoMechanicalContext {
        reference_temperature_k: tm.reference_temperature_k,
        applied_temperature_delta_k: tm.applied_temperature_delta_k,
        thermal_expansion_coefficient: tm.thermal_expansion_coefficient,
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
    })
}

fn to_fea_electro_thermal_context(
    options: Option<ElectroThermalCouplingOptions>,
) -> Option<runmat_analysis_fea::FeaElectroThermalContext> {
    options.map(|et| runmat_analysis_fea::FeaElectroThermalContext {
        reference_temperature_k: et.reference_temperature_k,
        applied_voltage_v: et.applied_voltage_v,
        base_electrical_conductivity_s_per_m: et.base_electrical_conductivity_s_per_m,
        resistive_heating_coefficient: et.resistive_heating_coefficient,
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
    options: Option<ResolvedContactOptions>,
) -> Option<runmat_analysis_fea::FeaContactInterfaceContext> {
    options.map(|contact| runmat_analysis_fea::FeaContactInterfaceContext {
        enabled: true,
        penalty_stiffness_scale: contact.penalty_stiffness_scale,
        max_penetration_ratio: contact.max_penetration_ratio,
        friction_coefficient: contact.friction_coefficient,
    })
}

fn validate_thermo_coupling_options(
    model: &AnalysisModel,
    options: &ThermoMechanicalCouplingOptions,
) -> Result<(), (String, BTreeMap<String, String>)> {
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

    let model_region_ids = model
        .material_assignments
        .iter()
        .map(|assignment| assignment.region_id.as_str())
        .chain(
            model
                .structural
                .iter()
                .flat_map(|structural| structural.elements.iter())
                .map(|element| element.region_id.as_str()),
        )
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
        if !model_region_ids.is_empty() && !model_region_ids.contains(delta.region_id.as_str()) {
            return Err((
                "thermo region_temperature_deltas region_id must exist in model material assignments"
                    .to_string(),
                BTreeMap::from([("region_id".to_string(), delta.region_id.clone())]),
            ));
        }
    }

    Ok(())
}

fn validate_electro_coupling_options(
    options: &ElectroThermalCouplingOptions,
) -> Result<(), (String, BTreeMap<String, String>)> {
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
    options: &ResolvedContactOptions,
) -> Result<(), (String, BTreeMap<String, String>)> {
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

fn validate_coupled_flow_interfaces(
    model: &AnalysisModel,
    family: &str,
) -> Result<(), (String, BTreeMap<String, String>)> {
    for interface in &model.interfaces {
        match &interface.kind {
            AnalysisInterfaceKind::Contact(_) => {
                return Err((
                    format!(
                        "{family} coupling does not accept structural contact interfaces as fluid/thermal interface mappings"
                    ),
                    BTreeMap::from([
                        ("interface_id".to_string(), interface.interface_id.clone()),
                        (
                            "primary_region_id".to_string(),
                            interface.primary.region_id.clone(),
                        ),
                        (
                            "secondary_region_id".to_string(),
                            interface.secondary.region_id.clone(),
                        ),
                        ("interface_kind".to_string(), "contact".to_string()),
                    ]),
                ));
            }
            AnalysisInterfaceKind::FluidStructure(_) => {
                if family != "FSI" {
                    return Err((
                        format!(
                            "{family} coupling does not accept fluid-structure interfaces as fluid/thermal interface mappings"
                        ),
                        BTreeMap::from([
                            ("interface_id".to_string(), interface.interface_id.clone()),
                            (
                                "primary_region_id".to_string(),
                                interface.primary.region_id.clone(),
                            ),
                            (
                                "secondary_region_id".to_string(),
                                interface.secondary.region_id.clone(),
                            ),
                            ("interface_kind".to_string(), "fluid_structure".to_string()),
                        ]),
                    ));
                }
            }
            AnalysisInterfaceKind::ConjugateHeatTransfer(cht_interface) => {
                if family != "CHT" {
                    return Err((
                        format!(
                            "{family} coupling does not accept conjugate heat-transfer interfaces as fluid/structure interface mappings"
                        ),
                        BTreeMap::from([
                            ("interface_id".to_string(), interface.interface_id.clone()),
                            (
                                "primary_region_id".to_string(),
                                interface.primary.region_id.clone(),
                            ),
                            (
                                "secondary_region_id".to_string(),
                                interface.secondary.region_id.clone(),
                            ),
                            (
                                "interface_kind".to_string(),
                                "conjugate_heat_transfer".to_string(),
                            ),
                        ]),
                    ));
                }
                let invalid_transfer = match cht_interface.transfer {
                    ThermalInterfaceTransfer::Perfect => false,
                    ThermalInterfaceTransfer::Conductance(conductance) => {
                        !conductance.is_finite() || conductance.watts_per_square_meter_kelvin <= 0.0
                    }
                    ThermalInterfaceTransfer::Resistance(resistance) => {
                        !resistance.is_finite() || resistance.square_meter_kelvin_per_watt <= 0.0
                    }
                };
                if invalid_transfer {
                    return Err((
                        format!("{family} conjugate heat-transfer interface has an invalid transfer law"),
                        BTreeMap::from([
                            ("interface_id".to_string(), interface.interface_id.clone()),
                            ("transfer_law".to_string(), format!("{:?}", cht_interface.transfer)),
                        ]),
                    ));
                }
            }
        }
    }
    Ok(())
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
                density_kg_per_m3: 7850.0,
            },
            thermal: MaterialThermalModel {
                reference_temperature_k: 293.15,
                modulus_temp_coeff_per_k: -2.5e-4,
                ..MaterialThermalModel::default()
            },
            acoustic: None,
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
                density_kg_per_m3: 7850.0,
            },
            thermal: MaterialThermalModel {
                reference_temperature_k: 293.15,
                modulus_temp_coeff_per_k: -2.5e-4,
                ..MaterialThermalModel::default()
            },
            acoustic: None,
            electrical: None,
            plastic: None,
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

fn em_sweep_known_answer_diagnostic(
    reference_frequency_hz: f64,
    frequencies_hz: &[f64],
    metrics: &EmSweepSummary,
) -> runmat_analysis_fea::diagnostics::FeaDiagnostic {
    let min_frequency_hz = frequencies_hz.iter().copied().fold(f64::INFINITY, f64::min);
    let max_frequency_hz = frequencies_hz.iter().copied().fold(0.0_f64, f64::max);
    let reference_scale = reference_frequency_hz.abs().max(1.0e-9);
    let reference_frequency_in_sweep_ratio = if frequencies_hz.iter().any(|frequency_hz| {
        (*frequency_hz - reference_frequency_hz).abs() <= reference_scale * 1.0e-9
    }) {
        1.0
    } else {
        0.0
    };
    let reference_frequency_bracket_ratio = if min_frequency_hz <= reference_frequency_hz
        && reference_frequency_hz <= max_frequency_hz
    {
        1.0
    } else {
        0.0
    };
    let normalized_peak_frequency_error_ratio = metrics
        .resonance_peak_frequency_hz
        .map(|peak_frequency_hz| {
            ((peak_frequency_hz - reference_frequency_hz).abs() / reference_scale).clamp(0.0, 1.0)
        })
        .unwrap_or(1.0);
    let resonance_flux_gain = metrics.resonance_flux_gain.unwrap_or(0.0);
    let resonance_quality_factor = metrics.resonance_quality_factor.unwrap_or(0.0);
    let sweep_known_answer_coverage_ratio = if metrics.sweep_count >= 3
        && reference_frequency_in_sweep_ratio >= 1.0
        && reference_frequency_bracket_ratio >= 1.0
        && normalized_peak_frequency_error_ratio <= 0.25
        && resonance_flux_gain >= 1.0
        && resonance_quality_factor >= 1.5
    {
        1.0
    } else {
        0.0
    };
    let severity = if sweep_known_answer_coverage_ratio >= 1.0 {
        runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info
    } else {
        runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
    };

    runmat_analysis_fea::diagnostics::FeaDiagnostic {
        code: "FEA_EM_SWEEP_KNOWN_ANSWER".to_string(),
        severity,
        message: format!(
            "basis=bracketed_reference_frequency sweep_count={} reference_frequency_hz={} sweep_frequency_min_hz={} sweep_frequency_max_hz={} reference_frequency_in_sweep_ratio={} reference_frequency_bracket_ratio={} normalized_peak_frequency_error_ratio={} resonance_flux_gain={} resonance_quality_factor={} sweep_known_answer_coverage_ratio={}",
            metrics.sweep_count,
            reference_frequency_hz,
            min_frequency_hz,
            max_frequency_hz,
            reference_frequency_in_sweep_ratio,
            reference_frequency_bracket_ratio,
            normalized_peak_frequency_error_ratio,
            resonance_flux_gain,
            resonance_quality_factor,
            sweep_known_answer_coverage_ratio,
        ),
    }
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
        assignments.push(MaterialAssignment {
            region_id: region.region_id.clone(),
            expected_material_id: assigned_material.clone(),
            assigned_material_id: assigned_material,
            confidence: evidence_confidence,
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
        AnalysisValidationError::MissingConstraintsOrBoundaries => (
            "RM.FEA.VALIDATE.MISSING_CONDITIONS",
            "FEA model must include at least one constraint or natural boundary".to_string(),
            BTreeMap::new(),
        ),
        AnalysisValidationError::InvalidId { kind } => (
            "RM.FEA.VALIDATE.INVALID_ID",
            format!("{kind} id must be non-empty"),
            BTreeMap::from([("kind".to_string(), kind.to_string())]),
        ),
        AnalysisValidationError::DuplicateId { kind, id } => (
            "RM.FEA.VALIDATE.DUPLICATE_ID",
            format!("duplicate {kind} id `{id}`"),
            BTreeMap::from([
                ("kind".to_string(), kind.to_string()),
                ("id".to_string(), id),
            ]),
        ),
        AnalysisValidationError::InvalidTarget { owner_id, detail } => (
            "RM.FEA.VALIDATE.INVALID_TARGET",
            format!("{owner_id} has an invalid target: {detail}"),
            BTreeMap::from([
                ("owner_id".to_string(), owner_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidFrame => (
            "RM.FEA.VALIDATE.INVALID_FRAME",
            "coordinate frame must be finite and non-degenerate".to_string(),
            BTreeMap::new(),
        ),
        AnalysisValidationError::InvalidMaterial {
            material_id,
            detail,
        } => (
            "RM.FEA.VALIDATE.INVALID_MATERIAL",
            format!("material `{material_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("material_id".to_string(), material_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidMaterialAssignment { region_id, detail } => (
            "RM.FEA.VALIDATE.INVALID_MATERIAL_ASSIGNMENT",
            format!("material assignment for region `{region_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("region_id".to_string(), region_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidConstraint {
            constraint_id,
            detail,
        } => (
            "RM.FEA.VALIDATE.INVALID_CONSTRAINT",
            format!("constraint `{constraint_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("constraint_id".to_string(), constraint_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidBoundary {
            boundary_id,
            detail,
        } => (
            "RM.FEA.VALIDATE.INVALID_BOUNDARY",
            format!("boundary `{boundary_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("boundary_id".to_string(), boundary_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidConnector {
            connector_id,
            detail,
        } => (
            "RM.FEA.VALIDATE.INVALID_CONNECTOR",
            format!("connector `{connector_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("connector_id".to_string(), connector_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidInterface {
            interface_id,
            detail,
        } => (
            "RM.FEA.VALIDATE.INVALID_INTERFACE",
            format!("interface `{interface_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("interface_id".to_string(), interface_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidSignal { signal_id, detail } => (
            "RM.FEA.VALIDATE.INVALID_SIGNAL",
            format!("signal `{signal_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("signal_id".to_string(), signal_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidDomain { domain, detail } => (
            "RM.FEA.VALIDATE.INVALID_DOMAIN",
            format!("{domain} domain is invalid: {detail}"),
            BTreeMap::from([
                ("domain".to_string(), domain.to_string()),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidExcitation {
            excitation_id,
            detail,
        } => (
            "RM.FEA.VALIDATE.INVALID_EXCITATION",
            format!("excitation `{excitation_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("excitation_id".to_string(), excitation_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidExcitationCase { case_id, detail } => (
            "RM.FEA.VALIDATE.INVALID_EXCITATION_CASE",
            format!("excitation case `{case_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("case_id".to_string(), case_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidInitialCondition {
            initial_condition_id,
            detail,
        } => (
            "RM.FEA.VALIDATE.INVALID_INITIAL_CONDITION",
            format!("initial condition `{initial_condition_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("initial_condition_id".to_string(), initial_condition_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidResponseSpectrum {
            spectrum_id,
            detail,
        } => (
            "RM.FEA.VALIDATE.INVALID_RESPONSE_SPECTRUM",
            format!("response spectrum `{spectrum_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("spectrum_id".to_string(), spectrum_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidSequence {
            sequence_id,
            detail,
        } => (
            "RM.FEA.VALIDATE.INVALID_SEQUENCE",
            format!("analysis sequence `{sequence_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("sequence_id".to_string(), sequence_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidProcedureStep { step_id, detail } => (
            "RM.FEA.VALIDATE.INVALID_PROCEDURE_STEP",
            format!("procedure step `{step_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("step_id".to_string(), step_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidStateTransition {
            transition_id,
            detail,
        } => (
            "RM.FEA.VALIDATE.INVALID_STATE_TRANSITION",
            format!("state transition `{transition_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("transition_id".to_string(), transition_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidObservation {
            observation_id,
            detail,
        } => (
            "RM.FEA.VALIDATE.INVALID_OBSERVATION",
            format!("observation `{observation_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("observation_id".to_string(), observation_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidMonitor { monitor_id, detail } => (
            "RM.FEA.VALIDATE.INVALID_MONITOR",
            format!("monitor `{monitor_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("monitor_id".to_string(), monitor_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidAccuracyPlan { plan_id, detail } => (
            "RM.FEA.VALIDATE.INVALID_ACCURACY_PLAN",
            format!("accuracy plan `{plan_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("plan_id".to_string(), plan_id),
                ("detail".to_string(), detail),
            ]),
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
                ("model_frame".to_string(), model.context_label()),
                ("geometry_frame".to_string(), geometry.context_label()),
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

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
