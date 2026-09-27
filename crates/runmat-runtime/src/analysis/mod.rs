use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::ErrorKind;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock, RwLock};
use std::time::Instant;

use chrono::Utc;
use runmat_analysis_core::{
    validate_model_against_geometry, AnalysisConstraint, AnalysisField, AnalysisInterfaceKind,
    AnalysisModel, AnalysisProcedure, AnalysisValidationError, BoundaryOperator, ContactFriction,
    CoordinateFrame, CoupledPhysicsFamily, ExcitationAction, MagneticConstraintKind, PhysicsFamily,
    ProcedureStepId, StateKind, StateOutput, ThermalInterfaceTransfer,
};
use runmat_analysis_fea::execution::{
    CanonicalSemanticState, CanonicalSemanticStateInput, CanonicalSemanticStatePayload,
    SequenceIdentity, SolverResultIdentity, StepIdentity, StructuralLinearWorkload, StudyIdentity,
};
use runmat_analysis_fea::physics::cfd::CfdDomainTopology;
use runmat_analysis_fea::physics::cht::{couple_thermal_flow, ChtCouplingOptions};
use runmat_analysis_fea::physics::fsi::{couple_fluid_structure, FsiCouplingOptions};
use runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind;
use runmat_analysis_fea::solve::preconditioner::SpdPreconditionerKind;
use runmat_analysis_fea::{
    run_electromagnetic_frequency_response, run_modal_with_options, run_nonlinear_with_options,
    run_thermal_with_options, run_transient_with_options, AnalysisRunIdentity, ComputeBackend,
    ElectromagneticSolveOptions, ExecutionBackend, FeaBucklingRunResult, FeaProgressHandler,
    FeaRunError, FeaRunResult, ModalSolveOptions, SolverMethod, SolverPreconditioner,
    ThermalProcedureControl, ThermalSolveOptions,
};
use runmat_geometry_core::UnitSystem;
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
#[cfg(test)]
mod excitation_authoring;
mod execution_dispatch;
mod execution_error;
#[cfg(feature = "plot-core")]
pub mod field_scene;
pub(crate) mod field_selection;
#[cfg(feature = "plot-core")]
mod figures;
mod formulation_admission;
mod frequency_result;
mod linear_static_execution;
mod linear_static_result;
pub(crate) mod meshing;
mod operations;
mod policy;
pub mod render_tiles;
mod response_study_result;
mod result_policy;
mod solver_mesh_artifact;
mod source;
pub mod storage;
mod structural_study;
mod study_contract;
mod study_execution;
mod study_placement;
mod study_result;
mod study_sweep;
mod typed_study_publication;
pub use response_study_result::{
    StructuralStudyModalResponseSpectrumNumericalResult,
    StructuralStudyRandomResponseNumericalResult,
};
pub mod value_admission;

#[cfg(test)]
use excitation_authoring::{append_constant_excitation, ConstantExcitation};
use execution_dispatch::{execution_operation, run_operation_for_identity, AnalysisOperation};
use execution_error::{admit_runtime_proxy_for_operation, map_fea_run_error};
use formulation_admission::{RuntimeDomainSource, RuntimeDomainTopology, RuntimeProxyFormulation};
use linear_static_execution::analysis_run_linear_static_with_executor;
use result_policy::{classify_run_status, evaluate_production_publication};
#[cfg(test)]
use structural_study::persist_immutable_study_evidence;
#[cfg(test)]
use structural_study::run_evidence_stage;
pub use structural_study::{
    analysis_complete_structural_study_intent_op, analysis_prepare_structural_study_intent_op,
    structural_study_execution_failure, PreparedStructuralStudy, PreparedStructuralStudyIntent,
    StructuralStudyCompletionIdentity, StructuralStudyExecutionFailure,
    StructuralStudyExecutionOutcome, STRUCTURAL_STUDY_INTENT_MAX_BYTES,
};
use structural_study::{
    complete_study_run, persist_content_addressed_study_evidence, study_completion_from_prepared,
    study_evidence_root, CompletedStudyStep, CompletedStudyStepRun, PreparedStudySequence,
    PreparedStudyStep,
};
pub use study_contract::analysis_study_content_digest;
use study_contract::{
    sanitize_study_sweep_id, study_fingerprint, study_fingerprint_from_digest, study_issue_message,
    study_operation_sequence, validate_study_issue_codes,
};
use study_execution::execute_study_step;
pub use study_placement::{
    analysis_execute_structural_study_with_placement,
    analysis_execute_structural_sweep_with_placement, StructuralStudyPlacementEntry,
    StructuralStudyPlacementExecutor, StructuralStudyPlacementFuture,
    StructuralStudyPlacementResult,
};
pub use study_result::{
    StructuralStudyResult, ThermalStudyCompletion, ThermalStudyObjectSource, ThermalStudySource,
};
pub use study_sweep::{
    analysis_complete_study_sweep_op, analysis_study_sweep_definition,
    analysis_study_sweep_identity, AnalysisStudySweepRunOutcome,
};
pub use typed_study_publication::publish_structural_study_result;

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
    ) -> Result<runmat_analysis_fea::execution::StructuralLinearWorkloadResult, String>;
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
    pub result: runmat_analysis_fea::execution::StructuralLinearWorkloadResult,
    pub numerical_receipt: runmat_analysis_fea::execution::StructuralLinearNumericalReceiptArtifact,
}

#[derive(Debug, Clone)]
pub struct StructuralStudySequenceNumericalResult {
    pub frozen_project_digest: Option<runmat_execution::Digest>,
    pub steps: Vec<StructuralStudyStepNumericalResult>,
    pub states: Vec<runmat_analysis_fea::execution::CanonicalSemanticState>,
}

#[derive(Debug, Clone)]
pub struct StructuralStudyTransientNumericalResult {
    pub frozen_project_digest: Option<runmat_execution::Digest>,
    pub step_id: ProcedureStepId,
    pub result: runmat_analysis_fea::execution::StructuralTransientSolverResult,
    pub numerical_receipt: runmat_analysis_fea::execution::StructuralTransientNumericalReceipt,
}

#[derive(Debug, Clone)]
pub struct StructuralStudyModalNumericalResult {
    pub frozen_project_digest: Option<runmat_execution::Digest>,
    pub step_id: ProcedureStepId,
    pub result: runmat_analysis_fea::execution::StructuralModalSolverResult,
    pub numerical_receipt: runmat_analysis_fea::execution::StructuralModalNumericalReceipt,
}

#[derive(Debug, Clone)]
pub struct StructuralStudyModalTransientNumericalResult {
    pub frozen_project_digest: Option<runmat_execution::Digest>,
    pub modal_step_id: ProcedureStepId,
    pub transient_step_id: ProcedureStepId,
    pub modal_result: runmat_analysis_fea::execution::StructuralModalSolverResult,
    pub modal_numerical_receipt: runmat_analysis_fea::execution::StructuralModalNumericalReceipt,
    pub basis_state: runmat_analysis_fea::execution::CanonicalSemanticState,
    pub transient_result: runmat_analysis_fea::execution::StructuralTransientSolverResult,
    pub transient_numerical_receipt:
        runmat_analysis_fea::execution::StructuralTransientNumericalReceipt,
}

#[derive(Debug, Clone)]
pub struct StructuralStudyBucklingNumericalResult {
    pub frozen_project_digest: Option<runmat_execution::Digest>,
    pub preload_step_id: ProcedureStepId,
    pub buckling_step_id: ProcedureStepId,
    pub preload_result: runmat_analysis_fea::execution::StructuralLinearWorkloadResult,
    pub preload_solver_result_identity: runmat_analysis_fea::execution::SolverResultIdentity,
    pub preload_numerical_receipt:
        runmat_analysis_fea::execution::StructuralLinearNumericalReceiptArtifact,
    pub prestress_state: runmat_analysis_fea::execution::CanonicalSemanticState,
    pub buckling_result: runmat_analysis_fea::execution::StructuralBucklingSolverResult,
    pub buckling_numerical_receipt:
        runmat_analysis_fea::execution::StructuralBucklingNumericalReceipt,
}

#[derive(Debug, Clone)]
pub struct StructuralStudyPrestressedModalNumericalResult {
    pub frozen_project_digest: Option<runmat_execution::Digest>,
    pub preload_step_id: ProcedureStepId,
    pub modal_step_id: ProcedureStepId,
    pub preload_result: runmat_analysis_fea::execution::StructuralLinearWorkloadResult,
    pub preload_solver_result_identity: runmat_analysis_fea::execution::SolverResultIdentity,
    pub preload_numerical_receipt:
        runmat_analysis_fea::execution::StructuralLinearNumericalReceiptArtifact,
    pub prestress_state: runmat_analysis_fea::execution::CanonicalSemanticState,
    pub modal_result: runmat_analysis_fea::execution::StructuralModalSolverResult,
    pub modal_numerical_receipt: runmat_analysis_fea::execution::StructuralModalNumericalReceipt,
}

#[derive(Debug, Clone)]
pub enum StructuralStudyPrestressedFrequencyReductionResult {
    Direct,
    ModalReduced(Box<StructuralStudyPrestressedModalFrequencyReductionResult>),
}

#[derive(Debug, Clone)]
pub struct StructuralStudyPrestressedModalFrequencyReductionResult {
    pub modal_step_id: ProcedureStepId,
    pub modal_result: runmat_analysis_fea::execution::StructuralModalSolverResult,
    pub modal_numerical_receipt: runmat_analysis_fea::execution::StructuralModalNumericalReceipt,
    pub basis_state: runmat_analysis_fea::execution::CanonicalSemanticState,
}

#[derive(Debug, Clone)]
pub struct StructuralStudyPrestressedFrequencyNumericalResult {
    pub frozen_project_digest: Option<runmat_execution::Digest>,
    pub preload_step_id: ProcedureStepId,
    pub frequency_step_id: ProcedureStepId,
    pub preload_result: runmat_analysis_fea::execution::StructuralLinearWorkloadResult,
    pub preload_solver_result_identity: runmat_analysis_fea::execution::SolverResultIdentity,
    pub preload_numerical_receipt:
        runmat_analysis_fea::execution::StructuralLinearNumericalReceiptArtifact,
    pub prestress_state: runmat_analysis_fea::execution::CanonicalSemanticState,
    pub reduction: StructuralStudyPrestressedFrequencyReductionResult,
    pub frequency: StructuralStudyFrequencyNumericalResult,
}

mod study_numerical_result;
pub use study_numerical_result::StructuralStudyNumericalResult;

pub use frequency_result::{
    StructuralFrequencyArtifactRoot, StructuralFrequencyArtifactRootIndex,
    StructuralFrequencyHistoryArtifactRoot, StructuralFrequencySampleArtifactRoots,
    StructuralStudyFrequencyNumericalResult,
};

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
    PathBuf::from(".artifacts")
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
    _solver_context: runmat_analysis_fea::FeaProgressContextGuard,
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
    let solver_context = install_fea_solver_context();
    FeaProgressHandlerGuard {
        previous,
        _solver_context: solver_context,
    }
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
    analysis_authoring_workflow, analysis_language_reference_paths, analysis_output_operation_ids,
    analysis_procedure_ids, AnalysisAcousticRunOptions, AnalysisBucklingRunOptions,
    AnalysisCfdRunOptions, AnalysisChtRunOptions, AnalysisDiagnosticsArtifactPayload,
    AnalysisDocumentCheckResult, AnalysisDocumentKind, AnalysisDocumentRunResult,
    AnalysisElectromagneticRunOptions, AnalysisExecutionPlan, AnalysisExecutionStep,
    AnalysisFieldBinding, AnalysisFieldBindingsArtifactPayload, AnalysisFieldDescriptor,
    AnalysisFieldKind, AnalysisFieldLocation, AnalysisFieldPageResult, AnalysisFieldPageValues,
    AnalysisFieldPagingDescriptor, AnalysisFieldRequestOptions, AnalysisFieldSelection,
    AnalysisFieldStatistic, AnalysisFieldStorage, AnalysisFieldStorageRef, AnalysisFsiRunOptions,
    AnalysisImmutableValueSpec, AnalysisModalRunOptions, AnalysisModelSourceEvidence,
    AnalysisModelSourceFileEvidence, AnalysisModelSourceKind, AnalysisNonlinearRunOptions,
    AnalysisObjectArtifactMetadata, AnalysisRandomResponseRunOptions, AnalysisRenderMesh,
    AnalysisRenderRegion, AnalysisRenderTopology, AnalysisRenderTopologySource,
    AnalysisRenderTriangleRange, AnalysisResponseSpectrumRunOptions, AnalysisResultsCompareData,
    AnalysisResultsCompareQuery, AnalysisResultsData, AnalysisResultsQuery, AnalysisResultsSummary,
    AnalysisRunDatasetFieldPagingPolicy, AnalysisRunDatasetPayload, AnalysisRunDatasetStudyRef,
    AnalysisRunOptions, AnalysisRunResult, AnalysisRuntimeCapabilities, AnalysisSolverOptions,
    AnalysisStructuralFrequencyRunOptions, AnalysisStructuralFrequencySolver,
    AnalysisStructuralTimeStepPolicy, AnalysisStudyIssue, AnalysisStudyPlanData,
    AnalysisStudyPlanStepData, AnalysisStudyRunData, AnalysisStudySpec,
    AnalysisStudyStatePublicationData, AnalysisStudyStepNumericalResultData,
    AnalysisStudyStepRunData, AnalysisStudySweepData, AnalysisStudySweepFailureEntry,
    AnalysisStudySweepFailureKind, AnalysisStudySweepPlanData, AnalysisStudySweepPlanEntry,
    AnalysisStudySweepRunEntry, AnalysisStudySweepSpec, AnalysisStudySweepValidateData,
    AnalysisStudySweepValidateEntry, AnalysisStudyValidateResult, AnalysisThermalRunOptions,
    AnalysisTransientRunOptions, AnalysisTrendKindSummary, AnalysisTrendsData, AnalysisTrendsQuery,
    AnalysisValidateResult, BucklingResultsData, ContactEnforcementOptions,
    ElectromagneticResultsData, FallbackEvent, FallbackReason, FallbackStage, ModalFrequencyBasis,
    ModalFrequencyUnits, ModalResultsData, NonlinearMethod, NonlinearResultsData,
    PlasticityConstitutiveOptions, PrecisionMode, PreconditionerMode, ProductionPublicationReason,
    ProductionPublicationReasonCode, QualityGate, QualityPolicy, QualityReason, QualityReasonCode,
    RunProvenance, RunStatus, StructuralFrequencyResultsData, StructuralRandomResponseResultsData,
    StructuralResponseSpectrumResultsData, ThermalResultsData, TransientIntegrationMethod,
    TransientResultsData, ANALYSIS_ARTIFACT_MANIFEST_KIND, ANALYSIS_DATASET_ARTIFACT_KIND,
    ANALYSIS_DIAGNOSTICS_ARTIFACT_KIND, ANALYSIS_DIAGNOSTICS_SCHEMA_VERSION,
    ANALYSIS_FIELD_BINDINGS_ARTIFACT_KIND, ANALYSIS_FIELD_BINDINGS_SCHEMA_VERSION,
    ANALYSIS_FIELD_DEFAULT_MATERIALIZE_LIMIT, ANALYSIS_FIELD_DEFAULT_PAGE_SIZE,
    ANALYSIS_OBJECT_ARTIFACT_METADATA_SCHEMA_VERSION, ANALYSIS_RUN_DATASET_KIND,
    ANALYSIS_RUN_DATASET_SCHEMA_VERSION,
};
#[cfg(feature = "plot-core")]
pub use figures::{
    analysis_generate_study_field_figures, analysis_generate_study_run_figures,
    analysis_generate_study_run_mesh_figure, AnalysisFigureGenerationOptions,
    AnalysisFigureMeshSource, AnalysisGeneratedFigure, AnalysisGeneratedFigureKind,
};
pub use operations::{
    analysis_field_page_by_run_id_op, analysis_results_by_run_id_op, analysis_results_compare_op,
    analysis_results_op, analysis_trends_op,
};
pub(crate) use operations::{analysis_run_field_ids, collect_analysis_result_fields};
pub use runmat_analysis_core::FormulationRequirement;
pub use runmat_analysis_fea::{FeaProgressEvent, FeaProgressPhase, FeaProgressStatus};
pub use runmat_analysis_language::{
    operation_pattern_from_value_fact, remote_execution_profile_facts, ExecutionProfileFacts,
    FunctionFacts, FunctionRequest, LogicalSourcePath, SourceDigest,
};
pub use source::{
    check_fea_path, check_fea_path_in_root, check_fea_source_in_root, check_fea_sources_in_root,
    logical_source_path, physical_source_path, prepare_fea_path_in_root,
    prepare_fea_sources_in_root, source_digest, CheckedFeaSource, FeaCheckStatus,
    FeaDeclaredMaterial, FeaExecutionBinding, FeaExecutionTarget, FeaModelMemberKind,
    FeaRootSelectionError, FeaRootSelector, FeaSemanticProgram, FeaSourceCheckResult,
    FeaSourceIdentity, FeaSourceIndex, FeaSourceIndexEntry, FeaSourceSnapshotBuilder,
    FeaSourceSnapshotRequests, FilesystemFeaHost, PreparedFeaSource, PreparedFeaSourceSnapshot,
    ProjectFeaHost, ResolvedFeaRoot,
};

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

pub fn analysis_validate_study_op(
    spec: &AnalysisStudySpec,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisStudyValidateResult>, OperationErrorEnvelope> {
    let mut issue_codes = validate_study_issue_codes(spec);
    let model_validation_detail =
        validate_model_against_geometry(&spec.model, spec.geometry.units, &CoordinateFrame::Global)
            .err()
            .map(|error| error.to_string());
    issue_codes.sort();
    issue_codes.dedup();
    let issues: Vec<AnalysisStudyIssue> = issue_codes
        .iter()
        .map(|code| AnalysisStudyIssue {
            code: code.clone(),
            message: if code == "RM.FEA.STUDY.MODEL_INVALID" {
                model_validation_detail
                    .as_ref()
                    .map(|detail| format!("study model failed FEA validation: {detail}"))
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
    let model = &spec.model;
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
            "model_id": model.model_id.0.clone(),
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
            model_id: model.model_id.0.clone(),
            sequence_id: model.sequence.sequence_id.clone(),
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
    let mut study = prepare_study_run(spec, context)?;
    let solver_mesh = resolve_solver_mesh_artifact(
        &study.model,
        study.solver_mesh_artifact_path.as_deref(),
        ANALYSIS_RUN_STUDY_OPERATION,
        ANALYSIS_RUN_STUDY_OP_VERSION,
        "RM.FEA.RUN_STUDY.PORTABLE_MESH_REQUIRED",
        &study.context,
    )?
    .ok_or_else(|| {
        operation_error(
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
        )
    })?;

    if let [step] = study.steps.as_slice() {
        if let AnalysisSolverOptions::StructuralModes(mut options) =
            step.authored_execution.solver_options.clone()
        {
            options.solver_mesh_artifact_path = None;
            let workload =
                structural_study::prepare_structural_modal_workload(&study, step, &options)?;
            return Ok(PreparedStructuralStudy {
                study,
                execution: structural_study::PreparedStructuralStudyExecution::StructuralModal {
                    workload: Box::new(workload),
                },
                solver_mesh,
            });
        }
    }

    if let [modal_step, response_step] = study.steps.as_slice() {
        if response_step.procedure_step.procedure
            == runmat_analysis_core::AnalysisProcedure::ModalResponseSpectrum
        {
            if let (
                AnalysisSolverOptions::StructuralModes(mut modal_options),
                AnalysisSolverOptions::StructuralResponseSpectrum(mut response_options),
            ) = (
                modal_step.authored_execution.solver_options.clone(),
                response_step.authored_execution.solver_options.clone(),
            ) {
                modal_options.solver_mesh_artifact_path = None;
                response_options.solver_mesh_artifact_path = None;
                structural_study::complete_response_spectrum_modal_options(
                    &study.model,
                    &response_step.procedure_step,
                    &mut modal_options,
                )
                .map_err(|message| {
                    operation_error(
                        ANALYSIS_RUN_STUDY_OPERATION,
                        ANALYSIS_RUN_STUDY_OP_VERSION,
                        &study.context,
                        OperationErrorSpec {
                            error_code: "RM.FEA.RUN_STUDY.INVALID_PORTABLE_INTENT",
                            error_type: OperationErrorType::Contract,
                            retryable: false,
                            severity: OperationErrorSeverity::Fatal,
                        },
                        message,
                        BTreeMap::new(),
                    )
                })?;
                let modal = structural_study::prepare_structural_modal_workload(
                    &study,
                    modal_step,
                    &modal_options,
                )?;
                let response = structural_study::prepare_structural_response_spectrum_template(
                    &study,
                    response_step,
                    &response_options,
                )?;
                let workloads = BTreeMap::from([
                    (
                        modal_step.procedure_step.step_id.clone(),
                        runmat_analysis_fea::execution::StructuralStepWorkload::ModalExtraction {
                            workload: Box::new(modal),
                        },
                    ),
                    (
                        response_step.procedure_step.step_id.clone(),
                        runmat_analysis_fea::execution::StructuralStepWorkload::ModalResponseSpectrum {
                            template: Box::new(response),
                        },
                    ),
                ]);
                runmat_analysis_fea::execution::StructuralStudyWorkloads::HeterogeneousSequence {
                    workloads: workloads.clone(),
                }
                .validate()
                .map_err(|error| {
                    operation_error(
                        ANALYSIS_RUN_STUDY_OPERATION,
                        ANALYSIS_RUN_STUDY_OP_VERSION,
                        &study.context,
                        OperationErrorSpec {
                            error_code: "RM.FEA.RUN_STUDY.INVALID_PORTABLE_INTENT",
                            error_type: OperationErrorType::Contract,
                            retryable: false,
                            severity: OperationErrorSeverity::Fatal,
                        },
                        error.to_string(),
                        BTreeMap::new(),
                    )
                })?;
                study.steps[0].authored_execution.solver_options =
                    AnalysisSolverOptions::StructuralModes(modal_options);
                study.steps[1].authored_execution.solver_options =
                    AnalysisSolverOptions::StructuralResponseSpectrum(response_options);
                return Ok(PreparedStructuralStudy {
                    study,
                    execution: structural_study::PreparedStructuralStudyExecution::StructuralModalResponseSpectrum {
                        workloads,
                    },
                    solver_mesh,
                });
            }
        }
    }

    if let [modal_step, response_step] = study.steps.as_slice() {
        if response_step.procedure_step.procedure
            == runmat_analysis_core::AnalysisProcedure::RandomResponse
        {
            if let (
                AnalysisSolverOptions::StructuralModes(mut modal_options),
                AnalysisSolverOptions::StructuralRandomResponse(mut response_options),
            ) = (
                modal_step.authored_execution.solver_options.clone(),
                response_step.authored_execution.solver_options.clone(),
            ) {
                modal_options.solver_mesh_artifact_path = None;
                response_options.solver_mesh_artifact_path = None;
                let modal = structural_study::prepare_structural_modal_workload(
                    &study,
                    modal_step,
                    &modal_options,
                )?;
                let response = structural_study::prepare_structural_random_response_template(
                    &study,
                    response_step,
                    &response_options,
                )?;
                let workloads = BTreeMap::from([
                    (
                        modal_step.procedure_step.step_id.clone(),
                        runmat_analysis_fea::execution::StructuralStepWorkload::ModalExtraction {
                            workload: Box::new(modal),
                        },
                    ),
                    (
                        response_step.procedure_step.step_id.clone(),
                        runmat_analysis_fea::execution::StructuralStepWorkload::RandomResponse {
                            template: Box::new(response),
                        },
                    ),
                ]);
                runmat_analysis_fea::execution::StructuralStudyWorkloads::HeterogeneousSequence {
                    workloads: workloads.clone(),
                }
                .validate()
                .map_err(|error| {
                    operation_error(
                        ANALYSIS_RUN_STUDY_OPERATION,
                        ANALYSIS_RUN_STUDY_OP_VERSION,
                        &study.context,
                        OperationErrorSpec {
                            error_code: "RM.FEA.RUN_STUDY.INVALID_PORTABLE_INTENT",
                            error_type: OperationErrorType::Contract,
                            retryable: false,
                            severity: OperationErrorSeverity::Fatal,
                        },
                        error.to_string(),
                        BTreeMap::new(),
                    )
                })?;
                study.steps[0].authored_execution.solver_options =
                    AnalysisSolverOptions::StructuralModes(modal_options);
                study.steps[1].authored_execution.solver_options =
                    AnalysisSolverOptions::StructuralRandomResponse(response_options);
                return Ok(PreparedStructuralStudy {
                    study,
                    execution:
                        structural_study::PreparedStructuralStudyExecution::StructuralRandomResponse {
                            workloads,
                        },
                    solver_mesh,
                });
            }
        }
    }

    if let [preload_step, modal_step, frequency_step] = study.steps.as_slice() {
        if modal_step.procedure_step.procedure
            == runmat_analysis_core::AnalysisProcedure::PrestressedModalExtraction
            && frequency_step.procedure_step.procedure
                == runmat_analysis_core::AnalysisProcedure::PrestressedHarmonicResponse
        {
            if let (
                AnalysisSolverOptions::StructuralLinear(mut preload_options),
                AnalysisSolverOptions::StructuralModes(mut modal_options),
                AnalysisSolverOptions::StructuralFrequency(mut frequency_options),
            ) = (
                preload_step.authored_execution.solver_options.clone(),
                modal_step.authored_execution.solver_options.clone(),
                frequency_step.authored_execution.solver_options.clone(),
            ) {
                if preload_options.solver_mesh_artifact_path.is_none() {
                    preload_options.solver_mesh_artifact_path =
                        study.solver_mesh_artifact_path.clone();
                }
                let preload = prepare_linear_static_step_run(
                    &study.model,
                    &preload_step.procedure_step.step_id,
                    preload_step.authored_execution.backend,
                    preload_step.authored_execution.requirement,
                    preload_options,
                    study.context.clone(),
                )?;
                modal_options.solver_mesh_artifact_path = None;
                let modal = structural_study::prepare_structural_prestressed_modal_template(
                    &study,
                    modal_step,
                    &modal_options,
                )?;
                frequency_options.solver_mesh_artifact_path = None;
                let frequency =
                    structural_study::prepare_structural_prestressed_frequency_template(
                        &study,
                        frequency_step,
                        &frequency_options,
                    )?;
                let workloads = BTreeMap::from([
                    (
                        preload_step.procedure_step.step_id.clone(),
                        runmat_analysis_fea::execution::StructuralStepWorkload::Linear {
                            workload: Box::new(preload.workload),
                        },
                    ),
                    (
                        modal_step.procedure_step.step_id.clone(),
                        runmat_analysis_fea::execution::StructuralStepWorkload::PrestressedModal {
                            template: Box::new(modal),
                        },
                    ),
                    (
                        frequency_step.procedure_step.step_id.clone(),
                        runmat_analysis_fea::execution::StructuralStepWorkload::PrestressedFrequency {
                            template: Box::new(frequency),
                        },
                    ),
                ]);
                runmat_analysis_fea::execution::StructuralStudyWorkloads::HeterogeneousSequence {
                    workloads: workloads.clone(),
                }
                .validate()
                .map_err(|error| {
                    operation_error(
                        ANALYSIS_RUN_STUDY_OPERATION,
                        ANALYSIS_RUN_STUDY_OP_VERSION,
                        &study.context,
                        OperationErrorSpec {
                            error_code: "RM.FEA.RUN_STUDY.INVALID_PORTABLE_INTENT",
                            error_type: OperationErrorType::Contract,
                            retryable: false,
                            severity: OperationErrorSeverity::Fatal,
                        },
                        error.to_string(),
                        BTreeMap::new(),
                    )
                })?;
                return Ok(PreparedStructuralStudy {
                    study,
                    execution:
                        structural_study::PreparedStructuralStudyExecution::StructuralPrestressedFrequency {
                            workloads,
                        },
                    solver_mesh,
                });
            }
        }
    }

    if let [preload_step, modal_step] = study.steps.as_slice() {
        if modal_step.procedure_step.procedure
            == runmat_analysis_core::AnalysisProcedure::PrestressedModalExtraction
        {
            if let (
                AnalysisSolverOptions::StructuralLinear(mut preload_options),
                AnalysisSolverOptions::StructuralModes(mut modal_options),
            ) = (
                preload_step.authored_execution.solver_options.clone(),
                modal_step.authored_execution.solver_options.clone(),
            ) {
                if preload_options.solver_mesh_artifact_path.is_none() {
                    preload_options.solver_mesh_artifact_path =
                        study.solver_mesh_artifact_path.clone();
                }
                let preload = prepare_linear_static_step_run(
                    &study.model,
                    &preload_step.procedure_step.step_id,
                    preload_step.authored_execution.backend,
                    preload_step.authored_execution.requirement,
                    preload_options,
                    study.context.clone(),
                )?;
                modal_options.solver_mesh_artifact_path = None;
                let modal = structural_study::prepare_structural_prestressed_modal_template(
                    &study,
                    modal_step,
                    &modal_options,
                )?;
                let workloads = BTreeMap::from([
                    (
                        preload_step.procedure_step.step_id.clone(),
                        runmat_analysis_fea::execution::StructuralStepWorkload::Linear {
                            workload: Box::new(preload.workload),
                        },
                    ),
                    (
                        modal_step.procedure_step.step_id.clone(),
                        runmat_analysis_fea::execution::StructuralStepWorkload::PrestressedModal {
                            template: Box::new(modal),
                        },
                    ),
                ]);
                runmat_analysis_fea::execution::StructuralStudyWorkloads::HeterogeneousSequence {
                    workloads: workloads.clone(),
                }
                .validate()
                .map_err(|error| {
                    operation_error(
                        ANALYSIS_RUN_STUDY_OPERATION,
                        ANALYSIS_RUN_STUDY_OP_VERSION,
                        &study.context,
                        OperationErrorSpec {
                            error_code: "RM.FEA.RUN_STUDY.INVALID_PORTABLE_INTENT",
                            error_type: OperationErrorType::Contract,
                            retryable: false,
                            severity: OperationErrorSeverity::Fatal,
                        },
                        error.to_string(),
                        BTreeMap::new(),
                    )
                })?;
                return Ok(PreparedStructuralStudy {
                    study,
                    execution:
                        structural_study::PreparedStructuralStudyExecution::StructuralPrestressedModal {
                            workloads,
                        },
                    solver_mesh,
                });
            }
        }
    }

    if let [preload_step, frequency_step] = study.steps.as_slice() {
        if frequency_step.procedure_step.procedure
            == runmat_analysis_core::AnalysisProcedure::PrestressedHarmonicResponse
        {
            if let (
                AnalysisSolverOptions::StructuralLinear(mut preload_options),
                AnalysisSolverOptions::StructuralFrequency(mut frequency_options),
            ) = (
                preload_step.authored_execution.solver_options.clone(),
                frequency_step.authored_execution.solver_options.clone(),
            ) {
                if preload_options.solver_mesh_artifact_path.is_none() {
                    preload_options.solver_mesh_artifact_path =
                        study.solver_mesh_artifact_path.clone();
                }
                let preload = prepare_linear_static_step_run(
                    &study.model,
                    &preload_step.procedure_step.step_id,
                    preload_step.authored_execution.backend,
                    preload_step.authored_execution.requirement,
                    preload_options,
                    study.context.clone(),
                )?;
                frequency_options.solver_mesh_artifact_path = None;
                let frequency =
                    structural_study::prepare_structural_prestressed_frequency_template(
                        &study,
                        frequency_step,
                        &frequency_options,
                    )?;
                let workloads = BTreeMap::from([
                    (
                        preload_step.procedure_step.step_id.clone(),
                        runmat_analysis_fea::execution::StructuralStepWorkload::Linear {
                            workload: Box::new(preload.workload),
                        },
                    ),
                    (
                        frequency_step.procedure_step.step_id.clone(),
                        runmat_analysis_fea::execution::StructuralStepWorkload::PrestressedFrequency {
                            template: Box::new(frequency),
                        },
                    ),
                ]);
                runmat_analysis_fea::execution::StructuralStudyWorkloads::HeterogeneousSequence {
                    workloads: workloads.clone(),
                }
                .validate()
                .map_err(|error| {
                    operation_error(
                        ANALYSIS_RUN_STUDY_OPERATION,
                        ANALYSIS_RUN_STUDY_OP_VERSION,
                        &study.context,
                        OperationErrorSpec {
                            error_code: "RM.FEA.RUN_STUDY.INVALID_PORTABLE_INTENT",
                            error_type: OperationErrorType::Contract,
                            retryable: false,
                            severity: OperationErrorSeverity::Fatal,
                        },
                        error.to_string(),
                        BTreeMap::new(),
                    )
                })?;
                return Ok(PreparedStructuralStudy {
                    study,
                    execution:
                        structural_study::PreparedStructuralStudyExecution::StructuralPrestressedFrequency {
                            workloads,
                        },
                    solver_mesh,
                });
            }
        }
    }

    if let [preload_step, buckling_step] = study.steps.as_slice() {
        if let (
            AnalysisSolverOptions::StructuralLinear(mut preload_options),
            AnalysisSolverOptions::StructuralBuckling(mut buckling_options),
        ) = (
            preload_step.authored_execution.solver_options.clone(),
            buckling_step.authored_execution.solver_options.clone(),
        ) {
            if preload_options.solver_mesh_artifact_path.is_none() {
                preload_options.solver_mesh_artifact_path = study.solver_mesh_artifact_path.clone();
            }
            let preload = prepare_linear_static_step_run(
                &study.model,
                &preload_step.procedure_step.step_id,
                preload_step.authored_execution.backend,
                preload_step.authored_execution.requirement,
                preload_options,
                study.context.clone(),
            )?;
            buckling_options.solver_mesh_artifact_path = None;
            let buckling = structural_study::prepare_structural_buckling_template(
                &study,
                buckling_step,
                &buckling_options,
            )?;
            let workloads = BTreeMap::from([
                (
                    preload_step.procedure_step.step_id.clone(),
                    runmat_analysis_fea::execution::StructuralStepWorkload::Linear {
                        workload: Box::new(preload.workload),
                    },
                ),
                (
                    buckling_step.procedure_step.step_id.clone(),
                    runmat_analysis_fea::execution::StructuralStepWorkload::Buckling {
                        template: Box::new(buckling),
                    },
                ),
            ]);
            runmat_analysis_fea::execution::StructuralStudyWorkloads::HeterogeneousSequence {
                workloads: workloads.clone(),
            }
            .validate()
            .map_err(|error| {
                operation_error(
                    ANALYSIS_RUN_STUDY_OPERATION,
                    ANALYSIS_RUN_STUDY_OP_VERSION,
                    &study.context,
                    OperationErrorSpec {
                        error_code: "RM.FEA.RUN_STUDY.INVALID_PORTABLE_INTENT",
                        error_type: OperationErrorType::Contract,
                        retryable: false,
                        severity: OperationErrorSeverity::Fatal,
                    },
                    error.to_string(),
                    BTreeMap::new(),
                )
            })?;
            return Ok(PreparedStructuralStudy {
                study,
                execution: structural_study::PreparedStructuralStudyExecution::StructuralBuckling {
                    workloads,
                },
                solver_mesh,
            });
        }
    }

    if let [modal_step, transient_step] = study.steps.as_slice() {
        if let (
            AnalysisSolverOptions::StructuralModes(mut modal_options),
            AnalysisSolverOptions::StructuralTransient(mut transient_options),
        ) = (
            modal_step.authored_execution.solver_options.clone(),
            transient_step.authored_execution.solver_options.clone(),
        ) {
            modal_options.solver_mesh_artifact_path = None;
            transient_options.solver_mesh_artifact_path = None;
            let modal = structural_study::prepare_structural_modal_workload(
                &study,
                modal_step,
                &modal_options,
            )?;
            let transient = structural_study::prepare_structural_modal_transient_template(
                &study,
                transient_step,
                &transient_options,
            )?;
            let workloads = BTreeMap::from([
                (
                    modal_step.procedure_step.step_id.clone(),
                    runmat_analysis_fea::execution::StructuralStepWorkload::ModalExtraction {
                        workload: Box::new(modal),
                    },
                ),
                (
                    transient_step.procedure_step.step_id.clone(),
                    runmat_analysis_fea::execution::StructuralStepWorkload::ModalTransient {
                        template: Box::new(transient),
                        block_plan:
                            runmat_analysis_fea::execution::StructuralTransientBlockPlan::new(
                                transient_options.maximum_intervals_per_block,
                            )
                            .map_err(|error| {
                                operation_error(
                                    ANALYSIS_RUN_STUDY_OPERATION,
                                    ANALYSIS_RUN_STUDY_OP_VERSION,
                                    &study.context,
                                    OperationErrorSpec {
                                        error_code: "RM.FEA.RUN_STUDY.INVALID_PORTABLE_INTENT",
                                        error_type: OperationErrorType::Contract,
                                        retryable: false,
                                        severity: OperationErrorSeverity::Fatal,
                                    },
                                    error.to_string(),
                                    BTreeMap::from([(
                                        "study_id".to_string(),
                                        study.spec.study_id.clone(),
                                    )]),
                                )
                            })?,
                    },
                ),
            ]);
            return Ok(PreparedStructuralStudy {
                study,
                execution:
                    structural_study::PreparedStructuralStudyExecution::StructuralModalTransient {
                        workloads,
                    },
                solver_mesh,
            });
        }
    }

    if let [step] = study.steps.as_slice() {
        if let AnalysisSolverOptions::StructuralFrequency(mut options) =
            step.authored_execution.solver_options.clone()
        {
            options.solver_mesh_artifact_path = None;
            let workload =
                structural_study::prepare_structural_frequency_workload(&study, step, &options)?;
            return Ok(PreparedStructuralStudy {
                study,
                execution:
                    structural_study::PreparedStructuralStudyExecution::StructuralFrequency {
                        workload: Box::new(workload),
                    },
                solver_mesh,
            });
        }
    }

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
    Ok(PreparedStructuralStudy {
        study,
        execution: structural_study::PreparedStructuralStudyExecution::LinearSequence { linears },
        solver_mesh,
    })
}

pub fn analysis_complete_structural_study_op(
    mut prepared: PreparedStructuralStudy,
    outcome: StructuralStudyExecutionOutcome,
) -> Result<OperationEnvelope<AnalysisStudyRunData>, OperationErrorEnvelope> {
    let result = match outcome {
        Ok(StructuralStudyNumericalResult::ThermalPrestress { .. }) => {
            return Err(structural_study_completion_failure(
                &prepared,
                "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
                OperationErrorType::Contract,
                OperationErrorSeverity::Fatal,
                "thermal prestress requires typed compiled-study publication",
            ))
        }
        Ok(StructuralStudyNumericalResult::LinearSequence(result)) => result,
        Ok(StructuralStudyNumericalResult::StructuralTransient(_)) => {
            return Err(structural_study_completion_failure(
                &prepared,
                "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
                OperationErrorType::Contract,
                OperationErrorSeverity::Fatal,
                "linear structural-study completion received a direct-transient result",
            ))
        }
        Ok(StructuralStudyNumericalResult::StructuralModal(result)) => {
            let workload = match &prepared.execution {
                structural_study::PreparedStructuralStudyExecution::StructuralModal {
                    workload,
                } => workload.as_ref().clone(),
                _ => {
                    return Err(structural_study_completion_failure(
                        &prepared,
                        "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
                        OperationErrorType::Contract,
                        OperationErrorSeverity::Fatal,
                        "structural modal result differs from the prepared study kind",
                    ));
                }
            };
            return structural_study::complete_modal_study(
                study_completion_from_prepared(prepared.study),
                prepared.solver_mesh,
                workload,
                *result,
                storage::next_run_id(),
            );
        }
        Ok(StructuralStudyNumericalResult::StructuralModalTransient(result)) => {
            let workloads = match &prepared.execution {
                structural_study::PreparedStructuralStudyExecution::StructuralModalTransient {
                    workloads,
                } => workloads.clone(),
                _ => {
                    return Err(structural_study_completion_failure(
                        &prepared,
                        "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
                        OperationErrorType::Contract,
                        OperationErrorSeverity::Fatal,
                        "modal-transient result differs from the prepared study kind",
                    ));
                }
            };
            return structural_study::complete_modal_transient_study(
                study_completion_from_prepared(prepared.study),
                prepared.solver_mesh,
                workloads,
                *result,
                storage::next_run_id(),
                storage::next_run_id(),
            );
        }
        Ok(StructuralStudyNumericalResult::StructuralModalResponseSpectrum(result)) => {
            let workloads = match &prepared.execution {
                structural_study::PreparedStructuralStudyExecution::StructuralModalResponseSpectrum {
                    workloads,
                } => workloads.clone(),
                _ => {
                    return Err(structural_study_completion_failure(
                        &prepared,
                        "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
                        OperationErrorType::Contract,
                        OperationErrorSeverity::Fatal,
                        "modal response-spectrum result differs from the prepared study kind",
                    ));
                }
            };
            return structural_study::complete_response_spectrum_study(
                study_completion_from_prepared(prepared.study),
                prepared.solver_mesh,
                workloads,
                *result,
                storage::next_run_id(),
                storage::next_run_id(),
            );
        }
        Ok(StructuralStudyNumericalResult::StructuralRandomResponse(result)) => {
            let workloads = match &prepared.execution {
                structural_study::PreparedStructuralStudyExecution::StructuralRandomResponse {
                    workloads,
                } => workloads.clone(),
                _ => {
                    return Err(structural_study_completion_failure(
                        &prepared,
                        "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
                        OperationErrorType::Contract,
                        OperationErrorSeverity::Fatal,
                        "random-response result differs from the prepared study kind",
                    ));
                }
            };
            return structural_study::complete_random_response_study(
                study_completion_from_prepared(prepared.study),
                prepared.solver_mesh,
                workloads,
                *result,
                storage::next_run_id(),
                storage::next_run_id(),
            );
        }
        Ok(StructuralStudyNumericalResult::StructuralBuckling(result)) => {
            let workloads = match &prepared.execution {
                structural_study::PreparedStructuralStudyExecution::StructuralBuckling {
                    workloads,
                } => workloads.clone(),
                _ => {
                    return Err(structural_study_completion_failure(
                        &prepared,
                        "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
                        OperationErrorType::Contract,
                        OperationErrorSeverity::Fatal,
                        "structural buckling result differs from the prepared study kind",
                    ));
                }
            };
            return structural_study::complete_buckling_study(
                study_completion_from_prepared(prepared.study),
                prepared.solver_mesh,
                workloads,
                *result,
                storage::next_run_id(),
                storage::next_run_id(),
            );
        }
        Ok(StructuralStudyNumericalResult::StructuralPrestressedModal(result)) => {
            let workloads = match &prepared.execution {
                structural_study::PreparedStructuralStudyExecution::StructuralPrestressedModal {
                    workloads,
                } => workloads.clone(),
                _ => {
                    return Err(structural_study_completion_failure(
                        &prepared,
                        "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
                        OperationErrorType::Contract,
                        OperationErrorSeverity::Fatal,
                        "prestressed-modal result differs from the prepared study kind",
                    ));
                }
            };
            return structural_study::complete_prestressed_modal_study(
                study_completion_from_prepared(prepared.study),
                prepared.solver_mesh,
                workloads,
                *result,
                storage::next_run_id(),
                storage::next_run_id(),
            );
        }
        Ok(StructuralStudyNumericalResult::StructuralPrestressedFrequency(result)) => {
            let modal_run_id = matches!(
                &result.reduction,
                StructuralStudyPrestressedFrequencyReductionResult::ModalReduced(_)
            )
            .then(storage::next_run_id);
            let workloads = match &prepared.execution {
                structural_study::PreparedStructuralStudyExecution::StructuralPrestressedFrequency {
                    workloads,
                } => workloads.clone(),
                _ => {
                    return Err(structural_study_completion_failure(
                        &prepared,
                        "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
                        OperationErrorType::Contract,
                        OperationErrorSeverity::Fatal,
                        "prestressed-frequency result differs from the prepared study kind",
                    ));
                }
            };
            return structural_study::complete_prestressed_frequency_study(
                study_completion_from_prepared(prepared.study),
                prepared.solver_mesh,
                workloads,
                *result,
                storage::next_run_id(),
                modal_run_id,
                storage::next_run_id(),
            );
        }
        Ok(StructuralStudyNumericalResult::StructuralFrequency(result)) => {
            let workload = match &prepared.execution {
                structural_study::PreparedStructuralStudyExecution::StructuralFrequency {
                    workload,
                } => workload.as_ref().clone(),
                structural_study::PreparedStructuralStudyExecution::LinearSequence { .. } => {
                    return Err(structural_study_completion_failure(
                        &prepared,
                        "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
                        OperationErrorType::Contract,
                        OperationErrorSeverity::Fatal,
                        "linear structural-study completion received a structural-frequency result",
                    ));
                }
                structural_study::PreparedStructuralStudyExecution::StructuralModal { .. }
                | structural_study::PreparedStructuralStudyExecution::StructuralModalTransient {
                    ..
                }
                | structural_study::PreparedStructuralStudyExecution::StructuralModalResponseSpectrum {
                    ..
                }
                | structural_study::PreparedStructuralStudyExecution::StructuralRandomResponse {
                    ..
                }
                | structural_study::PreparedStructuralStudyExecution::StructuralBuckling {
                    ..
                }
                | structural_study::PreparedStructuralStudyExecution::StructuralPrestressedModal {
                    ..
                }
                | structural_study::PreparedStructuralStudyExecution::StructuralPrestressedFrequency {
                    ..
                } => {
                    return Err(structural_study_completion_failure(
                        &prepared,
                        "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
                        OperationErrorType::Contract,
                        OperationErrorSeverity::Fatal,
                        "modal structural-study completion received a structural-frequency result",
                    ));
                }
            };
            return structural_study::complete_frequency_study(
                study_completion_from_prepared(prepared.study),
                prepared.solver_mesh,
                workload,
                *result,
                storage::next_run_id(),
            );
        }
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
    let structural_study::PreparedStructuralStudyExecution::LinearSequence { linears } =
        &mut prepared.execution
    else {
        return Err(structural_study_completion_failure(
            &prepared,
            "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
            OperationErrorType::Contract,
            OperationErrorSeverity::Fatal,
            "structural-frequency study completion received a linear sequence result",
        ));
    };
    let mut completed = Vec::with_capacity(result.steps.len());
    for step in result.steps {
        let linear = linears.remove(&step.step_id).ok_or_else(|| {
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
        step.numerical_receipt
            .validate_against(step.solver_result_identity, &step.result)
            .map_err(|error| study_sequence_contract_error(&prepared.study.context, error))?;
        let run = analysis_admit_linear_static_run_with_id(linear, step.result.run, None)?;
        completed.push(CompletedStudyStep {
            step_id: step.step_id,
            solver_result_identity: step.solver_result_identity,
            run: CompletedStudyStepRun::Standard(Box::new(run)),
            resolved_solver_options,
        });
    }
    if !linears.is_empty() {
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
            states.push(
                CanonicalSemanticState::new(CanonicalSemanticStateInput {
                    producer: step_identity,
                    declaration: declaration.clone(),
                    topology_digest,
                    source_result: solver_result_identity,
                    payload: semantic_state_payload(&run.data, declaration)
                        .map_err(|error| study_sequence_contract_error(&prepared.context, error))?,
                })
                .map_err(|error| study_sequence_contract_error(&prepared.context, error))?,
            );
        }
        completed.push(CompletedStudyStep {
            step_id: step.procedure_step.step_id.clone(),
            solver_result_identity,
            run: CompletedStudyStepRun::Standard(Box::new(run)),
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

fn semantic_state_payload(
    run: &AnalysisRunResult,
    declaration: &StateOutput,
) -> Result<CanonicalSemanticStatePayload, String> {
    if declaration.kind == StateKind::ModalBasis {
        let basis = run
            .modal_results
            .as_ref()
            .map(|modal| modal.basis.clone())
            .ok_or_else(|| {
                "modal procedure did not produce its declared modal-basis state".to_string()
            })?;
        Ok(CanonicalSemanticStatePayload::ModalBasis {
            basis: Box::new(basis),
        })
    } else {
        Ok(CanonicalSemanticStatePayload::Fields {
            fields: collect_semantic_state_fields(run),
        })
    }
}

fn collect_semantic_state_fields(run: &AnalysisRunResult) -> Vec<AnalysisField> {
    collect_analysis_result_fields(run)
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
        .collect()
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
    let aliases = runmat_analysis_core::GeometryRegionAliases::new(study_mesh.region_aliases)
        .map_err(|error| {
            operation_error(
                ANALYSIS_RUN_STUDY_OPERATION,
                ANALYSIS_RUN_STUDY_OP_VERSION,
                &prepared.context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RUN_STUDY.TOPOLOGY_RESOLUTION_FAILED",
                    error_type: OperationErrorType::Validation,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                error.to_string(),
                BTreeMap::new(),
            )
        })?;
    prepared.model =
        runmat_analysis_core::resolve_geometry_region_aliases(&prepared.model, &aliases).map_err(
            |error| {
                operation_error(
                    ANALYSIS_RUN_STUDY_OPERATION,
                    ANALYSIS_RUN_STUDY_OP_VERSION,
                    &prepared.context,
                    OperationErrorSpec {
                        error_code: "RM.FEA.RUN_STUDY.TOPOLOGY_RESOLUTION_FAILED",
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
    let model = spec.model.clone();
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

    if options.truncation.requested_mode_count() == 0 {
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
            "fea.run_modal options require a requested modal count greater than zero",
            BTreeMap::from([(
                "requested_mode_count".to_string(),
                options.truncation.requested_mode_count().to_string(),
            )]),
        ));
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
            truncation: options.truncation,
            convergence: options.convergence,
            residual_vector_directions: options.residual_vector_directions.clone(),
            solver_mesh: solver_mesh.clone(),
        },
    )
    .map_err(|err| map_fea_run_error(AnalysisOperation::Modal, model, &context, err))?;

    analysis_admit_modal_run_with_id(modal_run, backend, options, context, solver_mesh, None)
}

fn analysis_admit_modal_run_with_id(
    modal_run: runmat_analysis_fea::FeaModalRunResult,
    backend: ComputeBackend,
    options: AnalysisModalRunOptions,
    context: OperationContext,
    solver_mesh: Option<SolverMeshArtifact>,
    assigned_run_id: Option<&str>,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    let basis = modal_run.basis;
    let run = modal_run.run;
    let mode_shapes = modal_run.mode_shapes;
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
    let result_quality = if basis.modes().is_empty() {
        QualityGate::Fail
    } else if basis
        .modes()
        .iter()
        .map(|mode| mode.residual_norm())
        .fold(0.0_f64, f64::max)
        > options.convergence.relative_residual_tolerance()
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
                options.convergence.relative_residual_tolerance()
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
        run_id: assigned_run_id
            .map(str::to_string)
            .unwrap_or_else(storage::next_run_id),
        run,
        render_topology: render_topology_from_solver_mesh(solver_mesh.as_ref()),
        modal_results: Some(ModalResultsData {
            mode_indices: (0..basis.modes().len()).collect(),
            mode_shapes,
            basis,
            mode_units: ModalFrequencyUnits::Hz,
            frequency_basis,
        }),
        buckling_results: None,
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

fn analysis_admit_buckling_run_with_id(
    buckling: FeaBucklingRunResult,
    options: AnalysisBucklingRunOptions,
    context: OperationContext,
    solver_mesh: SolverMeshArtifact,
    assigned_run_id: &str,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    let FeaBucklingRunResult {
        run,
        load_factors,
        residual_norms,
        mode_shapes,
        rotation_mode_shapes,
        ..
    } = buckling;
    let maximum_residual = residual_norms.iter().copied().fold(0.0_f64, f64::max);
    let result_quality = if load_factors.is_empty()
        || load_factors
            .iter()
            .any(|factor| !factor.is_finite() || *factor <= 0.0)
        || maximum_residual > options.solver.relative_tolerance()
    {
        QualityGate::Fail
    } else {
        QualityGate::Pass
    };
    let solver_convergence = result_quality;
    let mut quality_reasons = Vec::new();
    if result_quality != QualityGate::Pass {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::SolverNotConverged,
            detail: format!(
                "buckling residual {} exceeds threshold {} or no positive mode was retained",
                maximum_residual,
                options.solver.relative_tolerance()
            ),
        });
    }
    let numerically_acceptable = result_quality == QualityGate::Pass;
    let (production_publishable, production_publication_reasons) =
        evaluate_production_publication(&run, numerically_acceptable);
    let run_status = classify_run_status(result_quality, numerically_acceptable);
    let provenance = RunProvenance {
        requested_backend: run.requested_backend,
        execution_backend: run.execution_backend,
        solver_backend: run.solver_backend,
        solver_device_apply_k_ratio: run.solver_device_apply_k_ratio,
        solver_host_sync_count: run.solver_host_sync_count,
        precision_mode: contracts::format_precision_mode(options.precision_mode),
        deterministic_mode: options.deterministic_mode,
        solver_method: run.solver_method,
        preconditioner: run.preconditioner,
        quality_policy: contracts::format_quality_policy(options.quality_policy),
        fallback_events: Vec::new(),
    };
    let result = AnalysisRunResult {
        run_id: assigned_run_id.to_string(),
        run,
        render_topology: render_topology_from_solver_mesh(Some(&solver_mesh)),
        modal_results: None,
        buckling_results: Some(BucklingResultsData {
            mode_indices: (1..=load_factors.len()).collect(),
            load_factors,
            residual_norms,
            mode_shapes,
            rotation_mode_shapes,
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
        provenance,
    };
    persist_fea_run_result_with_progress(
        AnalysisOperation::Buckling.name(),
        AnalysisOperation::Buckling.version(),
        "RM.FEA.RUN_BUCKLING.ARTIFACT_STORE_FAILED",
        &context,
        &result,
    )?;
    Ok(OperationEnvelope::new(
        AnalysisOperation::Buckling.name(),
        AnalysisOperation::Buckling.version(),
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
    let has_pressure_source = runmat_analysis_fea::physics::acoustic::has_nonzero_pressure_source(
        model,
        options.mode_count,
    )
    .map_err(|detail| {
        operation_error(
            AnalysisOperation::Acoustic.name(),
            AnalysisOperation::Acoustic.version(),
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_ACOUSTIC.EXCITATION_REALIZATION_FAILED",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            detail,
            BTreeMap::from([("analysis_model_id".to_string(), model.model_id.0.clone())]),
        )
    })?;
    if !has_pressure_source {
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
    )
    .map_err(|detail| {
        operation_error(
            AnalysisOperation::Acoustic.name(),
            AnalysisOperation::Acoustic.version(),
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_ACOUSTIC.EXCITATION_REALIZATION_FAILED",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            detail,
            BTreeMap::from([("analysis_model_id".to_string(), model.model_id.0.clone())]),
        )
    })?;
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
        buckling_results: None,
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
    if let Err(detail) =
        runmat_analysis_fea::physics::cfd::validate_authored_boundary_conditions(model)
    {
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

    let minimum_sample_count = if cfd_procedure == AnalysisProcedure::TransientResponse {
        2
    } else {
        1
    };
    if options.sample_count < minimum_sample_count {
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
            format!(
                "fea.run_cfd options require at least {minimum_sample_count} time samples for this procedure"
            ),
            BTreeMap::from([("sample_count".to_string(), options.sample_count.to_string())]),
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
    let mut run = runmat_analysis_fea::physics::cfd::solve_reduced_finite_volume(
        model,
        cfd_domain,
        backend,
        identity,
        &runmat_analysis_fea::physics::cfd::ReducedCfdSolveOptions {
            sample_count: options.sample_count,
            max_linear_iters: options.max_linear_iters,
            tolerance: options.tolerance,
            residual_warn_threshold: options.residual_warn_threshold,
        },
        solver_mesh.as_ref(),
    )
    .map_err(|detail| {
        operation_error(
            AnalysisOperation::Cfd.name(),
            AnalysisOperation::Cfd.version(),
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_CFD.EXCITATION_REALIZATION_FAILED",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            detail,
            BTreeMap::new(),
        )
    })?;
    let solve_ms = solve_start.elapsed().as_secs_f64() * 1000.0;
    run.diagnostics
        .push(runmat_analysis_fea::diagnostics::FeaDiagnostic {
            code: "FEA_CFD_COST".to_string(),
            severity: runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info,
            message: format!(
                "solve_ms={} sample_count={} max_linear_iters={} tolerance={}",
                solve_ms, options.sample_count, options.max_linear_iters, options.tolerance,
            ),
        });
    let mut fallback_events = Vec::new();
    if backend == ComputeBackend::Gpu && run.execution_backend != ExecutionBackend::Accelerator {
        fallback_events.push(FallbackEvent::solver_backend());
    }

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
        buckling_results: None,
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
    if options.sample_count < 2 || options.max_linear_iters == 0 {
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
            "fea.run_cht options require at least two time samples and max_linear_iters greater than zero",
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

    if model.thermo_mechanical.is_none() {
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
                snapshot_count: options.sample_count,
            },
            residual_target: options.residual_warn_threshold,
            solver_mesh: Some(solver_mesh.clone()),
        },
    )
    .map_err(|err| map_fea_run_error(AnalysisOperation::Cht, model, &context, err))?;

    let topology = CfdDomainTopology::from_model(model, Some(&solver_mesh));
    let coupling = couple_thermal_flow(
        model,
        cfd_domain,
        &topology,
        &thermal_run,
        ChtCouplingOptions {
            sample_count: options.sample_count,
            procedure: cht_procedure,
            residual_warn_threshold: options.residual_warn_threshold,
        },
    )
    .map_err(|detail| {
        operation_error(
            AnalysisOperation::Cht.name(),
            AnalysisOperation::Cht.version(),
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_CHT.COUPLING_FAILED",
                error_type: OperationErrorType::Backend,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            detail,
            BTreeMap::new(),
        )
    })?;
    let max_cfd_momentum_residual = coupling.max_cfd_momentum_residual;
    let max_cfd_continuity_residual = coupling.max_cfd_continuity_residual;
    let max_cht_transport_residual = coupling.max_transport_residual;
    let mut run = thermal_run.run.clone();
    run.identity = identity;
    run.solver_method = SolverMethod::ChtConjugateProjection;
    run.preconditioner = SolverPreconditioner::ThermalCfdProjection;
    run.fields.extend(coupling.fields);
    run.diagnostics.extend(coupling.diagnostics);
    let solve_ms = solve_start.elapsed().as_secs_f64() * 1000.0;
    run.diagnostics
        .push(runmat_analysis_fea::diagnostics::FeaDiagnostic {
            code: "FEA_CHT_COST".to_string(),
            severity: runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info,
            message: format!(
                "solve_ms={} sample_count={} max_linear_iters={} tolerance={}",
                solve_ms, options.sample_count, options.max_linear_iters, options.tolerance,
            ),
        });

    let mut fallback_events = Vec::new();
    if backend == ComputeBackend::Gpu && run.execution_backend != ExecutionBackend::Accelerator {
        fallback_events.push(FallbackEvent::solver_backend());
    }

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
        buckling_results: None,
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
    if options.sample_count < 2 || options.max_linear_iters == 0 {
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
            "fea.run_fsi options require at least two time samples and max_linear_iters greater than zero",
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
    let coupling = couple_fluid_structure(
        model,
        cfd_domain,
        &topology,
        FsiCouplingOptions {
            sample_count: options.sample_count,
            max_linear_iters: options.max_linear_iters,
            tolerance: options.tolerance,
            residual_warn_threshold: options.residual_warn_threshold,
        },
    )
    .map_err(|detail| {
        operation_error(
            AnalysisOperation::Fsi.name(),
            AnalysisOperation::Fsi.version(),
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_FSI.COUPLING_FAILED",
                error_type: OperationErrorType::Backend,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            detail,
            BTreeMap::new(),
        )
    })?;
    let max_cfd_momentum_residual = coupling.max_cfd_momentum_residual;
    let max_cfd_continuity_residual = coupling.max_cfd_continuity_residual;
    let max_interface_residual = coupling.max_interface_residual;
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
        diagnostics: coupling.diagnostics,
        spatial_projection_evidence: Vec::new(),
        structural_static_balance: None,
        structural_constraint_result: None,
        structural_connector_result: None,
        observations: Vec::new(),
        observation_histories: Vec::new(),
        series_reductions: Vec::new(),
        fields: coupling.fields,
    };
    let solve_ms = solve_start.elapsed().as_secs_f64() * 1000.0;
    run.diagnostics
        .push(runmat_analysis_fea::diagnostics::FeaDiagnostic {
            code: "FEA_FSI_COST".to_string(),
            severity: runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info,
            message: format!(
                "solve_ms={} sample_count={} max_linear_iters={} tolerance={}",
                solve_ms, options.sample_count, options.max_linear_iters, options.tolerance,
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
        buckling_results: None,
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
            formulation_requirement,
            procedure: match thermal_procedure {
                AnalysisProcedure::Static => ThermalProcedureControl::Steady,
                AnalysisProcedure::TransientResponse => ThermalProcedureControl::BackwardEuler {
                    snapshot_count: options.snapshot_count,
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
        buckling_results: None,
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
    if let Some((material, thermal)) = model.materials.iter().find_map(|material| {
        let thermal = material.thermal.as_ref()?;
        (!thermal.conductivity_w_per_mk.is_finite()
            || thermal.conductivity_w_per_mk <= 0.0
            || !thermal.specific_heat_j_per_kgk.is_finite()
            || thermal.specific_heat_j_per_kgk <= 0.0)
            .then_some((material, thermal))
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
                    thermal.conductivity_w_per_mk.to_string(),
                ),
                (
                    "specific_heat_j_per_kgk".to_string(),
                    thermal.specific_heat_j_per_kgk.to_string(),
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
            && step.family == PhysicsFamily::Structural
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
            "FEA model must include a structural transient step for fea.run_transient",
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                ("geometry_id".to_string(), model.geometry_id.clone()),
            ]),
        ));
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
            time_step_policy: match options.time_step_policy {
                contracts::AnalysisStructuralTimeStepPolicy::Fixed { interval_count } => {
                    runmat_analysis_fea::solve::transient::StructuralTimeStepPolicy::Fixed {
                        interval_count,
                    }
                }
            },
            residual_target: options.residual_target,
            progress_operation: AnalysisOperation::Transient.name().to_string(),
            solver_mesh: solver_mesh.clone(),
        },
    )
    .map_err(|err| map_fea_run_error(AnalysisOperation::Transient, model, &context, err))?;

    analysis_admit_transient_run_with_id(
        backend,
        options,
        &context,
        solver_mesh,
        transient_run,
        None,
    )
}

fn analysis_admit_transient_run_with_id(
    backend: ComputeBackend,
    options: AnalysisTransientRunOptions,
    context: &OperationContext,
    solver_mesh: Option<SolverMeshArtifact>,
    transient_run: runmat_analysis_fea::FeaTransientRunResult,
    assigned_run_id: Option<&str>,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
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
        run_id: assigned_run_id
            .map(str::to_owned)
            .unwrap_or_else(storage::next_run_id),
        run,
        render_topology: render_topology_from_solver_mesh(solver_mesh.as_ref()),
        modal_results: None,
        buckling_results: None,
        thermal_results: None,
        transient_results: Some(TransientResultsData {
            transient_payload_version: "transient_results/v1".to_string(),
            initial_state: transient_run.initial_state,
            time_points_s: transient_run.time_points_s,
            displacement_snapshots: transient_run.displacement_snapshots,
            rotation_snapshots: transient_run.rotation_snapshots,
            velocity_snapshots: transient_run.velocity_snapshots,
            angular_velocity_snapshots: transient_run.angular_velocity_snapshots,
            acceleration_snapshots: transient_run.acceleration_snapshots,
            angular_acceleration_snapshots: transient_run.angular_acceleration_snapshots,
            strain_snapshots: transient_run.strain_snapshots,
            stress_snapshots: transient_run.stress_snapshots,
            strain_energy_density_snapshots: transient_run.strain_energy_density_snapshots,
            von_mises_snapshots: transient_run.von_mises_snapshots,
            kinetic_energy_snapshots: transient_run.kinetic_energy_snapshots,
            strain_energy_snapshots: transient_run.strain_energy_snapshots,
            residual_norm_snapshots: transient_run.residual_norm_snapshots,
            reaction_force_snapshots: transient_run.reaction_force_snapshots,
            reaction_moment_snapshots: transient_run.reaction_moment_snapshots,
            external_work_snapshots: transient_run.external_work_snapshots,
            support_work_snapshots: transient_run.support_work_snapshots,
            damping_dissipation_snapshots: transient_run.damping_dissipation_snapshots,
            energy_balance_residual_snapshots: transient_run.energy_balance_residual_snapshots,
            relative_energy_balance_error_snapshots: transient_run
                .relative_energy_balance_error_snapshots,
            dynamic_balance_snapshots: transient_run.dynamic_balance_snapshots,
            electro_thermal_temperature_snapshots: transient_run
                .electro_thermal_temperature_snapshots,
            electro_thermal_thermal_residual_snapshots: transient_run
                .electro_thermal_thermal_residual_snapshots,
            residual_norms: transient_run.residual_norms,
            integration_method: TransientIntegrationMethod::NewmarkAverageAcceleration,
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
            precision_mode: contracts::format_precision_mode(PrecisionMode::Fp64),
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
        context,
        &result,
    )?;

    Ok(OperationEnvelope::new(
        AnalysisOperation::Transient.name(),
        AnalysisOperation::Transient.version(),
        context,
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
        buckling_results: None,
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
) -> Result<StructuralLinearWorkload, OperationErrorEnvelope> {
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
        schema_version: 2,
        model: model.clone(),
        step_id: step_id.clone(),
        backend,
        formulation_requirement,
        preconditioner: requested_preconditioner,
        algebra_backend: requested_solver_backend,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
        require_solver_mesh_for_solid: true,
        state_input: None,
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

    let requested_frequency_hz = if options.sweep_enabled {
        options.sweep_frequency_hz.as_slice()
    } else {
        &[]
    };
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
    let frequency_response = run_electromagnetic_frequency_response(
        model,
        backend,
        solve_options,
        requested_frequency_hz,
    )
    .map_err(|error| match error {
        FeaRunError::FrequencyPlan(reason) => operation_error(
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
            BTreeMap::from([("reason".to_string(), reason.to_string())]),
        ),
        FeaRunError::MissingElectromagneticSource { frequency_hz } => operation_error(
            AnalysisOperation::Electromagnetic.name(),
            AnalysisOperation::Electromagnetic.version(),
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_ELECTROMAGNETIC.MISSING_ELECTROMAGNETIC_SOURCE",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "fea.run_electromagnetic requires a nonzero electric-current or current-density excitation at every requested frequency",
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                ("excitation_count".to_string(), model.excitations.len().to_string()),
                ("frequency_hz".to_string(), frequency_hz.to_string()),
            ]),
        ),
        error => map_fea_run_error(AnalysisOperation::Electromagnetic, model, &context, error),
    })?;
    let sweep_frequency_hz = frequency_response.plan().frequencies_hz().to_vec();
    let sweep_metrics = frequency_response.summary().clone();
    let sweep_peak_flux_density = sweep_metrics.peak_flux_density.clone();
    let sweep_solve_quality = sweep_metrics.solve_quality.clone();
    let em_run = frequency_response.into_primary();
    let run = em_run.run.clone();
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
        buckling_results: None,
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
                (
                    "material_target".to_string(),
                    assignment.target_label().to_owned(),
                ),
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
        volume_element_count: mesh.topology.volume_elements.len(),
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
        AnalysisValidationError::InvalidRandomExcitation {
            excitation_id,
            detail,
        } => (
            "RM.FEA.VALIDATE.INVALID_RANDOM_EXCITATION",
            format!("random excitation `{excitation_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("excitation_id".to_string(), excitation_id),
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
