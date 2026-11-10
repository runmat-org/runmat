use runmat_time::Instant;
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use runmat_analysis_core::{
    validate_model_against_geometry, AnalysisConstraint, AnalysisField, AnalysisInterfaceKind,
    AnalysisModel, AnalysisProcedure, BoundaryOperator, CoordinateFrame, CoupledPhysicsFamily,
    ExcitationAction, PhysicsFamily, ProcedureStepId, StateKind, StateOutput,
};
use runmat_analysis_fea::execution::{
    CanonicalSemanticState, CanonicalSemanticStateInput, CanonicalSemanticStatePayload,
    SequenceIdentity, SolverResultIdentity, StepIdentity, StructuralLinearWorkload, StudyIdentity,
};
#[cfg(test)]
use runmat_analysis_fea::physics::cfd::CfdDomainTopology;
use runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind;
use runmat_analysis_fea::solve::preconditioner::SpdPreconditionerKind;
use runmat_analysis_fea::{
    run_modal_with_options, run_nonlinear_with_options, run_transient_with_options,
    AnalysisRunIdentity, ComputeBackend, ExecutionBackend, FeaBucklingRunResult,
    FeaProgressHandler, FeaRunResult, ModalSolveOptions,
};
use runmat_meshing_core::SolverMeshArtifact;

use crate::operations::{
    operation_error, OperationContext, OperationEnvelope, OperationErrorEnvelope,
    OperationErrorSeverity, OperationErrorSpec, OperationErrorType,
};
use policy::{
    breach_rate_greater_than, breach_rate_less_than, thermo_field_coverage_min_for_policy,
    thermo_spatial_gradient_threshold_for_policy, thermo_thresholds_for_policy,
    EM_ASSIGNMENT_COVERAGE_MIN_BALANCED, EM_BOUNDARY_ANCHOR_MIN_BALANCED,
    EM_BOUNDARY_ENERGY_MIN_BALANCED, EM_BOUNDARY_LOCALIZATION_MIN_BALANCED,
    EM_BOUNDARY_PENALTY_CONTRIBUTION_MAX_BALANCED, EM_CONDITIONING_MAX_BALANCED,
    EM_CONDUCTIVITY_SPREAD_THRESHOLD_BALANCED, EM_ENERGY_IMBALANCE_MAX_BALANCED,
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

mod artifact_root;
pub mod fea_run_admission;
mod evidence_filesystem;
use evidence_filesystem::{atomic_write_bytes, fs_create_dir_all};
mod modal_run_admission;
mod render_topology;
use modal_run_admission::analysis_admit_modal_run_with_id;
use render_topology::render_topology_from_solver_mesh;
#[cfg(feature = "plot-core")]
use render_topology::render_topology_from_solver_mesh_at;
pub use render_topology::{DroppedMidsides, SurfaceDetail};
pub mod assessment;
mod nonlinear_result;
pub use artifact_root::{
    AdmittedFrequencySamplePublication, AnalysisArtifactRoot, FrequencySampleArtifactRootInventory,
    FrequencySampleArtifactRoots, FrequencySamplePublicationAccess, ObservationHistoryArtifactRoot,
};
mod contracts;
#[cfg(test)]
mod excitation_authoring;
mod execution_dispatch;
mod execution_error;
mod validation_error;
use validation_error::map_validate_error;
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
mod prepared_study_publication;
pub mod render_tiles;
mod response_study_result;
mod result_policy;
mod solver_mesh_artifact;
mod source;
pub mod storage;
mod structural_owned_completion;
mod structural_study;
pub use structural_owned_completion::analysis_complete_structural_study_op;
mod study_completion_identity;
mod study_contract;
mod study_execution;
mod study_placement;
mod study_publication;
pub use prepared_study_publication::{
    assessment_evidence_identity, AssessmentExecutionReceipt, PreparedStudyPublication,
};
mod study_result;
pub mod study_result_manifest;
mod study_sweep;
mod thermal_execution;
pub use fea_run_admission::entrypoints::analysis_run_thermal_with_options_op;
pub use response_study_result::{
    StructuralStudyModalResponseSpectrumNumericalResult,
    StructuralStudyRandomResponseNumericalResult,
};
pub mod value_admission;

#[cfg(test)]
use excitation_authoring::{append_constant_excitation, ConstantExcitation};
use execution_dispatch::{execution_operation, run_operation_for_identity, AnalysisOperation};
pub use execution_error::ACCELERATOR_REQUIRED_ERROR_CODE;
use execution_error::{admit_runtime_proxy_for_operation, map_fea_run_error};
use formulation_admission::RuntimeProxyFormulation;
use linear_static_execution::analysis_run_linear_static_with_executor;
use result_policy::classify_run_status;
pub use structural_study::{
    analysis_complete_structural_study_intent_op, analysis_prepare_structural_study_intent_op,
    structural_study_execution_failure, PreparedStructuralStudy, PreparedStructuralStudyIntent,
    StructuralStudyExecutionFailure, StructuralStudyExecutionOutcome,
    STRUCTURAL_STUDY_INTENT_MAX_BYTES,
};
use structural_study::{
    complete_study_run, persist_content_addressed_study_evidence, study_completion_from_prepared,
    study_evidence_root, CompletedStudyStep, CompletedStudyStepRun, PreparedStudySequence,
    PreparedStudyStep,
};
#[cfg(test)]
use structural_study::{persist_immutable_study_evidence, run_evidence_stage};
pub use study_completion_identity::split_attempt_run_id;
pub use study_completion_identity::AnalysisStudyCompletionIdentity;
pub use study_contract::analysis_study_content_digest;
use study_contract::{
    sanitize_study_sweep_id, study_fingerprint, study_fingerprint_from_digest, study_issue_message,
    study_operation_sequence, validate_study_issue_codes,
};
use study_execution::{
    collect_semantic_state_fields, execute_study_step, in_process_study_topology_digest,
};
mod study_step_execution;
pub use study_placement::{
    analysis_execute_structural_study_with_placement,
    analysis_execute_structural_sweep_with_placement, StructuralStudyPlacementEntry,
    StructuralStudyPlacementExecutor, StructuralStudyPlacementFuture,
    StructuralStudyPlacementResult,
};
pub use study_publication::*;
pub use study_result::*;
pub use study_step_execution::{
    analysis_execute_study_step, analysis_execute_study_step_with_capacity,
    analysis_execute_study_sweep_with_step_executor,
    analysis_execute_study_with_step_executor, analysis_validate_study_step_transfer,
    AnalysisStudyStepExecutor, AnalysisStudyStepFuture, AnalysisStudyStepHost,
    AnalysisStudyStepRequest, AnalysisStudyStepResult,
};
pub use study_sweep::{
    analysis_complete_study_sweep_op, analysis_study_sweep_definition,
    analysis_study_sweep_identity, AnalysisStudySweepRunOutcome,
};

mod configuration;
use configuration::current_fea_runtime_config;
pub use configuration::{
    configure_fea_runtime, default_fea_artifact_root, replace_fea_runtime, FeaRuntimeConfig,
};

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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StructuralStudyStepNumericalResult {
    pub step_id: ProcedureStepId,
    pub solver_result_identity: runmat_analysis_fea::execution::SolverResultIdentity,
    pub result: runmat_analysis_fea::execution::StructuralLinearWorkloadResult,
    pub numerical_receipt: runmat_analysis_fea::execution::StructuralLinearNumericalReceiptArtifact,
    /// The static solve path the step's admitted plan selected; `None` when unknown.
    /// Provenance only, never identity.
    #[serde(default)]
    pub static_solve_path: Option<runmat_analysis_fea::planning::resource::StaticSolvePath>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StructuralStudySequenceNumericalResult {
    pub frozen_project_digest: Option<runmat_execution::Digest>,
    /// The effective memory ceiling the local placement admitted, min(authored, machine);
    /// `None` for imported or remote completions. Provenance only, never identity.
    pub effective_memory_ceiling_bytes: Option<u64>,
    pub steps: Vec<StructuralStudyStepNumericalResult>,
    pub states: Vec<runmat_analysis_fea::execution::CanonicalSemanticState>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct StructuralStudyTransientNumericalResult {
    pub frozen_project_digest: Option<runmat_execution::Digest>,
    pub step_id: ProcedureStepId,
    pub result: runmat_analysis_fea::execution::StructuralTransientSolverResult,
    pub numerical_receipt: runmat_analysis_fea::execution::StructuralTransientNumericalReceipt,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct StructuralStudyModalNumericalResult {
    pub frozen_project_digest: Option<runmat_execution::Digest>,
    pub step_id: ProcedureStepId,
    pub result: runmat_analysis_fea::execution::StructuralModalSolverResult,
    pub numerical_receipt: runmat_analysis_fea::execution::StructuralModalNumericalReceipt,
}

#[derive(Debug, Clone, serde::Serialize)]
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

#[derive(Debug, Clone, serde::Serialize)]
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

#[derive(Debug, Clone, serde::Serialize)]
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

mod study_numerical_result;
pub use study_numerical_result::StructuralStudyNumericalResult;

pub use frequency_result::{
    StructuralFrequencyArtifactRootIndex, StructuralFrequencyFinalizedHistoryArtifactRoots,
    StructuralFrequencyHistoryArtifactRoot, StructuralStudyFrequencyNumericalResult,
    StructuralStudyModalFrequencyNumericalResult, StructuralStudyPrestressedFrequencyNumericalResult,
    StructuralStudyPrestressedFrequencyReductionResult,
    StructuralStudyPrestressedModalFrequencyReductionResult,
};

impl PreparedStructuralLinearRun {
    pub fn workload(&self) -> &StructuralLinearWorkload {
        &self.workload
    }

    pub fn solver_mesh(&self) -> Option<&SolverMeshArtifact> {
        self.solver_mesh.as_ref()
    }
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
            .map(runmat_analysis_fea::FeaCancellationToken::from_token)
            .unwrap_or_default(),
        attempt_id: None,
    })
}

pub use contracts::{
    analysis_authoring_workflow, analysis_language_reference_paths, analysis_output_operation_ids,
    analysis_procedure_ids, AcousticFrequencyEntryEvidence, AcousticFrequencyResultsData,
    AnalysisAcousticRunOptions, AnalysisBucklingRunOptions, AnalysisCfdRunOptions,
    AnalysisDiagnosticsArtifactPayload, AnalysisDocumentCheckResult, AnalysisDocumentKind,
    AnalysisDocumentRunResult, AnalysisExecutionCandidate, AnalysisExecutionCapability,
    AnalysisExecutionChoices, AnalysisExecutionDuration, AnalysisExecutionMode,
    AnalysisExecutionPlan, AnalysisExecutionReadiness, AnalysisExecutionStep, AnalysisFieldBinding,
    AnalysisFieldBindingsArtifactPayload, AnalysisFieldDescriptor, AnalysisFieldKind,
    AnalysisFieldLocation, AnalysisFieldPageResult, AnalysisFieldPageValues,
    AnalysisFieldPagingDescriptor, AnalysisFieldRequestOptions, AnalysisFieldSelection,
    AnalysisFieldStorage, AnalysisFieldStorageRef,
    AnalysisImmutableValueSpec, AnalysisLineParametersRunOptions, AnalysisModalRunOptions,
    AnalysisModelSourceEvidence,
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
    AnalysisValidateResult, BucklingResultsData, ContactEnforcementOptions, FallbackEvent,
    FallbackReason, FallbackStage, ModalFrequencyBasis, ModalFrequencyUnits, ModalResultsData,
    NonlinearMethod, NonlinearResultsData, PrecisionMode, PreconditionerMode, QualityGate,
    QualityPolicy, QualityReason, QualityReasonCode, RunProvenance, RunStatus,
    StaticSolvePlacement, StructuralFrequencyResultsData, StructuralRandomResponseResultsData,
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
    AnalysisGeneratedFigure, AnalysisGeneratedFigureKind,
};

/// Which mesh a result figure is drawn on; request data, independent of plotting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisFigureMeshSource {
    Auto,
    Solver,
    Cad,
    CadReference,
}

pub use operations::{
    analysis_field_page_by_run_id_op, analysis_observation_history_sample_by_run_id_op,
    analysis_results_by_run_id_op, analysis_results_compare_op, analysis_results_op,
    analysis_trends_op, bounded_default_results_query, RESULTS_INLINE_FIELD_BYTES,
};
pub(crate) use operations::{analysis_run_field_ids, collect_analysis_result_fields};
pub use runmat_analysis_fea::{FeaProgressEvent, FeaProgressPhase, FeaProgressStatus};
pub use runmat_analysis_language::{
    operation_pattern_from_value_fact, remote_execution_profile_facts, ExecutionProfileFacts,
    FunctionFacts, FunctionRequest, LogicalSourcePath, SourceDigest,
};
pub use source::{
    check_fea_path, check_fea_path_in_root, check_fea_source_in_root, check_fea_sources_in_root,
    logical_source_path, physical_source_path, prepare_fea_path_in_root,
    prepare_fea_sources_in_root, source_digest, CheckedFeaSource, FeaBrowseOutcome, FeaBrowsePage,
    FeaBrowseRequest, FeaCheckStatus, FeaDeclaredMaterial, FeaExecutionBinding, FeaExecutionTarget,
    FeaModelMemberKind, FeaQueryCursor, FeaQueryDetail, FeaQueryError, FeaQueryFilter,
    FeaRootSelectionError, FeaRootSelector, FeaSemanticProgram, FeaSourceCheckResult,
    FeaSourceIdentity, FeaSourceIndex, FeaSourceIndexEntry, FeaSourceSnapshotBuilder,
    FeaSourceSnapshotRequests, FeaStudyMemberKind, FeaWorkspaceCandidate, FeaWorkspaceEdit,
    FeaWorkspaceError, FeaWorkspaceFile, FeaWorkspaceSnapshot, FilesystemFeaHost,
    PreparedFeaSource, PreparedFeaSourceSnapshot, ProjectFeaHost, ResolvedFeaRoot,
};

const ANALYSIS_VALIDATE_STUDY_OPERATION: &str = "fea.validate_study";
const ANALYSIS_VALIDATE_STUDY_OP_VERSION: &str = "fea.validate_study/v1";
const ANALYSIS_PLAN_STUDY_OPERATION: &str = "fea.plan_study";
const ANALYSIS_PLAN_STUDY_OP_VERSION: &str = "fea.plan_study/v1";
const ANALYSIS_PLAN_STUDY_SWEEP_OPERATION: &str = "fea.plan_study_sweep";
const ANALYSIS_PLAN_STUDY_SWEEP_OP_VERSION: &str = "fea.plan_study_sweep/v1";
pub const ANALYSIS_RUN_STUDY_OPERATION: &str = "fea.run_study";
pub const ANALYSIS_RUN_STUDY_OP_VERSION: &str = "fea.run_study/v1";
const ANALYSIS_VALIDATE_STUDY_SWEEP_OPERATION: &str = "fea.validate_study_sweep";
const ANALYSIS_VALIDATE_STUDY_SWEEP_OP_VERSION: &str = "fea.validate_study_sweep/v1";
pub(crate) const ANALYSIS_RUN_STUDY_SWEEP_OPERATION: &str = "fea.run_study_sweep";
pub(crate) const ANALYSIS_RUN_STUDY_SWEEP_OP_VERSION: &str = "fea.run_study_sweep/v1";
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

mod standalone_publication;
pub use fea_run_admission::entrypoints::{
    analysis_run_acoustic_with_options_op, analysis_run_modal_with_options_op,
    analysis_run_nonlinear_with_options_op, analysis_run_transient_with_options_op,
};
use standalone_publication::{persist_fea_run_result_with_progress, publish_standalone_run};

pub fn analysis_validate_study_op(
    spec: &AnalysisStudySpec,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisStudyValidateResult>, OperationErrorEnvelope> {
    let mut issue_codes = validate_study_issue_codes(spec);
    let model_validation_detail =
        validate_model_against_geometry(&spec.model, &CoordinateFrame::Global)
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

fn analysis_run_study_with_structural_executor(
    spec: &AnalysisStudySpec,
    context: OperationContext,
    structural_executor: Option<&dyn StructuralLinearExecutor>,
) -> Result<OperationEnvelope<AnalysisStudyRunData>, OperationErrorEnvelope> {
    fea_run_admission::entrypoints::run(
        fea_run_admission::FeaRunKind::MeshAndSolve,
        analysis_study_content_digest(spec),
        ANALYSIS_RUN_STUDY_OPERATION,
        ANALYSIS_RUN_STUDY_OP_VERSION,
        &context,
        || {
            analysis_run_study_with_structural_executor_admitted(
                spec,
                context.clone(),
                structural_executor,
            )
        },
    )
}

fn analysis_run_study_with_structural_executor_admitted(
    spec: &AnalysisStudySpec,
    context: OperationContext,
    structural_executor: Option<&dyn StructuralLinearExecutor>,
) -> Result<OperationEnvelope<AnalysisStudyRunData>, OperationErrorEnvelope> {
    let prepared = prepare_study_run(spec, context)?;
    let mut completion = study_step_execution::StudyStepCompletion::new(&prepared)?;
    for step in &prepared.steps {
        let (run, resolved_solver_options) = execute_study_step(
            &prepared.model,
            &step.procedure_step,
            &step.authored_execution,
            prepared.solver_mesh_artifact_path.as_deref(),
            &prepared.context,
            structural_executor,
        )?;
        completion.push(&prepared, step, run, resolved_solver_options)?;
    }
    completion.finish(prepared)
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
    let study_mesh = meshing::resolve_study_mesh(spec).map_err(|error| match error {
        meshing::StudyMeshingError::Execution(error) => {
            meshing::async_meshing_error(error, &prepared.context)
        }
        meshing::StudyMeshingError::Message(message) => operation_error(
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
        ),
    })?;
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
    analysis_validate(&model, &CoordinateFrame::Global, context.clone())?;
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
    let identity = analysis_study_sweep_identity(spec, &context)?;
    fea_run_admission::entrypoints::run_classified(
        fea_run_admission::FeaRunKind::MeshAndSolve,
        identity.digest(),
        ANALYSIS_RUN_STUDY_SWEEP_OPERATION,
        ANALYSIS_RUN_STUDY_SWEEP_OP_VERSION,
        &context,
        || analysis_run_study_sweep_admitted(spec, context.clone()),
        |result| fea_run_admission::entrypoints::sweep_outcome(&result.data),
    )
}

fn analysis_run_study_sweep_admitted(
    spec: &AnalysisStudySweepSpec,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisStudySweepData>, OperationErrorEnvelope> {
    let mut outcomes = Vec::with_capacity(spec.studies.len());
    for study in &spec.studies {
        match analysis_run_study_op(study, context.clone()) {
            Ok(run) => outcomes.push(AnalysisStudySweepRunOutcome::Succeeded(Box::new(run))),
            Err(error) => {
                outcomes.push(
                    if error.error_type == OperationErrorType::Cancelled
                        || crate::interrupt::is_cancelled()
                    {
                        AnalysisStudySweepRunOutcome::Cancelled(error)
                    } else {
                        AnalysisStudySweepRunOutcome::Failed(error)
                    },
                );
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
    geometry_frame: &CoordinateFrame,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisValidateResult>, OperationErrorEnvelope> {
    validate_model_against_geometry(model, geometry_frame)
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

fn execute_modal_with_options(
    model: &AnalysisModel,
    backend: ComputeBackend,
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
            truncation: options.truncation,
            convergence: options.convergence,
            residual_vector_directions: options.residual_vector_directions.clone(),
            solver_mesh: solver_mesh.clone(),
        },
    )
    .map_err(|err| map_fea_run_error(AnalysisOperation::Modal, model, &context, err))?;

    analysis_admit_modal_run_with_id(
        modal_run,
        backend,
        options,
        context,
        solver_mesh.as_ref(),
        None,
    )
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
        effective_memory_ceiling_bytes: None,
        static_solve_path: None,
    };
    let result = AnalysisRunResult {
        observation_scalars: Vec::new(),
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
        model_validity: QualityGate::Pass,
        solver_convergence,
        result_quality,
        run_status,
        numerically_acceptable,
        quality_reasons,
        provenance,
    };

    Ok(OperationEnvelope::new(
        AnalysisOperation::Buckling.name(),
        AnalysisOperation::Buckling.version(),
        &context,
        result,
    ))
}

fn execute_acoustic_with_options(
    model: &AnalysisModel,
    backend: ComputeBackend,
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
                | BoundaryOperator::AcousticSphericalRadiation { .. }
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
        render_topology,
        modal_results: None,
        buckling_results: None,
        thermal_results: None,
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
    fea_run_admission::entrypoints::run_model(model, AnalysisOperation::Cfd, &context, || {
        execute_cfd_with_options(model, backend, options, context.clone())
    })
}

fn execute_cfd_with_options(
    model: &AnalysisModel,
    _backend: ComputeBackend,
    _options: AnalysisCfdRunOptions,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    Err(operation_error(
        AnalysisOperation::Cfd.name(),
        AnalysisOperation::Cfd.version(),
        &context,
        OperationErrorSpec {
            error_code: "RM.FEA.RUN_CFD.AUTHORED_STUDY_REQUIRED",
            error_type: OperationErrorType::Validation,
            retryable: false,
            severity: OperationErrorSeverity::Error,
        },
        "canonical finite-volume execution requires an authored steady fluid study with real geometry, meshing and governed task publication; the reduced channel fallback has been removed",
        BTreeMap::from([("analysis_model_id".to_string(), model.model_id.0.clone())]),
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

/// The reduced partitioned coupled-flow realization is retired; authored studies run coupled flow.
pub fn analysis_run_cht_op(
    model: &AnalysisModel,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    Err(operation_error(
        AnalysisOperation::Cht.name(),
        AnalysisOperation::Cht.version(),
        &context,
        OperationErrorSpec {
            error_code: "RM.FEA.RUN_CHT.AUTHORED_STUDY_REQUIRED",
            error_type: OperationErrorType::Validation,
            retryable: false,
            severity: OperationErrorSeverity::Error,
        },
        "coupled flow execution requires an authored study with real geometry, meshing and governed task publication; the reduced partitioned realization has been removed",
        BTreeMap::from([("analysis_model_id".to_string(), model.model_id.0.clone())]),
    ))
}

/// The reduced partitioned coupled-flow realization is retired; authored studies run coupled flow.
pub fn analysis_run_fsi_op(
    model: &AnalysisModel,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    Err(operation_error(
        AnalysisOperation::Fsi.name(),
        AnalysisOperation::Fsi.version(),
        &context,
        OperationErrorSpec {
            error_code: "RM.FEA.RUN_FSI.AUTHORED_STUDY_REQUIRED",
            error_type: OperationErrorType::Validation,
            retryable: false,
            severity: OperationErrorSeverity::Error,
        },
        "coupled flow execution requires an authored study with real geometry, meshing and governed task publication; the reduced partitioned realization has been removed",
        BTreeMap::from([("analysis_model_id".to_string(), model.model_id.0.clone())]),
    ))
}

fn validate_thermal_run_model(
    model: &AnalysisModel,
    context: &OperationContext,
) -> Result<(), OperationErrorEnvelope> {
    if let Some((material, thermal)) = model.materials.iter().find_map(|material| {
        let thermal = material.thermal.as_ref()?;
        (!thermal.conductivity_w_per_mk.is_finite()
            || thermal.conductivity_w_per_mk <= runmat_types::physical::Quantity::new(0.0)
            || !thermal.specific_heat_j_per_kgk.is_finite()
            || thermal.specific_heat_j_per_kgk <= runmat_types::physical::Quantity::new(0.0))
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
                    thermal.conductivity_w_per_mk.canonical_value().to_string(),
                ),
                (
                    "specific_heat_j_per_kgk".to_string(),
                    thermal
                        .specific_heat_j_per_kgk
                        .canonical_value()
                        .to_string(),
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
            .is_some_and(|temperature| {
                temperature.is_finite() && temperature.canonical_value() > 0.0
            }))
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
                    .is_some_and(|temperature| {
                        temperature.is_finite() && temperature.canonical_value() > 0.0
                    })
                    || !coefficient.is_finite()
                    || coefficient.canonical_value() < 0.0
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

fn execute_transient_with_options(
    model: &AnalysisModel,
    backend: ComputeBackend,
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
    let run_status = classify_run_status(result_quality, numerically_acceptable);

    let solver_backend = run.solver_backend;
    let execution_backend = run.execution_backend;
    let solver_device_apply_k_ratio = run.solver_device_apply_k_ratio;
    let solver_host_sync_count = run.solver_host_sync_count;
    let solver_method = run.solver_method;
    let selected_preconditioner = run.preconditioner;

    let result = AnalysisRunResult {
        observation_scalars: Vec::new(),
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
            precision_mode: contracts::format_precision_mode(PrecisionMode::Fp64),
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
        AnalysisNonlinearRunOptions::default(),
        context,
    )
}

fn execute_nonlinear_with_options(
    model: &AnalysisModel,
    backend: ComputeBackend,
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
    if options.linear_tolerance <= 0.0 || !options.linear_tolerance.is_finite() {
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
            "fea.run_nonlinear options require finite positive linear_tolerance",
            BTreeMap::from([(
                "linear_tolerance".to_string(),
                options.linear_tolerance.to_string(),
            )]),
        ));
    }
    if options.minimum_load_increment <= 0.0 || !options.minimum_load_increment.is_finite() {
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
            "fea.run_nonlinear options require finite positive minimum_load_increment",
            BTreeMap::from([(
                "minimum_load_increment".to_string(),
                options.minimum_load_increment.to_string(),
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
    if options.max_accepted_increments == 0 {
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
            "fea.run_nonlinear options require max_accepted_increments greater than zero",
            BTreeMap::from([(
                "max_accepted_increments".to_string(),
                options.max_accepted_increments.to_string(),
            )]),
        ));
    }

    let has_contact = model
        .interfaces
        .iter()
        .any(|interface| matches!(interface.kind, AnalysisInterfaceKind::Contact(_)));
    if options.contact_enforcement.is_some() && !has_contact {
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
    let contact_options =
        has_contact.then(|| contact_enforcement(options.contact_enforcement.as_ref()));
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
            increment_count: options.increment_count,
            max_newton_iters: options.max_newton_iters,
            tolerance: options.tolerance,
            linear_tolerance: options.linear_tolerance,
            max_load_cutbacks: options.max_load_cutbacks,
            minimum_load_increment: options.minimum_load_increment,
            max_accepted_increments: options.max_accepted_increments,
            line_search: options.line_search,
            max_line_search_backtracks: options.max_line_search_backtracks,
            line_search_reduction: options.line_search_reduction,
            solver_mesh: solver_mesh.clone(),
            contact_enforcement: contact_options,
        },
    )
    .map_err(|err| map_fea_run_error(AnalysisOperation::Nonlinear, model, &context, err))?;

    let (run, nonlinear_run) = nonlinear_result::data(nonlinear_run);
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
    let result_quality = if nonlinear_run.load_factors.is_empty()
        || nonlinear_run.displacement_snapshots.is_empty()
        || nonlinear_run.residual_norms.iter().any(|r| !r.is_finite())
        || nonlinear_run.increment_norms.iter().any(|v| !v.is_finite())
    {
        QualityGate::Fail
    } else if max_nonlinear_residual > options.tolerance {
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
                "nonlinear equilibrium residual exceeds tolerance {}",
                options.tolerance
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
        render_topology: render_topology_from_solver_mesh(solver_mesh.as_ref()),
        modal_results: None,
        buckling_results: None,
        thermal_results: None,
        transient_results: None,
        nonlinear_results: Some(nonlinear_run),
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
        AnalysisOperation::Nonlinear.name(),
        AnalysisOperation::Nonlinear.version(),
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
    fea_run_admission::entrypoints::run_model(
        model,
        AnalysisOperation::LinearStatic,
        &context,
        || analysis_run_linear_static_with_executor(model, backend, options, context.clone(), None),
    )
}

pub fn analysis_prepare_linear_static_run_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
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
    prepare_linear_static_step_run(model, &step_id, backend, options, context)
}

fn prepare_linear_static_step_run(
    model: &AnalysisModel,
    step_id: &ProcedureStepId,
    backend: ComputeBackend,
    options: AnalysisRunOptions,
    context: OperationContext,
) -> Result<PreparedStructuralLinearRun, OperationErrorEnvelope> {
    let _solver_context = install_fea_solver_context();
    let workload = prepare_linear_static_workload_template(model, step_id, backend, &options)?;
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
    options: &AnalysisRunOptions,
) -> Result<StructuralLinearWorkload, OperationErrorEnvelope> {
    let requested_preconditioner = match options.preconditioner_mode {
        PreconditionerMode::Auto => SpdPreconditionerKind::Auto,
        PreconditionerMode::Jacobi => SpdPreconditionerKind::Jacobi,
        PreconditionerMode::Ilu => SpdPreconditionerKind::Ilu0,
        PreconditionerMode::Amg => SpdPreconditionerKind::Amg,
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
    let envelope = analysis_admit_linear_static_run_with_id(
        prepared,
        run,
        assigned_run_id,
        Default::default(),
    )?;
    persist_linear_static_run(&context, &envelope.data)?;
    Ok(envelope)
}

fn analysis_admit_linear_static_run_with_id(
    prepared: PreparedStructuralLinearRun,
    run: FeaRunResult,
    assigned_run_id: Option<&str>,
    placement: StaticSolvePlacement,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    let result = linear_static_result::admit_linear_static_result(
        linear_static_result::LinearStaticResultInput {
            model: &prepared.workload.model,
            solver_mesh: prepared.solver_mesh.as_ref(),
            requested_backend: prepared.workload.backend,
            options: &prepared.options,
            assigned_run_id,
            placement,
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

/// Penalty enforcement controls applied to every authored contact interface. Interfaces and
/// their friction law are read by the solver from the analysis model itself.
fn contact_enforcement(
    enforcement: Option<&ContactEnforcementOptions>,
) -> runmat_analysis_fea::FeaContactEnforcement {
    let defaults = runmat_analysis_fea::FeaContactEnforcement::default();
    runmat_analysis_fea::FeaContactEnforcement {
        penalty_stiffness_scale: enforcement.map_or(defaults.penalty_stiffness_scale, |value| {
            value.penalty_stiffness_scale
        }),
        max_penetration_ratio: enforcement.map_or(defaults.max_penetration_ratio, |value| {
            value.max_penetration_ratio
        }),
    }
}

fn validate_contact_interface_options(
    options: &runmat_analysis_fea::FeaContactEnforcement,
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

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

mod document_run_result;
pub use document_run_result::{
    analysis_document_study_result, project_study_field_bindings, project_study_result_views,
};
