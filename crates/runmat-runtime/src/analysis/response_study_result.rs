//! Response results retain both receipt-source and terminal resource accounting.
use runmat_analysis_core::ProcedureStepId;

#[derive(Debug, Clone, serde::Serialize)]
pub struct StructuralStudyModalResponseSpectrumNumericalResult {
    pub frozen_project_digest: Option<runmat_execution::Digest>,
    pub modal_step_id: ProcedureStepId,
    pub response_step_id: ProcedureStepId,
    pub modal_result: runmat_analysis_fea::execution::StructuralModalSolverResult,
    pub modal_numerical_receipt: runmat_analysis_fea::execution::StructuralModalNumericalReceipt,
    pub basis_state: runmat_analysis_fea::execution::CanonicalSemanticState,
    pub resolved_response_workload:
        runmat_analysis_fea::execution::StructuralModalResponseSpectrumWorkload,
    pub response_result:
        runmat_analysis_fea::execution::StructuralModalResponseSpectrumSolverResult,
    pub response_numerical_receipt:
        runmat_analysis_fea::execution::StructuralModalResponseSpectrumNumericalReceipt,
    pub response_accuracy: runmat_analysis_fea::results::accuracy::AccuracyArtifact,
    /// Immutable pre-publication evidence named by the receipt.
    pub response_publication_baseline_usage:
        runmat_analysis_fea::planning::resource::ResourceUsageEvidence,
    /// Complete usage including the final publication attempt.
    pub response_resource_usage: runmat_analysis_fea::planning::resource::ResourceUsageEvidence,
    pub response_publication: runmat_analysis_fea::results::publication::PublishedAnalysisResult,
    pub field_recovery: runmat_analysis_fea::solve::dynamics::ResponseSpectrumFieldRecovery,
    pub envelope_state: runmat_analysis_fea::execution::CanonicalSemanticState,
    pub response_observations:
        Option<runmat_analysis_fea::results::observation::ObservationPublicationCatalog>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct StructuralStudyRandomResponseNumericalResult {
    pub frozen_project_digest: Option<runmat_execution::Digest>,
    pub modal_step_id: ProcedureStepId,
    pub response_step_id: ProcedureStepId,
    pub modal_result: runmat_analysis_fea::execution::StructuralModalSolverResult,
    pub modal_numerical_receipt: runmat_analysis_fea::execution::StructuralModalNumericalReceipt,
    pub basis_state: runmat_analysis_fea::execution::CanonicalSemanticState,
    pub resolved_response_workload:
        runmat_analysis_fea::execution::StructuralRandomResponseWorkload,
    pub response_result: runmat_analysis_fea::execution::StructuralRandomResponseSolverResult,
    pub response_numerical_receipt:
        runmat_analysis_fea::execution::StructuralRandomResponseNumericalReceipt,
    pub response_accuracy: runmat_analysis_fea::results::accuracy::AccuracyArtifact,
    /// Immutable pre-publication evidence named by the receipt.
    pub response_publication_baseline_usage:
        runmat_analysis_fea::planning::resource::ResourceUsageEvidence,
    /// Complete usage including the final publication attempt.
    pub response_resource_usage: runmat_analysis_fea::planning::resource::ResourceUsageEvidence,
    pub response_publication: runmat_analysis_fea::results::publication::PublishedAnalysisResult,
    pub statistics_state: runmat_analysis_fea::execution::CanonicalSemanticState,
    pub response_observations:
        Option<runmat_analysis_fea::results::observation::ObservationPublicationCatalog>,
}
