//! Public structural result carriers; original wire closures live in the parent owner.
use super::*;

#[derive(Clone, Debug, PartialEq)]
pub struct ExactStructuralStudyResult {
    pub thermal: Option<Box<runmat_analysis_execution::CompletedThermalStudy>>,
    pub frozen_project_digest: Option<Digest>,
    pub compiled_study: Option<runmat_analysis_fea::study::CompiledAnalysisStudy>,
    pub mesh: ExactMeshingDagRunResult,
    pub region_resolution: TopologyRegionResolution,
    pub resolved_workloads: StructuralStudyWorkloads,
    pub analysis: ExactStructuralStudyAnalysisResult,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExactStructuralStudyAnalysisResult {
    LinearSequence {
        resolved_workloads:
            BTreeMap<runmat_analysis_core::ProcedureStepId, StructuralLinearWorkload>,
        sequence: CompletedStructuralSequence,
    },
    StructuralTransient {
        resolved_workload: Box<StructuralTransientWorkload>,
        result: Box<StructuralTransientSolverResult>,
        numerical_receipt: StructuralTransientNumericalReceipt,
    },
    StructuralModal {
        resolved_workload: Box<StructuralModalWorkload>,
        result: Box<StructuralModalSolverResult>,
        numerical_receipt: StructuralModalNumericalReceipt,
    },
    StructuralModalTransient(Box<ExactStructuralModalTransientAnalysisResult>),
    StructuralModalResponseSpectrum(Box<ExactStructuralModalResponseSpectrumAnalysisResult>),
    StructuralRandomResponse(Box<ExactStructuralRandomResponseAnalysisResult>),
    StructuralBuckling(Box<ExactStructuralBucklingAnalysisResult>),
    StructuralPrestressedModal(Box<ExactStructuralPrestressedModalAnalysisResult>),
    StructuralPrestressedFrequency(Box<ExactStructuralPrestressedFrequencyAnalysisResult>),
    StructuralPrestressedModalFrequency(
        Box<ExactStructuralPrestressedModalFrequencyAnalysisResult>,
    ),
    StructuralFrequency(Box<ExactStructuralFrequencyAnalysisResult>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactStructuralModalTransientAnalysisResult {
    pub modal_workload: Box<StructuralModalWorkload>,
    pub transient_workload: Box<StructuralTransientWorkload>,
    pub modal_result: Box<StructuralModalSolverResult>,
    pub modal_numerical_receipt: StructuralModalNumericalReceipt,
    pub basis_state: runmat_analysis_fea::execution::CanonicalSemanticState,
    pub transient_result: Box<StructuralTransientSolverResult>,
    pub transient_numerical_receipt: StructuralTransientNumericalReceipt,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactStructuralModalResponseSpectrumAnalysisResult {
    pub modal_workload: Box<StructuralModalWorkload>,
    pub response_workload:
        Box<runmat_analysis_fea::execution::StructuralModalResponseSpectrumWorkload>,
    pub modal_result: Box<StructuralModalSolverResult>,
    pub modal_numerical_receipt: StructuralModalNumericalReceipt,
    pub basis_state: runmat_analysis_fea::execution::CanonicalSemanticState,
    pub response_result:
        Box<runmat_analysis_fea::execution::StructuralModalResponseSpectrumSolverResult>,
    pub response_numerical_receipt:
        runmat_analysis_fea::execution::StructuralModalResponseSpectrumNumericalReceipt,
    pub response_accuracy: runmat_analysis_fea::results::accuracy::AccuracyArtifact,
    pub response_publication_baseline_usage: runmat_analysis_execution::PreparedResourceUsage,
    pub response_resource_usage: runmat_analysis_execution::PreparedResourceUsage,
    pub response_publication: runmat_analysis_execution::ImportedPublishedResult,
    pub field_recovery: runmat_analysis_fea::solve::dynamics::ResponseSpectrumFieldRecovery,
    pub envelope_state: runmat_analysis_fea::execution::CanonicalSemanticState,
    pub response_observations:
        Option<runmat_analysis_fea::results::observation::ObservationPublicationCatalog>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactStructuralRandomResponseAnalysisResult {
    pub modal_workload: Box<StructuralModalWorkload>,
    pub response_workload: Box<runmat_analysis_fea::execution::StructuralRandomResponseWorkload>,
    pub modal_result: Box<StructuralModalSolverResult>,
    pub modal_numerical_receipt: StructuralModalNumericalReceipt,
    pub basis_state: runmat_analysis_fea::execution::CanonicalSemanticState,
    pub response_result: Box<runmat_analysis_fea::execution::StructuralRandomResponseSolverResult>,
    pub response_numerical_receipt:
        runmat_analysis_fea::execution::StructuralRandomResponseNumericalReceipt,
    pub response_accuracy: runmat_analysis_fea::results::accuracy::AccuracyArtifact,
    pub response_publication_baseline_usage: runmat_analysis_execution::PreparedResourceUsage,
    pub response_resource_usage: runmat_analysis_execution::PreparedResourceUsage,
    pub response_publication: runmat_analysis_execution::ImportedPublishedResult,
    pub statistics_state: runmat_analysis_fea::execution::CanonicalSemanticState,
    pub response_observations:
        Option<runmat_analysis_fea::results::observation::ObservationPublicationCatalog>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactStructuralBucklingAnalysisResult {
    pub preload_workload: Box<StructuralLinearWorkload>,
    pub buckling_workload: Box<StructuralBucklingWorkload>,
    pub preload_result: Box<runmat_analysis_fea::execution::StructuralLinearWorkloadResult>,
    pub preload_result_identity: runmat_analysis_fea::execution::SolverResultIdentity,
    pub preload_numerical_receipt:
        runmat_analysis_fea::execution::StructuralLinearNumericalReceiptArtifact,
    pub prestress_state: runmat_analysis_fea::execution::CanonicalSemanticState,
    pub buckling_result: Box<StructuralBucklingSolverResult>,
    pub buckling_numerical_receipt: StructuralBucklingNumericalReceipt,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactStructuralPrestressedModalAnalysisResult {
    pub preload_workload: Box<StructuralLinearWorkload>,
    pub modal_workload: Box<StructuralModalWorkload>,
    pub preload_result: Box<runmat_analysis_fea::execution::StructuralLinearWorkloadResult>,
    pub preload_result_identity: runmat_analysis_fea::execution::SolverResultIdentity,
    pub preload_numerical_receipt:
        runmat_analysis_fea::execution::StructuralLinearNumericalReceiptArtifact,
    pub prestress_state: runmat_analysis_fea::execution::CanonicalSemanticState,
    pub modal_result: Box<StructuralModalSolverResult>,
    pub modal_numerical_receipt: StructuralModalNumericalReceipt,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactStructuralPrestressedFrequencyAnalysisResult {
    pub preload_workload: Box<StructuralLinearWorkload>,
    pub preload_result: Box<runmat_analysis_fea::execution::StructuralLinearWorkloadResult>,
    pub preload_result_identity: runmat_analysis_fea::execution::SolverResultIdentity,
    pub preload_numerical_receipt:
        runmat_analysis_fea::execution::StructuralLinearNumericalReceiptArtifact,
    pub prestress_state: runmat_analysis_fea::execution::CanonicalSemanticState,
    pub frequency: ExactStructuralFrequencyAnalysisResult,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactStructuralPrestressedModalFrequencyAnalysisResult {
    pub preload_workload: Box<StructuralLinearWorkload>,
    pub modal_workload: Box<StructuralModalWorkload>,
    pub preload_result: Box<runmat_analysis_fea::execution::StructuralLinearWorkloadResult>,
    pub preload_result_identity: runmat_analysis_fea::execution::SolverResultIdentity,
    pub preload_numerical_receipt:
        runmat_analysis_fea::execution::StructuralLinearNumericalReceiptArtifact,
    pub prestress_state: runmat_analysis_fea::execution::CanonicalSemanticState,
    pub modal_result: Box<StructuralModalSolverResult>,
    pub modal_numerical_receipt: StructuralModalNumericalReceipt,
    pub basis_state: runmat_analysis_fea::execution::CanonicalSemanticState,
    pub frequency: ExactStructuralFrequencyAnalysisResult,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactStructuralFrequencyAnalysisResult {
    pub resolved_workload: Box<StructuralFrequencyWorkload>,
    pub result: Box<StructuralFrequencySolverResult>,
    pub numerical_receipt: StructuralFrequencyNumericalReceipt,
    pub entries: Vec<ExactStructuralFrequencyEntryResult>,
    pub histories: Vec<ExactStructuralFrequencyHistoryResult>,
    pub resource_usage: runmat_analysis_execution::PreparedResourceUsage,
    pub publication: Box<runmat_analysis_fea::execution::StructuralFrequencyPublicationManifest>,
    pub object_catalog: PublishedObjectCatalog,
    pub artifact_roots: runmat_runtime::analysis::StructuralFrequencyArtifactRootIndex,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactStructuralFrequencyEntryResult {
    pub sample: runmat_analysis_fea::results::field::RecoveredFieldSample,
    pub recovered_fields: runmat_analysis_execution::ImportedRecoveredFields,
    pub observations: Option<runmat_analysis_execution::ImportedObservations>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactStructuralFrequencyHistoryResult {
    pub observation_id: runmat_analysis_core::ObservationId,
    pub checkpoint: runmat_analysis_execution::ImportedObservationHistoryCheckpointTask,
    pub finalization: Option<
        crate::analysis_frequency_observations::CompletedStructuralFrequencyHistoryFinalization,
    >,
}
