//! Portable execution contracts owned by the analysis domain.
//!
//! This module identifies analysis work and numerical objects. Scheduling, placement, transport,
//! encryption, and object storage remain owned by RunMat's generic execution crates.

mod accuracy;
mod accuracy_source;
mod buckling_result;
mod buckling_workload;
mod error;
mod field;
mod field_plan;
mod frequency_assembly;
mod frequency_balance;
mod frequency_entry;
mod frequency_publication;
mod frequency_receipt;
mod frequency_result;
mod frequency_sample;
mod frequency_scope;
mod frequency_workload;
mod identity;
mod modal_result;
mod modal_workload;
mod object;
mod observation;
mod observation_assembly;
mod observation_history_checkpoint;
mod observation_partition;
mod publication;
mod random_response_interval;
mod random_response_result;
mod random_response_scope;
mod random_response_workload;
mod response_spectrum_entry;
mod response_spectrum_result;
mod response_spectrum_scope;
mod response_spectrum_workload;
mod signal;
mod solve;
mod state;
pub(crate) mod structural_identity;
mod study_workloads;
mod sweep;
mod transient_block;
mod transient_checkpoint;
mod transient_coordinates;
mod transient_result;
mod transient_schedule;
mod transient_segment;
mod transient_template;
mod transient_workload;

pub use accuracy::{AccuracyEvaluationRequest, AccuracySourceKind};
pub use accuracy_source::AccuracySourceReceipt;
pub use buckling_result::{
    StructuralBucklingMode, StructuralBucklingNumericalReceipt, StructuralBucklingSolverResult,
    StructuralBucklingSolverResultInput,
};
pub use buckling_workload::{
    StructuralBucklingSolverPolicy, StructuralBucklingTemplate, StructuralBucklingWorkload,
    StructuralBucklingWorkloadInput,
};
pub use error::{AnalysisExecutionError, AnalysisExecutionResult};
pub use field::{FieldRecoveryRequest, FieldRecoverySource};
pub use field_plan::{FieldRecoveryCost, FieldRecoveryPlan, FieldRecoveryRequirement};
pub use frequency_assembly::{
    StructuralFrequencyAssemblyEntry, StructuralFrequencyAssemblyOutput,
    StructuralFrequencyResultAssemblyRequest,
};
pub use frequency_balance::{ComplexStructuralBalance, StructuralComplexVector};
pub use frequency_entry::{
    StructuralFrequencyEntryRequest, StructuralFrequencyEntryResult,
    StructuralFrequencyEntryResultManifest,
};
pub use frequency_publication::{
    StructuralFrequencyPublicationHistory, StructuralFrequencyPublicationInput,
    StructuralFrequencyPublicationManifest, StructuralFrequencyPublicationSample,
};
pub use frequency_receipt::{
    StructuralFrequencyEntrySolverEvidence, StructuralFrequencyNumericalReceipt,
    StructuralFrequencySolutionSpace, StructuralFrequencySolverEvidenceInput,
};
pub use frequency_result::{
    StructuralFrequencyResultEntry, StructuralFrequencySolverResult,
    StructuralFrequencySolverResultManifest,
};
pub use frequency_sample::StructuralFrequencyPhysicalSample;
pub use frequency_scope::{
    StructuralFrequencyScope, StructuralFrequencyScopeInput, StructuralPhasorConvention,
};
pub use frequency_workload::{
    StructuralFrequencySolverPlan, StructuralFrequencyStateInputs, StructuralFrequencyWorkload,
    StructuralFrequencyWorkloadInput, StructuralModalFrequencyTemplate,
    StructuralPrestressedFrequencySolverTemplate, StructuralPrestressedFrequencyTemplate,
};
pub use identity::{
    AccuracyEvaluationRequestIdentity, AccuracyIdentity, AnalysisAttemptIdentity,
    AnalysisResultIdentity, AnalysisTaskIdentity, AnalysisWorkKind, AnalysisWorkloadIdentity,
    ConstraintPlanIdentity, DiscretizationIdentity, ExecutionPlanIdentity,
    FieldRecoveryRequestIdentity, GeneratorIdentity, NumericalMaterializationIdentity,
    NumericalPlanIdentity, NumericalReceiptIdentity, ObservationAssemblyRequestIdentity,
    ObservationEvaluationRequestIdentity, ObservationHistoryCheckpointRequestIdentity,
    ObservationIdentity, ObservationPartitionEvaluationRequestIdentity,
    ObservationPartitionReductionRequestIdentity, OperatorIdentity, ProblemIdentity,
    PublicationIdentity, PublicationRequestIdentity, SequenceIdentity,
    SignalMaterializationIdentity, SignalRealizationIdentity, SolverExcitationIdentity,
    SolverResultIdentity, StateIdentity, StateTransitionIdentity, StepIdentity,
    StructuralBucklingWorkloadIdentity, StructuralDampingOperatorIdentity,
    StructuralDofLayoutIdentity, StructuralExcitationSetIdentity, StructuralFormulationIdentity,
    StructuralFrequencyEntryRequestIdentity, StructuralFrequencyEntryResultIdentity,
    StructuralFrequencyPhysicalSampleIdentity, StructuralFrequencyProblemIdentity,
    StructuralFrequencyPublicationIdentity, StructuralFrequencyResultAssemblyRequestIdentity,
    StructuralFrequencyScopeIdentity, StructuralFrequencyWorkloadIdentity,
    StructuralKinematicsIdentity, StructuralMassOperatorIdentity,
    StructuralModalResponseSpectrumAssemblyRequestIdentity,
    StructuralModalResponseSpectrumDirectionRequestIdentity,
    StructuralModalResponseSpectrumDirectionResultIdentity,
    StructuralModalResponseSpectrumScopeIdentity, StructuralModalResponseSpectrumWorkloadIdentity,
    StructuralModalWorkloadIdentity, StructuralRandomResponseAssemblyRequestIdentity,
    StructuralRandomResponseIntervalRequestIdentity,
    StructuralRandomResponseIntervalResultIdentity, StructuralRandomResponseScopeIdentity,
    StructuralRandomResponseWorkloadIdentity, StructuralStiffnessOperatorIdentity,
    StructuralTransientBlockRequestIdentity, StructuralTransientCheckpointIdentity,
    StructuralTransientCheckpointScopeIdentity, StructuralTransientPhysicalSampleIdentity,
    StructuralTransientResultAssemblyRequestIdentity, StructuralTransientScheduleIdentity,
    StructuralTransientSegmentIdentity, StructuralTransientSegmentSampleIdentity,
    StructuralTransientSolverOptionsIdentity, StudyIdentity, SweepIdentity, SweepResultIdentity,
};
pub use modal_result::{StructuralModalNumericalReceipt, StructuralModalSolverResult};
pub use modal_workload::{
    StructuralModalWorkload, StructuralModalWorkloadInput, StructuralPrestressedModalTemplate,
};
pub use object::{
    chunk_numerical_payload, chunk_partitioned_numerical_payload, AnalysisChunkDescriptor,
    AnalysisNumericalObjectKind, AnalysisObjectManifest, AnalysisPayloadPartition,
    AnalysisPayloadPartitionDescriptor, ChunkedAnalysisPayload, ANALYSIS_CHUNK_MEDIA_TYPE,
    ANALYSIS_CHUNK_VALUE_SCHEMA, ANALYSIS_MANIFEST_MAX_ENCODED_BYTES, ANALYSIS_MANIFEST_MEDIA_TYPE,
    ANALYSIS_MANIFEST_VALUE_SCHEMA, DEFAULT_ANALYSIS_CHUNK_BYTES,
};
pub use observation::ObservationEvaluationRequest;
pub use observation_assembly::ObservationAssemblyRequest;
pub use observation_history_checkpoint::ObservationHistoryCheckpointRequest;
pub use observation_partition::{
    ObservationPartitionEvaluationRequest, ObservationPartitionReductionRequest,
};
pub use publication::ResultPublicationRequest;
pub use random_response_interval::{
    StructuralRandomResponseIntervalRequest, StructuralRandomResponseIntervalResult,
};
pub use random_response_result::{
    StructuralRandomResponseAssemblyRequest, StructuralRandomResponseNumericalReceipt,
    StructuralRandomResponseSolverResult, StructuralRandomStatistics,
};
pub use random_response_scope::{
    StructuralRandomResponseScope, StructuralRandomResponseScopeInput,
};
pub use random_response_workload::{
    StructuralRandomResponseTemplate, StructuralRandomResponseWorkload,
    StructuralRandomResponseWorkloadInput,
};
pub use response_spectrum_entry::{
    StructuralModalResponseSpectrumDirectionRequest, StructuralModalResponseSpectrumDirectionResult,
};
pub use response_spectrum_result::{
    StructuralModalResponseSpectrumAssemblyRequest,
    StructuralModalResponseSpectrumNumericalReceipt, StructuralModalResponseSpectrumSolverResult,
    StructuralResponseEnvelope,
};
pub use response_spectrum_scope::{
    StructuralModalResponseSpectrumScope, StructuralModalResponseSpectrumScopeInput,
};
pub use response_spectrum_workload::{
    StructuralModalResponseSpectrumTemplate, StructuralModalResponseSpectrumWorkload,
    StructuralModalResponseSpectrumWorkloadInput,
};
pub use signal::{
    CanonicalSignalContent, CanonicalSignalKnot, CanonicalSignalRealization,
    CanonicalSignalRealizationInput, CanonicalSignalValues, SignalInputAdmission,
};
pub use solve::{
    StructuralLinearNumericalReceipt, StructuralLinearNumericalReceiptArtifact,
    StructuralLinearWorkload, StructuralLinearWorkloadResult,
};
pub use state::{
    BoundThermalStateTransfer, CanonicalSemanticState, CanonicalSemanticStateInput,
    CanonicalSemanticStatePayload, CanonicalStateTransition, CanonicalStateTransitionInput,
    SemanticStatePublicationRequest, SemanticStatePublicationSource,
};
pub use study_workloads::{
    StructuralStepWorkload, StructuralStudyWorkloads, StructuralTransientBlockPlan,
};
pub use sweep::{
    ComposedSweepResult, SweepCompositionPolicy, SweepDefinition, SweepEntryDisposition,
    SweepEntryOutcome,
};
pub use transient_block::StructuralTransientBlockRequest;
pub use transient_checkpoint::{
    StructuralTransientBalanceAccumulator, StructuralTransientCheckpoint,
    StructuralTransientCheckpointInput, StructuralTransientCheckpointScope,
    StructuralTransientContinuationPoint, StructuralTransientContinuationState,
    StructuralTransientCoordinateSpace, StructuralTransientNumericalState,
    StructuralTransientObservationCursor, StructuralTransientSolverCounters,
};
pub use transient_coordinates::{ModalProjectionEvidence, ModalTransientCoordinateState};
pub use transient_result::{
    StructuralTransientBlockReceipt, StructuralTransientNumericalReceipt,
    StructuralTransientResultAssemblyRequest, StructuralTransientResultEntry,
    StructuralTransientSolverResult, StructuralTransientSolverResultManifest,
};
pub use transient_schedule::{
    StructuralTransientContinuationApproach, StructuralTransientIntegrator,
    StructuralTransientPrecision, StructuralTransientSchedule, StructuralTransientScheduleArrival,
    StructuralTransientSchedulePoint,
};
pub use transient_segment::{
    StructuralTransientBlockSolverEvidence, StructuralTransientPhysicalSample,
    StructuralTransientPhysicalSampleInput, StructuralTransientSampleStep,
    StructuralTransientSegment, StructuralTransientSegmentEntry,
    StructuralTransientSegmentManifest, StructuralTransientSegmentSample,
    StructuralTransientSegmentSampleInput,
};
pub use transient_template::StructuralModalTransientTemplate;
pub(crate) use transient_workload::StructuralTransientCheckpointScopeInput;
pub use transient_workload::{StructuralTransientSolverPlan, StructuralTransientWorkload};

#[cfg(test)]
mod tests;
