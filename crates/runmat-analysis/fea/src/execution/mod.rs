//! Portable execution contracts owned by the analysis domain.
//!
//! This module identifies analysis work and numerical objects. Scheduling, placement, transport,
//! encryption, and object storage remain owned by RunMat's generic execution crates.

mod accuracy;
mod error;
mod field;
mod identity;
mod object;
mod observation;
mod observation_assembly;
mod observation_history_checkpoint;
mod observation_partition;
mod publication;
mod signal;
mod solve;
mod state;
mod sweep;
mod transient_block;
mod transient_checkpoint;
mod transient_result;
mod transient_schedule;
mod transient_segment;
mod transient_workload;

pub use accuracy::AccuracyEvaluationRequest;
pub use error::{AnalysisExecutionError, AnalysisExecutionResult};
pub use field::FieldRecoveryRequest;
pub use identity::{
    AccuracyEvaluationRequestIdentity, AccuracyIdentity, AnalysisAttemptIdentity,
    AnalysisResultIdentity, AnalysisTaskIdentity, AnalysisWorkKind, AnalysisWorkloadIdentity,
    ConstraintPlanIdentity, DiscretizationIdentity, ExecutionPlanIdentity,
    FieldRecoveryRequestIdentity, GeneratorIdentity, NumericalPlanIdentity,
    NumericalReceiptIdentity, ObservationAssemblyRequestIdentity,
    ObservationEvaluationRequestIdentity, ObservationHistoryCheckpointRequestIdentity,
    ObservationIdentity, ObservationPartitionEvaluationRequestIdentity,
    ObservationPartitionReductionRequestIdentity, OperatorIdentity, ProblemIdentity,
    PublicationIdentity, PublicationRequestIdentity, SequenceIdentity,
    SignalMaterializationIdentity, SignalRealizationIdentity, SolverExcitationIdentity,
    SolverResultIdentity, StateIdentity, StateTransitionIdentity, StepIdentity,
    StructuralDampingOperatorIdentity, StructuralDofLayoutIdentity,
    StructuralExcitationSetIdentity, StructuralFormulationIdentity, StructuralKinematicsIdentity,
    StructuralMassOperatorIdentity, StructuralStiffnessOperatorIdentity,
    StructuralTransientBlockRequestIdentity, StructuralTransientCheckpointIdentity,
    StructuralTransientCheckpointScopeIdentity, StructuralTransientPhysicalSampleIdentity,
    StructuralTransientResultAssemblyRequestIdentity, StructuralTransientScheduleIdentity,
    StructuralTransientSegmentIdentity, StructuralTransientSegmentSampleIdentity,
    StructuralTransientSolverOptionsIdentity, StudyIdentity, SweepIdentity, SweepResultIdentity,
};
pub use object::{
    chunk_numerical_payload, chunk_partitioned_numerical_payload, AnalysisChunkDescriptor,
    AnalysisNumericalObjectKind, AnalysisObjectManifest, AnalysisPayloadPartition,
    AnalysisPayloadPartitionDescriptor, ChunkedAnalysisPayload, ANALYSIS_CHUNK_MEDIA_TYPE,
    ANALYSIS_CHUNK_VALUE_SCHEMA, ANALYSIS_MANIFEST_MEDIA_TYPE, ANALYSIS_MANIFEST_VALUE_SCHEMA,
    DEFAULT_ANALYSIS_CHUNK_BYTES,
};
pub use observation::ObservationEvaluationRequest;
pub use observation_assembly::ObservationAssemblyRequest;
pub use observation_history_checkpoint::ObservationHistoryCheckpointRequest;
pub use observation_partition::{
    ObservationPartitionEvaluationRequest, ObservationPartitionExecutionTarget,
    ObservationPartitionReductionRequest,
};
pub use publication::ResultPublicationRequest;
pub use signal::{
    CanonicalSignalContent, CanonicalSignalKnot, CanonicalSignalRealization,
    CanonicalSignalRealizationInput, CanonicalSignalValues, SignalInputAdmission,
};
pub use solve::{
    StructuralLinearNumericalReceipt, StructuralLinearNumericalReceiptArtifact,
    StructuralLinearWorkload, StructuralLinearWorkloadResult,
};
pub use state::{
    CanonicalSemanticState, CanonicalSemanticStateInput, CanonicalStateTransition,
    CanonicalStateTransitionInput, SemanticStatePublicationRequest,
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
    StructuralTransientObservationCursor, StructuralTransientSolverCounters,
};
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
pub use transient_workload::{
    StructuralTransientCheckpointScopeInput, StructuralTransientWorkload,
};

#[cfg(test)]
mod tests;
