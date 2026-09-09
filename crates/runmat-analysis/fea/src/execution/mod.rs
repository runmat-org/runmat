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
mod publication;
mod signal;
mod solve;
mod state;
mod sweep;

pub use accuracy::AccuracyEvaluationRequest;
pub use error::{AnalysisExecutionError, AnalysisExecutionResult};
pub use field::FieldRecoveryRequest;
pub use identity::{
    AccuracyEvaluationRequestIdentity, AccuracyIdentity, AnalysisAttemptIdentity,
    AnalysisResultIdentity, AnalysisTaskIdentity, AnalysisWorkKind, AnalysisWorkloadIdentity,
    ConstraintPlanIdentity, DiscretizationIdentity, ExecutionPlanIdentity,
    FieldRecoveryRequestIdentity, GeneratorIdentity, NumericalPlanIdentity,
    NumericalReceiptIdentity, ObservationEvaluationRequestIdentity, ObservationIdentity,
    OperatorIdentity, ProblemIdentity, PublicationIdentity, PublicationRequestIdentity,
    SequenceIdentity, SignalMaterializationIdentity, SignalRealizationIdentity,
    SolverExcitationIdentity, SolverResultIdentity, StateIdentity, StateTransitionIdentity,
    StepIdentity, StudyIdentity, SweepIdentity, SweepResultIdentity,
};
pub use object::{
    chunk_numerical_payload, chunk_partitioned_numerical_payload, AnalysisChunkDescriptor,
    AnalysisNumericalObjectKind, AnalysisObjectManifest, AnalysisPayloadPartition,
    AnalysisPayloadPartitionDescriptor, ChunkedAnalysisPayload, ANALYSIS_CHUNK_MEDIA_TYPE,
    ANALYSIS_CHUNK_VALUE_SCHEMA, ANALYSIS_MANIFEST_MEDIA_TYPE, ANALYSIS_MANIFEST_VALUE_SCHEMA,
    DEFAULT_ANALYSIS_CHUNK_BYTES,
};
pub use observation::ObservationEvaluationRequest;
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
pub use sweep::{ComposedSweepResult, SweepEntryDisposition, SweepEntryOutcome};

#[cfg(test)]
mod tests;
