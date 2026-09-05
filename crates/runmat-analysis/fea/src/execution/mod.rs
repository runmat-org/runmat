//! Portable execution contracts owned by the analysis domain.
//!
//! This module identifies analysis work and numerical objects. Scheduling, placement, transport,
//! encryption, and object storage remain owned by RunMat's generic execution crates.

mod error;
mod identity;
mod object;
mod solve;

pub use error::{AnalysisExecutionError, AnalysisExecutionResult};
pub use identity::{
    AccuracyIdentity, AnalysisAttemptIdentity, AnalysisResultIdentity, AnalysisTaskIdentity,
    AnalysisWorkKind, AnalysisWorkloadIdentity, ConstraintPlanIdentity, DiscretizationIdentity,
    ExecutionPlanIdentity, GeneratorIdentity, NumericalPlanIdentity, ObservationIdentity,
    OperatorIdentity, ProblemIdentity, PublicationIdentity, SequenceIdentity,
    SignalRealizationIdentity, SolverExcitationIdentity, StateTransitionIdentity, StepIdentity,
    StudyIdentity,
};
pub use object::{
    chunk_numerical_payload, AnalysisChunkDescriptor, AnalysisNumericalObjectKind,
    AnalysisObjectManifest, ChunkedAnalysisPayload, ANALYSIS_CHUNK_MEDIA_TYPE,
    ANALYSIS_CHUNK_VALUE_SCHEMA, ANALYSIS_MANIFEST_MEDIA_TYPE, ANALYSIS_MANIFEST_VALUE_SCHEMA,
    DEFAULT_ANALYSIS_CHUNK_BYTES,
};
pub use solve::{StructuralLinearWorkload, StructuralLinearWorkloadResult};

#[cfg(test)]
mod tests;
