//! Portable execution contracts shared by native, browser, test, and remote hosts.

pub mod collective;
pub mod distributed;
mod error;
pub mod executable;
pub mod gang;
pub mod handle;
pub mod host;
pub mod identity;
pub mod placement;
pub mod protocol;
pub mod resource;
pub mod schema;
pub mod security;
pub mod state;
pub mod task;
pub mod value;

pub use collective::{
    CollectiveInvocation, CollectiveMessageTag, CollectiveRequest, CollectiveResponse,
    CollectiveSequence, DistributedBuildContribution, ReceiveSelection,
};
pub use distributed::{
    validate_partition_layouts, CompositeHandle, CompositeSnapshot, DistributedOwnedPartition,
    DistributedPartition, DistributedPartitionLayout, DistributedShardSnapshot,
    DistributedValueHandle, DistributedValueSnapshot, PartitionRange, PartitionSelection,
};
pub use error::ContractError;
pub use executable::{
    ExecutableComponentDescriptor, ExecutableComponentKind, ExecutableComponentPayload,
    ExecutableComponentRevisions, ExecutableEntrypointKind, ExecutableIdentity,
    ExecutableOptionalSection, ExecutableSectionSupport, ExecutableUnitAdmission,
    ExecutableUnitEnvelope, ExecutableUnitManifest, SectionRequirement,
    EXECUTABLE_UNIT_ENVELOPE_MAX_BYTES, EXECUTABLE_UNIT_SCHEMA_VERSION,
};
pub use gang::{GangHandle, GangRequest, GangSnapshot, SpmdOutputValue, SpmdTaskContext};
pub use handle::{
    ExecutionHandleSnapshot, ExecutionHandleState, FutureHandle, JobHandle, OutputContract,
    PoolBackend, PoolHandle, PoolRequest, PoolSnapshot, TaskHandle, TaskResultClaim,
};
pub use identity::{
    CompositeId, DistributedObjectId, ExecutionScopeId, FutureId, GangId, JobId, PoolId, RunId,
    TaskId,
};
pub use identity::{
    Digest, DomainContribution, LanguageCompatibilityMode, ProgramEnvironment, ProgramRevision,
};
pub use placement::{
    CandidateExecutionLocation, CandidateOutputResidency, CandidatePreparationState,
    CandidateResourceDemand, EstimateConfidence, EstimateSource, ExecutionCandidateDescriptor,
    ExecutionCandidateKind, ExecutionCostComponents, ExecutionCostEstimate, PlacementDecision,
    PlacementFeedback, PlacementGraph, PlacementGraphCandidate, PlacementGraphEdge,
    PlacementGraphLimits, PlacementGraphNode, PlacementInvalidation, PlacementPlanRequest,
    PlacementResourceSnapshot, PlacementRevision, PlacementSignature, ProviderResourceSnapshot,
    SelectedExecutionCandidate,
};
pub use runmat_types::ProgramFunctionId;
pub use state::{CancellationReason, PoolState};
pub use task::{
    ParallelChunk, ParallelRandomStream, ParallelRandomnessContext, ParallelTaskContext,
    ParallelTaskGraph, ProgramCallFrame, ProgramCallable, ProgramExecutionAssignment,
    ProgramInvocationContext, ProgramRuntimeFailure, ProgramSourceSpan, RetryPolicy,
};
