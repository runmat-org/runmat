//! Portable execution contracts shared by native, browser, test, and remote hosts.

pub mod cancellation;
pub use cancellation::{CancellationSource, CancellationToken};
pub mod collective;
pub mod console;
pub mod distributed;
mod environment;
pub use environment::{EnvironmentSnapshot, EnvironmentStorageProfile};
mod error;
pub mod executable;
pub mod gang;
pub mod handle;
pub mod host;
pub mod identity;
pub mod placement;
pub mod profiling;
pub mod process_output;
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
    ExecutableComponentRevisions, ExecutableDependencyDescriptor, ExecutableDependencyProduct,
    ExecutableEntrypointKind, ExecutableIdentity, ExecutableOptionalSection,
    ExecutableSectionSupport, ExecutableSourceMapV2, ExecutableSourceRecordV2,
    ExecutableUnitAdmission, ExecutableUnitEnvelope, ExecutableUnitManifest, SectionRequirement,
    ValidatedExecutableEntrypoint, EXECUTABLE_DEPENDENCY_MAX_PRODUCTS,
    EXECUTABLE_UNIT_ENVELOPE_MAX_BYTES, EXECUTABLE_UNIT_SCHEMA_VERSION,
};
pub use gang::{GangHandle, GangRequest, GangSnapshot, SpmdOutputValue, SpmdTaskContext};
pub use handle::{
    ExecutionHandleSnapshot, ExecutionHandleState, FutureHandle, JobHandle, OutputContract,
    PoolBackend, PoolHandle, PoolRequest, PoolSnapshot, TaskHandle, TaskResultClaim,
};
pub use identity::{
    CompositeId, DistributedObjectId, ExecutionScopeId, FunctionBindingOrigin, FutureId, GangId,
    JobId, PoolId, RunId, TaskId,
};
pub use identity::{
    Digest, DigestStream, DomainContribution, ForeignPolicyRef, LanguageCompatibilityMode,
    ProgramEnvironment, ProgramRevision,
};
pub use placement::{
    admit_candidate_resources, single_candidate_resources_admitted, AdmittedResourceDemand,
    CandidateCostEvidence, CandidateCostUnavailable, CandidateExecutionLocation,
    CandidateObservationEvidence, CandidateOutputPresentation, CandidateOutputResidency,
    CandidatePeakLiveDemand, CandidatePreparationState, CandidateResourceDemand,
    CandidateResourceFacts, CompactSelectedExecutionCandidate, EstimateConfidence, EstimateSource,
    ExecutionCandidateDescriptor, ExecutionCandidateKind, ExecutionCostComponents,
    ExecutionCostEstimate, ObservedCandidateDuration, PlacementDecision,
    PlacementExecutionAuthorityRevision, PlacementFeedback, PlacementGraph,
    PlacementGraphCandidate, PlacementGraphEdge, PlacementGraphLimits, PlacementGraphNode,
    PlacementGraphOutputFact, PlacementInvalidation, PlacementOperationId, PlacementPermitIdentity,
    PlacementPlanRequest, PlacementResourceSnapshot, PlacementRevision, PlacementRevisionRef,
    PlacementSignature, PlacementSignatureRef, ProviderInputStageDemand, ProviderInputStageDemands,
    ProviderReadbackDemand, ProviderReadbackDemands, ProviderResourceSnapshot, ResourceAccount,
    ResourceAdmissionContext, ResourceAdmissionError, ResourceBoundUnavailable, ResourceBudgetKind,
    ResourceByteBound, ResourceSnapshotError, SelectedExecutionCandidate, SourceOutputOrdinal,
    WholeWorkCandidate, WholeWorkConstraint, WholeWorkDecision, WholeWorkDecisionKind,
    WholeWorkObservation, WholeWorkObservationOutcome, WholeWorkPlacementRequest,
    WholeWorkPlacementRequestRef,
};
pub use runmat_types::ProgramFunctionId;
pub use state::{CancellationReason, PoolState};
pub use task::{
    OwnedBuiltinBinding, OwnedBuiltinDenseOutputContract, OwnedBuiltinResponseContract,
    OwnedBuiltinWorkRequest, OwnedBuiltinWorkResponse, ParallelChunk, ParallelRandomStream,
    ParallelRandomnessContext, ParallelTaskContext, ParallelTaskGraph, ProgramCallFrame,
    ProgramCallable, ProgramDetachedDiagnosticSource, ProgramExecutionAssignment,
    ProgramInvocationContext, ProgramRuntimeFailure, ProgramSourceSpan, RetryPolicy,
    TaskExecutableIdentity, OWNED_BUILTIN_FAILURE_WIRE_BYTES,
    OWNED_BUILTIN_MAX_RESPONSE_WIRE_BYTES, OWNED_BUILTIN_VALUE_LIMITS,
    OWNED_BUILTIN_WORK_SCHEMA_VERSION,
};
