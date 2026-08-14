//! Portable execution contracts shared by native, browser, test, and remote hosts.

mod error;
pub mod executable;
pub mod handle;
pub mod identity;
pub mod placement;
pub mod protocol;
pub mod resource;
pub mod schema;
pub mod security;
pub mod state;
pub mod task;
pub mod value;

pub use error::ContractError;
pub use executable::{
    ExecutableComponentDescriptor, ExecutableComponentKind, ExecutableComponentPayload,
    ExecutableComponentRevisions, ExecutableEntrypointKind, ExecutableIdentity,
    ExecutableOptionalSection, ExecutableSectionSupport, ExecutableUnitEnvelope,
    ExecutableUnitManifest, SectionRequirement, EXECUTABLE_UNIT_ENVELOPE_MAX_BYTES,
    EXECUTABLE_UNIT_SCHEMA_VERSION,
};
pub use handle::{FutureHandle, JobHandle, OutputContract, PoolHandle, TaskHandle};
pub use identity::{Digest, DomainContribution, ProgramEnvironment, ProgramRevision};
pub use identity::{ExecutionScopeId, FutureId, JobId, PoolId, RunId, TaskId};
pub use placement::{
    CandidateOutputResidency, CandidatePreparationState, EstimateConfidence, EstimateSource,
    ExecutionCandidateDescriptor, ExecutionCandidateKind, ExecutionCostComponents,
    ExecutionCostEstimate,
};
pub use state::CancellationReason;
