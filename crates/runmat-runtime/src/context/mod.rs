//! Explicit, session-owned runtime capabilities and invocation state.
//!
//! [`RuntimeContext`] is the one context propagated by Core, the VM, nested
//! calls, and future native/foreign/parallel adapters. Legacy ambient APIs are
//! migrated through [`legacy`] and must not become new semantic dependencies.

mod capability;
mod preferences;
mod user_path;
mod host_document;
pub mod host_effect;
pub(crate) mod host_presentation;
pub(crate) mod provider_operation;
#[cfg(test)]
pub(crate) mod document_test_support;
#[cfg(test)]
pub(crate) mod physical_test_support;
mod runtime;
mod scope;
mod services;
mod state;
#[cfg(test)]
mod cancellation_tests;

pub mod legacy;
pub mod preparation;

pub use capability::{RuntimeCapability, RuntimeCapabilityError};
pub use host_document::{HostDocumentRequest, HostTextLocation, HostTextSelection};
#[cfg(target_arch = "wasm32")]
pub use preferences::BrowserPreferenceService;
#[cfg(not(target_arch = "wasm32"))]
pub use preferences::NativePreferenceService;
pub use preferences::{
    IsolatedMemoryPreferenceService, PreferenceGroups, PreferenceServiceError, PreferenceValues,
    RuntimePreferenceService,
};
pub use runtime::{
    ProgramExecutionAssignmentGuard, ProgramExecutionJobGuard, RuntimeContext,
    RuntimeContextLocalId, RuntimeExecutionStack, RuntimeExecutionStackGuard, RuntimeLanguageMode,
    DEFAULT_CALLSTACK_LIMIT, DEFAULT_ERROR_NAMESPACE,
};
pub use scope::{ContextFuture, RuntimeContextGuard};
#[cfg(not(target_arch = "wasm32"))]
pub use services::RuntimeProcessService;
pub use services::{
    ForeignCall, HostInteraction, NativeCapability, ParallelCapability, RuntimeAccelerationService,
    RuntimeBuiltinService, RuntimeCallRequest, RuntimeCallRouter, RuntimeCallService,
    RuntimeCollectiveService, RuntimeDistributedCallRequest, RuntimeDistributedInvocation,
    RuntimeDistributedRetirement, RuntimeDistributedService, RuntimeDistributedSnapshot,
    RuntimeErrorService, RuntimeForeignService, RuntimeHostEffectResourceService,
    RuntimeHostService, RuntimeHttpService, RuntimeNativeService, RuntimeObjectService,
    RuntimeParallelResources, RuntimeParallelService, RuntimePlacementFeedbackSource,
    RuntimePlacementService, RuntimeServiceFuture, RuntimeServicePorts, RuntimeSpmdAdmission,
    RuntimeSpmdOutput, RuntimeSpmdRetainedOutput, RuntimeSpmdService,
    RuntimeWholeWorkObservationPermit, RuntimeWholeWorkProbeLease, RuntimeWholeWorkSelection,
    RuntimeWorkspaceService,
};
pub(crate) use state::RuntimeContextState;
pub use user_path::{
    RuntimeUserPathService, UserPathMutation, UserPathServiceError, UserPathSetting,
    UserPathSnapshot,
};
