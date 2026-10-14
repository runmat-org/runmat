//! Explicit, session-owned runtime capabilities and invocation state.
//!
//! [`RuntimeContext`] is the one context propagated by Core, the VM, nested
//! calls, and future native/foreign/parallel adapters. Legacy ambient APIs are
//! migrated through [`legacy`] and must not become new semantic dependencies.

mod capability;
mod cleanup_lifecycle;
mod callable_artifacts;
mod graphics_presentation;
mod graphics_callbacks;
#[cfg(not(target_arch = "wasm32"))]
mod foreign_callback_lineage;
mod arity;
mod catch;
pub(crate) mod call_projection;
mod local_tasks;
mod local_task_admission;
mod native_continuation;
mod preferences;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod process_invocation;
mod user_path;
mod host_document;
mod host_dialog;
mod managed_observations;
pub(crate) mod output_projection;
pub mod host_effect;
pub(crate) mod host_presentation;
pub(crate) mod provider_operation;
#[cfg(test)]
pub(crate) mod document_test_support;
#[cfg(test)]
pub(crate) mod physical_test_support;
mod runtime;
mod super_constructor;
mod scope;
mod source_evaluation;
mod services;
mod state;
mod timing;
mod timer;
mod tcp;
mod vm_fusion_completion;
mod vm_fusion_host_route;
mod vm_fusion_host_resources;
pub use vm_fusion_host_resources::checked_single_add_min_host_scratch;
#[cfg(test)]
mod cancellation_tests;

pub mod legacy;
pub mod preparation;

pub use arity::RuntimeInvocationArity;
pub use callable_artifacts::{
    RuntimeCallableArtifactBundle, RuntimeCallableArtifactCommit, RuntimeCallableArtifactError,
    RuntimeCallableArtifactLimits, RuntimeCallableArtifactRetirement,
    RuntimeCallableArtifactService, RuntimePreparedCallableAdmission,
};
pub use capability::{RuntimeCapability, RuntimeCapabilityError};
pub use catch::{ActiveCatchFuture, ActiveCatchGuard, UnboxedActiveCatchFuture};
pub use cleanup_lifecycle::RuntimeLifecycleHost;
pub use graphics_callbacks::{
    GraphicsCallbackDisposition, GraphicsCallbackHandler, GraphicsCallbackHostEvent,
    GraphicsCallbackHostFailure, GraphicsCallbackHostTicket, GraphicsCallbackIngress,
    GraphicsCallbackIngressError, GraphicsCallbackIngressFailure, GraphicsCallbackKind,
    GraphicsCallbackRegistration, GraphicsCallbackResponder, GraphicsCallbackResult,
    RuntimeGraphicsCallbackQueue, WeakRuntimeGraphicsCallbackQueue,
};
pub use graphics_presentation::{
    GraphicsCallbackPolicy, GraphicsCallbackReceipt, GraphicsFlushDisposition, GraphicsFlushRate,
    GraphicsFlushReceipt, GraphicsFlushRequest, GraphicsPresentation,
    RuntimeGraphicsFlushOperation, RuntimeGraphicsPresentationService,
};
pub(crate) use host_dialog::{select_directory, select_file_open, select_file_save};
pub(crate) use host_document::request as request_document;
pub use host_document::{HostDocumentRequest, HostTextLocation, HostTextSelection};
pub(crate) use local_task_admission::LocalTaskCommandOrigin;
#[cfg(target_arch = "wasm32")]
pub use local_tasks::BrowserLocalTaskSupervisor;
pub use local_tasks::{
    CanonicalLocalTaskAdmission, LocalTaskOwnerToken, RuntimeLocalTaskFuture,
    RuntimeLocalTaskLease, RuntimeLocalTaskOutcome, RuntimeLocalTaskService,
};
#[cfg(not(target_arch = "wasm32"))]
pub use local_tasks::{RuntimeLocalTaskChildFactory, RuntimeLocalTaskSupervisor};
pub use managed_observations::{ManagedObservationLease, ManagedObservationPool};
pub use native_continuation::{
    run_native_continuation, run_owned_request_continuation, run_selected_callback_continuation,
    NativeContinuationPlacement,
};
#[cfg(target_arch = "wasm32")]
pub use preferences::BrowserPreferenceService;
#[cfg(not(target_arch = "wasm32"))]
pub use preferences::NativePreferenceService;
pub use preferences::{
    IsolatedMemoryPreferenceService, PreferenceGroups, PreferenceServiceError, PreferenceValues,
    RuntimePreferenceService,
};
pub use runtime::{
    ProgramExecutionAssignmentGuard, ProgramExecutionJobGuard, ProgramSourceSelectionError,
    RuntimeContext, RuntimeContextLocalId, RuntimeExecutionStack, RuntimeExecutionStackGuard,
    RuntimeLanguageMode, RuntimeLanguagePolicy, DEFAULT_CALLSTACK_LIMIT, DEFAULT_ERROR_NAMESPACE,
};
pub use scope::{ContextFuture, RuntimeContextGuard};
pub(crate) use services::complete_process_output_requires_evidence;
#[cfg(test)]
pub(crate) use services::test_managed_selection;
#[cfg(not(target_arch = "wasm32"))]
pub use services::RuntimeProcessService;
pub use services::{
    ForeignCall, HostInteraction, NativeCapability, ParallelCapability, RuntimeAccelerationService,
    RuntimeBaseWorkspaceService, RuntimeBuiltinService, RuntimeCallRequest, RuntimeCallRouter,
    RuntimeCallService, RuntimeCollectiveService, RuntimeDistributedCallRequest,
    RuntimeDistributedInvocation, RuntimeDistributedRetirement, RuntimeDistributedService,
    RuntimeDistributedSnapshot, RuntimeErrorService, RuntimeForeignService,
    RuntimeGraphObservationPermit, RuntimeGraphPlacementSelection,
    RuntimeHostEffectResourceService, RuntimeHostService, RuntimeHttpService,
    RuntimeManagedWholeWorkSelection, RuntimeNativeService, RuntimeObjectService,
    RuntimeParallelResources, RuntimeParallelService, RuntimePlacementFeedbackSource,
    RuntimePlacementService, RuntimeServiceFuture, RuntimeServicePorts, RuntimeSpmdAdmission,
    RuntimeSpmdOutput, RuntimeSpmdRetainedOutput, RuntimeSpmdService,
    RuntimeWholeWorkObservationAuthority, RuntimeWholeWorkObservationPermit,
    RuntimeWholeWorkProbeLease, RuntimeWholeWorkSelection, RuntimeWorkspaceService,
};
pub use source_evaluation::{
    RuntimeSourceEvaluationAuthority, RuntimeSourceEvaluationService, SessionSourceEvaluation,
    SourceEvaluationPolicy, SourceEvaluationProvenance,
};
pub(crate) use state::{RuntimeContextState, SourceState};
pub(crate) use super_constructor::RetainedSuperConstructorTarget;
pub use super_constructor::{
    RetainedClassDeclaration, RetainedConstructorDeclaration, RetainedSuperConstructorBinding,
};
pub use tcp::{
    RuntimeTcpBind, RuntimeTcpByteOrder, RuntimeTcpClient, RuntimeTcpClientStatus,
    RuntimeTcpCloseScope, RuntimeTcpConnect, RuntimeTcpInvocation, RuntimeTcpIssuer,
    RuntimeTcpReadMode, RuntimeTcpReadPublication, RuntimeTcpReadResult, RuntimeTcpServer,
    RuntimeTcpService,
};
pub(crate) use timer::{TimerAdmissionError, TimerFailureReceipt, TimerPhase};
pub use timing::RuntimeTimingService;
pub(crate) use timing::StopwatchToken;
pub use user_path::{
    RuntimeUserPathService, UserPathMutation, UserPathServiceError, UserPathSetting,
    UserPathSnapshot,
};
pub use vm_fusion_completion::{
    RuntimeVmFusionGenerationPermit, VmFusionCompletionRight, VmFusionFeedbackLease,
    VmFusionInstalledRegion, VmFusionRegionRegistration, VmFusionSemanticIdentity,
    VmFusionSessionRegistration,
};
pub use vm_fusion_host_route::{VmFusionHostInstruction, VmFusionHostRoute};

#[cfg(test)]
mod display_tests;
