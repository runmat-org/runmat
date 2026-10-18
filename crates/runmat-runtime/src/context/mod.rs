//! Explicit, session-owned runtime capabilities and invocation state.
//!
//! [`RuntimeContext`] is the one context propagated by Core, the VM, nested
//! calls, and future native/foreign/parallel adapters. Legacy ambient APIs are
//! migrated through [`legacy`] and must not become new semantic dependencies.

mod capability;
mod cancellation_context;
pub(crate) use cancellation_context::RuntimeCancellationContext;
mod cleanup_lifecycle;
mod callable_artifacts;
mod graphics_presentation;
#[cfg(all(feature = "gui", not(target_arch = "wasm32")))]
mod native_graphics;
#[cfg(all(feature = "gui", not(target_arch = "wasm32")))]
pub use native_graphics::NativeGraphicsAttachment;
#[cfg(all(feature = "gui", not(target_arch = "wasm32")))]
pub use runmat_plot::gui::host_loop::WindowOpenPort as NativeWindowOpenPort;
#[cfg(all(feature = "gui", not(target_arch = "wasm32")))]
pub use runmat_plot::gui::WindowConfig as NativeWindowConfig;
mod graphics_callbacks;
pub(crate) use graphics_callbacks::GraphicsCallbackMutationScope;
#[cfg(not(target_arch = "wasm32"))]
mod foreign_callback_lineage;
mod arity;
mod catch;
pub(crate) mod call_projection;
mod local_tasks;
mod local_task_admission;
mod native_continuation;
mod preferences;
pub(crate) mod poll_projection;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod process_invocation;
mod user_path;
mod host_document;
mod host_dialog;
mod managed_observations;
pub(crate) mod output_projection;
pub(crate) mod overlay_storage;
pub mod host_effect;
pub mod host_allocation;
pub(crate) mod host_presentation;
pub(crate) mod provider_operation;
#[cfg(test)]
pub(crate) mod document_test_support;
#[cfg(test)]
pub(crate) mod physical_test_support;
mod runtime;
#[cfg(feature = "plot-core")]
pub(crate) use runtime::BackgroundCallbackAuthority;
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
    GraphicsBoundCallbackResponder, GraphicsCallbackDisposition, GraphicsCallbackDriveError,
    GraphicsCallbackHandler, GraphicsCallbackHostEvent, GraphicsCallbackHostFailure,
    GraphicsCallbackHostTicket, GraphicsCallbackIngress, GraphicsCallbackIngressError,
    GraphicsCallbackIngressFailure, GraphicsCallbackKind, GraphicsCallbackRegistration,
    GraphicsCallbackResponder, GraphicsCallbackResult, GraphicsCallbackRolePayload,
    GraphicsSelectedBoundCallbackResponder, RuntimeGraphicsCallbackQueue,
    WeakRuntimeGraphicsCallbackQueue,
};
#[cfg(feature = "plot-core")]
pub(crate) use graphics_presentation::complete_selected as complete_selected_graphics;
#[cfg(all(test, feature = "plot-core"))]
pub(crate) use graphics_presentation::GraphicsFlushEffectSource;
#[cfg(feature = "plot-core")]
pub(crate) use graphics_presentation::{
    attach_disposition as attach_graphics_disposition,
    await_completion as await_graphics_completion,
};
#[cfg(all(feature = "gui", not(target_arch = "wasm32")))]
pub(crate) use graphics_presentation::{retire_selected_scene, stage_selected_scene};
pub use graphics_presentation::{
    GraphicsCallbackPolicy, GraphicsCallbackReceipt, GraphicsFlushDisposition, GraphicsFlushRate,
    GraphicsFlushReceipt, GraphicsFlushRequest, GraphicsPresentation, GraphicsSceneStaging,
    RuntimeGraphicsFlushOperation, RuntimeGraphicsPresentationService,
};
#[cfg(feature = "plot-core")]
pub use graphics_presentation::{RuntimeGraphicsScene, RuntimeGraphicsSceneProposal};
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
    run_native_continuation, run_owned_request_continuation,
    run_rooted_function_native_continuation, run_selected_callback_continuation,
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
pub use scope::{
    ContextFuture, PreparedRuntimeFutureScope, PreparedRuntimeScope, RuntimeContextGuard,
};
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
    RuntimeParallelResources, RuntimeParallelService, RuntimePlacementFeedbackDelivery,
    RuntimePlacementFeedbackSource, RuntimePlacementService, RuntimeServiceFuture,
    RuntimeServicePorts, RuntimeSpmdAdmission, RuntimeSpmdGangLease, RuntimeSpmdOutput,
    RuntimeSpmdRetainedOutput, RuntimeSpmdService, RuntimeWholeWorkObservationAuthority,
    RuntimeWholeWorkObservationPermit, RuntimeWholeWorkProbeLease, RuntimeWholeWorkSelection,
    RuntimeWorkspaceService,
};
pub use source_evaluation::{
    execute_assignin_effect, execute_selected_dynamic_workspace, DynamicWorkspaceInvocation,
    NumericTextEvaluation, RuntimeSourceEvaluationAuthority, RuntimeSourceEvaluationService,
    SessionSourceEvaluation, SessionSourcePurpose, SourceEvaluationPolicy,
    SourceEvaluationProvenance,
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
    RuntimeVmFusionGenerationPermit, VmFusionCompletionRight, VmFusionDeclineRight,
    VmFusionFeedbackLease, VmFusionInstalledRegion, VmFusionRegionRegistration,
    VmFusionSemanticIdentity, VmFusionSessionRegistration,
};
pub use vm_fusion_host_route::{VmFusionHostInstruction, VmFusionHostOperation, VmFusionHostRoute};

#[cfg(test)]
mod display_tests;
