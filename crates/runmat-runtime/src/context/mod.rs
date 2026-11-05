//! Explicit, session-owned runtime capabilities and invocation state.
//!
//! [`RuntimeContext`] is the one context propagated by Core, the VM, nested
//! calls, and future native/foreign/parallel adapters. Legacy ambient APIs are
//! migrated through [`legacy`] and must not become new semantic dependencies.

mod cancellation_context;
mod capability;
pub(crate) mod gpu_device;
pub(crate) use cancellation_context::RuntimeCancellationContext;
pub use cancellation_context::RuntimeCancellationScope;
mod callable_artifacts;
mod cleanup_completion;
mod cleanup_lifecycle;
pub use cleanup_completion::RuntimeFunctionCleanup;
mod current_class_product;
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
mod arity;
pub(crate) mod call_projection;
mod catch;
#[cfg(test)]
pub(crate) mod document_test_support;
mod environment;
#[cfg(not(target_arch = "wasm32"))]
mod foreign_callback_lineage;
pub mod host_allocation;
mod host_dialog;
mod host_document;
pub mod host_effect;
pub(crate) mod host_presentation;
mod local_task_admission;
mod local_tasks;
mod managed_observations;
mod native_continuation;
pub(crate) mod output_projection;
pub(crate) mod overlay_storage;
#[cfg(test)]
pub(crate) mod physical_test_support;
pub(crate) mod poll_projection;
mod portable_host;
mod retained_console;
pub use retained_console::{
    RuntimeConsoleAdmission, RuntimeConsoleScopeIdentity, RuntimeConsoleScopeRef,
    RuntimeRetainedConsoleSource,
};
#[cfg(not(target_arch = "wasm32"))]
mod console_scope;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use console_scope::RuntimeConsoleScope;
#[cfg(not(target_arch = "wasm32"))]
mod process_console;
#[cfg(not(target_arch = "wasm32"))]
pub use process_console::ProcessConsoleDecoder;
mod acceleration_policy;
mod preferences;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod process_invocation;
pub(crate) mod provider_operation;
pub use acceleration_policy::{
    current_acceleration_policy, RuntimeAccelerationPolicy, RuntimeProviderPreference,
};
mod selected_random;
mod user_path;
pub(crate) use portable_host::{accepts_queued_interaction, check_active_host};
pub use portable_host::{
    NonCapturingPortableHost, PortableConsoleHost, PortableConsoleSink,
    PortableForwardingConsoleHost,
};
mod runtime;
#[cfg(feature = "plot-core")]
pub(crate) use runtime::BackgroundCallbackAuthority;
mod native_region_host_route;
mod scope;
mod services;
mod source_evaluation;
mod state;
mod super_constructor;
mod tcp;
mod timer;
mod timing;
mod vm_fusion_completion;
mod vm_fusion_host_outputs;
mod vm_fusion_host_resources;
mod vm_fusion_host_route;
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
pub use current_class_product::{
    admit_current_class_product_with, CurrentClassProductFuture, CurrentClassProductResolution,
    CurrentClassProductResolver,
};
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
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use native_continuation::run_selected_callback_continuation_with;
pub use native_continuation::{
    run_native_continuation, run_native_continuation_with, run_owned_request_continuation,
    run_rooted_function_native_continuation, run_selected_callback_continuation,
    NativeContinuationPlacement,
};
pub use native_region_host_route::{NativeRegionHostOperation, NativeRegionHostRoute};
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
    RuntimeLanguageMode, RuntimeLanguagePolicy, SelectedGraphFunction,
    SelectedGraphFunctionResolver, SelectedMethodDispatch, DEFAULT_CALLSTACK_LIMIT,
    DEFAULT_ERROR_NAMESPACE,
};
pub(crate) use scope::{
    with_original_scope, LazyRuntimeScope, PreparedUnselectedScope, RuntimeScopeFactory,
};
pub use scope::{
    ContextFuture, DeferredRuntimeFuture, PreparedRuntimeFutureScope, PreparedRuntimeScope,
    RuntimeContextGuard,
};
pub(crate) use services::complete_process_output_requires_evidence;
#[cfg(test)]
pub(crate) use services::test_managed_selection;
#[cfg(not(target_arch = "wasm32"))]
pub use services::RuntimeProcessService;
pub use services::{
    ForeignCall, ForeignCallCompletion, HostInteraction, NativeCapability,
    NativeLinkedCallbackPort, NativeLinkedDynamicPort, NativeLinkedHostPorts, ParallelCapability,
    RuntimeAccelerationService, RuntimeAnalysisService, RuntimeBaseWorkspaceService,
    RuntimeBuiltinService, RuntimeCallRequest, RuntimeCallRouter, RuntimeCallService,
    RuntimeCollectiveService, RuntimeDistributedCallRequest, RuntimeDistributedInvocation,
    RuntimeDistributedRetirement, RuntimeDistributedService, RuntimeDistributedSnapshot,
    RuntimeErrorService, RuntimeForeignService, RuntimeGraphObservationPermit,
    RuntimeGraphPlacementSelection, RuntimeHostEffectResourceService, RuntimeHostService,
    RuntimeHttpService, RuntimeManagedWholeWorkSelection, RuntimeNativeService,
    RuntimeObjectService, RuntimeParallelResources, RuntimeParallelService,
    RuntimePlacementFeedbackDelivery, RuntimePlacementFeedbackSource, RuntimePlacementService,
    RuntimeProgramCallable, RuntimeProgramService, RuntimeSelectedLocalServices,
    RuntimeServiceFuture, RuntimeServicePorts, RuntimeSignalMaterializationRequest,
    RuntimeSpmdAdmission, RuntimeSpmdGangLease, RuntimeSpmdOutput, RuntimeSpmdRetainedOutput,
    RuntimeSpmdService, RuntimeWholeWorkObservationAuthority, RuntimeWholeWorkObservationPermit,
    RuntimeWholeWorkProbeLease, RuntimeWholeWorkSelection, RuntimeWorkspaceService,
};
pub use source_evaluation::{
    execute_assignin_effect, execute_assignin_effect_view, execute_selected_dynamic_workspace,
    execute_selected_dynamic_workspace_retained, DynamicWorkspaceInvocation, NumericTextEvaluation,
    RuntimeSourceEvaluationAuthority, RuntimeSourceEvaluationService, SessionSourceEvaluation,
    SessionSourcePurpose, SourceEvaluationPolicy, SourceEvaluationProvenance,
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
pub(crate) use timing::StopwatchToken;
pub use timing::{DeterministicTimingService, RuntimeCalendarInstant, RuntimeTimingService};
pub use user_path::{
    RuntimeUserPathService, UserPathMutation, UserPathServiceError, UserPathSetting,
    UserPathSnapshot,
};
pub use vm_fusion_completion::{
    RuntimeVmFusionGenerationPermit, VmFusionCompletionRight, VmFusionDeclineRight,
    VmFusionFeedbackLease, VmFusionInstalledRegion, VmFusionRegionRegistration,
    VmFusionSemanticIdentity, VmFusionSessionRegistration,
};
pub use vm_fusion_host_outputs::{
    VmFusionHostOutputEntry, VmFusionHostOutputs, VmFusionHostRealSlotSpec, VmFusionHostRetirement,
};
pub use vm_fusion_host_route::{VmFusionHostInstruction, VmFusionHostOperation, VmFusionHostRoute};

#[cfg(test)]
mod display_tests;
