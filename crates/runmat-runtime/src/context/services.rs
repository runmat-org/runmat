use super::{
    RuntimeCallableArtifactService, RuntimeCapability, RuntimeCapabilityError, RuntimeContext,
    RuntimeGraphicsPresentationService, RuntimePreferenceService, RuntimeTcpService,
    RuntimeTimingService, RuntimeUserPathService,
};
use crate::builtin::RuntimeBuiltinBinding;
use crate::class_registry::RuntimeClass;
use crate::warning_store::RuntimeWarning;
use crate::RuntimeError;
use runmat_types::{BuiltinId, DistributedValueContract, DistributionScheme, LabRank};
use runmat_value::Value;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::Arc;

pub type RuntimeServiceFuture<T> = Pin<Box<dyn Future<Output = T> + 'static>>;

mod call;
mod selected_external;
mod selected_local;
mod selected_network;
mod selected_timing;
pub use call::{RuntimeCallRequest, RuntimeCallRouter, RuntimeCallService};
pub use selected_external::{
    NativeLinkedCallbackPort, NativeLinkedDynamicPort, NativeLinkedHostPorts,
};
pub use selected_local::RuntimeSelectedLocalServices;
use selected_timing::RuntimeTimingPort;

/// Invocation-scoped builtin authority used by exact compiled products.
/// When installed, the dispatcher must not fall back to process-global
/// discovery for a missing name.
pub trait RuntimeBuiltinService {
    fn bindings_by_name(&self, name: &str) -> Vec<RuntimeBuiltinBinding>;

    /// Stable generation for this exact selected scoped binding while it
    /// remains installed. A mutable service must change this when the binding
    /// or registration changes. None declines session-local timing feedback.
    /// Function addresses may validate membership but are not revision data.
    fn binding_registration_revision(
        &self,
        _binding: RuntimeBuiltinBinding,
    ) -> Option<runmat_execution::Digest> {
        None
    }
}

pub trait RuntimeWorkspaceService {
    /// Retain the current immutable read generation without synchronizing or
    /// publishing pending workspace effects. The default declines rather than
    /// constructing a detached Value/name mirror.
    fn capture_workspace_generation_with_context(
        &self,
        _context: &RuntimeContext,
    ) -> Result<Option<crate::workspace::LogicalWorkspaceGeneration>, RuntimeError> {
        Ok(None)
    }

    /// Capture one immutable current target generation and its matching
    /// completion authority. Implementations must keep the issuing service
    /// alive in the returned capture and must not construct a detached Value
    /// mirror. The default refuses before effects.
    fn capture_dynamic_workspace_target_with_context(
        self: Rc<Self>,
        _context: &RuntimeContext,
    ) -> Result<Option<crate::workspace::DynamicWorkspaceTargetCapture>, RuntimeError> {
        Ok(None)
    }

    /// Exact lexical/dynamic caller workspace for `evalin('caller', ...)`.
    /// Absence is authoritative and must never be replaced by current/base.
    fn caller_workspace_with_context(
        &self,
        _context: &RuntimeContext,
    ) -> Result<Option<Rc<dyn RuntimeWorkspaceService>>, RuntimeError> {
        Ok(None)
    }

    /// Capture the exact transactional identity used by a later whole-target
    /// completion. Generic adapters do not gain batch authority merely by
    /// supporting individual assignments.
    fn capture_workspace_completion_identity_with_context(
        &self,
        _context: &RuntimeContext,
    ) -> Result<Option<crate::workspace::WorkspaceCompletionIdentity>, RuntimeError> {
        Ok(None)
    }

    /// Install a normalized whole-target completion as one prepared mutation.
    /// Implementations must reject an identity they did not issue and must not
    /// fall back to committing a prefix through the scalar mutation methods.
    fn apply_workspace_completion_with_context(
        &self,
        context: &RuntimeContext,
        _identity: crate::workspace::WorkspaceCompletionIdentity,
        source: &dyn crate::workspace::WorkspaceCompletionSource,
    ) -> Result<(), RuntimeError> {
        if source.is_empty() {
            return self.ensure_effect_authority_with_context(context);
        }
        Err(crate::runtime_error::semantic_error(
            "RunMat:Workspace:ExactCompletionUnavailable",
            "exact transactional workspace completion is unavailable",
        ))
    }

    /// Prepare every Base value, name, delta and publication owner before a
    /// composite Native/lexical completion becomes visible. The returned
    /// token is bound to this service's exact completion identity; the
    /// default provides no whole-target authority.
    fn prepare_workspace_completion_with_context(
        &self,
        _context: &RuntimeContext,
        _identity: crate::workspace::WorkspaceCompletionIdentity,
        _source: &dyn crate::workspace::WorkspaceCompletionSource,
    ) -> Result<crate::workspace::PreparedWorkspaceCompletion, RuntimeError> {
        Err(crate::runtime_error::semantic_error(
            "RunMat:Workspace:ExactCompletionUnavailable",
            "exact transactional workspace completion is unavailable",
        ))
    }

    /// Operation-bearing retirement keeps publication and callback admission
    /// attached to the invocation that requested the effect.
    fn detach_selected_with_context(
        &self,
        _context: &RuntimeContext,
        selection: &crate::workspace::WorkspaceDetachSelection,
    ) -> Result<crate::workspace::DetachedWorkspaceValues, RuntimeError> {
        self.detach_selected(selection)
    }

    /// Retire selected bindings from this exact workspace before callbacks.
    /// Legacy adapters without atomic authority fail closed.
    fn detach_selected(
        &self,
        _selection: &crate::workspace::WorkspaceDetachSelection,
    ) -> Result<crate::workspace::DetachedWorkspaceValues, RuntimeError> {
        Err(crate::runtime_error::semantic_error(
            "RunMat:Workspace:ExactDetachUnavailable",
            "exact current-workspace detachment is unavailable",
        ))
    }

    fn visit_retained_values(&self, _visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        None
    }
    fn lookup(&self, name: &str) -> Option<Value>;
    fn snapshot(&self) -> Vec<(String, Value)>;
    fn global_names(&self) -> Vec<String>;

    /// Checked workspace reads used by canonical session workspace owners.
    /// Legacy services may retain their infallible read contract through the
    /// defaults, while authority-bearing services override these methods so a
    /// closed or temporarily unavailable owner is never mistaken for an empty
    /// workspace.
    fn try_lookup(&self, name: &str) -> Result<Option<Value>, RuntimeError> {
        Ok(self.lookup(name))
    }

    fn try_snapshot(&self) -> Result<Vec<(String, Value)>, RuntimeError> {
        Ok(self.snapshot())
    }

    fn try_global_names(&self) -> Result<Vec<String>, RuntimeError> {
        Ok(self.global_names())
    }

    fn try_lookup_with_context(
        &self,
        _context: &RuntimeContext,
        name: &str,
    ) -> Result<Option<Value>, RuntimeError> {
        self.try_lookup(name)
    }

    fn try_snapshot_with_context(
        &self,
        _context: &RuntimeContext,
    ) -> Result<Vec<(String, Value)>, RuntimeError> {
        self.try_snapshot()
    }

    fn try_global_names_with_context(
        &self,
        _context: &RuntimeContext,
    ) -> Result<Vec<String>, RuntimeError> {
        self.try_global_names()
    }

    /// Admit an ordered projection for this exact current-workspace service.
    ///
    /// Sequential caller-frame adapters return `None`; a service backed by a
    /// transactional owner may return a projection without exposing ownership
    /// or close authority. Callers must not substitute the session base port
    /// when this current-frame operation is unavailable.
    fn begin_workspace_projection(
        &self,
    ) -> Result<Option<crate::workspace::BaseWorkspaceProjection>, RuntimeError> {
        Ok(None)
    }

    /// Confirm that this exact workspace service may still authorize a new
    /// externally visible effect. Callers retain the service across suspension
    /// and invoke this checkpoint immediately before and after awaited effect
    /// work; they must not re-resolve a replacement from ambient context.
    ///
    /// Ordinary frame-local services have no independently revocable owner and
    /// therefore remain current for as long as their retained service exists.
    /// Authority-bearing services override this method to report closure or
    /// another typed liveness failure.
    fn ensure_effect_authority(&self) -> Result<(), RuntimeError> {
        Ok(())
    }

    fn ensure_effect_authority_with_context(
        &self,
        _context: &RuntimeContext,
    ) -> Result<(), RuntimeError> {
        self.ensure_effect_authority()
    }

    /// Admit an ordered executor-owned mutation without reconciling an
    /// externally changed workspace view. Executor register stores use this
    /// path after their exact frame lifecycle check; external/reentrant
    /// effects continue to use [`Self::ensure_effect_authority`].
    fn ensure_mutation_authority(&self) -> Result<(), RuntimeError> {
        self.ensure_effect_authority()
    }

    /// Check a register mutation under the captured invocation without
    /// reconciling an external workspace effect.
    fn ensure_mutation_authority_with_context(
        &self,
        _context: &RuntimeContext,
    ) -> Result<(), RuntimeError> {
        self.ensure_mutation_authority()
    }

    /// Revision of an authority-backed workspace after its latest admission
    /// checkpoint. Exact executor adapters use this to avoid materializing a
    /// complete workspace when no externally visible state changed.
    fn workspace_revision(
        &self,
    ) -> Result<Option<crate::workspace::WorkspaceRevision>, RuntimeError> {
        Ok(None)
    }

    fn workspace_revision_with_context(
        &self,
        _context: &RuntimeContext,
    ) -> Result<Option<crate::workspace::WorkspaceRevision>, RuntimeError> {
        self.workspace_revision()
    }

    /// Names whose base-workspace identity resolves through the separate
    /// session-global value store. Canonical base services override this with
    /// their revisioned namespace authority. Executor frame services must
    /// explicitly route authorized operations through their captured delegate.
    /// A generic current-workspace service does not acquire this authority by
    /// implementing ordinary value reads and writes, so the defaults fail
    /// closed.
    fn base_global_binding_names(&self) -> Result<Vec<String>, RuntimeError> {
        Err(base_namespace_capability_error(
            "enumerate base-workspace global bindings",
        ))
    }

    fn is_base_global_binding(&self, _name: &str) -> Result<bool, RuntimeError> {
        Err(base_namespace_capability_error(
            "inspect a base-workspace global binding",
        ))
    }

    fn bind_base_global(&self, _name: &str) -> Result<(), RuntimeError> {
        Err(base_namespace_capability_error(
            "bind a base-workspace global",
        ))
    }

    fn unbind_base_global(&self, _name: &str) -> Result<(), RuntimeError> {
        Err(base_namespace_capability_error(
            "unbind a base-workspace global",
        ))
    }

    fn assign_base_local(&self, name: &str, value: Value) -> Result<(), RuntimeError> {
        self.unbind_base_global(name)?;
        self.assign(name, value)
    }

    fn assign(&self, name: &str, value: Value) -> Result<(), RuntimeError>;

    /// Operation-bearing mutation entrypoints. Canonical managed services use
    /// the current context to admit storage before changing their namespace.
    /// Defaults retain the behavior of external services without that policy.
    fn assign_with_context(
        &self,
        _context: &RuntimeContext,
        name: &str,
        value: Value,
    ) -> Result<(), RuntimeError> {
        self.assign(name, value)
    }

    /// Prepare the publication owner before cloning a register-held value.
    fn assign_ref_with_context(
        &self,
        context: &RuntimeContext,
        name: &str,
        value: &Value,
    ) -> Result<(), RuntimeError> {
        self.assign_with_context(context, name, value.clone())
    }

    fn bind_base_global_with_context(
        &self,
        _context: &RuntimeContext,
        name: &str,
    ) -> Result<(), RuntimeError> {
        self.bind_base_global(name)
    }

    /// Bind the workspace spelling to its exact session destination. A
    /// generic adapter may not erase a named alias or legacy slot to a bool.
    fn bind_base_global_target_with_context(
        &self,
        _context: &RuntimeContext,
        _name: &str,
        _target: &crate::workspace::GlobalBindingTarget,
    ) -> Result<(), RuntimeError> {
        Err(base_namespace_capability_error(
            "bind a typed base-workspace global",
        ))
    }

    fn unbind_base_global_with_context(
        &self,
        _context: &RuntimeContext,
        name: &str,
    ) -> Result<(), RuntimeError> {
        self.unbind_base_global(name)
    }

    fn assign_base_local_with_context(
        &self,
        _context: &RuntimeContext,
        name: &str,
        value: Value,
    ) -> Result<(), RuntimeError> {
        self.assign_base_local(name, value)
    }

    /// Publish an executor register write whose register file is updated by
    /// the caller immediately after admission. Frame adapters override this to
    /// avoid scheduling the same write for callback reconciliation.
    fn assign_executor_slot(
        &self,
        _slot: usize,
        name: &str,
        value: Value,
    ) -> Result<(), RuntimeError> {
        self.assign(name, value)
    }

    /// Admit the publication from its borrowed register value before any
    /// publication alias is cloned. Managed workspace owners override this
    /// boundary to prepare their storage against the current operation.
    fn assign_executor_slot_ref(
        &self,
        _context: &RuntimeContext,
        slot: usize,
        name: &str,
        value: &Value,
    ) -> Result<(), RuntimeError> {
        self.assign_executor_slot(slot, name, value.clone())
    }

    /// Publish an executor register generation that was admitted before the
    /// workspace effect began. Frame services override this boundary to move
    /// the exact retained generation into their register owner; compatibility
    /// services may inspect it only through the existing borrowed boundary.
    fn assign_executor_slot_lease_with_context(
        &self,
        context: &RuntimeContext,
        slot: usize,
        name: &str,
        value: crate::workspace::WorkspaceSlotReadLease,
    ) -> Result<(), RuntimeError> {
        self.assign_executor_slot_ref(context, slot, name, value.value())
    }

    /// Establish a named session-global alias for one executor slot. The
    /// service owns frame scope and decides whether the declaration also
    /// removes a shadowed base-workspace value.
    fn declare_global_binding(&self, _slot: usize, _name: &str) -> Result<(), RuntimeError> {
        Err(base_namespace_capability_error(
            "declare a workspace global binding",
        ))
    }
    fn clear(&self) -> Result<(), RuntimeError>;
    fn remove(&self, name: &str) -> Result<(), RuntimeError>;

    fn declare_global_binding_with_context(
        &self,
        _context: &RuntimeContext,
        slot: usize,
        name: &str,
    ) -> Result<(), RuntimeError> {
        self.declare_global_binding(slot, name)
    }

    /// Establish a legacy slot-addressed global while retaining the compiler
    /// name used by the current workspace frame. Services without exact slot
    /// authority must refuse before mutating either namespace.
    fn declare_legacy_global_binding_with_context(
        &self,
        _context: &RuntimeContext,
        _slot: usize,
        _name: &str,
    ) -> Result<(), RuntimeError> {
        Err(base_namespace_capability_error(
            "declare a legacy workspace global binding",
        ))
    }

    fn clear_with_context(&self, _context: &RuntimeContext) -> Result<(), RuntimeError> {
        self.clear()
    }

    fn remove_with_context(
        &self,
        _context: &RuntimeContext,
        name: &str,
    ) -> Result<(), RuntimeError> {
        self.remove(name)
    }
}

fn base_namespace_capability_error(operation: &'static str) -> RuntimeError {
    RuntimeCapabilityError::new(RuntimeCapability::BaseWorkspace, operation).into_runtime_error()
}

/// Session-lifetime authority for the canonical base workspace.
///
/// This is deliberately separate from [`RuntimeWorkspaceService`], whose
/// installed value may describe a procedure or caller frame. Possessing a
/// current-frame service never grants base-workspace ownership; the host must
/// compose this port explicitly for the session.
pub trait RuntimeBaseWorkspaceService {
    fn owner(&self) -> crate::workspace::BaseWorkspaceOwner;

    fn invocation_projection(
        &self,
    ) -> Option<std::rc::Rc<crate::workspace::BaseWorkspaceProjectionService>> {
        None
    }

    /// Publish one base write under the selected service's invocation owner.
    /// Standalone hosts retain the canonical direct-owner implementation;
    /// Core installs its observed projection for a running source request.
    fn assign_logical_ref_with_context(
        &self,
        variables: &crate::workspace::session::SessionVariableAuthority,
        context: &RuntimeContext,
        name: &str,
        value: &Value,
    ) -> Result<(), RuntimeError> {
        crate::workspace::assign_logical_base_value_ref_with_context(
            &self.owner(),
            variables,
            context,
            name,
            value,
        )
    }
}

pub trait RuntimeObjectService {
    /// Complete retained values, including published static properties and
    /// implementation-owned captures. `None` means alias visibility is not
    /// available; callers must not interpret it as an empty root set.
    fn visit_retained_values(&self, _visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        None
    }

    fn class(&self, name: &str) -> Option<RuntimeClass>;
    fn register_class(&self, class: RuntimeClass) -> Result<(), RuntimeError>;
    fn static_property(&self, class: &str, property: &str) -> Option<Value>;
    fn set_static_property(
        &self,
        class: &str,
        property: &str,
        value: Value,
    ) -> Result<(), RuntimeError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostInteraction {
    Line { prompt: String, echo: bool },
    KeyPress { prompt: String },
}

pub trait RuntimeHostService {
    fn console(&self, stream: crate::console::ConsoleStream, text: crate::console::ConsoleText);

    /// A retaining host owns the normal console inventory. `evalc` is separate.
    fn owns_console_retention(&self) -> bool {
        false
    }

    fn console_limits(&self) -> Option<runmat_execution::console::ConsoleLimits> {
        None
    }

    fn console_admission(&self) -> Option<std::sync::Arc<dyn super::RuntimeConsoleAdmission>> {
        None
    }

    /// Retain the original child owner until the outer terminal is encoded.
    /// Only callers requesting console capture need this capability.
    fn retain_console_source(
        &self,
        _source: std::sync::Arc<dyn super::RuntimeRetainedConsoleSource>,
    ) -> Result<(), RuntimeError> {
        Err(
            crate::build_runtime_error("host cannot retain child console scopes")
                .with_identifier("RunMat:console:CaptureUnavailable")
                .build(),
        )
    }

    /// Sticky host failures cannot be hidden by a caught language exception.
    fn check_execution(&self) -> Result<(), RuntimeError> {
        Ok(())
    }

    /// Portable hosts reject interaction even when an ambient response is queued.
    fn accepts_queued_interaction(&self) -> bool {
        true
    }

    fn interact(
        &self,
        request: HostInteraction,
    ) -> RuntimeServiceFuture<Result<crate::interaction::InteractionResponse, RuntimeError>>;

    fn warning(&self, warning: RuntimeWarning);

    /// Complete only after the application host accepts the requested document
    /// interaction. Hosts without an editor/viewer must return an error.
    fn open_document(
        &self,
        _request: super::HostDocumentRequest,
    ) -> RuntimeServiceFuture<Result<(), RuntimeError>> {
        Box::pin(async { Err(super::host_document::unavailable()) })
    }

    fn select_file_open(
        &self,
        _request: runmat_filesystem::OpenFileDialogRequest,
    ) -> RuntimeServiceFuture<
        Result<Option<runmat_filesystem::OpenFileDialogSelection>, RuntimeError>,
    > {
        Box::pin(async { Ok(None) })
    }

    fn select_file_save(
        &self,
        _request: runmat_filesystem::SaveFileDialogRequest,
    ) -> RuntimeServiceFuture<
        Result<Option<runmat_filesystem::SaveFileDialogSelection>, RuntimeError>,
    > {
        Box::pin(async { Ok(None) })
    }

    fn select_directory(
        &self,
        _request: runmat_filesystem::DirectoryDialogRequest,
    ) -> RuntimeServiceFuture<
        Result<Option<runmat_filesystem::DirectoryDialogSelection>, RuntimeError>,
    > {
        Box::pin(async { Ok(None) })
    }
}

/// Host-owned asynchronous HTTP authority. The returned future must not start
/// I/O until first poll and must abort any submitted request when dropped.
/// This includes a request awaiting response-body bytes, not only headers.
pub trait RuntimeHttpService {
    fn send(
        &self,
        runtime: super::RuntimeContext,
        request: crate::http::HttpRequest,
    ) -> RuntimeServiceFuture<Result<crate::http::HttpResponse, crate::http::TransportError>>;

    /// Open one response body without collecting it. The service must reject
    /// an advertised length above `accepted_limit` before returning the head,
    /// and reject any body chunk that would cross it before publishing bytes.
    /// The caller still owns its own written-byte limit and partial-file RAII.
    /// This bounds total accepted bytes, not an undocumented transport-chunk
    /// or TLS allocation peak; finite memory quotas need separate admission.
    fn open_bounded(
        &self,
        _runtime: super::RuntimeContext,
        request: crate::http::HttpRequest,
        _accepted_limit: u64,
    ) -> RuntimeServiceFuture<Result<crate::http::HttpBodyStream, crate::http::TransportError>>
    {
        Box::pin(async move {
            Err(crate::http::TransportError::unsupported_streaming(
                &request.url,
            ))
        })
    }
}

/// Native host-owned process execution. The future starts no child until
/// polled, and dropping it must terminate its owned process tree.
#[cfg(not(target_arch = "wasm32"))]
pub trait RuntimeProcessService {
    fn output_complete(
        &self,
        command: runmat_process_host::HostCommand,
    ) -> RuntimeServiceFuture<
        runmat_process_host::ProcessHostResult<runmat_process_host::CompleteProcessOutput>,
    >;

    fn output_complete_with_input(
        &self,
        command: runmat_process_host::HostCommand,
        input: Vec<u8>,
    ) -> RuntimeServiceFuture<
        runmat_process_host::ProcessHostResult<runmat_process_host::CompleteProcessOutput>,
    >;
}

/// Complete native child output is not currently physically bounded. Keep
/// this admission fact at the service boundary for all process consumers.
pub(crate) fn complete_process_output_requires_evidence(
    runtime: Option<&super::RuntimeContext>,
) -> bool {
    runtime
        .and_then(|runtime| runtime.service_ports().parallel())
        .is_some_and(|port| port.placement_resources().memory_available_bytes.is_some())
}

pub trait RuntimeErrorService {
    fn report(&self, error: &RuntimeError);
}

pub trait RuntimeAccelerationService {
    fn numeric_region_capability(
        &self,
    ) -> Option<crate::numeric_region::service::NumericRegionCapability> {
        None
    }

    fn execute_numeric_region(
        &self,
        context: super::RuntimeContext,
        request: crate::numeric_region::service::NativeNumericRegionRequest,
    ) -> Result<
        crate::execution::RuntimeContinuationFuture<
            'static,
            crate::numeric_region::service::NativeNumericRegionOutcome,
        >,
        RuntimeError,
    > {
        crate::execution::admit_native_continuation_with(&context, || async move {
            drop(request);
            Ok(crate::numeric_region::service::NativeNumericRegionOutcome::Ineligible)
        })
    }

    /// Transitional transfer-ownership preparation for an already selected
    /// inventory implementation. Never applies to canonical or overridden calls.
    /// L23 removes this port once their owning domains declare admitted forms.
    fn prepare_legacy(
        &self,
        _context: super::RuntimeContext,
        _name: String,
        args: Vec<Value>,
    ) -> RuntimeServiceFuture<Result<Vec<Value>, RuntimeError>> {
        Box::pin(async move { Ok(args) })
    }

    /// Called only after the owning domain has admitted the complete call.
    /// Controls and parsed options remain in that domain, never in this service.
    /// Implementations must guard fresh resources across preparation itself:
    /// errors or cancellation before returning the owned carrier must release
    /// those resources without releasing borrowed inputs.
    fn prepare_admitted(
        &self,
        _context: super::RuntimeContext,
        request: super::preparation::AdmittedBuiltinPreparation,
    ) -> RuntimeServiceFuture<Result<super::preparation::PreparedBuiltinOperands, RuntimeError>>
    {
        Box::pin(async move {
            request
                .into_operands()
                .with_unbounded_values(super::preparation::PreparedBuiltinOperands::passthrough)?
        })
    }
}

/// Deferred candidate evidence with no runtime context/service ownership. Send
/// prevents a source from retaining the session's Rc graph; callbacks still do
/// not deliver feedback themselves. The owning session polls nonblocking.
pub trait RuntimePlacementFeedbackSource: Send {
    fn signature(&self) -> &runmat_execution::PlacementSignature;
    fn take_ready(&mut self) -> Option<RuntimePlacementFeedbackDelivery>;
    /// A completed cancellation or abandonment has no feedback sample. The
    /// owner may retire its source without manufacturing a failed observation.
    fn terminal_without_feedback(&self) -> bool {
        false
    }
}

/// Completion delivery preserves the selecting authority domain. A whole-work
/// selection retains its exact permit and exploratory lease until owner-thread
/// insertion; it is never represented as a graph observation permit.
pub enum RuntimePlacementFeedbackDelivery {
    Graph {
        feedback: runmat_execution::PlacementFeedback,
        permit: std::sync::Arc<dyn RuntimeGraphObservationPermit>,
    },
    WholeWork {
        selected: RuntimeWholeWorkSelection,
        elapsed_ns: u64,
        outcome: runmat_execution::WholeWorkObservationOutcome,
    },
    ManagedWholeWork {
        selected: RuntimeManagedWholeWorkSelection,
        elapsed_ns: u64,
        outcome: runmat_execution::WholeWorkObservationOutcome,
    },
}

impl RuntimePlacementFeedbackDelivery {
    /// Portable diagnostic snapshot only; this does not transfer authority.
    pub fn snapshot(&self) -> Option<runmat_execution::PlacementFeedback> {
        match self {
            // Finite completion transports the original compact selection;
            // it does not reconstruct a portable signature for diagnostics.
            Self::ManagedWholeWork { .. } => None,
            Self::Graph { feedback, .. } => Some(feedback.clone()),
            Self::WholeWork {
                selected,
                elapsed_ns,
                outcome,
            } => {
                let succeeded = match outcome {
                    runmat_execution::WholeWorkObservationOutcome::Success => true,
                    runmat_execution::WholeWorkObservationOutcome::OperationalFailure => false,
                    _ => return None,
                };
                Some(runmat_execution::PlacementFeedback {
                    signature: selected.decision.signature.clone(),
                    candidate: selected.decision.selected.candidate.clone(),
                    kind: selected.decision.selected.kind,
                    total_elapsed_ns: *elapsed_ns,
                    succeeded,
                })
            }
        }
    }
}

/// A process-local reservation for one exploratory whole-work invocation.
/// Dropping it releases the reservation without reporting a provider failure.
pub trait RuntimeWholeWorkProbeLease: Send {}

/// Read-only completion authority returned by the selecting service. Only
/// that service can revoke its concrete permit on profile replacement or
/// typed invalidation; consumers cannot re-enable a retired observation.
pub trait RuntimeWholeWorkObservationPermit: Send + Sync {
    fn is_current(&self) -> bool;
}

/// Selection-owned graph observation authority. Its concrete owner binds the
/// exact signature and selected candidate before the caller begins effects.
/// The permit is not a placement decision or a provider execution lease.
pub trait RuntimeGraphObservationPermit: Send + Sync + std::fmt::Debug {
    fn authorizes(
        &self,
        signature: &runmat_execution::PlacementSignature,
        candidate: &str,
        kind: runmat_execution::ExecutionCandidateKind,
    ) -> bool;
}

pub struct RuntimeGraphPlacementSelection {
    pub decision: runmat_execution::PlacementDecision,
    pub observation_permit: Option<std::sync::Arc<dyn RuntimeGraphObservationPermit>>,
}

/// The finite managed route retains a value slot/generation lease instead of
/// allocating one dynamic observation permit per invocation.
pub enum RuntimeWholeWorkObservationAuthority {
    Dynamic(std::sync::Arc<dyn RuntimeWholeWorkObservationPermit>),
}

impl RuntimeWholeWorkObservationAuthority {
    pub fn is_current(&self) -> bool {
        match self {
            Self::Dynamic(permit) => permit.is_current(),
        }
    }
}

pub struct RuntimeWholeWorkSelection {
    pub decision: runmat_execution::WholeWorkDecision,
    pub probe: Option<Box<dyn RuntimeWholeWorkProbeLease>>,
    /// Selection-owned validity; a matching invalidation or profile import
    /// revokes late completion even before a deferred source is registered.
    pub observation_permit: Option<RuntimeWholeWorkObservationAuthority>,
}

/// Fixed-size complete-call decision retained by a managed finite route.
/// Portable presentation is reconstructed only at an explicit export boundary.
pub struct RuntimeManagedWholeWorkSelection {
    pub identity: runmat_execution::PlacementPermitIdentity,
    pub selected: runmat_execution::CompactSelectedExecutionCandidate,
    pub kind: runmat_execution::WholeWorkDecisionKind,
    pub predicted_total_ns: Option<u64>,
    pub observation_lease: Arc<super::ManagedObservationLease>,
    /// Retains the synthetic probe until the selection drops; no reads are required.
    #[cfg(test)]
    pub(crate) _test_probe: Option<Box<dyn RuntimeWholeWorkProbeLease>>,
    #[cfg(test)]
    pub(crate) test_signature: Box<runmat_execution::PlacementSignature>,
}

impl RuntimeManagedWholeWorkSelection {
    pub fn observation_is_current(&self) -> bool {
        self.observation_lease.is_current()
    }

    #[cfg(test)]
    pub(crate) fn test_observation(
        &self,
        elapsed_ns: u64,
        outcome: runmat_execution::WholeWorkObservationOutcome,
    ) -> runmat_execution::WholeWorkObservation {
        runmat_execution::WholeWorkObservation {
            signature: (*self.test_signature).clone(),
            selected: self.selected.portable(),
            elapsed_ns,
            outcome,
        }
    }
}

#[cfg(test)]
pub(crate) fn test_managed_selection(
    request: runmat_execution::WholeWorkPlacementRequestRef<'_>,
    test_probe: Option<Box<dyn RuntimeWholeWorkProbeLease>>,
    ledger: &super::host_effect::ManagedHostEffectLedger,
) -> Result<RuntimeManagedWholeWorkSelection, RuntimeError> {
    let candidate = &request.candidates[0];
    let identity = request
        .signature
        .permit_identity()
        .expect("typed test signature");
    let pool = super::ManagedObservationPool::synthetic_unaccounted_for_tests(ledger).map_err(
        |source| {
            crate::build_runtime_error("test managed observation authority is unavailable")
                .with_identifier("RunMat:Placement:StaleResourceAuthority")
                .with_source(source)
                .build()
        },
    )?;
    Ok(RuntimeManagedWholeWorkSelection {
        identity,
        selected: runmat_execution::CompactSelectedExecutionCandidate {
            node: runmat_execution::PlacementOperationId::new(request.signature.operation)
                .expect("typed test operation"),
            candidate: candidate.identity_digest(),
            kind: candidate.kind,
        },
        kind: runmat_execution::WholeWorkDecisionKind::Unmeasured,
        predicted_total_ns: None,
        observation_lease: pool
            .acquire(identity, candidate.identity_digest())
            .map_err(|source| {
                crate::build_runtime_error("test managed observation selection is unavailable")
                    .with_identifier("RunMat:Placement:StaleResourceAuthority")
                    .with_source(source)
                    .build()
            })?,
        _test_probe: test_probe,
        test_signature: Box::new(request.signature.to_owned()),
    })
}

impl RuntimeWholeWorkSelection {
    pub fn observation_is_current(&self) -> bool {
        self.observation_permit
            .as_ref()
            .is_none_or(|permit| permit.is_current())
    }
}

/// Host-owned, operation-specific evidence for a complete side-effecting call.
/// A provider capability digest and an archive size limit are not substitutes
/// for this installed service's revision and peak-memory evidence. An adapter
/// must return None unless it can certify the exact request's transport,
/// filesystem, codec, and result-storage bounds; a configured size cap alone
/// is not a peak-memory proof.
pub trait RuntimeHostEffectResourceService {
    fn admission(
        &self,
        request: &super::host_effect::HostEffectRequest<'_>,
        runtime: &super::RuntimeContext,
    ) -> Option<super::host_effect::HostEffectResourceAdmission>;

    /// Optional bounded pre-scan authority. This reserves a certified peak
    /// before opening a source whose exact result geometry is learned from
    /// its header; it does not create a second placement probe or feedback.
    fn stage_admission(
        &self,
        _request: &super::host_effect::HostEffectRequest<'_>,
        _runtime: &super::RuntimeContext,
    ) -> Option<super::host_effect::HostEffectStageAdmission> {
        None
    }

    /// The operation owner supplies checked, closed typed post-scan facts.
    /// The resource adapter matches the declared variant; shared placement
    /// verifies workload/facts identity but never calculates domain geometry.
    /// No fallback to pre-scan admission exists.
    fn final_admission_after_stage(
        &self,
        _request: &super::host_effect::HostEffectRequest<'_>,
        _facts: super::host_effect::HostEffectFinalFacts<'_>,
        _runtime: &super::RuntimeContext,
    ) -> Option<super::host_effect::HostEffectResourceAdmission> {
        None
    }
}

#[derive(Debug, thiserror::Error)]
enum UnsupportedWholeWorkSelection {
    #[error("ordinary whole-work selection is not implemented")]
    Ordinary,
    #[error("managed whole-work selection is not implemented")]
    Managed,
}

/// Session-owned execution-placement authority. The runtime exposes only
/// executor-neutral contracts; candidate generation and policy remain in their
/// owning executor/acceleration crates.
pub trait RuntimePlacementService {
    /// Select and register one graph decision's exact observation right on
    /// this service's owner thread. The default still plans/executes, but
    /// deliberately returns no permission to learn from a later completion.
    fn plan_with_observation_permit(
        &self,
        request: runmat_execution::PlacementPlanRequest,
    ) -> Result<RuntimeGraphPlacementSelection, RuntimeError> {
        self.plan(request)
            .map(|decision| RuntimeGraphPlacementSelection {
                decision,
                observation_permit: None,
            })
    }

    /// Only the selected graph observation right can publish feedback. A
    /// missing or revoked permit suppresses the sample, never execution.
    fn observe_selected_graph(
        &self,
        permit: &dyn RuntimeGraphObservationPermit,
        feedback: runmat_execution::PlacementFeedback,
    ) -> Result<(), RuntimeError> {
        if permit.authorizes(&feedback.signature, &feedback.candidate, feedback.kind) {
            self.observe(feedback)
        } else {
            Ok(())
        }
    }

    /// Owning placement policy supplies revisions for complete-route feedback.
    /// A third-party service that does not support this contract returns None
    /// rather than inventing catalog/compiler/policy identities in the caller.
    fn whole_work_revision(
        &self,
        _program: Option<runmat_execution::ProgramRevision>,
        _provider: runmat_execution::Digest,
    ) -> Option<runmat_execution::PlacementRevision> {
        None
    }

    /// Managed provider revision borrows the installed program facts. The
    /// portable owned hook above is not a finite-admission fallback.
    fn whole_work_revision_ref<'a>(
        &self,
        _program: Option<&'a runmat_execution::ProgramRevision>,
        _provider: runmat_execution::Digest,
    ) -> Option<runmat_execution::PlacementRevisionRef<'a>> {
        None
    }

    /// Nonpersistable revision for a selected compiled/runtime binding.
    /// The caller supplies a certified registration generation, not a name
    /// or function-pointer hash. The placement owner adds its session identity.
    fn whole_work_runtime_binding_session_revision(
        &self,
        _program: Option<runmat_execution::ProgramRevision>,
        _binding: runmat_builtins::BuiltinBindingIdentity,
        _registration: runmat_execution::Digest,
    ) -> Option<runmat_execution::PlacementRevision> {
        None
    }

    /// VM-host registration of an actually installed fusion region. The
    /// placement session supplies its own nonpersistable authority; it never
    /// accepts a serialized profile or free-form operation digest as proof.
    fn whole_work_vm_fusion_region_session_revision(
        &self,
        _program: Option<runmat_execution::ProgramRevision>,
        _region: super::VmFusionRegionRegistration,
        _liveness: std::sync::Weak<()>,
    ) -> Option<super::VmFusionSessionRegistration> {
        None
    }

    /// A complete host-service route has no provider capability digest.
    /// Services that cannot identify the installed host authority return None;
    /// callers with finite resource limits must then decline before effects.
    fn whole_work_host_service_revision(
        &self,
        _program: Option<runmat_execution::ProgramRevision>,
        _host_service: runmat_execution::Digest,
    ) -> Option<runmat_execution::PlacementRevision> {
        None
    }

    /// Exact borrowed managed revision. The portable owned hook above is not
    /// an implicit fallback: cloning its program facts would escape the
    /// staged invocation's allocation authority.
    fn whole_work_host_service_revision_ref<'a>(
        &self,
        _program: Option<&'a runmat_execution::ProgramRevision>,
        _host_service: runmat_execution::Digest,
    ) -> Option<runmat_execution::PlacementRevisionRef<'a>> {
        None
    }

    /// Select one complete, already admitted invocation. Third-party
    /// implementations remain conservative until they support this contract.
    fn select_whole_work(
        &self,
        _request: runmat_execution::WholeWorkPlacementRequest,
    ) -> Result<RuntimeWholeWorkSelection, RuntimeError> {
        Err(
            crate::build_runtime_error("whole-work placement is unavailable")
                .with_identifier("RunMat:Placement:Unavailable")
                .with_source(UnsupportedWholeWorkSelection::Ordinary)
                .build(),
        )
    }

    fn try_select_whole_work(
        &self,
        request: runmat_execution::WholeWorkPlacementRequest,
    ) -> Result<Option<RuntimeWholeWorkSelection>, RuntimeError> {
        match self.select_whole_work(request) {
            Ok(selected) => Ok(Some(selected)),
            Err(error)
                if std::error::Error::source(&error)
                    .and_then(|source| source.downcast_ref::<UnsupportedWholeWorkSelection>())
                    .is_some_and(|source| {
                        matches!(source, UnsupportedWholeWorkSelection::Ordinary)
                    }) =>
            {
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    /// Select already admitted routes borrowed from their owning invocation.
    /// A one-route caller can pass a stack array. Services lacking this
    /// contract fail closed; silently converting to the owned multi-route
    /// request would conceal allocations from finite host-effect admission.
    fn select_whole_work_borrowed(
        &self,
        _request: runmat_execution::WholeWorkPlacementRequestRef<'_>,
    ) -> Result<RuntimeWholeWorkSelection, RuntimeError> {
        Err(
            crate::build_runtime_error("borrowed whole-work placement is unavailable")
                .with_identifier("RunMat:Placement:Unavailable")
                .build(),
        )
    }

    /// Finite host-effect calls require a placement service with a retained,
    /// capacity-charged in-flight identity owner. An ordinary borrowed policy
    /// implementation must not masquerade as that managed authority.
    fn select_whole_work_borrowed_managed(
        &self,
        _request: runmat_execution::WholeWorkPlacementRequestRef<'_>,
    ) -> Result<RuntimeManagedWholeWorkSelection, RuntimeError> {
        Err(
            crate::build_runtime_error("managed whole-work placement is unavailable")
                .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
                .with_source(UnsupportedWholeWorkSelection::Managed)
                .build(),
        )
    }

    fn try_select_whole_work_borrowed_managed(
        &self,
        request: runmat_execution::WholeWorkPlacementRequestRef<'_>,
    ) -> Result<Option<RuntimeManagedWholeWorkSelection>, RuntimeError> {
        match self.select_whole_work_borrowed_managed(request) {
            Ok(selected) => Ok(Some(selected)),
            Err(error)
                if std::error::Error::source(&error)
                    .and_then(|source| source.downcast_ref::<UnsupportedWholeWorkSelection>())
                    .is_some_and(|source| {
                        matches!(source, UnsupportedWholeWorkSelection::Managed)
                    }) =>
            {
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    /// A managed selector may retain one finite in-flight observation lease.
    /// Without this authority callers can execute a resource-admitted route
    /// but must not publish placement learning feedback.
    fn managed_whole_work_available(&self) -> bool {
        false
    }

    fn observe_managed_whole_work(
        &self,
        _selected: &RuntimeManagedWholeWorkSelection,
        _elapsed_ns: u64,
        _outcome: runmat_execution::WholeWorkObservationOutcome,
    ) -> Result<(), RuntimeError> {
        Err(
            crate::build_runtime_error("managed whole-work feedback is unavailable")
                .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
                .build(),
        )
    }

    /// A cancelled or abandoned invocation is not a failure observation.
    fn observe_whole_work(
        &self,
        observation: runmat_execution::WholeWorkObservation,
    ) -> Result<(), RuntimeError> {
        use runmat_execution::WholeWorkObservationOutcome;
        let succeeded = match observation.outcome {
            WholeWorkObservationOutcome::Success => true,
            WholeWorkObservationOutcome::OperationalFailure => false,
            WholeWorkObservationOutcome::Cancelled | WholeWorkObservationOutcome::Abandoned => {
                return Ok(());
            }
        };
        self.observe(runmat_execution::PlacementFeedback {
            signature: observation.signature,
            candidate: observation.selected.candidate,
            kind: observation.selected.kind,
            total_elapsed_ns: observation.elapsed_ns,
            succeeded,
        })
    }

    /// Only the selection that owns an unrevoked observation permit may
    /// publish complete-call evidence. This is shared by host effects, host
    /// sinks, and other whole-work consumers rather than a builtin timer.
    /// Runtime placement ports are Rc-owned and PlacementSession is !Sync;
    /// permit validation and profile insertion are serialized on the owner
    /// thread. Provider callbacks only complete Send receipts for later
    /// owner-thread draining.
    fn observe_selected_whole_work(
        &self,
        selected: &RuntimeWholeWorkSelection,
        elapsed_ns: u64,
        outcome: runmat_execution::WholeWorkObservationOutcome,
    ) -> Result<(), RuntimeError> {
        if !selected.observation_is_current() {
            return Ok(());
        }
        self.observe_whole_work(runmat_execution::WholeWorkObservation {
            signature: selected.decision.signature.clone(),
            selected: selected.decision.selected.clone(),
            elapsed_ns,
            outcome,
        })
    }

    /// A duration measured for this exact signature and candidate. The
    /// default makes third-party services conservative: absence of a lookup
    /// does not authorize inventing a numeric route cost.
    fn observed_candidate_duration(
        &self,
        _signature: &runmat_execution::PlacementSignature,
        _candidate: &str,
        _kind: runmat_execution::ExecutionCandidateKind,
    ) -> runmat_execution::CandidateObservationEvidence {
        runmat_execution::CandidateObservationEvidence::Unavailable(
            runmat_execution::CandidateCostUnavailable::NoObservations,
        )
    }

    fn plan(
        &self,
        request: runmat_execution::PlacementPlanRequest,
    ) -> Result<runmat_execution::PlacementDecision, RuntimeError>;

    fn observe(&self, feedback: runmat_execution::PlacementFeedback) -> Result<(), RuntimeError>;

    fn observe_deferred_feedback(
        &self,
        delivery: RuntimePlacementFeedbackDelivery,
    ) -> Result<(), RuntimeError> {
        match delivery {
            RuntimePlacementFeedbackDelivery::Graph { feedback, permit } => {
                self.observe_selected_graph(permit.as_ref(), feedback)
            }
            RuntimePlacementFeedbackDelivery::WholeWork {
                selected,
                elapsed_ns,
                outcome,
            } => self.observe_selected_whole_work(&selected, elapsed_ns, outcome),
            RuntimePlacementFeedbackDelivery::ManagedWholeWork {
                selected,
                elapsed_ns,
                outcome,
            } => self.observe_managed_whole_work(&selected, elapsed_ns, outcome),
        }
    }

    /// Optional bounded retention. Rejection is dropped measurement, never an
    /// execution failure; returning the source lets the caller release it safely.
    fn defer_feedback(
        &self,
        source: Box<dyn RuntimePlacementFeedbackSource>,
    ) -> Result<(), Box<dyn RuntimePlacementFeedbackSource>> {
        Err(source)
    }

    /// Called at evaluation/placement/profile boundaries, never a device wait.
    fn drain_feedback(&self) {}

    fn invalidate(&self, invalidation: runmat_execution::PlacementInvalidation);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeCapability {
    SharedLibrary,
    ExecutableMemory,
    ObjectEmission,
}

pub trait RuntimeNativeService {
    fn supports(&self, capability: NativeCapability) -> bool;
}

#[derive(Debug)]
pub struct ForeignCall {
    pub adapter: String,
    pub symbol: String,
    pub arguments: crate::call::RootedFunctionArguments,
    pub requested_outputs: usize,
}

/// A selected foreign operation is terminal, while a pre-effect qualified
/// target miss returns the exact original call for language-source fallback.
pub struct ForeignCallCompletion(ForeignCallOutcome);

enum ForeignCallOutcome {
    Invoked(Result<crate::execution::RootedValueSequence, RuntimeError>),
    Declined(ForeignCall),
}

impl ForeignCallCompletion {
    pub fn invoked(result: Result<crate::execution::RootedValueSequence, RuntimeError>) -> Self {
        Self(ForeignCallOutcome::Invoked(result))
    }

    pub(crate) fn declined_before_marshalling(call: ForeignCall) -> Self {
        Self(ForeignCallOutcome::Declined(call))
    }

    pub(crate) fn into_owner_return(
        self,
    ) -> Result<Result<crate::execution::RootedValueSequence, RuntimeError>, ForeignCall> {
        match self.0 {
            ForeignCallOutcome::Invoked(result) => Ok(result),
            ForeignCallOutcome::Declined(call) => Err(call),
        }
    }

    pub(crate) fn invoked_result(
        &self,
    ) -> Option<&Result<crate::execution::RootedValueSequence, RuntimeError>> {
        match &self.0 {
            ForeignCallOutcome::Invoked(result) => Some(result),
            ForeignCallOutcome::Declined(_) => None,
        }
    }

    /// Direct foreign calls have no language-source fallback owner.
    pub fn into_invoked_result(
        self,
    ) -> Result<crate::execution::RootedValueSequence, RuntimeError> {
        match self.0 {
            ForeignCallOutcome::Invoked(result) => result,
            ForeignCallOutcome::Declined(_) => Err(crate::runtime_error::semantic_error(
                "RunMat:Foreign:TargetUnavailable",
                "foreign target was not selected for a direct call",
            )),
        }
    }
}

pub trait RuntimeForeignService {
    /// Complete adapter-owned Values, including retained callback captures.
    fn visit_retained_values(&self, _visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        None
    }
    fn execution_stack_requirement(&self) -> runmat_types::ExecutionStackRequirement {
        runmat_types::ExecutionStackRequirement::Any
    }

    fn invoke(
        &self,
        context: super::RuntimeContext,
        call: ForeignCall,
    ) -> RuntimeServiceFuture<ForeignCallCompletion>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelCapability {
    Pool,
    Parfor,
    Spmd,
    DistributedValues,
    Collectives,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeParallelResources {
    pub cpu_millicores_available: u32,
    pub memory_available_bytes: Option<u64>,
    pub epoch: u64,
}

impl Default for RuntimeParallelResources {
    fn default() -> Self {
        Self {
            cpu_millicores_available: 1_000,
            memory_available_bytes: None,
            epoch: 0,
        }
    }
}

pub trait RuntimeParallelService {
    fn supports(&self, capability: ParallelCapability) -> bool;

    /// Side-effect-free scheduler capacity for placement admission. A future
    /// RM-1067 pool/scheduler adapter overrides this with its current lease;
    /// absence preserves the single-core local runtime budget.
    fn placement_resources(&self) -> RuntimeParallelResources {
        RuntimeParallelResources::default()
    }

    /// Optional host-managed allocation owner for finite host-effect routes.
    /// This must be the *same* quota authority whose atomically observed free
    /// bytes and epoch are reported by `placement_resources`, not a fresh
    /// ledger constructed from that snapshot for each call. The default
    /// refuses finite host-effect allocation rather than inventing capacity.
    fn managed_host_effect_ledger(&self) -> Option<super::host_effect::ManagedHostEffectLedger> {
        None
    }
}

#[derive(Clone)]
pub struct RuntimeSpmdAdmission {
    pub gang: runmat_execution::GangSnapshot,
    pub labs: Vec<Rc<dyn RuntimeCollectiveService>>,
    /// The host's exact gang generation remains live through an unpolled
    /// admission future, a suspended VM invocation, and every cloned lab
    /// admission. Its last drop retires the gang synchronously.
    pub lifetime: Rc<dyn RuntimeSpmdGangLease>,
}

/// Host-owned lifecycle for exactly one admitted SPMD gang generation.
/// `retire` is idempotent; dropping the final owner performs the same local
/// retirement without polling an async future or scheduling detached work.
pub trait RuntimeSpmdGangLease {
    fn gang_handle(&self) -> &runmat_execution::GangHandle;
    fn retire(&self) -> Result<(), RuntimeError>;
}

#[derive(Debug, Clone)]
pub struct RuntimeSpmdOutput {
    pub value: runmat_types::RegionValueId,
    pub fact: runmat_types::ValueFact,
    pub entries: Vec<Option<runmat_execution::SpmdOutputValue>>,
}

#[derive(Debug, Clone)]
pub enum RuntimeSpmdRetainedOutput {
    Composite(runmat_execution::CompositeHandle),
    Distributed(runmat_execution::DistributedValueHandle),
}

impl RuntimeSpmdAdmission {
    pub fn validate(&self) -> Result<(), RuntimeError> {
        self.gang.validate().map_err(|error| {
            crate::runtime_error::semantic_error(
                "RunMat:parallel:InvalidGangAdmission",
                error.to_string(),
            )
        })?;
        if self.lifetime.gang_handle() != &self.gang.handle
            || self.labs.len() != self.gang.handle.labs.0 as usize
            || self
                .labs
                .iter()
                .zip(&self.gang.ranks)
                .any(|(service, rank)| {
                    service.context().gang != self.gang.handle || service.context().rank != *rank
                })
        {
            return Err(crate::runtime_error::semantic_error(
                "RunMat:parallel:InvalidGangAdmission",
                "SPMD admission must provide one ordered collective context per stable lab rank",
            ));
        }
        Ok(())
    }
}

/// Host-owned admission and lifecycle authority for one SPMD gang. Collective
/// services are scoped per lab while the coordinator remains shared.
pub trait RuntimeSpmdService {
    fn admit(
        &self,
        request: runmat_execution::GangRequest,
        available_labs: runmat_types::LabCount,
        region: runmat_types::ParallelRegionId,
    ) -> RuntimeServiceFuture<Result<RuntimeSpmdAdmission, RuntimeError>>;

    /// Records that one admitted lab completed its region body. The gang
    /// authority uses this typed lifecycle transition to distinguish a live
    /// collective wait from a wait whose remaining peers have already exited.
    fn rank_finished(
        &self,
        gang: &runmat_execution::GangHandle,
        rank: runmat_types::LabRank,
    ) -> Result<(), RuntimeError>;

    /// Records that one admitted lab failed while executing its region body.
    /// Implementations use this typed lifecycle event to release peers blocked
    /// in collectives; the caller retains and reports the originating error.
    fn rank_failed(
        &self,
        gang: &runmat_execution::GangHandle,
        rank: runmat_types::LabRank,
    ) -> Result<(), RuntimeError>;

    fn retain_outputs(
        &self,
        gang: runmat_execution::GangHandle,
        region: runmat_types::ParallelRegionId,
        outputs: Vec<RuntimeSpmdOutput>,
    ) -> RuntimeServiceFuture<Result<Vec<RuntimeSpmdRetainedOutput>, RuntimeError>>;
}

#[derive(Debug, Clone)]
pub enum RuntimeDistributedInvocation {
    Client,
    Worker(runmat_execution::SpmdTaskContext),
}

#[derive(Debug, Clone)]
pub struct RuntimeDistributedCallRequest {
    pub builtin: BuiltinId,
    pub arguments: crate::call::BuiltinArguments,
    pub literals: runmat_types::LiteralContext,
    pub requested_outputs: usize,
    pub outputs: Vec<runmat_types::ValueFact>,
    pub invocation: RuntimeDistributedInvocation,
}

/// Live distributed metadata visible to language/runtime consumers. Payloads
/// remain behind the execution-owned store; unlike a transport snapshot this
/// record never invents object references for locally retained partitions.
#[derive(Debug, Clone)]
pub struct RuntimeDistributedSnapshot {
    pub handle: runmat_value::SharedDistributedHandle,
    pub partitions: Vec<runmat_execution::DistributedPartitionLayout>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RuntimeDistributedRetirement {
    pub distributed: usize,
    pub composites: usize,
}

/// Runtime operations over execution-service-owned distributed values.
/// Implementations preserve partition ownership and decide locality; callers
/// never obtain raw partition storage through this port.
pub trait RuntimeDistributedService {
    /// Complete local Values retained after request ownership is transferred.
    fn visit_retained_values(&self, _visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        None
    }
    fn retire_pool(
        &self,
        pool: runmat_execution::PoolHandle,
    ) -> Result<RuntimeDistributedRetirement, RuntimeError>;

    fn create(
        &self,
        contract: DistributedValueContract,
        input: Value,
        scheme: DistributionScheme,
        pool: runmat_execution::PoolSnapshot,
    ) -> RuntimeServiceFuture<Result<runmat_execution::DistributedValueHandle, RuntimeError>>;

    fn create_worker(
        &self,
        contract: DistributedValueContract,
        input: Value,
        scheme: DistributionScheme,
        context: runmat_execution::SpmdTaskContext,
    ) -> RuntimeServiceFuture<Result<runmat_execution::DistributedValueHandle, RuntimeError>>;

    fn build_worker(
        &self,
        contract: DistributedValueContract,
        local_part: Value,
        global_shape: Vec<u64>,
        scheme: DistributionScheme,
        layouts: Vec<runmat_execution::DistributedPartitionLayout>,
        context: runmat_execution::SpmdTaskContext,
    ) -> RuntimeServiceFuture<Result<runmat_execution::DistributedValueHandle, RuntimeError>>;

    fn inspect(
        &self,
        handle: runmat_value::SharedDistributedHandle,
    ) -> RuntimeServiceFuture<Result<RuntimeDistributedSnapshot, RuntimeError>>;

    fn export_local(
        &self,
        handle: runmat_value::SharedDistributedHandle,
        rank: LabRank,
    ) -> RuntimeServiceFuture<Result<runmat_execution::DistributedShardSnapshot, RuntimeError>>;

    fn local_part(
        &self,
        handle: runmat_value::SharedDistributedHandle,
        rank: LabRank,
    ) -> RuntimeServiceFuture<Result<crate::execution::RootedValue, RuntimeError>>;

    fn materialize(
        &self,
        handle: runmat_value::SharedDistributedHandle,
    ) -> RuntimeServiceFuture<Result<crate::execution::RootedValue, RuntimeError>>;

    fn redistribute(
        &self,
        handle: runmat_value::SharedDistributedHandle,
        scheme: DistributionScheme,
    ) -> RuntimeServiceFuture<Result<runmat_execution::DistributedValueHandle, RuntimeError>>;

    fn invoke(
        &self,
        request: RuntimeDistributedCallRequest,
    ) -> RuntimeServiceFuture<Result<runmat_value::ValueSequence, RuntimeError>>;

    fn composite_entry(
        &self,
        handle: runmat_value::SharedCompositeHandle,
        rank: LabRank,
    ) -> RuntimeServiceFuture<Result<Option<crate::execution::RootedValue>, RuntimeError>>;
}

/// SPMD worker communication. The installed service owns the current rank and
/// transports typed collective payloads through the admitted gang.
pub trait RuntimeCollectiveService {
    fn context(&self) -> &runmat_execution::SpmdTaskContext;

    /// Allocate the next invocation sequence for one compiler-owned call site.
    /// Sequence state is scoped to the lab service, so repeated calls in loops
    /// cannot collide while every rank retains the same deterministic order.
    fn next_sequence(
        &self,
        id: runmat_types::CollectiveId,
    ) -> Result<runmat_execution::CollectiveSequence, RuntimeError>;

    fn execute(
        &self,
        request: runmat_execution::CollectiveRequest,
    ) -> RuntimeServiceFuture<Result<runmat_execution::CollectiveResponse, RuntimeError>>;
}

/// Immutable analyzed callable selected from the executable active in this invocation.
///
/// The product contains everything a portable execution adapter needs to prove callable identity,
/// validate static facts, and execute the exact frozen program without rediscovering source.
#[derive(Clone, Debug)]
pub struct RuntimeProgramCallable {
    pub owner_identity: String,
    pub display_name: String,
    pub source_digest: runmat_execution::Digest,
    pub executable_source_digest: runmat_execution::Digest,
    pub executable: runmat_execution::ExecutableUnitEnvelope,
}

#[derive(Clone, Debug)]
pub struct RuntimeSignalMaterializationRequest {
    pub signal: runmat_analysis_core::PhysicalSignal,
    pub callable: runmat_execution::value::CallableValue,
    pub program: RuntimeProgramCallable,
    pub maximum_wall_millis: u64,
}

/// Host-owned execution of an admitted generated physical input.
///
/// The runtime owns language values and the active program product. The host owns process,
/// accelerator, and distributed execution. The returned signal is canonical solver data and
/// contains no callback or runtime handle.
pub trait RuntimeAnalysisService {
    /// An owned mesher returns only canonical numerical mesh/evidence. The
    /// parent retains admission and persistence. None keeps the native route.
    fn generate_mesh(
        &self,
        _request: crate::meshing::MeshingExecutionRequest,
    ) -> Option<
        RuntimeServiceFuture<
            Result<crate::meshing::MeshingExecutionOutput, crate::meshing::MeshingExecutionError>,
        >,
    > {
        None
    }

    fn materialize_signal(
        &self,
        request: RuntimeSignalMaterializationRequest,
    ) -> RuntimeServiceFuture<Result<runmat_analysis_core::PhysicalSignal, crate::RuntimeError>>;

    /// Execute a validated study through the host's canonical placement adapter.
    ///
    /// The runtime retains admission and publication; the host supplies local or distributed
    /// process placement without making language builtins depend on a native runner crate.
    fn execute_study(
        &self,
        study: crate::analysis::AnalysisStudySpec,
        context: crate::operations::OperationContext,
    ) -> RuntimeServiceFuture<
        Result<
            crate::operations::OperationEnvelope<crate::analysis::AnalysisStudyRunData>,
            crate::operations::OperationErrorEnvelope,
        >,
    >;

    fn execute_sweep(
        &self,
        sweep: crate::analysis::AnalysisStudySweepSpec,
        context: crate::operations::OperationContext,
    ) -> RuntimeServiceFuture<
        Result<
            crate::operations::OperationEnvelope<crate::analysis::AnalysisStudySweepData>,
            crate::operations::OperationErrorEnvelope,
        >,
    >;

    /// Execute a typed `.fea` study document by path through the host's public study runner
    /// (`runmat run`), returning its authenticated publication. Hosts without a native
    /// study runner refuse explicitly.
    fn execute_document(
        &self,
        path: std::path::PathBuf,
    ) -> RuntimeServiceFuture<Result<crate::analysis::AnalysisStudyRunData, crate::RuntimeError>>
    {
        Box::pin(async move {
            Err(crate::build_runtime_error(format!(
                "this host cannot execute the FEA document `{}`; run it with a native RunMat build",
                path.display()
            ))
            .with_identifier("RunMat:fea:OperationFailed")
            .build())
        })
    }
}

/// Session-owned authority over the exact immutable executable active in an invocation.
pub trait RuntimeProgramService {
    /// Freeze this function's original product after validating its active caller.
    fn freeze_executable(
        &self,
        origin: runmat_execution::FunctionBindingOrigin,
        invocation_revision: &runmat_execution::ProgramRevision,
    ) -> Result<runmat_execution::ExecutableUnitEnvelope, crate::RuntimeError>;

    fn freeze_bound_callable(
        &self,
        origin: runmat_execution::FunctionBindingOrigin,
        display_name: &str,
    ) -> Result<RuntimeProgramCallable, crate::RuntimeError>;
}

/// Narrow, typed ports composed by the host. An absent port is meaningful and
/// produces a stable capability error through the corresponding `require_*`
/// accessor; there is no string-keyed service locator.
/// Invocation clones share the immutable composition. Caller-workspace and
/// removable foreign-host projections remain inline; changing either does not
/// allocate or copy the service table. Other reconfiguration makes a successor.
#[derive(Clone)]
pub struct RuntimeServicePorts {
    workspace: Option<Rc<dyn RuntimeWorkspaceService>>,
    // Keep the removable host/session edge outside the shared service table.
    // Captured callbacks and console scopes can break this cycle in O(1).
    foreign: Option<Rc<dyn RuntimeForeignService>>,
    shared: Rc<ServicePortSnapshot>,
}

#[path = "services/snapshot.rs"]
mod snapshot;
use snapshot::ServicePortSnapshot;

impl Default for RuntimeServicePorts {
    fn default() -> Self {
        Self::with_initial_timing(RuntimeTimingPort::Hosted(super::timing::platform_service()))
    }
}

impl RuntimeServicePorts {
    /// Selected deterministic construction never instantiates an ambient
    /// clock, even transiently before the admitted factory is installed.
    pub(crate) fn selected_unavailable() -> Self {
        Self::with_initial_timing(RuntimeTimingPort::SelectedUnavailable)
    }

    /// Create the ambient timing owner only after a verified HostedNative
    /// Clock selection. Other selected programs never instantiate it.
    pub(crate) fn selected_with_hosted_clock() -> Self {
        Self::with_initial_timing(RuntimeTimingPort::Hosted(super::timing::platform_service()))
    }

    fn with_initial_timing(timing: RuntimeTimingPort) -> Self {
        Self {
            workspace: None,
            foreign: None,
            shared: Rc::new(ServicePortSnapshot {
                call: None,
                builtin: None,
                base_workspace: None,
                object: None,
                host: None,
                graphics_presentation: None,
                graphics_callbacks: None,
                http: None,
                preferences: None,
                user_path: None,
                #[cfg(not(target_arch = "wasm32"))]
                process: None,
                tcp: None,
                error: None,
                acceleration: None,
                placement: None,
                host_effect_resources: None,
                native: None,
                parallel: None,
                spmd: None,
                distributed: None,
                collective: None,
                timing,
                local_tasks: None,
                source_evaluation: None,
                selected_source_evaluation: None,
                selected_local: None,
                selected_network: None,
                selected_ui: None,
                selected_callback: None,
                selected_dynamic: None,
                #[cfg(not(target_arch = "wasm32"))]
                blocking_cpu: Some(crate::execution::native_default_blocking_cpu_service()),
                #[cfg(target_arch = "wasm32")]
                blocking_cpu: None,
                owned_builtin_work: None,
                callable_artifacts: None,
                program: None,
                analysis: None,
                fea_run_admission: None,
                sync_fea_run_admission: None,
            }),
        }
    }
}

impl std::fmt::Debug for RuntimeServicePorts {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = formatter.debug_struct("RuntimeServicePorts");
        debug
            .field("call", &self.shared.call.is_some())
            .field("builtin", &self.shared.builtin.is_some())
            .field("workspace", &self.workspace.is_some())
            .field("base_workspace", &self.shared.base_workspace.is_some())
            .field("object", &self.shared.object.is_some())
            .field("host", &self.shared.host.is_some())
            .field(
                "graphics_presentation",
                &self.shared.graphics_presentation.is_some(),
            )
            .field(
                "graphics_callbacks",
                &self.shared.graphics_callbacks.is_some(),
            )
            .field("http", &self.shared.http.is_some())
            .field("preferences", &self.shared.preferences.is_some())
            .field("user_path", &self.shared.user_path.is_some())
            .field("tcp", &self.shared.tcp.is_some())
            .field("error", &self.shared.error.is_some())
            .field("acceleration", &self.shared.acceleration.is_some())
            .field("placement", &self.shared.placement.is_some())
            .field(
                "host_effect_resources",
                &self.shared.host_effect_resources.is_some(),
            )
            .field("native", &self.shared.native.is_some())
            .field("foreign", &self.foreign.is_some())
            .field("parallel", &self.shared.parallel.is_some())
            .field("spmd", &self.shared.spmd.is_some())
            .field("distributed", &self.shared.distributed.is_some())
            .field("collective", &self.shared.collective.is_some());
        debug.field("timing", &"installed");
        debug.field("local_tasks", &self.shared.local_tasks.is_some());
        debug.field(
            "source_evaluation",
            &self.shared.source_evaluation.is_some(),
        );
        debug.field(
            "selected_source_evaluation",
            &self.shared.selected_source_evaluation.is_some(),
        );
        debug.field("selected_local", &self.shared.selected_local.is_some());
        debug.field("selected_network", &self.shared.selected_network.is_some());
        debug.field("selected_ui", &self.shared.selected_ui.is_some());
        debug.field(
            "selected_callback",
            &self.shared.selected_callback.is_some(),
        );
        debug.field("selected_dynamic", &self.shared.selected_dynamic.is_some());
        debug.field("blocking_cpu", &self.shared.blocking_cpu.is_some());
        debug.field(
            "owned_builtin_work",
            &self.shared.owned_builtin_work.is_some(),
        );
        debug.field(
            "callable_artifacts",
            &self.shared.callable_artifacts.is_some(),
        );
        debug.field("program", &self.shared.program.is_some());
        debug.field("analysis", &self.shared.analysis.is_some());
        #[cfg(not(target_arch = "wasm32"))]
        debug.field("process", &self.shared.process.is_some());
        debug.finish()
    }
}

macro_rules! port_accessors {
    ($with:ident, $get:ident, $require:ident, $field:ident, $trait_name:ident, $cap:ident) => {
        pub fn $with(mut self, service: Rc<dyn $trait_name>) -> Self {
            Rc::make_mut(&mut self.shared).$field = Some(service);
            self
        }

        pub fn $get(&self) -> Option<&Rc<dyn $trait_name>> {
            self.shared.$field.as_ref()
        }

        pub fn $require(
            &self,
            operation: impl Into<String>,
        ) -> Result<&Rc<dyn $trait_name>, RuntimeCapabilityError> {
            self.shared
                .$field
                .as_ref()
                .ok_or_else(|| RuntimeCapabilityError::new(RuntimeCapability::$cap, operation))
        }
    };
}

impl RuntimeServicePorts {
    pub fn sync_fea_run_admission(
        &self,
    ) -> Option<&Arc<dyn crate::analysis::fea_run_admission::FeaRunAdmission>> {
        self.shared.sync_fea_run_admission.as_ref()
    }
    /// Session-owned native admission; installation does not acquire a lease.
    pub fn with_sync_fea_run_admission(
        mut self,
        admission: Arc<dyn crate::analysis::fea_run_admission::FeaRunAdmission>,
    ) -> Self {
        Rc::make_mut(&mut self.shared).sync_fea_run_admission = Some(admission);
        self
    }
    pub fn fea_run_admission(
        &self,
    ) -> Option<&Rc<dyn crate::analysis::fea_run_admission::AsyncFeaRunAdmission>> {
        self.shared.fea_run_admission.as_ref()
    }
    pub fn with_fea_run_admission(
        mut self,
        admission: Rc<dyn crate::analysis::fea_run_admission::AsyncFeaRunAdmission>,
    ) -> Self {
        Rc::make_mut(&mut self.shared).fea_run_admission = Some(admission);
        self
    }
    /// Omit the physical owned-CPU service. NeverYield embeddings execute
    /// supported kernels inline; cooperative native hosts require a CPU service.
    pub fn without_blocking_cpu(mut self) -> Self {
        Rc::make_mut(&mut self.shared).blocking_cpu = None;
        self
    }

    pub fn with_blocking_cpu(
        mut self,
        service: Rc<dyn crate::execution::RuntimeBlockingCpuService>,
    ) -> Self {
        Rc::make_mut(&mut self.shared).blocking_cpu = Some(service);
        self
    }

    pub fn blocking_cpu(&self) -> Option<&Rc<dyn crate::execution::RuntimeBlockingCpuService>> {
        self.shared.blocking_cpu.as_ref()
    }

    pub fn with_owned_builtin_work(
        mut self,
        service: Rc<dyn crate::execution::RuntimeOwnedBuiltinWorkService>,
    ) -> Self {
        Rc::make_mut(&mut self.shared).owned_builtin_work = Some(service);
        self
    }

    pub fn owned_builtin_work(
        &self,
    ) -> Option<&Rc<dyn crate::execution::RuntimeOwnedBuiltinWorkService>> {
        self.shared.owned_builtin_work.as_ref()
    }

    /// Only effect-routing ports affect the command's device-route proof.
    /// Workspace/base-workspace replacements during ordinary Core execution
    /// must not erase an otherwise exact canonical admission.
    pub fn same_device_route_owners(&self, other: &Self) -> bool {
        fn same<T: ?Sized>(left: Option<&Rc<T>>, right: Option<&Rc<T>>) -> bool {
            match (left, right) {
                (Some(left), Some(right)) => Rc::ptr_eq(left, right),
                (None, None) => true,
                _ => false,
            }
        }
        same(self.shared.call.as_ref(), other.shared.call.as_ref())
            && same(
                self.shared.acceleration.as_ref(),
                other.shared.acceleration.as_ref(),
            )
            && same(self.foreign.as_ref(), other.foreign.as_ref())
    }

    pub fn with_local_tasks(mut self, service: Rc<dyn super::RuntimeLocalTaskService>) -> Self {
        Rc::make_mut(&mut self.shared).local_tasks = Some(service);
        self
    }

    pub fn local_tasks(&self) -> Option<&Rc<dyn super::RuntimeLocalTaskService>> {
        self.shared.local_tasks.as_ref()
    }

    pub fn require_local_tasks(
        &self,
        operation: impl Into<String>,
    ) -> Result<&Rc<dyn super::RuntimeLocalTaskService>, RuntimeCapabilityError> {
        self.shared
            .local_tasks
            .as_ref()
            .ok_or_else(|| RuntimeCapabilityError::new(RuntimeCapability::LocalTasks, operation))
    }

    pub fn with_timing(mut self, service: Rc<dyn RuntimeTimingService>) -> Self {
        Rc::make_mut(&mut self.shared).timing = match self.shared.timing.clone() {
            RuntimeTimingPort::Hosted(_) => RuntimeTimingPort::Hosted(service),
            RuntimeTimingPort::SelectedUnavailable | RuntimeTimingPort::Selected { .. } => {
                RuntimeTimingPort::SelectedUnavailable
            }
        };
        self
    }

    /// Restrict a selected invocation before its exact Clock is installed.
    /// Generic timing injection cannot remove this restriction.
    pub fn without_timing(mut self) -> Self {
        Rc::make_mut(&mut self.shared).timing = RuntimeTimingPort::SelectedUnavailable;
        self
    }

    pub fn timing(&self) -> Result<&Rc<dyn RuntimeTimingService>, crate::RuntimeError> {
        match &self.shared.timing {
            RuntimeTimingPort::Hosted(service) | RuntimeTimingPort::Selected { service, .. } => {
                Ok(service)
            }
            RuntimeTimingPort::SelectedUnavailable => Err(crate::build_runtime_error(
                "Clock is not admitted for this selected invocation",
            )
            .with_identifier("RunMat:Clock:ServiceUnavailable")
            .build()),
        }
    }
    pub fn with_call(mut self, service: Rc<dyn RuntimeCallService>) -> Self {
        Rc::make_mut(&mut self.shared).call = Some(service);
        self
    }

    pub fn call(&self) -> Option<&Rc<dyn RuntimeCallService>> {
        self.shared
            .selected_callback
            .as_ref()
            .map(|selected| &selected.call)
            .or(self.shared.call.as_ref())
    }

    pub fn require_call(
        &self,
        operation: impl Into<String>,
    ) -> Result<&Rc<dyn RuntimeCallService>, RuntimeCapabilityError> {
        self.call()
            .ok_or_else(|| RuntimeCapabilityError::new(RuntimeCapability::Call, operation))
    }
    pub fn with_builtin(mut self, service: Rc<dyn RuntimeBuiltinService>) -> Self {
        Rc::make_mut(&mut self.shared).builtin = Some(service);
        self
    }

    pub fn builtin(&self) -> Option<&Rc<dyn RuntimeBuiltinService>> {
        self.shared
            .selected_dynamic
            .as_ref()
            .map(|selected| &selected.service)
            .or(self.shared.builtin.as_ref())
    }

    pub fn require_builtin(
        &self,
        operation: impl Into<String>,
    ) -> Result<&Rc<dyn RuntimeBuiltinService>, RuntimeCapabilityError> {
        self.builtin()
            .ok_or_else(|| RuntimeCapabilityError::new(RuntimeCapability::Builtin, operation))
    }
    /// Caller-frame projection; changing it never copies the immutable ports.
    pub fn with_workspace(mut self, service: Rc<dyn RuntimeWorkspaceService>) -> Self {
        self.workspace = Some(service);
        self
    }
    pub fn workspace(&self) -> Option<&Rc<dyn RuntimeWorkspaceService>> {
        self.workspace.as_ref()
    }
    pub fn require_workspace(
        &self,
        operation: impl Into<String>,
    ) -> Result<&Rc<dyn RuntimeWorkspaceService>, RuntimeCapabilityError> {
        self.workspace
            .as_ref()
            .ok_or_else(|| RuntimeCapabilityError::new(RuntimeCapability::Workspace, operation))
    }
    port_accessors!(
        with_base_workspace,
        base_workspace,
        require_base_workspace,
        base_workspace,
        RuntimeBaseWorkspaceService,
        BaseWorkspace
    );
    pub fn with_source_evaluation(
        mut self,
        service: Rc<dyn super::RuntimeSourceEvaluationService>,
    ) -> Self {
        Rc::make_mut(&mut self.shared).source_evaluation = Some(service);
        Rc::make_mut(&mut self.shared).selected_source_evaluation = None;
        self
    }

    pub fn with_selected_source_evaluation(
        mut self,
        service: Rc<dyn super::RuntimeSourceEvaluationService>,
        issuer: runmat_execution::host::VerifiedNativeServiceIssuer,
    ) -> Result<Self, crate::RuntimeError> {
        if issuer.binding().kind
            != runmat_execution::host::NativeDeploymentServiceKind::SourceEvaluation
        {
            return Err(crate::build_runtime_error(
                "selected source-evaluation port has a different service identity",
            )
            .with_identifier("RunMat:SourceEvaluation:SelectedServiceMismatch")
            .build());
        }
        Rc::make_mut(&mut self.shared).source_evaluation = Some(service);
        Rc::make_mut(&mut self.shared).selected_source_evaluation = Some(issuer);
        Ok(self)
    }

    pub fn source_evaluation(&self) -> Option<&Rc<dyn super::RuntimeSourceEvaluationService>> {
        self.shared.source_evaluation.as_ref()
    }

    pub fn require_source_evaluation(
        &self,
        operation: impl Into<String>,
    ) -> Result<&Rc<dyn super::RuntimeSourceEvaluationService>, super::RuntimeCapabilityError> {
        self.shared.source_evaluation.as_ref().ok_or_else(|| {
            super::RuntimeCapabilityError::new(
                super::RuntimeCapability::SourceEvaluation,
                operation,
            )
        })
    }

    pub fn require_selected_source_evaluation(
        &self,
        expected: &runmat_execution::host::VerifiedNativeServiceIssuer,
    ) -> Result<&Rc<dyn super::RuntimeSourceEvaluationService>, crate::RuntimeError> {
        let service = self
            .require_source_evaluation("execute selected native source evaluation")
            .map_err(super::RuntimeCapabilityError::into_runtime_error)?;
        if self.shared.selected_source_evaluation.as_ref() != Some(expected) {
            return Err(crate::build_runtime_error(
                "live source-evaluation port differs from the verified selected service issuer",
            )
            .with_identifier("RunMat:SourceEvaluation:SelectedServiceMismatch")
            .build());
        }
        Ok(service)
    }
    port_accessors!(
        with_callable_artifacts,
        callable_artifacts,
        require_callable_artifacts,
        callable_artifacts,
        RuntimeCallableArtifactService,
        CallableArtifacts
    );

    /// Remove an inherited caller-workspace service before entering an
    /// isolated procedure frame. The callee may then install its own scoped
    /// workspace authority without exposing or mutating the caller frame.
    pub fn without_workspace(mut self) -> Self {
        self.workspace = None;
        self
    }

    /// Remove session base-workspace authority when constructing an isolated
    /// runtime such as a parallel lab. Ordinary procedure frames use
    /// [`Self::without_workspace`] and retain this separate session authority.
    pub fn without_base_workspace(mut self) -> Self {
        Rc::make_mut(&mut self.shared).base_workspace = None;
        self
    }

    /// Parallel labs cannot borrow their parent's window, surface, or event
    /// queue authority merely because their service ports were cloned.
    pub fn without_graphics_presentation(mut self) -> Self {
        Rc::make_mut(&mut self.shared).graphics_presentation = None;
        Rc::make_mut(&mut self.shared).graphics_callbacks = None;
        if let Some(selected) = &mut Rc::make_mut(&mut self.shared).selected_ui {
            selected.graphics = None;
            selected.callbacks = None;
        }
        self
    }

    /// Parallel labs do not inherit the parent session's definition
    /// publication authority. A worker must install its own artifact owner.
    pub fn without_callable_artifacts(mut self) -> Self {
        Rc::make_mut(&mut self.shared).callable_artifacts = None;
        self
    }
    /// Session-local callback registration authority. The port is weak so a
    /// selected callback context cannot keep its owning queue alive in a
    /// context → presentation → queue cycle.
    pub fn with_graphics_callback_queue(
        mut self,
        queue: super::WeakRuntimeGraphicsCallbackQueue,
    ) -> Self {
        Rc::make_mut(&mut self.shared).graphics_callbacks = Some(queue);
        self
    }

    pub fn graphics_callback_queue(&self) -> Option<&super::WeakRuntimeGraphicsCallbackQueue> {
        if let Some(selected) = &self.shared.selected_ui {
            selected.callbacks.as_ref()
        } else {
            self.shared.graphics_callbacks.as_ref()
        }
    }
    port_accessors!(
        with_object,
        object,
        require_object,
        object,
        RuntimeObjectService,
        Object
    );
    pub fn with_host(mut self, service: Rc<dyn RuntimeHostService>) -> Self {
        Rc::make_mut(&mut self.shared).host = Some(service);
        self
    }

    pub fn host(&self) -> Option<&Rc<dyn RuntimeHostService>> {
        self.shared
            .selected_ui
            .as_ref()
            .map(|selected| &selected.host)
            .or(self.shared.host.as_ref())
    }

    pub fn require_host(
        &self,
        operation: impl Into<String>,
    ) -> Result<&Rc<dyn RuntimeHostService>, RuntimeCapabilityError> {
        self.host()
            .ok_or_else(|| RuntimeCapabilityError::new(RuntimeCapability::Host, operation))
    }

    pub fn with_graphics_presentation(
        mut self,
        service: Rc<dyn RuntimeGraphicsPresentationService>,
    ) -> Self {
        Rc::make_mut(&mut self.shared).graphics_presentation = Some(service);
        self
    }

    pub fn graphics_presentation(&self) -> Option<&Rc<dyn RuntimeGraphicsPresentationService>> {
        if let Some(selected) = &self.shared.selected_ui {
            selected.graphics.as_ref()
        } else {
            self.shared.graphics_presentation.as_ref()
        }
    }

    pub fn require_graphics_presentation(
        &self,
        operation: impl Into<String>,
    ) -> Result<&Rc<dyn RuntimeGraphicsPresentationService>, RuntimeCapabilityError> {
        self.graphics_presentation().ok_or_else(|| {
            RuntimeCapabilityError::new(RuntimeCapability::GraphicsPresentation, operation)
        })
    }
    pub fn with_http(mut self, service: Rc<dyn RuntimeHttpService>) -> Self {
        Rc::make_mut(&mut self.shared).http = Some(service);
        self
    }

    pub fn http(&self) -> Option<&Rc<dyn RuntimeHttpService>> {
        self.shared.http.as_ref()
    }

    pub fn require_http(
        &self,
        operation: impl Into<String>,
    ) -> Result<&Rc<dyn RuntimeHttpService>, RuntimeCapabilityError> {
        let operation = operation.into();
        if !self.selected_http_matches() {
            return Err(RuntimeCapabilityError::new(
                RuntimeCapability::Http,
                operation,
            ));
        }
        self.shared
            .http
            .as_ref()
            .ok_or_else(|| RuntimeCapabilityError::new(RuntimeCapability::Http, operation))
    }
    #[cfg(not(target_arch = "wasm32"))]
    port_accessors!(
        with_process,
        process,
        require_process,
        process,
        RuntimeProcessService,
        Process
    );
    port_accessors!(
        with_preferences,
        preferences,
        require_preferences,
        preferences,
        RuntimePreferenceService,
        Preferences
    );
    port_accessors!(
        with_user_path,
        user_path,
        require_user_path,
        user_path,
        RuntimeUserPathService,
        UserPath
    );
    pub fn with_tcp(mut self, service: Rc<dyn RuntimeTcpService>) -> Self {
        Rc::make_mut(&mut self.shared).tcp = Some(service);
        self
    }

    pub fn tcp(&self) -> Option<&Rc<dyn RuntimeTcpService>> {
        self.shared.tcp.as_ref()
    }

    pub fn require_tcp(
        &self,
        operation: impl Into<String>,
    ) -> Result<&Rc<dyn RuntimeTcpService>, RuntimeCapabilityError> {
        let operation = operation.into();
        if !self.selected_tcp_matches() {
            return Err(RuntimeCapabilityError::new(
                RuntimeCapability::Tcp,
                operation,
            ));
        }
        self.shared
            .tcp
            .as_ref()
            .ok_or_else(|| RuntimeCapabilityError::new(RuntimeCapability::Tcp, operation))
    }
    port_accessors!(
        with_error,
        error,
        require_error,
        error,
        RuntimeErrorService,
        Error
    );
    port_accessors!(
        with_acceleration,
        acceleration,
        require_acceleration,
        acceleration,
        RuntimeAccelerationService,
        Acceleration
    );
    port_accessors!(
        with_placement,
        placement,
        require_placement,
        placement,
        RuntimePlacementService,
        Placement
    );
    port_accessors!(
        with_host_effect_resources,
        host_effect_resources,
        require_host_effect_resources,
        host_effect_resources,
        RuntimeHostEffectResourceService,
        Placement
    );
    port_accessors!(
        with_native,
        native,
        require_native,
        native,
        RuntimeNativeService,
        Native
    );
    pub fn with_foreign(mut self, service: Rc<dyn RuntimeForeignService>) -> Self {
        self.foreign = Some(service);
        self
    }

    pub fn foreign(&self) -> Option<&Rc<dyn RuntimeForeignService>> {
        self.foreign.as_ref()
    }

    pub fn require_foreign(
        &self,
        operation: impl Into<String>,
    ) -> Result<&Rc<dyn RuntimeForeignService>, RuntimeCapabilityError> {
        self.foreign
            .as_ref()
            .ok_or_else(|| RuntimeCapabilityError::new(RuntimeCapability::Foreign, operation))
    }

    /// Remove the strong foreign-runtime edge while retaining every other
    /// session authority. A long-lived foreign callback can retain this
    /// context and restore the foreign port from a weak reference while it
    /// executes without creating a host/session ownership cycle.
    pub fn without_foreign(mut self) -> Self {
        self.foreign = None;
        self
    }
    port_accessors!(
        with_parallel,
        parallel,
        require_parallel,
        parallel,
        RuntimeParallelService,
        Parallel
    );
    port_accessors!(
        with_spmd,
        spmd,
        require_spmd,
        spmd,
        RuntimeSpmdService,
        Spmd
    );
    port_accessors!(
        with_distributed,
        distributed,
        require_distributed,
        distributed,
        RuntimeDistributedService,
        Distributed
    );
    port_accessors!(
        with_collective,
        collective,
        require_collective,
        collective,
        RuntimeCollectiveService,
        Collective
    );
    port_accessors!(
        with_program,
        program,
        require_program,
        program,
        RuntimeProgramService,
        Program
    );
    port_accessors!(
        with_analysis,
        analysis,
        require_analysis,
        analysis,
        RuntimeAnalysisService,
        Analysis
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cloned_ports_share_composition_without_sharing_workspace_projection() {
        let original = RuntimeServicePorts::default();
        let workspace: Rc<dyn RuntimeWorkspaceService> = Rc::new(ValueOnlyWorkspace::default());
        let caller = original.clone().with_workspace(Rc::clone(&workspace));
        assert!(Rc::ptr_eq(&original.shared, &caller.shared));
        assert!(original.workspace().is_none());
        assert!(Rc::ptr_eq(caller.workspace().unwrap(), &workspace));

        let cleared = caller.clone().without_workspace();
        assert!(Rc::ptr_eq(&caller.shared, &cleared.shared));
        assert!(cleared.workspace().is_none());
        assert!(caller.workspace().is_some());

        let foreign: Rc<dyn RuntimeForeignService> = Rc::new(crate::foreign::ForeignRuntime::new(
            crate::foreign::ForeignPlatform::Native,
        ));
        let foreign_ports = caller.clone().with_foreign(Rc::clone(&foreign));
        let callback_ports = foreign_ports.clone().without_foreign();
        assert!(Rc::ptr_eq(&caller.shared, &foreign_ports.shared));
        assert!(Rc::ptr_eq(&foreign_ports.shared, &callback_ports.shared));
        assert!(Rc::ptr_eq(foreign_ports.foreign().unwrap(), &foreign));
        assert!(callback_ports.foreign().is_none());
        assert!(!foreign_ports.same_device_route_owners(&callback_ports));

        let successor = caller.clone().without_timing();
        assert!(!Rc::ptr_eq(&caller.shared, &successor.shared));
        assert!(successor.timing().is_err());
        assert!(caller.timing().is_ok());
        assert!(Rc::ptr_eq(successor.workspace().unwrap(), &workspace));
        assert_eq!(
            std::mem::size_of::<RuntimeServicePorts>(),
            std::mem::size_of::<Rc<ServicePortSnapshot>>()
                + std::mem::size_of::<Option<Rc<dyn RuntimeWorkspaceService>>>()
                + std::mem::size_of::<Option<Rc<dyn RuntimeForeignService>>>(),
        );
    }

    #[test]
    fn optional_whole_work_defaults_are_typed_and_do_not_hide_spoofed_errors() {
        use runmat_execution::*;
        struct DefaultSelector;
        struct SpoofSelector;
        macro_rules! graph_only {
            () => {
                fn plan(&self, _: PlacementPlanRequest) -> Result<PlacementDecision, RuntimeError> {
                    panic!("not part of optional whole-work selection")
                }
                fn observe(&self, _: PlacementFeedback) -> Result<(), RuntimeError> {
                    Ok(())
                }
                fn invalidate(&self, _: PlacementInvalidation) {}
            };
        }
        impl RuntimePlacementService for DefaultSelector {
            graph_only!();
        }
        impl RuntimePlacementService for SpoofSelector {
            graph_only!();
            fn select_whole_work(
                &self,
                _: WholeWorkPlacementRequest,
            ) -> Result<RuntimeWholeWorkSelection, RuntimeError> {
                Err(crate::build_runtime_error("custom ordinary failure")
                    .with_identifier("RunMat:Placement:Unavailable")
                    .build())
            }
            fn select_whole_work_borrowed_managed(
                &self,
                _: WholeWorkPlacementRequestRef<'_>,
            ) -> Result<RuntimeManagedWholeWorkSelection, RuntimeError> {
                Err(crate::build_runtime_error("custom managed failure")
                    .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
                    .build())
            }
        }
        let digest = Digest::sha256(b"optional-whole-work-default-fixture");
        let request = WholeWorkPlacementRequest {
            signature: PlacementSignature {
                region: None,
                operation: "optional.default.fixture".into(),
                runtime_facts: digest,
                provider_provenance: None,
                graph_output_authority: None,
                revision: PlacementRevision {
                    program: None,
                    catalog: digest,
                    compiler: digest,
                    authority: PlacementExecutionAuthorityRevision::HostService(digest),
                    policy: digest,
                },
            },
            constraint: WholeWorkConstraint::AnyLegal,
            candidates: Vec::new(),
            resources: PlacementResourceSnapshot::default(),
            deterministic: true,
            require_transactional_results: true,
        };
        assert!(DefaultSelector
            .try_select_whole_work(request.clone())
            .unwrap()
            .is_none());
        assert!(DefaultSelector
            .try_select_whole_work_borrowed_managed(request.as_ref())
            .unwrap()
            .is_none());
        assert!(
            matches!(SpoofSelector.try_select_whole_work(request.clone()), Err(error) if error.message == "custom ordinary failure")
        );
        assert!(
            matches!(SpoofSelector.try_select_whole_work_borrowed_managed(request.as_ref()), Err(error) if error.message == "custom managed failure")
        );
    }

    #[derive(Default)]
    struct ValueOnlyWorkspace {
        assignments: std::cell::Cell<usize>,
    }

    impl RuntimeWorkspaceService for ValueOnlyWorkspace {
        fn lookup(&self, _name: &str) -> Option<Value> {
            None
        }

        fn snapshot(&self) -> Vec<(String, Value)> {
            Vec::new()
        }

        fn global_names(&self) -> Vec<String> {
            Vec::new()
        }

        fn assign(&self, _name: &str, _value: Value) -> Result<(), RuntimeError> {
            self.assignments.set(self.assignments.get() + 1);
            Ok(())
        }

        fn clear(&self) -> Result<(), RuntimeError> {
            Ok(())
        }

        fn remove(&self, _name: &str) -> Result<(), RuntimeError> {
            Ok(())
        }
    }

    #[test]
    fn value_only_workspace_cannot_claim_global_namespace_authority() {
        let workspace = ValueOnlyWorkspace::default();
        let context = RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let failures = [
            workspace.base_global_binding_names().unwrap_err(),
            workspace.is_base_global_binding("g").unwrap_err(),
            workspace.bind_base_global("g").unwrap_err(),
            workspace.unbind_base_global("g").unwrap_err(),
            workspace
                .assign_base_local("g", Value::Num(1.0))
                .unwrap_err(),
            workspace.declare_global_binding(0, "g").unwrap_err(),
            workspace
                .bind_base_global_with_context(&context, "g")
                .unwrap_err(),
            workspace
                .unbind_base_global_with_context(&context, "g")
                .unwrap_err(),
            workspace
                .assign_base_local_with_context(&context, "g", Value::Num(1.0))
                .unwrap_err(),
            workspace
                .declare_global_binding_with_context(&context, 0, "g")
                .unwrap_err(),
            workspace
                .declare_legacy_global_binding_with_context(&context, 0, "g")
                .unwrap_err(),
        ];

        assert!(failures.iter().all(|error| {
            error.identifier() == Some(RuntimeCapabilityError::IDENTIFIER)
                && error
                    .message()
                    .contains(RuntimeCapability::BaseWorkspace.as_str())
        }));
        assert_eq!(workspace.assignments.get(), 0);
    }

    #[test]
    fn third_party_placement_service_has_no_invented_observation() {
        struct ExternalPlacement;
        impl RuntimePlacementService for ExternalPlacement {
            fn plan(
                &self,
                _: runmat_execution::PlacementPlanRequest,
            ) -> Result<runmat_execution::PlacementDecision, RuntimeError> {
                unreachable!("observation lookup does not plan")
            }
            fn observe(&self, _: runmat_execution::PlacementFeedback) -> Result<(), RuntimeError> {
                unreachable!("observation lookup does not submit feedback")
            }
            fn invalidate(&self, _: runmat_execution::PlacementInvalidation) {}
        }
        let digest = runmat_execution::Digest::sha256(b"revision");
        let signature = runmat_execution::PlacementSignature {
            region: None,
            operation: "route".into(),
            runtime_facts: digest,
            provider_provenance: None,
            graph_output_authority: None,
            revision: runmat_execution::PlacementRevision {
                program: None,
                catalog: digest,
                compiler: digest,
                authority: runmat_execution::PlacementExecutionAuthorityRevision::Provider(digest),
                policy: digest,
            },
        };
        assert_eq!(
            ExternalPlacement.observed_candidate_duration(
                &signature,
                "provider.route",
                runmat_execution::ExecutionCandidateKind::ProviderGraph,
            ),
            runmat_execution::CandidateObservationEvidence::Unavailable(
                runmat_execution::CandidateCostUnavailable::NoObservations,
            )
        );
    }

    #[test]
    fn absent_ports_report_stable_typed_capabilities() {
        fn missing<T>(result: Result<T, RuntimeCapabilityError>) -> RuntimeCapabilityError {
            match result {
                Ok(_) => panic!("expected absent runtime port"),
                Err(error) => error,
            }
        }
        let ports = RuntimeServicePorts::default();
        let failures = [
            missing(ports.require_builtin("resolve builtin")),
            missing(ports.require_call("invoke")),
            missing(ports.require_workspace("lookup")),
            missing(ports.require_base_workspace("base workspace projection")),
            missing(ports.require_object("class lookup")),
            missing(ports.require_host("console write")),
            missing(ports.require_http("HTTP request")),
            missing(ports.require_preferences("read preference")),
            missing(ports.require_user_path("hydrate user path")),
            missing(ports.require_error("report error")),
            missing(ports.require_acceleration("resident operation")),
            missing(ports.require_placement("placement plan")),
            missing(ports.require_native("load library")),
            missing(ports.require_foreign("foreign call")),
            missing(ports.require_parallel("parfor")),
            missing(ports.require_distributed("distributed array")),
            missing(ports.require_collective("SPMD barrier")),
            missing(ports.require_local_tasks("schedule callback")),
            missing(ports.require_source_evaluation("evaluate session source")),
            missing(ports.require_graphics_presentation("flush graphics")),
            missing(ports.require_program("freeze callable")),
            missing(ports.require_analysis("materialize signal")),
        ];
        assert_eq!(
            failures
                .iter()
                .map(|failure| failure.capability)
                .collect::<Vec<_>>(),
            vec![
                RuntimeCapability::Builtin,
                RuntimeCapability::Call,
                RuntimeCapability::Workspace,
                RuntimeCapability::BaseWorkspace,
                RuntimeCapability::Object,
                RuntimeCapability::Host,
                RuntimeCapability::Http,
                RuntimeCapability::Preferences,
                RuntimeCapability::UserPath,
                RuntimeCapability::Error,
                RuntimeCapability::Acceleration,
                RuntimeCapability::Placement,
                RuntimeCapability::Native,
                RuntimeCapability::Foreign,
                RuntimeCapability::Parallel,
                RuntimeCapability::Distributed,
                RuntimeCapability::Collective,
                RuntimeCapability::LocalTasks,
                RuntimeCapability::SourceEvaluation,
                RuntimeCapability::GraphicsPresentation,
                RuntimeCapability::Program,
                RuntimeCapability::Analysis,
            ]
        );
        assert!(failures.iter().all(|failure| {
            failure.clone().into_runtime_error().identifier()
                == Some(RuntimeCapabilityError::IDENTIFIER)
        }));
    }

    #[test]
    fn parallel_lab_does_not_inherit_cloned_parent_graphics_authority() {
        struct ParentGraphics;
        impl RuntimeGraphicsPresentationService for ParentGraphics {
            fn flush(
                &self,
                _: super::super::GraphicsFlushRequest,
            ) -> std::pin::Pin<Box<dyn super::super::RuntimeGraphicsFlushOperation>> {
                panic!("a cloned lab must never submit the parent's graphics request")
            }
        }

        let parent = super::super::RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_graphics_presentation(std::rc::Rc::new(ParentGraphics)),
        );
        assert!(parent.service_ports().graphics_presentation().is_some());
        let lab = parent
            .fork_parallel_lab(parent.service_ports().clone())
            .expect("idle session fork");
        assert!(lab.service_ports().graphics_presentation().is_none());
        assert_eq!(
            lab.service_ports()
                .require_graphics_presentation("drawnow")
                .err()
                .map(|error| error.capability),
            Some(RuntimeCapability::GraphicsPresentation)
        );
        assert!(parent.service_ports().graphics_presentation().is_some());
    }
}
