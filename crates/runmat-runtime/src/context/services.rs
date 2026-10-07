use super::{
    RuntimeCallableArtifactService, RuntimeCapability, RuntimeCapabilityError,
    RuntimeGraphicsPresentationService, RuntimePreferenceService, RuntimeSourceEvaluationService,
    RuntimeTcpService, RuntimeTimingService, RuntimeUserPathService,
};
use crate::builtin::RuntimeBuiltinBinding;
use crate::class_registry::RuntimeClass;
use crate::warning_store::RuntimeWarning;
use crate::RuntimeError;
use runmat_types::{
    BuiltinId, CallableFallbackPolicy, CallableIdentity, DistributedValueContract,
    DistributionScheme, LabRank, SourceId,
};
use runmat_value::{Value, ValueSequence};
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;

pub type RuntimeServiceFuture<T> = Pin<Box<dyn Future<Output = T> + 'static>>;

#[derive(Debug, Clone)]
pub struct RuntimeCallRequest {
    pub identity: CallableIdentity,
    pub binding: Option<runmat_value::BoundFunctionBinding>,
    pub arguments: Vec<Value>,
    pub requested_outputs: usize,
}

pub trait RuntimeCallService {
    /// Complete implementation-owned Values; unavailable is not an empty set.
    fn visit_retained_values(&self, _visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        None
    }
    fn resolve(&self, name: &str) -> Option<usize>;

    fn invoke(
        &self,
        runtime: super::RuntimeContext,
        request: RuntimeCallRequest,
    ) -> RuntimeServiceFuture<Result<ValueSequence, RuntimeError>>;

    fn source_functions(&self, _source_id: SourceId) -> Vec<(String, usize)> {
        Vec::new()
    }
}

/// Canonical call router for callbacks that re-enter the active RunMat
/// executor. The executor installs its semantic and external invokers in the
/// invocation context; this service keeps foreign adapters independent of VM,
/// JIT, and AOT implementation details.
#[derive(Debug, Default)]
pub struct RuntimeCallRouter;

impl RuntimeCallService for RuntimeCallRouter {
    fn visit_retained_values(&self, _visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        // This router has no state; pending requests belong to caller futures.
        Some(())
    }
    fn resolve(&self, name: &str) -> Option<usize> {
        crate::user_functions::resolve_semantic_function_by_name(name)
    }

    fn invoke(
        &self,
        runtime: super::RuntimeContext,
        request: RuntimeCallRequest,
    ) -> RuntimeServiceFuture<Result<ValueSequence, RuntimeError>> {
        let fallback_policy = match &request.identity {
            CallableIdentity::DynamicName(_)
            | CallableIdentity::Imported(_)
            | CallableIdentity::Method(_) => CallableFallbackPolicy::RuntimeNameResolution,
            CallableIdentity::ExternalName(_) => CallableFallbackPolicy::ExternalBoundary,
            _ => CallableFallbackPolicy::None,
        };
        Box::pin(async move {
            if let Some(binding) = request.binding {
                return crate::call::descriptor::execute_callable_descriptor_in_context(
                    &runtime,
                    crate::call::descriptor::CallableDescriptor::bound(
                        binding,
                        request.arguments,
                        request.requested_outputs,
                        crate::call::descriptor::CallableCallKind::Direct,
                    ),
                )
                .await;
            }
            crate::call::descriptor::execute_callable_descriptor_in_context(
                &runtime,
                crate::call::descriptor::CallableDescriptor::resolved(
                    request.identity,
                    request.arguments,
                    request.requested_outputs,
                    fallback_policy,
                    crate::call::descriptor::CallableCallKind::Direct,
                ),
            )
            .await
        })
    }

    fn source_functions(&self, source_id: SourceId) -> Vec<(String, usize)> {
        crate::user_functions::source_functions_for(source_id)
            .into_iter()
            .map(|function| (function.name, function.function))
            .collect()
    }
}

/// Invocation-scoped builtin authority used by exact compiled products.
/// When installed, the dispatcher must not fall back to process-global
/// discovery for a missing name.
pub trait RuntimeBuiltinService {
    fn bindings_by_name(&self, name: &str) -> Vec<RuntimeBuiltinBinding>;
}

pub trait RuntimeWorkspaceService {
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

    /// Admit an ordered executor-owned mutation without reconciling an
    /// externally changed workspace view. Executor register stores use this
    /// path after their exact frame lifecycle check; external/reentrant
    /// effects continue to use [`Self::ensure_effect_authority`].
    fn ensure_mutation_authority(&self) -> Result<(), RuntimeError> {
        self.ensure_effect_authority()
    }

    /// Revision of an authority-backed workspace after its latest admission
    /// checkpoint. Exact executor adapters use this to avoid materializing a
    /// complete workspace when no externally visible state changed.
    fn workspace_revision(
        &self,
    ) -> Result<Option<crate::workspace::WorkspaceRevision>, RuntimeError> {
        Ok(None)
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
    fn console(&self, stream: crate::console::ConsoleStream, text: String);

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
    fn supports_operation(&self, operation: &str) -> bool;

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
            Ok(super::preparation::PreparedBuiltinOperands::passthrough(
                request.into_operands(),
            ))
        })
    }
}

/// Deferred candidate evidence with no runtime context/service ownership. Send
/// prevents a source from retaining the session's Rc graph; callbacks still do
/// not deliver feedback themselves. The owning session polls nonblocking.
pub trait RuntimePlacementFeedbackSource: Send {
    fn signature(&self) -> &runmat_execution::PlacementSignature;
    fn take_ready(&mut self) -> Option<runmat_execution::PlacementFeedback>;
    /// A completed cancellation or abandonment has no feedback sample. The
    /// owner may retire its source without manufacturing a failed observation.
    fn terminal_without_feedback(&self) -> bool {
        false
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
    pub observation_lease: super::ManagedObservationLease,
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

/// Session-owned execution-placement authority. The runtime exposes only
/// executor-neutral contracts; candidate generation and policy remain in their
/// owning executor/acceleration crates.
pub trait RuntimePlacementService {
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
                .build(),
        )
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
                .build(),
        )
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

#[derive(Debug, Clone)]
pub struct ForeignCall {
    pub adapter: String,
    pub symbol: String,
    pub arguments: Vec<Value>,
    pub requested_outputs: usize,
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
    ) -> RuntimeServiceFuture<Result<runmat_value::ValueSequence, RuntimeError>>;
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
        if self.labs.len() != self.gang.handle.labs.0 as usize
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

    fn retire(
        &self,
        gang: runmat_execution::GangHandle,
    ) -> RuntimeServiceFuture<Result<(), RuntimeError>>;

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
    pub arguments: Vec<Value>,
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
    pub handle: runmat_execution::DistributedValueHandle,
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
        handle: runmat_execution::DistributedValueHandle,
    ) -> RuntimeServiceFuture<Result<RuntimeDistributedSnapshot, RuntimeError>>;

    fn export_local(
        &self,
        handle: runmat_execution::DistributedValueHandle,
        rank: LabRank,
    ) -> RuntimeServiceFuture<Result<runmat_execution::DistributedShardSnapshot, RuntimeError>>;

    fn local_part(
        &self,
        handle: runmat_execution::DistributedValueHandle,
        rank: LabRank,
    ) -> RuntimeServiceFuture<Result<Value, RuntimeError>>;

    fn materialize(
        &self,
        handle: runmat_execution::DistributedValueHandle,
    ) -> RuntimeServiceFuture<Result<Value, RuntimeError>>;

    fn redistribute(
        &self,
        handle: runmat_execution::DistributedValueHandle,
        scheme: DistributionScheme,
    ) -> RuntimeServiceFuture<Result<runmat_execution::DistributedValueHandle, RuntimeError>>;

    fn invoke(
        &self,
        request: RuntimeDistributedCallRequest,
    ) -> RuntimeServiceFuture<Result<runmat_value::ValueSequence, RuntimeError>>;

    fn composite_entry(
        &self,
        handle: runmat_execution::CompositeHandle,
        rank: LabRank,
    ) -> RuntimeServiceFuture<Result<Option<Value>, RuntimeError>>;
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

/// Narrow, typed ports composed by the host. An absent port is meaningful and
/// produces a stable capability error through the corresponding `require_*`
/// accessor; there is no string-keyed service locator.
#[derive(Clone)]
pub struct RuntimeServicePorts {
    call: Option<Rc<dyn RuntimeCallService>>,
    builtin: Option<Rc<dyn RuntimeBuiltinService>>,
    workspace: Option<Rc<dyn RuntimeWorkspaceService>>,
    base_workspace: Option<Rc<dyn RuntimeBaseWorkspaceService>>,
    object: Option<Rc<dyn RuntimeObjectService>>,
    host: Option<Rc<dyn RuntimeHostService>>,
    graphics_presentation: Option<Rc<dyn RuntimeGraphicsPresentationService>>,
    graphics_callbacks: Option<super::WeakRuntimeGraphicsCallbackQueue>,
    http: Option<Rc<dyn RuntimeHttpService>>,
    preferences: Option<Rc<dyn RuntimePreferenceService>>,
    user_path: Option<Rc<dyn RuntimeUserPathService>>,
    #[cfg(not(target_arch = "wasm32"))]
    process: Option<Rc<dyn RuntimeProcessService>>,
    tcp: Option<Rc<dyn RuntimeTcpService>>,
    error: Option<Rc<dyn RuntimeErrorService>>,
    acceleration: Option<Rc<dyn RuntimeAccelerationService>>,
    placement: Option<Rc<dyn RuntimePlacementService>>,
    host_effect_resources: Option<Rc<dyn RuntimeHostEffectResourceService>>,
    native: Option<Rc<dyn RuntimeNativeService>>,
    foreign: Option<Rc<dyn RuntimeForeignService>>,
    parallel: Option<Rc<dyn RuntimeParallelService>>,
    spmd: Option<Rc<dyn RuntimeSpmdService>>,
    distributed: Option<Rc<dyn RuntimeDistributedService>>,
    collective: Option<Rc<dyn RuntimeCollectiveService>>,
    timing: Rc<dyn RuntimeTimingService>,
    local_tasks: Option<Rc<dyn super::RuntimeLocalTaskService>>,
    source_evaluation: Option<Rc<dyn super::RuntimeSourceEvaluationService>>,
    callable_artifacts: Option<Rc<dyn super::RuntimeCallableArtifactService>>,
}

impl Default for RuntimeServicePorts {
    fn default() -> Self {
        Self {
            call: None,
            builtin: None,
            workspace: None,
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
            foreign: None,
            parallel: None,
            spmd: None,
            distributed: None,
            collective: None,
            timing: super::timing::platform_service(),
            local_tasks: None,
            source_evaluation: None,
            callable_artifacts: None,
        }
    }
}

impl std::fmt::Debug for RuntimeServicePorts {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = formatter.debug_struct("RuntimeServicePorts");
        debug
            .field("call", &self.call.is_some())
            .field("builtin", &self.builtin.is_some())
            .field("workspace", &self.workspace.is_some())
            .field("base_workspace", &self.base_workspace.is_some())
            .field("object", &self.object.is_some())
            .field("host", &self.host.is_some())
            .field(
                "graphics_presentation",
                &self.graphics_presentation.is_some(),
            )
            .field("graphics_callbacks", &self.graphics_callbacks.is_some())
            .field("http", &self.http.is_some())
            .field("preferences", &self.preferences.is_some())
            .field("user_path", &self.user_path.is_some())
            .field("tcp", &self.tcp.is_some())
            .field("error", &self.error.is_some())
            .field("acceleration", &self.acceleration.is_some())
            .field("placement", &self.placement.is_some())
            .field(
                "host_effect_resources",
                &self.host_effect_resources.is_some(),
            )
            .field("native", &self.native.is_some())
            .field("foreign", &self.foreign.is_some())
            .field("parallel", &self.parallel.is_some())
            .field("spmd", &self.spmd.is_some())
            .field("distributed", &self.distributed.is_some())
            .field("collective", &self.collective.is_some());
        debug.field("timing", &"installed");
        debug.field("local_tasks", &self.local_tasks.is_some());
        debug.field("source_evaluation", &self.source_evaluation.is_some());
        debug.field("callable_artifacts", &self.callable_artifacts.is_some());
        #[cfg(not(target_arch = "wasm32"))]
        debug.field("process", &self.process.is_some());
        debug.finish()
    }
}

macro_rules! port_accessors {
    ($with:ident, $get:ident, $require:ident, $field:ident, $trait_name:ident, $cap:ident) => {
        pub fn $with(mut self, service: Rc<dyn $trait_name>) -> Self {
            self.$field = Some(service);
            self
        }

        pub fn $get(&self) -> Option<&Rc<dyn $trait_name>> {
            self.$field.as_ref()
        }

        pub fn $require(
            &self,
            operation: impl Into<String>,
        ) -> Result<&Rc<dyn $trait_name>, RuntimeCapabilityError> {
            self.$field
                .as_ref()
                .ok_or_else(|| RuntimeCapabilityError::new(RuntimeCapability::$cap, operation))
        }
    };
}

impl RuntimeServicePorts {
    pub fn with_local_tasks(mut self, service: Rc<dyn super::RuntimeLocalTaskService>) -> Self {
        self.local_tasks = Some(service);
        self
    }

    pub fn local_tasks(&self) -> Option<&Rc<dyn super::RuntimeLocalTaskService>> {
        self.local_tasks.as_ref()
    }

    pub fn require_local_tasks(
        &self,
        operation: impl Into<String>,
    ) -> Result<&Rc<dyn super::RuntimeLocalTaskService>, RuntimeCapabilityError> {
        self.local_tasks
            .as_ref()
            .ok_or_else(|| RuntimeCapabilityError::new(RuntimeCapability::LocalTasks, operation))
    }

    pub fn with_timing(mut self, service: Rc<dyn RuntimeTimingService>) -> Self {
        self.timing = service;
        self
    }

    pub fn timing(&self) -> &Rc<dyn RuntimeTimingService> {
        &self.timing
    }
    /// A staged effect may finish only under the same installed owners that
    /// reserved the scan and supplied the original placement revision.
    pub(crate) fn same_host_effect_owners(&self, other: &Self) -> bool {
        fn same<T: ?Sized>(left: Option<&Rc<T>>, right: Option<&Rc<T>>) -> bool {
            match (left, right) {
                (Some(left), Some(right)) => Rc::ptr_eq(left, right),
                (None, None) => true,
                _ => false,
            }
        }
        same(self.placement(), other.placement())
            && same(self.host_effect_resources(), other.host_effect_resources())
            && same(self.parallel(), other.parallel())
    }

    port_accessors!(
        with_call,
        call,
        require_call,
        call,
        RuntimeCallService,
        Call
    );
    port_accessors!(
        with_builtin,
        builtin,
        require_builtin,
        builtin,
        RuntimeBuiltinService,
        Builtin
    );
    port_accessors!(
        with_workspace,
        workspace,
        require_workspace,
        workspace,
        RuntimeWorkspaceService,
        Workspace
    );
    port_accessors!(
        with_base_workspace,
        base_workspace,
        require_base_workspace,
        base_workspace,
        RuntimeBaseWorkspaceService,
        BaseWorkspace
    );
    port_accessors!(
        with_source_evaluation,
        source_evaluation,
        require_source_evaluation,
        source_evaluation,
        RuntimeSourceEvaluationService,
        SourceEvaluation
    );
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
        self.base_workspace = None;
        self
    }

    /// Parallel labs cannot borrow their parent's window, surface, or event
    /// queue authority merely because their service ports were cloned.
    pub fn without_graphics_presentation(mut self) -> Self {
        self.graphics_presentation = None;
        self.graphics_callbacks = None;
        self
    }

    /// Parallel labs do not inherit the parent session's definition
    /// publication authority. A worker must install its own artifact owner.
    pub fn without_callable_artifacts(mut self) -> Self {
        self.callable_artifacts = None;
        self
    }
    /// Session-local callback registration authority. The port is weak so a
    /// selected callback context cannot keep its owning queue alive in a
    /// context → presentation → queue cycle.
    pub fn with_graphics_callback_queue(
        mut self,
        queue: super::WeakRuntimeGraphicsCallbackQueue,
    ) -> Self {
        self.graphics_callbacks = Some(queue);
        self
    }

    pub fn graphics_callback_queue(&self) -> Option<&super::WeakRuntimeGraphicsCallbackQueue> {
        self.graphics_callbacks.as_ref()
    }
    port_accessors!(
        with_object,
        object,
        require_object,
        object,
        RuntimeObjectService,
        Object
    );
    port_accessors!(
        with_host,
        host,
        require_host,
        host,
        RuntimeHostService,
        Host
    );
    port_accessors!(
        with_graphics_presentation,
        graphics_presentation,
        require_graphics_presentation,
        graphics_presentation,
        RuntimeGraphicsPresentationService,
        GraphicsPresentation
    );
    port_accessors!(
        with_http,
        http,
        require_http,
        http,
        RuntimeHttpService,
        Http
    );
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
    port_accessors!(with_tcp, tcp, require_tcp, tcp, RuntimeTcpService, Tcp);
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
    port_accessors!(
        with_foreign,
        foreign,
        require_foreign,
        foreign,
        RuntimeForeignService,
        Foreign
    );

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
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let failures = [
            workspace.base_global_binding_names().unwrap_err(),
            workspace.is_base_global_binding("g").unwrap_err(),
            workspace.bind_base_global("g").unwrap_err(),
            workspace.unbind_base_global("g").unwrap_err(),
            workspace
                .assign_base_local("g", Value::Num(1.0))
                .unwrap_err(),
            workspace.declare_global_binding(0, "g").unwrap_err(),
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
        let lab = parent.fork_parallel_lab(parent.service_ports().clone());
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
