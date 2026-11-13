//! Shared real-placement feedback owner; optional typed selection capture is a specialization.
//! This delegates policy, revision, permits, and deferred fences to PlacementSession.
use runmat_accelerate::placement::PlacementSession;
use runmat_execution::{
    CandidateExecutionLocation, CandidateObservationEvidence, CandidatePreparationState, Digest,
    ExecutionCandidateKind, PlacementDecision, PlacementFeedback, PlacementInvalidation,
    PlacementPlanRequest, PlacementRevision, PlacementSignature, ProgramRevision,
};
use runmat_runtime::context::{
    RuntimePlacementFeedbackSource, RuntimePlacementService, VmFusionRegionRegistration,
    VmFusionSessionRegistration,
};
use runmat_runtime::RuntimeError;
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::{Arc, Mutex, Weak};

/// Observe only what the real deferred source actually publishes while the
/// owning PlacementSession still performs its normal nonblocking delivery.
struct CapturingDeferredSource {
    inner: Box<dyn RuntimePlacementFeedbackSource>,
    delivered: Arc<Mutex<Vec<PlacementFeedback>>>,
}

impl RuntimePlacementFeedbackSource for CapturingDeferredSource {
    fn signature(&self) -> &PlacementSignature {
        self.inner.signature()
    }

    fn take_ready(&mut self) -> Option<runmat_runtime::context::RuntimePlacementFeedbackDelivery> {
        let ready = self.inner.take_ready()?;
        if let Some(snapshot) = ready.snapshot() {
            self.delivered.lock().unwrap().push(snapshot);
        }
        Some(ready)
    }

    fn terminal_without_feedback(&self) -> bool {
        self.inner.terminal_without_feedback()
    }
}

/// Capture the exact selected request while retaining the real session's
/// revision, planning, permit, and deferred-fence behavior.
pub(crate) struct FeedbackPlacement<C> {
    pub(crate) session: Rc<PlacementSession>,
    pub(crate) reduction_selection: RefCell<Vec<C>>,
    pub(crate) delivered: Arc<Mutex<Vec<PlacementFeedback>>>,
    /// One-shot test fault at the real selection/publication boundary.
    pub(crate) invalidate_selected_provider: Cell<bool>,
    pub(crate) revoked_before_publication: Cell<usize>,
}

pub(crate) trait SelectionCapture: Sized {
    fn accepts(operation: &str) -> bool;
    fn capture(
        request: PlacementPlanRequest,
        signature: PlacementSignature,
        candidate: String,
        kind: ExecutionCandidateKind,
        preparation: CandidatePreparationState,
        device_id: u32,
    ) -> Self;
}

impl SelectionCapture for () {
    fn accepts(_: &str) -> bool {
        false
    }
    fn capture(
        _: PlacementPlanRequest,
        _: PlacementSignature,
        _: String,
        _: ExecutionCandidateKind,
        _: CandidatePreparationState,
        _: u32,
    ) -> Self {
    }
}

/// Reproduce the production `session_facts` digest from the selected device's
/// registered provider, never from a fixture-provided digest or candidate name.
pub(crate) fn selected_provider_capability_digest(device_id: u32) -> Digest {
    let provider = runmat_accelerate_api::provider_for_device(device_id)
        .expect("selected provider device has an authentic registered owner");
    Digest::sha256(
        serde_json::to_vec(&provider.capability_snapshot())
            .expect("provider capability snapshot must serialize"),
    )
}

impl<C: SelectionCapture> RuntimePlacementService for FeedbackPlacement<C> {
    fn whole_work_revision(
        &self,
        program: Option<ProgramRevision>,
        provider: Digest,
    ) -> Option<PlacementRevision> {
        RuntimePlacementService::whole_work_revision(self.session.as_ref(), program, provider)
    }

    fn whole_work_runtime_binding_session_revision(
        &self,
        program: Option<ProgramRevision>,
        binding: runmat_builtins::BuiltinBindingIdentity,
        registration: Digest,
    ) -> Option<PlacementRevision> {
        RuntimePlacementService::whole_work_runtime_binding_session_revision(
            self.session.as_ref(),
            program,
            binding,
            registration,
        )
    }

    fn whole_work_vm_fusion_region_session_revision(
        &self,
        program: Option<ProgramRevision>,
        region: VmFusionRegionRegistration,
        liveness: Weak<()>,
    ) -> Option<VmFusionSessionRegistration> {
        RuntimePlacementService::whole_work_vm_fusion_region_session_revision(
            self.session.as_ref(),
            program,
            region,
            liveness,
        )
    }

    fn whole_work_host_service_revision(
        &self,
        program: Option<ProgramRevision>,
        host_service: Digest,
    ) -> Option<PlacementRevision> {
        RuntimePlacementService::whole_work_host_service_revision(
            self.session.as_ref(),
            program,
            host_service,
        )
    }

    fn whole_work_host_service_revision_ref<'a>(
        &self,
        program: Option<&'a ProgramRevision>,
        host_service: Digest,
    ) -> Option<runmat_execution::PlacementRevisionRef<'a>> {
        RuntimePlacementService::whole_work_host_service_revision_ref(
            self.session.as_ref(),
            program,
            host_service,
        )
    }

    fn select_whole_work(
        &self,
        request: runmat_execution::WholeWorkPlacementRequest,
    ) -> Result<runmat_runtime::context::RuntimeWholeWorkSelection, RuntimeError> {
        RuntimePlacementService::select_whole_work(self.session.as_ref(), request)
    }

    fn select_whole_work_borrowed(
        &self,
        request: runmat_execution::WholeWorkPlacementRequestRef<'_>,
    ) -> Result<runmat_runtime::context::RuntimeWholeWorkSelection, RuntimeError> {
        RuntimePlacementService::select_whole_work_borrowed(self.session.as_ref(), request)
    }

    fn select_whole_work_borrowed_managed(
        &self,
        request: runmat_execution::WholeWorkPlacementRequestRef<'_>,
    ) -> Result<runmat_runtime::context::RuntimeManagedWholeWorkSelection, RuntimeError> {
        RuntimePlacementService::select_whole_work_borrowed_managed(self.session.as_ref(), request)
    }

    fn observe_selected_whole_work(
        &self,
        selected: &runmat_runtime::context::RuntimeWholeWorkSelection,
        elapsed_ns: u64,
        outcome: runmat_execution::WholeWorkObservationOutcome,
    ) -> Result<(), RuntimeError> {
        RuntimePlacementService::observe_selected_whole_work(
            self.session.as_ref(),
            selected,
            elapsed_ns,
            outcome,
        )
    }

    fn observe_managed_whole_work(
        &self,
        selected: &runmat_runtime::context::RuntimeManagedWholeWorkSelection,
        elapsed_ns: u64,
        outcome: runmat_execution::WholeWorkObservationOutcome,
    ) -> Result<(), RuntimeError> {
        RuntimePlacementService::observe_managed_whole_work(
            self.session.as_ref(),
            selected,
            elapsed_ns,
            outcome,
        )
    }

    fn observed_candidate_duration(
        &self,
        signature: &PlacementSignature,
        candidate: &str,
        kind: ExecutionCandidateKind,
    ) -> CandidateObservationEvidence {
        RuntimePlacementService::observed_candidate_duration(
            self.session.as_ref(),
            signature,
            candidate,
            kind,
        )
    }

    fn plan(&self, request: PlacementPlanRequest) -> Result<PlacementDecision, RuntimeError> {
        self.plan_with_observation_permit(request)
            .map(|selected| selected.decision)
    }

    fn plan_with_observation_permit(
        &self,
        request: PlacementPlanRequest,
    ) -> Result<runmat_runtime::context::RuntimeGraphPlacementSelection, RuntimeError> {
        let signature = C::accepts(&request.signature.operation).then(|| request.signature.clone());
        let reduction_request = signature.as_ref().map(|_| request.clone());
        let reduction_graph = signature.as_ref().map(|_| format!("{:?}", request.graph));
        let offered = request
            .graph
            .nodes
            .iter()
            .flat_map(|node| node.candidates.iter())
            .filter(|candidate| candidate.descriptor.kind == ExecutionCandidateKind::ProviderFusion)
            .map(|candidate| {
                (
                    candidate.descriptor.identity.clone(),
                    candidate.descriptor.preparation,
                    candidate.descriptor.execution_location,
                )
            })
            .collect::<Vec<_>>();
        let selection =
            RuntimePlacementService::plan_with_observation_permit(self.session.as_ref(), request)?;
        let decision = &selection.decision;
        if let Some(signature) = signature {
            assert_eq!(decision.signature, signature);
            let selected = decision
                .selections
                .iter()
                .filter(|choice| choice.kind == ExecutionCandidateKind::ProviderFusion)
                .collect::<Vec<_>>();
            assert_eq!(
                selected.len(),
                1,
                "one exact physical reduction selection; graph={:?}; decision={:?}",
                reduction_graph,
                decision,
            );
            let (preparation, location) = offered
                .iter()
                .find(|(candidate, _, _)| candidate == &selected[0].candidate)
                .map(|(_, preparation, location)| (*preparation, *location))
                .expect("selected physical reduction was offered in this exact request");
            let CandidateExecutionLocation::Provider { device_id } = location else {
                panic!("selected provider fusion has a provider execution location");
            };
            let provenance = selected_provider_capability_digest(device_id);
            assert_eq!(signature.provider_provenance, Some(provenance));
            if self.invalidate_selected_provider.replace(false) {
                let permit = selection
                    .observation_permit
                    .as_ref()
                    .expect("real session issued exact graph observation authority");
                assert!(permit.authorizes(&signature, &selected[0].candidate, selected[0].kind));
                self.session
                    .invalidate(PlacementInvalidation::Provider { digest: provenance });
                assert!(!permit.authorizes(&signature, &selected[0].candidate, selected[0].kind));
                self.revoked_before_publication
                    .set(self.revoked_before_publication.get() + 1);
            }
            self.reduction_selection.borrow_mut().push(C::capture(
                reduction_request.expect("captured exact VM planner request"),
                signature,
                selected[0].candidate.clone(),
                selected[0].kind,
                preparation,
                device_id,
            ));
        }
        Ok(selection)
    }

    fn observe(&self, feedback: PlacementFeedback) -> Result<(), RuntimeError> {
        RuntimePlacementService::observe(self.session.as_ref(), feedback.clone())?;
        // Direct-ready feedback never enters the deferred source. The session's
        // pending drain calls record_feedback internally, not this wrapper.
        self.delivered.lock().unwrap().push(feedback);
        Ok(())
    }

    fn defer_feedback(
        &self,
        source: Box<dyn RuntimePlacementFeedbackSource>,
    ) -> Result<(), Box<dyn RuntimePlacementFeedbackSource>> {
        RuntimePlacementService::defer_feedback(
            self.session.as_ref(),
            Box::new(CapturingDeferredSource {
                inner: source,
                delivered: self.delivered.clone(),
            }),
        )
    }

    fn drain_feedback(&self) {
        RuntimePlacementService::drain_feedback(self.session.as_ref());
    }

    fn invalidate(&self, invalidation: PlacementInvalidation) {
        RuntimePlacementService::invalidate(self.session.as_ref(), invalidation);
    }
}
