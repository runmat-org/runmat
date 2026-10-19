//! Test-only observation of real SharedRuntime selections. All policy, permits,
//! revisions, and feedback remain owned by the canonical PlacementSession.

use super::*;
use runmat_execution::{
    CandidateObservationEvidence, ExecutionCandidateKind, PlacementDecision, PlacementFeedback,
    PlacementInvalidation, PlacementPlanRequest, PlacementRevision, PlacementSignature,
    ProgramRevision,
};
use runmat_runtime::context::{
    RuntimeGraphPlacementSelection, RuntimePlacementFeedbackSource, RuntimePlacementService,
    VmFusionRegionRegistration, VmFusionSessionRegistration,
};
use runmat_runtime::RuntimeError;
use std::sync::Weak;

pub(super) struct SelectedReduction {
    pub(super) signature: PlacementSignature,
    pub(super) candidate: String,
    pub(super) kind: ExecutionCandidateKind,
}

pub(super) struct CpuCapturingPlacement {
    pub(super) inner: Rc<CapturingPlacement>,
    pub(super) selected: RefCell<Vec<SelectedReduction>>,
}

impl RuntimePlacementService for CpuCapturingPlacement {
    fn whole_work_revision(
        &self,
        program: Option<ProgramRevision>,
        provider: runmat_execution::Digest,
    ) -> Option<PlacementRevision> {
        RuntimePlacementService::whole_work_revision(self.inner.as_ref(), program, provider)
    }

    fn whole_work_runtime_binding_session_revision(
        &self,
        program: Option<ProgramRevision>,
        binding: runmat_builtins::BuiltinBindingIdentity,
        registration: runmat_execution::Digest,
    ) -> Option<PlacementRevision> {
        RuntimePlacementService::whole_work_runtime_binding_session_revision(
            self.inner.as_ref(),
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
            self.inner.as_ref(),
            program,
            region,
            liveness,
        )
    }

    fn whole_work_host_service_revision(
        &self,
        program: Option<ProgramRevision>,
        host_service: runmat_execution::Digest,
    ) -> Option<PlacementRevision> {
        RuntimePlacementService::whole_work_host_service_revision(
            self.inner.as_ref(),
            program,
            host_service,
        )
    }

    fn whole_work_host_service_revision_ref<'a>(
        &self,
        program: Option<&'a ProgramRevision>,
        host_service: runmat_execution::Digest,
    ) -> Option<runmat_execution::PlacementRevisionRef<'a>> {
        RuntimePlacementService::whole_work_host_service_revision_ref(
            self.inner.as_ref(),
            program,
            host_service,
        )
    }

    fn select_whole_work(
        &self,
        request: runmat_execution::WholeWorkPlacementRequest,
    ) -> Result<runmat_runtime::context::RuntimeWholeWorkSelection, RuntimeError> {
        RuntimePlacementService::select_whole_work(self.inner.as_ref(), request)
    }

    fn select_whole_work_borrowed(
        &self,
        request: runmat_execution::WholeWorkPlacementRequestRef<'_>,
    ) -> Result<runmat_runtime::context::RuntimeWholeWorkSelection, RuntimeError> {
        RuntimePlacementService::select_whole_work_borrowed(self.inner.as_ref(), request)
    }

    fn select_whole_work_borrowed_managed(
        &self,
        request: runmat_execution::WholeWorkPlacementRequestRef<'_>,
    ) -> Result<runmat_runtime::context::RuntimeManagedWholeWorkSelection, RuntimeError> {
        RuntimePlacementService::select_whole_work_borrowed_managed(self.inner.as_ref(), request)
    }

    fn observe_selected_whole_work(
        &self,
        selected: &runmat_runtime::context::RuntimeWholeWorkSelection,
        elapsed_ns: u64,
        outcome: runmat_execution::WholeWorkObservationOutcome,
    ) -> Result<(), RuntimeError> {
        RuntimePlacementService::observe_selected_whole_work(
            self.inner.as_ref(),
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
            self.inner.as_ref(),
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
            self.inner.as_ref(),
            signature,
            candidate,
            kind,
        )
    }

    fn plan(&self, request: PlacementPlanRequest) -> Result<PlacementDecision, RuntimeError> {
        self.plan_with_observation_permit(request)
            .map(|selection| selection.decision)
    }

    fn plan_with_observation_permit(
        &self,
        request: PlacementPlanRequest,
    ) -> Result<RuntimeGraphPlacementSelection, RuntimeError> {
        let reduction = request.signature.operation == "fusion.reduction";
        let offered = reduction.then(|| {
            request
                .graph
                .nodes
                .iter()
                .flat_map(|node| &node.candidates)
                .map(|candidate| {
                    (
                        candidate.descriptor.identity.clone(),
                        candidate.descriptor.kind,
                    )
                })
                .collect::<Vec<_>>()
        });
        let selection = RuntimePlacementService::plan_with_observation_permit(
            self.inner.session.as_ref(),
            request,
        )?;
        if let Some(offered) = offered {
            let [choice] = selection.decision.selections.as_slice() else {
                panic!("one canonical reduction selection required");
            };
            assert!(offered.iter().any(|candidate| {
                candidate.0 == choice.candidate && candidate.1 == choice.kind
            }));
            assert!(selection.observation_permit.as_ref().is_some_and(|permit| {
                permit.authorizes(
                    &selection.decision.signature,
                    &choice.candidate,
                    choice.kind,
                )
            }));
            self.selected.borrow_mut().push(SelectedReduction {
                signature: selection.decision.signature.clone(),
                candidate: choice.candidate.clone(),
                kind: choice.kind,
            });
        }
        Ok(selection)
    }

    fn observe(&self, feedback: PlacementFeedback) -> Result<(), RuntimeError> {
        RuntimePlacementService::observe(self.inner.as_ref(), feedback)
    }

    fn defer_feedback(
        &self,
        source: Box<dyn RuntimePlacementFeedbackSource>,
    ) -> Result<(), Box<dyn RuntimePlacementFeedbackSource>> {
        RuntimePlacementService::defer_feedback(self.inner.as_ref(), source)
    }

    fn drain_feedback(&self) {
        RuntimePlacementService::drain_feedback(self.inner.as_ref());
    }

    fn invalidate(&self, invalidation: PlacementInvalidation) {
        RuntimePlacementService::invalidate(self.inner.as_ref(), invalidation);
    }
}
