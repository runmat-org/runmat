//! Auto's actual selected route can be host or provider. Other integration
//! witnesses keep their stricter provider-only capture unchanged.

use super::*;
use runmat_execution::{
    CandidateExecutionLocation, CandidateObservationEvidence, Digest, PlacementDecision,
    PlacementFeedback, PlacementInvalidation, PlacementPlanRequest, PlacementRevision,
    ProgramRevision,
};
use runmat_runtime::context::{
    RuntimeGraphPlacementSelection, RuntimePlacementFeedbackSource, RuntimePlacementService,
    VmFusionRegionRegistration, VmFusionSessionRegistration,
};
use runmat_runtime::RuntimeError;
use std::sync::Weak;

#[derive(Clone, Copy)]
pub(super) enum AutoSelectedRoute {
    Host,
    Provider { device_id: u32 },
}

pub(super) struct AutoSelection {
    pub(super) request: PlacementPlanRequest,
    pub(super) signature: PlacementSignature,
    pub(super) candidate: String,
    pub(super) kind: ExecutionCandidateKind,
    pub(super) preparation: CandidatePreparationState,
    pub(super) route: AutoSelectedRoute,
}

pub(super) struct AutoPlacement {
    pub(super) base: CapturingPlacement,
    pub(super) selections: RefCell<Vec<AutoSelection>>,
}

impl AutoPlacement {
    pub(super) fn new(session: Rc<PlacementSession>) -> Self {
        Self {
            base: CapturingPlacement {
                session,
                reduction_selection: RefCell::new(Vec::new()),
                delivered: Arc::new(Mutex::new(Vec::new())),
                invalidate_selected_provider: Cell::new(false),
                revoked_before_publication: Cell::new(0),
            },
            selections: RefCell::new(Vec::new()),
        }
    }
}

impl RuntimePlacementService for AutoPlacement {
    fn whole_work_revision(
        &self,
        program: Option<ProgramRevision>,
        provider: Digest,
    ) -> Option<PlacementRevision> {
        RuntimePlacementService::whole_work_revision(&self.base, program, provider)
    }

    fn whole_work_runtime_binding_session_revision(
        &self,
        program: Option<ProgramRevision>,
        binding: runmat_builtins::BuiltinBindingIdentity,
        registration: Digest,
    ) -> Option<PlacementRevision> {
        RuntimePlacementService::whole_work_runtime_binding_session_revision(
            &self.base,
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
            &self.base, program, region, liveness,
        )
    }

    fn whole_work_host_service_revision(
        &self,
        program: Option<ProgramRevision>,
        host_service: Digest,
    ) -> Option<PlacementRevision> {
        RuntimePlacementService::whole_work_host_service_revision(&self.base, program, host_service)
    }

    fn whole_work_host_service_revision_ref<'a>(
        &self,
        program: Option<&'a ProgramRevision>,
        host_service: Digest,
    ) -> Option<runmat_execution::PlacementRevisionRef<'a>> {
        RuntimePlacementService::whole_work_host_service_revision_ref(
            &self.base,
            program,
            host_service,
        )
    }

    fn select_whole_work(
        &self,
        request: runmat_execution::WholeWorkPlacementRequest,
    ) -> Result<runmat_runtime::context::RuntimeWholeWorkSelection, RuntimeError> {
        RuntimePlacementService::select_whole_work(&self.base, request)
    }

    fn select_whole_work_borrowed(
        &self,
        request: runmat_execution::WholeWorkPlacementRequestRef<'_>,
    ) -> Result<runmat_runtime::context::RuntimeWholeWorkSelection, RuntimeError> {
        RuntimePlacementService::select_whole_work_borrowed(&self.base, request)
    }

    fn select_whole_work_borrowed_managed(
        &self,
        request: runmat_execution::WholeWorkPlacementRequestRef<'_>,
    ) -> Result<runmat_runtime::context::RuntimeManagedWholeWorkSelection, RuntimeError> {
        RuntimePlacementService::select_whole_work_borrowed_managed(&self.base, request)
    }

    fn observe_selected_whole_work(
        &self,
        selected: &runmat_runtime::context::RuntimeWholeWorkSelection,
        elapsed_ns: u64,
        outcome: runmat_execution::WholeWorkObservationOutcome,
    ) -> Result<(), RuntimeError> {
        RuntimePlacementService::observe_selected_whole_work(
            &self.base, selected, elapsed_ns, outcome,
        )
    }

    fn observe_managed_whole_work(
        &self,
        selected: &runmat_runtime::context::RuntimeManagedWholeWorkSelection,
        elapsed_ns: u64,
        outcome: runmat_execution::WholeWorkObservationOutcome,
    ) -> Result<(), RuntimeError> {
        RuntimePlacementService::observe_managed_whole_work(
            &self.base, selected, elapsed_ns, outcome,
        )
    }

    fn observed_candidate_duration(
        &self,
        signature: &PlacementSignature,
        candidate: &str,
        kind: ExecutionCandidateKind,
    ) -> CandidateObservationEvidence {
        RuntimePlacementService::observed_candidate_duration(&self.base, signature, candidate, kind)
    }

    fn plan(&self, request: PlacementPlanRequest) -> Result<PlacementDecision, RuntimeError> {
        self.plan_with_observation_permit(request)
            .map(|selected| selected.decision)
    }

    fn plan_with_observation_permit(
        &self,
        request: PlacementPlanRequest,
    ) -> Result<RuntimeGraphPlacementSelection, RuntimeError> {
        let reduction = request.signature.operation == "fusion.reduction";
        let selected = RuntimePlacementService::plan_with_observation_permit(
            self.base.session.as_ref(),
            request.clone(),
        )?;
        if reduction {
            assert_eq!(selected.decision.signature, request.signature);
            let [choice] = selected.decision.selections.as_slice() else {
                panic!("one exact selected Auto reduction route");
            };
            let [node] = request.graph.nodes.as_slice() else {
                panic!("one exact Auto reduction graph node");
            };
            assert_eq!(choice.node, node.identity);
            let offered = node
                .candidates
                .iter()
                .find(|candidate| {
                    candidate.descriptor.identity == choice.candidate
                        && candidate.descriptor.kind == choice.kind
                })
                .expect("selected Auto route was offered in the exact request");
            let route = match (choice.kind, offered.descriptor.execution_location) {
                (ExecutionCandidateKind::SharedRuntime, CandidateExecutionLocation::Host) => {
                    AutoSelectedRoute::Host
                }
                (
                    ExecutionCandidateKind::ProviderFusion,
                    CandidateExecutionLocation::Provider { device_id },
                ) => {
                    assert_eq!(
                        request.signature.provider_provenance,
                        Some(
                            typed_reduction_placement::feedback::selected_provider_capability_digest(
                                device_id
                            )
                        )
                    );
                    AutoSelectedRoute::Provider { device_id }
                }
                _ => panic!("Auto selected route kind and execution location disagree"),
            };
            assert!(selected.observation_permit.as_ref().is_some_and(|permit| {
                permit.authorizes(&selected.decision.signature, &choice.candidate, choice.kind)
            }));
            if self.base.invalidate_selected_provider.replace(false) {
                let AutoSelectedRoute::Provider { device_id } = route else {
                    panic!("provider fault injection requires a selected provider route");
                };
                let provenance =
                    typed_reduction_placement::feedback::selected_provider_capability_digest(
                        device_id,
                    );
                let permit = selected
                    .observation_permit
                    .as_ref()
                    .expect("selected graph observation permit");
                self.base
                    .session
                    .invalidate(PlacementInvalidation::Provider { digest: provenance });
                assert!(!permit.authorizes(&request.signature, &choice.candidate, choice.kind));
                self.base
                    .revoked_before_publication
                    .set(self.base.revoked_before_publication.get() + 1);
            }
            let preparation = offered.descriptor.preparation;
            self.selections.borrow_mut().push(AutoSelection {
                request,
                signature: selected.decision.signature.clone(),
                candidate: choice.candidate.clone(),
                kind: choice.kind,
                preparation,
                route,
            });
        }
        Ok(selected)
    }

    fn observe(&self, feedback: PlacementFeedback) -> Result<(), RuntimeError> {
        RuntimePlacementService::observe(&self.base, feedback)
    }

    fn defer_feedback(
        &self,
        source: Box<dyn RuntimePlacementFeedbackSource>,
    ) -> Result<(), Box<dyn RuntimePlacementFeedbackSource>> {
        RuntimePlacementService::defer_feedback(&self.base, source)
    }

    fn drain_feedback(&self) {
        RuntimePlacementService::drain_feedback(&self.base)
    }

    fn invalidate(&self, invalidation: PlacementInvalidation) {
        RuntimePlacementService::invalidate(&self.base, invalidation)
    }
}
