//! Actual-adapter qualification of two measured routes for one installed VM
//! reduction. The test delegates every decision, permit, and receipt to the
//! canonical PlacementSession; it never selects a candidate itself.

use super::*;
use runmat_execution::{
    CandidateObservationEvidence, CandidatePreparationState, ExecutionCandidateKind,
    PlacementDecision, PlacementFeedback, PlacementInvalidation, PlacementPlanRequest,
    PlacementRevision, PlacementSignature, ProgramRevision,
};
use runmat_runtime::context::{
    RuntimeGraphPlacementSelection, RuntimePlacementFeedbackSource, RuntimePlacementService,
    VmFusionRegionRegistration, VmFusionSessionRegistration,
};
use runmat_runtime::RuntimeError;
use std::sync::Weak;

#[derive(Clone)]
pub(super) struct MeasuredChoice {
    pub(super) signature: PlacementSignature,
    pub(super) candidate: String,
    pub(super) kind: ExecutionCandidateKind,
    pub(super) from_cache: bool,
    pub(super) predicted_total_ns: u64,
    pub(super) selected_observed_before: bool,
    pub(super) cpu_observed_candidate: Option<String>,
    pub(super) warm_provider_observed_candidate: Option<String>,
    pub(super) cpu_observation_before: Option<CandidateObservationEvidence>,
    pub(super) warm_provider_observation_before: Option<CandidateObservationEvidence>,
    pub(super) offers: Vec<(
        String,
        ExecutionCandidateKind,
        CandidatePreparationState,
        bool,
    )>,
    pub(super) receipts_before: usize,
    pub(super) resources: runmat_execution::PlacementResourceSnapshot,
}

pub(super) struct MeasuredPlacement {
    pub(super) session: Rc<PlacementSession>,
    pub(super) choices: RefCell<Vec<MeasuredChoice>>,
    pub(super) delivered: Arc<Mutex<Vec<PlacementFeedback>>>,
}

struct MeasuredDeferred {
    source: Box<dyn RuntimePlacementFeedbackSource>,
    delivered: Arc<Mutex<Vec<PlacementFeedback>>>,
}

impl RuntimePlacementFeedbackSource for MeasuredDeferred {
    fn signature(&self) -> &PlacementSignature {
        self.source.signature()
    }

    fn take_ready(&mut self) -> Option<runmat_runtime::context::RuntimePlacementFeedbackDelivery> {
        let ready = self.source.take_ready()?;
        if let Some(snapshot) = ready.snapshot() {
            self.delivered.lock().unwrap().push(snapshot);
        }
        Some(ready)
    }

    fn terminal_without_feedback(&self) -> bool {
        self.source.terminal_without_feedback()
    }
}

impl RuntimePlacementService for MeasuredPlacement {
    fn whole_work_revision(
        &self,
        program: Option<ProgramRevision>,
        provider: runmat_execution::Digest,
    ) -> Option<PlacementRevision> {
        RuntimePlacementService::whole_work_revision(self.session.as_ref(), program, provider)
    }

    fn whole_work_runtime_binding_session_revision(
        &self,
        program: Option<ProgramRevision>,
        binding: runmat_builtins::BuiltinBindingIdentity,
        registration: runmat_execution::Digest,
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
        host_service: runmat_execution::Digest,
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
        host_service: runmat_execution::Digest,
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
    ) -> Result<RuntimeGraphPlacementSelection, RuntimeError> {
        let reduction = request.signature.operation == "fusion.reduction";
        let observed = if reduction {
            request
                .graph
                .nodes
                .iter()
                .flat_map(|node| &node.candidates)
                .map(|candidate| {
                    (
                        candidate.descriptor.identity.clone(),
                        candidate.descriptor.kind,
                        candidate.descriptor.preparation,
                        self.session.observed_candidate_duration(
                            &request.signature,
                            &candidate.descriptor.identity,
                            candidate.descriptor.kind,
                        ),
                    )
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let resources = request.resources.clone();
        let selected =
            RuntimePlacementService::plan_with_observation_permit(self.session.as_ref(), request)?;
        if reduction {
            let [choice] = selected.decision.selections.as_slice() else {
                panic!("one typed reduction choice is required");
            };
            assert!(matches!(
                choice.kind,
                ExecutionCandidateKind::SharedRuntime | ExecutionCandidateKind::ProviderFusion
            ));
            assert!(selected.observation_permit.as_ref().is_some_and(|permit| {
                permit.authorizes(&selected.decision.signature, &choice.candidate, choice.kind)
            }));
            let warm_provider_observation = observed
                .iter()
                .filter(|(candidate, kind, preparation, evidence)| {
                    *kind == ExecutionCandidateKind::ProviderFusion
                        && *preparation == CandidatePreparationState::Warm
                        && matches!(evidence, CandidateObservationEvidence::Observed(_))
                        && (choice.kind != ExecutionCandidateKind::ProviderFusion
                            || candidate == &choice.candidate)
                })
                .min_by_key(|(_, _, _, evidence)| super::performance::observed_cost(*evidence).2);
            self.choices.borrow_mut().push(MeasuredChoice {
                signature: selected.decision.signature.clone(),
                candidate: choice.candidate.clone(),
                kind: choice.kind,
                from_cache: selected.decision.from_cache,
                predicted_total_ns: selected.decision.predicted_total_ns,
                selected_observed_before: observed.iter().any(|(candidate, kind, _, evidence)| {
                    candidate == &choice.candidate
                        && *kind == choice.kind
                        && matches!(evidence, CandidateObservationEvidence::Observed(_))
                }),
                cpu_observed_candidate: observed.iter().find_map(
                    |(candidate, kind, _, evidence)| {
                        (*kind == ExecutionCandidateKind::SharedRuntime
                            && matches!(evidence, CandidateObservationEvidence::Observed(_)))
                        .then(|| candidate.clone())
                    },
                ),
                warm_provider_observed_candidate: warm_provider_observation
                    .map(|(candidate, _, _, _)| candidate.clone()),
                cpu_observation_before: observed.iter().find_map(|(_, kind, _, evidence)| {
                    (*kind == ExecutionCandidateKind::SharedRuntime
                        && matches!(evidence, CandidateObservationEvidence::Observed(_)))
                    .then_some(*evidence)
                }),
                warm_provider_observation_before: warm_provider_observation
                    .map(|(_, _, _, evidence)| *evidence),
                offers: observed
                    .iter()
                    .map(|(candidate, kind, preparation, evidence)| {
                        (
                            candidate.clone(),
                            *kind,
                            *preparation,
                            matches!(evidence, CandidateObservationEvidence::Observed(_)),
                        )
                    })
                    .collect(),
                receipts_before: 0,
                resources,
            });
        }
        Ok(selected)
    }

    fn observe(&self, feedback: PlacementFeedback) -> Result<(), RuntimeError> {
        RuntimePlacementService::observe(self.session.as_ref(), feedback.clone())?;
        self.delivered.lock().unwrap().push(feedback);
        Ok(())
    }

    fn defer_feedback(
        &self,
        source: Box<dyn RuntimePlacementFeedbackSource>,
    ) -> Result<(), Box<dyn RuntimePlacementFeedbackSource>> {
        RuntimePlacementService::defer_feedback(
            self.session.as_ref(),
            Box::new(MeasuredDeferred {
                source,
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
