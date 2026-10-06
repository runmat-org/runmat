use super::*;
use crate::context::host_effect::{
    HostEffectOutput, HostEffectRequest, HostEffectResourceEvidence, HostEffectWorkload,
    ManagedHostEffectLedger,
};
use crate::context::{
    ParallelCapability, RuntimeContext, RuntimeHostEffectResourceService, RuntimeParallelResources,
    RuntimeParallelService, RuntimePlacementService, RuntimeServicePorts,
    RuntimeWholeWorkProbeLease, RuntimeWholeWorkSelection,
};
use runmat_execution::{
    CandidateResourceDemand, Digest, PlacementDecision, PlacementFeedback, PlacementInvalidation,
    PlacementPlanRequest, PlacementRevision, ResourceByteBound, WholeWorkDecision,
    WholeWorkDecisionKind, WholeWorkObservation, WholeWorkPlacementRequest,
};
use std::cell::{Cell, RefCell};
use std::path::Path;
use std::rc::Rc;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

pub(super) struct Budget(Option<ManagedHostEffectLedger>);
impl Budget {
    pub(super) fn new(available: Option<u64>) -> Self {
        Self(available.map(ManagedHostEffectLedger::new))
    }
}
impl RuntimeParallelService for Budget {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> RuntimeParallelResources {
        let snapshot = self.0.as_ref().map(ManagedHostEffectLedger::snapshot);
        RuntimeParallelResources {
            memory_available_bytes: snapshot.map(|owner| owner.free_bytes()),
            epoch: snapshot.map_or(0, |owner| owner.epoch),
            ..Default::default()
        }
    }

    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        self.0.clone()
    }
}

pub(super) struct Resources {
    pub(super) filesystem: Arc<dyn runmat_filesystem::FsProvider>,
    pub(super) evidence: Option<HostEffectResourceEvidence>,
    pub(super) calls: Cell<usize>,
    pub(super) workloads: RefCell<Vec<Vec<u8>>>,
}
impl Resources {
    pub(super) fn new(
        filesystem: Arc<dyn runmat_filesystem::FsProvider>,
        evidence: Option<HostEffectResourceEvidence>,
    ) -> Self {
        Self {
            filesystem,
            evidence,
            calls: Cell::new(0),
            workloads: RefCell::new(Vec::new()),
        }
    }
}
impl RuntimeHostEffectResourceService for Resources {
    fn admission(
        &self,
        request: &HostEffectRequest<'_>,
        _: &RuntimeContext,
    ) -> Option<crate::context::host_effect::HostEffectResourceAdmission> {
        self.calls.set(self.calls.get() + 1);
        self.workloads
            .borrow_mut()
            .push(request.workload.identity_bytes());
        if !Arc::ptr_eq(request.filesystem, &self.filesystem)
            || request.domain_facts != Digest::sha256(request.workload.identity_bytes())
        {
            return None;
        }
        let evidence = match request.workload {
            HostEffectWorkload::AudioDecode { source, .. }
                if request.operation == AUDIOREAD_BUILTIN_NAME
                    && source == Path::new("/audio/tone.wav") =>
            {
                self.evidence
            }
            HostEffectWorkload::AudioMetadata {
                source,
                output: HostEffectOutput::EffectOnly | HostEffectOutput::ReturnedValue,
            } if request.operation == AUDIOINFO_BUILTIN_NAME
                && source == Path::new("/audio/tone.wav") =>
            {
                self.evidence
            }
            _ => None,
        };
        evidence.map(crate::context::host_effect::test_admission)
    }
}

#[derive(Default)]
pub(super) struct Placement {
    pub(super) selections: Cell<usize>,
    pub(super) observations: RefCell<Vec<WholeWorkObservation>>,
    pub(super) probes: Arc<AtomicUsize>,
    pub(super) demands: RefCell<Vec<CandidateResourceDemand>>,
    pub(super) managed_ledger: RefCell<Option<ManagedHostEffectLedger>>,
}
struct Probe(Arc<AtomicUsize>);
impl RuntimeWholeWorkProbeLease for Probe {}
impl Drop for Probe {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}
impl RuntimePlacementService for Placement {
    fn whole_work_host_service_revision_ref<'a>(
        &self,
        program: Option<&'a runmat_execution::ProgramRevision>,
        service: Digest,
    ) -> Option<runmat_execution::PlacementRevisionRef<'a>> {
        let digest = Digest::sha256(b"audio-admission-test-placement");
        Some(runmat_execution::PlacementRevisionRef {
            program,
            catalog: digest,
            compiler: digest,
            authority: runmat_execution::PlacementExecutionAuthorityRevision::HostService(service),
            policy: digest,
        })
    }

    fn whole_work_host_service_revision(
        &self,
        program: Option<runmat_execution::ProgramRevision>,
        service: Digest,
    ) -> Option<PlacementRevision> {
        let digest = Digest::sha256(b"audio-admission-test-placement");
        Some(PlacementRevision {
            program,
            catalog: digest,
            compiler: digest,
            authority: runmat_execution::PlacementExecutionAuthorityRevision::HostService(service),
            policy: digest,
        })
    }
    fn select_whole_work(
        &self,
        request: WholeWorkPlacementRequest,
    ) -> Result<RuntimeWholeWorkSelection, crate::RuntimeError> {
        self.select_whole_work_borrowed(request.as_ref())
    }
    fn select_whole_work_borrowed(
        &self,
        request: runmat_execution::WholeWorkPlacementRequestRef<'_>,
    ) -> Result<RuntimeWholeWorkSelection, crate::RuntimeError> {
        request.validate().expect("test audio candidate is valid");
        let candidate = &request.candidates[0];
        if let Some(available) = request.resources.memory_available_bytes {
            match (
                candidate.resources.retained_bytes,
                candidate.resources.scratch_bytes,
            ) {
                (ResourceByteBound::Known(retained), ResourceByteBound::Known(scratch))
                    if retained
                        .checked_add(scratch)
                        .is_some_and(|total| total <= available) => {}
                _ => {
                    return Err(build_runtime_error("test audio memory limit")
                        .with_identifier("RunMat:Placement:ResourceLimit")
                        .build())
                }
            }
        }
        self.demands.borrow_mut().push(candidate.resources);
        self.selections.set(self.selections.get() + 1);
        self.probes.fetch_add(1, Ordering::SeqCst);
        Ok(RuntimeWholeWorkSelection {
            decision: WholeWorkDecision {
                signature: request.signature.to_owned(),
                selected: candidate.selected(request.signature.operation),
                kind: WholeWorkDecisionKind::Unmeasured,
                predicted_total_ns: None,
            },
            probe: Some(Box::new(Probe(self.probes.clone()))),
            observation_permit: None,
        })
    }
    fn select_whole_work_borrowed_managed(
        &self,
        request: runmat_execution::WholeWorkPlacementRequestRef<'_>,
    ) -> Result<crate::context::RuntimeManagedWholeWorkSelection, crate::RuntimeError> {
        request.validate().expect("typed audio candidate");
        let candidate = &request.candidates[0];
        if let Some(available) = request.resources.memory_available_bytes {
            let demand = match (
                candidate.resources.retained_bytes,
                candidate.resources.scratch_bytes,
            ) {
                (ResourceByteBound::Known(retained), ResourceByteBound::Known(scratch)) => {
                    retained.checked_add(scratch)
                }
                _ => None,
            };
            if demand.is_none_or(|demand| demand > available) {
                return Err(build_runtime_error("test audio memory limit")
                    .with_identifier("RunMat:Placement:ResourceLimit")
                    .build());
            }
        }
        self.demands.borrow_mut().push(candidate.resources);
        self.selections.set(self.selections.get() + 1);
        self.probes.fetch_add(1, Ordering::SeqCst);
        crate::context::test_managed_selection(
            request,
            Some(Box::new(Probe(self.probes.clone()))),
            self.managed_ledger
                .borrow()
                .as_ref()
                .expect("staged audio ledger"),
        )
    }
    fn observe_managed_whole_work(
        &self,
        selected: &crate::context::RuntimeManagedWholeWorkSelection,
        elapsed_ns: u64,
        outcome: runmat_execution::WholeWorkObservationOutcome,
    ) -> Result<(), crate::RuntimeError> {
        self.observations
            .borrow_mut()
            .push(selected.test_observation(elapsed_ns, outcome));
        Ok(())
    }
    fn observe_whole_work(
        &self,
        observation: WholeWorkObservation,
    ) -> Result<(), crate::RuntimeError> {
        self.observations.borrow_mut().push(observation);
        Ok(())
    }
    fn plan(&self, _: PlacementPlanRequest) -> Result<PlacementDecision, crate::RuntimeError> {
        unreachable!("audio host effects use whole-work selection")
    }
    fn observe(&self, _: PlacementFeedback) -> Result<(), crate::RuntimeError> {
        unreachable!("audio host effects use whole-work observation")
    }
    fn invalidate(&self, _: PlacementInvalidation) {}
}

pub(super) fn evidence(
    retained: ResourceByteBound,
    scratch: ResourceByteBound,
) -> HostEffectResourceEvidence {
    HostEffectResourceEvidence {
        authority_revision: Digest::sha256(b"fixture-audio-filesystem-codec-revision"),
        scratch_bytes: scratch,
        returned_value_bytes: retained,
    }
}

pub(super) fn context(
    provider: Arc<dyn runmat_filesystem::FsProvider>,
    evidence: Option<HostEffectResourceEvidence>,
    available: Option<u64>,
) -> (RuntimeContext, Rc<Resources>, Rc<Placement>) {
    let resources = Rc::new(Resources::new(provider, evidence));
    let placement = Rc::new(Placement::default());
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_parallel(Rc::new(Budget::new(available)))
                .with_host_effect_resources(resources.clone())
                .with_placement(placement.clone()),
        );
    (runtime, resources, placement)
}
