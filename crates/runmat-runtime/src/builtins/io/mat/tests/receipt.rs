//! Synthetic selection authority for exercising real outer MAT futures. It
//! deliberately supplies unavailable bounds and must never certify a finite
//! production resource route.

use crate::context::host_effect::{
    HostEffectRequest, HostEffectResourceAdmission, HostEffectResourceEvidence,
    HostEffectResourceLivenessLease,
};
use crate::context::{
    RuntimeContext, RuntimeHostEffectResourceService, RuntimePlacementService, RuntimeServicePorts,
    RuntimeWholeWorkProbeLease, RuntimeWholeWorkSelection,
};
use runmat_execution::{
    Digest, PlacementDecision, PlacementExecutionAuthorityRevision, PlacementFeedback,
    PlacementInvalidation, PlacementPlanRequest, PlacementRevision, ResourceBoundUnavailable,
    ResourceByteBound, WholeWorkDecision, WholeWorkDecisionKind, WholeWorkObservation,
    WholeWorkObservationOutcome, WholeWorkPlacementRequest,
};
use runmat_filesystem::FsProvider;
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

mod fs;
pub(super) use fs::PendingWriteFs;

struct Live(Arc<AtomicBool>);
impl HostEffectResourceLivenessLease for Live {
    fn is_current(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

struct Resources {
    filesystem: Arc<dyn FsProvider>,
    live: Arc<AtomicBool>,
}
impl RuntimeHostEffectResourceService for Resources {
    fn admission(
        &self,
        request: &HostEffectRequest<'_>,
        _: &RuntimeContext,
    ) -> Option<HostEffectResourceAdmission> {
        if !Arc::ptr_eq(request.filesystem, &self.filesystem)
            || request.domain_facts != Digest::sha256(request.workload.identity_bytes())
            || !matches!(
                request.operation,
                "save" | "matfile" | "matlab.io.MatFile.subsref" | "matlab.io.MatFile.subsasgn"
            )
        {
            return None;
        }
        Some(HostEffectResourceAdmission {
            evidence: HostEffectResourceEvidence {
                authority_revision: Digest::sha256(b"synthetic-mat-receipt-test"),
                scratch_bytes: ResourceByteBound::Unavailable(
                    ResourceBoundUnavailable::NoPeakMemoryBound,
                ),
                returned_value_bytes: ResourceByteBound::Unavailable(
                    ResourceBoundUnavailable::NoRetainedValueBound,
                ),
            },
            liveness_lease: Arc::new(Live(self.live.clone())),
            output_allocation_permit: None,
        })
    }
}

#[derive(Default)]
pub(super) struct Placement {
    selections: Cell<usize>,
    observations: RefCell<Vec<WholeWorkObservationOutcome>>,
    probes: Arc<AtomicUsize>,
}

struct Probe(Arc<AtomicUsize>);
impl RuntimeWholeWorkProbeLease for Probe {}
impl Drop for Probe {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

impl RuntimePlacementService for Placement {
    fn whole_work_host_service_revision(
        &self,
        program: Option<runmat_execution::ProgramRevision>,
        host_service: Digest,
    ) -> Option<PlacementRevision> {
        let revision = Digest::sha256(b"synthetic-mat-placement-test");
        Some(PlacementRevision {
            program,
            catalog: revision,
            compiler: revision,
            authority: PlacementExecutionAuthorityRevision::HostService(host_service),
            policy: revision,
        })
    }

    fn select_whole_work(
        &self,
        request: WholeWorkPlacementRequest,
    ) -> Result<RuntimeWholeWorkSelection, crate::RuntimeError> {
        request
            .validate()
            .expect("typed MAT host-effect test request");
        let candidate = &request.candidates[0];
        self.selections.set(self.selections.get() + 1);
        self.probes.fetch_add(1, Ordering::SeqCst);
        Ok(RuntimeWholeWorkSelection {
            decision: WholeWorkDecision {
                signature: request.signature.clone(),
                selected: candidate.selected(&request.signature.operation),
                kind: WholeWorkDecisionKind::Unmeasured,
                predicted_total_ns: None,
            },
            probe: Some(Box::new(Probe(self.probes.clone()))),
            observation_permit: None,
        })
    }

    fn observe_whole_work(
        &self,
        observation: WholeWorkObservation,
    ) -> Result<(), crate::RuntimeError> {
        self.observations.borrow_mut().push(observation.outcome);
        Ok(())
    }

    fn plan(&self, _: PlacementPlanRequest) -> Result<PlacementDecision, crate::RuntimeError> {
        unreachable!("MAT host-effect test selects complete work")
    }
    fn observe(&self, _: PlacementFeedback) -> Result<(), crate::RuntimeError> {
        unreachable!("MAT host-effect test observes complete work")
    }
    fn invalidate(&self, _: PlacementInvalidation) {}
}

pub(super) struct Witness {
    pub(super) runtime: RuntimeContext,
    placement: Rc<Placement>,
    live: Arc<AtomicBool>,
}

impl Witness {
    pub(super) fn new(filesystem: Arc<dyn FsProvider>) -> Self {
        let placement = Rc::new(Placement::default());
        let live = Arc::new(AtomicBool::new(true));
        let runtime =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_service_ports(
                    RuntimeServicePorts::default()
                        .with_placement(placement.clone())
                        .with_host_effect_resources(Rc::new(Resources {
                            filesystem,
                            live: live.clone(),
                        })),
                );
        Self {
            runtime,
            placement,
            live,
        }
    }

    pub(super) fn revoke(&self) {
        self.live.store(false, Ordering::SeqCst);
    }

    pub(super) fn selections(&self) -> usize {
        self.placement.selections.get()
    }

    pub(super) fn probes(&self) -> usize {
        self.placement.probes.load(Ordering::SeqCst)
    }

    pub(super) fn outcomes(&self) -> Vec<WholeWorkObservationOutcome> {
        self.placement.observations.borrow().clone()
    }
}
