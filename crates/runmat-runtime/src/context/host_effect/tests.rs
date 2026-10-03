use super::*;
use crate::context::{
    ParallelCapability, RuntimeHostEffectResourceService, RuntimeParallelResources,
    RuntimeParallelService, RuntimePlacementService, RuntimeServicePorts,
    RuntimeWholeWorkProbeLease,
};
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::{
    CancellationReason, PlacementDecision, PlacementFeedback, PlacementInvalidation,
    PlacementPlanRequest, PlacementRevision, ResourceBoundUnavailable, WholeWorkDecision,
    WholeWorkDecisionKind, WholeWorkObservation,
};
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

struct Resources(Option<HostEffectResourceEvidence>);
impl RuntimeHostEffectResourceService for Resources {
    fn admission(&self, request: &HostEffectRequest) -> Option<HostEffectResourceEvidence> {
        (request.operation == "unzip"
            && request.domain_facts == Digest::sha256(b"source-and-destination"))
        .then_some(self.0)
        .flatten()
    }
}

struct Parallel(Option<u64>);
impl RuntimeParallelService for Parallel {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> RuntimeParallelResources {
        RuntimeParallelResources {
            memory_available_bytes: self.0,
            ..RuntimeParallelResources::default()
        }
    }
}

#[derive(Default)]
struct Placement {
    selections: Cell<usize>,
    observations: RefCell<Vec<WholeWorkObservation>>,
    probes: Arc<AtomicUsize>,
    revision_available: Cell<bool>,
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
        if !self.revision_available.get() {
            return None;
        }
        let digest = Digest::sha256(b"host-effect-test-placement");
        Some(PlacementRevision {
            program,
            catalog: digest,
            compiler: digest,
            authority: runmat_execution::PlacementExecutionAuthorityRevision::HostService(
                host_service,
            ),
            policy: digest,
        })
    }
    fn select_whole_work(
        &self,
        request: WholeWorkPlacementRequest,
    ) -> Result<RuntimeWholeWorkSelection, crate::RuntimeError> {
        request.validate().unwrap();
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
            if !demand.is_some_and(|demand| demand <= available) {
                return Err(build_runtime_error("test placement resource limit")
                    .with_identifier("RunMat:Placement:ResourceLimit")
                    .build());
            }
        }
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
        self.observations.borrow_mut().push(observation);
        Ok(())
    }
    fn plan(&self, _: PlacementPlanRequest) -> Result<PlacementDecision, crate::RuntimeError> {
        unreachable!("host effect uses complete-work selection")
    }
    fn observe(&self, _: PlacementFeedback) -> Result<(), crate::RuntimeError> {
        unreachable!("host effect uses complete-work observation")
    }
    fn invalidate(&self, _: PlacementInvalidation) {}
}

fn request(output: HostEffectOutput) -> HostEffectRequest {
    HostEffectRequest {
        operation: "unzip",
        workload: HostEffectWorkload::ArchiveExtract {
            source: HostEffectSource::LocalArchive,
            output,
        },
        domain_facts: Digest::sha256(b"source-and-destination"),
    }
}

fn runtime(
    placement: Rc<Placement>,
    evidence: Option<HostEffectResourceEvidence>,
    available: Option<u64>,
) -> RuntimeContext {
    placement.revision_available.set(true);
    RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_placement(placement)
                .with_host_effect_resources(Rc::new(Resources(evidence)))
                .with_parallel(Rc::new(Parallel(available))),
        )
}

fn bounds(retained: ResourceByteBound, scratch: ResourceByteBound) -> HostEffectResourceEvidence {
    HostEffectResourceEvidence {
        authority_revision: Digest::sha256(b"typed-service-revision"),
        scratch_bytes: scratch,
        returned_value_bytes: retained,
    }
}

#[test]
fn finite_missing_or_unknown_bound_rejects_before_selection() {
    let placement = Rc::new(Placement::default());
    for evidence in [
        None,
        Some(bounds(
            ResourceByteBound::Known(4),
            ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoPeakMemoryBound),
        )),
        Some(bounds(
            ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoRetainedValueBound),
            ResourceByteBound::Known(4),
        )),
    ] {
        let runtime = runtime(placement.clone(), evidence, Some(32));
        let error = select_host_effect(Some(&runtime), request(HostEffectOutput::ReturnedValue))
            .err()
            .expect("finite unknown must reject");
        assert_eq!(error.code, "RunMat:Placement:InsufficientResourceEvidence");
    }
    assert_eq!(placement.selections.get(), 0);

    let runtime = runtime(
        placement.clone(),
        Some(bounds(
            ResourceByteBound::Known(4),
            ResourceByteBound::Known(4),
        )),
        Some(32),
    );
    placement.revision_available.set(false);
    let error = select_host_effect(Some(&runtime), request(HostEffectOutput::ReturnedValue))
        .err()
        .expect("finite missing revision must reject");
    assert_eq!(error.code, "RunMat:Placement:InsufficientResourceEvidence");
    assert_eq!(placement.selections.get(), 0);
}

#[test]
fn finite_effect_only_uses_zero_retained_but_still_budgets_scratch() {
    let placement = Rc::new(Placement::default());
    let evidence = Some(bounds(
        ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoRetainedValueBound),
        ResourceByteBound::Known(24),
    ));
    let enough = runtime(placement.clone(), evidence, Some(24));
    let receipt = select_host_effect(Some(&enough), request(HostEffectOutput::EffectOnly))
        .unwrap()
        .expect("effect-only call admitted");
    assert_eq!(placement.probes.load(Ordering::SeqCst), 1);
    receipt.finish(WholeWorkObservationOutcome::Success);
    assert_eq!(placement.probes.load(Ordering::SeqCst), 0);
    assert_eq!(placement.observations.borrow().len(), 1);
    assert!(placement.observations.borrow()[0].elapsed_ns > 0);
    let failed = select_host_effect(Some(&enough), request(HostEffectOutput::EffectOnly))
        .unwrap()
        .unwrap();
    failed.finish(WholeWorkObservationOutcome::OperationalFailure);
    assert_eq!(placement.observations.borrow().len(), 2);
    assert_eq!(
        placement.observations.borrow()[1].outcome,
        WholeWorkObservationOutcome::OperationalFailure
    );
    let insufficient = runtime(placement.clone(), evidence, Some(23));
    assert!(
        select_host_effect(Some(&insufficient), request(HostEffectOutput::EffectOnly)).is_err()
    );
    assert_eq!(placement.selections.get(), 2);
}

#[test]
fn finite_returned_value_charges_retained_and_scratch_together() {
    let placement = Rc::new(Placement::default());
    let evidence = Some(bounds(
        ResourceByteBound::Known(12),
        ResourceByteBound::Known(24),
    ));
    let enough = runtime(placement.clone(), evidence, Some(36));
    select_host_effect(Some(&enough), request(HostEffectOutput::ReturnedValue))
        .unwrap()
        .unwrap()
        .finish(WholeWorkObservationOutcome::Success);
    let short = runtime(placement.clone(), evidence, Some(35));
    assert!(select_host_effect(Some(&short), request(HostEffectOutput::ReturnedValue)).is_err());
    assert_eq!(placement.selections.get(), 1);
}

#[test]
fn dropped_and_cancelled_receipts_release_probe_without_failure_feedback() {
    let placement = Rc::new(Placement::default());
    let runtime = runtime(
        placement.clone(),
        Some(bounds(
            ResourceByteBound::Known(8),
            ResourceByteBound::Known(8),
        )),
        Some(16),
    );
    drop(select_host_effect(Some(&runtime), request(HostEffectOutput::ReturnedValue)).unwrap());
    assert_eq!(placement.probes.load(Ordering::SeqCst), 0);
    assert!(placement.observations.borrow().is_empty());

    let cancellation = CancellationSource::new();
    let cancelled = runtime.with_cancellation_token(cancellation.token());
    let receipt = select_host_effect(Some(&cancelled), request(HostEffectOutput::ReturnedValue))
        .unwrap()
        .unwrap();
    cancellation.cancel(CancellationReason::User);
    receipt.finish(WholeWorkObservationOutcome::OperationalFailure);
    assert_eq!(placement.probes.load(Ordering::SeqCst), 0);
    assert!(placement.observations.borrow().is_empty());
}
