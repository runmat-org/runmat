use super::*;
use crate::RuntimeError;
use std::sync::atomic::AtomicBool;

struct Reserved {
    live: Arc<AtomicBool>,
    drops: Arc<AtomicUsize>,
    ledger: ManagedHostEffectLedger,
    inner: Arc<ManagedHostEffectStageReservation>,
}

impl HostEffectStageReservation for Reserved {
    fn managed_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.ledger.clone())
    }
    fn reserved_bytes(&self) -> u64 {
        self.inner.reserved_bytes()
    }
    fn capacity_bytes(&self) -> u64 {
        self.inner.capacity_bytes()
    }
    fn promote_scan_envelope(&self, bytes: u64) -> Result<(), HostEffectAllocationError> {
        self.inner.promote_scan_envelope(bytes)
    }
    fn reserve_live_exact(
        &self,
        kind: HostEffectLiveAllocationKind,
        bytes: usize,
    ) -> Result<Arc<dyn HostEffectLiveAllocationReservation>, HostEffectAllocationError> {
        self.inner.reserve_live_exact(kind, bytes)
    }
    fn reserve_output_headroom(
        &self,
        bytes: u64,
    ) -> Result<Arc<dyn HostEffectOutputHeadroom>, HostEffectAllocationError> {
        self.inner.reserve_output_headroom(bytes)
    }
    fn is_current(&self) -> bool {
        self.live.load(Ordering::SeqCst) && self.inner.is_current()
    }
}

impl Drop for Reserved {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

struct StagedResources {
    revision: Digest,
    stage_scratch: ResourceByteBound,
    final_scratch: ResourceByteBound,
    reserved: u64,
    separate_final_permit: bool,
    live: Arc<AtomicBool>,
    drops: Arc<AtomicUsize>,
    ledger: ManagedHostEffectLedger,
}

struct ManagedParallel(ManagedHostEffectLedger);

impl RuntimeParallelService for ManagedParallel {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }

    fn placement_resources(&self) -> RuntimeParallelResources {
        let snapshot = self.0.snapshot();
        RuntimeParallelResources {
            memory_available_bytes: Some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
            ..RuntimeParallelResources::default()
        }
    }

    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}

impl RuntimeHostEffectResourceService for StagedResources {
    fn admission(
        &self,
        request: &HostEffectRequest<'_>,
        _: &RuntimeContext,
    ) -> Option<HostEffectResourceAdmission> {
        (request.operation == "unzip" && Arc::ptr_eq(request.filesystem, fixture_fs())).then(|| {
            test_admission(HostEffectResourceEvidence {
                authority_revision: self.revision,
                scratch_bytes: self.final_scratch,
                returned_value_bytes: ResourceByteBound::Known(8),
            })
        })
    }

    fn stage_admission(
        &self,
        request: &HostEffectRequest<'_>,
        _: &RuntimeContext,
    ) -> Option<HostEffectStageAdmission> {
        (request.operation == "unzip" && Arc::ptr_eq(request.filesystem, fixture_fs()))
            .then(|| {
                let inner = self.ledger.reserve_stage(self.reserved).ok()?;
                Some(HostEffectStageAdmission {
                    evidence: HostEffectStageEvidence {
                        authority_revision: self.revision,
                        scratch_bytes: self.stage_scratch,
                    },
                    reservation: Arc::new(Reserved {
                        live: self.live.clone(),
                        drops: self.drops.clone(),
                        ledger: self.ledger.clone(),
                        inner,
                    }),
                })
            })
            .flatten()
    }

    fn final_admission_after_stage(
        &self,
        request: &HostEffectRequest<'_>,
        facts: HostEffectFinalFacts<'_>,
        runtime: &RuntimeContext,
    ) -> Option<HostEffectResourceAdmission> {
        let mut admission = matches!(facts, HostEffectFinalFacts::TestOnly(_))
            .then(|| self.admission(request, runtime))
            .flatten()?;
        if self.separate_final_permit {
            admission.output_allocation_permit = Some(Arc::new(SeparatePermit));
        }
        Some(admission)
    }
}

#[derive(Debug)]
struct SeparatePermit;
impl runmat_value::ExactHostAllocationPermit for SeparatePermit {
    fn reserve_exact(
        &self,
        _: usize,
    ) -> Result<
        Arc<dyn runmat_value::ExactHostAllocationReservation>,
        runmat_value::ExactHostNumericAllocationError,
    > {
        Err(runmat_value::ExactHostNumericAllocationError::ExceedsPermit)
    }
}

fn staged_runtime(
    stage_scratch: ResourceByteBound,
    final_scratch: ResourceByteBound,
    reserved: u64,
    budget: u64,
) -> (
    RuntimeContext,
    Rc<Placement>,
    Arc<AtomicBool>,
    Arc<AtomicUsize>,
) {
    let placement = Rc::new(Placement::default());
    placement.revision_available.set(true);
    let live = Arc::new(AtomicBool::new(true));
    let drops = Arc::new(AtomicUsize::new(0));
    let ledger = ManagedHostEffectLedger::new(budget);
    *placement.managed_ledger.borrow_mut() = Some(ledger.clone());
    let resources = Rc::new(StagedResources {
        revision: Digest::sha256(b"staged-resource-v1"),
        stage_scratch,
        final_scratch,
        reserved,
        separate_final_permit: false,
        live: live.clone(),
        drops: drops.clone(),
        ledger: ledger.clone(),
    });
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_placement(placement.clone())
                .with_host_effect_resources(resources)
                .with_parallel(Rc::new(ManagedParallel(ledger))),
        );
    (runtime, placement, live, drops)
}

#[path = "stage/managed.rs"]
mod managed;

struct OwnedRevisionOnly;

impl RuntimePlacementService for OwnedRevisionOnly {
    fn plan(
        &self,
        _: runmat_execution::PlacementPlanRequest,
    ) -> Result<runmat_execution::PlacementDecision, RuntimeError> {
        panic!("missing borrowed revision must reject before planning")
    }

    fn observe(&self, _: runmat_execution::PlacementFeedback) -> Result<(), RuntimeError> {
        panic!("rejected stage must not publish placement feedback")
    }

    fn invalidate(&self, _: runmat_execution::PlacementInvalidation) {}

    fn whole_work_host_service_revision(
        &self,
        program: Option<runmat_execution::ProgramRevision>,
        host_service: Digest,
    ) -> Option<PlacementRevision> {
        let digest = Digest::sha256(b"owned-only-placement");
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
}

#[test]
fn finite_stage_requires_borrowed_revision_and_retires_offered_reservation() {
    let (runtime, placement, _live, drops) = staged_runtime(
        ResourceByteBound::Known(16),
        ResourceByteBound::Known(16),
        16,
        64,
    );
    let owned_only = runtime.clone().with_service_ports(
        runtime
            .service_ports()
            .clone()
            .with_placement(Rc::new(OwnedRevisionOnly)),
    );
    let error = match admit_host_effect_stage(&owned_only, request(HostEffectOutput::ReturnedValue))
    {
        Ok(_) => panic!("owned-only placement cannot certify a finite staged call"),
        Err(error) => error,
    };
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:InsufficientResourceEvidence")
    );
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(placement.selections.get(), 0);
}

#[test]
fn stage_reserves_before_effect_and_final_selection_owns_one_probe_and_observation() {
    let (runtime, placement, _live, drops) = staged_runtime(
        ResourceByteBound::Known(16),
        ResourceByteBound::Known(16),
        16,
        64,
    );
    let stage =
        admit_host_effect_stage(&runtime, request(HostEffectOutput::ReturnedValue)).unwrap();
    assert!(Arc::ptr_eq(stage.filesystem(), fixture_fs()));
    assert_eq!(placement.selections.get(), 0);
    assert_eq!(placement.probes.load(Ordering::SeqCst), 0);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    stage.checkpoint().unwrap();
    let stage_epoch = stage.active_epoch();
    let receipt = select_host_effect_after_stage(
        &runtime,
        request(HostEffectOutput::ReturnedValue),
        HostEffectFinalFacts::TestOnly(Digest::sha256(b"source-and-destination")),
        stage,
    )
    .unwrap();
    assert!(Arc::ptr_eq(&stage_epoch, &receipt.active_epoch));
    assert_eq!(placement.selections.get(), 1);
    assert_eq!(placement.probes.load(Ordering::SeqCst), 1);
    receipt.finish(WholeWorkObservationOutcome::Success);
    assert_eq!(placement.observations.borrow().len(), 1);
    assert_eq!(placement.probes.load(Ordering::SeqCst), 0);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn missing_or_excess_scan_reservation_rejects_before_placement() {
    for (scratch, reserved) in [
        (
            ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound),
            16,
        ),
        (ResourceByteBound::Known(32), 16),
        (ResourceByteBound::Known(16), 65),
    ] {
        let (runtime, placement, _, drops) =
            staged_runtime(scratch, ResourceByteBound::Known(32), reserved, 64);
        let error = admit_host_effect_stage(&runtime, request(HostEffectOutput::ReturnedValue))
            .err()
            .expect("incomplete stage cannot authorize source open");
        assert_eq!(
            error.identifier(),
            Some("RunMat:Placement:InsufficientResourceEvidence")
        );
        assert_eq!(placement.selections.get(), 0);
        assert_eq!(drops.load(Ordering::SeqCst), usize::from(reserved <= 64));
    }
}

#[test]
fn revoked_or_abandoned_stage_never_selects_or_observes() {
    let (runtime, placement, live, drops) = staged_runtime(
        ResourceByteBound::Known(16),
        ResourceByteBound::Known(16),
        16,
        64,
    );
    let stage =
        admit_host_effect_stage(&runtime, request(HostEffectOutput::ReturnedValue)).unwrap();
    live.store(false, Ordering::SeqCst);
    let error = select_host_effect_after_stage(
        &runtime,
        request(HostEffectOutput::ReturnedValue),
        HostEffectFinalFacts::TestOnly(Digest::sha256(b"source-and-destination")),
        stage,
    )
    .err()
    .expect("revoked stage");
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:StaleResourceAuthority")
    );
    assert_eq!(placement.selections.get(), 0);
    assert!(placement.observations.borrow().is_empty());
    assert_eq!(drops.load(Ordering::SeqCst), 1);

    live.store(true, Ordering::SeqCst);
    let abandoned =
        admit_host_effect_stage(&runtime, request(HostEffectOutput::ReturnedValue)).unwrap();
    drop(abandoned);
    assert_eq!(drops.load(Ordering::SeqCst), 2);
    assert_eq!(placement.probes.load(Ordering::SeqCst), 0);
}

#[test]
fn cancellation_after_scan_releases_reservation_without_a_probe() {
    let (runtime, placement, _, drops) = staged_runtime(
        ResourceByteBound::Known(16),
        ResourceByteBound::Known(16),
        16,
        64,
    );
    let cancellation = CancellationSource::new();
    let runtime = runtime.with_cancellation_token(cancellation.token());
    let stage =
        admit_host_effect_stage(&runtime, request(HostEffectOutput::ReturnedValue)).unwrap();
    cancellation.cancel(CancellationReason::User);
    let error = select_host_effect_after_stage(
        &runtime,
        request(HostEffectOutput::ReturnedValue),
        HostEffectFinalFacts::TestOnly(Digest::sha256(b"source-and-destination")),
        stage,
    )
    .err()
    .expect("cancelled scan cannot select");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(placement.selections.get(), 0);
    assert_eq!(placement.probes.load(Ordering::SeqCst), 0);
    assert!(placement.observations.borrow().is_empty());
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn final_smaller_scratch_still_retains_original_stage_precharge() {
    let (runtime, placement, _, drops) = staged_runtime(
        ResourceByteBound::Known(16),
        ResourceByteBound::Known(8),
        16,
        64,
    );
    let stage =
        admit_host_effect_stage(&runtime, request(HostEffectOutput::ReturnedValue)).unwrap();
    let receipt = select_host_effect_after_stage(
        &runtime,
        request(HostEffectOutput::ReturnedValue),
        HostEffectFinalFacts::TestOnly(Digest::sha256(b"source-and-destination")),
        stage,
    )
    .expect("final smaller bound does not release live stage precharge");
    assert_eq!(placement.selections.get(), 1);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    receipt.finish(WholeWorkObservationOutcome::Success);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn stage_cannot_be_replayed_on_another_filesystem_owner() {
    let (runtime, placement, _, drops) = staged_runtime(
        ResourceByteBound::Known(16),
        ResourceByteBound::Known(16),
        16,
        64,
    );
    let stage =
        admit_host_effect_stage(&runtime, request(HostEffectOutput::ReturnedValue)).unwrap();
    let foreign: Arc<dyn runmat_filesystem::FsProvider> =
        Arc::new(runmat_filesystem::MemoryFsProvider::new());
    let original = request(HostEffectOutput::ReturnedValue);
    let other = HostEffectRequest {
        operation: original.operation,
        workload: original.workload,
        domain_facts: original.domain_facts,
        filesystem: &foreign,
    };
    let error = select_host_effect_after_stage(
        &runtime,
        other,
        HostEffectFinalFacts::TestOnly(Digest::sha256(b"source-and-destination")),
        stage,
    )
    .err()
    .expect("foreign filesystem is not the opened source owner");
    assert_eq!(error.identifier(), Some("RunMat:Placement:InvalidStage"));
    assert_eq!(placement.selections.get(), 0);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn stage_cannot_be_selected_from_another_runtime_session() {
    let (runtime, placement, _, drops) = staged_runtime(
        ResourceByteBound::Known(16),
        ResourceByteBound::Known(16),
        16,
        64,
    );
    let stage =
        admit_host_effect_stage(&runtime, request(HostEffectOutput::ReturnedValue)).unwrap();
    let other = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(runtime.service_ports().clone());
    let error = select_host_effect_after_stage(
        &other,
        request(HostEffectOutput::ReturnedValue),
        HostEffectFinalFacts::TestOnly(Digest::sha256(b"source-and-destination")),
        stage,
    )
    .err()
    .expect("foreign session cannot consume staged source");
    assert_eq!(error.identifier(), Some("RunMat:Placement:InvalidStage"));
    assert_eq!(placement.selections.get(), 0);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn staged_receipt_suppresses_feedback_when_scan_reservation_is_revoked() {
    let (runtime, placement, live, drops) = staged_runtime(
        ResourceByteBound::Known(16),
        ResourceByteBound::Known(16),
        16,
        64,
    );
    let stage =
        admit_host_effect_stage(&runtime, request(HostEffectOutput::ReturnedValue)).unwrap();
    let receipt = select_host_effect_after_stage(
        &runtime,
        request(HostEffectOutput::ReturnedValue),
        HostEffectFinalFacts::TestOnly(Digest::sha256(b"source-and-destination")),
        stage,
    )
    .unwrap();
    assert_eq!(placement.selections.get(), 1);
    live.store(false, Ordering::SeqCst);
    assert_eq!(
        receipt.checkpoint().unwrap_err().identifier(),
        Some("RunMat:Placement:StaleResourceAuthority")
    );
    receipt.finish(WholeWorkObservationOutcome::Success);
    assert!(placement.observations.borrow().is_empty());
    assert_eq!(placement.probes.load(Ordering::SeqCst), 0);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn staged_final_rejects_a_second_output_quota_authority() {
    let (runtime, placement, _, _) = staged_runtime(
        ResourceByteBound::Known(16),
        ResourceByteBound::Known(16),
        16,
        64,
    );
    let drops = Arc::new(AtomicUsize::new(0));
    let resources = Rc::new(StagedResources {
        revision: Digest::sha256(b"staged-resource-v1"),
        stage_scratch: ResourceByteBound::Known(16),
        final_scratch: ResourceByteBound::Known(16),
        reserved: 16,
        separate_final_permit: true,
        live: Arc::new(AtomicBool::new(true)),
        drops: drops.clone(),
        ledger: runtime
            .service_ports()
            .parallel()
            .and_then(|owner| owner.managed_host_effect_ledger())
            .expect("original managed budget"),
    });
    let ports = runtime
        .service_ports()
        .clone()
        .with_host_effect_resources(resources);
    let runtime = runtime.with_service_ports(ports);
    let stage =
        admit_host_effect_stage(&runtime, request(HostEffectOutput::ReturnedValue)).unwrap();
    let error = select_host_effect_after_stage(
        &runtime,
        request(HostEffectOutput::ReturnedValue),
        HostEffectFinalFacts::TestOnly(Digest::sha256(b"source-and-destination")),
        stage,
    )
    .err()
    .expect("independent output authority is contradictory");
    assert_eq!(error.identifier(), Some("RunMat:Placement:InvalidStage"));
    assert_eq!(placement.selections.get(), 0);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn same_session_cannot_rebind_placement_parallel_or_cancellation_authority() {
    for replacement in 0..4 {
        let (runtime, placement, live, drops) = staged_runtime(
            ResourceByteBound::Known(16),
            ResourceByteBound::Known(16),
            16,
            64,
        );
        let stage =
            admit_host_effect_stage(&runtime, request(HostEffectOutput::ReturnedValue)).unwrap();
        let changed = match replacement {
            0 => {
                let other = Rc::new(Placement::default());
                other.revision_available.set(true);
                runtime
                    .clone()
                    .with_service_ports(runtime.service_ports().clone().with_placement(other))
            }
            1 => runtime.clone().with_service_ports(
                runtime
                    .service_ports()
                    .clone()
                    .with_parallel(Rc::new(Parallel(Some(64)))),
            ),
            2 => runtime.clone().with_service_ports(
                runtime
                    .service_ports()
                    .clone()
                    .with_host_effect_resources(Rc::new(StagedResources {
                        revision: Digest::sha256(b"staged-resource-v1"),
                        stage_scratch: ResourceByteBound::Known(16),
                        final_scratch: ResourceByteBound::Known(16),
                        reserved: 16,
                        separate_final_permit: false,
                        live,
                        drops: Arc::new(AtomicUsize::new(0)),
                        ledger: runtime
                            .service_ports()
                            .parallel()
                            .and_then(|owner| owner.managed_host_effect_ledger())
                            .expect("original managed budget"),
                    })),
            ),
            _ => runtime
                .clone()
                .with_cancellation_token(CancellationSource::new().token()),
        };
        assert_eq!(runtime.local_identity(), changed.local_identity());
        let error = select_host_effect_after_stage(
            &changed,
            request(HostEffectOutput::ReturnedValue),
            HostEffectFinalFacts::TestOnly(Digest::sha256(b"source-and-destination")),
            stage,
        )
        .err()
        .expect("rebound authority cannot consume original scan");
        assert_eq!(error.identifier(), Some("RunMat:Placement:InvalidStage"));
        assert_eq!(placement.selections.get(), 0);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}
