use super::*;
use std::sync::atomic::AtomicBool;

struct Reserved {
    live: Arc<AtomicBool>,
    drops: Arc<AtomicUsize>,
    bytes: u64,
}

impl HostEffectStageReservation for Reserved {
    fn reserved_bytes(&self) -> u64 {
        self.bytes
    }
    fn is_current(&self) -> bool {
        self.live.load(Ordering::SeqCst)
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
    live: Arc<AtomicBool>,
    drops: Arc<AtomicUsize>,
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
        (request.operation == "unzip" && Arc::ptr_eq(request.filesystem, fixture_fs())).then(|| {
            HostEffectStageAdmission {
                evidence: HostEffectStageEvidence {
                    authority_revision: self.revision,
                    scratch_bytes: self.stage_scratch,
                },
                reservation: Arc::new(Reserved {
                    live: self.live.clone(),
                    drops: self.drops.clone(),
                    bytes: self.reserved,
                }),
            }
        })
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
    let resources = Rc::new(StagedResources {
        revision: Digest::sha256(b"staged-resource-v1"),
        stage_scratch,
        final_scratch,
        reserved,
        live: live.clone(),
        drops: drops.clone(),
    });
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_placement(placement.clone())
                .with_host_effect_resources(resources)
                .with_parallel(Rc::new(Parallel(Some(budget)))),
        );
    (runtime, placement, live, drops)
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
    let receipt =
        select_host_effect_after_stage(&runtime, request(HostEffectOutput::ReturnedValue), stage)
            .unwrap();
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
        assert_eq!(drops.load(Ordering::SeqCst), 1);
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
    let error =
        select_host_effect_after_stage(&runtime, request(HostEffectOutput::ReturnedValue), stage)
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
    let error =
        select_host_effect_after_stage(&runtime, request(HostEffectOutput::ReturnedValue), stage)
            .err()
            .expect("cancelled scan cannot select");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(placement.selections.get(), 0);
    assert_eq!(placement.probes.load(Ordering::SeqCst), 0);
    assert!(placement.observations.borrow().is_empty());
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn final_call_must_match_original_owner_identity_and_peak() {
    let (runtime, placement, _, drops) = staged_runtime(
        ResourceByteBound::Known(16),
        ResourceByteBound::Known(8),
        16,
        64,
    );
    let stage =
        admit_host_effect_stage(&runtime, request(HostEffectOutput::ReturnedValue)).unwrap();
    let error =
        select_host_effect_after_stage(&runtime, request(HostEffectOutput::ReturnedValue), stage)
            .err()
            .expect("final route omits staged peak");
    assert_eq!(error.identifier(), Some("RunMat:Placement:InvalidStage"));
    assert_eq!(placement.selections.get(), 0);
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
    let error = select_host_effect_after_stage(&runtime, other, stage)
        .err()
        .expect("foreign filesystem is not the opened source owner");
    assert_eq!(error.identifier(), Some("RunMat:Placement:InvalidStage"));
    assert_eq!(placement.selections.get(), 0);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}
