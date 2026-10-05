use super::*;
use runmat_value::ExactHostAllocationPermit;

struct ManagedResources {
    ledger: ManagedHostEffectLedger,
    revision: Digest,
}

impl RuntimeHostEffectResourceService for ManagedResources {
    fn admission(
        &self,
        _: &HostEffectRequest<'_>,
        _: &RuntimeContext,
    ) -> Option<HostEffectResourceAdmission> {
        None
    }

    fn stage_admission(
        &self,
        request: &HostEffectRequest<'_>,
        _: &RuntimeContext,
    ) -> Option<HostEffectStageAdmission> {
        if request.operation != "unzip" || !Arc::ptr_eq(request.filesystem, fixture_fs()) {
            return None;
        }
        Some(HostEffectStageAdmission {
            evidence: HostEffectStageEvidence {
                authority_revision: self.revision,
                scratch_bytes: ResourceByteBound::Known(8),
            },
            reservation: self.ledger.reserve_stage(8).ok()?,
        })
    }

    fn final_admission_after_stage(
        &self,
        request: &HostEffectRequest<'_>,
        facts: HostEffectFinalFacts<'_>,
        _: &RuntimeContext,
    ) -> Option<HostEffectResourceAdmission> {
        (request.operation == "unzip"
            && Arc::ptr_eq(request.filesystem, fixture_fs())
            && matches!(facts, HostEffectFinalFacts::TestOnly(_)))
        .then(|| {
            test_admission(HostEffectResourceEvidence {
                authority_revision: self.revision,
                scratch_bytes: ResourceByteBound::Known(8),
                returned_value_bytes: ResourceByteBound::Known(4),
            })
        })
    }
}

fn managed_runtime(
    ledger: &ManagedHostEffectLedger,
    capacity: u64,
) -> (RuntimeContext, Rc<Placement>) {
    let placement = Rc::new(Placement::default());
    placement.revision_available.set(true);
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_placement(placement.clone())
                .with_host_effect_resources(Rc::new(ManagedResources {
                    ledger: ledger.clone(),
                    revision: Digest::sha256(b"managed-stage-v1"),
                }))
                .with_parallel(Rc::new(Parallel(Some(capacity)))),
        );
    (runtime, placement)
}

fn selected_receipt(runtime: &RuntimeContext) -> HostEffectReceipt {
    let stage = admit_host_effect_stage(runtime, request(HostEffectOutput::ReturnedValue))
        .expect("precharged stage");
    select_host_effect_after_stage(
        runtime,
        request(HostEffectOutput::ReturnedValue),
        HostEffectFinalFacts::TestOnly(Digest::sha256(b"source-and-destination")),
        stage,
    )
    .expect("one selected call")
}

#[test]
fn managed_claims_retain_physical_output_and_one_stage_envelope_across_receipt() {
    let ledger = ManagedHostEffectLedger::new(12);
    let (runtime, placement) = managed_runtime(&ledger, 12);
    let stage = admit_host_effect_stage(&runtime, request(HostEffectOutput::ReturnedValue))
        .expect("precharged stage");
    let scan = stage
        .reserve_live_exact(HostEffectLiveAllocationKind::SourceStaging, 4)
        .expect("source allocation claim");
    let receipt = select_host_effect_after_stage(
        &runtime,
        request(HostEffectOutput::ReturnedValue),
        HostEffectFinalFacts::TestOnly(Digest::sha256(b"source-and-destination")),
        stage,
    )
    .expect("one selected call");
    let parsed = receipt
        .reserve_live_exact(HostEffectLiveAllocationKind::ParsedMetadata, 4)
        .expect("post-header temporary claim");
    let permit = receipt
        .output_allocation_permit()
        .expect("staged output permit");
    let output = runmat_value::ExactHostNumericAllocation::new(
        runmat_value::NumericDType::F32,
        1,
        permit.as_ref(),
    )
    .expect("physical four-byte output")
    .into_host_buffer();
    assert_eq!(ledger.snapshot().live_reserved_bytes, 12);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 12);
    assert_eq!(ledger.snapshot().peak_claimed_bytes, 12);
    receipt.finish(WholeWorkObservationOutcome::Success);
    drop(permit);
    assert_eq!(placement.observations.borrow().len(), 1);
    drop(scan);
    drop(parsed);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 4);
    drop(output);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[test]
fn simultaneous_receipts_compete_for_retained_output_capacity() {
    let ledger = ManagedHostEffectLedger::new(20);
    let (runtime, placement) = managed_runtime(&ledger, 20);
    let first = selected_receipt(&runtime);
    let first_output = first
        .output_allocation_permit()
        .expect("first output permit")
        .reserve_exact(4)
        .expect("first retained output");
    let second = selected_receipt(&runtime);
    assert!(matches!(
        second
            .output_allocation_permit()
            .expect("second output permit")
            .reserve_exact(4),
        Err(runmat_value::ExactHostNumericAllocationError::ExceedsPermit)
    ));
    assert_eq!(ledger.snapshot().live_reserved_bytes, 20);
    first.finish(WholeWorkObservationOutcome::Success);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 12);
    let second_output = second
        .output_allocation_permit()
        .expect("same second output permit")
        .reserve_exact(4)
        .expect("capacity returned after first stage");
    assert_eq!(ledger.snapshot().live_reserved_bytes, 16);
    second.finish(WholeWorkObservationOutcome::Success);
    drop(first_output);
    drop(second_output);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    assert_eq!(placement.observations.borrow().len(), 2);
}
