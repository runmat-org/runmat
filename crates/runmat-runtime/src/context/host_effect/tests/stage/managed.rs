use super::*;
use crate::context::host_allocation::HostAllocationPlan;
use std::error::Error;

struct ManagedResources {
    ledger: ManagedHostEffectLedger,
    revision: Digest,
    returned_value_bytes: u64,
}

struct PooledResources {
    pool: ManagedHostEffectStagePool,
    revision: Digest,
}

impl RuntimeHostEffectResourceService for PooledResources {
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
            reservation: self.pool.acquire(8).ok()?.into_reservation(),
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
            let value_bytes =
                HostAllocationPlan::retained_tensor_bytes(runmat_value::NumericDType::U64, &[1, 1])
                    .expect("known exact output layout and claims");
            test_admission(HostEffectResourceEvidence {
                authority_revision: self.revision,
                scratch_bytes: ResourceByteBound::Known(8),
                returned_value_bytes: ResourceByteBound::Known(value_bytes),
            })
        })
    }
}

fn pooled_runtime(ledger: &ManagedHostEffectLedger) -> RuntimeContext {
    let placement = Rc::new(Placement::default());
    placement.revision_available.set(true);
    *placement.managed_ledger.borrow_mut() = Some(ledger.clone());
    RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_placement(placement)
                .with_host_effect_resources(Rc::new(PooledResources {
                    pool: ManagedHostEffectStagePool::new(ledger, 1).unwrap(),
                    revision: Digest::sha256(b"pooled-stage-v1"),
                }))
                .with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
        )
}

#[test]
fn pooled_stage_admission_releases_only_its_session_slot() {
    let ledger = ManagedHostEffectLedger::new(16);
    let first = pooled_runtime(&ledger);
    let second = pooled_runtime(&ledger);
    let first_stage = admit_host_effect_stage(&first, request(HostEffectOutput::ReturnedValue))
        .expect("first session admission");
    let second_stage = admit_host_effect_stage(&second, request(HostEffectOutput::ReturnedValue))
        .expect("second session admission");
    assert_eq!(ledger.snapshot().live_reserved_bytes, 16);
    drop(first_stage);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 8);
    second_stage
        .checkpoint()
        .expect("second owner remains current");
    let reused = admit_host_effect_stage(&first, request(HostEffectOutput::ReturnedValue))
        .expect("first owner reuses its retired slot");
    assert_eq!(ledger.snapshot().live_reserved_bytes, 16);
    drop(reused);
    drop(second_stage);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn pooled_receipt_charges_exact_output_claim_until_value_drop() {
    let ledger = ManagedHostEffectLedger::new(256);
    let runtime = pooled_runtime(&ledger);
    let stage = admit_host_effect_stage(&runtime, request(HostEffectOutput::ReturnedValue))
        .expect("same-ledger stage");
    let receipt = select_host_effect_after_stage(
        &runtime,
        request(HostEffectOutput::ReturnedValue),
        HostEffectFinalFacts::TestOnly(Digest::sha256(b"source-and-destination")),
        stage,
    )
    .expect("one selected complete call");
    let permit = receipt
        .output_allocation_permit()
        .expect("pooled output owner");
    let value_bytes =
        HostAllocationPlan::retained_tensor_bytes(runmat_value::NumericDType::U64, &[1, 1])
            .unwrap();
    assert_eq!(ledger.snapshot().live_reserved_bytes, 8 + value_bytes);
    let tensor = runmat_value::ExactHostTensor2dAllocation::new(
        runmat_value::NumericDType::U64,
        1,
        1,
        permit.as_ref(),
    )
    .unwrap()
    .into_tensor()
    .unwrap();
    assert_eq!(ledger.snapshot().live_claimed_bytes, value_bytes);
    receipt.finish(WholeWorkObservationOutcome::Success);
    assert_eq!(ledger.snapshot().live_reserved_bytes, value_bytes);
    assert!(matches!(
        runmat_value::ExactHostTensor2dAllocation::new(
            runmat_value::NumericDType::U64,
            1,
            1,
            permit.as_ref(),
        ),
        Err(runmat_value::ExactHostNumericAllocationError::ExceedsPermit)
    ));
    drop(permit);
    assert_eq!(ledger.snapshot().live_reserved_bytes, value_bytes);
    drop(tensor);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    admit_host_effect_stage(&runtime, request(HostEffectOutput::ReturnedValue))
        .expect("pool slot is reusable after final output owner");
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
                returned_value_bytes: ResourceByteBound::Known(self.returned_value_bytes),
            })
        })
    }
}

fn managed_runtime(
    ledger: &ManagedHostEffectLedger,
    capacity: u64,
) -> (RuntimeContext, Rc<Placement>) {
    managed_runtime_with_output(ledger, capacity, 4)
}

fn managed_runtime_with_output(
    ledger: &ManagedHostEffectLedger,
    capacity: u64,
    returned_value_bytes: u64,
) -> (RuntimeContext, Rc<Placement>) {
    assert_eq!(ledger.snapshot().capacity_bytes, capacity);
    let placement = Rc::new(Placement::default());
    placement.revision_available.set(true);
    *placement.managed_ledger.borrow_mut() = Some(ledger.clone());
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_placement(placement.clone())
                .with_host_effect_resources(Rc::new(ManagedResources {
                    ledger: ledger.clone(),
                    revision: Digest::sha256(b"managed-stage-v1"),
                    returned_value_bytes,
                }))
                .with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
        );
    (runtime, placement)
}

#[test]
fn direct_only_finite_service_may_decline_stage_without_a_managed_owner() {
    let placement = Rc::new(Placement::default());
    let runtime = runtime_context(
        placement.clone(),
        Some(bounds(
            ResourceByteBound::Known(4),
            ResourceByteBound::Known(8),
        )),
        Some(12),
    );
    assert!(
        try_admit_host_effect_stage(&runtime, request(HostEffectOutput::ReturnedValue))
            .expect("direct-only service may decline staging")
            .is_none()
    );
    let receipt = select_host_effect(Some(&runtime), request(HostEffectOutput::ReturnedValue))
        .expect("direct service's complete evidence is still selected")
        .expect("one complete direct route");
    assert_eq!(placement.selections.get(), 1);
    receipt.finish(WholeWorkObservationOutcome::Success);
}

#[test]
fn offered_stage_without_parallel_managed_owner_fails_before_selection() {
    let stage_ledger = ManagedHostEffectLedger::new(12);
    let placement = Rc::new(Placement::default());
    placement.revision_available.set(true);
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_placement(placement.clone())
                .with_parallel(Rc::new(Parallel(Some(12))))
                .with_host_effect_resources(Rc::new(ManagedResources {
                    ledger: stage_ledger.clone(),
                    revision: Digest::sha256(b"managed-stage-v1"),
                    returned_value_bytes: 4,
                })),
        );
    let error = try_admit_host_effect_stage(&runtime, request(HostEffectOutput::ReturnedValue))
        .err()
        .expect("offered stage requires the same installed managed owner");
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:InsufficientResourceEvidence")
    );
    assert_eq!(placement.selections.get(), 0);
    assert_eq!(stage_ledger.snapshot().live_reserved_bytes, 0);
}

#[test]
fn foreign_stage_ledger_rejects_before_selection_and_releases_its_charge() {
    let parallel_ledger = ManagedHostEffectLedger::new(12);
    let stage_ledger = ManagedHostEffectLedger::new(12);
    let placement = Rc::new(Placement::default());
    placement.revision_available.set(true);
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_placement(placement.clone())
                .with_parallel(Rc::new(ManagedParallel(parallel_ledger.clone())))
                .with_host_effect_resources(Rc::new(ManagedResources {
                    ledger: stage_ledger.clone(),
                    revision: Digest::sha256(b"managed-stage-v1"),
                    returned_value_bytes: 4,
                })),
        );
    let error = admit_host_effect_stage(&runtime, request(HostEffectOutput::ReturnedValue))
        .err()
        .expect("a stage charged to a different managed owner must fail");
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:StaleResourceAuthority")
    );
    assert_eq!(placement.selections.get(), 0);
    assert_eq!(stage_ledger.snapshot().live_reserved_bytes, 0);
    assert_eq!(parallel_ledger.snapshot().live_reserved_bytes, 0);
}

struct OverreservedResources(ManagedHostEffectLedger);

impl RuntimeHostEffectResourceService for OverreservedResources {
    fn admission(
        &self,
        _: &HostEffectRequest<'_>,
        _: &RuntimeContext,
    ) -> Option<HostEffectResourceAdmission> {
        None
    }

    fn stage_admission(
        &self,
        _: &HostEffectRequest<'_>,
        _: &RuntimeContext,
    ) -> Option<HostEffectStageAdmission> {
        Some(HostEffectStageAdmission {
            evidence: HostEffectStageEvidence {
                authority_revision: Digest::sha256(b"overreserved-stage"),
                scratch_bytes: ResourceByteBound::Known(8),
            },
            reservation: self.0.reserve_stage(8).ok()?,
        })
    }

    fn final_admission_after_stage(
        &self,
        _: &HostEffectRequest<'_>,
        _: HostEffectFinalFacts<'_>,
        _: &RuntimeContext,
    ) -> Option<HostEffectResourceAdmission> {
        Some(test_admission(HostEffectResourceEvidence {
            authority_revision: Digest::sha256(b"overreserved-stage"),
            scratch_bytes: ResourceByteBound::Known(4),
            returned_value_bytes: ResourceByteBound::Known(6),
        }))
    }
}

#[test]
fn placement_counts_live_precharge_when_final_scratch_bound_shrinks() {
    let ledger = ManagedHostEffectLedger::new(12);
    let placement = Rc::new(Placement::default());
    placement.revision_available.set(true);
    *placement.managed_ledger.borrow_mut() = Some(ledger.clone());
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_placement(placement.clone())
                .with_parallel(Rc::new(ManagedParallel(ledger.clone())))
                .with_host_effect_resources(Rc::new(OverreservedResources(ledger.clone()))),
        );
    let stage = admit_host_effect_stage(&runtime, request(HostEffectOutput::ReturnedValue))
        .expect("stage precharges eight bytes before effects");
    assert_eq!(ledger.snapshot().free_bytes(), 4);
    let error = select_host_effect_after_stage(
        &runtime,
        request(HostEffectOutput::ReturnedValue),
        HostEffectFinalFacts::TestOnly(Digest::sha256(b"source-and-destination")),
        stage,
    )
    .err()
    .expect("eight live scratch plus six output exceeds twelve");
    assert_eq!(error.identifier(), Some("RunMat:Placement:ResourceLimit"));
    assert_eq!(placement.selections.get(), 0);
    assert!(placement.observations.borrow().is_empty());
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
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
fn exact_byte_owner_uses_public_stage_permit_and_retains_charge_until_drop() {
    let ledger = ManagedHostEffectLedger::new(64);
    let (runtime, _) = managed_runtime(&ledger, 64);
    let stage = admit_host_effect_stage(&runtime, request(HostEffectOutput::ReturnedValue))
        .expect("precharged stage");
    let metadata = stage.live_claim_metadata_bytes();
    let total = 8 + metadata as u64;
    stage
        .promote_scan_envelope(total)
        .expect("claim metadata envelope");
    let view = stage.reservation();
    let mut bytes = stage
        .allocate_exact_bytes(HostEffectLiveAllocationKind::CodecWorkspace, 8)
        .expect("exact stage-owned bytes");
    bytes.append(b"12345678").expect("within admitted capacity");
    assert_eq!(bytes.as_bytes(), b"12345678");
    assert_eq!(ledger.snapshot().live_claimed_bytes, total);
    drop(stage);
    assert!(!view.is_current());
    assert!(matches!(
        view.reserve_live_exact(HostEffectLiveAllocationKind::CodecWorkspace, 1),
        Err(HostEffectAllocationError::InvalidInvocation)
    ));
    assert_eq!(ledger.snapshot().live_reserved_bytes, total);
    drop(bytes);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[test]
fn exact_byte_stage_admission_rejects_cancelled_and_revoked_authority() {
    let ledger = ManagedHostEffectLedger::new(64);
    let (runtime, _) = managed_runtime(&ledger, 64);
    let cancellation = CancellationSource::new();
    let runtime = runtime.with_cancellation_token(cancellation.token());
    let stage = admit_host_effect_stage(&runtime, request(HostEffectOutput::ReturnedValue))
        .expect("precharged stage");
    let metadata = stage.live_claim_metadata_bytes() as u64;
    let total = 8 + metadata;
    stage
        .promote_scan_envelope(total)
        .expect("claim metadata envelope");
    let bytes = stage
        .allocate_exact_bytes(HostEffectLiveAllocationKind::CodecWorkspace, 4)
        .expect("live exact bytes");
    cancellation.cancel(CancellationReason::User);
    assert!(matches!(
        stage.allocate_exact_bytes(HostEffectLiveAllocationKind::CodecWorkspace, 1),
        Err(HostEffectAllocationError::Cancelled)
    ));
    assert_eq!(ledger.snapshot().live_claimed_bytes, 4 + metadata);
    drop(stage);
    drop(bytes);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

    let revoked_ledger = ManagedHostEffectLedger::new(64);
    let (runtime, _) = managed_runtime(&revoked_ledger, 64);
    let stage = admit_host_effect_stage(&runtime, request(HostEffectOutput::ReturnedValue))
        .expect("second precharged stage");
    stage
        .promote_scan_envelope(total)
        .expect("claim metadata envelope");
    let bytes = stage
        .allocate_exact_bytes(HostEffectLiveAllocationKind::CodecWorkspace, 4)
        .expect("second live exact bytes");
    revoked_ledger.revoke();
    assert!(matches!(
        stage.allocate_exact_bytes(HostEffectLiveAllocationKind::CodecWorkspace, 1),
        Err(HostEffectAllocationError::Revoked)
    ));
    assert_eq!(revoked_ledger.snapshot().live_claimed_bytes, 4 + metadata);
    drop(stage);
    drop(bytes);
    assert_eq!(revoked_ledger.snapshot().live_reserved_bytes, 0);
}

#[test]
fn finished_receipt_closes_exact_byte_stage_view_without_releasing_live_buffer() {
    let ledger = ManagedHostEffectLedger::new(12);
    let (runtime, placement) = managed_runtime(&ledger, 12);
    let stage = admit_host_effect_stage(&runtime, request(HostEffectOutput::ReturnedValue))
        .expect("precharged stage");
    let view = stage.reservation();
    let claim = stage
        .reserve_live_exact(HostEffectLiveAllocationKind::CodecWorkspace, 4)
        .expect("legacy shared temporary claim");
    let bytes = super::super::super::ExactHostEffectBytes::new(
        4,
        super::super::super::HostEffectLiveAllocationClaim::Shared(claim),
    )
    .expect("live exact bytes");
    let receipt = select_host_effect_after_stage(
        &runtime,
        request(HostEffectOutput::ReturnedValue),
        HostEffectFinalFacts::TestOnly(Digest::sha256(b"source-and-destination")),
        stage,
    )
    .expect("selected call");
    receipt.finish(WholeWorkObservationOutcome::Success);
    assert!(matches!(
        view.reserve_live_exact(HostEffectLiveAllocationKind::CodecWorkspace, 1),
        Err(HostEffectAllocationError::InvalidInvocation)
    ));
    assert_eq!(placement.observations.borrow().len(), 1);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 4);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 8);
    drop(bytes);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[test]
fn selected_receipt_precharges_output_before_physical_allocation() {
    let ledger = ManagedHostEffectLedger::new(12);
    let (runtime, placement) = managed_runtime(&ledger, 12);
    let receipt = selected_receipt(&runtime);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 12);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert!(matches!(
        ledger.reserve_stage(1).err(),
        Some(HostEffectAllocationError::ExceedsCapacity)
    ));
    receipt.finish(WholeWorkObservationOutcome::Success);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    assert_eq!(placement.observations.borrow().len(), 1);
}

#[test]
fn shared_ledger_rebind_invalidates_only_issuing_session() {
    let ledger = ManagedHostEffectLedger::new(24);
    let (first_runtime, first_placement) = managed_runtime(&ledger, 24);
    let (second_runtime, second_placement) = managed_runtime(&ledger, 24);
    let first = admit_host_effect_stage(&first_runtime, request(HostEffectOutput::ReturnedValue))
        .expect("first session stage");
    let second = admit_host_effect_stage(&second_runtime, request(HostEffectOutput::ReturnedValue))
        .expect("second session stage");
    first_runtime.invalidate_host_effect_session_authority();
    assert_eq!(
        first.checkpoint().unwrap_err().identifier(),
        Some("RunMat:Placement:StaleResourceAuthority")
    );
    second.checkpoint().expect("other issuer remains current");
    assert!(ledger.snapshot().current);
    let receipt = select_host_effect_after_stage(
        &second_runtime,
        request(HostEffectOutput::ReturnedValue),
        HostEffectFinalFacts::TestOnly(Digest::sha256(b"source-and-destination")),
        second,
    )
    .expect("other issuer may select and precharge output");
    assert_eq!(first_placement.selections.get(), 0);
    assert_eq!(second_placement.selections.get(), 1);
    drop(first);
    receipt.finish(WholeWorkObservationOutcome::Success);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[test]
fn foreign_managed_observation_owner_rejects_selection_before_receipt() {
    let ledger = ManagedHostEffectLedger::new(12);
    let (runtime, placement) = managed_runtime(&ledger, 12);
    *placement.managed_ledger.borrow_mut() = Some(ManagedHostEffectLedger::new(12));
    let stage = admit_host_effect_stage(&runtime, request(HostEffectOutput::ReturnedValue))
        .expect("original owner stage");
    let error = select_host_effect_after_stage(
        &runtime,
        request(HostEffectOutput::ReturnedValue),
        HostEffectFinalFacts::TestOnly(Digest::sha256(b"source-and-destination")),
        stage,
    )
    .err()
    .expect("foreign observation cannot authorize an effect");
    assert_eq!(error.identifier(), Some("RunMat:Placement:InvalidDecision"));
    assert_eq!(placement.selections.get(), 1);
    assert!(placement.observations.borrow().is_empty());
    assert_eq!(placement.probes.load(Ordering::SeqCst), 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[test]
fn selected_headroom_capacity_failure_preserves_typed_resource_cause() {
    let ledger = ManagedHostEffectLedger::new(12);
    let (runtime, placement) = managed_runtime(&ledger, 12);
    let competing = Rc::new(RefCell::new(None));
    let retained = competing.clone();
    let concurrent_ledger = ledger.clone();
    *placement.on_select.borrow_mut() = Some(Box::new(move || {
        *retained.borrow_mut() = Some(
            concurrent_ledger
                .reserve_stage(4)
                .expect("other invocation claims free capacity during selection"),
        );
    }));
    let stage = admit_host_effect_stage(&runtime, request(HostEffectOutput::ReturnedValue))
        .expect("initial eight-byte stage");
    let error = select_host_effect_after_stage(
        &runtime,
        request(HostEffectOutput::ReturnedValue),
        HostEffectFinalFacts::TestOnly(Digest::sha256(b"source-and-destination")),
        stage,
    )
    .err()
    .expect("selected output headroom cannot race with other reservations");
    assert_eq!(error.identifier(), Some("RunMat:Placement:ResourceLimit"));
    assert!(matches!(
        error
            .source()
            .and_then(|source| source.downcast_ref::<HostEffectAllocationError>()),
        Some(HostEffectAllocationError::ExceedsCapacity)
    ));
    assert!(placement.observations.borrow().is_empty());
    assert_eq!(ledger.snapshot().live_reserved_bytes, 4);
    drop(competing.borrow_mut().take());
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[test]
fn revoked_selected_headroom_reports_stale_authority_with_cause() {
    let ledger = ManagedHostEffectLedger::new(12);
    let (runtime, placement) = managed_runtime(&ledger, 12);
    let concurrent_ledger = ledger.clone();
    *placement.on_select.borrow_mut() = Some(Box::new(move || concurrent_ledger.revoke()));
    let stage = admit_host_effect_stage(&runtime, request(HostEffectOutput::ReturnedValue))
        .expect("initial stage");
    let error = select_host_effect_after_stage(
        &runtime,
        request(HostEffectOutput::ReturnedValue),
        HostEffectFinalFacts::TestOnly(Digest::sha256(b"source-and-destination")),
        stage,
    )
    .err()
    .expect("revoked owner cannot reserve output");
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:StaleResourceAuthority")
    );
    assert!(matches!(
        error
            .source()
            .and_then(|source| source.downcast_ref::<HostEffectAllocationError>()),
        Some(HostEffectAllocationError::Revoked)
    ));
    assert!(placement.observations.borrow().is_empty());
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[test]
fn managed_claims_retain_physical_output_and_one_stage_envelope_across_receipt() {
    let physical_output_bytes = runmat_value::ExactHostNumericAllocation::requested_bytes(
        runmat_value::NumericDType::F32,
        1,
    )
    .expect("canonical exact float owner layout") as u64;
    // `new` obtains an owned exact claim. Its managed claim-control owner is
    // distinct from the numeric backing and remains live with that backing.
    let output_bytes =
        crate::context::host_allocation::HostAllocationPlan::retained_exact_output_bytes(
            physical_output_bytes,
            1,
        )
        .expect("one physical output and its claim-control owner");
    let capacity = 8 + output_bytes;
    let ledger = ManagedHostEffectLedger::new(capacity);
    let (runtime, placement) = managed_runtime_with_output(&ledger, capacity, output_bytes);
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
    assert_eq!(ledger.snapshot().live_reserved_bytes, capacity);
    assert_eq!(ledger.snapshot().live_claimed_bytes, capacity);
    assert_eq!(ledger.snapshot().peak_claimed_bytes, capacity);
    receipt.finish(WholeWorkObservationOutcome::Success);
    assert!(matches!(
        permit.reserve_exact(1),
        Err(runmat_value::ExactHostNumericAllocationError::ExceedsPermit)
    ));
    assert_eq!(placement.observations.borrow().len(), 1);
    drop(scan);
    drop(parsed);
    // A retained permission view must not pin the obsolete scan precharge.
    assert_eq!(ledger.snapshot().live_reserved_bytes, output_bytes);
    drop(output);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    drop(permit);
}

#[test]
fn abandoned_receipt_invalidates_retained_permit_and_stage_view() {
    let ledger = ManagedHostEffectLedger::new(12);
    let (runtime, placement) = managed_runtime(&ledger, 12);
    let stage = admit_host_effect_stage(&runtime, request(HostEffectOutput::ReturnedValue))
        .expect("precharged stage");
    let stage_view = stage.reservation();
    let receipt = select_host_effect_after_stage(
        &runtime,
        request(HostEffectOutput::ReturnedValue),
        HostEffectFinalFacts::TestOnly(Digest::sha256(b"source-and-destination")),
        stage,
    )
    .expect("one selected call");
    let permit = receipt.output_allocation_permit().expect("output permit");
    let retained = permit.reserve_exact(4).expect("retained output charge");
    assert_eq!(ledger.snapshot().live_reserved_bytes, 12);
    drop(receipt);
    assert!(!stage_view.is_current());
    assert!(matches!(
        stage_view.reserve_live_exact(HostEffectLiveAllocationKind::CodecWorkspace, 1),
        Err(HostEffectAllocationError::InvalidInvocation)
    ));
    assert!(matches!(
        permit.reserve_exact(1),
        Err(runmat_value::ExactHostNumericAllocationError::ExceedsPermit)
    ));
    assert_eq!(ledger.snapshot().live_reserved_bytes, 4);
    assert!(placement.observations.borrow().is_empty());
    drop(retained);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[test]
fn observation_callback_cannot_admit_new_temporary_or_output_work() {
    let ledger = ManagedHostEffectLedger::new(12);
    let (runtime, placement) = managed_runtime(&ledger, 12);
    let stage = admit_host_effect_stage(&runtime, request(HostEffectOutput::ReturnedValue))
        .expect("precharged stage");
    let view = stage.reservation();
    let receipt = select_host_effect_after_stage(
        &runtime,
        request(HostEffectOutput::ReturnedValue),
        HostEffectFinalFacts::TestOnly(Digest::sha256(b"source-and-destination")),
        stage,
    )
    .expect("selected call");
    let output_permit = receipt.output_allocation_permit().expect("output permit");
    let retained = output_permit.reserve_exact(4).expect("retained result");
    let observed = Rc::new(Cell::new(false));
    let observed_callback = observed.clone();
    *placement.on_observe.borrow_mut() = Some(Box::new(move || {
        observed_callback.set(true);
        assert!(!view.is_current());
        assert!(matches!(
            view.reserve_live_exact(HostEffectLiveAllocationKind::CodecWorkspace, 1),
            Err(HostEffectAllocationError::InvalidInvocation)
        ));
        assert!(matches!(
            output_permit.reserve_exact(1),
            Err(runmat_value::ExactHostNumericAllocationError::ExceedsPermit)
        ));
    }));
    receipt.finish(WholeWorkObservationOutcome::Success);
    assert!(observed.get());
    assert_eq!(ledger.snapshot().live_reserved_bytes, 4);
    drop(retained);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[test]
fn captured_cancellation_blocks_cloned_allocation_views_before_receipt_drop() {
    let ledger = ManagedHostEffectLedger::new(12);
    let (runtime, placement) = managed_runtime(&ledger, 12);
    let cancellation = CancellationSource::new();
    let runtime = runtime.with_cancellation_token(cancellation.token());
    let stage = admit_host_effect_stage(&runtime, request(HostEffectOutput::ReturnedValue))
        .expect("precharged stage");
    let view = stage.reservation();
    let receipt = select_host_effect_after_stage(
        &runtime,
        request(HostEffectOutput::ReturnedValue),
        HostEffectFinalFacts::TestOnly(Digest::sha256(b"source-and-destination")),
        stage,
    )
    .expect("selected call");
    let output_permit = receipt.output_allocation_permit().expect("output permit");
    cancellation.cancel(CancellationReason::User);
    assert!(matches!(
        view.reserve_live_exact(HostEffectLiveAllocationKind::CodecWorkspace, 1),
        Err(HostEffectAllocationError::Cancelled)
    ));
    assert!(matches!(
        output_permit.reserve_exact(1),
        Err(runmat_value::ExactHostNumericAllocationError::ExceedsPermit)
    ));
    drop(receipt);
    assert!(placement.observations.borrow().is_empty());
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
    let second_stage = admit_host_effect_stage(&runtime, request(HostEffectOutput::ReturnedValue))
        .expect("second scan can use remaining eight bytes");
    let error = select_host_effect_after_stage(
        &runtime,
        request(HostEffectOutput::ReturnedValue),
        HostEffectFinalFacts::TestOnly(Digest::sha256(b"source-and-destination")),
        second_stage,
    )
    .err()
    .expect("second output headroom must be secured before return");
    assert_eq!(error.identifier(), Some("RunMat:Placement:ResourceLimit"));
    assert_eq!(ledger.snapshot().live_reserved_bytes, 12);
    first.finish(WholeWorkObservationOutcome::Success);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 4);
    let second = selected_receipt(&runtime);
    let second_output = second
        .output_allocation_permit()
        .expect("same second output permit")
        .reserve_exact(4)
        .expect("second selected receipt already owns headroom");
    assert_eq!(ledger.snapshot().live_reserved_bytes, 16);
    second.finish(WholeWorkObservationOutcome::Success);
    drop(first_output);
    drop(second_output);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    assert_eq!(placement.observations.borrow().len(), 2);
}
