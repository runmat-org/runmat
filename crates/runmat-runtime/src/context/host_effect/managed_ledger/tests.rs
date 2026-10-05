use super::*;

fn temp(
    stage: &ManagedHostEffectStageReservation,
    bytes: usize,
) -> Result<Arc<dyn HostEffectLiveAllocationReservation>, HostEffectAllocationError> {
    stage.reserve_live_exact(HostEffectLiveAllocationKind::CodecWorkspace, bytes)
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn two_temporary_claims_share_one_precharged_envelope() {
    let ledger = ManagedHostEffectLedger::new(8);
    let stage = ledger.reserve_stage(8).expect("one stage");
    let first = temp(&stage, 4).expect("first buffer");
    let second = temp(&stage, 4).expect("second buffer");
    assert_eq!(
        temp(&stage, 1).err(),
        Some(HostEffectAllocationError::ExceedsStageEnvelope)
    );
    assert_eq!(ledger.snapshot().live_reserved_bytes, 8);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 8);
    assert_eq!(ledger.snapshot().peak_claimed_bytes, 8);
    drop(first);
    let replacement = temp(&stage, 4).expect("dropped charge reused");
    assert_eq!(ledger.snapshot().live_claimed_bytes, 8);
    drop(stage);
    assert_eq!(
        ledger.snapshot().live_reserved_bytes,
        8,
        "claims retain stage"
    );
    drop(second);
    drop(replacement);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn retained_output_adds_global_charge_during_stage_overlap() {
    let ledger = ManagedHostEffectLedger::new(12);
    let stage = ledger.reserve_stage(8).expect("stage");
    let input = temp(&stage, 4).expect("live scan buffer");
    let output = stage.reserve_output_bytes(4).expect("retained output");
    assert_eq!(ledger.snapshot().live_reserved_bytes, 12);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 8);
    assert_eq!(ledger.snapshot().peak_reserved_bytes, 12);
    assert_eq!(ledger.snapshot().peak_claimed_bytes, 8);
    assert_eq!(
        stage.reserve_output_bytes(1).err(),
        Some(HostEffectAllocationError::ExceedsCapacity)
    );
    drop(input);
    drop(stage);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 4);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 4);
    drop(output);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn revocation_and_overflow_are_typed_and_keep_existing_charges() {
    let full = ManagedHostEffectLedger::new(u64::MAX);
    let full_stage = full.reserve_stage(u64::MAX).expect("full stage");
    assert_eq!(
        full_stage.reserve_output_bytes(1).err(),
        Some(HostEffectAllocationError::Overflow)
    );
    drop(full_stage);
    assert_eq!(full.snapshot().live_reserved_bytes, 0);

    let ledger = ManagedHostEffectLedger::new(8);
    let stage = ledger.reserve_stage(8).expect("stage");
    let claim = temp(&stage, 4).expect("live claim");
    ledger.revoke();
    assert_eq!(
        temp(&stage, 1).err(),
        Some(HostEffectAllocationError::Revoked)
    );
    assert_eq!(
        stage.reserve_output_bytes(1).err(),
        Some(HostEffectAllocationError::Revoked)
    );
    assert_eq!(
        ledger.reserve_stage(1).err(),
        Some(HostEffectAllocationError::Revoked)
    );
    assert_eq!(ledger.snapshot().live_reserved_bytes, 8);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 4);
    drop(claim);
    drop(stage);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn independent_stages_charge_atomically_under_concurrent_claims() {
    let ledger = ManagedHostEffectLedger::new(16);
    let first = ledger.reserve_stage(8).expect("first stage");
    let second = ledger.reserve_stage(8).expect("second stage");
    let worker = std::thread::spawn(move || {
        let claim = temp(&first, 8).expect("first worker claim");
        assert_eq!(
            temp(&first, 1).err(),
            Some(HostEffectAllocationError::ExceedsStageEnvelope)
        );
        claim
    });
    let other = temp(&second, 8).expect("second worker claim");
    let first_claim = worker.join().expect("worker completion");
    assert_eq!(ledger.snapshot().peak_claimed_bytes, 16);
    assert_eq!(ledger.snapshot().peak_reserved_bytes, 16);
    assert_eq!(
        ledger.reserve_stage(1).err(),
        Some(HostEffectAllocationError::ExceedsCapacity)
    );
    drop(first_claim);
    drop(other);
    drop(second);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}
