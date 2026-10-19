use super::*;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn actual_upload_size_mismatch_retires_owned_output_not_borrowed() {
    let owner = Owner::new(16);
    let _selected = ThreadProviderGuard::set(Some(owner));
    let borrowed = allocation(owner, vec![1, 1], NumericElementType::F64);
    let host = Value::Num(4.0);
    let selected = plan(
        owner,
        vec![
            resident_entry(&borrowed.handle, ProviderElementType::F64),
            host_entry(owner),
        ],
        vec![
            InputAction::Borrow(&borrowed.handle),
            InputAction::Upload {
                source: &host,
                target: ProviderPrecision::F64,
            },
        ],
        vec![],
        32,
    );
    assert!(selected.has_physical_effects());
    let error = block_on(selected.prepare()).unwrap_err();
    assert!(error.is_after_effect());
    assert_eq!(owner.evidence_count(), 1);
    assert_eq!(owner.upload_count(), 1);
    assert_eq!(owner.probes.lock().unwrap().len(), 1);
    assert_eq!(owner.probes.lock().unwrap()[0].retirements(), 1);
    assert_eq!(borrowed.probe.retirements(), 0);
    borrowed.handle.allocation_lease().validate_live().unwrap();
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn earlier_upload_then_later_failure_retains_after_effect_phase() {
    let owner = Owner::new(8);
    owner.fail_upload(2);
    let _selected = ThreadProviderGuard::set(Some(owner));
    let first = Value::Num(5.0);
    let second = Value::Num(6.0);
    let selected = plan(
        owner,
        vec![host_entry(owner), host_entry(owner)],
        vec![
            InputAction::Upload {
                source: &first,
                target: ProviderPrecision::F64,
            },
            InputAction::Upload {
                source: &second,
                target: ProviderPrecision::F64,
            },
        ],
        vec![],
        32,
    );
    let error = block_on(selected.prepare()).unwrap_err();
    assert!(error.is_after_effect());
    assert_eq!(owner.evidence_count(), 2);
    assert_eq!(owner.upload_count(), 2);
    assert_eq!(owner.probes.lock().unwrap().len(), 1);
    assert_eq!(owner.probes.lock().unwrap()[0].retirements(), 1);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn all_host_plan_uses_captured_provider_not_active_other_facade() {
    let owner = Owner::new(8);
    let other = Owner::same_device_facade(8, owner.device_id());
    assert_eq!(owner.device_id(), other.device_id());
    assert!(!runmat_accelerate_api::same_provider_instance(owner, other));
    let host = Value::Num(7.0);
    let selected = plan(
        owner,
        vec![host_entry(owner)],
        vec![InputAction::Upload {
            source: &host,
            target: ProviderPrecision::F64,
        }],
        vec![],
        32,
    );
    let _active_other = ThreadProviderGuard::set(Some(other));
    let prepared = block_on(selected.prepare()).expect("registered selected owner must execute");
    assert_eq!(prepared.len(), 1);
    assert_eq!(prepared[0].device_id(), owner.device_id());
    assert!(runmat_accelerate_api::same_provider_instance(
        runmat_runtime::builtins::common::gpu_helpers::exact_provider_for_handle(&prepared[0])
            .expect("durable registered owner"),
        owner,
    ));
    assert_eq!(owner.evidence_count(), 1);
    assert_eq!(owner.upload_count(), 1);
    assert_eq!(other.evidence_count(), 0);
    assert_eq!(other.upload_count(), 0);
    assert_eq!(owner.probes.lock().unwrap()[0].retirements(), 0);
    drop(prepared);
    assert_eq!(owner.probes.lock().unwrap()[0].retirements(), 1);
}
