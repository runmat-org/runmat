use super::*;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn rejected_native_single_has_no_upload_or_probe_effect() {
    let provider = provider_with_admission(false, false);
    let resident = provider
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let prepared = prepare_resident_scalar_power(
        provider,
        &resident.handle,
        PowerScalarControl::SingleBits(1.8f32.to_bits()),
        RealPowerOperand::Base,
        &[1, 1],
        ProviderOperationIdentity::new("test.power"),
    )
    .unwrap();
    assert!(prepared.is_none());
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.probes.load(Ordering::SeqCst), 0);
    assert_eq!(resident.probe.retirements(), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn malformed_scalar_shape_and_foreign_owner_reject_before_effects() {
    let owner_provider = provider(false);
    let foreign = provider(false);
    let resident = foreign
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    for shape in [&[2, 1][..], &[1, 1][..]] {
        assert!(prepare_resident_scalar_power(
            owner_provider,
            &resident.handle,
            PowerScalarControl::SingleBits(2.0f32.to_bits()),
            RealPowerOperand::Base,
            shape,
            ProviderOperationIdentity::new("test.power")
        )
        .is_err());
    }
    assert_eq!(owner_provider.uploads.load(Ordering::SeqCst), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn malformed_borrowed_alias_keeps_caller_lease_live() {
    let owner = TestAllocationOwner::new();
    let allocation = owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let error = super::validate::reject(allocation.handle.clone(), "aliased cast");
    assert!(error.to_string().contains("aliased cast"));
    assert_eq!(allocation.probe.retirements(), 0);
    assert!(allocation.handle.allocation_lease().validate_live().is_ok());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn malformed_owned_output_retains_failed_retirement_cause() {
    let owner = TestAllocationOwner::new();
    let allocation = owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    allocation
        .probe
        .fail_retirement("injected POWER cleanup failure");
    let error = super::validate::reject(allocation.handle, "malformed result");
    assert!(error.to_string().contains("injected POWER cleanup failure"));
    assert_eq!(allocation.probe.retirements(), 1);
}
