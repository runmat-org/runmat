use super::*;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn original_single_bits_survive_preparation_and_execution_on_double_provider() {
    let provider = provider(false);
    let resident = provider
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let bits = (-0.0f32).to_bits();
    let prepared = prepare_resident_scalar_power(
        provider,
        &resident.handle,
        PowerScalarControl::SingleBits(bits),
        RealPowerOperand::Exponent,
        &[1, 1],
        ProviderOperationIdentity::new("test.power"),
    )
    .unwrap()
    .unwrap();
    assert_eq!(prepared.mode(), RealPowerArithmeticMode::NativeSingle);
    assert_eq!(prepared.control(), PowerScalarControl::SingleBits(bits));
    assert_eq!(
        prepared.physical_recipe().scalar,
        PowerScalarConstruction::ExactTypedUpload
    );
    assert!(!prepared.physical_recipe().single_publication);
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
    let output = ready(execute_resident_scalar_power(prepared)).unwrap();
    assert_eq!(output.domain, PowerResultDomain::Real);
    assert_eq!(output.handle.shape(), &[2, 1]);
    assert_eq!(
        crate::handle_precision(&output.handle),
        Some(ProviderPrecision::F32)
    );
    assert_eq!(provider.last_bits.load(Ordering::SeqCst), bits);
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
    assert_eq!(provider.probes.load(Ordering::SeqCst), 1);
    assert_eq!(resident.probe.retirements(), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn rank_one_native_single_scalar_upload_uses_actual_broadcast_shape() {
    let provider = provider(false);
    let resident = provider
        .owner
        .allocate_zeroed(vec![4], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let prepared = prepare_resident_scalar_power(
        provider,
        &resident.handle,
        PowerScalarControl::SingleBits(1.8f32.to_bits()),
        RealPowerOperand::Base,
        &[1, 1],
        ProviderOperationIdentity::new("test.power"),
    )
    .unwrap()
    .unwrap();
    assert_eq!(prepared.output_shape(), &[1, 4]);
    let output = ready(execute_resident_scalar_power(prepared)).unwrap();
    assert_eq!(output.handle.shape(), &[1, 4]);
    assert_eq!(
        crate::handle_precision(&output.handle),
        Some(ProviderPrecision::F32)
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn rank_one_native_single_rejects_provider_wrong_physical_shape() {
    let provider = provider(false);
    provider.wrong_power_shape.store(true, Ordering::SeqCst);
    let resident = provider
        .owner
        .allocate_zeroed(vec![4], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let prepared = prepare_resident_scalar_power(
        provider,
        &resident.handle,
        PowerScalarControl::SingleBits(1.8f32.to_bits()),
        RealPowerOperand::Base,
        &[1, 1],
        ProviderOperationIdentity::new("test.power"),
    )
    .unwrap()
    .unwrap();
    let error = ready(execute_resident_scalar_power(prepared))
        .err()
        .unwrap();
    assert!(error.to_string().contains("invalid POWER native result"));
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
    assert_eq!(provider.probes.load(Ordering::SeqCst), 1);
    assert_eq!(resident.probe.retirements(), 0);
}
