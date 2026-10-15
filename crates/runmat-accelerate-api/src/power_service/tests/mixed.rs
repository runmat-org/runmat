use super::*;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn mixed_single_base_double_exponent_widens_and_publishes_once() {
    let mode = RealPowerArithmeticMode::MixedDoubleThenSingle {
        single_operand: RealPowerOperand::Base,
    };
    let provider = provider_with_mode(false, true, mode);
    let resident = provider
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let prepared = prepare_resident_scalar_power(
        provider,
        &resident.handle,
        PowerScalarControl::DoubleBits(2.0f64.to_bits()),
        RealPowerOperand::Base,
        &[1, 1],
        ProviderOperationIdentity::new("test.power"),
    )
    .unwrap()
    .unwrap();
    assert_eq!(prepared.mode(), mode);
    assert!(prepared.physical_recipe().widen_resident);
    assert!(prepared.physical_recipe().single_publication);
    assert_eq!(
        prepared.physical_recipe().scalar,
        PowerScalarConstruction::ExactTypedUpload
    );
    let output = ready(execute_resident_scalar_power(prepared)).unwrap();
    assert_eq!(
        crate::handle_precision(&output.handle),
        Some(ProviderPrecision::F32)
    );
    assert_eq!(
        provider.last_double_bits.load(Ordering::SeqCst),
        2.0f64.to_bits()
    );
    assert_eq!(provider.widenings.load(Ordering::SeqCst), 1);
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
    assert_eq!(provider.fills.load(Ordering::SeqCst), 0);
    assert_eq!(provider.publications.load(Ordering::SeqCst), 1);
    assert_eq!(resident.probe.retirements(), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn mixed_negative_zero_double_control_uses_exact_typed_upload() {
    let mode = RealPowerArithmeticMode::MixedDoubleThenSingle {
        single_operand: RealPowerOperand::Base,
    };
    let provider = provider_with_mode(false, true, mode);
    let resident = provider
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let bits = (-0.0f64).to_bits();
    let prepared = prepare_resident_scalar_power(
        provider,
        &resident.handle,
        PowerScalarControl::DoubleBits(bits),
        RealPowerOperand::Base,
        &[1, 1],
        ProviderOperationIdentity::new("test.power"),
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        prepared.physical_recipe().scalar,
        PowerScalarConstruction::ExactTypedUpload
    );
    let output = ready(execute_resident_scalar_power(prepared)).unwrap();
    assert_eq!(
        crate::handle_precision(&output.handle),
        Some(ProviderPrecision::F32)
    );
    assert_eq!(provider.last_double_bits.load(Ordering::SeqCst), bits);
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
    assert_eq!(provider.fills.load(Ordering::SeqCst), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn mixed_exact_upload_rejects_alias_to_widened_operand_before_probe() {
    let mode = RealPowerArithmeticMode::MixedDoubleThenSingle {
        single_operand: RealPowerOperand::Base,
    };
    let provider = provider_with_mode(false, true, mode);
    provider.alias_widened.store(true, Ordering::SeqCst);
    let resident = provider
        .owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let prepared = prepare_resident_scalar_power(
        provider,
        &resident.handle,
        PowerScalarControl::DoubleBits((-0.0f64).to_bits()),
        RealPowerOperand::Base,
        &[1, 1],
        ProviderOperationIdentity::new("test.power"),
    )
    .unwrap()
    .unwrap();
    let error = ready(execute_resident_scalar_power(prepared))
        .err()
        .unwrap();
    assert!(error.to_string().contains("aliasing a borrowed input"));
    assert_eq!(provider.widenings.load(Ordering::SeqCst), 1);
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
    assert_eq!(provider.probes.load(Ordering::SeqCst), 0);
    assert_eq!(
        provider
            .widened_probe
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .retirements(),
        1
    );
    assert_eq!(resident.probe.retirements(), 0);
    assert!(resident.handle.allocation_lease().validate_live().is_ok());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn mixed_double_base_single_exponent_retains_original_single_class() {
    let mode = RealPowerArithmeticMode::MixedDoubleThenSingle {
        single_operand: RealPowerOperand::Exponent,
    };
    let provider = provider_with_mode(false, true, mode);
    let resident = provider
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let control = 1.8f32;
    let prepared = prepare_resident_scalar_power(
        provider,
        &resident.handle,
        PowerScalarControl::SingleBits(control.to_bits()),
        RealPowerOperand::Base,
        &[1, 1],
        ProviderOperationIdentity::new("test.power"),
    )
    .unwrap()
    .unwrap();
    assert_eq!(prepared.mode(), mode);
    assert!(!prepared.physical_recipe().widen_resident);
    let output = ready(execute_resident_scalar_power(prepared)).unwrap();
    assert_eq!(
        crate::handle_precision(&output.handle),
        Some(ProviderPrecision::F32)
    );
    assert_eq!(
        provider.last_double_bits.load(Ordering::SeqCst),
        f64::from(control).to_bits()
    );
    assert_eq!(provider.widenings.load(Ordering::SeqCst), 0);
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
    assert_eq!(provider.fills.load(Ordering::SeqCst), 0);
    assert_eq!(provider.publications.load(Ordering::SeqCst), 1);
}
