use super::*;

#[test]
fn certified_power_reshape_and_capability_failures_retire_only_private_outputs() {
    for fault in [
        None,
        Some(ExactFault::ReshapeError),
        Some(ExactFault::WrongReshapeOwner),
        Some(ExactFault::BorrowedReshapeAlias),
        Some(ExactFault::RetractCapability),
    ] {
        let provider = provider_with_mode(
            false,
            true,
            RealPowerArithmeticMode::MixedDoubleThenSingle {
                single_operand: RealPowerOperand::Base,
            },
        );
        provider.exact_scalar.store(true, Ordering::SeqCst);
        *provider.exact_fault.lock().unwrap() = fault;
        let resident = provider
            .owner
            .allocate_zeroed(vec![2], NumericElementType::F32, GpuTensorStorage::Real)
            .unwrap();
        let prepared = prepare_resident_scalar_power(
            provider,
            &resident.handle,
            PowerScalarControl::DoubleBits(2.0_f64.to_bits()),
            RealPowerOperand::Base,
            &[1, 1],
            ProviderOperationIdentity::new("test.power"),
        )
        .unwrap()
        .unwrap();
        assert_ne!(
            prepared.output_shape(),
            resident.handle.shape(),
            "fixture must exercise shape reconciliation"
        );
        let result = ready(execute_resident_scalar_power(prepared));
        if fault.is_none() {
            let output = result.unwrap();
            assert_eq!(output.handle.shape(), [1, 2]);
            assert_eq!(
                provider
                    .exact_output_probe
                    .lock()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .retirements(),
                0
            );
            output.handle.release().unwrap();
        } else {
            let error = result.err().unwrap();
            if matches!(fault, Some(ExactFault::ReshapeError)) {
                assert!(error.to_string().contains("exact reshape primary failure"));
            }
        }
        assert_eq!(
            provider
                .exact_output_probe
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .retirements(),
            1
        );
        if let Some(probe) = provider.exact_reshape_probe.lock().unwrap().as_ref() {
            assert_eq!(probe.retirements(), 1);
        }
        assert_eq!(resident.probe.retirements(), 0);
        resident.handle.allocation_lease().validate_live().unwrap();
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn unsupported_probe_after_typed_upload_is_terminal() {
    let provider = provider(true);
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
    .unwrap()
    .unwrap();
    let error = ready(execute_resident_scalar_power(prepared))
        .err()
        .unwrap();
    assert!(error.to_string().contains("post-upload probe failure"));
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
    assert_eq!(provider.probes.load(Ordering::SeqCst), 1);
    assert!(!error.is_unsupported());
    assert_eq!(resident.probe.retirements(), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn unavailable_lease_between_prepare_and_execute_rejects_before_effects() {
    let provider = provider(false);
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
    .unwrap()
    .unwrap();
    resident.probe.mark_unavailable();
    assert!(resident.handle.allocation_lease().validate_live().is_err());
    let error = ready(execute_resident_scalar_power(prepared))
        .err()
        .unwrap();
    assert!(error.to_string().contains("no longer live"));
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.probes.load(Ordering::SeqCst), 0);
    assert_eq!(resident.probe.retirements(), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn feasibility_retracted_between_prepare_and_execute_rejects_before_effects() {
    let provider = provider(false);
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
    .unwrap()
    .unwrap();
    provider.admit.store(false, Ordering::SeqCst);
    let error = ready(execute_resident_scalar_power(prepared))
        .err()
        .unwrap();
    assert!(error.to_string().contains("feasibility changed"));
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.probes.load(Ordering::SeqCst), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn selected_complex_domain_rejected_after_native_scalar_upload_before_power() {
    let provider = provider(false);
    provider.real_only.store(true, Ordering::SeqCst);
    provider.complex_domain.store(true, Ordering::SeqCst);
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
    .unwrap()
    .unwrap();
    let error = ready(execute_resident_scalar_power(prepared))
        .err()
        .unwrap();
    assert!(error.to_string().contains("selected POWER result domain"));
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
    assert_eq!(provider.probes.load(Ordering::SeqCst), 1);
    assert_eq!(provider.power_calls.load(Ordering::SeqCst), 0);
    assert_eq!(resident.probe.retirements(), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn dropping_unpolled_prepared_program_keeps_borrowed_input_alive() {
    let provider = provider(false);
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
    .unwrap()
    .unwrap();
    let future = execute_resident_scalar_power(prepared);
    drop(future);
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.probes.load(Ordering::SeqCst), 0);
    assert_eq!(resident.probe.retirements(), 0);
    assert!(resident.handle.allocation_lease().validate_live().is_ok());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn cancelling_after_scalar_upload_retires_only_intermediate() {
    struct Noop;
    impl std::task::Wake for Noop {
        fn wake(self: std::sync::Arc<Self>) {}
    }
    let provider =
        provider_with_suspension(false, true, RealPowerArithmeticMode::NativeSingle, true);
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
    .unwrap()
    .unwrap();
    let mut future = Box::pin(execute_resident_scalar_power(prepared));
    let waker = std::task::Waker::from(std::sync::Arc::new(Noop));
    assert!(matches!(
        future
            .as_mut()
            .poll(&mut std::task::Context::from_waker(&waker)),
        std::task::Poll::Pending
    ));
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
    assert_eq!(provider.probes.load(Ordering::SeqCst), 1);
    let uploaded = provider
        .uploaded_probe
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .clone();
    drop(future);
    assert_eq!(uploaded.retirements(), 1);
    assert_eq!(resident.probe.retirements(), 0);
    assert!(resident.handle.allocation_lease().validate_live().is_ok());
}
