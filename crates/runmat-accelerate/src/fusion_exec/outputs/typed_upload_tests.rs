use super::*;

fn assert_upload_failure(error: &ProviderExecutionError) {
    let mut cause: &(dyn std::error::Error + 'static) = error;
    loop {
        if let Some(failure) = cause.downcast_ref::<UploadFailure>() {
            assert_eq!(failure.0, "typed fixture upload failed");
            return;
        }
        cause = cause.source().expect("typed upload cause must be retained");
    }
}

fn assert_terminal_unsupported(error: &ProviderExecutionError) {
    assert!(error.is_after_effect());
    // This predicate authorizes replay; it is not just a cause-type query.
    assert!(!error.is_unsupported());
    let mut cause: &(dyn std::error::Error + 'static) = error;
    loop {
        if cause.is::<runmat_accelerate_api::UnsupportedProviderOperation>() {
            return;
        }
        cause = cause.source().expect("unsupported cause must be retained");
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn preserves_single_payload_when_provider_defaults_double() {
    let owner = new_owner();
    assert_eq!(owner.precision(), ProviderPrecision::F64);
    let _selected = ThreadProviderGuard::set(Some(owner));
    let source = [1.25_f32, -0.0, f32::from_bits(0x7fc0_1234)];
    let output = owner
        .allocations
        .allocate_zeroed(vec![1, 3], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    *owner.upload_result.lock().unwrap() = Some(output.handle.clone());

    let mut admission = FusionOutputAdmission::with_borrowed_handles(std::iter::empty());
    let uploaded = admission
        .upload_typed(
            owner,
            &HostNumericTensorView {
                data: HostNumericDataView::F32(&source),
                shape: &[1, 3],
                storage: GpuTensorStorage::Real,
            },
        )
        .unwrap();
    assert_eq!(uploaded.physical().element_type(), NumericElementType::F32);
    assert_eq!(uploaded.shape(), [1, 3]);
    assert!(runmat_accelerate_api::same_provider_instance(
        exact_provider_for_handle(&uploaded).unwrap(),
        owner
    ));
    assert_eq!(
        owner.uploads.lock().unwrap().as_slice(),
        [UploadCall {
            shape: vec![1, 3],
            storage: GpuTensorStorage::Real,
            payload: UploadPayload::F32(source.iter().map(|value| value.to_bits()).collect()),
        }]
    );
    drop(output.handle);
    assert_eq!(output.probe.retirements(), 0);
    drop(uploaded);
    assert_eq!(output.probe.retirements(), 1);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn preserves_double_payload_when_provider_defaults_single() {
    let owner = new_owner_with_precision(ProviderPrecision::F32);
    assert_eq!(owner.precision(), ProviderPrecision::F32);
    let _selected = ThreadProviderGuard::set(Some(owner));
    let source = [-0.0_f64, f64::from_bits(0x7ff8_0000_0000_1234)];
    let output = owner
        .allocations
        .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    *owner.upload_result.lock().unwrap() = Some(output.handle.clone());

    let mut admission = FusionOutputAdmission::with_borrowed_handles(std::iter::empty());
    let uploaded = admission
        .upload_typed(
            owner,
            &HostNumericTensorView {
                data: HostNumericDataView::F64(&source),
                shape: &[1, 2],
                storage: GpuTensorStorage::Real,
            },
        )
        .unwrap();
    assert_eq!(uploaded.physical().element_type(), NumericElementType::F64);
    assert_eq!(uploaded.shape(), [1, 2]);
    assert_eq!(
        owner.uploads.lock().unwrap().as_slice(),
        [UploadCall {
            shape: vec![1, 2],
            storage: GpuTensorStorage::Real,
            payload: UploadPayload::F64(source.iter().map(|value| value.to_bits()).collect()),
        }]
    );
    drop(output.handle);
    assert_eq!(output.probe.retirements(), 0);
    drop(uploaded);
    assert_eq!(output.probe.retirements(), 1);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn rejects_unsupported_or_malformed_input_before_provider_effects() {
    let owner = new_owner();
    let _selected = ThreadProviderGuard::set(Some(owner));
    let mut admission = FusionOutputAdmission::with_borrowed_handles(std::iter::empty());
    for host in [
        HostNumericTensorView {
            data: HostNumericDataView::U64(&[u64::MAX]),
            shape: &[1, 1],
            storage: GpuTensorStorage::Real,
        },
        HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.0, -2.0]),
            shape: &[1, 1],
            storage: GpuTensorStorage::ComplexInterleaved,
        },
    ] {
        let error = admission.upload_typed(owner, &host).unwrap_err();
        assert!(error.is_unsupported());
        assert!(!error.is_after_effect());
    }
    let malformed = HostNumericTensorView {
        data: HostNumericDataView::F32(&[1.0]),
        shape: &[1, 2],
        storage: GpuTensorStorage::Real,
    };
    let error = admission.upload_typed(owner, &malformed).unwrap_err();
    assert!(!error.is_after_effect());
    assert!(owner.uploads.lock().unwrap().is_empty());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn retires_malformed_outputs_but_preserves_prepared_input_aliases() {
    let owner = new_owner();
    let _selected = ThreadProviderGuard::set(Some(owner));
    let source = [1.0_f32, 2.0];
    let host = HostNumericTensorView {
        data: HostNumericDataView::F32(&source),
        shape: &[1, 2],
        storage: GpuTensorStorage::Real,
    };
    let foreign = TestAllocationOwner::new();
    let cases = [
        owner
            .allocations
            .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
            .unwrap(),
        owner
            .allocations
            .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap(),
        foreign
            .allocate_zeroed(vec![1, 2], NumericElementType::F32, GpuTensorStorage::Real)
            .unwrap(),
    ];
    let mut admission = FusionOutputAdmission::with_borrowed_handles(std::iter::empty());
    for output in cases {
        *owner.upload_result.lock().unwrap() = Some(output.handle);
        let error = admission.upload_typed(owner, &host).unwrap_err();
        assert!(runmat_accelerate_api::is_invalid_numeric_upload_result(
            &error
        ));
        assert!(error.is_after_effect());
        assert_eq!(output.probe.retirements(), 1);
    }

    let borrowed = owner
        .allocations
        .allocate_zeroed(vec![1, 2], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let mut admission = FusionOutputAdmission::with_borrowed_handles([&borrowed.handle]);
    *owner.upload_result.lock().unwrap() = Some(borrowed.handle.clone());
    let error = admission.upload_typed(owner, &host).unwrap_err();
    assert!(runmat_accelerate_api::is_invalid_numeric_upload_result(
        &error
    ));
    assert_eq!(borrowed.probe.retirements(), 0);
    drop(admission);
    drop(borrowed.handle);
    assert_eq!(borrowed.probe.retirements(), 1);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn local_failure_is_preeffect_but_prior_stage_remains_terminal_after_expiry() {
    let owner = new_owner();
    let _selected = ThreadProviderGuard::set(Some(owner));
    let host = HostNumericTensorView {
        data: HostNumericDataView::F32(&[3.0, 4.0]),
        shape: &[1, 2],
        storage: GpuTensorStorage::Real,
    };
    let mut local = FusionOutputAdmission::with_borrowed_handles(std::iter::empty());
    *owner.upload_failure.lock().unwrap() = Some("typed fixture upload failed");
    let error = local.upload_typed(owner, &host).unwrap_err();
    assert!(!error.is_after_effect());
    assert_upload_failure(&error);
    owner.uploads.lock().unwrap().clear();

    let borrowed = owner
        .allocations
        .allocate_zeroed(vec![1, 2], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let staged = owner
        .allocations
        .allocate_zeroed(vec![1, 2], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let mut admission = FusionOutputAdmission::with_borrowed_handles([&borrowed.handle]);
    let staged_handle = admission
        .admit(staged.handle, owner, &[1, 2], ProviderPrecision::F32)
        .unwrap();

    let unsupported = HostNumericTensorView {
        data: HostNumericDataView::U64(&[u64::MAX]),
        shape: &[1, 1],
        storage: GpuTensorStorage::Real,
    };
    let error = admission.upload_typed(owner, &unsupported).unwrap_err();
    assert_terminal_unsupported(&error);
    assert!(owner.uploads.lock().unwrap().is_empty());

    drop(staged_handle);
    assert_eq!(staged.probe.retirements(), 1);
    let error = admission.upload_typed(owner, &unsupported).unwrap_err();
    assert_terminal_unsupported(&error);
    let malformed = HostNumericTensorView {
        data: HostNumericDataView::F32(&[3.0]),
        shape: &[1, 2],
        storage: GpuTensorStorage::Real,
    };
    let error = admission.upload_typed(owner, &malformed).unwrap_err();
    assert!(error.is_after_effect());
    assert!(owner.uploads.lock().unwrap().is_empty());

    *owner.upload_failure.lock().unwrap() = Some("typed fixture upload failed");
    let error = admission.upload_typed(owner, &host).unwrap_err();
    assert!(error.is_after_effect());
    assert_upload_failure(&error);
    assert_eq!(owner.uploads.lock().unwrap().len(), 1);
    assert_eq!(borrowed.probe.retirements(), 0);
    drop(admission);
    assert_eq!(staged.probe.retirements(), 1);
    drop(borrowed.handle);
    assert_eq!(borrowed.probe.retirements(), 1);
}
