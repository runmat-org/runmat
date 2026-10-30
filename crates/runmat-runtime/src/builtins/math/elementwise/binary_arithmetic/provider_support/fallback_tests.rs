use super::*;
use runmat_accelerate_api::{GpuHandleProvenance, GpuTensorStorage, NumericElementType};
use runmat_value::{IntegerStorage, Tensor};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn explicit_floating_decline_is_not_host_authority() {
    let owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
    let source = owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let explicit = source
        .handle
        .with_provenance(GpuHandleProvenance::Explicit)
        .unwrap();
    let automatic = explicit
        .clone()
        .with_provenance(GpuHandleProvenance::Automatic)
        .unwrap();
    let scalar = Value::Num(1.5);
    let error = HostFallbackAdmission::new(
        None,
        &runmat_builtins::PLUS_CATALOG_ENTRY,
        &Value::GpuTensor(explicit.clone()),
        &scalar,
    )
    .err()
    .expect("explicit arithmetic needs a provider");
    assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
    assert_eq!(
        error.identifier(),
        Some("RunMat:gpu:ExplicitProviderRequired")
    );
    HostFallbackAdmission::new(
        None,
        &runmat_builtins::PLUS_CATALOG_ENTRY,
        &Value::GpuTensor(automatic),
        &scalar,
    )
    .expect("automatic input can decline before effects");
    explicit.allocation_lease().validate_live().unwrap();
    assert_eq!(source.probe.retirements(), 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn exact_integer_form_requires_provider_publication_before_download() {
    crate::builtins::common::test_support::with_uncertified_integer_provider(
        |provider, fixture| {
            let source = Tensor::new_integer(IntegerStorage::I32(vec![2, 4]), vec![2, 1]).unwrap();
            let resident = gpu_helpers::upload_tensor(provider, &source)
                .unwrap()
                .with_provenance(GpuHandleProvenance::Explicit)
                .unwrap();
            let scalar = Value::Num(2.0);
            provider.reset_telemetry();
            for entry in [
                &runmat_builtins::PLUS_CATALOG_ENTRY,
                &runmat_builtins::MINUS_CATALOG_ENTRY,
            ] {
                assert_eq!(
                    integer_materialization(entry, &Value::GpuTensor(resident.clone()), &scalar)
                        .unwrap(),
                    Some((BuiltinIntegerClass::Int32, vec![2, 1]))
                );
                assert_eq!(
                    integer_materialization(entry, &scalar, &Value::GpuTensor(resident.clone()))
                        .unwrap(),
                    Some((BuiltinIntegerClass::Int32, vec![2, 1]))
                );
                let error = HostFallbackAdmission::new(
                    None,
                    entry,
                    &Value::GpuTensor(resident.clone()),
                    &scalar,
                )
                .err()
                .expect("fixture lacks exact integer upload evidence");
                assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
                assert!(
                    fixture.upload_evidence_queries() > 0,
                    "upload evidence gate queried"
                );
                resident.allocation_lease().validate_live().unwrap();
            }
        },
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn explicit_integer_fallback_rejects_mismatched_class_and_shape_before_gather() {
    let owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
    let left = owner
        .allocate_zeroed(vec![2, 3], NumericElementType::I32, GpuTensorStorage::Real)
        .unwrap()
        .handle
        .with_provenance(GpuHandleProvenance::Explicit)
        .unwrap();
    let wrong_class = owner
        .allocate_zeroed(vec![2, 3], NumericElementType::I64, GpuTensorStorage::Real)
        .unwrap()
        .handle;
    let wrong_shape = owner
        .allocate_zeroed(vec![4, 3], NumericElementType::I32, GpuTensorStorage::Real)
        .unwrap()
        .handle;
    for (right, detail) in [
        (wrong_class, "same integer class"),
        (wrong_shape, "dimension"),
    ] {
        let error = HostFallbackAdmission::new(
            None,
            &runmat_builtins::PLUS_CATALOG_ENTRY,
            &Value::GpuTensor(left.clone()),
            &Value::GpuTensor(right),
        )
        .err()
        .expect("invalid integer form must reject before readback");
        assert!(error.to_string().contains(detail));
        assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
    }
    left.allocation_lease().validate_live().unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn integer_form_does_not_certify_nonscalar_double_or_unsupported_storage() {
    let owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
    let integer = owner
        .allocate_zeroed(vec![2, 1], NumericElementType::I32, GpuTensorStorage::Real)
        .unwrap()
        .handle
        .with_provenance(GpuHandleProvenance::Explicit)
        .unwrap();
    let nonscalar = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
    let error = HostFallbackAdmission::new(
        None,
        &runmat_builtins::TIMES_CATALOG_ENTRY,
        &Value::GpuTensor(integer.clone()),
        &nonscalar,
    )
    .err()
    .expect("integer with nonscalar Double must fail before readback");
    assert!(error.to_string().contains("scalar double"));
    assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);

    let complex = owner
        .allocate_zeroed(
            vec![2, 1],
            NumericElementType::F64,
            GpuTensorStorage::ComplexInterleaved,
        )
        .unwrap()
        .handle;
    let error = HostFallbackAdmission::new(
        None,
        &runmat_builtins::TIMES_CATALOG_ENTRY,
        &Value::GpuTensor(integer.clone()),
        &Value::GpuTensor(complex),
    )
    .err()
    .expect("complex operand is outside integer materialization form");
    assert_eq!(
        error.identifier(),
        Some("RunMat:gpu:ExplicitProviderRequired")
    );
    integer.allocation_lease().validate_live().unwrap();
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[test]
#[ignore = "requires an actual WGPU adapter with native integer transfer"]
fn wgpu_exact_integer_form_materializes_the_certified_result_class() {
    let _guard = crate::builtins::common::test_support::accel_test_lock();
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("required WGPU adapter");
    let input = Tensor::new_integer(IntegerStorage::I32(vec![2, 4]), vec![2, 1]).unwrap();
    let resident = gpu_helpers::upload_tensor(provider, &input)
        .unwrap()
        .with_provenance(GpuHandleProvenance::Explicit)
        .unwrap();
    let admitted = HostFallbackAdmission::new(
        None,
        &runmat_builtins::PLUS_CATALOG_ENTRY,
        &Value::GpuTensor(resident.clone()),
        &Value::Num(2.0),
    )
    .expect("exact catalog form and physical integer publication");
    let result = Tensor::new_integer(IntegerStorage::I32(vec![4, 6]), vec![2, 1]).unwrap();
    let Value::GpuTensor(output) = admitted.finish(Value::Tensor(result)).unwrap() else {
        panic!("explicit integer fallback must remain resident")
    };
    assert!(runmat_accelerate_api::handle_is_explicit(&output));
    assert_eq!(
        runmat_accelerate_api::handle_integer_type(&output),
        Some(runmat_accelerate_api::IntegerElementType::I32)
    );
    assert!(!output.shares_allocation_with(&resident));
    resident.allocation_lease().validate_live().unwrap();
}
