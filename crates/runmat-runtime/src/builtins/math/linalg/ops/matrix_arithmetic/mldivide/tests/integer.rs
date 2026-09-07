use super::*;

#[test]
fn resident_integer_result_restores_to_the_input_owner_not_the_ambient_provider() {
    let _guard = test_support::accel_test_lock();
    let owner = Box::leak(Box::new(
        runmat_accelerate::simple_provider::InProcessProvider::new(),
    ));
    let ambient = Box::leak(Box::new(
        runmat_accelerate::simple_provider::InProcessProvider::new(),
    ));
    unsafe {
        runmat_accelerate_api::register_provider(owner);
        runmat_accelerate_api::register_provider(ambient);
    }
    let rhs =
        Tensor::new_integer(IntegerStorage::U64(vec![8, 12]), vec![1, 2]).expect("integer rhs");
    let rhs_handle = gpu_helpers::upload_tensor(owner, &rhs).expect("owner upload");

    let result = mldivide_builtin(
        Value::Int(IntValue::U64(2)),
        Value::GpuTensor(rhs_handle.clone()),
    )
    .expect("integer divide");
    let Value::GpuTensor(result_handle) = result else {
        panic!("expected resident result")
    };
    let result_owner =
        runmat_accelerate_api::provider_for_handle(&result_handle).expect("restored owner");
    assert!(std::ptr::eq(result_owner, owner));
    assert!(!std::ptr::eq(result_owner, ambient));
    let gathered = block_on(owner.download_integer(&result_handle)).expect("owner download");
    assert_eq!(
        gathered.data,
        runmat_accelerate_api::HostIntegerDataOwned::U64(vec![4, 6])
    );
    owner.free(&rhs_handle).expect("free rhs");
    owner.free(&result_handle).expect("free result");
}

#[test]
#[cfg(feature = "wgpu")]
fn wgpu_integer_scalar_mldivide_preserves_all_classes_and_residency() {
    let _guard = test_support::accel_test_lock();
    if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .is_err()
    {
        return;
    }
    let provider = runmat_accelerate_api::provider().expect("registered WGPU provider");

    for (scalar, rhs, expected) in integer_scalar_mldivide_cases() {
        let rhs = Tensor::new_integer(rhs, vec![1, 3]).expect("WGPU integer RHS");
        let rhs_handle = gpu_helpers::upload_tensor(provider, &rhs).expect("upload RHS");
        let result_handle =
            match mldivide_builtin(Value::Int(scalar), Value::GpuTensor(rhs_handle.clone())) {
                Ok(Value::GpuTensor(handle)) => handle,
                Ok(other) => {
                    provider.free(&rhs_handle).expect("free RHS");
                    panic!("expected resident integer result, got {other:?}");
                }
                Err(error) => {
                    provider.free(&rhs_handle).expect("free RHS");
                    panic!("WGPU integer scalar-left division failed: {error}");
                }
            };

        let owner =
            runmat_accelerate_api::provider_for_handle(&result_handle).expect("result WGPU owner");
        let owner_matches = std::ptr::eq(owner, provider);
        let result_type = runmat_accelerate_api::handle_integer_type(&result_handle);
        let gathered = test_support::gather(Value::GpuTensor(result_handle.clone()));

        provider.free(&rhs_handle).expect("free RHS");
        provider.free(&result_handle).expect("free result");

        assert!(owner_matches, "result must restore to the WGPU RHS owner");
        assert_eq!(result_type, Some(integer_element_type(&expected)));
        let gathered = gathered.expect("gather WGPU integer result");
        assert_eq!(gathered.integer_storage(), Some(&expected));
    }
}

#[test]
fn complex_single_scalar_left_division_preserves_single_storage() {
    let divisor = ComplexTensor::from_f32(vec![(2.0, 1.0)], vec![1, 1]).expect("divisor");
    let rhs = ComplexTensor::from_f32(vec![(8.0, 4.0), (12.0, 6.0)], vec![2, 1]).expect("rhs");
    let Value::ComplexTensor(result) =
        mldivide_builtin(Value::ComplexTensor(divisor), Value::ComplexTensor(rhs))
            .expect("complex single scalar solve")
    else {
        panic!("expected complex tensor")
    };
    assert_eq!(result.numeric_dtype(), runmat_value::NumericDType::F32);
    assert_eq!(result.shape, vec![2, 1]);
}

#[test]
fn solve_provider_uses_resident_owner_and_rejects_mixed_owners() {
    let _guard = test_support::accel_test_lock();
    let first = Box::leak(Box::new(
        runmat_accelerate::simple_provider::InProcessProvider::new(),
    ));
    let second = Box::leak(Box::new(
        runmat_accelerate::simple_provider::InProcessProvider::new(),
    ));
    unsafe {
        runmat_accelerate_api::register_provider(first);
        runmat_accelerate_api::register_provider(second);
    }
    let first_handle = first
        .upload(&HostTensorView {
            data: &[1.0, 0.0, 0.0, 1.0],
            shape: &[2, 2],
        })
        .expect("first upload");
    let second_handle = second
        .upload(&HostTensorView {
            data: &[1.0, 0.0, 0.0, 1.0],
            shape: &[2, 2],
        })
        .expect("second upload");

    let error = mldivide_builtin(
        Value::GpuTensor(first_handle.clone()),
        Value::GpuTensor(second_handle.clone()),
    )
    .expect_err("mixed provider solve must reject");
    assert_eq!(error.identifier(), Some("RunMat:gpu:MixedProviders"));

    first.free(&first_handle).expect("free first");
    second.free(&second_handle).expect("free second");
}

#[test]
fn integer_scalar_left_division_preserves_uint64_storage() {
    let values = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]), vec![1, 2])
        .expect("integer values");
    let result = mldivide_builtin(Value::Num(2.0), Value::Tensor(values)).expect("mldivide");
    assert_eq!(
        result,
        Value::Tensor(
            Tensor::new_integer(
                IntegerStorage::U64(vec![1_u64 << 63, 1_u64 << 62]),
                vec![1, 2],
            )
            .expect("integer result")
        )
    );
}

#[test]
fn integer_matrix_left_division_is_rejected() {
    let coefficients = Tensor::new_integer(IntegerStorage::I32(vec![1, 0, 0, 1]), vec![2, 2])
        .expect("integer coefficients");
    let err = mldivide_builtin(Value::Tensor(coefficients), Value::Int(IntValue::I32(1)))
        .expect_err("integer matrix solve must reject");
    assert_eq!(err.identifier(), MLDIVIDE_ERROR_INVALID_INPUT.identifier);
    assert!(err.message().contains("only supported for scalar"));
}

#[test]
fn mldivide_complex_scalar_promotion_reads_typed_integer_storage_exactly() {
    let divisor =
        Tensor::new_integer(IntegerStorage::I64(vec![2]), vec![1, 1]).expect("integer scalar");
    let rhs = ComplexTensor::new(vec![(6.0, 4.0), (2.0, -8.0)], vec![2, 1]).unwrap();

    let result =
        mldivide_builtin(Value::Tensor(divisor), Value::ComplexTensor(rhs)).expect("mldivide");
    let Value::ComplexTensor(out) = result else {
        panic!("expected complex tensor result");
    };
    assert_eq!(out.shape, vec![2, 1]);
    assert!((out.materialize_f64()[0].0 - 3.0).abs() < 1e-12);
    assert!((out.materialize_f64()[0].1 - 2.0).abs() < 1e-12);
    assert!((out.materialize_f64()[1].0 - 1.0).abs() < 1e-12);
    assert!((out.materialize_f64()[1].1 + 4.0).abs() < 1e-12);
}

#[test]
fn mldivide_host_real_reads_typed_integer_storage_exactly() {
    let lhs = Tensor::new_integer(IntegerStorage::I16(vec![2]), vec![1, 1]).expect("typed divisor");
    let rhs = Tensor::new_integer(IntegerStorage::I16(vec![6, 10]), vec![2, 1]).expect("typed rhs");

    let out = mldivide_host_real_for_provider(&lhs, &rhs).expect("host mldivide");

    assert_eq!(out.materialize_f64(), vec![3.0, 5.0]);
    assert!(out.integer_storage().is_none());
}
