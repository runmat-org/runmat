use super::*;

#[test]
fn complex_single_scalar_right_division_preserves_single_storage() {
    let lhs = ComplexTensor::from_f32(vec![(8.0, 4.0), (12.0, 6.0)], vec![1, 2]).expect("lhs");
    let divisor = ComplexTensor::from_f32(vec![(2.0, 1.0)], vec![1, 1]).expect("divisor");
    let Value::ComplexTensor(result) =
        mrdivide_builtin(Value::ComplexTensor(lhs), Value::ComplexTensor(divisor))
            .expect("complex single scalar solve")
    else {
        panic!("expected complex tensor")
    };
    assert_eq!(result.numeric_dtype(), runmat_value::NumericDType::F32);
    assert_eq!(result.shape, vec![1, 2]);
}

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
    let input =
        Tensor::new_integer(IntegerStorage::U64(vec![8, 12]), vec![1, 2]).expect("integer row");
    let input_handle = gpu_helpers::upload_tensor(owner, &input).expect("owner upload");

    let result = mrdivide_builtin(
        Value::GpuTensor(input_handle.clone()),
        Value::Int(IntValue::U64(2)),
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

    owner.free(&input_handle).expect("free input");
    owner.free(&result_handle).expect("free result");
}

#[test]
fn integer_matrix_right_division_by_scalar_preserves_uint64_storage() {
    let values = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]), vec![1, 2])
        .expect("integer values");
    let result = mrdivide_builtin(Value::Tensor(values), Value::Num(1.0)).expect("mrdivide");
    assert_eq!(
        result,
        Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]), vec![1, 2])
                .expect("integer result")
        )
    );
}

#[test]
fn integer_array_scalar_right_division_is_exact_for_all_classes() {
    for (array, scalar, expected) in integer_scalar_mrdivide_cases() {
        let array = Value::Tensor(Tensor::new_integer(array, vec![1, 3]).expect("integer array"));
        for divisor in [
            Value::Int(scalar.clone()),
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::from_scalar(scalar), vec![1, 1])
                    .expect("integer scalar tensor"),
            ),
            Value::Num(2.0),
        ] {
            let result = mrdivide_builtin(array.clone(), divisor).expect("integer scalar mrdivide");
            assert_eq!(
                result,
                Value::Tensor(
                    Tensor::new_integer(expected.clone(), vec![1, 3]).expect("integer result")
                )
            );
        }
    }
}

#[test]
fn integer_scalar_divisor_rejects_nonscalar_double_numerator() {
    let numerator = Tensor::new(vec![2.0, 4.0], vec![1, 2]).expect("numerator");
    let err = mrdivide_builtin(Value::Tensor(numerator), Value::Int(IntValue::I32(2)))
        .expect_err("integer scalar divisor needs scalar numerator");
    assert_eq!(err.identifier(), MRDIVIDE_ERROR_INVALID_INPUT.identifier);
    assert!(err
        .message()
        .contains("integer arrays can only be combined with scalar double values"));
}

#[test]
fn integer_mrdivide_rejects_nonscalar_divisors_and_mixed_classes() {
    let lhs = Value::Tensor(
        Tensor::new_integer(IntegerStorage::I16(vec![6, 4]), vec![1, 2]).expect("lhs"),
    );
    let rhs = Value::Tensor(
        Tensor::new_integer(IntegerStorage::I16(vec![2, 2]), vec![1, 2]).expect("rhs"),
    );
    let error = mrdivide_builtin(lhs, rhs).expect_err("nonscalar integer divisor must reject");
    assert_eq!(error.identifier(), MRDIVIDE_ERROR_INVALID_INPUT.identifier);
    assert!(error.message().contains("scalar right division"));

    let lhs = Value::Tensor(
        Tensor::new_integer(IntegerStorage::I16(vec![6, 4]), vec![1, 2]).expect("lhs"),
    );
    let error = mrdivide_builtin(lhs, Value::Int(IntValue::U16(2)))
        .expect_err("mixed integer classes must reject");
    assert_eq!(error.identifier(), MRDIVIDE_ERROR_INVALID_INPUT.identifier);
    assert!(error.message().contains("same integer class"));
}

#[test]
fn mrdivide_complex_scalar_promotion_reads_typed_integer_storage_exactly() {
    let lhs = ComplexTensor::new(vec![(6.0, 4.0), (2.0, -8.0)], vec![1, 2]).unwrap();
    let divisor =
        Tensor::new_integer(IntegerStorage::I64(vec![2]), vec![1, 1]).expect("integer scalar");

    let result =
        mrdivide_builtin(Value::ComplexTensor(lhs), Value::Tensor(divisor)).expect("mrdivide");
    let Value::ComplexTensor(out) = result else {
        panic!("expected complex tensor result");
    };
    assert_eq!(out.shape, vec![1, 2]);
    assert!((out.materialize_f64()[0].0 - 3.0).abs() < 1e-12);
    assert!((out.materialize_f64()[0].1 - 2.0).abs() < 1e-12);
    assert!((out.materialize_f64()[1].0 - 1.0).abs() < 1e-12);
    assert!((out.materialize_f64()[1].1 + 4.0).abs() < 1e-12);
}

#[test]
fn mrdivide_host_real_reads_typed_integer_storage_exactly() {
    let lhs = Tensor::new_integer(IntegerStorage::I16(vec![6, 10]), vec![1, 2]).expect("typed lhs");
    let rhs = Tensor::new_integer(IntegerStorage::I16(vec![2]), vec![1, 1]).expect("typed divisor");

    let out = mrdivide_host_real_for_provider(&lhs, &rhs).expect("host mrdivide");

    assert_eq!(out.materialize_f64(), vec![3.0, 5.0]);
    assert!(out.integer_storage().is_none());
}
