use super::super::*;
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn power_scalar_numbers() {
    let result = power_builtin(Value::Num(2.0), Value::Num(3.0), Vec::new()).expect("power");
    match result {
        Value::Num(v) => assert!((v - 8.0).abs() < 1e-12),
        other => panic!("expected scalar numeric result, got {other:?}"),
    }
}

#[test]
fn power_integer_arrays_preserve_storage_and_uint64_precision() {
    let base = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 2, 0]), vec![1, 3])
        .expect("integer base");
    let exponent = Tensor::new_integer(IntegerStorage::U64(vec![1, 64, 0]), vec![1, 3])
        .expect("integer exponent");
    let result =
        power_builtin(Value::Tensor(base), Value::Tensor(exponent), Vec::new()).expect("power");
    assert_eq!(
        result,
        Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, u64::MAX, 1]), vec![1, 3])
                .expect("integer result")
        )
    );
}

#[test]
fn power_integer_scalar_exponent_preserves_exact_uint64_value() {
    let base =
        Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).expect("integer base");
    let result = power_builtin(Value::Tensor(base), Value::Num(1.0), Vec::new()).expect("power");
    assert_eq!(result, Value::Int(IntValue::U64(u64::MAX)));
}

#[test]
fn power_integer_exponent_domain_has_stable_public_error() {
    let base =
        Tensor::new_integer(IntegerStorage::I32(vec![2, -2]), vec![2, 1]).expect("integer base");
    for exponent in [Value::Num(-1.0), Value::Num(0.5), Value::Num(f64::NAN)] {
        let error = power_builtin(Value::Tensor(base.clone()), exponent, Vec::new())
            .expect_err("invalid integer exponent");
        assert_eq!(error.identifier(), Some("RunMat:power:InvalidInput"));
        assert!(error.message().contains("nonnegative integer values"));
    }

    let exponent = Tensor::new_integer(IntegerStorage::I32(vec![2, -1, 0]), vec![1, 3]).unwrap();
    let error = power_builtin(
        Value::Tensor(Tensor::new_integer(IntegerStorage::I32(vec![-2, 0]), vec![2, 1]).unwrap()),
        Value::Tensor(exponent),
        Vec::new(),
    )
    .expect_err("broadcast negative integer exponent");
    assert_eq!(error.identifier(), Some("RunMat:power:InvalidInput"));
    assert!(error.message().contains("nonnegative integer values"));
}

#[test]
fn power_integer_zero_and_negative_base_edges_remain_exact() {
    let base = Tensor::new_integer(IntegerStorage::I16(vec![0, -2, -2]), vec![1, 3]).unwrap();
    let exponent = Tensor::new_integer(IntegerStorage::I16(vec![0, 2, 3]), vec![1, 3]).unwrap();
    let result = power_builtin(Value::Tensor(base), Value::Tensor(exponent), Vec::new()).unwrap();
    assert_eq!(
        result,
        Value::Tensor(
            Tensor::new_integer(IntegerStorage::I16(vec![1, 4, -8]), vec![1, 3]).unwrap()
        )
    );
}
