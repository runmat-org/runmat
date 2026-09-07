use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn integer_matrix_times_scalar_preserves_uint64_storage() {
    let a = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]), vec![2, 1])
        .expect("integer matrix");
    let result = mtimes_builtin(Value::Tensor(a), Value::Num(1.0)).expect("mtimes");
    assert_eq!(
        result,
        Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]), vec![2, 1])
                .expect("integer result")
        )
    );
}

#[test]
fn same_class_integer_array_scalar_mtimes_is_exact_in_both_orders() {
    for (array, scalar, expected) in integer_scalar_mtimes_cases() {
        let array = Value::Tensor(Tensor::new_integer(array, vec![1, 3]).expect("integer array"));
        for (lhs, rhs) in [
            (array.clone(), Value::Int(scalar.clone())),
            (
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::from_scalar(scalar.clone()), vec![1, 1])
                        .expect("integer scalar tensor"),
                ),
                array.clone(),
            ),
        ] {
            let result = mtimes_builtin(lhs, rhs).expect("same-class integer scalar mtimes");
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
fn integer_scalar_mtimes_rejects_nonscalar_floating_partner() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let err = mtimes_builtin(Value::Int(IntValue::I32(2)), Value::Tensor(a))
        .expect_err("nonscalar floating partner must reject");
    assert_eq!(err.identifier(), MTIMES_ERROR_INVALID_INPUT.identifier);
    assert!(err
        .message()
        .contains("integer arrays can only be combined with scalar double values"));
}

#[test]
fn integer_mtimes_rejects_matrix_products_and_mixed_integer_classes() {
    let lhs = Value::Tensor(
        Tensor::new_integer(IntegerStorage::I16(vec![1, 2, 3, 4]), vec![2, 2]).expect("lhs"),
    );
    let rhs = Value::Tensor(
        Tensor::new_integer(IntegerStorage::I16(vec![5, 6, 7, 8]), vec![2, 2]).expect("rhs"),
    );
    let error = mtimes_builtin(lhs, rhs).expect_err("integer matrix product must reject");
    assert_eq!(error.identifier(), MTIMES_ERROR_INVALID_INPUT.identifier);
    assert!(error.message().contains("other input must be scalar"));

    let lhs = Value::Tensor(
        Tensor::new_integer(IntegerStorage::I16(vec![1, 2]), vec![1, 2]).expect("lhs"),
    );
    let error = mtimes_builtin(lhs, Value::Int(IntValue::U16(2)))
        .expect_err("mixed integer classes must reject");
    assert_eq!(error.identifier(), MTIMES_ERROR_INVALID_INPUT.identifier);
    assert!(error.message().contains("same integer class"));
}
