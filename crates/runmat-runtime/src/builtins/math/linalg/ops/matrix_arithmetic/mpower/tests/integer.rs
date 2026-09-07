use super::*;

#[test]
fn typed_mpower_exponent_parser_rejects_unrepresentable_uint64() {
    assert_eq!(
        exponent::parse(&Value::Int(IntValue::I64(-1))).expect("signed exponent"),
        Some(-1)
    );
    let err = exponent::parse(&Value::Int(IntValue::U64(u64::MAX)))
        .expect_err("unrepresentable typed exponent must not saturate");
    assert_eq!(err.identifier(), MPOWER_ERROR_INVALID_ARGUMENT.identifier);

    let exponent =
        Tensor::new_integer(IntegerStorage::I16(vec![2]), vec![1, 1]).expect("typed exponent");
    assert_eq!(
        exponent::parse(&Value::Tensor(exponent)).expect("typed tensor exponent"),
        Some(2)
    );

    let wide = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
        .expect("wide typed exponent");
    let err = exponent::parse(&Value::Tensor(wide))
        .expect_err("unrepresentable typed tensor exponent must not use f64 mirror");
    assert_eq!(err.identifier(), MPOWER_ERROR_INVALID_ARGUMENT.identifier);
}

#[test]
fn scalar_integer_mpower_preserves_exact_uint64_storage() {
    let result =
        mpower_builtin(Value::Int(IntValue::U64(u64::MAX)), Value::Num(1.0)).expect("mpower");
    assert_eq!(result, Value::Int(IntValue::U64(u64::MAX)));

    let base = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
        .expect("integer scalar tensor");
    let result = mpower_builtin(Value::Tensor(base), Value::Num(1.0)).expect("mpower");
    assert_eq!(result, Value::Int(IntValue::U64(u64::MAX)));

    let result = mpower_builtin(Value::Num(2.0), Value::Int(IntValue::U8(3))).expect("mpower");
    assert_eq!(result, Value::Int(IntValue::U8(8)));
}

#[test]
fn matrix_integer_mpower_preserves_every_class() {
    let cases = [
        (
            IntegerStorage::I8(vec![1, 2, 3, 4]),
            IntegerStorage::I8(vec![7, 10, 15, 22]),
        ),
        (
            IntegerStorage::I16(vec![1, 2, 3, 4]),
            IntegerStorage::I16(vec![7, 10, 15, 22]),
        ),
        (
            IntegerStorage::I32(vec![1, 2, 3, 4]),
            IntegerStorage::I32(vec![7, 10, 15, 22]),
        ),
        (
            IntegerStorage::I64(vec![1, 2, 3, 4]),
            IntegerStorage::I64(vec![7, 10, 15, 22]),
        ),
        (
            IntegerStorage::U8(vec![1, 2, 3, 4]),
            IntegerStorage::U8(vec![7, 10, 15, 22]),
        ),
        (
            IntegerStorage::U16(vec![1, 2, 3, 4]),
            IntegerStorage::U16(vec![7, 10, 15, 22]),
        ),
        (
            IntegerStorage::U32(vec![1, 2, 3, 4]),
            IntegerStorage::U32(vec![7, 10, 15, 22]),
        ),
        (
            IntegerStorage::U64(vec![1, 2, 3, 4]),
            IntegerStorage::U64(vec![7, 10, 15, 22]),
        ),
    ];
    for (input, expected) in cases {
        let matrix = Tensor::new_integer(input, vec![2, 2]).expect("integer matrix");
        let result =
            mpower_builtin(Value::Tensor(matrix), Value::Num(2.0)).expect("integer matrix power");
        let Value::Tensor(result) = result else {
            panic!("expected integer matrix");
        };
        assert_eq!(result.integer_storage(), Some(&expected));
    }
}

#[test]
fn matrix_integer_mpower_saturates_in_destination_class() {
    let matrix = Tensor::new_integer(IntegerStorage::I8(vec![100; 4]), vec![2, 2]).unwrap();
    let result =
        mpower_builtin(Value::Tensor(matrix), Value::Num(2.0)).expect("integer matrix power");
    let Value::Tensor(result) = result else {
        panic!("expected integer matrix");
    };
    assert_eq!(
        result.integer_storage(),
        Some(&IntegerStorage::I8(vec![i8::MAX; 4]))
    );
}
