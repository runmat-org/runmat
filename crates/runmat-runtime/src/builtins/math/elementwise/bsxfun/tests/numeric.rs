use super::*;

#[test]
fn bsxfun_plus_preserves_all_exact_integer_classes() {
    let cases = [
        (
            IntegerStorage::I8(vec![i8::MAX, -5]),
            IntegerStorage::I8(vec![1]),
            IntegerStorage::I8(vec![i8::MAX, -4]),
        ),
        (
            IntegerStorage::I16(vec![i16::MAX, -5]),
            IntegerStorage::I16(vec![1]),
            IntegerStorage::I16(vec![i16::MAX, -4]),
        ),
        (
            IntegerStorage::I32(vec![i32::MAX, -5]),
            IntegerStorage::I32(vec![1]),
            IntegerStorage::I32(vec![i32::MAX, -4]),
        ),
        (
            IntegerStorage::I64(vec![i64::MAX, -5]),
            IntegerStorage::I64(vec![1]),
            IntegerStorage::I64(vec![i64::MAX, -4]),
        ),
        (
            IntegerStorage::U8(vec![u8::MAX, 5]),
            IntegerStorage::U8(vec![1]),
            IntegerStorage::U8(vec![u8::MAX, 6]),
        ),
        (
            IntegerStorage::U16(vec![u16::MAX, 5]),
            IntegerStorage::U16(vec![1]),
            IntegerStorage::U16(vec![u16::MAX, 6]),
        ),
        (
            IntegerStorage::U32(vec![u32::MAX, 5]),
            IntegerStorage::U32(vec![1]),
            IntegerStorage::U32(vec![u32::MAX, 6]),
        ),
        (
            IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993]),
            IntegerStorage::U64(vec![1]),
            IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_994]),
        ),
    ];
    for (left, right, expected) in cases {
        let left = Tensor::new_integer(left, vec![2, 1]).expect("left");
        let right = Tensor::new_integer(right, vec![1, 1]).expect("right");
        let result = call(
            Value::FunctionHandle("plus".to_string()),
            Value::Tensor(left),
            Value::Tensor(right),
        )
        .expect("integer bsxfun plus");
        let Value::Tensor(result) = result else {
            panic!("expected typed tensor, got {result:?}");
        };
        assert_eq!(result.integer_storage(), Some(&expected));
    }
}

#[test]
fn bsxfun_preserves_native_single_and_integer_predicate_outputs() {
    let single = Tensor::from_f32(vec![1.25, 2.5], vec![2, 1]).expect("single");
    let one = Tensor::from_f32(vec![1.0], vec![1, 1]).expect("single scalar");
    let result = call(
        Value::FunctionHandle("plus".to_string()),
        Value::Tensor(single),
        Value::Tensor(one),
    )
    .expect("single plus");
    let Value::Tensor(result) = result else {
        panic!("expected single tensor");
    };
    assert_eq!(
        result.into_numeric_storage().expect("single storage"),
        NumericStorage::F32(vec![2.25, 3.5])
    );

    let wide = Tensor::new_integer(
        IntegerStorage::U64(vec![9_007_199_254_740_993, 9_007_199_254_740_995]),
        vec![2, 1],
    )
    .expect("wide");
    let threshold =
        Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_994]), vec![1, 1])
            .expect("threshold");
    let result = call(
        Value::FunctionHandle("gt".to_string()),
        Value::Tensor(wide),
        Value::Tensor(threshold),
    )
    .expect("exact integer predicate");
    let Value::LogicalArray(result) = result else {
        panic!("expected logical output");
    };
    assert_eq!(result.data, vec![0, 1]);
}

#[test]
fn bsxfun_empty_known_callbacks_preserve_output_class() {
    let empty =
        Tensor::new_integer(IntegerStorage::U64(Vec::new()), vec![0, 1]).expect("empty integer");
    let scalar = Value::Int(IntValue::U64(9_007_199_254_740_993));
    let result = call(
        Value::FunctionHandle("plus".to_string()),
        Value::Tensor(empty),
        scalar,
    )
    .expect("empty integer plus");
    let Value::Tensor(result) = result else {
        panic!("expected typed empty tensor");
    };
    assert_eq!(
        result.into_numeric_storage().expect("uint64 storage"),
        NumericStorage::U64(Vec::new())
    );
}
