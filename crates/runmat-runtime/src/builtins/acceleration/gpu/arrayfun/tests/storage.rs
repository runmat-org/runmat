use super::*;

#[test]
fn typed_integer_input_elements_are_extracted_from_exact_storage() {
    let tensor = Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
        .expect("integer tensor");
    let input = ArrayInput {
        data: ArrayData::Tensor(tensor),
        shape: vec![1, 1],
        strides: vec![1, 1],
    };

    assert_eq!(
        input.value_at(0, &[1, 1]).expect("value"),
        Value::Int(runmat_value::IntValue::U64(9_007_199_254_740_993))
    );
}

#[test]
fn uniform_classifier_preserves_typed_integer_tensor_storage() {
    let tensor = Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
        .expect("integer tensor");

    match classify_value(&Value::Tensor(tensor)).expect("classify") {
        ClassifiedValue::Integer(value) => {
            assert_eq!(value, IntValue::U64(9_007_199_254_740_993));
        }
        _ => panic!("expected integer classification"),
    }
}

#[test]
fn uniform_classifier_preserves_wide_integer_scalar() {
    match classify_value(&Value::Int(IntValue::I64(i64::MIN))).expect("classify") {
        ClassifiedValue::Integer(value) => assert_eq!(value, IntValue::I64(i64::MIN)),
        _ => panic!("expected integer classification"),
    }
}

#[test]
fn uniform_collector_preserves_every_integer_class() {
    let cases = [
        (
            IntValue::I8(i8::MIN),
            IntValue::I8(i8::MAX),
            IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
        ),
        (
            IntValue::I16(i16::MIN),
            IntValue::I16(i16::MAX),
            IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
        ),
        (
            IntValue::I32(i32::MIN),
            IntValue::I32(i32::MAX),
            IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
        ),
        (
            IntValue::I64(i64::MIN),
            IntValue::I64(i64::MAX),
            IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
        ),
        (
            IntValue::U8(u8::MIN),
            IntValue::U8(u8::MAX),
            IntegerStorage::U8(vec![u8::MIN, u8::MAX]),
        ),
        (
            IntValue::U16(u16::MIN),
            IntValue::U16(u16::MAX),
            IntegerStorage::U16(vec![u16::MIN, u16::MAX]),
        ),
        (
            IntValue::U32(u32::MIN),
            IntValue::U32(u32::MAX),
            IntegerStorage::U32(vec![u32::MIN, u32::MAX]),
        ),
        (
            IntValue::U64(9_007_199_254_740_993),
            IntValue::U64(u64::MAX),
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
        ),
    ];

    for (first, second, expected) in cases {
        let mut collector = UniformCollector::Pending;
        collector.push(&Value::Int(first)).expect("first value");
        collector.push(&Value::Int(second)).expect("second value");
        let Value::Tensor(tensor) = collector.finish(&[1, 2]).expect("finish") else {
            panic!("expected integer tensor");
        };
        assert_eq!(tensor.integer_storage(), Some(&expected));
    }
}

#[test]
fn uniform_collector_rejects_mixed_integer_classes() {
    let mut collector = UniformCollector::Pending;
    collector
        .push(&Value::Int(IntValue::U8(1)))
        .expect("first value");
    let error = collector
        .push(&Value::Int(IntValue::U16(2)))
        .expect_err("mixed integer classes must fail");
    assert_eq!(
        error.identifier(),
        ARRAYFUN_ERROR_UNIFORM_OUTPUT_TYPE.identifier
    );
}

#[test]
fn arrayfun_preserves_every_integer_input_and_uniform_output_class() {
    for (storage, callback) in [
        (IntegerStorage::I8(vec![i8::MIN, i8::MAX]), "int8"),
        (IntegerStorage::I16(vec![i16::MIN, i16::MAX]), "int16"),
        (IntegerStorage::I32(vec![i32::MIN, i32::MAX]), "int32"),
        (IntegerStorage::I64(vec![i64::MIN, i64::MAX]), "int64"),
        (IntegerStorage::U8(vec![0, u8::MAX]), "uint8"),
        (IntegerStorage::U16(vec![0, u16::MAX]), "uint16"),
        (IntegerStorage::U32(vec![0, u32::MAX]), "uint32"),
        (
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            "uint64",
        ),
    ] {
        let result = call(
            Value::FunctionHandle(callback.to_string()),
            vec![Value::Tensor(
                Tensor::new_integer(storage.clone(), vec![1, 2]).expect("integer input"),
            )],
        )
        .expect("arrayfun integer identity cast");
        let Value::Tensor(result) = result else {
            panic!("expected typed integer tensor");
        };
        assert_eq!(result.integer_storage(), Some(&storage));
    }
}
