use super::*;
use runmat_value::{ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage};

#[test]
fn uniform_output_preserves_all_integer_classes() {
    let cases = [
        ("int8", IntValue::I8(i8::MIN)),
        ("int16", IntValue::I16(i16::MIN)),
        ("int32", IntValue::I32(i32::MIN)),
        ("int64", IntValue::I64(i64::MIN)),
        ("uint8", IntValue::U8(u8::MAX)),
        ("uint16", IntValue::U16(u16::MAX)),
        ("uint32", IntValue::U32(u32::MAX)),
        ("uint64", IntValue::U64(u64::MAX)),
    ];
    for (name, value) in cases {
        let result = call(
            Value::FunctionHandle(name.into()),
            vec![cell(vec![Value::Int(value.clone())], &[1, 1])],
        )
        .unwrap();
        let Value::Tensor(output) = result else {
            panic!("expected integer tensor")
        };
        assert_eq!(output.numeric_dtype().class_name(), name);
        assert_eq!(
            output.numeric_value_at(0).unwrap().into_int_value(),
            Some(value)
        );
    }
}

#[test]
fn uniform_output_preserves_single_and_complex_storage() {
    let single = Value::Tensor(Tensor::from_f32(vec![1.25], vec![1, 1]).unwrap());
    let result = call(
        Value::FunctionHandle("single".into()),
        vec![cell(vec![single], &[1, 1])],
    )
    .unwrap();
    let Value::Tensor(output) = result else {
        panic!("expected single tensor")
    };
    assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F32);

    let exact = IntegerComplexStorage::new(
        IntegerStorage::U64(vec![u64::MAX]),
        IntegerStorage::U64(vec![u64::MAX - 1]),
    )
    .unwrap();
    let complex = Value::ComplexTensor(ComplexTensor::new_integer(exact, vec![1, 1]).unwrap());
    let result = call(
        Value::FunctionHandle("conj".into()),
        vec![cell(vec![complex], &[1, 1])],
    )
    .unwrap();
    let Value::ComplexTensor(output) = result else {
        panic!("expected complex tensor")
    };
    assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::U64);
}

#[test]
fn uniform_output_rejects_mixed_integer_classes() {
    let error = call(
        Value::FunctionHandle("abs".into()),
        vec![cell(
            vec![Value::Int(IntValue::I8(1)), Value::Int(IntValue::I16(2))],
            &[1, 2],
        )],
    )
    .unwrap_err();
    assert!(error.message().contains("same data type"));
}
