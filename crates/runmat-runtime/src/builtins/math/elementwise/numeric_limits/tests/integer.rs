use runmat_value::{ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, Tensor, Value};

use super::{intmax, intmin};

#[test]
fn supports_common_classes() {
    assert_eq!(
        intmax(Vec::new()).unwrap(),
        Value::Int(IntValue::I32(i32::MAX))
    );
    assert_eq!(
        intmin(vec![Value::from("uint16")]).unwrap(),
        Value::Int(IntValue::U16(0))
    );
    assert_eq!(
        intmax(vec![Value::from("uint32")]).unwrap(),
        Value::Int(IntValue::U32(u32::MAX))
    );
}

#[test]
fn supports_all_class_names_and_exact_wide_bounds() {
    let cases = [
        ("int8", IntValue::I8(i8::MIN), IntValue::I8(i8::MAX)),
        ("int16", IntValue::I16(i16::MIN), IntValue::I16(i16::MAX)),
        ("int32", IntValue::I32(i32::MIN), IntValue::I32(i32::MAX)),
        ("int64", IntValue::I64(i64::MIN), IntValue::I64(i64::MAX)),
        ("uint8", IntValue::U8(0), IntValue::U8(u8::MAX)),
        ("uint16", IntValue::U16(0), IntValue::U16(u16::MAX)),
        ("uint32", IntValue::U32(0), IntValue::U32(u32::MAX)),
        ("uint64", IntValue::U64(0), IntValue::U64(u64::MAX)),
    ];

    for (class, minimum, maximum) in cases {
        assert_eq!(
            intmin(vec![Value::from(class)]).unwrap(),
            Value::Int(minimum)
        );
        assert_eq!(
            intmax(vec![Value::from(class)]).unwrap(),
            Value::Int(maximum)
        );
    }
}

#[test]
fn like_copies_every_prototype_class() {
    let prototypes = [
        (
            IntegerStorage::I8(vec![7]),
            IntValue::I8(i8::MIN),
            IntValue::I8(i8::MAX),
        ),
        (
            IntegerStorage::I16(vec![7]),
            IntValue::I16(i16::MIN),
            IntValue::I16(i16::MAX),
        ),
        (
            IntegerStorage::I32(vec![7]),
            IntValue::I32(i32::MIN),
            IntValue::I32(i32::MAX),
        ),
        (
            IntegerStorage::I64(vec![9_007_199_254_740_993]),
            IntValue::I64(i64::MIN),
            IntValue::I64(i64::MAX),
        ),
        (
            IntegerStorage::U8(vec![7]),
            IntValue::U8(0),
            IntValue::U8(u8::MAX),
        ),
        (
            IntegerStorage::U16(vec![7]),
            IntValue::U16(0),
            IntValue::U16(u16::MAX),
        ),
        (
            IntegerStorage::U32(vec![7]),
            IntValue::U32(0),
            IntValue::U32(u32::MAX),
        ),
        (
            IntegerStorage::U64(vec![9_007_199_254_740_993]),
            IntValue::U64(0),
            IntValue::U64(u64::MAX),
        ),
    ];

    for (storage, minimum, maximum) in prototypes {
        let prototype = Tensor::new_integer(storage, vec![1, 1]).unwrap();
        assert_eq!(
            intmin(vec![Value::from("like"), Value::Tensor(prototype.clone())]).unwrap(),
            Value::Int(minimum)
        );
        assert_eq!(
            intmax(vec![Value::from("like"), Value::Tensor(prototype)]).unwrap(),
            Value::Int(maximum)
        );
    }
}

#[test]
fn like_copies_complexity_without_losing_uint64_max() {
    let prototype = ComplexTensor::new_integer(
        IntegerComplexStorage::new(
            IntegerStorage::U64(vec![9_007_199_254_740_993]),
            IntegerStorage::U64(vec![1]),
        )
        .unwrap(),
        vec![1, 1],
    )
    .unwrap();

    let output = intmax(vec![Value::from("like"), Value::ComplexTensor(prototype)]).unwrap();
    let Value::ComplexTensor(output) = output else {
        panic!("expected complex integer scalar")
    };
    assert_eq!(output.shape, vec![1, 1]);
    assert_eq!(
        output.integer_storage().cloned(),
        Some(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![u64::MAX]),
                IntegerStorage::U64(vec![0]),
            )
            .unwrap()
        )
    );
}

#[test]
fn like_rejects_noninteger_prototypes_and_bad_syntax() {
    assert!(intmin(vec![Value::from("like"), Value::Num(0.0)]).is_err());
    assert!(intmax(vec![Value::Bool(true)]).is_err());
    assert!(intmax(vec![Value::from("uint8"), Value::from("uint16")]).is_err());
    assert!(intmin(vec![Value::from("int")]).is_err());
}
