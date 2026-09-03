use super::*;
use futures::executor::block_on;
use runmat_builtins::{LCM_ERROR_INVALID_INPUT, LCM_ERROR_OVERFLOW, LCM_ERROR_SIZE_MISMATCH};
use runmat_value::{IntValue, IntegerStorage, NumericDType, Tensor};

#[test]
fn double_array_and_scalar_preserve_shape() {
    let input = Tensor::new(vec![5.0, 17.0, 10.0, 60.0], vec![2, 2]).unwrap();
    let Value::Tensor(output) =
        block_on(lcm_builtin(Value::Tensor(input), Value::Num(45.0))).unwrap()
    else {
        panic!("expected tensor")
    };
    assert_eq!(output.shape, vec![2, 2]);
    assert_eq!(output.materialize_f64(), vec![45.0, 765.0, 90.0, 180.0]);
}

#[test]
fn preserves_every_native_integer_class_with_double_scalar() {
    let cases = [
        (
            IntegerStorage::I8(vec![6, 10]),
            IntegerStorage::I8(vec![30, 30]),
        ),
        (
            IntegerStorage::I16(vec![6, 10]),
            IntegerStorage::I16(vec![30, 30]),
        ),
        (
            IntegerStorage::I32(vec![6, 10]),
            IntegerStorage::I32(vec![30, 30]),
        ),
        (
            IntegerStorage::I64(vec![6, 10]),
            IntegerStorage::I64(vec![30, 30]),
        ),
        (
            IntegerStorage::U8(vec![6, 10]),
            IntegerStorage::U8(vec![30, 30]),
        ),
        (
            IntegerStorage::U16(vec![6, 10]),
            IntegerStorage::U16(vec![30, 30]),
        ),
        (
            IntegerStorage::U32(vec![6, 10]),
            IntegerStorage::U32(vec![30, 30]),
        ),
        (
            IntegerStorage::U64(vec![6, 10]),
            IntegerStorage::U64(vec![30, 30]),
        ),
    ];
    for (input, expected) in cases {
        let input = Tensor::new_integer(input, vec![1, 2]).unwrap();
        let Value::Tensor(output) =
            block_on(lcm_builtin(Value::Tensor(input), Value::Num(15.0))).unwrap()
        else {
            panic!("expected tensor")
        };
        assert_eq!(output.integer_storage(), Some(&expected));
    }
}

#[test]
fn retains_exact_uint64_values_above_flintmax() {
    let expected = IntegerStorage::U64(vec![9_007_199_254_740_993, 9_007_199_254_740_995]);
    let input = Tensor::new_integer(expected.clone(), vec![1, 2]).unwrap();
    let Value::Tensor(output) =
        block_on(lcm_builtin(Value::Tensor(input), Value::Num(1.0))).unwrap()
    else {
        panic!("expected tensor")
    };
    assert_eq!(output.integer_storage(), Some(&expected));
}

#[test]
fn rejects_invalid_domains_shapes_classes_and_overflow() {
    for value in [
        Value::Num(0.0),
        Value::Num(-2.0),
        Value::Num(2.5),
        Value::Complex(2.0, 0.0),
    ] {
        let error = block_on(lcm_builtin(value, Value::Num(3.0))).unwrap_err();
        assert_eq!(error.identifier(), LCM_ERROR_INVALID_INPUT.identifier);
    }
    let left = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
    let right = Tensor::new(vec![5.0, 7.0, 11.0], vec![1, 3]).unwrap();
    let error = block_on(lcm_builtin(Value::Tensor(left), Value::Tensor(right))).unwrap_err();
    assert_eq!(error.identifier(), LCM_ERROR_SIZE_MISMATCH.identifier);
    let error = block_on(lcm_builtin(
        Value::Int(IntValue::U8(2)),
        Value::Int(IntValue::U16(4)),
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), LCM_ERROR_INVALID_INPUT.identifier);
    let error = block_on(lcm_builtin(
        Value::Int(IntValue::U8(200)),
        Value::Int(IntValue::U8(201)),
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), LCM_ERROR_OVERFLOW.identifier);
}

#[test]
fn preserves_native_single_storage() {
    let left = Tensor::from_f32(vec![6.0, 10.0], vec![1, 2]).unwrap();
    let right = Tensor::new_with_dtype(vec![15.0, 21.0], vec![1, 2], NumericDType::F64).unwrap();
    let Value::Tensor(output) =
        block_on(lcm_builtin(Value::Tensor(left), Value::Tensor(right))).unwrap()
    else {
        panic!("expected tensor")
    };
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
    assert_eq!(output.as_f32_slice(), Some([30.0, 210.0].as_slice()));
}
