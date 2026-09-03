use super::*;
use futures::executor::block_on;
use runmat_builtins::{GCD_ERROR_INVALID_INPUT, GCD_ERROR_SIZE_MISMATCH};
use runmat_value::{IntValue, IntegerStorage, NumericDType, Tensor};

#[test]
fn handles_negative_and_zero_values() {
    let left = Tensor::new(vec![-5.0, 17.0, 10.0, 0.0], vec![2, 2]).unwrap();
    let right = Tensor::new(vec![-15.0, 3.0, 100.0, 0.0], vec![2, 2]).unwrap();
    let Value::Tensor(output) =
        block_on(gcd_builtin(Value::Tensor(left), Value::Tensor(right))).unwrap()
    else {
        panic!("expected tensor")
    };
    assert_eq!(output.shape, vec![2, 2]);
    assert_eq!(output.materialize_f64(), vec![5.0, 1.0, 10.0, 0.0]);
}

#[test]
fn preserves_all_native_integer_classes_with_double_scalar() {
    let cases = [
        (
            IntegerStorage::I8(vec![-12, 18]),
            IntegerStorage::I8(vec![6, 6]),
        ),
        (
            IntegerStorage::I16(vec![-12, 18]),
            IntegerStorage::I16(vec![6, 6]),
        ),
        (
            IntegerStorage::I32(vec![-12, 18]),
            IntegerStorage::I32(vec![6, 6]),
        ),
        (
            IntegerStorage::I64(vec![-12, 18]),
            IntegerStorage::I64(vec![6, 6]),
        ),
        (
            IntegerStorage::U8(vec![12, 18]),
            IntegerStorage::U8(vec![6, 6]),
        ),
        (
            IntegerStorage::U16(vec![12, 18]),
            IntegerStorage::U16(vec![6, 6]),
        ),
        (
            IntegerStorage::U32(vec![12, 18]),
            IntegerStorage::U32(vec![6, 6]),
        ),
        (
            IntegerStorage::U64(vec![12, 18]),
            IntegerStorage::U64(vec![6, 6]),
        ),
    ];
    for (input, expected) in cases {
        let input = Tensor::new_integer(input, vec![1, 2]).unwrap();
        let Value::Tensor(output) =
            block_on(gcd_builtin(Value::Tensor(input), Value::Num(6.0))).unwrap()
        else {
            panic!("expected tensor")
        };
        assert_eq!(output.integer_storage(), Some(&expected));
    }
}

#[test]
fn extended_outputs_preserve_signed_class_and_bezout_identity() {
    let _guard = crate::output_count::push_output_count(Some(3));
    let left = Tensor::new_integer(IntegerStorage::I16(vec![30, -81]), vec![1, 2]).unwrap();
    let right = Tensor::new_integer(IntegerStorage::I16(vec![56, 57]), vec![1, 2]).unwrap();
    let Value::OutputList(outputs) =
        block_on(gcd_builtin(Value::Tensor(left), Value::Tensor(right))).unwrap()
    else {
        panic!("expected output list")
    };
    let storages = outputs
        .iter()
        .map(|value| match value {
            Value::Tensor(tensor) => tensor.integer_storage().cloned().unwrap(),
            _ => panic!("expected tensor"),
        })
        .collect::<Vec<_>>();
    let (IntegerStorage::I16(divisors), IntegerStorage::I16(first), IntegerStorage::I16(second)) =
        (&storages[0], &storages[1], &storages[2])
    else {
        panic!("expected int16 outputs")
    };
    assert_eq!(divisors, &[2, 3]);
    for index in 0..2 {
        assert_eq!(
            [30i32, -81][index] * i32::from(first[index])
                + [56i32, 57][index] * i32::from(second[index]),
            i32::from(divisors[index])
        );
    }
}

#[test]
fn preserves_single_and_rejects_invalid_inputs() {
    let left = Tensor::from_f32(vec![12.0, 18.0], vec![1, 2]).unwrap();
    let right = Tensor::new(vec![8.0, 30.0], vec![1, 2]).unwrap();
    let Value::Tensor(output) =
        block_on(gcd_builtin(Value::Tensor(left), Value::Tensor(right))).unwrap()
    else {
        panic!("expected tensor")
    };
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
    assert_eq!(output.as_f32_slice(), Some([4.0, 6.0].as_slice()));

    let error = block_on(gcd_builtin(Value::Num(2.5), Value::Num(1.0))).unwrap_err();
    assert_eq!(error.identifier(), GCD_ERROR_INVALID_INPUT.identifier);
    let left = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
    let right = Tensor::new(vec![2.0, 3.0, 4.0], vec![1, 3]).unwrap();
    let error = block_on(gcd_builtin(Value::Tensor(left), Value::Tensor(right))).unwrap_err();
    assert_eq!(error.identifier(), GCD_ERROR_SIZE_MISMATCH.identifier);
}

#[test]
fn extended_outputs_reject_unsigned_classes_and_excess_count() {
    {
        let _guard = crate::output_count::push_output_count(Some(2));
        let error = block_on(gcd_builtin(
            Value::Int(IntValue::U16(30)),
            Value::Int(IntValue::U16(56)),
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), GCD_ERROR_INVALID_INPUT.identifier);
    }
    let _guard = crate::output_count::push_output_count(Some(4));
    assert!(block_on(gcd_builtin(Value::Num(30.0), Value::Num(56.0))).is_err());
}
