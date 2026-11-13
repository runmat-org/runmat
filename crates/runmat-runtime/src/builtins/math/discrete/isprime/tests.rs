use super::*;
use futures::executor::block_on;
use runmat_value::{IntValue, IntegerStorage, LogicalArray, NumericStorage, Tensor};

fn call(value: Value) -> BuiltinResult<Value> {
    block_on(isprime_builtin(
        value,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
}

#[test]
fn classifies_scalars_and_full_width_integers() {
    assert_eq!(call(Value::Num(2.0)).unwrap(), Value::Bool(true));
    assert_eq!(
        call(Value::Int(IntValue::U64(u64::MAX))).unwrap(),
        Value::Bool(false)
    );
    assert_eq!(
        call(Value::Int(IntValue::U64(18_446_744_073_709_551_557))).unwrap(),
        Value::Bool(true)
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn tensor_inputs_preserve_rank_zero_and_rank_one_logical_shapes() {
    for (input, shape, expected) in [
        (
            Tensor::from_f32(vec![2.0], vec![]).expect("rank-zero single"),
            Vec::new(),
            vec![1],
        ),
        (
            Tensor::new_integer(IntegerStorage::U64(vec![2, 4]), vec![2])
                .expect("rank-one integer"),
            vec![2],
            vec![1, 0],
        ),
    ] {
        let Value::LogicalArray(output) = call(Value::Tensor(input)).expect("isprime") else {
            panic!("expected logical array");
        };
        assert_eq!(output.shape, shape);
        assert_eq!(output.data.as_slice(), expected.as_slice());
    }
}

#[test]
fn supports_every_numeric_storage_class_and_preserves_shape() {
    let storages = [
        NumericStorage::F64(vec![2.0, 4.0]),
        NumericStorage::F32(vec![2.0, 4.0]),
        NumericStorage::I8(vec![2, 4]),
        NumericStorage::I16(vec![2, 4]),
        NumericStorage::I32(vec![2, 4]),
        NumericStorage::I64(vec![2, 4]),
        NumericStorage::U8(vec![2, 4]),
        NumericStorage::U16(vec![2, 4]),
        NumericStorage::U32(vec![2, 4]),
        NumericStorage::U64(vec![2, 4]),
    ];
    for storage in storages {
        let input = Tensor::from_numeric_storage(storage, vec![2, 1]).unwrap();
        let Value::LogicalArray(out) = call(Value::Tensor(input)).unwrap() else {
            panic!()
        };
        assert_eq!(out, LogicalArray::new(vec![1, 0], vec![2, 1]).unwrap());
    }
}

#[test]
fn supports_empty_and_rejects_invalid_inputs_and_arity() {
    let input = Tensor::from_f32(Vec::new(), vec![0, 3]).unwrap();
    let Value::LogicalArray(out) = call(Value::Tensor(input)).unwrap() else {
        panic!()
    };
    assert_eq!(out.shape, vec![0, 3]);
    for value in [
        Value::Num(-1.0),
        Value::Num(2.5),
        Value::Num(u64::MAX as f64),
        Value::Complex(2.0, 0.0),
    ] {
        assert!(call(value).is_err());
    }
    assert!(block_on(isprime_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(3.0)])
    ))
    .is_err());
}
