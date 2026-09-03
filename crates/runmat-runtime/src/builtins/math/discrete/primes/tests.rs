use futures::executor::block_on;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{PRIMES_ERROR_INVALID_INPUT, PRIMES_ERROR_LIMIT};
use runmat_value::{IntValue, IntegerStorage, NumericDType, Tensor, Value};

use super::arguments::MAX_PRIMES_LIMIT;
use super::primes_builtin;
use crate::BuiltinResult;

fn call(value: Value) -> BuiltinResult<Tensor> {
    match block_on(primes_builtin(value, Vec::new()))? {
        Value::Tensor(tensor) => Ok(tensor),
        other => panic!("expected tensor output, got {other:?}"),
    }
}

#[test]
fn returns_double_row_and_empty_rows() {
    let output = call(Value::Num(25.0)).expect("primes");
    assert_eq!(output.shape, vec![1, 9]);
    assert_eq!(output.numeric_dtype(), NumericDType::F64);
    assert_eq!(
        output.materialize_f64(),
        vec![2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0]
    );
    for value in [Value::Num(1.0), Value::Num(0.0), Value::Num(-8.0)] {
        let output = call(value).expect("empty primes");
        assert_eq!(output.shape, vec![1, 0]);
    }
}

#[test]
fn preserves_floating_and_all_integer_classes() {
    let single = call(Value::Tensor(
        Tensor::from_f32(vec![12.0], vec![1, 1]).unwrap(),
    ))
    .expect("single primes");
    assert_eq!(single.numeric_dtype(), NumericDType::F32);

    let expected = vec![2, 3, 5, 7, 11];
    let cases = [
        (
            IntValue::I8(12),
            IntegerStorage::I8(expected.iter().map(|&v| v as i8).collect()),
        ),
        (
            IntValue::I16(12),
            IntegerStorage::I16(expected.iter().map(|&v| v as i16).collect()),
        ),
        (IntValue::I32(12), IntegerStorage::I32(expected.clone())),
        (
            IntValue::I64(12),
            IntegerStorage::I64(expected.iter().map(|&v| i64::from(v)).collect()),
        ),
        (
            IntValue::U8(12),
            IntegerStorage::U8(expected.iter().map(|&v| v as u8).collect()),
        ),
        (
            IntValue::U16(12),
            IntegerStorage::U16(expected.iter().map(|&v| v as u16).collect()),
        ),
        (
            IntValue::U32(12),
            IntegerStorage::U32(expected.iter().map(|&v| v as u32).collect()),
        ),
        (
            IntValue::U64(12),
            IntegerStorage::U64(expected.iter().map(|&v| v as u64).collect()),
        ),
    ];
    for (input, expected_storage) in cases {
        let output = call(Value::Int(input)).expect("integer primes");
        assert_eq!(output.shape, vec![1, 5]);
        assert_eq!(output.integer_storage(), Some(&expected_storage));
    }
}

#[test]
fn rejects_invalid_values_shapes_and_limits() {
    let nonscalar = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
    for value in [
        Value::Tensor(nonscalar),
        Value::Num(10.5),
        Value::Num(f64::INFINITY),
        Value::Bool(true),
        Value::Num(u64::MAX as f64),
    ] {
        let error = block_on(primes_builtin(value, Vec::new())).expect_err("invalid primes input");
        assert_eq!(error.identifier(), PRIMES_ERROR_INVALID_INPUT.identifier);
    }
    let error = block_on(primes_builtin(
        Value::Num((MAX_PRIMES_LIMIT + 1) as f64),
        Vec::new(),
    ))
    .expect_err("bounded sieve limit");
    assert_eq!(error.identifier(), PRIMES_ERROR_LIMIT.identifier);
}

#[test]
fn rejects_nonscalar_gpu_handle_before_gather() {
    let handle = GpuTensorHandle {
        shape: vec![1, 2],
        device_id: 0,
        buffer_id: 999,
        descriptor: Default::default(),
    };
    let error = block_on(primes_builtin(Value::GpuTensor(handle), Vec::new()))
        .expect_err("nonscalar GPU handle");
    assert_eq!(error.identifier(), PRIMES_ERROR_INVALID_INPUT.identifier);
}
