use super::super::statistical_means::{geomean, harmmean, rms, rmse};
use super::flagged::{kurtosis, mad, skewness};
use super::tabulation::tabulate;
use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_value::{
    IntValue, IntegerComplexStorage, IntegerStorage, NumericDType, StringArray, Tensor,
};

mod means_regression;
mod moments;
mod tabulation;

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-10,
        "expected {expected}, got {actual}"
    );
}

fn assert_extension_error(result: BuiltinResult<Value>, identifier: &'static str) {
    let error = result.unwrap_err();
    assert_eq!(error.identifier(), Some(identifier));
}

fn tensor_values(value: Value) -> (Vec<f64>, Vec<usize>) {
    match value {
        Value::Num(num) => (vec![num], vec![1, 1]),
        Value::Tensor(tensor) => (tensor.materialize_f64(), tensor.shape().to_vec()),
        other => panic!("expected numeric output, got {other:?}"),
    }
}

fn int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).unwrap();
    Value::Tensor(tensor)
}

fn mirrorless_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).unwrap();
    Value::Tensor(tensor)
}

fn complex_int_tensor(real: IntegerStorage, imag: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = runmat_value::ComplexTensor::new_integer(
        IntegerComplexStorage::new(real, imag).unwrap(),
        shape,
    )
    .unwrap();
    Value::ComplexTensor(tensor)
}
