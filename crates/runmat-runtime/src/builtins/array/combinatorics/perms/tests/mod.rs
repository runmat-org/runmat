mod host;
mod provider;

use super::*;
use crate::BuiltinResult;
use futures::executor::block_on;
use runmat_value::{ComplexTensor, Tensor, Value};

fn call(value: Value) -> BuiltinResult<Value> {
    block_on(perms_builtin(value, Vec::new()))
}

fn tensor_rows(tensor: &Tensor) -> Vec<Vec<f64>> {
    (0..tensor.rows)
        .map(|row| {
            (0..tensor.cols)
                .map(|column| tensor.materialize_f64()[column * tensor.rows + row])
                .collect()
        })
        .collect()
}

fn complex_rows(tensor: &ComplexTensor) -> Vec<Vec<(f64, f64)>> {
    (0..tensor.rows)
        .map(|row| {
            (0..tensor.cols)
                .map(|column| tensor.materialize_f64()[column * tensor.rows + row])
                .collect()
        })
        .collect()
}
