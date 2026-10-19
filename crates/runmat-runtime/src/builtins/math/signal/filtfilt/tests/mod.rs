//! Public filtfilt contract and provider route witnesses.

use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate_api::{
    handle_precision, provider_for_handle, AccelDownloadFuture, AccelNumericDownloadFuture,
    AccelProvider, GpuTensorHandle, HostTensorView, ProviderPrecision,
};
use runmat_builtins::builtin_function_by_name;
use runmat_value::{ComplexTensor, Tensor};

fn call(b: Value, a: Value, x: Value) -> BuiltinResult<Value> {
    block_on(evaluate(b, a, x))
}

fn tensor_data(value: Value) -> (Vec<f64>, Vec<usize>) {
    let tensor = test_support::gather(value).expect("gather tensor");
    (tensor.materialize_f64(), tensor.shape().to_vec())
}

fn approx(lhs: &[f64], rhs: &[f64], tol: f64) {
    assert_eq!(lhs.len(), rhs.len());
    for (idx, (a, b)) in lhs.iter().zip(rhs.iter()).enumerate() {
        assert!((a - b).abs() <= tol, "mismatch at {idx}: got {a}, want {b}");
    }
}

mod host;
mod provider_output;
mod provider_routes;
