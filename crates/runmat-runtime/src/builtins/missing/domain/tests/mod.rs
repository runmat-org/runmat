use super::detection::*;
use super::filling::*;
use super::numeric::*;
use super::removal::*;
use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};
use runmat_value::{IntegerStorage, StructArray};

fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(data, shape).unwrap())
}

async fn rmmissing_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    super::entrypoints::rmmissing_builtin(
        runtime,
        value,
        crate::builtins::common::test_support::builtin_arguments(rest),
    )
    .await
}

fn first_unrepresentable_usize_double() -> f64 {
    if usize::BITS == 64 {
        usize::MAX as f64
    } else {
        (usize::MAX as f64) + 1.0
    }
}

mod core;
mod errors;
mod filling;
mod filling_preflight;
mod moving;
mod nan;
mod removal;
mod standardize;
