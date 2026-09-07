use super::*;
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate_api::GpuTensorStorage;
use runmat_builtins::RDIVIDE_DESCRIPTOR;
use runmat_value::{
    CharArray, ComplexTensor, IntValue, IntegerStorage, LogicalArray, NumericStorage, Tensor,
};

const EPS: f64 = 1e-12;

fn rdivide_builtin(lhs: Value, rhs: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    block_on(super::rdivide_builtin(lhs, rhs, rest))
}

fn double_values(tensor: &Tensor) -> &[f64] {
    tensor.as_f64_slice().expect("double tensor")
}

mod contract;
mod host;
mod provider;
mod wgpu;
