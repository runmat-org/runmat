use super::*;
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate_api::HostTensorView;
use runmat_builtins::LDIVIDE_DESCRIPTOR;
use runmat_value::{
    CharArray, ComplexTensor, IntValue, IntegerStorage, LogicalArray, NumericStorage, SymbolicExpr,
    Tensor,
};

const EPS: f64 = 1e-12;
const GPU_EPS: f64 = 1e-6;

fn ldivide_builtin(lhs: Value, rhs: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    block_on(super::ldivide_builtin(lhs, rhs, rest))
}

mod contract;
mod host;
mod provider;
mod wgpu;
