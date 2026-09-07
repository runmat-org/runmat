use super::*;
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_builtins::POWER_DESCRIPTOR;
use runmat_value::{ComplexTensor, IntValue, IntegerStorage, SymbolicArray, SymbolicExpr, Tensor};

fn power_builtin(lhs: Value, rhs: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    block_on(super::power_builtin(lhs, rhs, rest))
}

mod contract;
mod numeric;
mod prototype;
mod provider;
mod symbolic;
mod wgpu;
