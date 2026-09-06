use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate_api::HostTensorView;
use runmat_value::{
    CharArray, ComplexTensor, IntValue, IntegerStorage, LogicalArray, NumericDType, SparseTensor,
    Tensor,
};

const EPS: f64 = 1e-12;

fn minus_builtin(lhs: Value, rhs: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    block_on(super::minus_builtin(lhs, rhs, rest))
}

mod host;
mod provider;
#[cfg(feature = "wgpu")]
mod wgpu;
