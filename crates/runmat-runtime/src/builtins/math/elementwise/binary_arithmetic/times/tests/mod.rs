use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate_api::HostTensorView;
use runmat_value::{
    CharArray, ComplexTensor, IntValue, IntegerStorage, LogicalArray, SparseTensor, Tensor,
};

const EPS: f64 = 1e-12;

fn times_builtin(lhs: Value, rhs: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    block_on(super::times_builtin(
        None,
        lhs,
        rhs,
        crate::builtins::common::test_support::builtin_arguments(rest),
    ))
    .map(|owned| owned.value().clone())
}

mod host;
mod provider;
#[cfg(feature = "wgpu")]
mod wgpu;
