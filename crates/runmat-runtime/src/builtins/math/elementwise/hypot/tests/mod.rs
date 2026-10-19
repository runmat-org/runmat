use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use futures::executor::block_on;
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::AccelProvider;
use runmat_builtins::{HYPOT_ERROR_INVALID_INPUT, HYPOT_ERROR_TOO_MANY_OUTPUTS};
use runmat_value::{
    CharArray, ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, LogicalArray,
    NumericStorage, Tensor, Value,
};

fn hypot_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    block_on(super::hypot_builtin(lhs, rhs))
}

mod contract;
mod host_behavior;
mod provider;
mod typed_storage;
#[cfg(feature = "wgpu")]
mod wgpu;
