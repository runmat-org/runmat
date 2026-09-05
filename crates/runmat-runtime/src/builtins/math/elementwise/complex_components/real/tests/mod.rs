use crate::builtins::common::{gpu_helpers, test_support};
use futures::executor::block_on;
use runmat_builtins::{REAL_DESCRIPTOR, REAL_ERROR_INVALID_INPUT};
use runmat_value::{
    CharArray, ComplexTensor, IntValue, LogicalArray, NumericStorage, Tensor, Value,
};

use super::*;

fn real_builtin(value: Value) -> BuiltinResult<Value> {
    block_on(super::real_builtin(value))
}

mod host;
mod provider;
#[cfg(feature = "wgpu")]
mod wgpu;
