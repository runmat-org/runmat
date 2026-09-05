use crate::builtins::common::{gpu_helpers, test_support};
use futures::executor::block_on;
use runmat_builtins::{IMAG_DESCRIPTOR, IMAG_ERROR_INVALID_INPUT};
use runmat_value::{
    CharArray, ComplexTensor, IntValue, LogicalArray, NumericStorage, StringArray, Tensor, Value,
};

use super::*;

fn imag_builtin(value: Value) -> BuiltinResult<Value> {
    block_on(super::imag_builtin(value))
}

mod host;
mod provider;
#[cfg(feature = "wgpu")]
mod wgpu;
