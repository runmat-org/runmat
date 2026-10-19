use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use futures::executor::block_on;
use runmat_builtins::{
    MPOWER_DESCRIPTOR, MPOWER_ERROR_INVALID_ARGUMENT, MPOWER_ERROR_INVALID_INPUT,
};
use runmat_value::{IntValue, IntegerStorage, Tensor};
fn unwrap_error(err: crate::RuntimeError) -> crate::RuntimeError {
    err
}

mod contract;
mod host;
mod integer;
mod provider;

fn mpower_builtin(base: Value, exponent: Value) -> BuiltinResult<Value> {
    block_on(super::mpower_builtin(base, exponent))
}
