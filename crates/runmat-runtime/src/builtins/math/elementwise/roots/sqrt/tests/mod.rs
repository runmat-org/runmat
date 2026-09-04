mod host;
mod provider;

use super::*;
use futures::executor::block_on;

fn sqrt_builtin(value: Value) -> BuiltinResult<Value> {
    block_on(super::sqrt_builtin(value))
}
