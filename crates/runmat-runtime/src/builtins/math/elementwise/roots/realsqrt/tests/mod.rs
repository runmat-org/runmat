mod host;
mod provider;

use super::*;
use futures::executor::block_on;

fn call(value: Value) -> BuiltinResult<Value> {
    block_on(super::realsqrt_builtin(value))
}
