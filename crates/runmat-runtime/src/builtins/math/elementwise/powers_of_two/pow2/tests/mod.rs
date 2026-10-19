mod host;
mod integer;
mod provider;

#[cfg(feature = "wgpu")]
mod wgpu;

use futures::executor::block_on;
use runmat_value::Value;

use crate::BuiltinResult;

pub(super) fn call(first: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    block_on(super::pow2_builtin(first, rest))
}
