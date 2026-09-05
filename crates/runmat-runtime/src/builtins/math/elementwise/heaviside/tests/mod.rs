mod provider;
mod semantics;
mod specification;

use futures::executor::block_on;
use runmat_value::Value;

use crate::BuiltinResult;

fn execute(value: Value) -> BuiltinResult<Value> {
    block_on(super::heaviside_builtin(value))
}
