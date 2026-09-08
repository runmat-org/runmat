mod collections;
mod structures;
mod validation;

use runmat_value::Value;

pub(super) fn run(target: Value, names: Value) -> crate::BuiltinResult<Value> {
    futures::executor::block_on(super::isfield_builtin(target, names))
}
