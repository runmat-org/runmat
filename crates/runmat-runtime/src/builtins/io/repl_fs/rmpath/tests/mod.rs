mod behavior;
mod validation;

use runmat_value::Value;

pub(super) fn run(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    futures::executor::block_on(super::execute::run(args))
}
