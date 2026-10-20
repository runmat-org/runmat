mod behavior;
mod compatibility;
mod validation;

use runmat_value::Value;

pub(super) fn run(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    futures::executor::block_on(super::execute::run(
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
}
