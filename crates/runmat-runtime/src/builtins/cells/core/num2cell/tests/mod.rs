mod admission;
mod containers;
mod managed_strings;
mod numeric;

use super::*;
use futures::executor::block_on;
use runmat_value::Value;

fn run(value: Value, dims: Vec<Value>) -> Value {
    block_on(num2cell_builtin(
        None,
        value,
        crate::builtins::common::test_support::builtin_arguments(dims),
    ))
    .expect("num2cell succeeds")
}
