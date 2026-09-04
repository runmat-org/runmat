mod arrays;
mod bins;
mod compatibility;
mod provider;
mod tables;

use futures::executor::block_on;
use runmat_value::Value;

use super::groupcounts_builtin;

fn call(first: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    block_on(groupcounts_builtin(first, rest))
}

fn output_list(value: Value) -> Vec<Value> {
    match value {
        Value::OutputList(values) => values,
        other => panic!("expected output list, got {other:?}"),
    }
}
