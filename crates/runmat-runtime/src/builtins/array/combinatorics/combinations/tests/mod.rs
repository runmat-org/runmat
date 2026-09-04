mod containers;
mod integers;
mod planning;
mod provider;

use super::*;
use crate::builtins::table::{table_height, table_variables};
use crate::BuiltinResult;
use futures::executor::block_on;
use runmat_value::{ObjectInstance, Value};

fn call(first: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    block_on(combinations_builtin(first, rest))
}

fn table(value: Value) -> ObjectInstance {
    let Value::Object(table) = value else {
        panic!("expected table")
    };
    table
}
