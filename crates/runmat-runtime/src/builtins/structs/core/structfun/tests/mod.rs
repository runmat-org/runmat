use super::structfun_builtin;
use futures::executor::block_on;
use runmat_value::{StructValue, Value};

fn call(function: Value, structure: StructValue, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let _compatibility = crate::compatibility::push_runmat_extensions_enabled(true);
    block_on(structfun_builtin(function, Value::Struct(structure), rest))
}

fn numbers() -> StructValue {
    let mut structure = StructValue::new();
    structure.insert("a", Value::Num(1.0));
    structure.insert("b", Value::Num(2.0));
    structure
}

mod callback;
mod invocation;
mod options;
mod output;
mod provider;
