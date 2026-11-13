use super::structfun_builtin;
use futures::executor::block_on;
use runmat_value::{StructValue, Value};

fn call(
    function: Value,
    structure: StructValue,
    rest: Vec<Value>,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    let _compatibility = crate::compatibility::push_runmat_extensions_enabled(true);
    block_on(structfun_builtin(
        None,
        function,
        Value::Struct(structure),
        crate::builtins::common::test_support::builtin_arguments(rest),
    ))
}

fn rooted_sequence(
    sequence: runmat_value::ValueSequence,
) -> crate::BuiltinResult<crate::execution::RootedValueSequence> {
    crate::execution::RootedValueSequence::new(sequence).map_err(|error| error.into_runtime_error())
}

fn numbers() -> StructValue {
    let mut structure = runmat_value::UnboundedStructValueBuilder::new();
    structure.insert("a", Value::Num(1.0));
    structure.insert("b", Value::Num(2.0));
    structure.finish()
}

mod callback;
mod invocation;
mod options;
mod output;
mod provider;
