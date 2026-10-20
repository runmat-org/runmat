mod arrays;
mod ordering;
mod outputs;
mod preservation;
mod validation;

use futures::executor::block_on;
use runmat_value::{StructValue, Value};

fn call(target: Value, order: Vec<Value>) -> crate::BuiltinResult<Value> {
    block_on(super::orderfields_builtin(
        None,
        target,
        crate::builtins::common::test_support::builtin_arguments(order),
    ))
    .map(|rooted| rooted.with_owned(|value| value))
}

fn structure(fields: &[(&str, Value)]) -> StructValue {
    let mut structure = runmat_value::UnboundedStructValueBuilder::new();
    for (name, value) in fields {
        structure.insert(*name, value.clone());
    }
    structure.finish()
}

fn field_order(structure: &StructValue) -> Vec<&str> {
    structure.field_names().collect()
}

fn error_identifier(error: crate::RuntimeError) -> String {
    error.identifier().unwrap_or_default().to_string()
}
