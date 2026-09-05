use super::operation::LogarithmOperation;
use crate::BuiltinResult;
use runmat_value::{ObjectInstance, StructValue, Value};

pub(super) async fn evaluate(
    operation: LogarithmOperation,
    object: ObjectInstance,
) -> BuiltinResult<Value> {
    let variables = crate::builtins::table::table_variables(&object)
        .map_err(|error| super::errors::invalid(operation, error.message()))?;
    let mut output = StructValue::new();
    for (name, value) in variables.fields {
        let transformed = super::engine::execute_non_table(operation, value)
            .await
            .map_err(|error| {
                super::errors::invalid(
                    operation,
                    format!(
                        "table variable {name} does not support {}: {}",
                        operation.name(),
                        error.message()
                    ),
                )
            })?;
        output.insert(name, transformed);
    }
    crate::builtins::table::table_replace_variables_like(&object, output)
        .map_err(|error| super::errors::internal(operation, error.message()))
}
