use runmat_value::{ObjectInstance, StructValue, Value};

use crate::BuiltinResult;

use super::operation::ExponentialOperation;

pub(super) async fn evaluate(
    operation: ExponentialOperation,
    object: ObjectInstance,
) -> BuiltinResult<Value> {
    let variables = crate::builtins::table::table_variables(&object)
        .map_err(|error| super::errors::invalid(operation, error.message))?;
    let mut output = StructValue::new();
    for (name, value) in variables.fields {
        super::extensions::ensure(operation, &value)?;
        if matches!(value, Value::Object(_)) {
            return Err(super::errors::invalid(
                operation,
                format!(
                    "table variable {name} does not support {}",
                    operation.name()
                ),
            ));
        }
        output.insert(
            name,
            super::engine::execute_non_table(operation, value).await?,
        );
    }
    crate::builtins::table::table_replace_variables_like(&object, output)
        .map_err(|error| super::errors::internal(operation, error.message))
}
