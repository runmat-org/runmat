use runmat_builtins::LogicalUnaryOperator;
use runmat_value::{StructValue, Value};

use crate::BuiltinResult;

use super::{contract, operand, provider};

pub(crate) async fn execute(value: Value, operation: LogicalUnaryOperator) -> BuiltinResult<Value> {
    let (name, invalid) = contract::unary(operation);
    if let Value::Object(object) = value {
        if crate::builtins::table::is_tabular_object(&object) {
            let variables = crate::builtins::table::table_variables(&object)
                .map_err(|error| operand::error_with_detail(name, invalid, error))?;
            let mut output = StructValue::new();
            for (variable_name, value) in variables.fields {
                let result = scalar_or_array(value, operation, name, invalid)
                    .await
                    .map_err(|error| {
                        operand::error_with_detail(
                            name,
                            invalid,
                            format!("table variable {variable_name}: {}", error.message()),
                        )
                    })?;
                output.insert(variable_name, result);
            }
            return crate::builtins::table::table_replace_variables_like(&object, output)
                .map_err(|error| operand::error_with_detail(name, invalid, error));
        }
        return Err(operand::error_with_detail(
            name,
            invalid,
            "unsupported object input",
        ));
    }
    scalar_or_array(value, operation, name, invalid).await
}

async fn scalar_or_array(
    value: Value,
    operation: LogicalUnaryOperator,
    name: &'static str,
    invalid: &'static runmat_builtins::BuiltinErrorDescriptor,
) -> BuiltinResult<Value> {
    let output_source = provider::select_unary_source(&value, name)?;
    if let Value::GpuTensor(handle) = &value {
        if let Some(output) = provider::unary_hook(handle, name)? {
            return Ok(output);
        }
    }
    let buffer = operand::from_value(name, value, invalid).await?;
    let data = match operation {
        LogicalUnaryOperator::Not => buffer
            .data
            .into_iter()
            .map(|bit| u8::from(bit == 0))
            .collect(),
    };
    let value = operand::into_value(name, data, buffer.shape, invalid)?;
    provider::restore_explicit(value, output_source.as_ref(), name)
}
