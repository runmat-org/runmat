use runmat_builtins::LogicalUnaryOperator;
use runmat_value::Value;

use crate::execution::{RootedValue, RootedValueLease, RootedValuesBuilder};
use crate::BuiltinResult;

use super::{contract, operand, provider};

pub(crate) async fn execute(
    value: Value,
    operation: LogicalUnaryOperator,
) -> BuiltinResult<RootedValue> {
    let _input =
        RootedValueLease::for_traced(&value).map_err(|error| error.into_runtime_error())?;
    let (name, invalid) = contract::unary(operation);
    if let Value::Object(object) = value {
        if crate::builtins::table::is_tabular_object(&object) {
            let variables = crate::builtins::table::table_variables(&object)
                .map_err(|error| operand::error_with_detail(name, invalid, error))?;
            let mut names = Vec::with_capacity(variables.fields.len());
            let mut output = RootedValuesBuilder::with_capacity(variables.fields.len());
            for (variable_name, value) in variables.into_fields_unbounded() {
                let result = scalar_or_array(value, operation, name, invalid)
                    .await
                    .map_err(|error| {
                        operand::error_with_detail(
                            name,
                            invalid,
                            format!("table variable {variable_name}: {}", error.message()),
                        )
                    })?;
                output.push_owned(result);
                names.push(variable_name);
            }
            let output = output
                .finish()
                .map_err(|error| error.into_runtime_error())?;
            return crate::builtins::table::table_replace_variables_like_owned(
                &object, names, output,
            )
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
) -> BuiltinResult<RootedValue> {
    let output_source = provider::select_unary_source(&value, name)?;
    if let Value::GpuTensor(handle) = &value {
        if let Some(output) = provider::unary_hook(handle, name)? {
            return RootedValue::new(output).map_err(|error| error.into_runtime_error());
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
    let output = provider::restore_explicit(value, output_source.as_ref(), name)?;
    RootedValue::new(output).map_err(|error| error.into_runtime_error())
}
