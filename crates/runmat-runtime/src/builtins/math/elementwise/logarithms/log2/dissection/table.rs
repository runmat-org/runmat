use super::super::super::errors;
use super::super::OPERATION;
use crate::BuiltinResult;
use runmat_value::{ObjectInstance, StructValue, Value};

pub(super) async fn evaluate(object: ObjectInstance) -> BuiltinResult<(Value, Value)> {
    let variables = crate::builtins::table::table_variables(&object)
        .map_err(|error| errors::invalid(OPERATION, error.message()))?;
    let mut fractions = StructValue::new();
    let mut exponents = StructValue::new();
    for (name, value) in variables.fields {
        let (fraction, exponent) = Box::pin(super::execute(value)).await.map_err(|error| {
            errors::invalid(
                OPERATION,
                format!(
                    "table variable {name} does not support log2 dissection: {}",
                    error.message()
                ),
            )
        })?;
        fractions.insert(name.clone(), fraction);
        exponents.insert(name, exponent);
    }
    let fractions = replace(&object, fractions)?;
    let exponents = replace(&object, exponents)?;
    Ok((fractions, exponents))
}

fn replace(object: &ObjectInstance, values: StructValue) -> BuiltinResult<Value> {
    crate::builtins::table::table_replace_variables_like(object, values)
        .map_err(|error| errors::internal(OPERATION, error.message()))
}
