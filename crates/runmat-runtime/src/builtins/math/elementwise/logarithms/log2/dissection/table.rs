use super::super::super::errors;
use super::super::OPERATION;
use crate::execution::{RootedValue, RootedValueLease, RootedValuesBuilder};
use crate::BuiltinResult;
use runmat_value::{ObjectInstance, Value};

pub(super) async fn evaluate(object: ObjectInstance) -> BuiltinResult<(RootedValue, RootedValue)> {
    let source = Value::Object(object.clone());
    let _source = RootedValueLease::for_values(std::iter::once(&source))
        .map_err(|cause| cause.into_runtime_error())?;
    let variables = crate::builtins::table::table_variables(&object)
        .map_err(|error| errors::invalid(OPERATION, error.message()))?;
    let mut names = Vec::with_capacity(variables.fields.len());
    let mut fractions = RootedValuesBuilder::with_capacity(variables.fields.len());
    let mut exponents = RootedValuesBuilder::with_capacity(variables.fields.len());
    for (name, value) in variables.into_fields_unbounded() {
        let (fraction, exponent) = Box::pin(super::execute(value)).await.map_err(|error| {
            errors::invalid(
                OPERATION,
                format!(
                    "table variable {name} does not support log2 dissection: {}",
                    error.message()
                ),
            )
        })?;
        names.push(name);
        fractions.push_owned(fraction);
        exponents.push_owned(exponent);
    }
    let fractions = replace(&object, names.clone(), fractions)?;
    let exponents = replace(&object, names, exponents)?;
    Ok((fractions, exponents))
}

fn replace(
    object: &ObjectInstance,
    names: Vec<String>,
    values: RootedValuesBuilder,
) -> BuiltinResult<RootedValue> {
    let values = values
        .finish()
        .map_err(|cause| cause.into_runtime_error())?;
    crate::builtins::table::table_replace_variables_like_owned(object, names, values)
        .map_err(|error| errors::internal(OPERATION, error.message()))
}
