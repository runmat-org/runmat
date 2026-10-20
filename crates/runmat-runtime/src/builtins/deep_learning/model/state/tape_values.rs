//! State-owner staging for tape annotations. The serialized State table stays
//! canonical and its metadata is preserved exactly.

use super::*;

pub(in crate::builtins::deep_learning) fn stage_annotated(
    network: &ObjectInstance,
    replacements: Vec<(StateKey, Value)>,
    function: &'static str,
) -> BuiltinResult<Value> {
    let expected = view(network, function)?;
    if expected.len() != replacements.len()
        || expected
            .iter()
            .zip(&replacements)
            .any(|(row, (key, _))| row.key != *key)
    {
        return Err(deep_learning_error(
            function,
            format!("{function}: tape State rows differ from canonical State"),
        ));
    }

    let Some(stored) = network.properties().get(STORAGE_PROPERTY) else {
        unreachable!("State view admitted canonical State")
    };
    let mut replacements = replacements.into_iter();
    map_values(network, stored, function, |key, _| {
        let (supplied_key, value) = replacements
            .next()
            .expect("State view admitted every canonical row");
        debug_assert_eq!(supplied_key, *key);
        Ok(value)
    })
}

/// Map a returned canonical State table to a new value without modifying the
/// input network or its serialized State. The mapper sees typed row identity,
/// never the public Layer/Parameter text columns. It is invoked only after all
/// source rows have passed canonical table admission.
pub(in crate::builtins::deep_learning) fn map_result_values(
    network: &ObjectInstance,
    state: Value,
    function: &'static str,
    mapper: impl FnMut(&StateKey, Value) -> BuiltinResult<Value>,
) -> BuiltinResult<Value> {
    map_values(network, &state, function, mapper)
}

fn map_values(
    network: &ObjectInstance,
    source: &Value,
    function: &'static str,
    mut mapper: impl FnMut(&StateKey, Value) -> BuiltinResult<Value>,
) -> BuiltinResult<Value> {
    let Some(Value::Cell(layers)) = network.properties().get("Layers") else {
        return Err(deep_learning_error(
            function,
            format!("{function}: network is missing Layers"),
        ));
    };
    table::validate_canonical(layers.data(), source, function)?;
    let keys = schema::keys(layers.data(), function)?;
    let Value::Object(stored) = source else {
        unreachable!("canonical State validation admitted a table")
    };
    let variables = crate::builtins::table::table_variables(stored)
        .map_err(|error| deep_learning_error(function, error.to_string()))?;
    let Some(Value::Cell(column)) = variables.fields.get("Value") else {
        unreachable!("canonical State validation admitted the Value column")
    };
    let context = crate::context::legacy::active();
    let prepared_column = crate::execution::structure::replacement::cell(column, context.as_ref())?;
    let _column_roots = prepared_column.1;
    let mut column = prepared_column.0;
    let slots = column
        .try_data_mut_unique()
        .expect("new admitted State slots are unique");
    for (key, slot) in keys.iter().zip(slots.iter_mut()) {
        let value = mapper(key, slot.clone())?;
        schema::validate_value(&value, key, function)?;
        *slot = value;
    }
    let prepared_variables =
        crate::execution::structure::replacement::record(&variables, context.as_ref())?;
    let _variables_roots = prepared_variables.1;
    let mut variables = prepared_variables.0;
    variables
        .try_replace_unique("Value", Value::Cell(column))
        .expect("new admitted State map contains Value");
    let replacement = crate::builtins::table::table_replace_variables_like(stored, variables)
        .map_err(|error| deep_learning_error(function, error.to_string()))?;
    table::validate_canonical(layers.data(), &replacement, function)?;
    Ok(replacement)
}

/// Called only after the staged table passed full canonical validation.
pub(in crate::builtins::deep_learning) fn install_annotated(
    network: &mut ObjectInstance,
    staged: Value,
) -> BuiltinResult<()> {
    network.replace_dl_property(STORAGE_PROPERTY, staged)?;
    Ok(())
}

#[cfg(test)]
mod tests;
