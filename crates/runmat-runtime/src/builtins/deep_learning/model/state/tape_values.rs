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

    let Some(Value::Cell(layers)) = network.properties.get("Layers") else {
        unreachable!("State view admitted Layers")
    };
    let Some(Value::Object(stored)) = network.properties.get(STORAGE_PROPERTY) else {
        unreachable!("State view admitted canonical State")
    };
    let mut variables = crate::builtins::table::table_variables(stored)
        .map_err(|error| deep_learning_error(function, error.to_string()))?;
    let Some(Value::Cell(column)) = variables.fields.get_mut("Value") else {
        unreachable!("State view admitted the Value column")
    };
    for (slot, (key, value)) in column.data.iter_mut().zip(replacements) {
        schema::validate_value(&value, &key, function)?;
        *slot = value;
    }
    let replacement = crate::builtins::table::table_replace_variables_like(stored, variables)
        .map_err(|error| deep_learning_error(function, error.to_string()))?;
    table::validate_canonical(&layers.data, &replacement, function)?;
    Ok(replacement)
}

/// Called only after the staged table passed full canonical validation.
pub(in crate::builtins::deep_learning) fn install_annotated(
    network: &mut ObjectInstance,
    staged: Value,
) {
    network.properties.insert(STORAGE_PROPERTY.into(), staged);
}
