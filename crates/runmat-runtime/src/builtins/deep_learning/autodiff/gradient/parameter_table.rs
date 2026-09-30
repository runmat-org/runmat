//! Borrowed public gradient-target schema, separate from network assignment.
use super::*;

/// Layer and Parameter are opaque string metadata, not execution roles.
/// Applying values to a network separately admits canonical typed row keys.
pub(super) fn values<'a>(
    table: &'a ObjectInstance,
    function: &'static str,
) -> BuiltinResult<Option<&'a CellArray>> {
    let columns = crate::builtins::table::table_variable_names_from_object(table)
        .map_err(|error| deep_learning_error(function, error.to_string()))?;
    if columns != ["Layer", "Parameter", "Value"] {
        return Ok(None);
    }
    let variables = crate::builtins::table::table_variables_ref(table)
        .map_err(|error| deep_learning_error(function, error.to_string()))?
        .ok_or_else(|| {
            deep_learning_error(
                function,
                format!("{function}: parameter table is missing Variables"),
            )
        })?;
    let text_column = |name| match variables.fields.get(name) {
        Some(Value::StringArray(array)) if array.shape == [array.data.len(), 1] => Ok(array),
        _ => Err(deep_learning_error(
            function,
            format!("{function}: {name} must be a string column in a parameter table"),
        )),
    };
    let layers = text_column("Layer")?;
    let parameters = text_column("Parameter")?;
    let Some(Value::Cell(values)) = variables.fields.get("Value") else {
        return Err(deep_learning_error(
            function,
            format!("{function}: parameter table Value must be a column cell array"),
        ));
    };
    if values.shape != [values.data.len(), 1] {
        return Err(deep_learning_error(
            function,
            format!("{function}: parameter table Value must be a column cell array"),
        ));
    }
    if layers.data.len() != parameters.data.len() || layers.data.len() != values.data.len() {
        return Err(deep_learning_error(
            function,
            format!("{function}: parameter table columns have different row counts"),
        ));
    }
    Ok(Some(values))
}
