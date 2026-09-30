//! Borrowed columns for the public network Learnables and State tables.
use super::*;
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum GradientRole {
    Learnable(parameters::ParameterRole),
    State(state::StateRole),
}

/// Admit the public Learnables/State table shape for differentiation. Other
/// tables keep the general recursive table behavior in the autodiff caller.
/// A table claiming this exact schema must have valid metadata; its Layer and
/// Parameter columns are identities, not differentiation targets.
pub(in crate::builtins::deep_learning) fn gradient_values<'a>(
    table: &'a ObjectInstance,
    function: &'static str,
) -> BuiltinResult<Option<&'a CellArray>> {
    let columns = crate::builtins::table::table_variable_names_from_object(table)
        .map_err(|error| deep_learning_error(function, error.to_string()))?;
    if columns != ["Layer", "Parameter", "Value"] {
        return Ok(None);
    }
    let fields = &variables(table, "network parameter", function)?.fields;
    let column = |name| match fields.get(name) {
        Some(Value::StringArray(array)) if array.shape == [array.data.len(), 1] => Ok(array),
        _ => Err(deep_learning_error(
            function,
            format!("{function}: {name} must be a string column in a network parameter table"),
        )),
    };
    let layers = column("Layer")?;
    let roles = column("Parameter")?;
    let values = values(table, "network parameter", function)?;
    if layers.data.len() != roles.data.len() || layers.data.len() != values.data.len() {
        return Err(deep_learning_error(
            function,
            format!("{function}: network parameter table columns have different row counts"),
        ));
    }
    let mut family = None;
    let mut seen = HashSet::with_capacity(layers.data.len());
    for (layer, role) in layers.data.iter().zip(&roles.data) {
        let role = if let Some(role) = parameters::ParameterRole::parse(role) {
            GradientRole::Learnable(role)
        } else if let Some(role) = state::StateRole::parse(role) {
            GradientRole::State(role)
        } else {
            return Err(deep_learning_error(
                function,
                format!("{function}: unknown network parameter role '{role}'"),
            ));
        };
        let is_state = matches!(role, GradientRole::State(_));
        if family.is_some_and(|prior| prior != is_state) {
            return Err(deep_learning_error(
                function,
                format!("{function}: Learnables and State roles cannot share a table"),
            ));
        }
        family = Some(is_state);
        if !seen.insert((layer, role)) {
            return Err(deep_learning_error(
                function,
                format!("{function}: duplicate network parameter row identity"),
            ));
        }
    }
    Ok(Some(values))
}

fn variables<'a>(
    table: &'a ObjectInstance,
    property: &str,
    function: &'static str,
) -> BuiltinResult<&'a StructValue> {
    crate::builtins::table::table_variables_ref(table)
        .map_err(|error| deep_learning_error(function, error.to_string()))?
        .ok_or_else(|| {
            deep_learning_error(
                function,
                format!("{function}: {property} table is missing Variables"),
            )
        })
}

pub(super) fn values<'a>(
    table: &'a ObjectInstance,
    property: &str,
    function: &'static str,
) -> BuiltinResult<&'a CellArray> {
    match variables(table, property, function)?.fields.get("Value") {
        Some(Value::Cell(cell)) if cell.shape == [cell.data.len(), 1] => Ok(cell),
        _ => Err(deep_learning_error(
            function,
            format!("{function}: {property}.Value must be a column cell array"),
        )),
    }
}

pub(super) fn text(
    table: &ObjectInstance,
    property: &str,
    name: &'static str,
    function: &'static str,
) -> BuiltinResult<Vec<String>> {
    match variables(table, property, function)?.fields.get(name) {
        Some(Value::StringArray(array)) if array.shape == [array.data.len(), 1] => {
            Ok(array.data.clone())
        }
        // Kept only as an input extension for older value-network callers.
        Some(Value::Cell(cell)) if cell.shape == [cell.data.len(), 1] => cell
            .data
            .iter()
            .map(|value| scalar_text(value, function))
            .collect(),
        _ => Err(deep_learning_error(
            function,
            format!("{function}: {property}.{name} must be a string column"),
        )),
    }
}

pub(super) fn require_columns(
    table: &ObjectInstance,
    property: &str,
    function: &'static str,
) -> BuiltinResult<()> {
    if !crate::builtins::table::is_tabular_object(table) {
        return Err(deep_learning_error(
            function,
            format!("{function}: {property} must be a table"),
        ));
    }
    let columns = crate::builtins::table::table_variable_names_from_object(table)
        .map_err(|error| deep_learning_error(function, error.to_string()))?;
    if columns != ["Layer", "Parameter", "Value"] {
        return Err(deep_learning_error(
            function,
            format!(
                "{function}: {property} must have Layer, Parameter, Value columns in that order"
            ),
        ));
    }
    Ok(())
}
