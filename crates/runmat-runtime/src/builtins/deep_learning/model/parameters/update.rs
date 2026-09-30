//! Whole-table validation and atomic replacement of canonical layer parameters.
use super::*;
use std::collections::HashMap;

/// All metadata and replacement values are admitted before the owned network is changed.
/// A failed replacement leaves its input network and GPU leases untouched.
pub(in crate::builtins) fn apply(
    mut network: ObjectInstance,
    replacement: Value,
    function: &'static str,
) -> BuiltinResult<Value> {
    let Value::Object(table) = replacement else {
        return Err(deep_learning_error(
            function,
            format!("{function}: Learnables must be a table"),
        ));
    };
    super::super::table_rows::require_columns(&table, "Learnables", function)?;
    let layer_names = super::super::table_rows::text(&table, "Learnables", "Layer", function)?;
    let parameter_names =
        super::super::table_rows::text(&table, "Learnables", "Parameter", function)?;
    let values = super::super::table_rows::values(&table, "Learnables", function)?;
    let parameters = view(&network, function)?;
    if layer_names.len() != parameters.len()
        || parameter_names.len() != parameters.len()
        || values.data.len() != parameters.len()
    {
        return Err(deep_learning_error(
            function,
            format!("{function}: Learnables rows do not match network parameters"),
        ));
    }
    let mut supplied = HashMap::with_capacity(parameters.len());
    for (row, (layer, role)) in layer_names.iter().zip(&parameter_names).enumerate() {
        let role = ParameterRole::parse(role).ok_or_else(|| {
            deep_learning_error(
                function,
                format!("{function}: unknown Learnables parameter role '{role}'"),
            )
        })?;
        if supplied.insert((layer.as_str(), role), row).is_some() {
            return Err(deep_learning_error(
                function,
                format!("{function}: duplicate Learnables parameter identity"),
            ));
        }
    }
    let mut initialized = true;
    let mut rows = Vec::with_capacity(parameters.len());
    for parameter in &parameters {
        let row = supplied
            .get(&(parameter.key.layer_name.as_str(), parameter.key.role))
            .copied()
            .ok_or_else(|| {
                deep_learning_error(
                    function,
                    format!("{function}: Learnables parameter identity differs from Layers"),
                )
            })?;
        validate_parameter(
            &values.data[row],
            &parameter.expected_shape,
            parameter.key.role,
            function,
        )?;
        initialized &= !empty_parameter(parameter_data(&values.data[row], function)?);
        rows.push(row);
    }
    let keys = parameters
        .into_iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    let Value::Cell(layers) = network
        .properties
        .get_mut("Layers")
        .expect("validated Layers")
    else {
        unreachable!("validated Layers")
    };
    for (key, row) in keys.iter().zip(rows) {
        let value = values.data[row].clone();
        let Value::Object(layer) = &mut layers.data[key.layer_index] else {
            unreachable!("validated layer")
        };
        if empty_parameter(parameter_data(&value, function)?) {
            layer.properties.remove(key.role.name());
        } else {
            layer.properties.insert(key.role.name().into(), value);
        }
    }
    initialized &= super::super::state::all_filled(&network, function)?;
    network
        .properties
        .insert("Initialized".into(), Value::Bool(initialized));
    Ok(Value::Object(network))
}
