//! Whole-table validation and atomic replacement of canonical layer parameters.
use super::*;
use runmat_value::NumericDType;
use std::collections::HashMap;

/// A selected native-precision replacement for one canonical layer role.
/// The key is obtained from `view`; it is not a parallel parameter store.
pub(in crate::builtins::deep_learning) struct ParameterUpdate {
    pub key: ParameterKey,
    pub value: Tensor,
}

fn install(
    network: &mut ObjectInstance,
    replacements: Vec<(ParameterKey, Option<Value>)>,
    function: &'static str,
) -> BuiltinResult<()> {
    let Value::Cell(layers) = network
        .properties
        .get_mut("Layers")
        .expect("parameter view admitted Layers")
    else {
        unreachable!("parameter view admitted Layers cell")
    };
    let layer_values =
        crate::builtins::deep_learning::layer_helpers::mutable_cell_values(layers, function)?;
    for (key, value) in replacements {
        let Value::Object(layer) = &mut layer_values[key.layer_index] else {
            unreachable!("parameter view admitted layer object")
        };
        match value {
            Some(value) => {
                layer.properties.insert(key.role.name().into(), value);
            }
            None => {
                layer.properties.remove(key.role.name());
            }
        }
    }
    Ok(())
}

/// Atomic selected-role publication for the native trainer. Full topology,
/// physical class, and canonical State admission happens before any edit.
pub(in crate::builtins::deep_learning) fn apply_updates(
    mut network: ObjectInstance,
    updates: Vec<ParameterUpdate>,
    function: &'static str,
) -> BuiltinResult<Value> {
    let rows = view(&network, function)?;
    let state_filled = super::super::state::all_filled(&network, function)?;
    let mut selected = HashMap::with_capacity(updates.len());
    for update in updates {
        if selected.insert(update.key, update.value).is_some() {
            return Err(deep_learning_error(
                function,
                format!("{function}: duplicate parameter update"),
            ));
        }
    }
    let mut replacements = Vec::with_capacity(selected.len());
    let mut initialized = state_filled;
    for row in rows {
        if let Some(value) = selected.remove(&row.key) {
            if !matches!(value.numeric_dtype(), NumericDType::F32 | NumericDType::F64) {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: native training updates must be single or double"),
                ));
            }
            validate_parameter(
                &Value::Tensor(value.clone()),
                &row.expected_shape,
                row.key.role,
                function,
            )?;
            if value.shape().contains(&0) {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: native training update must not be empty"),
                ));
            }
            let original = row.value.ok_or_else(|| {
                deep_learning_error(function, format!("{function}: parameter is uninitialized"))
            })?;
            if matches!(parameter_data(original, function)?, Value::GpuTensor(handle)
                if runmat_accelerate_api::handle_is_explicit(handle))
            {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: explicit gpuArray parameter cannot be replaced by a host training update"),
                ));
            }
            if floating_precision(original) != Some(value.numeric_dtype()) {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: native training update changes parameter precision"),
                ));
            }
            let replacement = match original {
                Value::Object(original_wrapper)
                    if original_wrapper.class_name.is(super::DLARRAY_CLASS) =>
                {
                    let mut wrapper = ObjectInstance::new(super::DLARRAY_CLASS.owned());
                    wrapper
                        .properties
                        .insert("Data".into(), Value::Tensor(value));
                    for property in ["Format", "Labels"] {
                        if let Some(metadata) = original_wrapper.properties.get(property) {
                            wrapper.properties.insert(property.into(), metadata.clone());
                        }
                    }
                    Value::Object(wrapper)
                }
                _ => Value::Tensor(value),
            };
            validate_parameter(&replacement, &row.expected_shape, row.key.role, function)?;
            replacements.push((row.key, Some(replacement)));
        } else {
            initialized &= row
                .value
                .map(|value| parameter_data(value, function).map(|data| !empty_parameter(data)))
                .transpose()?
                .unwrap_or(false);
        }
    }
    if !selected.is_empty() {
        return Err(deep_learning_error(
            function,
            format!("{function}: parameter update key differs from network Layers"),
        ));
    }
    install(&mut network, replacements, function)?;
    network
        .properties
        .insert("Initialized".into(), Value::Bool(initialized));
    Ok(Value::Object(network))
}

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
    initialized &= super::super::state::all_filled(&network, function)?;
    let replacements = parameters
        .into_iter()
        .zip(rows)
        .map(|(parameter, row)| {
            let value = &values.data[row];
            let replacement =
                (!empty_parameter(parameter_data(value, function)?)).then(|| value.clone());
            Ok((parameter.key, replacement))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    install(&mut network, replacements, function)?;
    network
        .properties
        .insert("Initialized".into(), Value::Bool(initialized));
    Ok(Value::Object(network))
}

#[cfg(test)]
mod tests;
