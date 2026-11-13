//! One canonical State table, with typed row admission before replacement.
use super::*;
use std::collections::HashMap;

pub(super) fn table(
    keys: &[StateKey],
    values: Vec<Value>,
    function: &'static str,
) -> BuiltinResult<Value> {
    let count = keys.len();
    let layer = string_array(
        keys.iter().map(|key| key.layer_name.clone()).collect(),
        vec![count, 1],
        function,
    )?;
    let parameter = string_array(
        keys.iter().map(|key| key.role.name().into()).collect(),
        vec![count, 1],
        function,
    )?;
    let values = CellArray::new(values, count, 1)
        .map(Value::Cell)
        .map_err(|error| deep_learning_error(function, error))?;
    crate::builtins::table::table_from_columns(
        vec!["Layer".into(), "Parameter".into(), "Value".into()],
        vec![layer, parameter, values],
    )
    .map_err(|error| deep_learning_error(function, error.to_string()))
}

#[cfg(test)]
pub(in crate::builtins::deep_learning::model) fn initial(
    layers: &[Value],
    function: &'static str,
) -> BuiltinResult<Value> {
    initial_with_defaults(layers, false, function)
}

pub(in crate::builtins::deep_learning::model) fn initial_with_defaults(
    layers: &[Value],
    initialize: bool,
    function: &'static str,
) -> BuiltinResult<Value> {
    let keys = schema::keys(layers, function)?;
    let mut precision = vec![None; layers.len()];
    if initialize {
        for row in super::super::parameters::view_layers(layers, function)? {
            if let Some(value) = row.value {
                if let Some(dtype @ (NumericDType::F32 | NumericDType::F64)) =
                    super::super::parameters::floating_precision(value)
                {
                    precision[row.key.layer_index] = Some(dtype);
                }
            }
        }
    }
    let mut values = Vec::with_capacity(keys.len());
    for key in &keys {
        let Value::Object(layer) = &layers[key.layer_index] else {
            unreachable!("validated layer")
        };
        let value = layer.properties().get(key.role.name());
        if let Some(value) = value {
            schema::validate_value(value, key, function)?;
        }
        let empty = value.is_none_or(|value| {
            super::super::parameters::numeric_payload_shape(value, key.role.name(), function)
                .is_ok_and(|shape| shape.contains(&0))
        });
        if initialize && empty {
            let single = precision[key.layer_index] == Some(NumericDType::F32);
            let shape = vec![key.channels, 1];
            let value = if single {
                let mut values = Vec::new();
                values.try_reserve_exact(key.channels).map_err(|_| {
                    deep_learning_error(
                        function,
                        format!("{function}: cannot allocate initial recurrent State"),
                    )
                })?;
                values.resize(key.channels, 0.0_f32);
                Tensor::from_f32(values, shape)
            } else {
                let mut values = Vec::new();
                values.try_reserve_exact(key.channels).map_err(|_| {
                    deep_learning_error(
                        function,
                        format!("{function}: cannot allocate initial recurrent State"),
                    )
                })?;
                values.resize(key.channels, 0.0_f64);
                Tensor::new(values, shape)
            }
            .map(Value::Tensor)
            .map_err(|error| deep_learning_error(function, error))?;
            values.push(value);
        } else {
            values.push(super::super::parameters::public_value(value)?);
        }
    }
    let state = table(&keys, values, function)?;
    validate_canonical(layers, &state, function)?;
    Ok(state)
}

pub(super) fn validate_canonical(
    layers: &[Value],
    stored: &Value,
    function: &'static str,
) -> BuiltinResult<()> {
    let Value::Object(table) = stored else {
        return Err(deep_learning_error(
            function,
            format!("{function}: canonical State must be a table"),
        ));
    };
    super::super::table_rows::require_columns(table, "State", function)?;
    let names = super::super::table_rows::text(table, "State", "Layer", function)?;
    let roles = super::super::table_rows::text(table, "State", "Parameter", function)?;
    let values = super::super::table_rows::values(table, "State", function)?;
    let keys = schema::keys(layers, function)?;
    if names.len() != keys.len() || roles.len() != keys.len() || values.data().len() != keys.len() {
        return Err(deep_learning_error(
            function,
            format!("{function}: canonical State row count differs from Layers"),
        ));
    }
    let mut batch = None;
    for (row, key) in keys.iter().enumerate() {
        if names[row] != key.layer_name || roles[row] != key.role.name() {
            return Err(deep_learning_error(
                function,
                format!("{function}: canonical State identity differs from Layers"),
            ));
        }
        schema::validate_value(&values.data()[row], key, function)?;
        if let Some(current) = schema::batch_size(&values.data()[row], function)? {
            if batch.is_some_and(|prior| prior != current) {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: State rows have inconsistent batch dimensions"),
                ));
            }
            batch = Some(current);
        }
    }
    Ok(())
}

pub(super) fn all_filled(
    layers: &[Value],
    stored: &Value,
    function: &'static str,
) -> BuiltinResult<bool> {
    validate_canonical(layers, stored, function)?;
    let Value::Object(table) = stored else {
        unreachable!("validated State table")
    };
    let values = super::super::table_rows::values(table, "State", function)?;
    for value in values.data() {
        if super::super::parameters::numeric_payload_shape(value, "State", function)?.contains(&0) {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(super) fn validated_replacement(
    layers: &[Value],
    replacement: Value,
    function: &'static str,
) -> BuiltinResult<Value> {
    let Value::Object(input) = replacement else {
        return Err(deep_learning_error(
            function,
            format!("{function}: State must be a table"),
        ));
    };
    super::super::table_rows::require_columns(&input, "State", function)?;
    let names = super::super::table_rows::text(&input, "State", "Layer", function)?;
    let roles = super::super::table_rows::text(&input, "State", "Parameter", function)?;
    let values = super::super::table_rows::values(&input, "State", function)?;
    let keys = schema::keys(layers, function)?;
    if names.len() != keys.len() || roles.len() != keys.len() || values.data().len() != keys.len() {
        return Err(deep_learning_error(
            function,
            format!("{function}: State rows do not match network state topology"),
        ));
    }
    let mut supplied = HashMap::with_capacity(keys.len());
    for (row, (name, role)) in names.iter().zip(&roles).enumerate() {
        let role = StateRole::parse(role).ok_or_else(|| {
            deep_learning_error(
                function,
                format!("{function}: unknown State parameter role '{role}'"),
            )
        })?;
        if supplied.insert((name.as_str(), role), row).is_some() {
            return Err(deep_learning_error(
                function,
                format!("{function}: duplicate State row identity"),
            ));
        }
    }
    let mut ordered = Vec::with_capacity(keys.len());
    let mut batch = None;
    for key in &keys {
        let row = supplied
            .get(&(key.layer_name.as_str(), key.role))
            .copied()
            .ok_or_else(|| {
                deep_learning_error(
                    function,
                    format!("{function}: State identity differs from Layers"),
                )
            })?;
        let value = &values.data()[row];
        schema::validate_value(value, key, function)?;
        if let Some(current) = schema::batch_size(value, function)? {
            if batch.is_some_and(|prior| prior != current) {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: State rows have inconsistent batch dimensions"),
                ));
            }
            batch = Some(current);
        }
        ordered.push(super::super::parameters::public_value(Some(value))?);
    }
    table(&keys, ordered, function)
}
