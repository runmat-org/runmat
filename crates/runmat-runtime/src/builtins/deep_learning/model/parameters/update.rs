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

fn install(network: &mut ObjectInstance, replacements: Vec<(ParameterKey, Option<Value>)>) {
    let Value::Cell(layers) = network
        .properties
        .get_mut("Layers")
        .expect("parameter view admitted Layers")
    else {
        unreachable!("parameter view admitted Layers cell")
    };
    for (key, value) in replacements {
        let Value::Object(layer) = &mut layers.data[key.layer_index] else {
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
            if value.shape.contains(&0) {
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
    install(&mut network, replacements);
    network
        .properties
        .insert("Initialized".into(), Value::Bool(initialized));
    Ok(Value::Object(network))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::deep_learning::layers::{
        feature_input_layer_builtin, fully_connected_layer_builtin,
    };
    use crate::builtins::deep_learning::model::dlnetwork_builtin;
    use futures::executor::block_on;

    fn network() -> ObjectInstance {
        let input = block_on(feature_input_layer_builtin(Value::Num(2.0), vec![])).unwrap();
        let dense = block_on(fully_connected_layer_builtin(
            Value::Num(2.0),
            vec![
                Value::String("Weights".into()),
                Value::Tensor(Tensor::from_f32(vec![1., 2., 3., 4.], vec![2, 2]).unwrap()),
                Value::String("Bias".into()),
                Value::Tensor(Tensor::from_f32(vec![0., 0.], vec![1, 2]).unwrap()),
            ],
        ))
        .unwrap();
        let Value::Object(network) = block_on(dlnetwork_builtin(vec![Value::Cell(
            CellArray::new(vec![input, dense], 2, 1).unwrap(),
        )]))
        .unwrap() else {
            panic!("network expected")
        };
        network
    }

    #[test]
    fn selected_native_updates_preserve_canonical_store_metadata_and_input_alias() {
        let mut input = network();
        input
            .properties
            .insert("UserMarker".into(), Value::String("keep".into()));
        let original = input.clone();
        let rows = view(&input, "test").unwrap();
        let weight_key = rows[0].key.clone();
        let new_weights = Tensor::from_f32(vec![5., 6., 7., 8.], vec![2, 2]).unwrap();
        let Value::Object(updated) = apply_updates(
            input,
            vec![ParameterUpdate {
                key: weight_key,
                value: new_weights,
            }],
            "test",
        )
        .unwrap() else {
            panic!("updated network expected")
        };
        assert_eq!(
            updated.properties.get("UserMarker"),
            Some(&Value::String("keep".into()))
        );
        assert_eq!(
            updated
                .properties
                .get(super::super::state::STORAGE_PROPERTY),
            original
                .properties
                .get(super::super::state::STORAGE_PROPERTY)
        );
        let updated_rows = view(&updated, "test").unwrap();
        let old_rows = view(&original, "test").unwrap();
        let updated_data = parameter_data(updated_rows[0].value.unwrap(), "test").unwrap();
        let old_data = parameter_data(old_rows[0].value.unwrap(), "test").unwrap();
        let Value::Tensor(updated_data) = updated_data else {
            panic!("updated tensor expected")
        };
        let Value::Tensor(old_data) = old_data else {
            panic!("original tensor expected")
        };
        assert_eq!(updated_data.as_f32_slice(), Some(&[5., 6., 7., 8.][..]));
        assert_eq!(old_data.as_f32_slice(), Some(&[1., 2., 3., 4.][..]));
        assert_eq!(updated_rows[1].value, old_rows[1].value);
    }

    #[test]
    fn selected_updates_reject_entire_malformed_batch_before_install() {
        let input = network();
        let rows = view(&input, "test").unwrap();
        let updates = vec![
            ParameterUpdate {
                key: rows[0].key.clone(),
                value: Tensor::from_f32(vec![5., 6., 7., 8.], vec![2, 2]).unwrap(),
            },
            ParameterUpdate {
                key: rows[1].key.clone(),
                value: Tensor::from_f32(vec![1., 2., 3.], vec![1, 3]).unwrap(),
            },
        ];
        let alias = input.clone();
        assert!(apply_updates(input, updates, "test").is_err());
        let old_rows = view(&alias, "test").unwrap();
        let Value::Tensor(weights) = parameter_data(old_rows[0].value.unwrap(), "test").unwrap()
        else {
            panic!("original weights expected")
        };
        assert_eq!(weights.as_f32_slice(), Some(&[1., 2., 3., 4.][..]));
    }

    #[test]
    fn selected_updates_reject_duplicate_foreign_and_precision_changing_keys() {
        let input = network();
        let key = view(&input, "test").unwrap()[0].key.clone();
        let value = Tensor::from_f32(vec![5.; 4], vec![2, 2]).unwrap();
        assert!(apply_updates(
            input.clone(),
            vec![
                ParameterUpdate {
                    key: key.clone(),
                    value: value.clone(),
                },
                ParameterUpdate {
                    key: key.clone(),
                    value: value.clone(),
                },
            ],
            "test",
        )
        .is_err());
        let mut foreign = key.clone();
        foreign.layer_name.push_str("-other");
        assert!(apply_updates(
            input.clone(),
            vec![ParameterUpdate {
                key: foreign,
                value: value.clone(),
            }],
            "test",
        )
        .is_err());
        assert!(apply_updates(
            input,
            vec![ParameterUpdate {
                key,
                value: Tensor::new(vec![5.; 4], vec![2, 2]).unwrap(),
            }],
            "test",
        )
        .is_err());
    }

    #[test]
    fn selected_updates_preserve_dlarray_format_without_reusing_wrapper_identity() {
        let mut input = network();
        let Value::Cell(layers) = input.properties.get_mut("Layers").unwrap() else {
            panic!("Layers cell expected")
        };
        let Value::Object(dense) = &mut layers.data[1] else {
            panic!("dense layer expected")
        };
        let original = dense.properties.remove("Weights").unwrap();
        let mut wrapper =
            ObjectInstance::new(crate::builtins::deep_learning::DLARRAY_CLASS.owned());
        wrapper.properties.insert("Data".into(), original);
        wrapper
            .properties
            .insert("Format".into(), Value::String(String::new()));
        wrapper
            .properties
            .insert("Labels".into(), Value::String(String::new()));
        dense
            .properties
            .insert("Weights".into(), Value::Object(wrapper));

        let key = view(&input, "test").unwrap()[0].key.clone();
        let Value::Object(updated) = apply_updates(
            input.clone(),
            vec![ParameterUpdate {
                key,
                value: Tensor::from_f32(vec![9.; 4], vec![2, 2]).unwrap(),
            }],
            "test",
        )
        .unwrap() else {
            panic!("updated network expected")
        };
        let rows = view(&updated, "test").unwrap();
        let Value::Object(replacement) = rows[0].value.unwrap() else {
            panic!("dlarray wrapper expected")
        };
        assert_eq!(
            replacement.properties.get("Format"),
            Some(&Value::String(String::new()))
        );
        assert_eq!(
            replacement.properties.get("Labels"),
            Some(&Value::String(String::new()))
        );
        assert_eq!(replacement.properties.len(), 3);
        let original = view(&input, "test").unwrap();
        let Value::Tensor(original) = parameter_data(original[0].value.unwrap(), "test").unwrap()
        else {
            panic!("original tensor expected")
        };
        assert_eq!(original.as_f32_slice(), Some(&[1., 2., 3., 4.][..]));
    }
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
    install(&mut network, replacements);
    network
        .properties
        .insert("Initialized".into(), Value::Bool(initialized));
    Ok(Value::Object(network))
}
