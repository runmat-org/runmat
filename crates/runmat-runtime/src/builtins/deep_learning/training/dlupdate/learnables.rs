//! Learnables tables and network reconstruction.
use super::*;

pub(super) async fn dlupdate_learnables_struct(
    function: &Value,
    reference: &StructValue,
    values: &[Value],
    requested_outputs: usize,
    depth: usize,
) -> BuiltinResult<Vec<Value>> {
    let structs = values
        .iter()
        .map(|value| match value {
            Value::Struct(st) if is_learnables_struct(st) => Ok(st),
            other => Err(deep_learning_error(
                "dlupdate",
                format!("dlupdate: expected matching learnables struct tree, got {other:?}"),
            )),
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let tables = structs.iter().cloned().cloned().collect::<Vec<_>>();
    require_matching_learnable_metadata(&tables)?;
    let value_columns = structs
        .iter()
        .map(|st| {
            st.fields.get("Value").cloned().ok_or_else(|| {
                deep_learning_error("dlupdate", "dlupdate: learnables struct is missing Value")
            })
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let updated_values = dlupdate_node(function, &value_columns, requested_outputs, depth).await?;
    updated_values
        .into_iter()
        .map(|value_column| {
            let mut out = reference.clone();
            out.insert("Value", value_column);
            Ok(Value::Struct(out))
        })
        .collect()
}

pub(super) async fn dlupdate_table(
    function: &Value,
    reference: &ObjectInstance,
    values: &[Value],
    requested_outputs: usize,
    depth: usize,
) -> BuiltinResult<Vec<Value>> {
    let variable_names = crate::builtins::table::table_variable_names_from_object(reference)
        .map_err(|err| deep_learning_error("dlupdate", err.to_string()))?;
    require_learnables_table(&variable_names)?;
    let reference_height = crate::builtins::table::table_height(reference)
        .map_err(|err| deep_learning_error("dlupdate", err.to_string()))?;
    let tables = values
        .iter()
        .map(|value| match value {
            Value::Object(object) if crate::builtins::table::is_tabular_object(object) => {
                if object.class_name != reference.class_name {
                    return Err(deep_learning_error(
                        "dlupdate",
                        "dlupdate: table parameter trees must have the same tabular class",
                    ));
                }
                let names = crate::builtins::table::table_variable_names_from_object(object)
                    .map_err(|err| deep_learning_error("dlupdate", err.to_string()))?;
                if names != variable_names {
                    return Err(deep_learning_error(
                        "dlupdate",
                        "dlupdate: table parameter trees must have the same variable names",
                    ));
                }
                let height = crate::builtins::table::table_height(object)
                    .map_err(|err| deep_learning_error("dlupdate", err.to_string()))?;
                if height != reference_height {
                    return Err(deep_learning_error(
                        "dlupdate",
                        "dlupdate: table parameter trees must have the same height",
                    ));
                }
                Ok(object)
            }
            other => Err(deep_learning_error(
                "dlupdate",
                format!("dlupdate: expected matching table tree, got {other:?}"),
            )),
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let table_variables = tables
        .iter()
        .map(|object| {
            crate::builtins::table::table_variables(object)
                .map_err(|err| deep_learning_error("dlupdate", err.to_string()))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;

    require_matching_learnable_metadata(&table_variables)?;
    let value_columns = table_variables
        .iter()
        .map(|vars| {
            vars.fields.get("Value").cloned().ok_or_else(|| {
                deep_learning_error("dlupdate", "dlupdate: learnables table is missing Value")
            })
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    if !value_columns
        .iter()
        .all(|value| matches!(value, Value::Cell(_)))
    {
        return Err(deep_learning_error(
            "dlupdate",
            "dlupdate: learnables table Value variable must be a cell array",
        ));
    }
    let updated_values = dlupdate_node(function, &value_columns, requested_outputs, depth).await?;

    updated_values
        .into_iter()
        .map(|value_column| {
            let mut variables = table_variables[0].clone();
            variables.insert("Value", value_column);
            crate::builtins::table::table_replace_variables_like(reference, variables)
                .map_err(|err| deep_learning_error("dlupdate", err.to_string()))
        })
        .collect()
}

pub(super) async fn dlupdate_network(
    function: &Value,
    reference: &ObjectInstance,
    values: &[Value],
    requested_outputs: usize,
    depth: usize,
) -> BuiltinResult<Vec<Value>> {
    let networks = values
        .iter()
        .map(|value| match value {
            Value::Object(object) if model::is_deep_learning_network_object(object) => {
                if object.class_name != reference.class_name {
                    return Err(deep_learning_error(
                        "dlupdate",
                        "dlupdate: dlnetwork parameter trees must have the same network class",
                    ));
                }
                Ok(object)
            }
            other => Err(deep_learning_error(
                "dlupdate",
                format!("dlupdate: expected matching dlnetwork tree, got {other:?}"),
            )),
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let learnables = networks
        .iter()
        .map(|object| {
            object.properties.get("Learnables").cloned().ok_or_else(|| {
                deep_learning_error("dlupdate", "dlupdate: dlnetwork is missing Learnables")
            })
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let updated = dlupdate_node(function, &learnables, requested_outputs, depth).await?;
    updated
        .into_iter()
        .map(|learnables| apply_learnables_to_network(reference.clone(), learnables))
        .collect()
}

fn apply_learnables_to_network(
    mut network: ObjectInstance,
    learnables: Value,
) -> BuiltinResult<Value> {
    let mut parameter_values = Vec::new();
    match &learnables {
        Value::Struct(st) => {
            let layer_names = string_cells(st.fields.get("Layer"), "Layer")?;
            let parameter_names = string_cells(st.fields.get("Parameter"), "Parameter")?;
            let Value::Cell(values) = st.fields.get("Value").ok_or_else(|| {
                deep_learning_error("dlupdate", "dlupdate: learnables tree is missing Value")
            })?
            else {
                return Err(deep_learning_error(
                    "dlupdate",
                    "dlupdate: learnables Value variable must be a cell array",
                ));
            };
            if layer_names.len() != parameter_names.len() || values.data.len() != layer_names.len()
            {
                return Err(deep_learning_error(
                    "dlupdate",
                    "dlupdate: learnables metadata and Value rows must match",
                ));
            }
            for idx in 0..layer_names.len() {
                parameter_values.push((
                    layer_names[idx].clone(),
                    parameter_names[idx].clone(),
                    values.data[idx].clone(),
                ));
            }
        }
        other => {
            return Err(deep_learning_error(
                "dlupdate",
                format!("dlupdate: expected Learnables struct, got {other:?}"),
            ))
        }
    }

    if let Some(Value::Cell(mut layers)) = network.properties.get("Layers").cloned() {
        for layer in &mut layers.data {
            let Value::Object(layer_object) = layer else {
                continue;
            };
            let layer_name = model::layer_name(layer_object);
            for (target_layer, parameter, value) in &parameter_values {
                if *target_layer == layer_name {
                    layer_object
                        .properties
                        .insert(parameter.clone(), value.clone());
                }
            }
        }
        network
            .properties
            .insert("Layers".to_string(), Value::Cell(layers));
    }
    network
        .properties
        .insert("Learnables".to_string(), learnables);
    Ok(Value::Object(network))
}

fn string_cells(value: Option<&Value>, label: &'static str) -> BuiltinResult<Vec<String>> {
    let Some(Value::Cell(cell)) = value else {
        return Err(deep_learning_error(
            "dlupdate",
            format!("dlupdate: learnables tree is missing {label} cell metadata"),
        ));
    };
    cell.data
        .iter()
        .map(|value| match value {
            Value::String(text) => Ok(text.clone()),
            other => Err(deep_learning_error(
                "dlupdate",
                format!("dlupdate: learnables {label} entries must be strings, got {other:?}"),
            )),
        })
        .collect()
}

fn require_learnables_table(variable_names: &[String]) -> BuiltinResult<()> {
    for required in ["Layer", "Parameter", "Value"] {
        if !variable_names.iter().any(|name| name == required) {
            return Err(deep_learning_error(
                "dlupdate",
                "dlupdate: learnables tables must contain Layer, Parameter, and Value variables",
            ));
        }
    }
    Ok(())
}

pub(super) fn is_learnables_struct(value: &StructValue) -> bool {
    ["Layer", "Parameter", "Value"]
        .iter()
        .all(|field| value.fields.contains_key(*field))
}

fn require_matching_learnable_metadata(tables: &[StructValue]) -> BuiltinResult<()> {
    let Some(reference) = tables.first() else {
        return Ok(());
    };
    for variable in ["Layer", "Parameter"] {
        let expected = reference.fields.get(variable).ok_or_else(|| {
            deep_learning_error(
                "dlupdate",
                format!("dlupdate: learnables table is missing {variable}"),
            )
        })?;
        for table in tables.iter().skip(1) {
            let actual = table.fields.get(variable).ok_or_else(|| {
                deep_learning_error(
                    "dlupdate",
                    format!("dlupdate: learnables table is missing {variable}"),
                )
            })?;
            if actual != expected {
                return Err(deep_learning_error(
                    "dlupdate",
                    "dlupdate: learnables table Layer and Parameter metadata must match",
                ));
            }
        }
    }
    Ok(())
}
