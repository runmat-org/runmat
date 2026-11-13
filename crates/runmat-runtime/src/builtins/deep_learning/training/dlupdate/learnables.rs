//! Learnables tables and network reconstruction.
use super::*;

pub(super) async fn dlupdate_learnables_struct(
    runtime: Option<&crate::context::RuntimeContext>,
    function: &Value,
    reference: &StructValue,
    values: &[Value],
    requested_outputs: usize,
    depth: usize,
) -> BuiltinResult<RootedValues> {
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
    let updated_values =
        dlupdate_node(runtime, function, &value_columns, requested_outputs, depth).await?;
    updated_values.with_owned_output_storage(|updated_values| {
        let updated_values = into_owned_outputs(updated_values)?;
        let mut rebuilt = RootedValuesBuilder::with_capacity(requested_outputs);
        for value_column in updated_values {
            let prepared_out =
                crate::execution::structure::replacement::record(reference, runtime)?;
            let _source_roots = prepared_out.1;
            let mut out = prepared_out.0;
            out.try_replace_unique("Value", value_column)
                .expect("learnables schema admitted Value in the private destination");
            rebuilt
                .push(Value::Struct(out))
                .map_err(|error| error.into_runtime_error())?;
        }
        rebuilt.finish().map_err(|error| error.into_runtime_error())
    })
}

pub(super) async fn dlupdate_table(
    runtime: Option<&crate::context::RuntimeContext>,
    function: &Value,
    reference: &ObjectInstance,
    values: &[Value],
    requested_outputs: usize,
    depth: usize,
) -> BuiltinResult<RootedValues> {
    let variable_names = crate::builtins::table::table_variable_names_from_object(reference)
        .map_err(|err| deep_learning_error("dlupdate", err.to_string()))?;
    require_learnables_table(&variable_names)?;
    let reference_height = crate::builtins::table::table_height(reference)
        .map_err(|err| deep_learning_error("dlupdate", err.to_string()))?;
    let tables = values
        .iter()
        .map(|value| match value {
            Value::Object(object) if crate::builtins::table::is_tabular_object(object) => {
                if object.class_name() != reference.class_name() {
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
    let updated_values =
        dlupdate_node(runtime, function, &value_columns, requested_outputs, depth).await?;

    updated_values.with_owned_output_storage(|updated_values| {
        let updated_values = into_owned_outputs(updated_values)?;
        let mut rebuilt = RootedValuesBuilder::with_capacity(requested_outputs);
        for value_column in updated_values {
            let prepared_variables =
                crate::execution::structure::replacement::record(&table_variables[0], runtime)?;
            let _source_roots = prepared_variables.1;
            let mut variables = prepared_variables.0;
            variables
                .try_replace_unique("Value", value_column)
                .expect("learnables schema admitted Value in the private destination");
            let table = crate::builtins::table::table_replace_variables_like(reference, variables)
                .map_err(|err| deep_learning_error("dlupdate", err.to_string()))?;
            rebuilt
                .push(table)
                .map_err(|error| error.into_runtime_error())?;
        }
        rebuilt.finish().map_err(|error| error.into_runtime_error())
    })
}

pub(super) async fn dlupdate_network(
    runtime: Option<&crate::context::RuntimeContext>,
    function: &Value,
    reference: &ObjectInstance,
    values: &[Value],
    requested_outputs: usize,
    depth: usize,
) -> BuiltinResult<RootedValues> {
    let networks = values
        .iter()
        .map(|value| match value {
            Value::Object(object) if model::is_deep_learning_network_object(object) => {
                if object.class_name() != reference.class_name() {
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
    model::validate_matching_networks(&networks, "dlupdate")?;
    let mut projected = RootedValuesBuilder::with_capacity(networks.len());
    for object in &networks {
        let learnables = model::project_learnables(object, "dlupdate")?;
        projected
            .push(learnables)
            .map_err(|error| error.into_runtime_error())?;
    }
    let learnables = projected
        .finish()
        .map_err(|error| error.into_runtime_error())?;
    let updated = dlupdate_node(
        runtime,
        function,
        learnables.values(),
        requested_outputs,
        depth,
    )
    .await?;
    updated.with_owned_output_storage(|updated| {
        let updated = into_owned_outputs(updated)?;
        let mut rebuilt = RootedValuesBuilder::with_capacity(requested_outputs);
        for learnables in updated {
            let network = model::apply_learnables(reference.clone(), learnables, "dlupdate")?;
            rebuilt
                .push(network)
                .map_err(|error| error.into_runtime_error())?;
        }
        rebuilt.finish().map_err(|error| error.into_runtime_error())
    })
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
        .all(|field| value.fields.contains_key(field))
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
