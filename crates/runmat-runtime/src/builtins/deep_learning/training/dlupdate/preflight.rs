//! Read-only tree admission before the first user callback is invoked.

use super::*;

pub(super) fn matching_trees(values: &[Value]) -> BuiltinResult<()> {
    let references = values.iter().collect::<Vec<_>>();
    matching_node(&references, 0)
}

fn matching_node(values: &[&Value], depth: usize) -> BuiltinResult<()> {
    if depth > DLUPDATE_MAX_DEPTH {
        return Err(deep_learning_error(
            "dlupdate",
            "dlupdate: parameter tree nesting exceeds supported recursion depth",
        ));
    }
    let Some(reference) = values.first() else {
        return Ok(());
    };
    match reference {
        Value::Struct(st) if is_learnables_struct(st) => {
            let mut columns = Vec::with_capacity(values.len());
            for value in values {
                let Value::Struct(other) = value else {
                    return Err(structure_mismatch());
                };
                if !is_learnables_struct(other) {
                    return Err(structure_mismatch());
                }
                for field in ["Layer", "Parameter"] {
                    if other.fields.get(field) != st.fields.get(field) {
                        return Err(deep_learning_error(
                            "dlupdate",
                            "dlupdate: learnables table Layer and Parameter metadata must match",
                        ));
                    }
                }
                columns.push(other.fields.get("Value").expect("learnables Value exists"));
            }
            matching_node(&columns, depth + 1)
        }
        Value::Struct(st) => {
            let fields = st.field_names().collect::<Vec<_>>();
            let mut peers = Vec::with_capacity(values.len());
            for value in values {
                let Value::Struct(other) = value else {
                    return Err(structure_mismatch());
                };
                if !other.field_names().eq(fields.iter().copied()) {
                    return Err(deep_learning_error(
                        "dlupdate",
                        "dlupdate: struct parameter trees must have the same fields in the same order",
                    ));
                }
                peers.push(other);
            }
            for field in fields {
                let children = peers
                    .iter()
                    .map(|peer| peer.fields.get(field).expect("matching field exists"))
                    .collect::<Vec<_>>();
                matching_node(&children, depth + 1)?;
            }
            Ok(())
        }
        Value::Cell(cell) => {
            let mut peers = Vec::with_capacity(values.len());
            for value in values {
                let Value::Cell(other) = value else {
                    return Err(structure_mismatch());
                };
                if other.shape != cell.shape {
                    return Err(deep_learning_error(
                        "dlupdate",
                        "dlupdate: cell parameter trees must have the same shape",
                    ));
                }
                peers.push(other);
            }
            for index in 0..cell.data().len() {
                let children = peers
                    .iter()
                    .map(|peer| &peer.data()[index])
                    .collect::<Vec<_>>();
                matching_node(&children, depth + 1)?;
            }
            Ok(())
        }
        Value::StructArray(array) => {
            let fields = array.field_names().collect::<Vec<_>>();
            let mut peers = Vec::with_capacity(values.len());
            for value in values {
                let Value::StructArray(other) = value else {
                    return Err(structure_mismatch());
                };
                if other.shape() != array.shape() || !other.field_names().eq(fields.iter().copied())
                {
                    return Err(deep_learning_error(
                        "dlupdate",
                        "dlupdate: structure-array parameter trees must have matching shapes and ordered fields",
                    ));
                }
                peers.push(other);
            }
            for field in fields {
                for index in 0..array.len() {
                    let children = peers
                        .iter()
                        .map(|peer| {
                            peer.field_values(field)
                                .and_then(|column| column.get(index))
                                .ok_or_else(|| {
                                    deep_learning_error(
                                        "dlupdate",
                                        "dlupdate: structure-array field storage is incomplete",
                                    )
                                })
                        })
                        .collect::<BuiltinResult<Vec<_>>>()?;
                    matching_node(&children, depth + 1)?;
                }
            }
            Ok(())
        }
        Value::Object(network) if model::is_deep_learning_network_object(network) => {
            let mut peers = Vec::with_capacity(values.len());
            for value in values {
                let Value::Object(other) = value else {
                    return Err(structure_mismatch());
                };
                if !model::is_deep_learning_network_object(other)
                    || other.class_name() != network.class_name()
                {
                    return Err(structure_mismatch());
                }
                peers.push(other);
            }
            model::validate_matching_networks(&peers, "dlupdate")
        }
        Value::Object(table) if crate::builtins::table::is_tabular_object(table) => {
            matching_tables(values, table, depth)
        }
        _ if values.iter().any(|value| is_tree_container(value)) => Err(structure_mismatch()),
        _ => Ok(()),
    }
}

fn matching_tables(
    values: &[&Value],
    reference: &ObjectInstance,
    depth: usize,
) -> BuiltinResult<()> {
    let reference_vars = crate::builtins::table::table_variables_ref(reference)
        .map_err(|error| deep_learning_error("dlupdate", error.to_string()))?;
    let names = reference_vars
        .map(|vars| vars.fields.keys().collect::<Vec<_>>())
        .unwrap_or_default();
    for field in ["Layer", "Parameter", "Value"] {
        if !names.contains(&field) {
            return Err(deep_learning_error(
                "dlupdate",
                "dlupdate: learnables tables must contain Layer, Parameter, and Value variables",
            ));
        }
    }
    let height = reference_vars
        .and_then(|vars| vars.fields.values().next())
        .map(crate::builtins::table::value_row_count)
        .transpose()
        .map_err(|error| deep_learning_error("dlupdate", error.to_string()))?
        .unwrap_or(0);
    let mut variables = Vec::with_capacity(values.len());
    for value in values {
        let Value::Object(other) = value else {
            return Err(structure_mismatch());
        };
        if !crate::builtins::table::is_tabular_object(other)
            || other.class_name() != reference.class_name()
        {
            return Err(structure_mismatch());
        }
        let other_vars = crate::builtins::table::table_variables_ref(other)
            .map_err(|error| deep_learning_error("dlupdate", error.to_string()))?;
        let other_names = other_vars
            .map(|vars| vars.fields.keys().collect::<Vec<_>>())
            .unwrap_or_default();
        let other_height = other_vars
            .and_then(|vars| vars.fields.values().next())
            .map(crate::builtins::table::value_row_count)
            .transpose()
            .map_err(|error| deep_learning_error("dlupdate", error.to_string()))?
            .unwrap_or(0);
        if other_names != names || other_height != height {
            return Err(structure_mismatch());
        }
        variables.push(other_vars.expect("required learnables table variables exist"));
    }
    for field in ["Layer", "Parameter"] {
        let expected = variables[0].fields.get(field);
        if variables
            .iter()
            .skip(1)
            .any(|other| other.fields.get(field) != expected)
        {
            return Err(deep_learning_error(
                "dlupdate",
                "dlupdate: learnables table Layer and Parameter metadata must match",
            ));
        }
    }
    let columns = variables
        .iter()
        .map(|vars| {
            vars.fields
                .get("Value")
                .expect("required table variable exists")
        })
        .collect::<Vec<_>>();
    if columns.iter().any(|value| !matches!(value, Value::Cell(_))) {
        return Err(deep_learning_error(
            "dlupdate",
            "dlupdate: learnables table Value variable must be a cell array",
        ));
    }
    matching_node(&columns, depth + 1)
}

fn structure_mismatch() -> crate::RuntimeError {
    deep_learning_error(
        "dlupdate",
        "dlupdate: all parameter trees must have matching container structure",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use std::error::Error;

    fn missing_callback() -> Value {
        Value::FunctionHandle("__missing_dlupdate_callback".into())
    }

    #[test]
    fn network_leaf_mismatch_is_symmetric_and_never_calls_the_callback() {
        let network = Value::Object(ObjectInstance::new(model::DLNETWORK_CLASS.to_string()));
        for trees in [
            [network.clone(), Value::Num(1.0)],
            [Value::Num(1.0), network.clone()],
        ] {
            let error = block_on(dlupdate_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    missing_callback(),
                    trees[0].clone(),
                    trees[1].clone(),
                ]),
            ))
            .expect_err("network/leaf mismatch must precede callback invocation");
            assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
            assert!(error.to_string().contains("matching container structure"));
            assert!(error.source().is_none());
        }
    }

    #[test]
    fn later_nested_shape_mismatch_precedes_an_earlier_leaf_callback() {
        let mut left = runmat_value::UnboundedStructValueBuilder::new();
        left.insert("First", Value::Num(1.0));
        left.insert(
            "Later",
            Value::Cell(CellArray::new(vec![Value::Num(2.0)], 1, 1).unwrap()),
        );
        let mut right = runmat_value::UnboundedStructValueBuilder::new();
        right.insert("First", Value::Num(3.0));
        right.insert(
            "Later",
            Value::Cell(CellArray::new(vec![Value::Num(4.0), Value::Num(5.0)], 1, 2).unwrap()),
        );
        let error = block_on(dlupdate_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                missing_callback(),
                Value::Struct(left.finish()),
                Value::Struct(right.finish()),
            ]),
        ))
        .expect_err("later mismatch must be admitted before first callback");
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        assert!(error.to_string().contains("same shape"));
        assert!(error.source().is_none());
    }
}
