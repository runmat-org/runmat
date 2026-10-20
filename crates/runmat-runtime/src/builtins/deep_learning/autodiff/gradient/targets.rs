//! Borrowed target admission before reverse execution or gradient allocation.
use super::*;

pub(super) fn node_ids(target: &Value) -> BuiltinResult<Vec<usize>> {
    let mut nodes = Vec::new();
    collect_node_ids(target, &mut nodes)?;
    Ok(nodes)
}

fn collect_node_ids(target: &Value, nodes: &mut Vec<usize>) -> BuiltinResult<()> {
    match target {
        Value::Object(object) if object.class_name().is(DLARRAY_CLASS) => {
            validate_tape_dlarray_data(target)?;
            nodes.push(dlarray_node_id(target).ok_or_else(|| {
                deep_learning_error(
                    "dlgradient",
                    "dlgradient: target dlarray was not traced inside the active dlfeval",
                )
            })?);
            Ok(())
        }
        Value::Object(object) if model::is_deep_learning_network_object(object) => {
            for row in model::parameters::view(object, "dlgradient")? {
                let value = row.value.ok_or_else(|| {
                    deep_learning_error(
                        "dlgradient",
                        "dlgradient: network target has an uninitialized learnable",
                    )
                })?;
                collect_node_ids(value, nodes)?;
            }
            Ok(())
        }
        Value::Object(object) if crate::builtins::table::is_tabular_object(object) => {
            if let Some(values) = parameter_table::values(object, "dlgradient")? {
                return collect_values(values.data().iter(), nodes);
            }
            let variables = crate::builtins::table::table_variables_ref(object)
                .map_err(|error| deep_learning_error("dlgradient", error.to_string()))?
                .ok_or_else(|| {
                    deep_learning_error("dlgradient", "dlgradient: table is missing Variables")
                })?;
            collect_values(variables.fields.values(), nodes)
        }
        Value::Struct(value) if is_learnables_struct(value) => {
            // The legacy structural form explicitly separates row metadata
            // from its derivative-bearing Value subtree.
            collect_node_ids(
                value
                    .fields
                    .get(Column::Value.name())
                    .expect("admitted structural Value"),
                nodes,
            )
        }
        Value::Struct(value) => collect_values(value.fields.values(), nodes),
        Value::StructArray(array) => {
            for element in array.elements() {
                collect_values(element.fields.values(), nodes)?;
            }
            Ok(())
        }
        Value::Cell(values) => collect_values(values.data().iter(), nodes),
        other => Err(deep_learning_error(
            "dlgradient",
            format!("dlgradient: target must be a traced dlarray, learnables tree, or dlnetwork, got {other:?}"),
        )),
    }
}

fn collect_values<'a>(
    values: impl Iterator<Item = &'a Value>,
    nodes: &mut Vec<usize>,
) -> BuiltinResult<()> {
    for value in values {
        collect_node_ids(value, nodes)?;
    }
    Ok(())
}
