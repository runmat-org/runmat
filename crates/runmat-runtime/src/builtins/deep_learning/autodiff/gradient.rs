use super::*;
use runmat_builtins::DeepLearningParameterTableColumn as Column;
use std::collections::HashMap;

mod parameter_table;

#[runtime_builtin(
    name = "dlgradient",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::autodiff::gradient"
)]
pub(crate) async fn dlgradient_builtin(loss: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.is_empty() {
        return Err(deep_learning_error(
            "dlgradient",
            "dlgradient: expected at least one dlarray, learnables, or dlnetwork target",
        ));
    }
    let requested = crate::output_count::current_output_count().unwrap_or(1);
    if requested > rest.len() {
        return Err(deep_learning_error(
            "dlgradient",
            format!(
                "dlgradient: requested {requested} outputs for {} gradient targets",
                rest.len()
            ),
        ));
    }
    validate_host_tape_dlarray_data(&loss)?;
    let Some(loss_node) = dlarray_node_id(&loss) else {
        return Err(deep_learning_error(
            "dlgradient",
            "dlgradient: loss must be a traced dlarray value produced inside dlfeval",
        ));
    };
    let loss_tensor = value_for_node(loss_node)?;
    if tensor::tensor_element_len(&loss_tensor) != 1 {
        return Err(deep_learning_error(
            "dlgradient",
            "dlgradient: loss must be scalar",
        ));
    }
    for target in rest.iter().take(requested) {
        targets::validate(target)?;
    }
    let gradients = backward(loss_node)?;
    let mut outputs = Vec::with_capacity(requested);
    for target in rest.iter().take(requested) {
        outputs.push(gradient_tree_for_target(target, &gradients)?);
    }
    if requested == 1 {
        Ok(outputs.remove(0))
    } else {
        Ok(Value::OutputList(outputs))
    }
}

mod backward;
mod targets;
use backward::backward;

#[cfg(test)]
mod tests;

pub(super) fn gradient_tree_for_target(
    target: &Value,
    gradients: &HashMap<usize, Tensor>,
) -> BuiltinResult<Value> {
    match target {
        Value::Object(object) if object.class_name == DLARRAY_CLASS => {
            validate_host_tape_dlarray_data(target)?;
            let Some(id) = dlarray_node_id(target) else {
                return Err(deep_learning_error(
                    "dlgradient",
                    "dlgradient: target dlarray was not traced inside the active dlfeval",
                ));
            };
            let tensor = if let Some(tensor) = gradients.get(&id).cloned() {
                tensor
            } else {
                let target_tensor = host_tensor_from_value(target, "dlgradient", "target")?;
                floating_tensor_from_f64(
                    vec![0.0; tensor::tensor_element_len(&target_tensor)],
                    target_tensor.shape.clone(),
                    target_tensor.numeric_dtype(),
                    "dlgradient",
                )?
            };
            let (format, labels) = dlarray_format_and_labels(target);
            Ok(super::super::object_with_identity(
                DLARRAY_CLASS,
                vec![
                    ("Data", Value::Tensor(tensor)),
                    ("Format", format),
                    ("Labels", labels),
                ],
            ))
        }
        Value::Object(object) if model::is_deep_learning_network_object(object) => {
            let Value::Object(table) = model::project_learnables(object, "dlgradient")? else {
                unreachable!("network learnables projection is a table")
            };
            gradient_network_table(&table, gradients)
        }
        Value::Object(object) if crate::builtins::table::is_tabular_object(object) => {
            if parameter_table::values(object, "dlgradient")?.is_some() {
                return gradient_network_table(object, gradients);
            }
            let variables = crate::builtins::table::table_variables(object)
                .map_err(|err| deep_learning_error("dlgradient", err.to_string()))?;
            let mut updated = StructValue::new();
            for (name, value) in variables.fields {
                updated.insert(name, gradient_tree_for_target(&value, gradients)?);
            }
            crate::builtins::table::table_replace_variables_like(object, updated)
                .map_err(|err| deep_learning_error("dlgradient", err.to_string()))
        }
        Value::Struct(st) => {
            if is_learnables_struct(st) {
                let mut out = st.clone();
                let value = st.fields.get(Column::Value.name()).ok_or_else(|| {
                    deep_learning_error("dlgradient", "dlgradient: learnables tree is missing Value")
                })?;
                out.insert(Column::Value.name(), gradient_tree_for_target(value, gradients)?);
                return Ok(Value::Struct(out));
            }
            let mut out = StructValue::new();
            for (name, value) in &st.fields {
                out.insert(name.clone(), gradient_tree_for_target(value, gradients)?);
            }
            Ok(Value::Struct(out))
        }
        Value::StructArray(array) => array
            .clone()
            .try_map_values(|value| gradient_tree_for_target(&value, gradients))
            .map(Value::StructArray),
        Value::Cell(cell) => {
            let data = cell
                .data
                .iter()
                .map(|value| gradient_tree_for_target(value, gradients))
                .collect::<BuiltinResult<Vec<_>>>()?;
            CellArray::new_with_shape(data, cell.shape.clone())
                .map(Value::Cell)
                .map_err(|err| deep_learning_error("dlgradient", err))
        }
        other => Err(deep_learning_error(
            "dlgradient",
            format!("dlgradient: target must be a traced dlarray, learnables tree, or dlnetwork, got {other:?}"),
        )),
    }
}

fn gradient_network_table(
    table: &ObjectInstance,
    gradients: &HashMap<usize, Tensor>,
) -> BuiltinResult<Value> {
    let values = parameter_table::values(table, "dlgradient")?.expect("parameter table schema");
    let gradients = values
        .data
        .iter()
        .map(|value| gradient_tree_for_target(value, gradients))
        .collect::<BuiltinResult<Vec<_>>>()?;
    let mut variables = crate::builtins::table::table_variables(table)
        .map_err(|error| deep_learning_error("dlgradient", error.to_string()))?;
    variables.insert(
        "Value",
        Value::Cell(
            CellArray::new_with_shape(gradients, values.shape.clone())
                .map_err(|error| deep_learning_error("dlgradient", error))?,
        ),
    );
    crate::builtins::table::table_replace_variables_like(table, variables)
        .map_err(|error| deep_learning_error("dlgradient", error.to_string()))
}

fn validate_host_tape_dlarray_data(value: &Value) -> BuiltinResult<()> {
    let Value::Object(object) = value else {
        return Ok(());
    };
    if object.class_name == DLARRAY_CLASS {
        match object.properties.get("Data") {
            Some(Value::GpuTensor(_)) => {
                return Err(unsupported_error(
                    "dlgradient",
                    "dlgradient: provider-resident dlarray Data requires unavailable tape kernels",
                ));
            }
            Some(Value::SparseTensor(_)) => {
                return Err(deep_learning_error(
                    "dlgradient",
                    "dlgradient: sparse dlarray Data are not supported by the host tape",
                ));
            }
            Some(Value::Num(_) | Value::Bool(_) | Value::LogicalArray(_)) => {}
            Some(Value::Tensor(tensor))
                if matches!(
                    tensor.numeric_dtype(),
                    NumericDType::F64 | NumericDType::F32
                ) => {}
            Some(_) => {
                return Err(deep_learning_error(
                    "dlgradient",
                    "dlgradient: dlarray Data must be real double, single, or logical host data",
                ));
            }
            None => {
                return Err(deep_learning_error(
                    "dlgradient",
                    "dlgradient: traced dlarray is missing Data",
                ));
            }
        }
    }
    Ok(())
}

fn is_learnables_struct(value: &StructValue) -> bool {
    Column::ALL
        .iter()
        .all(|column| value.fields.contains_key(column.name()))
}
