use super::*;
use std::collections::HashMap;

#[runtime_builtin(
    name = "dlgradient",
    category = "deep_learning",
    summary = "Differentiate a traced scalar dlarray loss with respect to dlarray variables.",
    keywords = "dlgradient,deep learning,automatic differentiation,gradient",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::DLGRADIENT_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::autodiff::gradient"
)]
pub(super) async fn dlgradient_builtin(loss: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
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
use backward::backward;

fn gradient_tree_for_target(
    target: &Value,
    gradients: &HashMap<usize, Tensor>,
) -> BuiltinResult<Value> {
    match target {
        Value::Object(object) if object.class_name == DLARRAY_CLASS => {
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
            Ok(super::object_with_identity(
                DLARRAY_CLASS,
                vec![
                    ("Data", Value::Tensor(tensor)),
                    ("Format", format),
                    ("Labels", labels),
                ],
            ))
        }
        Value::Object(object) if model::is_deep_learning_network_object(object) => {
            if let Some(learnables) = object.properties.get("Learnables") {
                gradient_tree_for_target(learnables, gradients)
            } else {
                Ok(Value::Struct(StructValue::new()))
            }
        }
        Value::Object(object) if crate::builtins::table::is_tabular_object(object) => {
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
                let value = st.fields.get("Value").ok_or_else(|| {
                    deep_learning_error("dlgradient", "dlgradient: learnables tree is missing Value")
                })?;
                out.insert("Value", gradient_tree_for_target(value, gradients)?);
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

fn is_learnables_struct(value: &StructValue) -> bool {
    ["Layer", "Parameter", "Value"]
        .iter()
        .all(|field| value.fields.contains_key(*field))
}
