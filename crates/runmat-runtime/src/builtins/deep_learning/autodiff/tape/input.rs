use super::super::*;
use super::{dlarray_object_node_id, push_source_node};

mod network;
use network::annotate_network_object;

pub(super) fn prepare_tape_inputs(values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    values.into_iter().map(annotate_tree).collect()
}

pub(in crate::builtins::deep_learning) fn annotate_tree(value: Value) -> BuiltinResult<Value> {
    if !tape_is_active() {
        return Ok(value);
    }
    match value {
        Value::Object(mut object) if object.class_name == DLARRAY_CLASS => {
            annotate_dlarray_object(&mut object)?;
            Ok(Value::Object(object))
        }
        Value::Object(mut object) if model::is_deep_learning_network_object(&object) => {
            annotate_network_object(&mut object)?;
            Ok(Value::Object(object))
        }
        Value::Object(object) if crate::builtins::table::is_tabular_object(&object) => {
            let variables = crate::builtins::table::table_variables(&object)
                .map_err(|err| deep_learning_error("dlfeval", err.to_string()))?;
            let mut updated = StructValue::new();
            for (name, value) in variables.fields {
                updated.insert(name, annotate_tree(value)?);
            }
            crate::builtins::table::table_replace_variables_like(&object, updated)
                .map_err(|err| deep_learning_error("dlfeval", err.to_string()))
        }
        Value::Object(object) => Ok(Value::Object(object)),
        Value::Struct(mut st) => {
            for value in st.fields.values_mut() {
                *value = annotate_tree(value.clone())?;
            }
            Ok(Value::Struct(st))
        }
        Value::StructArray(array) => array.try_map_values(annotate_tree).map(Value::StructArray),
        Value::Cell(cell) => {
            let data = cell
                .data
                .into_iter()
                .map(annotate_tree)
                .collect::<BuiltinResult<Vec<_>>>()?;
            CellArray::new_with_shape(data, cell.shape)
                .map(Value::Cell)
                .map_err(|err| deep_learning_error("dlfeval", err))
        }
        other => Ok(other),
    }
}

pub(in crate::builtins::deep_learning) fn annotate_dlarray_value(
    value: Value,
) -> BuiltinResult<Value> {
    if !tape_is_active() {
        return Ok(value);
    }
    match value {
        Value::Object(mut object) if object.class_name == DLARRAY_CLASS => {
            annotate_dlarray_object(&mut object)?;
            Ok(Value::Object(object))
        }
        other => Ok(other),
    }
}

fn annotate_dlarray_object(object: &mut ObjectInstance) -> BuiltinResult<()> {
    if dlarray_object_node_id(object).is_some() {
        return Ok(());
    }
    let data = object
        .properties
        .get("Data")
        .ok_or_else(|| deep_learning_error("dlarray", "dlarray: traced object is missing Data"))?;
    let tensor = host_tensor_from_value(data, "dlarray", "Data")?;
    annotate_dlarray_object_with_tensor(object, tensor)
}

fn annotate_dlarray_object_with_tensor(
    object: &mut ObjectInstance,
    tensor: Tensor,
) -> BuiltinResult<()> {
    if dlarray_object_node_id(object).is_some() {
        return Ok(());
    }
    let data = object
        .properties
        .get("Data")
        .ok_or_else(|| deep_learning_error("dlarray", "dlarray: traced object is missing Data"))?;
    let node = push_source_node(tensor, data)?;
    let tape_id = active_tape()
        .expect("node insertion requires active tape")
        .borrow()
        .id;
    object.properties.insert(
        AD_NODE_PROPERTY.to_string(),
        Value::Int(IntValue::U64(
            u64::try_from(node).expect("node index fits u64"),
        )),
    );
    object.properties.insert(
        AD_TAPE_PROPERTY.to_string(),
        Value::Int(IntValue::U64(tape_id)),
    );
    Ok(())
}
