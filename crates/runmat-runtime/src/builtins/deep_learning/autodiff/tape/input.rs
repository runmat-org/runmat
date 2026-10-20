use super::super::*;
use super::{
    attach_node_to_object, dlarray_object_node_id, push_resident_source_node, push_source_node,
    NodeValue,
};

mod network;
use network::annotate_network_object;

pub(super) async fn prepare_tape_inputs(
    values: &crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<crate::call::BuiltinArguments> {
    values
        .try_map_mixed_async(values.len(), runtime, async |index| {
            let value = annotate_tree(values.get(index).expect("tape input index").clone())?;
            let value = match runtime {
                Some(runtime) => crate::execution::RootedValue::in_context(value, runtime)?,
                None => crate::execution::RootedValue::new(value)
                    .map_err(|error| error.into_runtime_error())?,
            };
            Ok(crate::execution::ArgumentProjection::Rooted(value))
        })
        .await
}

pub(in crate::builtins::deep_learning) fn annotate_tree(value: Value) -> BuiltinResult<Value> {
    if !tape_is_active() {
        return Ok(value);
    }
    match value {
        Value::Object(mut object) if object.class_name().is(DLARRAY_CLASS) => {
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
            let Value::Struct(updated) = annotate_tree(Value::Struct(variables))? else {
                unreachable!("annotation preserves the variable structure")
            };
            crate::builtins::table::table_replace_variables_like(&object, updated)
                .map_err(|err| deep_learning_error("dlfeval", err.to_string()))
        }
        Value::Object(object) => Ok(Value::Object(object)),
        Value::Struct(st) => {
            if !st.fields.values().any(requires_annotation) {
                return Ok(Value::Struct(st));
            }
            let context = crate::context::legacy::active();
            let prepared_updated =
                crate::execution::structure::replacement::record(&st, context.as_ref())?;
            let _roots = prepared_updated.1;
            let mut updated = prepared_updated.0;
            for value in updated
                .fields
                .try_values_mut_unique()
                .expect("new admitted map is unique")
            {
                annotate_slot(value)?;
            }
            Ok(Value::Struct(updated))
        }
        Value::StructArray(array) => {
            if !array.any_value(requires_annotation) {
                return Ok(Value::StructArray(array));
            }
            let context = crate::context::legacy::active();
            let prepared_updated =
                crate::execution::structure::replacement::array(&array, context.as_ref())?;
            let _roots = prepared_updated.1;
            let mut updated = prepared_updated.0;
            for name in array.field_names() {
                for value in updated
                    .try_column_mut_unique(name)
                    .expect("new admitted column is unique")
                {
                    annotate_slot(value)?;
                }
            }
            Ok(Value::StructArray(updated))
        }
        Value::Cell(mut cell) => {
            if let Some(values) = cell.try_data_mut_unique() {
                for value in values {
                    annotate_slot(value)?;
                }
            } else if cell.data().iter().any(requires_annotation) {
                let context = crate::context::legacy::active();
                let prepared_updated =
                    crate::execution::structure::replacement::cell(&cell, context.as_ref())?;
                let _roots = prepared_updated.1;
                let mut updated = prepared_updated.0;
                for value in updated
                    .try_data_mut_unique()
                    .expect("new admitted Cell slots are unique")
                {
                    annotate_slot(value)?;
                }
                return Ok(Value::Cell(updated));
            }
            Ok(Value::Cell(cell))
        }
        other => Ok(other),
    }
}

/// A unique aggregate is private to this owned transformation. Temporarily
/// emptying one slot moves its child into the recursive annotator without a
/// Value-tree clone; an error drops the aggregate rather than publishing it.
fn annotate_slot(slot: &mut Value) -> BuiltinResult<()> {
    let child = std::mem::replace(slot, Value::Num(0.0));
    *slot = annotate_tree(child)?;
    Ok(())
}

/// Only skip a shared Cell replacement when no descendant can acquire a new
/// tape node. This walk neither clones children nor changes the source owner.
fn requires_annotation(value: &Value) -> bool {
    match value {
        Value::Object(object) if object.class_name().is(DLARRAY_CLASS) => {
            dlarray_object_node_id(object).is_none()
        }
        Value::Object(object) => {
            model::is_deep_learning_network_object(object)
                || crate::builtins::table::is_tabular_object(object)
        }
        Value::Struct(st) => st.fields.values().any(requires_annotation),
        Value::StructArray(array) => array.any_value(requires_annotation),
        Value::Cell(cell) => cell.data().iter().any(requires_annotation),
        _ => false,
    }
}

pub(in crate::builtins::deep_learning) fn annotate_dlarray_value(
    value: Value,
) -> BuiltinResult<Value> {
    if !tape_is_active() {
        return Ok(value);
    }
    match value {
        Value::Object(mut object) if object.class_name().is(DLARRAY_CLASS) => {
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
        .properties()
        .get("Data")
        .ok_or_else(|| deep_learning_error("dlarray", "dlarray: traced object is missing Data"))?;
    let value = match data {
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_storage(handle)
                != runmat_accelerate_api::GpuTensorStorage::Real
                || runmat_accelerate_api::handle_precision(handle).is_none()
            {
                return Err(deep_learning_error(
                    "dlarray",
                    "dlarray: resident tape source requires real single or double storage",
                ));
            }
            NodeValue::Resident(handle.clone())
        }
        _ => NodeValue::Host(host_tensor_from_value(data, "dlarray", "Data")?),
    };
    annotate_dlarray_object_with_value(object, value)
}

fn annotate_dlarray_object_with_value(
    object: &mut ObjectInstance,
    value: NodeValue,
) -> BuiltinResult<()> {
    if dlarray_object_node_id(object).is_some() {
        return Ok(());
    }
    let data = object
        .properties()
        .get("Data")
        .ok_or_else(|| deep_learning_error("dlarray", "dlarray: traced object is missing Data"))?;
    let node = match value {
        NodeValue::Host(tensor) => push_source_node(tensor, data)?,
        NodeValue::Resident(handle) => push_resident_source_node(handle)?,
    };
    attach_node_to_object(object, node)
}
