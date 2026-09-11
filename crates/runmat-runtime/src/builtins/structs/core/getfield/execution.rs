use super::arguments;
use super::errors::{self, BUILTIN_NAME};
use super::object;
use crate::builtins::structs::core::field_path::{self, IndexSelector};
use crate::{gather_if_needed_async, BuiltinResult};
use runmat_builtins::{
    GETFIELD_ERROR_INDEX_SELECTOR_TYPE, GETFIELD_ERROR_INDEX_SHAPE,
    GETFIELD_ERROR_NOT_ENOUGH_INPUTS, GETFIELD_INDEXED_RESIDENT_EXTENSION,
};
use runmat_value::Value;

pub(super) async fn getfield(
    base: Value,
    rest: Vec<Value>,
    enforce_public_builtin_extension: bool,
) -> BuiltinResult<Value> {
    if rest.is_empty() {
        return Err(errors::from_descriptor(&GETFIELD_ERROR_NOT_ENOUGH_INPUTS));
    }
    let parsed = arguments::parse(rest)?;
    let mut current = base;
    if let Some(index) = parsed.leading_index {
        current = apply_indices(current, &index, true).await?;
    }
    let field_count = parsed.fields.len();
    for (field_index, step) in parsed.fields.into_iter().enumerate() {
        current =
            object::get_field_value(current, &step.name, enforce_public_builtin_extension).await?;
        if let Some(index) = step.index {
            current = apply_indices(current, &index, field_index + 1 < field_count).await?;
        }
    }
    Ok(current)
}

async fn apply_indices(
    value: Value,
    selector: &IndexSelector,
    require_scalar_traversal: bool,
) -> BuiltinResult<Value> {
    if selector.components.is_empty() {
        return Err(errors::with_message(
            "getfield: index cell must contain at least one element",
            &GETFIELD_ERROR_INDEX_SELECTOR_TYPE,
        ));
    }
    let value = match value {
        Value::GpuTensor(handle) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &GETFIELD_INDEXED_RESIDENT_EXTENSION,
                BUILTIN_NAME,
            )?;
            gather_if_needed_async(&Value::GpuTensor(handle))
                .await
                .map_err(|flow| errors::remap(flow, Some("getfield: ")))?
        }
        other => other,
    };
    let plan = field_path::build_plan(&value, selector, false)
        .map_err(|error| errors::remap_index(error, Some("getfield: ")))?;
    let selected = crate::indexing::value::read_with_plan(value, &plan)
        .map_err(|error| errors::remap_index(error, Some("getfield: ")))?;
    if require_scalar_traversal && plan.indices.len() != 1 {
        return Err(errors::with_message(
            "getfield: intermediate indices must select one element",
            &GETFIELD_ERROR_INDEX_SHAPE,
        ));
    }
    Ok(selected)
}

/// Resolve ordinary dot-member access without applying the compatibility gate
/// for the public getfield object extension. Class and handle dot syntax is a
/// language operation, not a call to that extension.
pub(crate) async fn get_member_value(value: Value, name: &str) -> BuiltinResult<Value> {
    object::get_field_value(value, name, false).await
}
