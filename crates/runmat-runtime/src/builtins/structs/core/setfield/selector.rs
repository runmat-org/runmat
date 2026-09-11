use super::assignment::assign_into_value;
use super::errors::{self, BUILTIN_NAME};
use crate::builtins::structs::core::field_path::{self, FieldStep, IndexSelector};
use crate::{gather_if_needed_async, BuiltinResult};
use runmat_builtins::{SETFIELD_ERROR_INDEX_SHAPE, SETFIELD_INDEXED_RESIDENT_EXTENSION};
use runmat_value::Value;

#[async_recursion::async_recursion(?Send)]
pub(super) async fn assign_with_selector(
    value: Value,
    selector: &IndexSelector,
    rest: &[FieldStep],
    rhs: Value,
) -> BuiltinResult<Value> {
    if matches!(value, Value::GpuTensor(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SETFIELD_INDEXED_RESIDENT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let host_value = gather_if_needed_async(&value)
        .await
        .map_err(|flow| errors::remap(flow, Some("setfield: ")))?;
    let plan = field_path::build_plan(&host_value, selector, rest.is_empty())
        .map_err(|error| errors::remap_index(error, Some("setfield: ")))?;
    if rest.is_empty() {
        return crate::indexing::value::assign_with_plan(host_value, &plan, rhs)
            .await
            .map_err(|error| errors::remap_index(error, Some("setfield: ")));
    }
    if plan.indices.len() != 1 {
        return Err(errors::with_message(
            "setfield: nested traversal requires a scalar selection",
            &SETFIELD_ERROR_INDEX_SHAPE,
        ));
    }
    let current = crate::indexing::value::read_with_plan(host_value.clone(), &plan)
        .map_err(|error| errors::remap_index(error, Some("setfield: ")))?;
    let updated = assign_into_value(current, rest, rhs).await?;
    crate::indexing::value::assign_with_plan(host_value, &plan, updated)
        .await
        .map_err(|error| errors::remap_index(error, Some("setfield: ")))
}
