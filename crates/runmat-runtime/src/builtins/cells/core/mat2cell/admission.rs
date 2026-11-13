use runmat_builtins::MAT2CELL_INTEGER_PARTITIONS_EXTENSION;
use runmat_value::Value;

use super::error::{mat2cell_error_with_message, MAT2CELL_ERROR_INVALID_INPUT};
use crate::{gather_if_needed_async, BuiltinResult};

pub(super) async fn gather_input(value: Value) -> BuiltinResult<Value> {
    reject_explicit_residency(&value, "input array")?;
    gather_if_needed_async(&value).await
}

pub(super) async fn gather_partitions(
    values: crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<crate::call::BuiltinArguments> {
    if !values.iter().any(crate::dispatcher::value_contains_gpu) {
        for value in &values {
            validate_partition(value)?;
        }
        return Ok(values);
    }
    // Keep validation adjacent to each download: an invalid earlier
    // partition must reject before a later provider can be called.
    values
        .try_map_preserving_roots_async(values.len(), runtime, async |index| {
            let value = &values[index];
            validate_partition(value)?;
            gather_if_needed_async(value).await
        })
        .await
}

fn validate_partition(value: &Value) -> BuiltinResult<()> {
    reject_explicit_residency(value, "partition vector")?;
    if is_typed_integer(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &MAT2CELL_INTEGER_PARTITIONS_EXTENSION,
            "mat2cell",
        )?;
    }
    Ok(())
}

fn reject_explicit_residency(value: &Value, role: &str) -> BuiltinResult<()> {
    if matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
    {
        return Err(mat2cell_error_with_message(
            format!("mat2cell: explicit gpuArray {role} is not supported"),
            &MAT2CELL_ERROR_INVALID_INPUT,
        ));
    }
    Ok(())
}

fn is_typed_integer(value: &Value) -> bool {
    match value {
        Value::Int(_) => true,
        Value::Tensor(tensor) => tensor.integer_storage().is_some(),
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_integer_type(handle).is_some(),
        _ => false,
    }
}
