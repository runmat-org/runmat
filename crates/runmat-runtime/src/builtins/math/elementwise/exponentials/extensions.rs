use runmat_value::Value;

use crate::BuiltinResult;

use super::operation::ExponentialOperation;

pub(super) fn ensure(operation: ExponentialOperation, value: &Value) -> BuiltinResult<()> {
    if is_integer(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            operation.integer_extension(),
            operation.name(),
        )?;
    }
    if is_logical(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            operation.logical_extension(),
            operation.name(),
        )?;
    }
    if matches!(value, Value::CharArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            operation.character_extension(),
            operation.name(),
        )?;
    }
    Ok(())
}

fn is_integer(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::SparseTensor(sparse) if sparse.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn is_logical(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::SparseTensor(sparse) if sparse.is_logical())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}
