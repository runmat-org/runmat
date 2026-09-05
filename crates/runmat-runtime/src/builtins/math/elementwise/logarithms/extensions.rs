use super::operation::LogarithmOperation;
use crate::BuiltinResult;
use runmat_value::Value;

pub(super) async fn ensure(operation: LogarithmOperation, value: &Value) -> BuiltinResult<()> {
    if is_integer(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            operation.integer_extension(),
            operation.name(),
        )?;
        if !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(value)
            .await?
        {
            return Err(super::errors::invalid(
                operation,
                "integer input lies outside the exact binary64 interval",
            ));
        }
    }
    if crate::builtins::common::validation::value_has_logical_class(value) {
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
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}
