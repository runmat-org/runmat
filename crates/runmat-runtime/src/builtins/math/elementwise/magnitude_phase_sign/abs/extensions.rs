use runmat_builtins::{ABS_CHARACTER_INPUT_EXTENSION, ABS_LOGICAL_INPUT_EXTENSION};
use runmat_value::Value;

use crate::BuiltinResult;

use super::BUILTIN_NAME;

pub(super) fn ensure(value: &Value) -> BuiltinResult<()> {
    let logical = matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::SparseTensor(sparse) if sparse.is_logical())
        || matches!(
            value,
            Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle)
        );
    if logical {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ABS_LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(value, Value::CharArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ABS_CHARACTER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(
        value,
        Value::SparseTensor(sparse) if sparse.integer_storage().is_some()
    ) {
        crate::compatibility::ensure_sparse_integer_extension_enabled(BUILTIN_NAME)?;
    }
    Ok(())
}
