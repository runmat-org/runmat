use runmat_builtins::{
    HEAVISIDE_CHARACTER_INPUT_EXTENSION, HEAVISIDE_GPU_INPUT_EXTENSION,
    HEAVISIDE_INTEGER_INPUT_EXTENSION, HEAVISIDE_LOGICAL_INPUT_EXTENSION,
};
use runmat_value::Value;

use crate::BuiltinResult;

use super::BUILTIN_NAME;

pub(super) fn validate(value: &Value) -> BuiltinResult<()> {
    if is_integer(value) {
        require(&HEAVISIDE_INTEGER_INPUT_EXTENSION)?;
    }
    if is_logical(value) {
        require(&HEAVISIDE_LOGICAL_INPUT_EXTENSION)?;
    }
    if matches!(value, Value::CharArray(_)) {
        require(&HEAVISIDE_CHARACTER_INPUT_EXTENSION)?;
    }
    if matches!(value, Value::GpuTensor(_)) {
        require(&HEAVISIDE_GPU_INPUT_EXTENSION)?;
    }
    Ok(())
}

fn require(extension: &runmat_builtins::BuiltinExtensionDescriptor) -> BuiltinResult<()> {
    crate::compatibility::ensure_builtin_extension_enabled(extension, BUILTIN_NAME)
}

fn is_integer(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn is_logical(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}
