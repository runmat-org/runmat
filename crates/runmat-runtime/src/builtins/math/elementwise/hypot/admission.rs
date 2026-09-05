use runmat_builtins::{
    HYPOT_CHARACTER_INPUT_EXTENSION, HYPOT_ERROR_INVALID_INPUT, HYPOT_ERROR_TOO_MANY_OUTPUTS,
    HYPOT_INTEGER_INPUT_EXTENSION, HYPOT_LOGICAL_INPUT_EXTENSION,
};
use runmat_value::Value;

use crate::BuiltinResult;

use super::{errors, BUILTIN_NAME};

pub(super) fn output_count() -> BuiltinResult<()> {
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
        return Err(errors::with_detail(
            &HYPOT_ERROR_TOO_MANY_OUTPUTS,
            "only one output is defined",
        ));
    }
    Ok(())
}

pub(super) fn extensions(left: &Value, right: &Value) -> BuiltinResult<()> {
    for value in [left, right] {
        if is_logical(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &HYPOT_LOGICAL_INPUT_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        if matches!(value, Value::CharArray(_)) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &HYPOT_CHARACTER_INPUT_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
    }
    Ok(())
}

pub(super) fn integer_input(value: &Value) -> BuiltinResult<()> {
    if !crate::builtins::common::validation::value_has_native_integer_class(value) {
        return Ok(());
    }
    crate::compatibility::ensure_builtin_extension_enabled(
        &HYPOT_INTEGER_INPUT_EXTENSION,
        BUILTIN_NAME,
    )?;
    if !matches!(value, Value::GpuTensor(_))
        && !crate::builtins::common::validation::native_integer_value_is_exact_f64(value)
    {
        return Err(inexact_integer());
    }
    Ok(())
}

pub(super) fn gathered_integer_boundary(value: &Value) -> BuiltinResult<()> {
    if !crate::builtins::common::validation::native_integer_value_is_exact_f64(value) {
        return Err(inexact_integer());
    }
    Ok(())
}

fn inexact_integer() -> crate::RuntimeError {
    errors::terminal(
        &HYPOT_ERROR_INVALID_INPUT,
        "integer input must be exactly representable as double",
    )
}

fn is_logical(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}
