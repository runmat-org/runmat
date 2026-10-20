use super::*;

pub(super) fn ensure_gauspuls_time_extensions(value: &Value) -> BuiltinResult<()> {
    if is_typed_integer_value(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &GAUSPULS_INTEGER_TIME_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if is_logical_value(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &GAUSPULS_LOGICAL_TIME_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(value, Value::GpuTensor(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &GAUSPULS_RESIDENT_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

pub(super) fn ensure_gauspuls_control_extensions(
    values: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    for value in values {
        if is_typed_integer_value(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &GAUSPULS_INTEGER_CONTROL_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        if is_logical_value(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &GAUSPULS_LOGICAL_CONTROL_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        if is_single_value(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &GAUSPULS_SINGLE_CONTROL_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        if matches!(value, Value::GpuTensor(_)) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &GAUSPULS_RESIDENT_INPUT_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
    }
    Ok(())
}

pub(super) fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

pub(super) fn is_logical_value(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

pub(super) fn is_single_value(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_precision(handle) == Some(runmat_accelerate_api::ProviderPrecision::F32))
}
