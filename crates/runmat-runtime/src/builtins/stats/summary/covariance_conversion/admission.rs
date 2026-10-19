use super::*;

pub(super) fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(
            value,
            Value::GpuTensor(handle)
                if runmat_accelerate_api::handle_integer_type(handle).is_some()
        )
}

pub(super) fn is_logical_value(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(
            value,
            Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle)
        )
}

pub(super) fn is_single_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32
    ) || matches!(
        value,
        Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_precision(handle)
                == Some(runmat_accelerate_api::ProviderPrecision::F32)
                && runmat_accelerate_api::handle_integer_type(handle).is_none()
                && !runmat_accelerate_api::handle_is_logical(handle)
    )
}

pub(super) fn ensure_conversion_extensions(
    kind: ConversionKind,
    value: &Value,
) -> BuiltinResult<()> {
    match kind {
        ConversionKind::Corrcov => {
            if is_typed_integer_value(value) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &CORRCOV_INTEGER_DATA_EXTENSION,
                    "corrcov",
                )?;
            }
            if is_logical_value(value) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &CORRCOV_LOGICAL_DATA_EXTENSION,
                    "corrcov",
                )?;
            }
        }
        ConversionKind::Cov2corr => {
            if is_typed_integer_value(value) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &COV2CORR_INTEGER_DATA_EXTENSION,
                    "cov2corr",
                )?;
            }
            if is_logical_value(value) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &COV2CORR_LOGICAL_DATA_EXTENSION,
                    "cov2corr",
                )?;
            }
            if is_single_value(value) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &COV2CORR_SINGLE_DATA_EXTENSION,
                    "cov2corr",
                )?;
            }
            if matches!(value, Value::GpuTensor(_)) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &COV2CORR_GPU_INPUT_EXTENSION,
                    "cov2corr",
                )?;
            }
        }
    }
    Ok(())
}

pub(super) fn conversion_error(name: &'static str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(name)
        .with_identifier(format!("RunMat:{name}:InvalidArgument"))
        .build()
}

pub(super) fn internal_error(name: &'static str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(name)
        .with_identifier(format!("RunMat:{name}:Internal"))
        .build()
}
