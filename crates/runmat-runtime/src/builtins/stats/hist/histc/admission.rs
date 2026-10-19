use super::*;
pub(super) fn ensure_histc_input_extensions(value: &Value) -> BuiltinResult<()> {
    if matches!(value, Value::GpuTensor(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(&HISTC_GPU_EXTENSION, NAME)?;
        return Ok(());
    }
    if matches!(value, Value::Bool(_) | Value::LogicalArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(&HISTC_LOGICAL_EXTENSION, NAME)?;
    }
    let wide = matches!(
        value,
        Value::Int(runmat_value::IntValue::I64(_) | runmat_value::IntValue::U64(_))
    ) || matches!(value, Value::Tensor(tensor) if matches!(tensor.numeric_dtype(), NumericDType::I64 | NumericDType::U64))
        || matches!(value, Value::ComplexTensor(tensor) if tensor.integer_storage().is_some_and(|storage| matches!(storage.real().numeric_dtype(), NumericDType::I64 | NumericDType::U64)));
    if wide {
        crate::compatibility::ensure_builtin_extension_enabled(
            &HISTC_WIDE_INTEGER_EXTENSION,
            NAME,
        )?;
    }
    Ok(())
}

pub(super) async fn value_to_real_tensor(value: Value) -> BuiltinResult<Tensor> {
    let host = match value {
        Value::GpuTensor(_) => gpu_helpers::gather_value_async(&value)
            .await
            .map_err(|err| internal_error(format!("histc: {err}")))?,
        other => other,
    };
    host_to_real_tensor(host)
}

pub(super) fn validate_dimension_argument(dim: usize) -> BuiltinResult<usize> {
    if dim == 0 || dim > MAX_DIMENSION_ARGUMENT {
        return Err(invalid_argument(format!(
            "histc: dim must be <= {MAX_DIMENSION_ARGUMENT}"
        )));
    }
    Ok(dim)
}

pub(super) fn host_to_real_tensor(value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::Complex(re, _) => {
            Tensor::new(vec![re], vec![1, 1]).map_err(|err| internal_error(format!("histc: {err}")))
        }
        Value::ComplexTensor(tensor) => complex_real_tensor(tensor),
        other => tensor::value_into_tensor_for(NAME, other).map_err(invalid_argument),
    }
}

pub(super) fn complex_real_tensor(tensor: ComplexTensor) -> BuiltinResult<Tensor> {
    let (storage, shape) = tensor.into_complex_storage_and_shape();
    let data = match storage {
        runmat_value::ComplexStorage::Integer(storage) => storage
            .real()
            .exact_values()
            .into_iter()
            .map(|value| value.to_f64())
            .collect(),
        storage => storage
            .materialize_f64()
            .into_iter()
            .map(|(re, _)| re)
            .collect(),
    };
    Tensor::new(data, shape).map_err(|err| internal_error(format!("histc: {err}")))
}
