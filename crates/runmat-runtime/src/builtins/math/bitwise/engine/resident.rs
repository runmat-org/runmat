use super::*;

pub(super) fn is_single_bitwise_value(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor.numeric_dtype() == NumericDType::F32,
        Value::SparseTensor(sparse) => sparse.numeric_dtype() == Some(NumericDType::F32),
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_integer_type(handle).is_none()
                && !runmat_accelerate_api::handle_is_logical(handle)
                && runmat_accelerate_api::handle_precision(handle)
                    == Some(runmat_accelerate_api::ProviderPrecision::F32)
        }
        _ => false,
    }
}

pub(super) fn is_logical_bitwise_value(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

pub(super) fn bitwise_integer_class(value: &Value) -> Option<IntegerClass> {
    match value {
        Value::Int(value) => Some(value.integer_class()),
        Value::Tensor(tensor) => tensor.integer_storage().map(IntegerStorage::integer_class),
        Value::SparseTensor(sparse) => sparse.integer_storage().map(IntegerStorage::integer_class),
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_integer_class(handle),
        _ => None,
    }
}

pub(super) fn bitwise_result_as_logical(name: &'static str, value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Num(value) => Ok(Value::Bool(value != 0.0)),
        Value::Tensor(tensor) => LogicalArray::new(
            tensor
                .materialize_f64()
                .into_iter()
                .map(|value| u8::from(value != 0.0))
                .collect(),
            tensor.shape,
        )
        .map(Value::LogicalArray)
        .map_err(|error| error_with_detail(name, &ERROR_INVALID_INPUT, error)),
        other => Err(error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            format!("internal logical result had unsupported value {other:?}"),
        )),
    }
}

pub(super) fn restore_binary_bitwise_gpu_result(
    name: &'static str,
    value: Value,
    source: Option<&runmat_accelerate_api::GpuTensorHandle>,
) -> BuiltinResult<Value> {
    crate::builtins::common::resident_output::restore_to_source_provider(value, source)
        .map_err(|error| error_with_detail(name, &ERROR_INVALID_INPUT, error))
}
