use super::super::operation::LogarithmOperation;
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin};
use crate::BuiltinResult;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_value::Value;

pub(super) async fn fallback(
    operation: LogarithmOperation,
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
    metadata: &gpu_helpers::GpuHandleMetadataSnapshot,
) -> BuiltinResult<Value> {
    let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone())).await;
    gpu_helpers::restore_handle_metadata(handle, metadata);
    let gathered =
        gathered.map_err(|flow| map_control_flow_with_builtin(flow, operation.name()))?;
    let result = super::super::host::evaluate(operation, gathered)?;
    if matches!(result, Value::Complex(_, _) | Value::ComplexTensor(_)) {
        super::ensure_explicit_complex_extension(operation, handle)?;
    }
    if !runmat_accelerate_api::handle_is_explicit(handle)
        && !operation.restores_automatic_fallback()
    {
        return Ok(result);
    }
    restore(operation, provider, handle, result)
}

fn restore(
    operation: LogarithmOperation,
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
    result: Value,
) -> BuiltinResult<Value> {
    let exact_owner = gpu_helpers::exact_provider_for_handle(handle)
        .ok_or_else(|| super::super::errors::internal(operation, "GPU provider unavailable"))?;
    if !std::ptr::eq(provider, exact_owner) {
        return Err(super::super::errors::internal(
            operation,
            "GPU input owner changed during fallback",
        ));
    }
    gpu_helpers::restore_class_preserving_value(handle, result, operation.name())
        .map_err(|error| super::super::errors::internal(operation, error.message()))
}
