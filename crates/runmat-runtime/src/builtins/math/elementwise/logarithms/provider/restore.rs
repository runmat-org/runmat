use super::super::operation::LogarithmOperation;
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin};
use crate::BuiltinResult;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_value::Value;

pub(super) async fn fallback(
    operation: LogarithmOperation,
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> BuiltinResult<Value> {
    let gathered = {
        let _metadata = gpu_helpers::ResidencyRestoreGuard::new(handle);
        gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone())).await
    };
    let gathered =
        gathered.map_err(|flow| map_control_flow_with_builtin(flow, operation.name()))?;
    let result = super::super::host::evaluate(operation, gathered)?;
    super::publish_host_result(operation, provider, handle, result)
}

pub(in crate::builtins::math::elementwise::logarithms) fn publish_host_result(
    operation: LogarithmOperation,
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
    result: Value,
) -> BuiltinResult<Value> {
    if matches!(result, Value::Complex(_, _) | Value::ComplexTensor(_)) {
        super::ensure_explicit_complex_extension(operation, handle)?;
    }
    let domain = if matches!(result, Value::Complex(_, _) | Value::ComplexTensor(_)) {
        runmat_types::NumericDomain::Complex
    } else {
        runmat_types::NumericDomain::Real
    };
    if operation
        .fallback_publication()
        .publishes_host(handle, domain)
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
    if !runmat_accelerate_api::same_provider_instance(provider, exact_owner) {
        return Err(super::super::errors::internal(
            operation,
            "GPU input owner changed during fallback",
        ));
    }
    gpu_helpers::restore_class_preserving_value(handle, result, operation.name())
}
