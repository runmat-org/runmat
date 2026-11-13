mod contract;
mod restore;
#[cfg(test)]
mod owning_tests;

use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage};
use runmat_value::Value;

use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::BuiltinResult;

use super::operation::ExponentialOperation;

pub(super) use contract::valid_real;

pub(super) async fn evaluate(
    operation: ExponentialOperation,
    handle: GpuTensorHandle,
) -> BuiltinResult<Value> {
    let provider = gpu_helpers::require_resident_compute_owner(&handle, || {
        super::errors::internal(operation, "GPU provider unavailable for input")
    })?;
    gpu_helpers::expected_handle_numeric_element_type(&handle).map_err(|_| {
        super::errors::internal(
            operation,
            "GPU input class metadata contradicts its physical storage",
        )
    })?;
    let _residency = gpu_helpers::ResidencyRestoreGuard::new(&handle);

    if runmat_accelerate_api::handle_integer_type(&handle).is_some()
        || runmat_accelerate_api::handle_is_logical(&handle)
    {
        // The logical/integer language class resolves to Double independently
        // of the provider's physical encoding or preferred floating precision.
        let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone()))
            .await
            .map_err(|flow| map_control_flow_with_builtin(flow, operation.name()))?;
        let tensor = tensor::value_into_tensor_for(operation.name(), gathered)
            .map_err(|error| super::errors::invalid(operation, error))?;
        let output = super::host::evaluate_tensor(operation, tensor)?;
        return gpu_helpers::restore_class_preserving_value(
            &handle,
            Value::Tensor(output),
            operation.name(),
        );
    }

    if runmat_accelerate_api::handle_storage(&handle) == GpuTensorStorage::ComplexInterleaved {
        let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone())).await;
        let value =
            gathered.map_err(|flow| map_control_flow_with_builtin(flow, operation.name()))?;
        let output = super::host::evaluate(operation, value)?;
        return restore::value(operation, provider, &handle, output);
    }

    let provider_result = match operation {
        ExponentialOperation::Exp => provider.unary_exp(&handle).await,
        ExponentialOperation::Expm1 => provider.unary_expm1(&handle).await,
    };
    match provider_result {
        Ok(output) if valid_real(&output, &handle, provider, contract::precision(&handle)) => {
            let output = gpu_helpers::propagate_owned_output_provenance(output, [&handle])?;
            return gpu_helpers::resident_gpu_value(output);
        }
        Ok(output) => {
            let mut error =
                super::errors::internal(operation, "provider returned malformed unary output");
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            return Err(gpu_helpers::reject_gpu_output(output, error));
        }
        Err(error) if gpu_helpers::provider_hook_allows_host_fallback(&error) => {}
        Err(error) => {
            let mut failure = super::errors::internal(
                operation,
                format!("provider unary operation failed: {error}"),
            );
            failure = crate::runtime_error::with_provider_execution_source(failure, error.into());
            failure.gpu_gather_retry = crate::GpuGatherRetry::Never;
            return Err(failure);
        }
    }

    let gathered = gpu_helpers::gather_tensor_async(&handle).await;
    let tensor = gathered.map_err(|flow| map_control_flow_with_builtin(flow, operation.name()))?;
    let output = super::host::evaluate_tensor(operation, tensor)?;
    restore::real(
        operation,
        provider,
        &handle,
        output,
        contract::precision(&handle),
    )
}
