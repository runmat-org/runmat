mod contract;
mod restore;
#[cfg(test)]
mod owning_tests;

use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage, ProviderPrecision};
use runmat_value::Value;

use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::BuiltinResult;

use super::operation::ExponentialOperation;

pub(super) use contract::valid_real;

pub(super) async fn evaluate(
    operation: ExponentialOperation,
    handle: GpuTensorHandle,
) -> BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(&handle)
        .ok_or_else(|| super::errors::internal(operation, "GPU provider unavailable for input"))?;
    gpu_helpers::expected_handle_numeric_element_type(&handle).map_err(|_| {
        super::errors::internal(
            operation,
            "GPU input class metadata contradicts its physical storage",
        )
    })?;
    let _residency = gpu_helpers::ResidencyRestoreGuard::new(&handle);

    if runmat_accelerate_api::handle_integer_type(&handle).is_some() {
        let gathered = gpu_helpers::gather_tensor_async(&handle).await;
        let tensor =
            gathered.map_err(|flow| map_control_flow_with_builtin(flow, operation.name()))?;
        let output = super::host::evaluate_tensor(operation, tensor)?;
        if provider.precision() != ProviderPrecision::F64 {
            return Ok(tensor::tensor_into_value(output));
        }
        return restore::real(
            operation,
            provider,
            &handle,
            output,
            Some(ProviderPrecision::F64),
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
        Ok(mut output) if valid_real(&output, &handle, provider, contract::precision(&handle)) => {
            gpu_helpers::propagate_output_provenance(&mut output, &[&handle])?;
            return gpu_helpers::resident_gpu_value(output);
        }
        Ok(output) => {
            let mut error =
                super::errors::internal(operation, "provider returned malformed unary output");
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            return Err(gpu_helpers::reject_gpu_output(output, error));
        }
        Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {}
        Err(error) => {
            let mut failure = super::errors::internal(
                operation,
                format!("provider unary operation failed: {error}"),
            );
            failure.source = Some(error.into_boxed_dyn_error());
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
