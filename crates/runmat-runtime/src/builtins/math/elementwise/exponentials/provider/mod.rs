mod contract;
mod restore;

use runmat_accelerate_api::{
    GpuHandleProvenance, GpuTensorHandle, GpuTensorStorage, ProviderPrecision,
};
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
    let metadata = gpu_helpers::snapshot_handle_metadata(&handle);
    let provenance =
        runmat_accelerate_api::handle_provenance(&handle).unwrap_or(GpuHandleProvenance::Automatic);

    if runmat_accelerate_api::handle_integer_type(&handle).is_some() {
        let gathered = gpu_helpers::gather_tensor_async(&handle).await;
        gpu_helpers::restore_handle_metadata(&handle, &metadata);
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
        gpu_helpers::restore_handle_metadata(&handle, &metadata);
        let value =
            gathered.map_err(|flow| map_control_flow_with_builtin(flow, operation.name()))?;
        let output = super::host::evaluate(operation, value)?;
        return restore::value(operation, provider, &handle, output, provenance);
    }

    let provider_result = match operation {
        ExponentialOperation::Exp => provider.unary_exp(&handle).await,
        ExponentialOperation::Expm1 => provider.unary_expm1(&handle).await,
    };
    gpu_helpers::restore_handle_metadata(&handle, &metadata);
    match provider_result {
        Ok(mut output) if valid_real(&output, &handle, provider, contract::precision(&handle)) => {
            runmat_accelerate_api::set_handle_provenance(&mut output, provenance);
            return Ok(gpu_helpers::resident_gpu_value(output));
        }
        Ok(output) => {
            gpu_helpers::free_unprotected_exact_owner(&output, &[&handle]);
            return Err(super::errors::internal(
                operation,
                "provider returned malformed unary output",
            ));
        }
        Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {}
        Err(error) => {
            return Err(super::errors::internal(
                operation,
                format!("provider unary operation failed: {error}"),
            ));
        }
    }

    let gathered = gpu_helpers::gather_tensor_async(&handle).await;
    gpu_helpers::restore_handle_metadata(&handle, &metadata);
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
