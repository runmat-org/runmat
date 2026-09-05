pub(super) mod contract;
mod restore;

use super::super::domain_probe::{probe_gpu_lower_bound, GpuLowerBoundResult};
use super::operation::LogarithmOperation;
use crate::builtins::common::gpu_helpers;
use crate::BuiltinResult;
use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage};
use runmat_value::Value;

pub(super) async fn evaluate(
    operation: LogarithmOperation,
    handle: GpuTensorHandle,
) -> BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(&handle)
        .ok_or_else(|| super::errors::missing_provider(operation))?;
    gpu_helpers::expected_handle_numeric_element_type(&handle).map_err(|_| {
        super::errors::internal(
            operation,
            "GPU input class metadata contradicts its physical storage",
        )
    })?;
    let metadata = gpu_helpers::snapshot_handle_metadata(&handle);
    if requires_host_input(&handle) {
        return restore::fallback(operation, provider, &handle, &metadata).await;
    }

    match probe_gpu_lower_bound(provider, &handle, operation.real_boundary())
        .await
        .map_err(|error| super::errors::internal(operation, error.to_string()))?
    {
        GpuLowerBoundResult::Below => {
            ensure_explicit_complex_extension(operation, &handle)?;
            restore::fallback(operation, provider, &handle, &metadata).await
        }
        GpuLowerBoundResult::AtOrAbove => {
            let result = operation.evaluate_provider(provider, &handle).await;
            gpu_helpers::restore_handle_metadata(&handle, &metadata);
            match result {
                Ok(output) => contract::validate(operation, provider, &handle, output),
                Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {
                    restore::fallback(operation, provider, &handle, &metadata).await
                }
                Err(error) => Err(super::errors::internal(
                    operation,
                    format!("provider unary operation failed: {error}"),
                )),
            }
        }
        GpuLowerBoundResult::Unknown => {
            restore::fallback(operation, provider, &handle, &metadata).await
        }
    }
}

fn requires_host_input(handle: &GpuTensorHandle) -> bool {
    runmat_accelerate_api::handle_integer_type(handle).is_some()
        || runmat_accelerate_api::handle_is_logical(handle)
        || runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::ComplexInterleaved
}

pub(super) fn ensure_explicit_complex_extension(
    operation: LogarithmOperation,
    handle: &GpuTensorHandle,
) -> BuiltinResult<()> {
    if runmat_accelerate_api::handle_is_explicit(handle) {
        if runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::ComplexInterleaved {
            return Ok(());
        }
        let Some(extension) = operation.explicit_complex_extension() else {
            return Err(super::errors::explicit_complex_unsupported(operation));
        };
        crate::compatibility::ensure_builtin_extension_enabled(extension, operation.name())?;
    }
    Ok(())
}
