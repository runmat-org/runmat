pub(super) mod contract;
mod restore;
pub(super) use restore::publish_host_result;
#[cfg(test)]
mod owning_tests;

use super::super::domain_probe::{probe_gpu_lower_bound, GpuLowerBoundResult};
use super::operation::LogarithmOperation;
use crate::builtins::common::gpu_helpers;
use crate::BuiltinResult;
use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage};
use runmat_value::Value;

pub(super) async fn mapped_batch_equivalent(
    operation: LogarithmOperation,
    input: &Value,
) -> BuiltinResult<bool> {
    let Value::GpuTensor(handle) = input else {
        return Ok(true);
    };
    // These extension/class rules belong to canonical scalar admission, not
    // the real floating range probe. Do not interpret integer/logical storage
    // as real floating evidence merely because its physical encoding is float.
    if runmat_accelerate_api::handle_integer_type(handle).is_some()
        || runmat_accelerate_api::handle_is_logical(handle)
    {
        return Ok(false);
    }
    if !runmat_accelerate_api::handle_is_explicit(handle)
        || runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::ComplexInterleaved
        || (operation.explicit_complex_extension().is_some()
            && crate::compatibility::runmat_extensions_enabled())
    {
        return Ok(true);
    }
    let provider = gpu_helpers::exact_provider_for_handle(handle)
        .ok_or_else(|| super::errors::missing_provider(operation))?;
    let result = probe_gpu_lower_bound(provider, handle, operation.real_boundary())
        .await
        .map_err(|source| {
            let mut error = super::errors::internal(operation, source.to_string());
            error.source = Some(Box::new(source));
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })?;
    Ok(matches!(result, GpuLowerBoundResult::AtOrAbove))
}

pub(super) async fn evaluate(
    operation: LogarithmOperation,
    handle: GpuTensorHandle,
) -> BuiltinResult<Value> {
    let provider = gpu_helpers::require_resident_compute_owner(&handle, || {
        super::errors::missing_provider(operation)
    })?;
    gpu_helpers::expected_handle_numeric_element_type(&handle).map_err(|_| {
        super::errors::internal(
            operation,
            "GPU input class metadata contradicts its physical storage",
        )
    })?;
    let _residency = gpu_helpers::ResidencyRestoreGuard::new(&handle);
    if requires_host_input(&handle) {
        return restore::fallback(operation, provider, &handle).await;
    }

    match probe_gpu_lower_bound(provider, &handle, operation.real_boundary())
        .await
        .map_err(|error| {
            let mut failure = super::errors::internal(operation, error.to_string());
            failure.source = Some(Box::new(error));
            failure.gpu_gather_retry = crate::GpuGatherRetry::Never;
            failure
        })? {
        GpuLowerBoundResult::Below => {
            ensure_explicit_complex_extension(operation, &handle)?;
            restore::fallback(operation, provider, &handle).await
        }
        GpuLowerBoundResult::AtOrAbove => {
            let result = {
                let _metadata = gpu_helpers::ResidencyRestoreGuard::new(&handle);
                operation.evaluate_provider(provider, &handle).await
            };
            match result {
                Ok(output) => contract::validate(operation, provider, &handle, output),
                Err(error) if gpu_helpers::provider_hook_allows_host_fallback(&error) => {
                    restore::fallback(operation, provider, &handle).await
                }
                Err(error) => {
                    let mut failure = super::errors::internal(
                        operation,
                        format!("provider unary operation failed: {error}"),
                    );
                    failure =
                        crate::runtime_error::with_provider_execution_source(failure, error.into());
                    failure.gpu_gather_retry = crate::GpuGatherRetry::Never;
                    Err(failure)
                }
            }
        }
        GpuLowerBoundResult::Unknown => restore::fallback(operation, provider, &handle).await,
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
