use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};
use runmat_value::Value;

use crate::builtins::common::gpu_helpers;
use crate::builtins::math::elementwise::domain_probe::{
    probe_gpu_lower_bound, GpuLowerBoundResult,
};
use crate::BuiltinResult;

use super::{errors, host, BUILTIN_NAME};

pub(super) async fn evaluate(input: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = gpu_helpers::require_resident_compute_owner(&input, || {
        errors::internal("resident input has no exact owner")
    })?;
    let requires_host = runmat_accelerate_api::handle_integer_type(&input).is_some()
        || runmat_accelerate_api::handle_is_logical(&input)
        || runmat_accelerate_api::handle_storage(&input) == GpuTensorStorage::ComplexInterleaved;
    if requires_host {
        return fallback(&input, provider).await;
    }
    match probe_gpu_lower_bound(provider, &input, 0.0).await {
        Ok(GpuLowerBoundResult::Below | GpuLowerBoundResult::Unknown) => {
            fallback(&input, provider).await
        }
        Ok(GpuLowerBoundResult::AtOrAbove) => {
            let result = {
                let _metadata = gpu_helpers::ResidencyRestoreGuard::new(&input);
                provider.unary_sqrt(&input).await
            };
            match result {
                Ok(output) => super::super::super::unary_provider::validate_real_unary_output(
                    provider,
                    &input,
                    output,
                    BUILTIN_NAME,
                ),
                Err(error) if gpu_helpers::provider_hook_allows_host_fallback(&error) => {
                    fallback(&input, provider).await
                }
                Err(error) => {
                    let mut failure =
                        errors::internal(format!("provider unary_sqrt failed: {error}"));
                    failure =
                        crate::runtime_error::with_provider_execution_source(failure, error.into());
                    failure.gpu_gather_retry = crate::GpuGatherRetry::Never;
                    Err(failure)
                }
            }
        }
        Err(error) => {
            let mut failure = errors::internal(&error);
            failure.source = Some(Box::new(error));
            failure.gpu_gather_retry = crate::GpuGatherRetry::Never;
            Err(failure)
        }
    }
}

async fn fallback(
    input: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
) -> BuiltinResult<Value> {
    super::super::super::unary_provider::gather_compute_restore(
        input,
        provider,
        BUILTIN_NAME,
        |gathered| match gathered {
            Value::ComplexTensor(tensor) => super::complex::evaluate_tensor(tensor),
            other => host::evaluate(other),
        },
    )
    .await
}
