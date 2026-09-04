use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};
use runmat_builtins::{REALSQRT_ERROR_DOMAIN, REALSQRT_ERROR_INVALID_INPUT};
use runmat_value::Value;

use crate::builtins::common::gpu_helpers;
use crate::builtins::math::elementwise::domain_probe::{
    probe_gpu_lower_bound, GpuLowerBoundResult,
};
use crate::BuiltinResult;

use super::{errors, host, BUILTIN_NAME};

pub(super) async fn evaluate(input: GpuTensorHandle) -> BuiltinResult<Value> {
    reject_unsupported_storage(&input)?;
    let provider = gpu_helpers::exact_provider_for_handle(&input)
        .ok_or_else(|| errors::internal("resident input has no exact owner"))?;
    match probe_gpu_lower_bound(provider, &input, 0.0).await {
        Ok(GpuLowerBoundResult::Below) => Err(errors::with_detail(
            &REALSQRT_ERROR_DOMAIN,
            "gpuArray contains negative values",
        )),
        Ok(GpuLowerBoundResult::AtOrAbove) => match provider.unary_sqrt(&input).await {
            Ok(output) => super::super::super::unary_provider::validate_real_unary_output(
                provider,
                &input,
                output,
                BUILTIN_NAME,
            ),
            Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {
                fallback(&input, provider).await
            }
            Err(error) => Err(errors::internal(format!(
                "provider unary_sqrt failed: {error}"
            ))),
        },
        Ok(GpuLowerBoundResult::Unknown) => fallback(&input, provider).await,
        Err(error) => Err(errors::internal(error)),
    }
}

fn reject_unsupported_storage(input: &GpuTensorHandle) -> BuiltinResult<()> {
    if runmat_accelerate_api::handle_integer_type(input).is_some()
        || runmat_accelerate_api::handle_is_logical(input)
    {
        return Err(errors::with_detail(
            &REALSQRT_ERROR_INVALID_INPUT,
            "expected real single or double gpuArray input",
        ));
    }
    if runmat_accelerate_api::handle_storage(input) == GpuTensorStorage::ComplexInterleaved {
        return Err(errors::with_detail(
            &REALSQRT_ERROR_INVALID_INPUT,
            "complex gpuArray input is not supported",
        ));
    }
    Ok(())
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
            Value::Tensor(tensor) => host::evaluate_tensor(tensor),
            other => host::evaluate(other),
        },
    )
    .await
}
