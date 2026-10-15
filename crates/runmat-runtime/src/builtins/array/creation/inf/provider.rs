//! Owning-provider construction and explicit residency preservation.
use super::*;
use crate::builtins::array::creation::resident_constant::floating_constant;

pub(super) async fn inf_gpu(shape: &[usize], dtype: NumericDType) -> crate::BuiltinResult<Value> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Err(builtin_error(
            "inf: gpuArray output requires an active provider",
        ));
    };
    let precision = match dtype {
        NumericDType::F32 => ProviderPrecision::F32,
        NumericDType::F64 => ProviderPrecision::F64,
        _ => unreachable!("inf GPU output is floating"),
    };
    if provider.precision() != precision {
        return Err(builtin_error(
            "inf: active provider cannot preserve requested gpuArray precision",
        ));
    }
    floating_constant(provider, "inf", shape, f64::INFINITY, dtype, None).map(Value::GpuTensor)
}

#[async_recursion::async_recursion(?Send)]
pub(super) async fn inf_like_gpu(
    handle: &GpuTensorHandle,
    shape: &[usize],
) -> crate::BuiltinResult<Value> {
    if runmat_accelerate_api::handle_integer_type(handle).is_some() {
        return Err(inf_error(&INF_ERROR_INTEGER_LIKE_PROTOTYPE));
    }
    if runmat_accelerate_api::handle_is_logical(handle) {
        return Err(builtin_error(
            "inf: 'like' prototype must be single or double",
        ));
    }
    let provider = gpu_helpers::exact_provider_for_handle(handle)
        .ok_or_else(|| builtin_error("inf: provider cannot preserve explicit gpuArray output"))?;
    let precision =
        runmat_accelerate_api::handle_precision(handle).unwrap_or_else(|| provider.precision());
    floating_constant(
        provider,
        "inf",
        shape,
        f64::INFINITY,
        dtype_from_precision(precision),
        Some(handle),
    )
    .map(Value::GpuTensor)
}
