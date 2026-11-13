//! Owning-provider construction and explicit residency preservation.
use super::*;
use crate::builtins::array::creation::resident_constant::floating_constant;

pub(super) async fn nan_gpu(shape: &[usize], dtype: NumericDType) -> crate::BuiltinResult<Value> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Err(builtin_error(
            "nan: gpuArray output requires an active provider",
        ));
    };
    let precision = match dtype {
        NumericDType::F32 => ProviderPrecision::F32,
        NumericDType::F64 => ProviderPrecision::F64,
        _ => unreachable!("nan GPU output is floating"),
    };
    if provider.precision() != precision {
        return Err(builtin_error(
            "nan: active provider cannot preserve requested gpuArray precision",
        ));
    }
    floating_constant(provider, "nan", shape, f64::NAN, dtype, None).map(Value::GpuTensor)
}

#[async_recursion::async_recursion(?Send)]
pub(super) async fn nan_like_gpu(
    handle: &GpuTensorHandle,
    shape: &[usize],
) -> crate::BuiltinResult<Value> {
    if runmat_accelerate_api::handle_integer_type(handle).is_some() {
        return Err(nan_error(&NAN_ERROR_INTEGER_LIKE_PROTOTYPE));
    }
    if runmat_accelerate_api::handle_is_logical(handle) {
        return Err(builtin_error(
            "nan: 'like' prototype must be single or double",
        ));
    }
    let provider = gpu_helpers::exact_provider_for_handle(handle)
        .ok_or_else(|| builtin_error("nan: provider cannot preserve explicit gpuArray output"))?;
    let precision =
        runmat_accelerate_api::handle_precision(handle).unwrap_or_else(|| provider.precision());
    floating_constant(
        provider,
        "nan",
        shape,
        f64::NAN,
        dtype_from_precision(precision),
        Some(handle),
    )
    .map(Value::GpuTensor)
}
