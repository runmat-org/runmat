//! Explicit GPU zero construction with independent residency and element class.
use super::{builtin_error, GpuElementClass};
use crate::builtins::common::{gpu_helpers, random_args::validate_constructor_gpu_output, tensor};
use crate::BuiltinResult;
use runmat_accelerate_api::{GpuTensorStorage, IntegerElementType, ProviderPrecision};
use runmat_value::{NumericDType, Value};

pub(super) async fn build(shape: &[usize], element_class: GpuElementClass) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider()
        .ok_or_else(|| builtin_error("zeros: no acceleration provider is available"))?;
    let GpuElementClass::Numeric(dtype) = element_class else {
        return super::super::resident_constant::logical(provider, shape, false, &[], "zeros")
            .map(Value::GpuTensor);
    };
    let precision = match dtype {
        NumericDType::F32 => Some(ProviderPrecision::F32),
        NumericDType::F64 => Some(ProviderPrecision::F64),
        _ => None,
    };
    let handle = if precision == Some(provider.precision()) {
        match provider.zeros(shape) {
            Ok(handle) => handle,
            Err(cause)
                if crate::builtins::common::gpu_helpers::provider_hook_allows_host_fallback(
                    &cause,
                ) =>
            {
                upload_zeros(provider, shape, dtype)?
            }
            Err(cause) => {
                return Err(crate::build_runtime_error(format!(
                    "zeros: provider allocation failed: {cause}"
                ))
                .with_builtin("zeros")
                .with_source(runmat_accelerate_api::ProviderExecutionError::from(cause))
                .build())
            }
        }
    } else {
        upload_zeros(provider, shape, dtype)?
    };
    validate_constructor_gpu_output(
        "zeros",
        provider,
        handle,
        shape,
        GpuTensorStorage::Real,
        precision,
        dtype.integer_class().map(IntegerElementType::from),
        false,
    )
    .map(Value::GpuTensor)
}

fn upload_zeros(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    shape: &[usize],
    dtype: NumericDType,
) -> BuiltinResult<runmat_accelerate_api::GpuTensorHandle> {
    let host = tensor::zeros_with_dtype(shape, dtype)?;
    gpu_helpers::upload_tensor(provider, &host)
}

#[cfg(test)]
#[path = "gpu/tests.rs"]
mod tests;
