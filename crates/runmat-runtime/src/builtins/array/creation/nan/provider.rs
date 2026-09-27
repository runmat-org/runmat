//! Owning-provider construction and explicit residency preservation.
use super::*;
pub(super) async fn nan_gpu(shape: &[usize], dtype: NumericDType) -> crate::BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
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
        match provider.fill(shape, f64::NAN) {
            Ok(handle) => {
                if let Ok(handle) = validate_constructor_gpu_output(
                    "nan",
                    provider,
                    handle,
                    shape,
                    GpuTensorStorage::Real,
                    Some(precision),
                    None,
                    false,
                ) {
                    return Ok(Value::GpuTensor(handle));
                }
            }
            Err(err) => {
                log::debug!("nan_gpu: provider.fill failed ({err}); falling back to host upload");
            }
        }
        let host = nan_tensor(shape, dtype_from_precision(precision))?;
        if let Ok(gpu) = gpu_helpers::upload_tensor(provider, &host) {
            if let Ok(gpu) = validate_constructor_gpu_output(
                "nan",
                provider,
                gpu,
                shape,
                GpuTensorStorage::Real,
                Some(precision),
                None,
                false,
            ) {
                return Ok(Value::GpuTensor(gpu));
            }
        }
    }
    Err(builtin_error(
        "nan: gpuArray output requires an active provider",
    ))
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
    if let Some(provider) = gpu_helpers::exact_provider_for_handle(handle) {
        let precision =
            runmat_accelerate_api::handle_precision(handle).unwrap_or_else(|| provider.precision());
        let storage = runmat_accelerate_api::handle_storage(handle);
        if storage == GpuTensorStorage::ComplexInterleaved {
            let len = tensor::element_count(shape);
            let tensor = ComplexTensor::from_f64_values_with_dtype(
                vec![(f64::NAN, 0.0); len],
                shape.to_vec(),
                dtype_from_precision(precision),
            )
            .map_err(|e| builtin_error(format!("nan: {e}")))?;
            if let Ok(gpu) =
                gpu_helpers::upload_complex_tensor_protecting(provider, &tensor, &[handle])
            {
                if let Ok(gpu) =
                    crate::builtins::common::random_args::validate_constructor_gpu_output_protecting(
                        "nan",
                        provider,
                        gpu,
                        shape,
                        GpuTensorStorage::ComplexInterleaved,
                        Some(precision),
                        None,
                        false,
                        &[handle],
                    )
                {
                    return Ok(Value::GpuTensor(gpu));
                }
            }
            return Err(builtin_error(
                "nan: provider cannot preserve explicit complex gpuArray output",
            ));
        }
        let attempt = if handle.shape() == shape {
            provider.fill_like(handle, f64::NAN)
        } else {
            provider.fill(shape, f64::NAN)
        };
        if let Ok(gpu) = attempt {
            if let Ok(gpu) =
                crate::builtins::common::random_args::validate_constructor_gpu_output_protecting(
                    "nan",
                    provider,
                    gpu,
                    shape,
                    GpuTensorStorage::Real,
                    Some(precision),
                    None,
                    false,
                    &[handle],
                )
            {
                return Ok(Value::GpuTensor(gpu));
            }
        }

        let host = nan_tensor(shape, dtype_from_precision(precision))?;
        if let Ok(gpu) = gpu_helpers::upload_tensor_protecting(provider, &host, &[handle]) {
            if let Ok(gpu) =
                crate::builtins::common::random_args::validate_constructor_gpu_output_protecting(
                    "nan",
                    provider,
                    gpu,
                    shape,
                    GpuTensorStorage::Real,
                    Some(precision),
                    None,
                    false,
                    &[handle],
                )
            {
                return Ok(Value::GpuTensor(gpu));
            }
        }
    }

    Err(builtin_error(
        "nan: provider cannot preserve explicit gpuArray output",
    ))
}
