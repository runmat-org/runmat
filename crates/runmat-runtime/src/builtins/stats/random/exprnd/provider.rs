use super::*;
pub(super) fn try_gpu_exponential(
    mu: f64,
    shape: &[usize],
    borrowed: &[GpuTensorHandle],
) -> crate::BuiltinResult<Option<Value>> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Ok(None);
    };
    if provider.precision() != runmat_accelerate_api::ProviderPrecision::F64 {
        return Ok(None);
    }
    let protected: Vec<_> = borrowed.iter().collect();
    match provider.random_exponential(mu, shape) {
        Ok(handle)
            if !borrowed
                .iter()
                .any(|input| gpu_helpers::same_gpu_handle(&handle, input))
                && valid_new_exprnd_gpu_output(&handle, provider, shape) =>
        {
            let len = checked_element_count(shape)?;
            if let Err(error) = random::skip_uniform(len, "exprnd") {
                gpu_helpers::free_rejected_provider_output(&handle, &protected, provider);
                return Err(error);
            }
            Ok(Some(Value::GpuTensor(handle)))
        }
        Ok(handle) => {
            gpu_helpers::free_rejected_provider_output(&handle, &protected, provider);
            Err(exprnd_internal_error(
                "exprnd: provider random_exponential returned malformed output",
            ))
        }
        Err(err) if gpu_helpers::provider_hook_is_unsupported(&err) => Ok(None),
        Err(err) => Err(exprnd_internal_error(format!(
            "exprnd: provider random_exponential failed: {err}"
        ))),
    }
}

pub(super) fn valid_new_exprnd_gpu_output(
    output: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    shape: &[usize],
) -> bool {
    output.shape == shape
        && output.device_id == provider.device_id()
        && runmat_accelerate_api::handle_precision(output) == Some(ProviderPrecision::F64)
        && runmat_accelerate_api::handle_storage(output) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && !runmat_accelerate_api::handle_is_logical(output)
        && runmat_accelerate_api::provider_for_handle(output)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
}

pub(super) fn valid_exprnd_gpu_output(
    output: &GpuTensorHandle,
    source: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    shape: &[usize],
    precision: ProviderPrecision,
) -> bool {
    output.shape == shape
        && output.device_id == source.device_id
        && output.buffer_id != source.buffer_id
        && runmat_accelerate_api::handle_precision(output) == Some(precision)
        && runmat_accelerate_api::handle_storage(output) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && !runmat_accelerate_api::handle_is_logical(output)
        && runmat_accelerate_api::provider_for_handle(output)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
}
