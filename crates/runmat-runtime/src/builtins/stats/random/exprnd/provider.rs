use super::*;
pub(super) fn try_gpu_exponential(mu: f64, shape: &[usize]) -> crate::BuiltinResult<Option<Value>> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Ok(None);
    };
    if provider.precision() != runmat_accelerate_api::ProviderPrecision::F64 {
        return Ok(None);
    }
    match provider.random_exponential(mu, shape) {
        Ok(handle) if valid_new_exprnd_gpu_output(&handle, provider, shape) => {
            let len = checked_element_count(shape)?;
            if let Err(error) = random::skip_uniform(len, "exprnd") {
                free_exprnd_gpu_output_without_input(&handle);
                return Err(error);
            }
            Ok(Some(Value::GpuTensor(handle)))
        }
        Ok(handle) => {
            free_exprnd_gpu_output_without_input(&handle);
            Err(exprnd_internal_error(
                "exprnd: provider random_exponential returned malformed output",
            ))
        }
        Err(err) if err.to_string().contains("random_exponential not supported") => Ok(None),
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

pub(super) fn free_exprnd_gpu_output(output: &GpuTensorHandle, source: &GpuTensorHandle) {
    if output.device_id == source.device_id && output.buffer_id == source.buffer_id {
        return;
    }
    free_exprnd_gpu_output_without_input(output);
}

pub(super) fn free_exprnd_gpu_output_without_input(output: &GpuTensorHandle) {
    if let Some(owner) = runmat_accelerate_api::provider_for_handle(output)
        .filter(|owner| owner.device_id() == output.device_id)
    {
        let _ = owner.free(output);
    }
}
