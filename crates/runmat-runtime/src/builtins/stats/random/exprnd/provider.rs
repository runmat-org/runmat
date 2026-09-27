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
    match provider.random_exponential(mu, shape) {
        Ok(handle)
            if !borrowed
                .iter()
                .any(|input| handle.shares_allocation_with(input))
                && valid_new_exprnd_gpu_output(&handle, provider, shape) =>
        {
            let len = checked_element_count(shape)?;
            if let Err(error) = random::skip_uniform(len, "exprnd") {
                return Err(gpu_helpers::reject_gpu_output(handle, error));
            }
            Ok(Some(Value::GpuTensor(handle)))
        }
        Ok(handle) => Err(gpu_helpers::reject_gpu_output(
            handle,
            exprnd_internal_error("exprnd: provider random_exponential returned malformed output"),
        )),
        Err(err) if gpu_helpers::provider_hook_is_unsupported(&err) => Ok(None),
        Err(err) => Err(build_runtime_error(format!(
            "exprnd: provider random_exponential failed: {err}"
        ))
        .with_builtin(BUILTIN_NAME)
        .with_identifier(
            EXPRND_ERROR_INTERNAL
                .identifier
                .expect("internal identifier"),
        )
        .with_source(runmat_accelerate_api::ProviderExecutionError::from(err))
        .build()),
    }
}

pub(super) fn valid_new_exprnd_gpu_output(
    output: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    shape: &[usize],
) -> bool {
    output.shape() == shape
        && output.device_id() == provider.device_id()
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
    output.shape() == shape
        && output.device_id() == source.device_id()
        && !output.shares_allocation_with(source)
        && runmat_accelerate_api::handle_precision(output) == Some(precision)
        && runmat_accelerate_api::handle_storage(output) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && !runmat_accelerate_api::handle_is_logical(output)
        && runmat_accelerate_api::provider_for_handle(output)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
}
