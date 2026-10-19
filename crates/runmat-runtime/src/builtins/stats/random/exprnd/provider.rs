use super::*;
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
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
}
