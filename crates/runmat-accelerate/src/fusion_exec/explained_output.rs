//! Admission of the matrix pattern's real floating results.
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, NumericElementType, ProviderExecutionResult,
};
use runmat_runtime::builtins::common::gpu_helpers::exact_provider_for_handle;

pub(super) fn admit(
    output: GpuTensorHandle,
    provider: &'static dyn AccelProvider,
    shape: &[usize],
    element: NumericElementType,
    protected: &[&GpuTensorHandle],
) -> ProviderExecutionResult<GpuTensorHandle> {
    let valid = output.shape() == shape
        && output.physical().element_type() == element
        && output.physical().storage() == GpuTensorStorage::Real
        && !runmat_accelerate_api::handle_is_logical(&output)
        && output.device_id() == provider.device_id()
        && exact_provider_for_handle(&output)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
        && output.allocation_lease().validate_live().is_ok()
        && !protected
            .iter()
            .any(|input| input.shares_allocation_with(&output));
    if valid {
        Ok(output)
    } else {
        Err(super::outputs::reject_outputs(
            vec![output],
            "explained variance provider returned an invalid result",
        ))
    }
}
