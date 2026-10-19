use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_value::Value;

#[cfg(test)]
#[path = "binary/owning_tests.rs"]
mod owning_tests;

pub(crate) enum BinaryInputPlan<T> {
    Values(Box<(Value, Value)>),
    Structured(T),
}

pub(crate) fn matching_physical_inputs(left: &GpuTensorHandle, right: &GpuTensorHandle) -> bool {
    left.shape() == right.shape()
        && runmat_accelerate_api::handle_storage(left)
            == runmat_accelerate_api::handle_storage(right)
        && runmat_accelerate_api::handle_precision(left)
            == runmat_accelerate_api::handle_precision(right)
        && runmat_accelerate_api::handle_integer_type(left)
            == runmat_accelerate_api::handle_integer_type(right)
        && runmat_accelerate_api::handle_is_logical(left)
            == runmat_accelerate_api::handle_is_logical(right)
}

pub(crate) fn validate_resident_output(
    provider: &'static dyn AccelProvider,
    left: &GpuTensorHandle,
    right: &GpuTensorHandle,
    output: GpuTensorHandle,
    contract: &super::gpu_helpers::BinaryGpuOutputContract,
) -> crate::BuiltinResult<Value> {
    if !super::gpu_helpers::binary_gpu_output_matches(&output, left, right, provider, contract) {
        let mut error = crate::RuntimeError::new("provider returned a malformed binary output");
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        return Err(super::gpu_helpers::reject_gpu_output(output, error));
    }
    let output = super::gpu_helpers::propagate_owned_output_provenance(output, [left, right])?;
    super::gpu_helpers::resident_gpu_value(output)
}
