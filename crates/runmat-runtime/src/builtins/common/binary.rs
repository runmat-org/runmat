use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_value::Value;

pub(crate) enum BinaryInputPlan<T> {
    Values(Box<(Value, Value)>),
    Structured(T),
}

pub(crate) fn matching_physical_inputs(left: &GpuTensorHandle, right: &GpuTensorHandle) -> bool {
    left.shape == right.shape
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
) -> Result<Value, String> {
    if !super::gpu_helpers::binary_gpu_output_matches(&output, left, right, provider, contract) {
        super::gpu_helpers::free_rejected_provider_output(&output, &[left, right], provider);
        return Err("provider returned a malformed binary output".to_string());
    }
    let mut output = output;
    let provenance = if runmat_accelerate_api::handle_is_explicit(left)
        || runmat_accelerate_api::handle_is_explicit(right)
    {
        runmat_accelerate_api::GpuHandleProvenance::Explicit
    } else {
        runmat_accelerate_api::GpuHandleProvenance::Automatic
    };
    runmat_accelerate_api::set_handle_provenance(&mut output, provenance);
    Ok(super::gpu_helpers::resident_gpu_value(output))
}
