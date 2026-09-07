use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};

use crate::builtins::common::gpu_helpers;

pub(in crate::builtins::math::elementwise::binary_arithmetic) fn valid_real_binary_output(
    output: &GpuTensorHandle,
    class_source: &GpuTensorHandle,
    other_source: Option<&GpuTensorHandle>,
    owner: &dyn AccelProvider,
    expected_shape: &[usize],
) -> bool {
    output.shape == expected_shape
        && output.device_id == class_source.device_id
        && gpu_helpers::exact_provider_for_handle(output)
            .is_some_and(|output_owner| std::ptr::eq(output_owner, owner))
        && !gpu_helpers::same_gpu_handle(output, class_source)
        && other_source.is_none_or(|source| !gpu_helpers::same_gpu_handle(output, source))
        && runmat_accelerate_api::handle_storage(output) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_integer_type(output)
            == runmat_accelerate_api::handle_integer_type(class_source)
        && runmat_accelerate_api::handle_is_logical(output)
            == runmat_accelerate_api::handle_is_logical(class_source)
        && runmat_accelerate_api::handle_precision(output)
            == runmat_accelerate_api::handle_precision(class_source)
}
