use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage, ProviderPrecision};

use crate::builtins::common::gpu_helpers;

pub(super) fn valid_complex_binary_output(
    precision: ProviderPrecision,
    output: &GpuTensorHandle,
    left: &GpuTensorHandle,
    right: &GpuTensorHandle,
    owner: &dyn AccelProvider,
    expected_shape: &[usize],
) -> bool {
    output.shape() == expected_shape
        && output.device_id() == left.device_id()
        && gpu_helpers::exact_provider_for_handle(output)
            .is_some_and(|actual| runmat_accelerate_api::same_provider_instance(actual, owner))
        && output.allocation_lease().validate_live().is_ok()
        && !output.shares_allocation_with(left)
        && !output.shares_allocation_with(right)
        && output.physical().storage() == GpuTensorStorage::ComplexInterleaved
        && output.descriptor().storage == Some(GpuTensorStorage::ComplexInterleaved)
        && output.physical().element_type().precision() == Some(precision)
        && output.descriptor().element_type == Some(output.physical().element_type())
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && !runmat_accelerate_api::handle_is_logical(output)
        && gpu_helpers::gpu_class_metadata_matches(output, Some(precision), None, false)
}

pub(in crate::builtins::math::elementwise::binary_arithmetic) fn valid_real_binary_output(
    output: &GpuTensorHandle,
    class_source: &GpuTensorHandle,
    other_source: Option<&GpuTensorHandle>,
    owner: &dyn AccelProvider,
    expected_shape: &[usize],
) -> bool {
    output.shape() == expected_shape
        && output.device_id() == class_source.device_id()
        && gpu_helpers::exact_provider_for_handle(output).is_some_and(|output_owner| {
            runmat_accelerate_api::same_provider_instance(output_owner, owner)
        })
        && !output.shares_allocation_with(class_source)
        && other_source.is_none_or(|source| !output.shares_allocation_with(source))
        && output.descriptor().storage == Some(GpuTensorStorage::Real)
        && gpu_helpers::expected_handle_numeric_element_type(output).is_ok()
        && gpu_helpers::gpu_class_metadata_matches(
            output,
            runmat_accelerate_api::handle_precision(class_source),
            runmat_accelerate_api::handle_integer_type(class_source),
            runmat_accelerate_api::handle_is_logical(class_source),
        )
        && runmat_accelerate_api::handle_integer_type(output)
            == runmat_accelerate_api::handle_integer_type(class_source)
        && runmat_accelerate_api::handle_is_logical(output)
            == runmat_accelerate_api::handle_is_logical(class_source)
        && runmat_accelerate_api::handle_precision(output)
            == runmat_accelerate_api::handle_precision(class_source)
}
