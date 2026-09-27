use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};

use crate::builtins::common::{broadcast::broadcast_shapes, gpu_helpers};

pub(super) fn valid_binary(
    output: &GpuTensorHandle,
    lhs: &GpuTensorHandle,
    rhs: &GpuTensorHandle,
    owner: &'static dyn AccelProvider,
    builtin: &str,
) -> bool {
    broadcast_shapes(builtin, lhs.shape(), rhs.shape())
        .ok()
        .as_deref()
        == Some(output.shape())
        && valid(output, lhs, owner)
        && !same_handle(output, rhs)
}

pub(super) fn valid(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    owner: &'static dyn AccelProvider,
) -> bool {
    output.view().is_logical()
        && output.device_id() == input.device_id()
        && !same_handle(output, input)
        && runmat_accelerate_api::handle_storage(output) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_precision(output)
            == runmat_accelerate_api::handle_precision(input)
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && gpu_helpers::exact_provider_for_handle(output).is_some_and(|output_owner| {
            runmat_accelerate_api::same_provider_instance(output_owner, owner)
        })
}

pub(super) fn annotate<'a>(
    output: &mut GpuTensorHandle,
    inputs: impl IntoIterator<Item = &'a GpuTensorHandle>,
) -> crate::BuiltinResult<()> {
    gpu_helpers::propagate_output_provenance(output, inputs)
}

fn same_handle(lhs: &GpuTensorHandle, rhs: &GpuTensorHandle) -> bool {
    lhs.shares_allocation_with(rhs)
}
