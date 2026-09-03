use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};

use crate::builtins::common::{broadcast::broadcast_shapes, gpu_helpers};

pub(super) fn valid_binary(
    output: &GpuTensorHandle,
    lhs: &GpuTensorHandle,
    rhs: &GpuTensorHandle,
    owner: &'static dyn AccelProvider,
    builtin: &str,
) -> bool {
    broadcast_shapes(builtin, &lhs.shape, &rhs.shape)
        .ok()
        .as_deref()
        == Some(output.shape.as_slice())
        && valid(output, lhs, owner)
        && !same_handle(output, rhs)
}

pub(super) fn valid(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    owner: &'static dyn AccelProvider,
) -> bool {
    output.device_id == input.device_id
        && !same_handle(output, input)
        && runmat_accelerate_api::handle_storage(output) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_precision(output)
            == runmat_accelerate_api::handle_precision(input)
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && gpu_helpers::exact_provider_for_handle(output)
            .is_some_and(|output_owner| std::ptr::eq(output_owner, owner))
}

pub(super) fn annotate<'a>(
    output: &mut GpuTensorHandle,
    inputs: impl IntoIterator<Item = &'a GpuTensorHandle>,
) {
    runmat_accelerate_api::set_handle_logical(output, true);
    let provenance = inputs
        .into_iter()
        .filter_map(runmat_accelerate_api::handle_provenance)
        .find(|provenance| *provenance == runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic);
    runmat_accelerate_api::set_handle_provenance(output, provenance);
    runmat_accelerate_api::mark_residency(output);
}

pub(super) fn free_rejected(output: &GpuTensorHandle, protected: &[&GpuTensorHandle]) {
    if protected.iter().any(|handle| same_handle(output, handle)) {
        return;
    }
    if let Some(owner) = gpu_helpers::exact_provider_for_handle(output) {
        let _ = owner.free(output);
    }
}

fn same_handle(lhs: &GpuTensorHandle, rhs: &GpuTensorHandle) -> bool {
    lhs.device_id == rhs.device_id && lhs.buffer_id == rhs.buffer_id
}
