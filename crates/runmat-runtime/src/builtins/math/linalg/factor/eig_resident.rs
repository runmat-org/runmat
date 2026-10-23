//! Exact contextual readback through the input's actual allocation owner.
use super::*;
pub(super) async fn gather_eig_resident_value_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    handle: &GpuTensorHandle,
) -> BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(handle)
        .ok_or_else(|| eig_internal_error("eig: no acceleration provider owns the input handle"))?;
    gpu_helpers::download_value_preserving_residency_in_context(
        runtime,
        provider,
        handle,
        "RunMat:eig:Allocation",
        "eig input transfer admission failed",
    )
    .await
    .map_err(with_eig_context)
}
