//! Exact contextual readback through the input's actual allocation owner.
use super::*;
pub(super) async fn gather_eig_resident_value_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    handle: &GpuTensorHandle,
) -> BuiltinResult<Value> {
    let readback =
        runmat_accelerate_api::numeric_readback_for_handle(handle).map_err(|source| {
            crate::runtime_error::with_provider_execution_source(
                eig_internal_error("eig: no acceleration provider owns the input handle"),
                source.into(),
            )
        })?;
    gpu_helpers::download_readback_value_with_admission_in_context(
        runtime,
        &readback,
        "RunMat:eig:Allocation",
        "eig input transfer admission failed",
    )
    .await
    .map_err(|error| {
        let mut error = with_eig_context(error);
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        error
    })
}
