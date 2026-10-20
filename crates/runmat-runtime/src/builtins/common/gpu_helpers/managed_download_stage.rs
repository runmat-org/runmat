//! One staged provider operation with typed before/after-effect failure.
use super::{
    admission_error, adopt_settled_native_download, invalid, quote_exact_native_download,
    ExactDownloadFailure, ExactNativeDownloadQuote,
};
use crate::context::host_allocation::HostAllocationReceipt;
use crate::context::RuntimeContext;
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, HostNumericTensorOwned, ProviderFeasibilityQuery,
    ProviderQueryMetadataFootprint,
};
use runmat_value::Value;

/// One selected source through exactly one provider stage and native-buffer
/// adoption. A stage error is terminal after effect unless the provider
/// certifies typed pre-effect Unsupported. The returned Value retains its own
/// physical claims independently of the preparation receipt.
pub async fn download_exact_native(
    runtime: &RuntimeContext,
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
    receipt: &HostAllocationReceipt,
    quote: ExactNativeDownloadQuote,
) -> Result<Value, ExactDownloadFailure> {
    if quote_exact_native_download(provider, handle).ok() != Some(quote) {
        return Err(ExactDownloadFailure::before(invalid(
            "selected host source geometry changed",
        )));
    }
    let host =
        staged_numeric_download(runtime, provider, handle, receipt, quote.callback_bytes).await?;
    if host.shape != handle.shape() || host.storage != quote.storage {
        return Err(ExactDownloadFailure::after(invalid(
            "selected provider returned a different source geometry",
        )));
    }
    adopt_settled_native_download(host, quote, runtime, receipt)
        .map_err(ExactDownloadFailure::after)
}

async fn staged_numeric_download(
    runtime: &RuntimeContext,
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
    receipt: &HostAllocationReceipt,
    callback_bytes: u64,
) -> Result<HostNumericTensorOwned, ExactDownloadFailure> {
    receipt
        .checkpoint(runtime)
        .map_err(|source| ExactDownloadFailure::before(admission_error(runtime, source)))?;
    let claim = receipt
        .claim_retained_metadata(runtime, callback_bytes)
        .map_err(|source| ExactDownloadFailure::before(admission_error(runtime, source)))?;
    let query = ProviderFeasibilityQuery::exact_numeric_download(handle)
        .ok_or_else(|| ExactDownloadFailure::before(invalid("selected host query changed")))?;
    let rank = handle.shape().len();
    let footprint = ProviderQueryMetadataFootprint {
        operation: "transfer.download",
        input_ranks: &[rank],
        output_ranks: &[rank],
        preparation_shape_ranks: &[rank],
        preparation_representation_slots: 0,
    };
    if !footprint.matches(&query) || !query.matches_numeric_download(handle) {
        return Err(ExactDownloadFailure::before(invalid(
            "selected host query no longer matches its source",
        )));
    }
    let identity = handle.view_identity();
    let stage = provider
        .prepare_numeric_download_stage(handle, &query, claim)
        .map_err(|source| {
            let certified_before =
                runmat_accelerate_api::is_unsupported_provider_operation(&source);
            let error = crate::runtime_error::with_provider_execution_source(
                invalid("selected host provider staging failed"),
                runmat_accelerate_api::ProviderExecutionError::from(source),
            );
            if certified_before {
                ExactDownloadFailure::before(error)
            } else {
                ExactDownloadFailure::after(error)
            }
        })?;
    if !stage.matches_source(handle) {
        return Err(ExactDownloadFailure::after(invalid(
            "selected host provider staged a different source",
        )));
    }
    receipt
        .checkpoint(runtime)
        .map_err(|source| ExactDownloadFailure::after(admission_error(runtime, source)))?;
    let host =
        super::super::download_numeric_preserving_residency_staged_async(provider, handle, stage)
            .await
            .map_err(ExactDownloadFailure::after)?;
    receipt
        .checkpoint(runtime)
        .map_err(|source| ExactDownloadFailure::after(admission_error(runtime, source)))?;
    if handle.view_identity() != identity || !query.matches_numeric_download(handle) {
        return Err(ExactDownloadFailure::after(invalid(
            "selected host source changed during readback",
        )));
    }
    Ok(host)
}
