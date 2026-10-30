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

// A private operation adapter, not a second payload or provider registry. Both
// authorities use exactly the same query, stage receipt and adoption owner.
enum DownloadSource<'a, 'h> {
    Provider(&'a dyn AccelProvider, &'h GpuTensorHandle),
    Readback(
        &'a runmat_accelerate_api::NumericReadback<'h>,
        super::super::NumericDownloadDiagnostic,
    ),
}
impl<'h> DownloadSource<'_, 'h> {
    fn handle(&self) -> &'h GpuTensorHandle {
        match self {
            Self::Provider(_, handle) => handle,
            Self::Readback(readback, _) => readback.source(),
        }
    }
    fn quote(&self) -> Result<ExactNativeDownloadQuote, crate::RuntimeError> {
        match self {
            Self::Provider(provider, handle) => quote_exact_native_download(*provider, handle),
            Self::Readback(readback, _) => super::quote::quote_exact_native_readback(readback),
        }
    }
    fn prepare(
        &self,
        query: &ProviderFeasibilityQuery,
        claim: std::sync::Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>,
    ) -> anyhow::Result<runmat_accelerate_api::PreparedNumericDownloadStage> {
        match self {
            Self::Provider(provider, handle) => {
                provider.prepare_numeric_download_stage(handle, query, claim)
            }
            Self::Readback(readback, _) => readback.prepare_numeric_download_stage(query, claim),
        }
    }
    async fn download(
        &self,
        stage: runmat_accelerate_api::PreparedNumericDownloadStage,
    ) -> crate::BuiltinResult<HostNumericTensorOwned> {
        match self {
            Self::Provider(provider, handle) => {
                super::super::download_numeric_preserving_residency_staged_async(
                    *provider, handle, stage,
                )
                .await
            }
            Self::Readback(readback, diagnostic) => {
                super::super::download_numeric_readback_preserving_residency_async(
                    readback,
                    Some(stage),
                    *diagnostic,
                )
                .await
            }
        }
    }
}

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
    download_from_source(
        runtime,
        DownloadSource::Provider(provider, handle),
        receipt,
        quote,
    )
    .await
}

pub async fn download_exact_native_readback(
    runtime: &RuntimeContext,
    readback: &runmat_accelerate_api::NumericReadback<'_>,
    receipt: &HostAllocationReceipt,
    quote: ExactNativeDownloadQuote,
) -> Result<Value, ExactDownloadFailure> {
    download_exact_native_readback_with_diagnostics(
        runtime,
        readback,
        receipt,
        quote,
        super::super::NumericDownloadDiagnostic::Gpu,
    )
    .await
}

pub(crate) async fn download_exact_native_readback_with_diagnostics(
    runtime: &RuntimeContext,
    readback: &runmat_accelerate_api::NumericReadback<'_>,
    receipt: &HostAllocationReceipt,
    quote: ExactNativeDownloadQuote,
    diagnostic: super::super::NumericDownloadDiagnostic,
) -> Result<Value, ExactDownloadFailure> {
    download_from_source(
        runtime,
        DownloadSource::Readback(readback, diagnostic),
        receipt,
        quote,
    )
    .await
}

async fn download_from_source(
    runtime: &RuntimeContext,
    source: DownloadSource<'_, '_>,
    receipt: &HostAllocationReceipt,
    quote: ExactNativeDownloadQuote,
) -> Result<Value, ExactDownloadFailure> {
    let handle = source.handle();
    let current_quote = source.quote().map_err(ExactDownloadFailure::before)?;
    if current_quote != quote {
        return Err(ExactDownloadFailure::before(invalid(
            "selected host source geometry changed",
        )));
    }
    let host = staged_numeric_download(runtime, &source, receipt, quote.callback_bytes).await?;
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
    source: &DownloadSource<'_, '_>,
    receipt: &HostAllocationReceipt,
    callback_bytes: u64,
) -> Result<HostNumericTensorOwned, ExactDownloadFailure> {
    let handle = source.handle();
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
    let stage = source.prepare(&query, claim).map_err(|source| {
        let certified_before = runmat_accelerate_api::is_unsupported_provider_operation(&source);
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
    let host = source
        .download(stage)
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
