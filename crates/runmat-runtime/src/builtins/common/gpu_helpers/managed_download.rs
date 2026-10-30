//! Exact native provider download: one quote, one staged source, one retained value.

use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationError, HostAllocationPlan,
};
use crate::context::RuntimeContext;
use crate::{build_runtime_error, GpuGatherRetry, RuntimeError};
use runmat_accelerate_api::GpuTensorStorage;
use runmat_value::NumericDType;

#[path = "managed_download_adoption.rs"]
mod adoption;
#[path = "managed_download_quote.rs"]
mod quote;
#[path = "managed_download_stage.rs"]
mod stage;

pub use adoption::adopt_settled_native_download;
pub use quote::{quote_exact_native_download, quote_exact_native_readback};
pub(crate) use stage::download_exact_native_readback_with_diagnostics;
pub use stage::{download_exact_native, download_exact_native_readback};

/// Declared source-only host inspection, not a compute fallback. Reuses the
/// same finite readback quote, receipt, stage and final adoption as gather.
pub(crate) async fn download_readback_value_in_context(
    runtime: Option<&RuntimeContext>,
    readback: &runmat_accelerate_api::NumericReadback<'_>,
) -> Result<runmat_value::Value, RuntimeError> {
    download_readback_value_with_admission_in_context(
        runtime,
        readback,
        "RunMat:Placement:HostReadback",
        "native readback admission failed",
    )
    .await
}

pub(crate) async fn download_readback_value_with_admission_in_context(
    runtime: Option<&RuntimeContext>,
    readback: &runmat_accelerate_api::NumericReadback<'_>,
    admission_identifier: &'static str,
    admission_message: &'static str,
) -> Result<runmat_value::Value, RuntimeError> {
    use crate::context::host_allocation::{HostAllocationPolicy, HostAllocationReceipt};
    let allocation = |cause| {
        host_allocation_runtime_error(runtime, admission_identifier, admission_message, cause)
    };
    if HostAllocationReceipt::policy(runtime).map_err(allocation)?
        == HostAllocationPolicy::Unbounded
    {
        let host = super::download_numeric_readback_preserving_residency_async(
            readback,
            None,
            super::NumericDownloadDiagnostic::Gpu,
        )
        .await?;
        return super::value_from_numeric_download(
            host,
            runmat_accelerate_api::handle_is_logical(readback.source()),
        )
        .map_err(|cause| {
            super::provider_payload_mismatch(readback.source(), readback.source().shape(), cause)
        });
    }
    let runtime = runtime.ok_or_else(|| allocation(HostAllocationError::MissingManagedOwner))?;
    let quote = quote_exact_native_readback(readback)?;
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        quote.single_source_plan().map_err(allocation)?,
    )
    .map_err(allocation)?
    .ok_or_else(|| allocation(HostAllocationError::MissingManagedOwner))?;
    let budget = runtime.cpu_cooperation();
    budget
        .await_cancellable(download_exact_native_readback(
            runtime, readback, &receipt, quote,
        ))
        .await
        .map_err(|cause| cause.into_runtime_error_in_context(Some(runtime)))?
        .map_err(|failure| failure.error)
}

/// Download through the selected operation's authority. The returned native
/// carrier owns its backing independently of the temporary stage receipt.
pub(crate) async fn download_value_preserving_residency_in_context(
    runtime: Option<&RuntimeContext>,
    provider: &dyn runmat_accelerate_api::AccelProvider,
    input: &runmat_accelerate_api::GpuTensorHandle,
    admission_identifier: &'static str,
    admission_message: &'static str,
) -> Result<runmat_value::Value, RuntimeError> {
    use crate::context::host_allocation::{HostAllocationPolicy, HostAllocationReceipt};
    let allocation = |cause| {
        host_allocation_runtime_error(runtime, admission_identifier, admission_message, cause)
    };
    if HostAllocationReceipt::policy(runtime).map_err(allocation)?
        == HostAllocationPolicy::Unbounded
    {
        return super::download_value_preserving_residency_async(provider, input).await;
    }
    let runtime = runtime.ok_or_else(|| allocation(HostAllocationError::MissingManagedOwner))?;
    let mut budget = runtime.cpu_cooperation();
    budget
        .checkpoint(0)
        .await
        .map_err(|cause| cause.into_runtime_error_in_context(Some(runtime)))?;
    let quote = quote_exact_native_download(provider, input)?;
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        quote.single_source_plan().map_err(allocation)?,
    )
    .map_err(allocation)?
    .ok_or_else(|| allocation(HostAllocationError::MissingManagedOwner))?;
    let output = budget
        .await_cancellable(download_exact_native(
            runtime, provider, input, &receipt, quote,
        ))
        .await
        .map_err(|cause| cause.into_runtime_error_in_context(Some(runtime)))?
        .map_err(|failure| failure.error)?;
    receipt.checkpoint(runtime).map_err(allocation)?;
    Ok(output)
}

fn invalid(message: &'static str) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier("RunMat:Placement:InvalidHostReadback")
        .with_gpu_gather_retry(GpuGatherRetry::Never)
        .build()
}
fn admission_error(runtime: &RuntimeContext, cause: HostAllocationError) -> RuntimeError {
    host_allocation_runtime_error(
        Some(runtime),
        "RunMat:Placement:HostReadback",
        "selected numerical readback admission failed",
        cause,
    )
}
/// One source's checked physical geometry, including stage and final owner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExactNativeDownloadQuote {
    pub dtype: NumericDType,
    pub storage: GpuTensorStorage,
    pub logical: bool,
    pub host_workspace: u64,
    pub payload_bytes: u64,
    pub query_bytes: u64,
    pub shape_bytes: u64,
    pub callback_bytes: u64,
    pub retained_value_upper: u64,
    pub active_upper: u64,
    pub precharged_output_bytes: u64,
}
impl ExactNativeDownloadQuote {
    pub fn single_source_plan(self) -> Result<HostAllocationPlan, HostAllocationError> {
        let callback = HostAllocationPlan::retained_metadata_bytes(self.callback_bytes)?;
        HostAllocationPlan::new(
            self.active_upper,
            callback
                .checked_add(self.precharged_output_bytes)
                .ok_or(HostAllocationError::Overflow)?,
        )
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExactDownloadPhase {
    BeforeEffect,
    AfterEffect,
}
#[derive(Debug)]
pub struct ExactDownloadFailure {
    pub error: RuntimeError,
    pub phase: ExactDownloadPhase,
}
impl ExactDownloadFailure {
    fn before(error: RuntimeError) -> Self {
        Self {
            error,
            phase: ExactDownloadPhase::BeforeEffect,
        }
    }
    fn after(error: RuntimeError) -> Self {
        Self {
            error,
            phase: ExactDownloadPhase::AfterEffect,
        }
    }
}
