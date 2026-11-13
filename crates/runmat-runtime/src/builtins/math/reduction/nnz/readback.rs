//! The declared host count result, using the canonical exact download owner.
use super::super::super::numerical_execution::NumericalCall;
use super::super::*;
use crate::context::host_allocation::{HostAllocationPolicy, HostAllocationReceipt};
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
pub(super) async fn download(
    provider: &dyn AccelProvider,
    result: &GpuTensorHandle,
    call: &NumericalCall<'_>,
) -> BuiltinResult<Value> {
    call.check()?;
    let value = match HostAllocationReceipt::policy(call.runtime).map_err(|cause| {
        super::super::super::numerical_execution::control(call.kind, call.runtime, cause)
    })? {
        HostAllocationPolicy::Unbounded => {
            gpu_helpers::download_value_preserving_residency_async(provider, result).await?
        }
        HostAllocationPolicy::Managed => {
            let runtime = call
                .runtime
                .ok_or_else(|| super::failure("selected count readback context is absent"))?;
            let quote = gpu_helpers::quote_exact_native_download(provider, result)?;
            let plan = quote.single_source_plan().map_err(|cause| {
                super::super::super::numerical_execution::control(call.kind, call.runtime, cause)
            })?;
            let receipt = HostAllocationReceipt::admit(Some(runtime), plan)
                .map_err(|cause| {
                    super::super::super::numerical_execution::control(
                        call.kind,
                        call.runtime,
                        cause,
                    )
                })?
                .ok_or_else(|| super::failure("count readback allocation policy changed"))?;
            gpu_helpers::download_exact_native(runtime, provider, result, &receipt, quote)
                .await
                .map_err(|failure| failure.error)?
        }
    };
    call.check()?;
    Ok(value)
}
