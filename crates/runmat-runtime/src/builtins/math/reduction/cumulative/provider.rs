//! One no-attempt/error/result boundary for all eight scan hooks.
#[cfg(test)]
mod tests;
use crate::{
    builtins::common::gpu_helpers,
    context::{
        host_allocation::{
            HostAllocationCheckpoint, HostAllocationError, HostAllocationPolicy,
            HostAllocationReceipt,
        },
        RuntimeContext,
    },
    BuiltinResult,
};
use runmat_accelerate_api::{
    GpuTensorHandle, ProviderExecutionFuture, ProviderExtremaDemand, ProviderNanMode,
    ProviderScanDirection,
};
use runmat_builtins::{CumulativeKind, CumulativeMissingMode, CumulativeScanDirection};
mod result;
pub(in crate::builtins::math::reduction) use result::ResultValue;
async fn attempt<T>(
    future: ProviderExecutionFuture<'_, Option<T>>,
    kind: CumulativeKind,
) -> BuiltinResult<Option<T>> {
    future.await.map_err(|source| {
        crate::runtime_error::with_provider_execution_source(
            super::failure(kind, "provider cumulative execution failed"),
            source.with_unproven_effects(),
        )
    })
}
fn check(
    kind: CumulativeKind,
    runtime: Option<&RuntimeContext>,
    owner: Option<&HostAllocationCheckpoint>,
) -> BuiltinResult<()> {
    match (runtime, owner) {
        (Some(runtime), Some(owner)) => owner
            .check(runtime)
            .map_err(|e| super::admission::control(kind, Some(runtime), e)),
        _ => {
            if HostAllocationReceipt::policy(runtime)
                .map_err(|e| super::admission::control(kind, runtime, e))?
                == HostAllocationPolicy::Unbounded
            {
                Ok(())
            } else {
                Err(super::admission::control(
                    kind,
                    runtime,
                    HostAllocationError::StaleAuthority,
                ))
            }
        }
    }
}
pub(in crate::builtins::math::reduction) fn identity(
    kind: CumulativeKind,
    input: &GpuTensorHandle,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    admit(kind, input, runtime).map(|_| ())
}
fn admit(
    kind: CumulativeKind,
    input: &GpuTensorHandle,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<(
    &'static dyn runmat_accelerate_api::AccelProvider,
    Option<HostAllocationCheckpoint>,
)> {
    input.allocation_lease().validate_live().map_err(|source| {
        crate::build_runtime_error("cumulative input allocation is unavailable")
            .with_source(source)
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build()
    })?;
    let provider = gpu_helpers::require_resident_compute_owner(input, || {
        super::failure(kind, "resident input has no exact owning provider")
    })?;
    let owner = HostAllocationReceipt::capture_owner(runtime)
        .map_err(|e| super::admission::control(kind, runtime, e))?;
    check(kind, runtime, owner.as_ref())?;
    Ok((provider, owner))
}

pub(in crate::builtins::math::reduction) async fn execute(
    kind: CumulativeKind,
    input: &GpuTensorHandle,
    axis: usize,
    (direction, missing): (CumulativeScanDirection, CumulativeMissingMode),
    demand: ProviderExtremaDemand,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Option<ResultValue>> {
    let (provider, owner) = admit(kind, input, runtime)?;
    let direction = match direction {
        CumulativeScanDirection::Forward => ProviderScanDirection::Forward,
        CumulativeScanDirection::Reverse => ProviderScanDirection::Reverse,
    };
    let missing = match missing {
        CumulativeMissingMode::Include => ProviderNanMode::Include,
        CumulativeMissingMode::Omit => ProviderNanMode::Omit,
    };
    let integer = runmat_accelerate_api::handle_integer_type(input).is_some();
    let result = match (kind, integer) {
        (CumulativeKind::Sum, false) => {
            attempt(provider.cumsum_scan(input, axis, direction, missing), kind)
                .await?
                .map(ResultValue::Values)
        }
        (CumulativeKind::Sum, true) => {
            attempt(provider.integer_cumsum_scan(input, axis, direction), kind)
                .await?
                .map(ResultValue::Values)
        }
        (CumulativeKind::Product, false) => {
            attempt(provider.cumprod_scan(input, axis, direction, missing), kind)
                .await?
                .map(ResultValue::Values)
        }
        (CumulativeKind::Product, true) => {
            attempt(provider.integer_cumprod_scan(input, axis, direction), kind)
                .await?
                .map(ResultValue::Values)
        }
        (CumulativeKind::Minimum, false) => attempt(
            provider.cummin_scan(input, axis, direction, missing, demand),
            kind,
        )
        .await?
        .map(|r| ResultValue::Pair {
            values: r.values,
            indices: r.indices,
        }),
        (CumulativeKind::Minimum, true) => attempt(
            provider.integer_cummin_scan(input, axis, direction, demand),
            kind,
        )
        .await?
        .map(|r| ResultValue::Pair {
            values: r.values,
            indices: r.indices,
        }),
        (CumulativeKind::Maximum, false) => attempt(
            provider.cummax_scan(input, axis, direction, missing, demand),
            kind,
        )
        .await?
        .map(|r| ResultValue::Pair {
            values: r.values,
            indices: r.indices,
        }),
        (CumulativeKind::Maximum, true) => attempt(
            provider.integer_cummax_scan(input, axis, direction, demand),
            kind,
        )
        .await?
        .map(|r| ResultValue::Pair {
            values: r.values,
            indices: r.indices,
        }),
    };
    if let Err(error) = check(kind, runtime, owner.as_ref()) {
        return Err(match result {
            Some(result) => result.reject(error),
            None => error,
        });
    }
    let Some(result) = result else {
        return Ok(None);
    };
    result::validate(kind, input, provider, result, demand).map(Some)
}

pub(in crate::builtins::math::reduction) async fn fallback(
    kind: CumulativeKind,
    input: &GpuTensorHandle,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<runmat_value::Value> {
    // Allocation policy is not placement consent. No existing selected owner
    // authorizes these explicit-input scans to execute on the host.
    if runmat_accelerate_api::handle_is_explicit(input) {
        return Err(super::failure(
            kind,
            "the exact provider cannot execute this explicit-device scan form",
        ));
    }
    super::admission::host(kind, runtime)?;
    let provider = gpu_helpers::exact_provider_for_handle(input)
        .ok_or_else(|| super::failure(kind, "resident input has no exact owning provider"))?;
    gpu_helpers::download_value_preserving_residency_async(provider, input)
        .await
        .map_err(|mut e| {
            e.gpu_gather_retry = crate::GpuGatherRetry::Never;
            e
        })
}
