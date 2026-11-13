//! Shared selected-context and provider-error boundaries for numeric reductions.
use crate::{
    context::{
        host_allocation::{HostAllocationError, HostAllocationPolicy, HostAllocationReceipt},
        RuntimeContext,
    },
    BuiltinResult, RuntimeError,
};
use runmat_accelerate_api::{ProviderExecutionFuture, ProviderExecutionResult};
use runmat_builtins::NumericalReductionKind;
use runmat_value::Value;
mod call;
#[cfg(test)]
mod tests;
pub(super) use call::NumericalCall;

pub(super) fn terminal(mut error: RuntimeError) -> RuntimeError {
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}

pub(super) fn control(
    kind: NumericalReductionKind,
    runtime: Option<&RuntimeContext>,
    source: HostAllocationError,
) -> RuntimeError {
    let cancellation = runtime.map(RuntimeContext::cancellation_context);
    terminal(crate::builtin::builtin_control_error(
        cancellation.as_ref(),
        kind.catalog_entry().identity.name,
        source,
    ))
}

pub(super) fn host(
    kind: NumericalReductionKind,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    match HostAllocationReceipt::policy(runtime).map_err(|e| control(kind, runtime, e))? {
        HostAllocationPolicy::Unbounded => Ok(()),
        HostAllocationPolicy::Managed => Err(control(
            kind,
            runtime,
            HostAllocationError::MissingManagedOwner,
        )),
    }
}

pub(super) fn input_fallback<'a>(
    kind: NumericalReductionKind,
    runtime: Option<&RuntimeContext>,
    values: impl IntoIterator<Item = &'a Value>,
) -> BuiltinResult<()> {
    if values
        .into_iter()
        .any(crate::dispatcher::value_contains_explicit_gpu)
    {
        return Err(error(
            kind,
            "explicit gpuArray input has no admitted resident execution route",
        ));
    }
    host(kind, runtime)
}

pub(super) fn error(kind: NumericalReductionKind, detail: &str) -> RuntimeError {
    terminal(crate::runtime_descriptor_error_with_detail(
        kind.catalog_entry().identity.name,
        kind.internal_error(),
        detail,
    ))
}

pub(super) fn completed<T>(
    kind: NumericalReductionKind,
    result: ProviderExecutionResult<Option<T>>,
) -> BuiltinResult<Option<T>> {
    result.map_err(|source| {
        crate::runtime_error::with_provider_execution_source(
            error(kind, "provider execution failed"),
            source.with_unproven_effects(),
        )
    })
}

pub(super) async fn attempt<T>(
    kind: NumericalReductionKind,
    pending: ProviderExecutionFuture<'_, Option<T>>,
) -> BuiltinResult<Option<T>> {
    completed(kind, pending.await)
}
