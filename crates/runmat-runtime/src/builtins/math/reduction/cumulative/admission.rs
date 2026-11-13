use crate::{
    context::{
        host_allocation::{HostAllocationError, HostAllocationPolicy, HostAllocationReceipt},
        RuntimeContext,
    },
    BuiltinResult,
};
use runmat_builtins::CumulativeKind;
use runmat_value::Value;
#[cfg(test)]
mod tests;

pub(in crate::builtins::math::reduction) fn roots<'a>(
    runtime: Option<&RuntimeContext>,
    values: impl Iterator<Item = &'a Value> + Clone,
) -> BuiltinResult<crate::execution::RootedValueLease> {
    let result = match runtime {
        Some(runtime) => {
            crate::execution::RootedValueLease::for_values_iterator_in_context(values, runtime)
        }
        None => crate::execution::RootedValueLease::for_values(values)
            .map_err(|source| source.into_runtime_error()),
    };
    result.map_err(|mut error| {
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        error
    })
}

pub(in crate::builtins::math::reduction) fn control(
    kind: CumulativeKind,
    runtime: Option<&RuntimeContext>,
    source: HostAllocationError,
) -> crate::RuntimeError {
    let cancellation = runtime.map(RuntimeContext::cancellation_context);
    let mut error = crate::builtin::builtin_control_error(
        cancellation.as_ref(),
        kind.catalog_entry().identity.name,
        source,
    );
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}
pub(in crate::builtins::math::reduction) fn host(
    kind: CumulativeKind,
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
pub(in crate::builtins::math::reduction) fn input(
    kind: CumulativeKind,
    value: &Value,
) -> BuiltinResult<()> {
    if matches!(
        value,
        Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::Tensor(_)
            | Value::LogicalArray(_)
            | Value::Complex(..)
            | Value::ComplexTensor(_)
            | Value::GpuTensor(_)
    ) && !crate::builtins::common::validation::is_typed_complex_integer(value)
    {
        return Ok(());
    }
    let entry = kind.catalog_entry();
    Err(crate::runtime_descriptor_error_with_detail(
        entry.identity.name,
        &entry.descriptor.errors[1],
        "expected dense numeric or logical input; complex integers are unsupported",
    ))
}
