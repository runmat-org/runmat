//! Refresh a portable numeric cache only after exact unit/origin selection.
use runmat_runtime::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::execution::RootedValue;
use runmat_runtime::RuntimeError;
use runmat_value::{BoundFunctionAuthority, Closure, ClosureHeaderError, Value};

pub(super) fn normalize(
    runtime: &RuntimeContext,
    registry: &runmat_vm::FunctionRegistry,
    callback: RootedValue,
    function: runmat_hir::FunctionId,
) -> Result<Option<RootedValue>, RuntimeError> {
    let Value::Closure(closure) = callback.value() else {
        return Ok(None);
    };
    let Some(binding) = &closure.binding else {
        return Ok(None);
    };
    if !matches!(&binding.authority, BoundFunctionAuthority::Portable { .. })
        || binding.function == function.0
    {
        return Ok(None);
    }
    let error = |source| {
        host_allocation_runtime_error(
            Some(runtime),
            "RunMat:Callable:Binding",
            "portable callable cache normalization failed",
            source,
        )
    };
    let binding = registry.exact_bound_binding(function).ok_or_else(|| {
        runmat_runtime::runtime_error::semantic_error(
            "UnavailableBoundFunction",
            "selected executable definition is no longer available",
        )
    })?;
    // Every child is already an original immutable/shared owner. Only the
    // new physical header is constructed here; source-graph authority retains
    // its independent canonical metadata ownership.
    let mut data = closure.clone().into_data();
    data.binding = Some(binding);
    let bytes = u64::try_from(Closure::resolved_header_bytes())
        .map_err(|_| error(HostAllocationError::Overflow))?;
    let retained = HostAllocationPlan::retained_exact_output_bytes(bytes, 1).map_err(error)?;
    let plan = HostAllocationPlan::new(0, retained).map_err(error)?;
    let receipt = HostAllocationReceipt::admit(Some(runtime), plan).map_err(error)?;
    let closure = match &receipt {
        Some(receipt) => receipt
            .with_output_construction(runtime, |permit| {
                Closure::try_from_data_with_exact_header(data, permit)
            })
            .map_err(error)?
            .map_err(|source| match source {
                ClosureHeaderError::Allocation(source) => {
                    error(HostAllocationError::Numeric(source))
                }
                ClosureHeaderError::CapturesNotShared => {
                    runmat_runtime::runtime_error::semantic_error(
                        "CallableArguments",
                        "portable callable captures lack their original shared owner",
                    )
                }
            })?,
        None => Closure::from_data(data),
    };
    // Keep the original rooted target alive through the contextual successor
    // root admission. No raw managed Value escapes and no capture is copied.
    let normalized = RootedValue::in_context(Value::Closure(closure), runtime)?;
    if let Some(receipt) = &receipt {
        receipt.checkpoint(runtime).map_err(error)?;
    }
    drop(callback);
    Ok(Some(normalized))
}
