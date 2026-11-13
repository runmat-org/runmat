use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::RuntimeError;

pub(super) fn exact_receipt(
    runtime: &RuntimeContext,
    physical: usize,
    claims: usize,
) -> Result<Option<HostAllocationReceipt>, RuntimeError> {
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(physical).map_err(|_| allocation(runtime, HostAllocationError::Overflow))?,
        claims,
    )
    .map_err(|source| allocation(runtime, source))?;
    HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(0, retained).map_err(|source| allocation(runtime, source))?,
    )
    .map_err(|source| allocation(runtime, source))
}

pub(super) fn initialization<E>(
    runtime: &RuntimeContext,
    source: runmat_value::ExactArrayInitializationError<E>,
) -> RuntimeError
where
    E: Into<HostAllocationError>,
{
    match source {
        runmat_value::ExactArrayInitializationError::Allocation(source) => {
            allocation(runtime, HostAllocationError::Numeric(source))
        }
        runmat_value::ExactArrayInitializationError::Element(source) => {
            allocation(runtime, source.into())
        }
    }
}

pub(super) fn allocation(runtime: &RuntimeContext, source: HostAllocationError) -> RuntimeError {
    host_allocation_runtime_error(
        Some(runtime),
        "RunMat:graphics:HostAllocation",
        "prepare graphics object snapshot",
        source,
    )
}
