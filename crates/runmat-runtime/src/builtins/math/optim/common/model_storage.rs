//! Native model metadata reuses canonical array, object and collection owners.
use super::ScalarCooperation;
use crate::context::{
    host_allocation::{host_allocation_runtime_error, HostAllocationError},
    RuntimeContext,
};
use crate::{BuiltinResult, RuntimeError};
mod collections;
mod native;
mod shape;
pub(in crate::builtins::math::optim) fn array<T>(
    cooperation: &ScalarCooperation,
    len: usize,
) -> BuiltinResult<crate::context::host_allocation::PreparedHostArray<T>> {
    cooperation.check_allocation_epoch()?;
    let array =
        crate::context::host_allocation::prepare_host_array_optional(cooperation.runtime(), len)?;
    cooperation.check_allocation_epoch()?;
    Ok(array)
}
pub(in crate::builtins::math::optim) use collections::{Collections, Map, Vector};
pub(in crate::builtins::math::optim) use native::{allocate_gc, publish, retain_handle};
pub(in crate::builtins::math::optim) use shape::shape;

fn allocation(runtime: Option<&RuntimeContext>, source: HostAllocationError) -> RuntimeError {
    host_allocation_runtime_error(
        runtime,
        "RunMat:OptimModel:Allocation",
        "optimization model backing",
        source,
    )
}
