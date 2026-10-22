//! Typed object selector/path adapters over the canonical host array owner.
use super::{ObjectIndexComponent, ObjectSubscript};
use crate::{
    context::{
        host_allocation::{prepare_host_array, HostAllocationError, PreparedHostArray},
        RuntimeContext,
    },
    RuntimeError,
};
pub type PreparedObjectIndexComponents = PreparedHostArray<ObjectIndexComponent>;
pub type PreparedObjectSubscripts = PreparedHostArray<ObjectSubscript>;
pub fn prepare_components_in_context(
    context: &RuntimeContext,
    count: usize,
) -> Result<PreparedObjectIndexComponents, RuntimeError> {
    prepare_host_array(context, count)
}
pub fn prepare_subscripts_in_context(
    context: &RuntimeContext,
    count: usize,
) -> Result<PreparedObjectSubscripts, RuntimeError> {
    prepare_host_array(context, count)
}
pub(crate) use crate::context::host_allocation::initialize_host_array as array_in_context;
pub(crate) fn allocation_error(
    context: Option<&RuntimeContext>,
    source: HostAllocationError,
) -> RuntimeError {
    crate::context::host_allocation::host_allocation_runtime_error(
        context,
        "RunMat:Indexing:ControlAllocation",
        "indexing selector/path storage",
        source,
    )
}
