//! One original homogeneous ObjectArray destination. Borrowed source children
//! must remain rooted until publication. Checkpoints fence cancellation; index
//! initialization and class preparation are synchronous, not async work bounds.
use crate::context::{
    host_allocation::{
        host_allocation_runtime_error, HostAllocationError, HostAllocationPlan,
        HostAllocationReceipt,
    },
    RuntimeContext,
};
use crate::{execution::RootedValue, RuntimeError};
use runmat_value::{
    ObjectArray, ObjectArrayBuildError, ObjectArrayBuilder, PreparedObjectClassName,
    SelectedObjectClassFacts, SharedClassIdentity, SharedShape, Value,
};
use std::rc::Rc;
mod shape;
mod construction;
mod projection;
pub(crate) use construction::{Class, ObjectArrayConstruction};
pub(crate) use projection::{copy_source, select};
pub(crate) mod indices;
pub(crate) use shape::reshape;
#[cfg(test)]
mod tests;

fn allocation(runtime: Option<&RuntimeContext>, source: HostAllocationError) -> RuntimeError {
    host_allocation_runtime_error(
        runtime,
        "RunMat:ObjectArray:Allocation",
        "object array backing admission failed",
        source,
    )
}
fn build_error(runtime: Option<&RuntimeContext>, source: ObjectArrayBuildError) -> RuntimeError {
    match source {
        ObjectArrayBuildError::Allocation(source) => allocation(runtime, source.into()),
        source => crate::runtime_error::semantic_error_with_source(
            "ObjectArrayConstruction",
            "object array construction failed",
            source,
        ),
    }
}
