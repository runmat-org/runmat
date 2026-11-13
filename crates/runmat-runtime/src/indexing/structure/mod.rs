//! Executor-neutral structure-array indexing operations.

mod assignment;
mod deletion;

#[cfg(test)]
mod tests;

use crate::indexing::plan::IndexPlan;
use crate::runtime_error::semantic_error as mex;
use crate::RuntimeError;
use runmat_value::{StructArray, Value};

pub use assignment::assign_with_plan;
pub use assignment::assign_with_plan_in_context;
pub use deletion::{delete_scalar_indices_in_context, delete_with_plan_in_context};

pub fn read_with_plan(array: &StructArray, plan: &IndexPlan) -> Result<Value, RuntimeError> {
    let context = crate::context::legacy::active();
    read_with_plan_in_context(context.as_ref(), array, plan)
}

pub fn read_with_plan_in_context(
    context: Option<&crate::context::RuntimeContext>,
    array: &StructArray,
    plan: &IndexPlan,
) -> Result<Value, RuntimeError> {
    let length = plan
        .output_shape
        .iter()
        .try_fold(1usize, |length, &extent| length.checked_mul(extent))
        .ok_or_else(|| {
            mex(
                "StructIndexing",
                "structure selection shape overflows addressable storage",
            )
        })?;
    if length != plan.indices.len() {
        return Err(mex(
            "StructIndexing",
            "structure selection shape differs from its index count",
        ));
    }
    crate::aggregate::structure::operations::select_by_index_in_context(
        context,
        array,
        &plan.output_shape,
        |index| plan.indices.get(index).map(|&index| index as usize),
    )
    .map_err(|error| {
        crate::aggregate::structure::operations::remap(error, |message| {
            mex("StructIndexing", message)
        })
    })
}
