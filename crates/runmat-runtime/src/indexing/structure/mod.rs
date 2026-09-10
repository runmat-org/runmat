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

pub fn read_with_plan(array: &StructArray, plan: &IndexPlan) -> Result<Value, RuntimeError> {
    array
        .select_linear(&zero_based_indices(plan), plan.output_shape.clone())
        .map_err(|error| mex("StructIndexing", error))
}

fn zero_based_indices(plan: &IndexPlan) -> Vec<usize> {
    plan.indices.iter().map(|index| *index as usize).collect()
}
