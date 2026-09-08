//! Orchestration for array-block partitioning.

use runmat_value::Value;

use super::error::{mat2cell_error_with_message, MAT2CELL_ERROR_INVALID_INPUT};
use crate::BuiltinResult;

pub(super) async fn execute(value: Value, partitions: Vec<Value>) -> BuiltinResult<Value> {
    if partitions.is_empty() {
        return Err(mat2cell_error_with_message(
            "mat2cell: expected at least one partition vector",
            &MAT2CELL_ERROR_INVALID_INPUT,
        ));
    }
    let value = super::admission::gather_input(value).await?;
    let partitions = super::admission::gather_partitions(partitions).await?;
    let input = super::input::Input::new(value)?;
    let plan = super::partition::for_dimensions(input.dimensions(), &partitions)?;
    super::assembly::build(&input, plan)
}
