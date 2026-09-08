//! Orchestration for cell-block concatenation.

use runmat_value::{Tensor, Value};

use super::error::{cell2mat_error_with_message, invalid_input, CELL2MAT_ERROR_INTERNAL};
use crate::BuiltinResult;

pub(super) async fn execute(value: Value) -> BuiltinResult<Value> {
    let Value::Cell(cells) = value else {
        return Err(invalid_input("expected a cell array input"));
    };
    if cells.data.is_empty() {
        return empty_double();
    }
    let (kind, entries) = super::input::gather(&cells).await?;
    let plan = super::plan::build(&cells.shape, &entries, kind)?;
    super::concatenate::assemble(kind, &entries, &plan)
}

fn empty_double() -> BuiltinResult<Value> {
    Tensor::new(Vec::new(), vec![0, 0])
        .map(Value::Tensor)
        .map_err(|error| {
            cell2mat_error_with_message(format!("cell2mat: {error}"), &CELL2MAT_ERROR_INTERNAL)
        })
}
