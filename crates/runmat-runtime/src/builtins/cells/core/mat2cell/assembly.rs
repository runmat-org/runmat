use runmat_value::{CellArray, Value};

use super::error::{mat2cell_error_with_message, MAT2CELL_ERROR_INTERNAL};
use super::input::Input;
use super::partition::PartitionPlan;
use crate::builtins::cells::core::block_layout::column_major_coordinates;
use crate::BuiltinResult;

pub(super) fn build(input: &Input, plan: PartitionPlan) -> BuiltinResult<Value> {
    if plan.cell_count == 0 || plan.sizes.iter().any(Vec::is_empty) {
        return finish(Vec::new(), plan.cell_shape);
    }
    let mut cells = Vec::with_capacity(plan.cell_count);
    for linear in 0..plan.cell_count {
        let coordinates = column_major_coordinates(linear, &plan.cell_shape);
        let (start, sizes) = selection(&plan, &coordinates);
        cells.push(input.extract(&start, &sizes)?);
    }
    finish(cells, plan.cell_shape)
}

fn selection(plan: &PartitionPlan, coordinates: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let mut start = Vec::with_capacity(plan.sizes.len());
    let mut sizes = Vec::with_capacity(plan.sizes.len());
    for (dimension, parts) in plan.sizes.iter().enumerate() {
        start.push(plan.offsets[dimension][coordinates[dimension]]);
        sizes.push(parts[coordinates[dimension]]);
    }
    (start, sizes)
}

fn finish(cells: Vec<Value>, shape: Vec<usize>) -> BuiltinResult<Value> {
    CellArray::from_column_major(cells, shape)
        .map(Value::Cell)
        .map_err(|error| {
            mat2cell_error_with_message(format!("mat2cell: {error}"), &MAT2CELL_ERROR_INTERNAL)
        })
}
