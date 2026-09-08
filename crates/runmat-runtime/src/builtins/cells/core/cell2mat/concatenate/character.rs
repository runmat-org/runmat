use runmat_value::{CharArray, Value};

use crate::builtins::cells::core::block_layout::extend_with_ones;
use crate::builtins::cells::core::cell2mat::error::{
    cell2mat_error_with_message, CELL2MAT_ERROR_INTERNAL, CELL2MAT_ERROR_SIZE_EXCEEDED,
};
use crate::builtins::cells::core::cell2mat::input::{CellEntry, EntryData};
use crate::builtins::cells::core::cell2mat::plan::AssemblyPlan;
use crate::BuiltinResult;

pub(super) fn assemble(entries: &[CellEntry], plan: &AssemblyPlan) -> BuiltinResult<Value> {
    let rows = plan.output_shape.first().copied().unwrap_or(0);
    let columns = plan.output_shape.get(1).copied().unwrap_or(1);
    let mut output = vec!['\0'; rows.checked_mul(columns).ok_or_else(size_exceeded)?];
    for (entry, coordinates) in entries.iter().zip(&plan.cell_coordinates) {
        let EntryData::Character(data) = &entry.data else {
            continue;
        };
        if entry.len() == 0 {
            continue;
        }
        let shape = extend_with_ones(&entry.shape, 2);
        let row_offset = offset(plan, 0, coordinates.first().copied().unwrap_or(0))?;
        let column_offset = offset(plan, 1, coordinates.get(1).copied().unwrap_or(0))?;
        for (source, value) in data.iter().enumerate() {
            let local_row = source / shape[1];
            let local_column = source % shape[1];
            let destination = (row_offset + local_row)
                .checked_mul(columns)
                .and_then(|base| base.checked_add(column_offset + local_column))
                .ok_or_else(size_exceeded)?;
            output[destination] = *value;
        }
    }
    CharArray::new(output, rows, columns)
        .map(Value::CharArray)
        .map_err(|error| {
            cell2mat_error_with_message(format!("cell2mat: {error}"), &CELL2MAT_ERROR_INTERNAL)
        })
}

fn offset(plan: &AssemblyPlan, dimension: usize, coordinate: usize) -> BuiltinResult<usize> {
    plan.block_offsets
        .get(dimension)
        .and_then(|offsets| offsets.get(coordinate))
        .copied()
        .ok_or_else(size_exceeded)
}

fn size_exceeded() -> crate::RuntimeError {
    cell2mat_error_with_message(
        "cell2mat: character destination exceeds platform limits",
        &CELL2MAT_ERROR_SIZE_EXCEEDED,
    )
}
