use runmat_value::{CharArray, StringArray, Value};

use super::super::error::{
    mat2cell_error_with_message, MAT2CELL_ERROR_INTERNAL, MAT2CELL_ERROR_INVALID_PARTITION,
};
use super::super::extract;
use crate::BuiltinResult;

pub(super) fn string(
    array: &StringArray,
    shape: &[usize],
    start: &[usize],
    sizes: &[usize],
) -> BuiltinResult<Value> {
    let data = extract::block(&array.data, shape, start, sizes)?;
    if data.len() == 1 {
        return Ok(Value::String(data.into_iter().next().expect("one string")));
    }
    StringArray::new(data, extract::output_shape(sizes))
        .map(Value::StringArray)
        .map_err(internal)
}

pub(super) fn character(
    array: &CharArray,
    start: &[usize],
    sizes: &[usize],
) -> BuiltinResult<Value> {
    validate_character_rank(start, sizes)?;
    let row_start = start.first().copied().unwrap_or(0);
    let rows = sizes.first().copied().unwrap_or(1);
    let column_start = start.get(1).copied().unwrap_or(0);
    let columns = sizes.get(1).copied().unwrap_or(1);
    if row_start
        .checked_add(rows)
        .is_none_or(|end| end > array.rows)
        || column_start
            .checked_add(columns)
            .is_none_or(|end| end > array.cols)
    {
        return Err(partition_error());
    }
    let mut data = Vec::with_capacity(rows.saturating_mul(columns));
    for row in 0..rows {
        let base = (row_start + row)
            .checked_mul(array.cols)
            .and_then(|offset| offset.checked_add(column_start))
            .ok_or_else(partition_error)?;
        data.extend_from_slice(
            array
                .data
                .get(base..base + columns)
                .ok_or_else(partition_error)?,
        );
    }
    CharArray::new(data, rows, columns)
        .map(Value::CharArray)
        .map_err(internal)
}

fn validate_character_rank(start: &[usize], sizes: &[usize]) -> BuiltinResult<()> {
    if sizes
        .iter()
        .enumerate()
        .skip(2)
        .any(|(dimension, size)| *size != 1 || start.get(dimension).copied().unwrap_or(0) != 0)
    {
        return Err(partition_error());
    }
    Ok(())
}

fn partition_error() -> crate::RuntimeError {
    mat2cell_error_with_message(
        "mat2cell: character partition exceeds supported dimensions",
        &MAT2CELL_ERROR_INVALID_PARTITION,
    )
}

fn internal(error: impl std::fmt::Display) -> crate::RuntimeError {
    mat2cell_error_with_message(format!("mat2cell: {error}"), &MAT2CELL_ERROR_INTERNAL)
}
