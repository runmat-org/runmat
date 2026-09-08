use crate::builtins::cells::core::block_layout::{
    checked_element_count, column_major_coordinates, column_major_strides, extend_with_ones,
};
use crate::builtins::cells::core::mat2cell::error::{
    mat2cell_error_with_message, MAT2CELL_ERROR_INVALID_PARTITION, MAT2CELL_ERROR_SIZE_EXCEEDED,
};
use crate::BuiltinResult;

pub(super) fn block<T: Clone>(
    data: &[T],
    shape: &[usize],
    start: &[usize],
    sizes: &[usize],
) -> BuiltinResult<Vec<T>> {
    let source_shape = extend_with_ones(shape, sizes.len());
    validate_bounds(&source_shape, start, sizes)?;
    let count = checked_element_count(sizes).ok_or_else(size_exceeded)?;
    let strides = column_major_strides(&source_shape).ok_or_else(size_exceeded)?;
    let mut result = Vec::with_capacity(count);
    for output_index in 0..count {
        let coordinates = column_major_coordinates(output_index, sizes);
        let source_index = coordinates
            .iter()
            .enumerate()
            .try_fold(0usize, |linear, (dimension, coordinate)| {
                (start[dimension] + coordinate)
                    .checked_mul(strides[dimension])
                    .and_then(|offset| linear.checked_add(offset))
            })
            .ok_or_else(size_exceeded)?;
        result.push(data.get(source_index).ok_or_else(index_error)?.clone());
    }
    Ok(result)
}

pub(super) fn output_shape(sizes: &[usize]) -> Vec<usize> {
    if sizes.is_empty() {
        vec![1, 1]
    } else {
        sizes.to_vec()
    }
}

fn validate_bounds(shape: &[usize], start: &[usize], sizes: &[usize]) -> BuiltinResult<()> {
    for dimension in 0..sizes.len() {
        if start[dimension]
            .checked_add(sizes[dimension])
            .is_none_or(|end| end > shape[dimension])
        {
            return Err(index_error());
        }
    }
    Ok(())
}

fn index_error() -> crate::RuntimeError {
    mat2cell_error_with_message(
        "mat2cell: partition exceeds input bounds",
        &MAT2CELL_ERROR_INVALID_PARTITION,
    )
}

fn size_exceeded() -> crate::RuntimeError {
    mat2cell_error_with_message(
        "mat2cell: block size exceeds platform limits",
        &MAT2CELL_ERROR_SIZE_EXCEEDED,
    )
}
