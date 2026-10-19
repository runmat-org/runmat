use runmat_value::{CellArray, Value};

use super::{index_cell_value, mex, parse_cell_index_value_for_len};
use crate::RuntimeError;

#[cfg(test)]
mod tests;

pub(super) fn row_major_pos_from_linear(
    cell: &CellArray,
    index: usize,
) -> Result<usize, RuntimeError> {
    if index == 0 || index > cell.data.len() {
        return Err(mex("CellIndexOutOfBounds", "Cell index out of bounds"));
    }
    if cell.rows <= 1 || cell.cols <= 1 {
        return Ok(index - 1);
    }
    let zero_based = index - 1;
    let page_len = cell.rows.checked_mul(cell.cols).ok_or_else(|| {
        mex(
            "CellIndexOutOfBounds",
            "Cell array shape exceeds supported size",
        )
    })?;
    let page = zero_based / page_len;
    let within_page = zero_based % page_len;
    let row = within_page % cell.rows;
    let column = within_page / cell.rows;
    Ok(page * page_len + row * cell.cols + column)
}

pub(super) fn cell_selector_extents(
    cell: &CellArray,
    selector_count: usize,
) -> Result<Vec<usize>, RuntimeError> {
    (0..selector_count)
        .map(|position| cell_selector_extent(cell, selector_count, position))
        .collect()
}

pub(super) fn cell_selector_extent(
    cell: &CellArray,
    selector_count: usize,
    position: usize,
) -> Result<usize, RuntimeError> {
    if selector_count == 0 || position >= selector_count {
        return Err(mex(
            "CellIndexOutOfBounds",
            "Cell selector position is out of bounds",
        ));
    }
    if selector_count == 1 {
        return Ok(cell.data.len());
    }
    if position + 1 == selector_count {
        return cell.shape[position.min(cell.shape.len())..]
            .iter()
            .try_fold(1usize, |length, extent| length.checked_mul(*extent))
            .ok_or_else(shape_size_error);
    }
    Ok(cell.shape.get(position).copied().unwrap_or(1))
}

pub(super) fn linear_index_from_subscripts(
    cell: &CellArray,
    indices: &[usize],
) -> Result<usize, RuntimeError> {
    let extents = cell_selector_extents(cell, indices.len())?;
    let mut zero_based = 0usize;
    let mut stride = 1usize;
    for (&index, extent) in indices.iter().zip(extents) {
        if index == 0 || index > extent {
            return Err(mex(
                "CellSubscriptOutOfBounds",
                "Cell subscript out of bounds",
            ));
        }
        zero_based = zero_based
            .checked_add(
                (index - 1)
                    .checked_mul(stride)
                    .ok_or_else(index_size_error)?,
            )
            .ok_or_else(index_size_error)?;
        stride = stride.checked_mul(extent).ok_or_else(shape_size_error)?;
    }
    zero_based.checked_add(1).ok_or_else(index_size_error)
}

pub(super) fn expand_cell_subscripts(
    cell: &CellArray,
    indices: &[Value],
) -> Result<Vec<Value>, RuntimeError> {
    let extents = cell_selector_extents(cell, indices.len())?;
    let selections = indices
        .iter()
        .zip(extents)
        .map(|(value, extent)| selector_values(value, extent))
        .collect::<Result<Vec<_>, _>>()?;
    if selections.iter().any(Vec::is_empty) {
        return Ok(Vec::new());
    }

    let output_len = selections
        .iter()
        .try_fold(1usize, |length, selection| {
            length.checked_mul(selection.len())
        })
        .ok_or_else(selection_size_error)?;
    let mut values = Vec::with_capacity(output_len);
    let mut positions = vec![0usize; selections.len()];
    loop {
        let subscripts = selections
            .iter()
            .zip(&positions)
            .map(|(selection, position)| selection[*position])
            .collect::<Vec<_>>();
        values.push(index_cell_value(cell, &subscripts)?);

        let mut dimension = 0usize;
        while dimension < positions.len() {
            positions[dimension] += 1;
            if positions[dimension] < selections[dimension].len() {
                break;
            }
            positions[dimension] = 0;
            dimension += 1;
        }
        if dimension == positions.len() {
            return Ok(values);
        }
    }
}

fn selector_values(value: &Value, extent: usize) -> Result<Vec<usize>, RuntimeError> {
    let is_colon = matches!(value, Value::String(text) if text == ":")
        || matches!(value, Value::CharArray(chars) if chars.row_string().as_deref() == Some(":"));
    if is_colon {
        return Ok((1..=extent).collect());
    }
    Ok(vec![parse_cell_index_value_for_len(value, extent)?])
}

fn shape_size_error() -> RuntimeError {
    mex(
        "CellIndexOutOfBounds",
        "Cell array shape exceeds supported size",
    )
}

fn index_size_error() -> RuntimeError {
    mex(
        "CellIndexOutOfBounds",
        "Cell array index exceeds supported size",
    )
}

fn selection_size_error() -> RuntimeError {
    mex(
        "CellIndexOutOfBounds",
        "Cell selection exceeds supported size",
    )
}
