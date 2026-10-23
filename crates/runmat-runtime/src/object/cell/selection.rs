use runmat_value::{CellArray, Value};

use super::{mex, parse_cell_index_value_for_len};
use crate::RuntimeError;

#[cfg(test)]
mod tests;

pub(super) fn row_major_pos_from_linear(
    cell: &CellArray,
    index: usize,
) -> Result<usize, RuntimeError> {
    if index == 0 || index > cell.data().len() {
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
        return Ok(cell.data().len());
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
    linear_index_from_subscripts_by(cell, indices.len(), |axis, _| Ok(indices[axis]))
}

fn linear_index_from_subscripts_by(
    cell: &CellArray,
    count: usize,
    mut index_at: impl FnMut(usize, usize) -> Result<usize, RuntimeError>,
) -> Result<usize, RuntimeError> {
    for axis in 0..count {
        cell_selector_extent(cell, count, axis)?;
    }
    let mut zero_based = 0usize;
    let mut stride = 1usize;
    for axis in 0..count {
        let extent = cell_selector_extent(cell, count, axis)?;
        let index = index_at(axis, extent)?;
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
    CellSelection::new(cell, indices)?.collect()
}

fn is_colon(value: &Value) -> bool {
    matches!(value, Value::String(text) if text == ":")
        || matches!(value, Value::CharArray(chars) if chars.rows == 1 && chars.data().len() == 1 && chars.data()[0].raw() == u16::from(b':'))
}

/// The existing Cell brace policy as a repeatable view, rather than nested
/// selector/coordinate vectors. First selector still varies fastest.
pub(super) struct CellSelection<'a> {
    cell: &'a CellArray,
    indices: &'a [Value],
    len: usize,
}

impl<'a> CellSelection<'a> {
    pub(super) fn new(cell: &'a CellArray, indices: &'a [Value]) -> Result<Self, RuntimeError> {
        let len = match indices {
            [] => return Err(mex("CellIndexType", "Unsupported cell index type")),
            [value] if is_colon(value) => cell.data().len(),
            [Value::Num(_) | Value::Int(_)] => 1,
            [Value::Tensor(tensor)] => tensor.len(),
            [_] => return Err(mex("CellIndexType", "Unsupported cell index type")),
            _ => {
                for axis in 0..indices.len() {
                    cell_selector_extent(cell, indices.len(), axis)?;
                }
                // Preserve eager scalar parsing, including selectors after an
                // empty colon. Index bounds are checked when visiting values.
                for (axis, value) in indices.iter().enumerate() {
                    let extent = cell_selector_extent(cell, indices.len(), axis)?;
                    if !is_colon(value) {
                        parse_cell_index_value_for_len(value, extent)?;
                    }
                }
                if indices.iter().enumerate().any(|(axis, value)| {
                    is_colon(value)
                        && cell_selector_extent(cell, indices.len(), axis)
                            .expect("validated cell extent")
                            == 0
                }) {
                    0
                } else {
                    indices
                        .iter()
                        .enumerate()
                        .try_fold(1usize, |length, (axis, value)| {
                            let count = if is_colon(value) {
                                cell_selector_extent(cell, indices.len(), axis)?
                            } else {
                                1
                            };
                            length.checked_mul(count).ok_or_else(selection_size_error)
                        })?
                }
            }
        };
        Ok(Self { cell, indices, len })
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    pub(super) fn value_at(&self, ordinal: usize) -> Result<Value, RuntimeError> {
        self.value_ref_at(ordinal).cloned()
    }

    pub(super) fn value_ref_at(&self, ordinal: usize) -> Result<&'a Value, RuntimeError> {
        let linear = match self.indices {
            [value] if is_colon(value) => ordinal.checked_add(1).ok_or_else(index_size_error)?,
            [Value::Tensor(tensor)]
                if !(super::tensor_element_len(tensor) == 1
                    && tensor.shape().iter().product::<usize>() == 1) =>
            {
                super::parse_positive_cell_index(super::index_scalar_from_numeric(
                    tensor
                        .numeric_value_at(ordinal)
                        .expect("validated numeric tensor storage"),
                )?)?
            }
            [value] => parse_cell_index_value_for_len(value, self.cell.data().len())?,
            _ => {
                let mut remainder = ordinal;
                linear_index_from_subscripts_by(self.cell, self.indices.len(), |axis, extent| {
                    if is_colon(&self.indices[axis]) {
                        let index = remainder % extent + 1;
                        remainder /= extent;
                        Ok(index)
                    } else {
                        parse_cell_index_value_for_len(&self.indices[axis], extent)
                    }
                })?
            }
        };
        Ok(&self.cell.data()[row_major_pos_from_linear(self.cell, linear)?])
    }

    pub(super) fn collect(&self) -> Result<Vec<Value>, RuntimeError> {
        (0..self.len).map(|index| self.value_at(index)).collect()
    }
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
