use super::error::{
    cell2mat_error_with_message, CELL2MAT_ERROR_INVALID_CONTENTS, CELL2MAT_ERROR_SIZE_EXCEEDED,
};
use super::input::{CellEntry, ElementKind};
use crate::builtins::cells::core::block_layout::{
    checked_element_count, extend_with_ones, prefix_offsets,
};
use crate::BuiltinResult;

pub(super) struct AssemblyPlan {
    pub(super) cell_rank: usize,
    pub(super) cell_coordinates: Vec<Vec<usize>>,
    pub(super) block_offsets: Vec<Vec<usize>>,
    pub(super) output_shape: Vec<usize>,
    pub(super) output_len: usize,
}

pub(super) fn build(
    cell_shape: &[usize],
    entries: &[CellEntry],
    kind: ElementKind,
) -> BuiltinResult<AssemblyPlan> {
    let cell_rank = cell_shape.len();
    let cell_coordinates = (0..entries.len())
        .map(|linear| row_major_coordinates(linear, cell_shape))
        .collect::<Vec<_>>();
    let mut block_extents = cell_shape
        .iter()
        .map(|extent| vec![0; *extent])
        .collect::<Vec<_>>();
    let mut trailing_shape = None;
    for (entry, coordinates) in entries.iter().zip(&cell_coordinates) {
        record_grid_extents(entry, coordinates, cell_rank, &mut block_extents)?;
        record_trailing_shape(entry, cell_rank, &mut trailing_shape)?;
    }
    let mut output_shape = block_extents
        .iter()
        .map(|extents| checked_sum(extents))
        .collect::<BuiltinResult<Vec<_>>>()?;
    output_shape.extend(trailing_shape.unwrap_or_default());
    if output_shape.is_empty() {
        output_shape = vec![0, 0];
    }
    if kind == ElementKind::Character && output_shape.len() > 2 {
        return Err(invalid(
            "character contents must form a 2-D character array",
        ));
    }
    let output_len = checked_element_count(&output_shape).ok_or_else(size_exceeded)?;
    let block_offsets = block_extents
        .iter()
        .map(|extents| prefix_offsets(extents).ok_or_else(size_exceeded))
        .collect::<BuiltinResult<Vec<_>>>()?;
    Ok(AssemblyPlan {
        cell_rank,
        cell_coordinates,
        block_offsets,
        output_shape,
        output_len,
    })
}

fn record_grid_extents(
    entry: &CellEntry,
    coordinates: &[usize],
    cell_rank: usize,
    block_extents: &mut [Vec<usize>],
) -> BuiltinResult<()> {
    let shape = extend_with_ones(&entry.shape, cell_rank);
    for dimension in 0..cell_rank {
        let slot = block_extents
            .get_mut(dimension)
            .and_then(|extents| extents.get_mut(coordinates[dimension]))
            .ok_or_else(|| invalid("cell index is outside its declared shape"))?;
        if *slot == 0 {
            *slot = shape[dimension];
        } else if *slot != shape[dimension] {
            return Err(invalid(
                "cell-grid block sizes must agree along each shared coordinate",
            ));
        }
    }
    Ok(())
}

fn record_trailing_shape(
    entry: &CellEntry,
    cell_rank: usize,
    expected: &mut Option<Vec<usize>>,
) -> BuiltinResult<()> {
    let trailing = entry.shape.get(cell_rank..).unwrap_or_default();
    match expected {
        Some(shape) if shape.as_slice() != trailing => Err(invalid(
            "higher-dimensional extents must agree across all cells",
        )),
        Some(_) => Ok(()),
        None => {
            *expected = Some(trailing.to_vec());
            Ok(())
        }
    }
}

fn checked_sum(extents: &[usize]) -> BuiltinResult<usize> {
    extents
        .iter()
        .try_fold(0usize, |sum, extent| sum.checked_add(*extent))
        .ok_or_else(size_exceeded)
}

fn row_major_coordinates(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    let mut coordinates = vec![0; shape.len()];
    for (index, extent) in shape.iter().enumerate().rev() {
        if *extent > 0 {
            coordinates[index] = linear % *extent;
            linear /= *extent;
        }
    }
    coordinates
}

fn invalid(detail: &'static str) -> crate::RuntimeError {
    cell2mat_error_with_message(
        format!("cell2mat: {detail}"),
        &CELL2MAT_ERROR_INVALID_CONTENTS,
    )
}

fn size_exceeded() -> crate::RuntimeError {
    cell2mat_error_with_message(
        "cell2mat: resulting matrix exceeds platform limits",
        &CELL2MAT_ERROR_SIZE_EXCEEDED,
    )
}
