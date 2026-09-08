use super::super::error::{cell2mat_error_with_message, CELL2MAT_ERROR_SIZE_EXCEEDED};
use super::super::input::CellEntry;
use super::super::plan::AssemblyPlan;
use crate::builtins::cells::core::block_layout::{
    column_major_coordinates, column_major_strides, extend_with_ones,
};
use crate::BuiltinResult;

pub(super) fn for_each(
    entry: &CellEntry,
    cell_coordinates: &[usize],
    plan: &AssemblyPlan,
    mut visit: impl FnMut(usize, usize) -> BuiltinResult<()>,
) -> BuiltinResult<()> {
    if entry.len() == 0 {
        return Ok(());
    }
    let rank = plan.output_shape.len();
    let shape = extend_with_ones(&entry.shape, rank);
    let strides = column_major_strides(&plan.output_shape).ok_or_else(size_exceeded)?;
    let base = base_offsets(cell_coordinates, plan, rank)?;
    for source in 0..entry.len() {
        let local = column_major_coordinates(source, &shape);
        let destination = local
            .iter()
            .enumerate()
            .try_fold(0usize, |linear, (dimension, coordinate)| {
                (base[dimension] + coordinate)
                    .checked_mul(strides[dimension])
                    .and_then(|offset| linear.checked_add(offset))
            })
            .ok_or_else(size_exceeded)?;
        visit(source, destination)?;
    }
    Ok(())
}

fn base_offsets(
    coordinates: &[usize],
    plan: &AssemblyPlan,
    rank: usize,
) -> BuiltinResult<Vec<usize>> {
    let mut base = vec![0; rank];
    for dimension in 0..plan.cell_rank.min(plan.block_offsets.len()) {
        base[dimension] = plan.block_offsets[dimension]
            .get(coordinates[dimension])
            .copied()
            .ok_or_else(size_exceeded)?;
    }
    Ok(base)
}

fn size_exceeded() -> crate::RuntimeError {
    cell2mat_error_with_message(
        "cell2mat: destination offset exceeds platform limits",
        &CELL2MAT_ERROR_SIZE_EXCEEDED,
    )
}
