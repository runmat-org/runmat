use runmat_accelerate_api::GpuTensorHandle;

use super::super::SolveOrientation;

pub(super) fn output_shape(
    orientation: SolveOrientation,
    lhs: &GpuTensorHandle,
    rhs: &GpuTensorHandle,
) -> Vec<usize> {
    match orientation {
        SolveOrientation::Left => vec![matrix_columns(&lhs.shape), matrix_columns(&rhs.shape)],
        SolveOrientation::Right => vec![matrix_rows(&lhs.shape), matrix_rows(&rhs.shape)],
    }
}

pub(super) fn disallowed_scalar(
    orientation: SolveOrientation,
    lhs: &GpuTensorHandle,
    rhs: &GpuTensorHandle,
) -> bool {
    match orientation {
        SolveOrientation::Left => is_scalar(lhs) || is_scalar(rhs),
        SolveOrientation::Right => is_scalar(rhs),
    }
}

fn is_scalar(handle: &GpuTensorHandle) -> bool {
    crate::builtins::common::shape::is_scalar_shape(&handle.shape)
}

fn matrix_rows(shape: &[usize]) -> usize {
    shape.first().copied().unwrap_or(1)
}

fn matrix_columns(shape: &[usize]) -> usize {
    shape
        .get(1)
        .copied()
        .unwrap_or_else(|| shape.first().copied().unwrap_or(1))
}
