use runmat_value::{ComplexTensor, Tensor};

use crate::BuiltinResult;

use super::super::{errors, SolveOrientation};

pub(super) fn ensure_matrix(orientation: SolveOrientation, shape: &[usize]) -> BuiltinResult<()> {
    if shape.len() <= 2 || shape.iter().skip(2).all(|&extent| extent == 1) {
        Ok(())
    } else {
        Err(errors::invalid_input(
            orientation,
            format!(
                "{}: inputs must be 2-D matrices or vectors",
                orientation.name()
            ),
        ))
    }
}

pub(super) fn ensure_matching_real_dimension(
    orientation: SolveOrientation,
    lhs: &Tensor,
    rhs: &Tensor,
) -> BuiltinResult<()> {
    let matches = match orientation {
        SolveOrientation::Left => lhs.rows() == rhs.rows(),
        SolveOrientation::Right => lhs.cols() == rhs.cols(),
    };
    matching_dimension_result(orientation, matches)
}

pub(super) fn ensure_matching_complex_dimension(
    orientation: SolveOrientation,
    lhs: &ComplexTensor,
    rhs: &ComplexTensor,
) -> BuiltinResult<()> {
    let matches = match orientation {
        SolveOrientation::Left => lhs.rows == rhs.rows,
        SolveOrientation::Right => lhs.cols == rhs.cols,
    };
    matching_dimension_result(orientation, matches)
}

fn matching_dimension_result(orientation: SolveOrientation, matches: bool) -> BuiltinResult<()> {
    if matches {
        Ok(())
    } else {
        Err(errors::invalid_input(
            orientation,
            "Matrix dimensions must agree.",
        ))
    }
}

pub(super) fn has_empty_solve_dimension(
    orientation: SolveOrientation,
    lhs: &Tensor,
    rhs: &Tensor,
) -> bool {
    match orientation {
        SolveOrientation::Left => lhs.rows() == 0,
        SolveOrientation::Right => rhs.cols() == 0,
    }
}

pub(super) fn real_output_dimensions(
    orientation: SolveOrientation,
    lhs: &Tensor,
    rhs: &Tensor,
) -> (usize, usize) {
    match orientation {
        SolveOrientation::Left => (lhs.cols(), rhs.cols()),
        SolveOrientation::Right => (lhs.rows(), rhs.rows()),
    }
}

pub(super) fn has_empty_complex_solve_dimension(
    orientation: SolveOrientation,
    lhs: &ComplexTensor,
    rhs: &ComplexTensor,
) -> bool {
    match orientation {
        SolveOrientation::Left => lhs.rows == 0,
        SolveOrientation::Right => rhs.cols == 0,
    }
}

pub(super) fn complex_output_dimensions(
    orientation: SolveOrientation,
    lhs: &ComplexTensor,
    rhs: &ComplexTensor,
) -> (usize, usize) {
    match orientation {
        SolveOrientation::Left => (lhs.cols, rhs.cols),
        SolveOrientation::Right => (lhs.rows, rhs.rows),
    }
}
