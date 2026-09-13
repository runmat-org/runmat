use thiserror::Error;

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub(super) enum DenseSolveError {
    #[error("dense solve dimensions are inconsistent")]
    DimensionMismatch,
    #[error("dense solve contains a nonfinite value")]
    NonFinite,
    #[error("dense solve matrix is numerically singular")]
    Singular,
}

pub(super) fn solve_square(
    matrix: &[f64],
    right_hand_side: &[f64],
    size: usize,
) -> Result<Vec<f64>, DenseSolveError> {
    if size == 0
        || matrix.len()
            != size
                .checked_mul(size)
                .ok_or(DenseSolveError::DimensionMismatch)?
        || right_hand_side.len() != size
    {
        return Err(DenseSolveError::DimensionMismatch);
    }
    if matrix
        .iter()
        .chain(right_hand_side)
        .any(|value| !value.is_finite())
    {
        return Err(DenseSolveError::NonFinite);
    }
    let mut matrix = matrix.to_vec();
    let mut solution = right_hand_side.to_vec();
    let scale = matrix
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let pivot_floor = f64::EPSILON * size as f64 * scale.max(1.0);
    for pivot_column in 0..size {
        let pivot_row = (pivot_column..size)
            .max_by(|left, right| {
                matrix[*left * size + pivot_column]
                    .abs()
                    .total_cmp(&matrix[*right * size + pivot_column].abs())
            })
            .ok_or(DenseSolveError::Singular)?;
        let pivot = matrix[pivot_row * size + pivot_column];
        if pivot.abs() <= pivot_floor {
            return Err(DenseSolveError::Singular);
        }
        if pivot_row != pivot_column {
            for column in 0..size {
                matrix.swap(pivot_column * size + column, pivot_row * size + column);
            }
            solution.swap(pivot_column, pivot_row);
        }
        for row in pivot_column + 1..size {
            let factor =
                matrix[row * size + pivot_column] / matrix[pivot_column * size + pivot_column];
            matrix[row * size + pivot_column] = 0.0;
            for column in pivot_column + 1..size {
                matrix[row * size + column] -= factor * matrix[pivot_column * size + column];
            }
            solution[row] -= factor * solution[pivot_column];
        }
    }
    for row in (0..size).rev() {
        let known = (row + 1..size)
            .map(|column| matrix[row * size + column] * solution[column])
            .sum::<f64>();
        solution[row] = (solution[row] - known) / matrix[row * size + row];
    }
    if solution.iter().any(|value| !value.is_finite()) {
        return Err(DenseSolveError::NonFinite);
    }
    Ok(solution)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_pivoting_solves_a_nonsingular_system_and_rejects_rank_loss() {
        let solution = solve_square(&[0.0, 2.0, 1.0, 3.0], &[4.0, 7.0], 2).unwrap();
        assert!((solution[0] - 1.0).abs() <= 1.0e-14);
        assert!((solution[1] - 2.0).abs() <= 1.0e-14);
        assert_eq!(
            solve_square(&[1.0, 2.0, 2.0, 4.0], &[1.0, 2.0], 2),
            Err(DenseSolveError::Singular)
        );
    }
}
