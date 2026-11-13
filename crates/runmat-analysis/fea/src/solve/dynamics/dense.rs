use thiserror::Error;

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub(super) enum DenseSolveError {
    #[error("dense solve dimensions are inconsistent")]
    DimensionMismatch,
    #[error("dense solve contains a nonfinite value")]
    NonFinite,
    #[error("dense solve matrix is numerically singular")]
    Singular,
    #[error("dense solve numerical usage: {0}")]
    Usage(runmat_simulation_usage::LedgerError),
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
    let scale = matrix
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let pivot_floor = f64::EPSILON * size as f64 * scale.max(1.0);
    let lu = runmat_sparse_solvers::dense::DenseLu::from_row_major(matrix, size, pivot_floor)
        .map_err(|error| match error {
            runmat_sparse_solvers::SolverError::Usage(error) => DenseSolveError::Usage(error),
            _ => DenseSolveError::Singular,
        })?;
    let mut solution = right_hand_side.to_vec();
    lu.solve_in_place(&mut solution)
        .map_err(|error| match error {
            runmat_sparse_solvers::SolverError::Usage(error) => DenseSolveError::Usage(error),
            _ => DenseSolveError::Singular,
        })?;
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
