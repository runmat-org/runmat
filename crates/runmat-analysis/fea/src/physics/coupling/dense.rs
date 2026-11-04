#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum DenseCouplingError {
    #[error("{0}")]
    Message(String),
    #[error("dense coupling numerical usage: {0}")]
    Usage(runmat_simulation_usage::LedgerError),
}

impl From<String> for DenseCouplingError {
    fn from(message: String) -> Self {
        Self::Message(message)
    }
}
impl From<&'static str> for DenseCouplingError {
    fn from(message: &'static str) -> Self {
        Self::Message(message.into())
    }
}

pub(crate) fn solve(matrix: &[Vec<f64>], rhs: &[f64]) -> Result<Vec<f64>, DenseCouplingError> {
    let n = rhs.len();
    if matrix.len() != n || matrix.iter().any(|row| row.len() != n) {
        return Err("dense coupling operator must be square and match the right-hand side".into());
    }
    if matrix.iter().flatten().any(|value| !value.is_finite())
        || rhs.iter().any(|value| !value.is_finite())
    {
        return Err("dense coupling system must contain only finite values".into());
    }

    let flat: Vec<f64> = matrix.iter().flatten().copied().collect();
    let lu = runmat_sparse_solvers::dense::DenseLu::from_row_major(&flat, n, 1.0e-24).map_err(
        |error| match error {
            runmat_sparse_solvers::SolverError::Usage(error) => DenseCouplingError::Usage(error),
            runmat_sparse_solvers::SolverError::Singular { pivot } => DenseCouplingError::Message(
                format!("dense coupling operator is singular at pivot {pivot}"),
            ),
            other => DenseCouplingError::Message(other.to_string()),
        },
    )?;
    let mut solution = rhs.to_vec();
    lu.solve_in_place(&mut solution)
        .map_err(|error| match error {
            runmat_sparse_solvers::SolverError::Usage(error) => DenseCouplingError::Usage(error),
            other => DenseCouplingError::Message(other.to_string()),
        })?;
    if solution.iter().any(|value| !value.is_finite()) {
        return Err("dense coupling solve produced a nonfinite value".into());
    }
    Ok(solution)
}

pub(crate) fn apply(matrix: &[Vec<f64>], x: &[f64]) -> Result<Vec<f64>, DenseCouplingError> {
    if matrix.iter().any(|row| row.len() != x.len()) {
        return Err("dense coupling operator width must match the vector".into());
    }
    Ok(matrix
        .iter()
        .map(|row| {
            row.iter()
                .zip(x.iter())
                .map(|(coefficient, value)| coefficient * value)
                .sum::<f64>()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::{apply, solve};

    #[test]
    fn pivoted_solve_matches_an_independent_two_by_two_answer() {
        let matrix = vec![vec![0.0, 2.0], vec![1.0, 3.0]];
        let rhs = vec![4.0, 7.0];
        let solution = solve(&matrix, &rhs).expect("nonsingular system should solve");
        assert!((solution[0] - 1.0).abs() <= 1.0e-12);
        assert!((solution[1] - 2.0).abs() <= 1.0e-12);
        let reconstructed = apply(&matrix, &solution).expect("shape should remain valid");
        assert!((reconstructed[0] - rhs[0]).abs() <= 1.0e-12);
        assert!((reconstructed[1] - rhs[1]).abs() <= 1.0e-12);
    }

    #[test]
    fn singular_and_malformed_systems_fail_closed() {
        assert!(solve(&[vec![1.0, 2.0]], &[1.0]).is_err());
        assert!(solve(&[vec![1.0, 2.0], vec![2.0, 4.0]], &[1.0, 2.0]).is_err());
        assert!(apply(&[vec![1.0]], &[1.0, 2.0]).is_err());
    }
}
