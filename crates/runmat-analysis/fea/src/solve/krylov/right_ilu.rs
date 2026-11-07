//! Fixed-support nonsymmetric ILU(0), applied on the right of a general operator.
//!
//! Genuine `L U` with no fill, permutation, pivot modification, dimensional floor or fallback,
//! computed by the shared sparse layer (`Ilu0::strict`).

mod envelope;
use crate::algebra::RealSparseMatrix;
pub(crate) use envelope::SparseRightIluEnvelope;
use thiserror::Error;

/// Strict fixed-support ILU(0) from the shared sparse layer.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SparseRightIlu {
    factor: runmat_sparse_solvers::precond::Ilu0<f64>,
    dimension: usize,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub(crate) enum SparseRightIluError {
    #[error("right ILU requires a square sparse operator")]
    NonSquare,
    #[error("right ILU is missing the diagonal at row {row}")]
    MissingDiagonal { row: usize },
    #[error("right ILU pivot at row {row} is not finite and invertible")]
    InvalidPivot { row: usize },
    #[error("right ILU factor at entry {entry} is nonfinite")]
    NonfiniteFactor { entry: usize },
    #[error("right ILU vector dimension differs from its factor dimension")]
    DimensionMismatch,
    #[error("right ILU input at row {row} is nonfinite")]
    NonfiniteInput { row: usize },
    #[error("right ILU output at row {row} is nonfinite")]
    NonfiniteOutput { row: usize },
    #[error("right ILU operation cancelled")]
    Cancelled,
    #[error("right ILU numerical usage: {0}")]
    Usage(runmat_simulation_usage::LedgerError),
}

fn check_cancelled(cancelled: &impl Fn() -> bool) -> Result<(), SparseRightIluError> {
    if cancelled() {
        Err(SparseRightIluError::Cancelled)
    } else {
        Ok(())
    }
}

impl SparseRightIlu {
    /// Factors on the exact support of `matrix` (no fill, permutation, pivot modification or
    /// fallback).
    pub(crate) fn factor(
        matrix: &RealSparseMatrix,
        cancelled: impl Fn() -> bool,
    ) -> Result<Self, SparseRightIluError> {
        check_cancelled(&cancelled)?;
        if matrix.rows() != matrix.columns() {
            return Err(SparseRightIluError::NonSquare);
        }
        let arrays = crate::solve::shared_sparse::CsrArrays::of(matrix);
        let factor =
            runmat_sparse_solvers::precond::Ilu0::strict(arrays.view()).map_err(|error| {
                use runmat_sparse_solvers::SolverError;
                match error {
                    SolverError::Usage(error) => SparseRightIluError::Usage(error),
                    SolverError::MissingDiagonal { row } => {
                        SparseRightIluError::MissingDiagonal { row }
                    }
                    SolverError::Singular { pivot } => {
                        SparseRightIluError::InvalidPivot { row: pivot }
                    }
                    SolverError::NonFinite { entry } => {
                        SparseRightIluError::NonfiniteFactor { entry }
                    }
                    _ => SparseRightIluError::NonSquare,
                }
            })?;
        check_cancelled(&cancelled)?;
        Ok(Self {
            factor,
            dimension: matrix.rows(),
        })
    }

    #[cfg(test)]
    pub(crate) fn factors(&self) -> Vec<f64> {
        self.factor.values().to_vec()
    }
    #[cfg(test)]
    pub(crate) fn row_offsets(&self) -> Vec<usize> {
        self.factor.row_ptr().to_vec()
    }
    #[cfg(test)]
    pub(crate) fn column_indices(&self) -> Vec<usize> {
        self.factor.col_idx().to_vec()
    }
    #[cfg(test)]
    pub(crate) fn retained_bytes(&self) -> u64 {
        self.factor.retained_bytes()
    }

    /// Computes `U⁻¹ L⁻¹ input`. For right GMRES apply `A M⁻¹ y`; recover `x = M⁻¹ y`.
    /// Original physical residual/power acceptance remains the caller's duty.
    pub(crate) fn apply(
        &self,
        input: &[f64],
        cancelled: impl Fn() -> bool,
    ) -> Result<Vec<f64>, SparseRightIluError> {
        use runmat_sparse_solvers::Preconditioner;
        check_cancelled(&cancelled)?;
        if input.len() != self.dimension {
            return Err(SparseRightIluError::DimensionMismatch);
        }
        if let Some(row) = input.iter().position(|value| !value.is_finite()) {
            return Err(SparseRightIluError::NonfiniteInput { row });
        }
        let mut output = vec![0.0; input.len()];
        self.factor
            .apply(input, &mut output)
            .map_err(|error| match error {
                runmat_sparse_solvers::SolverError::Usage(error) => {
                    SparseRightIluError::Usage(error)
                }
                _ => SparseRightIluError::DimensionMismatch,
            })?;
        if let Some(row) = output.iter().position(|value| !value.is_finite()) {
            return Err(SparseRightIluError::NonfiniteOutput { row });
        }
        check_cancelled(&cancelled)?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests;
