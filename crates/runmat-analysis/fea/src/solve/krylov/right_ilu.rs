//! Fixed-support nonsymmetric ILU(0), applied on the right of a general operator.
//!
//! This is genuine `L U`, not the positive-pivot `L D Lᵀ` PCG preconditioner.
//! No fill, permutation, pivot modification, dimensional floor or fallback is used.

use crate::algebra::RealSparseMatrix;
use thiserror::Error;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SparseRightIlu {
    row_offsets: Vec<usize>,
    column_indices: Vec<usize>,
    factors: Vec<f64>,
    diagonal_entries: Vec<usize>,
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
}

fn check_cancelled(cancelled: &impl Fn() -> bool) -> Result<(), SparseRightIluError> {
    if cancelled() {
        Err(SparseRightIluError::Cancelled)
    } else {
        Ok(())
    }
}

fn check_pivot(value: f64, row: usize) -> Result<(), SparseRightIluError> {
    let inverse = value.recip();
    if !value.is_finite() || value == 0.0 || !inverse.is_finite() || inverse == 0.0 {
        Err(SparseRightIluError::InvalidPivot { row })
    } else {
        Ok(())
    }
}

impl SparseRightIlu {
    /// Retains exactly four heap vectors: offsets `n+1`, columns `nnz`, factors
    /// `nnz`, diagonal positions `n`. Setup has no auxiliary heap vectors/maps;
    /// CSR binary search locates retained updates. The source CSR coexists with
    /// this storage. Its validated sorted pattern is never changed.
    pub(crate) fn factor(
        matrix: &RealSparseMatrix,
        cancelled: impl Fn() -> bool,
    ) -> Result<Self, SparseRightIluError> {
        check_cancelled(&cancelled)?;
        if matrix.rows() != matrix.columns() {
            return Err(SparseRightIluError::NonSquare);
        }
        let n = matrix.rows();
        let mut result = Self {
            row_offsets: matrix
                .pattern()
                .row_offsets()
                .iter()
                .map(|&v| v as usize)
                .collect(),
            column_indices: matrix
                .pattern()
                .column_indices()
                .iter()
                .map(|&v| v as usize)
                .collect(),
            factors: matrix.values().to_vec(),
            diagonal_entries: Vec::with_capacity(n),
        };
        for row in 0..n {
            check_cancelled(&cancelled)?;
            let start = result.row_offsets[row];
            let end = result.row_offsets[row + 1];
            let diagonal = result.column_indices[start..end]
                .binary_search(&row)
                .map_err(|_| SparseRightIluError::MissingDiagonal { row })?;
            result.diagonal_entries.push(start + diagonal);
        }
        for row in 0..n {
            check_cancelled(&cancelled)?;
            let start = result.row_offsets[row];
            let end = result.row_offsets[row + 1];
            for lower in start..result.diagonal_entries[row] {
                check_cancelled(&cancelled)?;
                let pivot_row = result.column_indices[lower];
                let pivot = result.factors[result.diagonal_entries[pivot_row]];
                check_pivot(pivot, pivot_row)?;
                let multiplier = result.factors[lower] / pivot;
                if !multiplier.is_finite() {
                    return Err(SparseRightIluError::NonfiniteFactor { entry: lower });
                }
                result.factors[lower] = multiplier;
                for upper in
                    result.diagonal_entries[pivot_row] + 1..result.row_offsets[pivot_row + 1]
                {
                    check_cancelled(&cancelled)?;
                    let column = result.column_indices[upper];
                    if let Ok(offset) = result.column_indices[start..end].binary_search(&column) {
                        let target = start + offset;
                        let product = multiplier * result.factors[upper];
                        let updated = result.factors[target] - product;
                        if !product.is_finite() || !updated.is_finite() {
                            return Err(SparseRightIluError::NonfiniteFactor { entry: target });
                        }
                        result.factors[target] = updated;
                    }
                }
            }
            check_pivot(result.factors[result.diagonal_entries[row]], row)?;
        }
        check_cancelled(&cancelled)?;
        Ok(result)
    }

    /// Computes `U⁻¹ L⁻¹ input` with one `n`-value output allocation, reused
    /// in both sweeps. For right GMRES apply `A M⁻¹ y`; recover `x=M⁻¹ y`.
    /// Original physical residual/power acceptance remains the caller's duty.
    pub(crate) fn apply(
        &self,
        input: &[f64],
        cancelled: impl Fn() -> bool,
    ) -> Result<Vec<f64>, SparseRightIluError> {
        check_cancelled(&cancelled)?;
        if input.len() != self.diagonal_entries.len() {
            return Err(SparseRightIluError::DimensionMismatch);
        }
        for (row, value) in input.iter().enumerate() {
            check_cancelled(&cancelled)?;
            if !value.is_finite() {
                return Err(SparseRightIluError::NonfiniteInput { row });
            }
        }
        let mut output = input.to_vec();
        for row in 0..output.len() {
            check_cancelled(&cancelled)?;
            for entry in self.row_offsets[row]..self.diagonal_entries[row] {
                check_cancelled(&cancelled)?;
                let product = self.factors[entry] * output[self.column_indices[entry]];
                output[row] -= product;
                if !output[row].is_finite() {
                    return Err(SparseRightIluError::NonfiniteOutput { row });
                }
            }
        }
        for row in (0..output.len()).rev() {
            check_cancelled(&cancelled)?;
            for entry in self.diagonal_entries[row] + 1..self.row_offsets[row + 1] {
                check_cancelled(&cancelled)?;
                let product = self.factors[entry] * output[self.column_indices[entry]];
                output[row] -= product;
                if !output[row].is_finite() {
                    return Err(SparseRightIluError::NonfiniteOutput { row });
                }
            }
            output[row] /= self.factors[self.diagonal_entries[row]];
            if !output[row].is_finite() {
                return Err(SparseRightIluError::NonfiniteOutput { row });
            }
        }
        check_cancelled(&cancelled)?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests;
