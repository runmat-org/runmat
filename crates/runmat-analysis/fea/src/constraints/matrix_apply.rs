//! Sparse application of canonical constraint matrices and their transposes.

use super::{ConstraintAlgebraError, ResolvedConstraintMatrix};

impl ResolvedConstraintMatrix {
    pub fn apply(&self, full: &[f64]) -> Result<Vec<f64>, ConstraintAlgebraError> {
        self.validate()?;
        if full.len() != self.full_dof_count {
            return Err(ConstraintAlgebraError::OperatorDimensionMismatch);
        }
        Ok(self.apply_validated(full))
    }

    pub(crate) fn apply_validated(&self, full: &[f64]) -> Vec<f64> {
        (0..self.row_count())
            .map(|row| {
                let start = self.row_offsets[row];
                let end = self.row_offsets[row + 1];
                (start..end)
                    .map(|entry| self.coefficients[entry] * full[self.column_indices[entry]])
                    .sum()
            })
            .collect()
    }

    pub fn transpose_apply(&self, rows: &[f64]) -> Result<Vec<f64>, ConstraintAlgebraError> {
        self.validate()?;
        if rows.len() != self.row_count() {
            return Err(ConstraintAlgebraError::OperatorDimensionMismatch);
        }
        Ok(self.transpose_apply_validated(rows))
    }

    pub(crate) fn transpose_apply_validated(&self, rows: &[f64]) -> Vec<f64> {
        let mut full = vec![0.0; self.full_dof_count];
        for (row, value) in rows.iter().copied().enumerate() {
            let start = self.row_offsets[row];
            let end = self.row_offsets[row + 1];
            for entry in start..end {
                full[self.column_indices[entry]] += self.coefficients[entry] * value;
            }
        }
        full
    }
}
