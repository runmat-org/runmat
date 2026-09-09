use std::{collections::BTreeMap, fmt};

use crate::algebra::PreparedRealLinearSystem;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpdPreconditionerKind {
    Jacobi,
    Ilu0,
}

impl SpdPreconditionerKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Jacobi => "jacobi",
            Self::Ilu0 => "ilu0",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreconditionerError {
    MissingDiagonal { row: usize },
    InvalidPivot { row: usize },
}

impl fmt::Display for PreconditionerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "preconditioner setup failed: {self:?}")
    }
}

impl std::error::Error for PreconditionerError {}

pub trait SpdPreconditioner {
    fn kind(&self) -> SpdPreconditionerKind;
    fn apply(&self, residual: &[f64]) -> Vec<f64>;
}

struct JacobiPreconditioner {
    inverse_diagonal: Vec<f64>,
}

impl JacobiPreconditioner {
    fn try_new(system: &PreparedRealLinearSystem) -> Result<Self, PreconditionerError> {
        let mut inverse_diagonal = Vec::with_capacity(system.dof_count());
        for row in 0..system.dof_count() {
            let diagonal = system.matrix().get(row, row);
            if !diagonal.is_finite() {
                return Err(PreconditionerError::InvalidPivot { row });
            }
            if diagonal.abs() <= 1.0e-14 {
                let row_is_empty = system.matrix().pattern().row_range(row).is_empty();
                if row_is_empty && system.rhs()[row] == 0.0 {
                    inverse_diagonal.push(1.0);
                    continue;
                }
                return Err(PreconditionerError::InvalidPivot { row });
            }
            inverse_diagonal.push(diagonal.recip());
        }
        Ok(Self { inverse_diagonal })
    }
}

impl SpdPreconditioner for JacobiPreconditioner {
    fn kind(&self) -> SpdPreconditionerKind {
        SpdPreconditionerKind::Jacobi
    }
    fn apply(&self, residual: &[f64]) -> Vec<f64> {
        residual
            .iter()
            .zip(&self.inverse_diagonal)
            .map(|(value, inverse)| value * inverse)
            .collect()
    }
}

struct Ilu0Preconditioner {
    row_offsets: Vec<usize>,
    column_indices: Vec<usize>,
    factors: Vec<f64>,
    diagonal_entries: Vec<usize>,
}

impl Ilu0Preconditioner {
    fn factor(system: &PreparedRealLinearSystem) -> Result<Self, PreconditionerError> {
        let matrix = system.matrix();
        let n = matrix.rows();
        let row_offsets = matrix
            .pattern()
            .row_offsets()
            .iter()
            .map(|value| *value as usize)
            .collect::<Vec<_>>();
        let column_indices = matrix
            .pattern()
            .column_indices()
            .iter()
            .map(|value| *value as usize)
            .collect::<Vec<_>>();
        let mut factors = matrix.values().to_vec();
        let positions = (0..n)
            .map(|row| {
                matrix
                    .pattern()
                    .row_range(row)
                    .map(|entry| (matrix.pattern().column(entry), entry))
                    .collect::<BTreeMap<_, _>>()
            })
            .collect::<Vec<_>>();
        let diagonal_entries = positions
            .iter()
            .enumerate()
            .map(|(row, entries)| {
                entries
                    .get(&row)
                    .copied()
                    .ok_or(PreconditionerError::MissingDiagonal { row })
            })
            .collect::<Result<Vec<_>, _>>()?;

        for row in 0..n {
            let lower_entries = matrix
                .pattern()
                .row_range(row)
                .take_while(|entry| matrix.pattern().column(*entry) < row)
                .collect::<Vec<_>>();
            for lower_entry in lower_entries {
                let pivot_row = column_indices[lower_entry];
                let pivot = factors[diagonal_entries[pivot_row]];
                if !pivot.is_finite() || pivot.abs() <= 1.0e-14 {
                    return Err(PreconditionerError::InvalidPivot { row: pivot_row });
                }
                let multiplier = factors[lower_entry] / pivot;
                factors[lower_entry] = multiplier;
                for upper_entry in matrix.pattern().row_range(pivot_row) {
                    let column = column_indices[upper_entry];
                    if column <= pivot_row {
                        continue;
                    }
                    if let Some(&target) = positions[row].get(&column) {
                        factors[target] -= multiplier * factors[upper_entry];
                    }
                }
            }
            let pivot = factors[diagonal_entries[row]];
            if !pivot.is_finite() || pivot.abs() <= 1.0e-14 {
                return Err(PreconditionerError::InvalidPivot { row });
            }
        }
        Ok(Self {
            row_offsets,
            column_indices,
            factors,
            diagonal_entries,
        })
    }
}

impl SpdPreconditioner for Ilu0Preconditioner {
    fn kind(&self) -> SpdPreconditionerKind {
        SpdPreconditionerKind::Ilu0
    }
    fn apply(&self, residual: &[f64]) -> Vec<f64> {
        let n = self.diagonal_entries.len();
        let mut y = vec![0.0; n];
        for row in 0..n {
            let correction = (self.row_offsets[row]..self.diagonal_entries[row])
                .map(|entry| self.factors[entry] * y[self.column_indices[entry]])
                .sum::<f64>();
            y[row] = residual[row] - correction;
        }
        let mut result = vec![0.0; n];
        for row in (0..n).rev() {
            let correction = (self.diagonal_entries[row] + 1..self.row_offsets[row + 1])
                .map(|entry| self.factors[entry] * result[self.column_indices[entry]])
                .sum::<f64>();
            result[row] = (y[row] - correction) / self.factors[self.diagonal_entries[row]];
        }
        result
    }
}

pub fn build_spd_preconditioner(
    system: &PreparedRealLinearSystem,
    kind: SpdPreconditionerKind,
) -> Result<Box<dyn SpdPreconditioner>, PreconditionerError> {
    match kind {
        SpdPreconditionerKind::Jacobi => Ok(Box::new(JacobiPreconditioner::try_new(system)?)),
        SpdPreconditionerKind::Ilu0 => Ok(Box::new(Ilu0Preconditioner::factor(system)?)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{
        MatrixProperties, OperatorRole, PreparedRealLinearSystem, RealSparseAssembler,
        RealSparseMatrix,
    };

    #[test]
    fn ilu0_uses_canonical_sparse_pattern_and_is_finite() {
        let mut assembler = RealSparseAssembler::square(
            3,
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        );
        for row in 0..3 {
            assembler.add(row, row, 4.0);
            if row + 1 < 3 {
                assembler.add(row, row + 1, -1.0);
                assembler.add(row + 1, row, -1.0);
            }
        }
        let matrix = assembler.finish().unwrap();
        let system =
            PreparedRealLinearSystem::try_new(matrix, vec![1.0; 3], vec![false; 3]).unwrap();
        let applied = build_spd_preconditioner(&system, SpdPreconditionerKind::Ilu0)
            .unwrap()
            .apply(system.rhs());
        assert!(applied.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn missing_pivot_is_rejected_by_solver_admission() {
        let matrix = RealSparseMatrix::diagonal(
            vec![1.0, 0.0],
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        )
        .unwrap();
        assert!(matches!(
            PreparedRealLinearSystem::try_new(matrix, vec![1.0; 2], vec![false; 2]),
            Err(crate::algebra::RealSystemError::Coercivity(
                crate::algebra::CoercivityError::NonPositivePivot { pivot: 1 }
            ))
        ));
    }
}
