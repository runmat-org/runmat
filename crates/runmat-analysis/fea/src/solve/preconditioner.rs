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
            if !admissible_positive_pivot(diagonal) {
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
                if !admissible_positive_pivot(pivot) {
                    return Err(PreconditionerError::InvalidPivot { row: pivot_row });
                }
                let multiplier = factors[lower_entry] / pivot;
                if !multiplier.is_finite() {
                    return Err(PreconditionerError::InvalidPivot { row: pivot_row });
                }
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
            if !admissible_positive_pivot(pivot) {
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

fn admissible_positive_pivot(pivot: f64) -> bool {
    pivot > 0.0 && pivot.is_finite() && pivot.recip().is_finite()
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
        // PCG requires an SPD preconditioner. Accepted positive pivots define
        // M = L D Lᵀ; use that symmetric factor explicitly rather than the
        // independently rounded upper ILU entries. Back-substitution propagates
        // through the stored strict-lower rows in place, so no transpose copy
        // or additional full-size vector is retained.
        let mut result = y;
        for (row, value) in result.iter_mut().enumerate() {
            *value /= self.factors[self.diagonal_entries[row]];
        }
        for row in (0..n).rev() {
            let pivot_value = result[row];
            for entry in self.row_offsets[row]..self.diagonal_entries[row] {
                result[self.column_indices[entry]] -= self.factors[entry] * pivot_value;
            }
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
        let residual = crate::algebra::spmv(system.matrix(), &applied).unwrap();
        for (actual, expected) in residual.iter().zip(system.rhs()) {
            assert!((actual - expected).abs() <= 1.0e-12);
        }
        assert!(
            system
                .rhs()
                .iter()
                .zip(&applied)
                .map(|(left, right)| left * right)
                .sum::<f64>()
                > 0.0
        );
    }

    #[test]
    fn ilu0_rejects_negative_incomplete_pivot_on_a_coercive_operator() {
        // A is independently Cholesky-positive, but its zero-fill incomplete
        // factorization develops a negative final pivot. Absolute-value pivot
        // admission would incorrectly pass an indefinite CG preconditioner.
        let values = [
            [
                2.853656491730362,
                0.0,
                1.7390443217009306,
                1.6154918987303972,
                0.6468259897083044,
            ],
            [0.0, 4.752237345324829, 0.0, 2.2926174104213715, 0.0],
            [
                1.7390443217009306,
                0.0,
                5.047566675581038,
                0.0,
                -2.5766232535243034,
            ],
            [
                1.6154918987303972,
                2.2926174104213715,
                0.0,
                3.101011391496286,
                1.4569321684539318,
            ],
            [
                0.6468259897083044,
                0.0,
                -2.5766232535243034,
                1.4569321684539318,
                3.166770219337195,
            ],
        ];
        let mut assembler = RealSparseAssembler::square(
            5,
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        );
        for (row, entries) in values.iter().enumerate() {
            for (column, value) in entries.iter().copied().enumerate() {
                if value != 0.0 {
                    assembler.add(row, column, value);
                }
            }
        }
        let system = PreparedRealLinearSystem::try_new(
            assembler.finish().unwrap(),
            vec![1.0; 5],
            vec![false; 5],
        )
        .expect("the full sparse operator is coercive");
        assert!(matches!(
            build_spd_preconditioner(&system, SpdPreconditionerKind::Ilu0),
            Err(PreconditionerError::InvalidPivot { row: 4 })
        ));
    }

    #[test]
    fn ilu0_applies_a_symmetric_positive_preconditioner_with_missing_fill() {
        let mut assembler = RealSparseAssembler::square(
            4,
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        );
        for row in 0..4 {
            assembler.add(row, row, 6.0);
            let next = (row + 1) % 4;
            assembler.add(row, next, -1.0);
            assembler.add(next, row, -1.0);
        }
        let system = PreparedRealLinearSystem::try_new(
            assembler.finish().unwrap(),
            vec![0.0; 4],
            vec![false; 4],
        )
        .unwrap();
        let preconditioner =
            build_spd_preconditioner(&system, SpdPreconditionerKind::Ilu0).unwrap();
        let inverse_columns = (0..4)
            .map(|column| {
                let mut basis = vec![0.0; 4];
                basis[column] = 1.0;
                preconditioner.apply(&basis)
            })
            .collect::<Vec<_>>();
        for row in 0..4 {
            for column in 0..4 {
                assert!(
                    (inverse_columns[column][row] - inverse_columns[row][column]).abs() < 1.0e-14
                );
            }
        }
        let vector = [1.0, -2.0, 0.5, 3.0];
        let applied = preconditioner.apply(&vector);
        assert!(vector.iter().zip(applied).map(|(a, b)| a * b).sum::<f64>() > 0.0);
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
