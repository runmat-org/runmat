mod sparse_operator;
use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_core::{StructuralRotationDof, StructuralTranslationDof};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use runmat_meshing_core::StableDigest;

use crate::algebra::{
    spmv, RealSparseAssembler, RealSparseMatrix, RealSparseMatrixError, SparsePattern,
};

use super::{
    CanonicalStructuralConstraintPlan, CanonicalStructuralDof, CanonicalStructuralDofOwner,
    ConstraintPlanError, ReactionRecovery,
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedConstraintMatrix {
    pub full_dof_count: usize,
    pub row_offsets: Vec<usize>,
    pub column_indices: Vec<usize>,
    pub coefficients: Vec<f64>,
    pub right_hand_side: Vec<f64>,
    pub row_identities: Vec<Digest>,
    pub constraint_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AffineConstraintReduction {
    pub full_dof_count: usize,
    pub reduced_dof_count: usize,
    pub free_dof_indices: Vec<usize>,
    pub pivot_dof_indices: Vec<usize>,
    pub recovery_row_offsets: Vec<usize>,
    pub recovery_column_indices: Vec<usize>,
    pub recovery_coefficients: Vec<f64>,
    pub offset: Vec<f64>,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum ConstraintAlgebraError {
    #[error("constraint plan is invalid: {0}")]
    InvalidPlan(String),
    #[error("constraint row `{constraint_id}` references an unavailable solver DOF")]
    UnavailableDof { constraint_id: String },
    #[error("resolved constraint matrix is malformed: {0}")]
    InvalidMatrix(String),
    #[error("constraint matrix is rank deficient after canonical compilation")]
    RankDeficient,
    #[error("operator dimensions do not match the affine constraint reduction")]
    OperatorDimensionMismatch,
    #[error("reduced solution dimensions do not match the affine constraint reduction")]
    SolutionDimensionMismatch,
}

impl ResolvedConstraintMatrix {
    /// Resolves the canonical solid-mechanics numbering shared by planning and assembly:
    /// three translational DOFs per mesh node followed by six DOFs per reference point.
    pub fn from_solid_plan(
        plan: &CanonicalStructuralConstraintPlan,
        node_identities: &[StableDigest],
    ) -> Result<Self, ConstraintAlgebraError> {
        let mesh_dofs = node_identities.len().checked_mul(3).ok_or_else(|| {
            ConstraintAlgebraError::InvalidMatrix("solid DOF count overflowed".into())
        })?;
        let reference_dofs = plan.reference_points.len().checked_mul(6).ok_or_else(|| {
            ConstraintAlgebraError::InvalidMatrix("reference-point DOF count overflowed".into())
        })?;
        let full_dof_count = mesh_dofs.checked_add(reference_dofs).ok_or_else(|| {
            ConstraintAlgebraError::InvalidMatrix("solid DOF count overflowed".into())
        })?;
        let node_indices = node_identities
            .iter()
            .copied()
            .enumerate()
            .map(|(index, identity)| (identity, index))
            .collect::<BTreeMap<_, _>>();
        if node_indices.len() != node_identities.len() {
            return Err(ConstraintAlgebraError::InvalidMatrix(
                "solid node identities are not unique".into(),
            ));
        }
        let reference_indices = plan
            .reference_points
            .iter()
            .enumerate()
            .map(|(index, point)| (point.identity, index))
            .collect::<BTreeMap<_, _>>();
        Self::from_plan(plan, full_dof_count, |owner, dof| {
            let component = match dof {
                CanonicalStructuralDof::Translation(StructuralTranslationDof::Ux) => 0,
                CanonicalStructuralDof::Translation(StructuralTranslationDof::Uy) => 1,
                CanonicalStructuralDof::Translation(StructuralTranslationDof::Uz) => 2,
                CanonicalStructuralDof::Rotation(StructuralRotationDof::Rx) => 3,
                CanonicalStructuralDof::Rotation(StructuralRotationDof::Ry) => 4,
                CanonicalStructuralDof::Rotation(StructuralRotationDof::Rz) => 5,
            };
            match owner {
                CanonicalStructuralDofOwner::StructuralNode(identity) => {
                    (component < 3).then(|| {
                        node_indices
                            .get(&identity)
                            .copied()?
                            .checked_mul(3)?
                            .checked_add(component)
                    })?
                }
                CanonicalStructuralDofOwner::ReferencePoint(identity) => reference_indices
                    .get(&identity)
                    .copied()?
                    .checked_mul(6)?
                    .checked_add(mesh_dofs)?
                    .checked_add(component),
            }
        })
    }

    pub fn from_plan(
        plan: &CanonicalStructuralConstraintPlan,
        full_dof_count: usize,
        mut resolve: impl FnMut(CanonicalStructuralDofOwner, CanonicalStructuralDof) -> Option<usize>,
    ) -> Result<Self, ConstraintAlgebraError> {
        plan.validate()
            .map_err(|error| ConstraintAlgebraError::InvalidPlan(error.to_string()))?;
        let active_rows = plan
            .rows
            .iter()
            .filter(|row| row.reaction_recovery != ReactionRecovery::RedundantEquation);
        let mut matrix = Self {
            full_dof_count,
            row_offsets: vec![0],
            column_indices: Vec::new(),
            coefficients: Vec::new(),
            right_hand_side: Vec::new(),
            row_identities: Vec::new(),
            constraint_ids: Vec::new(),
        };
        for row in active_rows {
            let mut coefficients = BTreeMap::<usize, f64>::new();
            for term in &row.terms {
                let column = resolve(term.owner, term.dof).ok_or_else(|| {
                    ConstraintAlgebraError::UnavailableDof {
                        constraint_id: row.constraint_id.clone(),
                    }
                })?;
                if column >= full_dof_count {
                    return Err(ConstraintAlgebraError::UnavailableDof {
                        constraint_id: row.constraint_id.clone(),
                    });
                }
                *coefficients.entry(column).or_insert(0.0) += term.coefficient;
            }
            coefficients.retain(|_, value| *value != 0.0);
            if coefficients.is_empty() {
                return Err(ConstraintAlgebraError::InvalidMatrix(format!(
                    "constraint `{}` resolves to an empty row",
                    row.constraint_id
                )));
            }
            for (column, coefficient) in coefficients {
                matrix.column_indices.push(column);
                matrix.coefficients.push(coefficient);
            }
            matrix.row_offsets.push(matrix.column_indices.len());
            matrix.right_hand_side.push(row.right_hand_side);
            matrix.row_identities.push(row.identity);
            matrix.constraint_ids.push(row.constraint_id.clone());
        }
        matrix.validate()?;
        Ok(matrix)
    }

    pub fn row_count(&self) -> usize {
        self.right_hand_side.len()
    }

    pub fn validate(&self) -> Result<(), ConstraintAlgebraError> {
        if self.row_offsets.len() != self.row_count() + 1
            || self.row_offsets.first() != Some(&0)
            || self.row_offsets.last().copied() != Some(self.coefficients.len())
            || self.column_indices.len() != self.coefficients.len()
            || self.row_identities.len() != self.row_count()
            || self.constraint_ids.len() != self.row_count()
            || self.coefficients.iter().any(|value| !value.is_finite())
            || self.right_hand_side.iter().any(|value| !value.is_finite())
            || self.constraint_ids.iter().any(String::is_empty)
        {
            return Err(ConstraintAlgebraError::InvalidMatrix(
                "shape, identity, or numerical content is inconsistent".into(),
            ));
        }
        for row in 0..self.row_count() {
            let start = self.row_offsets[row];
            let end = self.row_offsets[row + 1];
            if start >= end
                || end > self.coefficients.len()
                || self.column_indices[start..end]
                    .iter()
                    .any(|column| *column >= self.full_dof_count)
                || !self.column_indices[start..end]
                    .windows(2)
                    .all(|pair| pair[0] < pair[1])
            {
                return Err(ConstraintAlgebraError::InvalidMatrix(format!(
                    "row {row} is empty, unordered, or out of bounds"
                )));
            }
        }
        Ok(())
    }

    pub fn affine_reduction(&self) -> Result<AffineConstraintReduction, ConstraintAlgebraError> {
        self.validate()?;
        let mut basis = Vec::<(usize, BTreeMap<usize, f64>, f64)>::new();
        for row in 0..self.row_count() {
            let start = self.row_offsets[row];
            let end = self.row_offsets[row + 1];
            let mut coefficients = self.column_indices[start..end]
                .iter()
                .copied()
                .zip(self.coefficients[start..end].iter().copied())
                .collect::<BTreeMap<_, _>>();
            let mut right_hand_side = self.right_hand_side[row];
            for (pivot, existing, existing_rhs) in &basis {
                let Some(factor) = coefficients.remove(pivot) else {
                    continue;
                };
                for (column, coefficient) in existing {
                    if column != pivot {
                        *coefficients.entry(*column).or_insert(0.0) -= factor * coefficient;
                    }
                }
                right_hand_side -= factor * existing_rhs;
                coefficients.retain(|_, coefficient| coefficient.abs() > 1.0e-11);
            }
            let Some((&pivot, &pivot_coefficient)) = coefficients.first_key_value() else {
                return Err(ConstraintAlgebraError::RankDeficient);
            };
            for coefficient in coefficients.values_mut() {
                *coefficient /= pivot_coefficient;
            }
            right_hand_side /= pivot_coefficient;
            basis.push((pivot, coefficients, right_hand_side));
        }

        for index in (0..basis.len()).rev() {
            let (right, left) = basis.split_at_mut(index);
            let (pivot, coefficients, right_hand_side) = &left[0];
            for (_, previous, previous_rhs) in right {
                let Some(factor) = previous.remove(pivot) else {
                    continue;
                };
                for (column, coefficient) in coefficients {
                    if column != pivot {
                        *previous.entry(*column).or_insert(0.0) -= factor * coefficient;
                    }
                }
                *previous_rhs -= factor * right_hand_side;
                previous.retain(|_, coefficient| coefficient.abs() > 1.0e-11);
            }
        }

        let pivot_dof_indices = basis.iter().map(|(pivot, _, _)| *pivot).collect::<Vec<_>>();
        let pivots = pivot_dof_indices.iter().copied().collect::<BTreeSet<_>>();
        let free_dof_indices = (0..self.full_dof_count)
            .filter(|dof| !pivots.contains(dof))
            .collect::<Vec<_>>();
        let free_columns = free_dof_indices
            .iter()
            .enumerate()
            .map(|(column, dof)| (*dof, column))
            .collect::<BTreeMap<_, _>>();
        let pivot_rows = basis
            .iter()
            .enumerate()
            .map(|(row, (pivot, _, _))| (*pivot, row))
            .collect::<BTreeMap<_, _>>();
        let mut reduction = AffineConstraintReduction {
            full_dof_count: self.full_dof_count,
            reduced_dof_count: free_dof_indices.len(),
            free_dof_indices,
            pivot_dof_indices,
            recovery_row_offsets: vec![0],
            recovery_column_indices: Vec::new(),
            recovery_coefficients: Vec::new(),
            offset: vec![0.0; self.full_dof_count],
        };
        for full_dof in 0..self.full_dof_count {
            if let Some(&column) = free_columns.get(&full_dof) {
                reduction.recovery_column_indices.push(column);
                reduction.recovery_coefficients.push(1.0);
            } else if let Some(&basis_row) = pivot_rows.get(&full_dof) {
                let (_, coefficients, right_hand_side) = &basis[basis_row];
                reduction.offset[full_dof] = *right_hand_side;
                for (&dof, &coefficient) in coefficients {
                    if dof == full_dof {
                        continue;
                    }
                    let Some(&column) = free_columns.get(&dof) else {
                        return Err(ConstraintAlgebraError::RankDeficient);
                    };
                    reduction.recovery_column_indices.push(column);
                    reduction.recovery_coefficients.push(-coefficient);
                }
            }
            reduction
                .recovery_row_offsets
                .push(reduction.recovery_column_indices.len());
        }
        reduction.validate()?;
        Ok(reduction)
    }
}

impl AffineConstraintReduction {
    pub(crate) fn from_constrained_dofs(constrained: &[bool]) -> Self {
        let free_dof_indices = constrained
            .iter()
            .enumerate()
            .filter_map(|(dof, constrained)| (!constrained).then_some(dof))
            .collect::<Vec<_>>();
        let pivot_dof_indices = constrained
            .iter()
            .enumerate()
            .filter_map(|(dof, constrained)| constrained.then_some(dof))
            .collect::<Vec<_>>();
        let free_columns = free_dof_indices
            .iter()
            .copied()
            .enumerate()
            .map(|(column, dof)| (dof, column))
            .collect::<BTreeMap<_, _>>();
        let mut recovery_row_offsets = Vec::with_capacity(constrained.len() + 1);
        let mut recovery_column_indices = Vec::with_capacity(free_dof_indices.len());
        let mut recovery_coefficients = Vec::with_capacity(free_dof_indices.len());
        recovery_row_offsets.push(0);
        for dof in 0..constrained.len() {
            if let Some(column) = free_columns.get(&dof) {
                recovery_column_indices.push(*column);
                recovery_coefficients.push(1.0);
            }
            recovery_row_offsets.push(recovery_column_indices.len());
        }
        Self {
            full_dof_count: constrained.len(),
            reduced_dof_count: free_dof_indices.len(),
            free_dof_indices,
            pivot_dof_indices,
            recovery_row_offsets,
            recovery_column_indices,
            recovery_coefficients,
            offset: vec![0.0; constrained.len()],
        }
    }

    pub fn validate(&self) -> Result<(), ConstraintAlgebraError> {
        if self.reduced_dof_count != self.free_dof_indices.len()
            || self
                .pivot_dof_indices
                .len()
                .checked_add(self.reduced_dof_count)
                != Some(self.full_dof_count)
            || Some(self.recovery_row_offsets.len()) != self.full_dof_count.checked_add(1)
            || self.recovery_row_offsets.first() != Some(&0)
            || self.recovery_row_offsets.last().copied() != Some(self.recovery_coefficients.len())
            || self.recovery_column_indices.len() != self.recovery_coefficients.len()
            || self.offset.len() != self.full_dof_count
            || self.offset.iter().any(|value| !value.is_finite())
            || self
                .recovery_coefficients
                .iter()
                .any(|value| !value.is_finite())
            || self
                .recovery_column_indices
                .iter()
                .any(|column| *column >= self.reduced_dof_count)
        {
            return Err(ConstraintAlgebraError::InvalidMatrix(
                "affine recovery shape or numerical content is inconsistent".into(),
            ));
        }
        if self
            .recovery_row_offsets
            .windows(2)
            .any(|r| r[0] > r[1] || r[1] > self.recovery_coefficients.len())
        {
            return Err(ConstraintAlgebraError::InvalidMatrix(
                "affine recovery offsets are unordered or out of bounds".into(),
            ));
        }
        let mut partition = Vec::new();
        partition
            .try_reserve_exact(self.full_dof_count)
            .map_err(|_| {
                ConstraintAlgebraError::InvalidMatrix(
                    "affine partition allocation unavailable".into(),
                )
            })?;
        partition.extend_from_slice(&self.free_dof_indices);
        partition.extend_from_slice(&self.pivot_dof_indices);
        partition.sort_unstable();
        if partition
            .iter()
            .enumerate()
            .any(|(row, value)| row != *value)
        {
            return Err(ConstraintAlgebraError::InvalidMatrix(
                "free and pivot DOFs do not partition the full space".into(),
            ));
        }
        Ok(())
    }

    pub fn expand(&self, reduced: &[f64]) -> Result<Vec<f64>, ConstraintAlgebraError> {
        self.validate()?;
        if reduced.len() != self.reduced_dof_count {
            return Err(ConstraintAlgebraError::SolutionDimensionMismatch);
        }
        Ok((0..self.full_dof_count)
            .map(|row| {
                let start = self.recovery_row_offsets[row];
                let end = self.recovery_row_offsets[row + 1];
                self.offset[row]
                    + (start..end)
                        .map(|entry| {
                            self.recovery_coefficients[entry]
                                * reduced[self.recovery_column_indices[entry]]
                        })
                        .sum::<f64>()
            })
            .collect())
    }

    pub fn expand_linear(&self, reduced: &[f64]) -> Result<Vec<f64>, ConstraintAlgebraError> {
        let mut expanded = self.expand(reduced)?;
        for (value, offset) in expanded.iter_mut().zip(&self.offset) {
            *value -= offset;
        }
        Ok(expanded)
    }

    pub fn transpose_apply(&self, full: &[f64]) -> Result<Vec<f64>, ConstraintAlgebraError> {
        self.validate()?;
        if full.len() != self.full_dof_count {
            return Err(ConstraintAlgebraError::OperatorDimensionMismatch);
        }
        let mut reduced = vec![0.0; self.reduced_dof_count];
        for (row, value) in full.iter().copied().enumerate() {
            let start = self.recovery_row_offsets[row];
            let end = self.recovery_row_offsets[row + 1];
            for entry in start..end {
                reduced[self.recovery_column_indices[entry]] +=
                    self.recovery_coefficients[entry] * value;
            }
        }
        Ok(reduced)
    }

    /// Computes the material-independent sparsity upper bound for `T' * K * T` without
    /// allocating values or depending on cancellations in a particular assembled operator.
    pub fn reduce_sparse_pattern(
        &self,
        operator: &SparsePattern,
    ) -> Result<SparsePattern, ConstraintAlgebraError> {
        self.validate()?;
        if operator.rows() != self.full_dof_count || operator.columns() != self.full_dof_count {
            return Err(ConstraintAlgebraError::OperatorDimensionMismatch);
        }
        let plan = super::AffineSparseSupportPlan::prepare(
            self,
            operator,
            operator,
            super::AffineSupportBudget {
                maximum_resident_bytes: u64::MAX,
                maximum_scalar_operations: u64::MAX,
            },
        )
        .map_err(|e| ConstraintAlgebraError::InvalidMatrix(e.to_string()))?;
        plan.materialize()
            .map_err(|e| ConstraintAlgebraError::InvalidMatrix(e.to_string()))
    }

    pub fn reduce_dense_operator(
        &self,
        operator: &[f64],
        right_hand_side: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), ConstraintAlgebraError> {
        self.validate()?;
        if operator.len() != self.full_dof_count.saturating_mul(self.full_dof_count)
            || right_hand_side.len() != self.full_dof_count
        {
            return Err(ConstraintAlgebraError::OperatorDimensionMismatch);
        }
        let zero = vec![0.0; self.reduced_dof_count];
        let offset = self.expand(&zero)?;
        let mut shifted_rhs = right_hand_side.to_vec();
        for row in 0..self.full_dof_count {
            shifted_rhs[row] -= (0..self.full_dof_count)
                .map(|column| operator[row * self.full_dof_count + column] * offset[column])
                .sum::<f64>();
        }
        let mut reduced_operator = vec![0.0; self.reduced_dof_count * self.reduced_dof_count];
        let mut reduced_rhs = vec![0.0; self.reduced_dof_count];
        for full_row in 0..self.full_dof_count {
            let row_start = self.recovery_row_offsets[full_row];
            let row_end = self.recovery_row_offsets[full_row + 1];
            for left in row_start..row_end {
                let reduced_row = self.recovery_column_indices[left];
                let left_value = self.recovery_coefficients[left];
                reduced_rhs[reduced_row] += left_value * shifted_rhs[full_row];
                for full_column in 0..self.full_dof_count {
                    let column_start = self.recovery_row_offsets[full_column];
                    let column_end = self.recovery_row_offsets[full_column + 1];
                    for right in column_start..column_end {
                        let reduced_column = self.recovery_column_indices[right];
                        reduced_operator[reduced_row * self.reduced_dof_count + reduced_column] +=
                            left_value
                                * operator[full_row * self.full_dof_count + full_column]
                                * self.recovery_coefficients[right];
                    }
                }
            }
        }
        Ok((reduced_operator, reduced_rhs))
    }

    pub fn reduce_sparse_system(
        &self,
        operator: &RealSparseMatrix,
        right_hand_side: &[f64],
    ) -> Result<(RealSparseMatrix, Vec<f64>), ConstraintAlgebraError> {
        let reduced_rhs = self.reduce_rhs(operator, right_hand_side)?;
        Ok((self.reduce_sparse_operator(operator)?, reduced_rhs))
    }

    /// Reduces the system into a given reduced pattern (no value cancellation). With the reduced
    /// pattern resource planning analyzed, the reduced operator has exactly that pattern, so its
    /// factorization repeats the planned ordering, fill and work. Entries of `T' K T` outside the
    /// pattern are rejected by the scatter.
    pub fn reduce_sparse_system_on_support(
        &self,
        operator: &RealSparseMatrix,
        reduced_support: &SparsePattern,
        right_hand_side: &[f64],
    ) -> Result<(RealSparseMatrix, Vec<f64>), ConstraintAlgebraError> {
        let reduced_rhs = self.reduce_rhs(operator, right_hand_side)?;
        Ok((
            self.reduce_sparse_operator_planned(operator, reduced_support)?,
            reduced_rhs,
        ))
    }

    pub(crate) fn reduce_rhs(
        &self,
        operator: &RealSparseMatrix,
        right_hand_side: &[f64],
    ) -> Result<Vec<f64>, ConstraintAlgebraError> {
        self.validate()?;
        if operator.rows() != self.full_dof_count
            || operator.columns() != self.full_dof_count
            || right_hand_side.len() != self.full_dof_count
        {
            return Err(ConstraintAlgebraError::OperatorDimensionMismatch);
        }
        let applied_offset = spmv(operator, &self.offset)
            .map_err(|_| ConstraintAlgebraError::OperatorDimensionMismatch)?;
        let shifted_rhs = right_hand_side
            .iter()
            .zip(applied_offset)
            .map(|(applied, internal)| applied - internal)
            .collect::<Vec<_>>();
        self.transpose_apply(&shifted_rhs)
    }
}

fn sparse_reduction_error(error: RealSparseMatrixError) -> ConstraintAlgebraError {
    ConstraintAlgebraError::InvalidMatrix(format!("sparse affine reduction failed: {error}"))
}

impl From<ConstraintPlanError> for ConstraintAlgebraError {
    fn from(error: ConstraintPlanError) -> Self {
        Self::InvalidPlan(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{MatrixProperties, OperatorRole};

    fn matrix(
        full_dof_count: usize,
        rows: &[&[(usize, f64)]],
        right_hand_side: Vec<f64>,
    ) -> ResolvedConstraintMatrix {
        let mut row_offsets = vec![0];
        let mut column_indices = Vec::new();
        let mut coefficients = Vec::new();
        for row in rows {
            column_indices.extend(row.iter().map(|(column, _)| *column));
            coefficients.extend(row.iter().map(|(_, coefficient)| *coefficient));
            row_offsets.push(column_indices.len());
        }
        ResolvedConstraintMatrix {
            full_dof_count,
            row_offsets,
            column_indices,
            coefficients,
            right_hand_side,
            row_identities: (0..rows.len())
                .map(|row| Digest::sha256([row as u8 + 1]))
                .collect(),
            constraint_ids: (0..rows.len()).map(|row| format!("row_{row}")).collect(),
        }
    }

    #[test]
    fn affine_reduction_recovers_coupled_equations() {
        let first = [(0, 1.0), (1, -1.0)];
        let second = [(1, 1.0), (2, -1.0)];
        let matrix = matrix(3, &[&first, &second], vec![1.0, 2.0]);

        let reduction = matrix.affine_reduction().unwrap();
        let expanded = reduction.expand(&[7.0]).unwrap();

        assert_eq!(reduction.free_dof_indices, vec![2]);
        assert_eq!(reduction.pivot_dof_indices, vec![0, 1]);
        assert_eq!(expanded, vec![10.0, 9.0, 7.0]);
    }

    #[test]
    fn dense_operator_reduction_applies_affine_lifting_and_congruence() {
        let first = [(0, 1.0), (1, -1.0)];
        let second = [(1, 1.0), (2, -1.0)];
        let reduction = matrix(3, &[&first, &second], vec![1.0, 2.0])
            .affine_reduction()
            .unwrap();

        let (operator, rhs) = reduction
            .reduce_dense_operator(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[0.0; 3])
            .unwrap();
        let reduced_solution = rhs[0] / operator[0];
        let solution = reduction.expand(&[reduced_solution]).unwrap();

        assert!((operator[0] - 3.0).abs() <= 1.0e-12);
        assert!((rhs[0] + 5.0).abs() <= 1.0e-12);
        assert!((solution[0] - 4.0 / 3.0).abs() <= 1.0e-12);
        assert!((solution[1] - 1.0 / 3.0).abs() <= 1.0e-12);
        assert!((solution[2] + 5.0 / 3.0).abs() <= 1.0e-12);
    }

    // witness: FEA-SHARED-OPERATORS-001/declared_sparse_operator_properties @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
    //   boundary: Sparse reduction matches dense lifting and preserves operator properties.
    #[test]
    fn sparse_reduction_matches_dense_lifting_and_preserves_operator_properties() {
        let first = [(0, 1.0), (1, -1.0)];
        let second = [(1, 1.0), (2, -1.0)];
        let reduction = matrix(3, &[&first, &second], vec![1.0, 2.0])
            .affine_reduction()
            .unwrap();
        let dense = [4.0, 1.0, 0.5, 1.0, 3.0, 0.25, 0.5, 0.25, 2.0];
        let load = [2.0, -1.0, 5.0];
        let mut assembler = RealSparseAssembler::square(
            3,
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        );
        for row in 0..3 {
            for column in 0..3 {
                assembler.add(row, column, dense[row * 3 + column]);
            }
        }
        let sparse = assembler.finish().unwrap();

        let (dense_operator, dense_rhs) = reduction.reduce_dense_operator(&dense, &load).unwrap();
        let (sparse_operator, sparse_rhs) = reduction.reduce_sparse_system(&sparse, &load).unwrap();
        let reduced_pattern = reduction.reduce_sparse_pattern(sparse.pattern()).unwrap();

        assert_eq!(sparse_operator.properties(), sparse.properties());
        assert_eq!(reduced_pattern, sparse_operator.pattern().clone());
        assert_eq!(sparse_rhs, dense_rhs);
        for row in 0..reduction.reduced_dof_count {
            for column in 0..reduction.reduced_dof_count {
                assert_eq!(
                    sparse_operator.get(row, column),
                    dense_operator[row * reduction.reduced_dof_count + column]
                );
            }
        }
    }

    #[test]
    fn prescribed_value_reduction_retains_free_operator_coupling() {
        let prescribed = [(0, 1.0)];
        let reduction = matrix(2, &[&prescribed], vec![2.0])
            .affine_reduction()
            .unwrap();
        let (operator, rhs) = reduction
            .reduce_dense_operator(&[2.0, 1.0, 1.0, 3.0], &[0.0, 4.0])
            .unwrap();
        let solution = reduction.expand(&[rhs[0] / operator[0]]).unwrap();

        assert_eq!(operator, vec![3.0]);
        assert_eq!(rhs, vec![2.0]);
        assert_eq!(solution[0], 2.0);
        assert!((solution[1] - 2.0 / 3.0).abs() <= 1.0e-12);
    }

    #[test]
    fn rank_deficient_resolved_matrix_fails_closed() {
        let first = [(0, 1.0), (1, -1.0)];
        let duplicate = [(0, 2.0), (1, -2.0)];
        let error = matrix(2, &[&first, &duplicate], vec![0.0, 0.0])
            .affine_reduction()
            .unwrap_err();

        assert_eq!(error, ConstraintAlgebraError::RankDeficient);
    }
}
