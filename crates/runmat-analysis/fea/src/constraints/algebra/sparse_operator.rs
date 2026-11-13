//! One numeric transformation authority for discovered and planned support.
use super::*;
impl AffineConstraintReduction {
    pub fn reduce_sparse_operator(
        &self,
        operator: &RealSparseMatrix,
    ) -> Result<RealSparseMatrix, ConstraintAlgebraError> {
        self.check_operator(operator)?;
        let reduced = RealSparseAssembler::square(
            self.reduced_dof_count,
            operator.role(),
            operator.properties(),
        );
        self.scatter_operator(operator, reduced)
    }
    pub fn reduce_sparse_operator_planned(
        &self,
        operator: &RealSparseMatrix,
        pattern: &SparsePattern,
    ) -> Result<RealSparseMatrix, ConstraintAlgebraError> {
        self.check_operator(operator)?;
        if pattern.rows() != self.reduced_dof_count || pattern.columns() != self.reduced_dof_count {
            return Err(ConstraintAlgebraError::OperatorDimensionMismatch);
        }
        let reduced = RealSparseAssembler::with_pattern(
            pattern
                .try_clone()
                .map_err(|e| ConstraintAlgebraError::InvalidMatrix(e.to_string()))?,
            operator.role(),
            operator.properties(),
        )
        .map_err(sparse_reduction_error)?;
        self.scatter_operator(operator, reduced)
    }
    fn check_operator(&self, operator: &RealSparseMatrix) -> Result<(), ConstraintAlgebraError> {
        self.validate()?;
        if operator.rows() != self.full_dof_count || operator.columns() != self.full_dof_count {
            return Err(ConstraintAlgebraError::OperatorDimensionMismatch);
        }
        Ok(())
    }
    fn scatter_operator(
        &self,
        operator: &RealSparseMatrix,
        mut reduced: RealSparseAssembler,
    ) -> Result<RealSparseMatrix, ConstraintAlgebraError> {
        for contribution in operator.contributions() {
            reduced.add_contribution(contribution.clone());
        }
        for full_row in 0..self.full_dof_count {
            let left_start = self.recovery_row_offsets[full_row];
            let left_end = self.recovery_row_offsets[full_row + 1];
            for operator_entry in operator.pattern().row_range(full_row) {
                let full_column = operator.pattern().column(operator_entry);
                let value = operator.values()[operator_entry];
                let right_start = self.recovery_row_offsets[full_column];
                let right_end = self.recovery_row_offsets[full_column + 1];
                for left in left_start..left_end {
                    for right in right_start..right_end {
                        reduced.add(
                            self.recovery_column_indices[left],
                            self.recovery_column_indices[right],
                            self.recovery_coefficients[left]
                                * value
                                * self.recovery_coefficients[right],
                        );
                    }
                }
            }
        }
        reduced.finish().map_err(sparse_reduction_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn planned_general_reduction_matches_independent_congruence_and_rejects_missing_support() {
        let reduction = AffineConstraintReduction {
            full_dof_count: 3,
            reduced_dof_count: 2,
            free_dof_indices: vec![0, 1],
            pivot_dof_indices: vec![2],
            recovery_row_offsets: vec![0, 1, 2, 4],
            recovery_column_indices: vec![0, 1, 0, 1],
            recovery_coefficients: vec![1.; 4],
            offset: vec![0.; 3],
        };
        let mut assembler = RealSparseAssembler::square(
            3,
            crate::algebra::OperatorRole::EffectiveTangent,
            crate::algebra::MatrixProperties::GENERAL,
        );
        for (r, row) in [[2., 3., 0.], [-1., 4., 1.], [0., 2., 5.]]
            .iter()
            .enumerate()
        {
            for (c, v) in row.iter().enumerate() {
                assembler.add(r, c, *v);
            }
        }
        let operator = assembler.finish().unwrap();
        let original = operator.clone();
        let support = reduction.reduce_sparse_pattern(operator.pattern()).unwrap();
        let reduced = reduction
            .reduce_sparse_operator_planned(&operator, &support)
            .unwrap();
        assert_eq!(reduced.values(), &[7., 10., 5., 12.]);
        assert_eq!(
            reduced.properties(),
            crate::algebra::MatrixProperties::GENERAL
        );
        assert_eq!(
            reduced,
            reduction.reduce_sparse_operator(&operator).unwrap()
        );
        assert_eq!(operator, original);
        let missing = SparsePattern::try_new(2, 2, vec![0, 1, 2], vec![0, 1]).unwrap();
        assert!(reduction
            .reduce_sparse_operator_planned(&operator, &missing)
            .is_err());
    }
}
