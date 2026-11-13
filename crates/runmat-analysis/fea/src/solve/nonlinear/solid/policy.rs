use super::{constraint_error, SolidNewtonError};
use crate::{assembly::AssemblySummary, constraints::AffineConstraintReduction};

#[derive(Debug, Clone, Copy)]
pub(crate) struct SolidNewtonPolicy {
    pub max_iterations: usize,
    pub relative_residual_tolerance: f64,
    pub linear_relative_tolerance: f64,
    pub line_search: bool,
    pub max_backtracks: usize,
    pub backtrack_reduction: f64,
    pub max_cutbacks: usize,
    pub max_accepted_subincrements: usize,
    pub minimum_load_increment: f64,
}

impl SolidNewtonPolicy {
    pub(crate) fn validate(self) -> Result<(), SolidNewtonError> {
        if self.max_iterations == 0
            || self.max_accepted_subincrements == 0
            || !self.relative_residual_tolerance.is_finite()
            || self.relative_residual_tolerance <= 0.0
            || self.relative_residual_tolerance >= 1.0
            || !self.linear_relative_tolerance.is_finite()
            || self.linear_relative_tolerance <= 0.0
            || self.linear_relative_tolerance >= self.relative_residual_tolerance
            || !self.backtrack_reduction.is_finite()
            || self.backtrack_reduction <= 0.0
            || self.backtrack_reduction >= 1.0
            || !self.minimum_load_increment.is_finite()
            || self.minimum_load_increment <= 0.0
        {
            return Err(SolidNewtonError::Invalid(
                "invalid Newton/linear/line-search/cutback policy".into(),
            ));
        }
        Ok(())
    }
}

pub(super) fn constraint_reduction(
    summary: &AssemblySummary,
) -> Result<AffineConstraintReduction, SolidNewtonError> {
    let reduction = if let Some(matrix) = &summary.constraint_matrix {
        matrix.affine_reduction().map_err(constraint_error)?
    } else if let Some(reduction) = &summary.affine_constraint_reduction {
        reduction.clone()
    } else {
        let mask = summary.operator.constrained_dofs();
        let free = mask
            .iter()
            .enumerate()
            .filter_map(|(i, fixed)| (!fixed).then_some(i))
            .collect::<Vec<_>>();
        let mut row_offsets = vec![0];
        let mut columns = Vec::new();
        let mut offset = vec![0.0; summary.dof_count];
        let mut ordinal = 0;
        for (i, fixed) in mask.iter().enumerate() {
            if *fixed {
                offset[i] = summary.operator.load()[i];
            } else {
                columns.push(ordinal);
                ordinal += 1;
            }
            row_offsets.push(columns.len());
        }
        AffineConstraintReduction {
            full_dof_count: summary.dof_count,
            reduced_dof_count: free.len(),
            free_dof_indices: free,
            pivot_dof_indices: mask
                .iter()
                .enumerate()
                .filter_map(|(i, fixed)| fixed.then_some(i))
                .collect(),
            recovery_row_offsets: row_offsets,
            recovery_coefficients: vec![1.0; columns.len()],
            recovery_column_indices: columns,
            offset,
        }
    };
    reduction.validate().map_err(constraint_error)?;
    if reduction.full_dof_count != summary.dof_count {
        return Err(SolidNewtonError::Invalid(
            "equality reduction/solid dimensions differ".into(),
        ));
    }
    Ok(reduction)
}
