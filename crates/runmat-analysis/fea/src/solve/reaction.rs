use std::collections::BTreeMap;

use thiserror::Error;

use crate::{assembly::AssemblySummary, constraints::solve_constraint_multipliers};

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub(crate) enum StructuralReactionError {
    #[error("structural reaction residual has the wrong dimension or a nonfinite value")]
    InvalidResidual,
    #[error("constraint-row reactions could not be recovered from the structural residual")]
    Recovery,
}

pub(crate) fn recover_structural_reaction(
    summary: &AssemblySummary,
    residual: &[f64],
) -> Result<Vec<f64>, StructuralReactionError> {
    if residual.len() != summary.dof_count || residual.iter().any(|value| !value.is_finite()) {
        return Err(StructuralReactionError::InvalidResidual);
    }
    if let Some(matrix) = summary.constraint_matrix.as_ref() {
        let mut rows = Vec::with_capacity(matrix.row_count());
        let mut projected_residual = Vec::with_capacity(matrix.row_count());
        for row_index in 0..matrix.row_count() {
            let start = matrix.row_offsets[row_index];
            let end = matrix.row_offsets[row_index + 1];
            let row = matrix.column_indices[start..end]
                .iter()
                .copied()
                .zip(matrix.coefficients[start..end].iter().copied())
                .collect::<BTreeMap<_, _>>();
            projected_residual.push(
                row.iter()
                    .map(|(dof, coefficient)| coefficient * residual[*dof])
                    .sum(),
            );
            rows.push(row);
        }
        let multipliers = solve_constraint_multipliers(&rows, &projected_residual)
            .ok_or(StructuralReactionError::Recovery)?;
        let mut reaction = vec![0.0; summary.dof_count];
        for (row, multiplier) in rows.iter().zip(multipliers) {
            for (dof, coefficient) in row {
                reaction[*dof] += coefficient * multiplier;
            }
        }
        return Ok(reaction);
    }
    Ok(residual
        .iter()
        .zip(summary.operator.constrained_dofs())
        .map(|(residual, constrained)| if *constrained { *residual } else { 0.0 })
        .collect())
}
