use std::collections::BTreeMap;

use thiserror::Error;

use crate::{
    assembly::AssemblySummary,
    constraints::solve_constraint_multipliers,
    planning::resource::{
        execute_scoped_slices_mut, LocalExecutionPlan, ScopedLocalExecutionError,
    },
};

#[derive(Debug, Error)]
pub(crate) enum StructuralReactionError {
    #[error("structural reaction residual has the wrong dimension or a nonfinite value")]
    InvalidResidual,
    #[error("constraint-row reactions could not be recovered from the structural residual")]
    Recovery,
    #[error("structural reaction local execution failed: {0}")]
    Scoped(#[from] ScopedLocalExecutionError<ReactionWorkError>),
}

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub(crate) enum ReactionWorkError {
    #[error("structural reaction recovery cancelled")]
    Cancelled,
}

pub(crate) fn recover_structural_reaction(
    summary: &AssemblySummary,
    residual: &[f64],
) -> Result<Vec<f64>, StructuralReactionError> {
    recover_structural_reaction_with_plan(summary, residual, None)
}

pub(crate) fn recover_structural_reaction_scoped(
    summary: &AssemblySummary,
    residual: &[f64],
    local_execution: LocalExecutionPlan,
) -> Result<Vec<f64>, StructuralReactionError> {
    recover_structural_reaction_with_plan(summary, residual, Some(local_execution))
}

fn recover_structural_reaction_with_plan(
    summary: &AssemblySummary,
    residual: &[f64],
    local_execution: Option<LocalExecutionPlan>,
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
    let constrained_dofs = summary.operator.constrained_dofs();
    if constrained_dofs.len() != residual.len() {
        return Err(StructuralReactionError::InvalidResidual);
    }
    if let Some(plan) = local_execution {
        // The admitted solve envelope already owns this one reaction vector.
        // Disjoint slices require no per-worker DOF-sized scratch or scatter.
        let mut reaction = vec![0.0; residual.len()];
        let cancellation = crate::progress::cancellation_token();
        execute_scoped_slices_mut(plan, &mut reaction, |chunk, output| {
            for (offset, value) in output.iter_mut().enumerate() {
                if cancellation.is_cancelled() {
                    return Err(ReactionWorkError::Cancelled);
                }
                let index = chunk.start() as usize + offset;
                if constrained_dofs[index] {
                    *value = residual[index];
                }
            }
            Ok(())
        })?;
        Ok(reaction)
    } else {
        Ok(residual
            .iter()
            .zip(constrained_dofs)
            .map(|(residual, constrained)| if *constrained { *residual } else { 0.0 })
            .collect())
    }
}
