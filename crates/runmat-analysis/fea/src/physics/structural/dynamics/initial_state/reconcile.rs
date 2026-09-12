use crate::constraints::{AffineConstraintReduction, ResolvedConstraintMatrix};

use super::{assignment::Assignment, StructuralInitialConstraintView, StructuralInitialStateError};

const RELATIVE_RESIDUAL_TOLERANCE: f64 = 1.0e-10;

#[derive(Debug, Clone, Copy)]
pub(super) struct ConstraintReconciliationEvidence {
    pub direct_constrained_dof_count: usize,
    pub constraint_equation_count: usize,
    pub maximum_displacement_residual: f64,
    pub maximum_velocity_residual: f64,
}

pub(super) fn reconcile_constraints(
    displacement: &mut [Option<Assignment>],
    velocity: &mut [Option<Assignment>],
    constraints: StructuralInitialConstraintView<'_>,
) -> Result<ConstraintReconciliationEvidence, StructuralInitialStateError> {
    validate_view(displacement.len(), constraints)?;
    apply_direct_defaults(displacement, velocity, constraints)?;

    let derived_reduction;
    let reduction = match (constraints.affine_reduction, constraints.constraint_matrix) {
        (Some(reduction), _) => Some(reduction),
        (None, Some(matrix)) => {
            derived_reduction = matrix.affine_reduction().map_err(invalid_constraints)?;
            Some(&derived_reduction)
        }
        (None, None) => None,
    };
    if let Some(reduction) = reduction {
        apply_affine_defaults(displacement, reduction, false)?;
        apply_affine_defaults(velocity, reduction, true)?;
    }

    let displacement_values = values(displacement);
    let velocity_values = values(velocity);
    let (maximum_displacement_residual, maximum_velocity_residual) = constraints
        .constraint_matrix
        .map(|matrix| {
            validate_equations(
                matrix,
                displacement,
                velocity,
                &displacement_values,
                &velocity_values,
            )
        })
        .transpose()?
        .unwrap_or((0.0, 0.0));
    Ok(ConstraintReconciliationEvidence {
        direct_constrained_dof_count: constraints
            .direct_constrained_dofs
            .iter()
            .filter(|constrained| **constrained)
            .count(),
        constraint_equation_count: constraints
            .constraint_matrix
            .map_or(0, ResolvedConstraintMatrix::row_count),
        maximum_displacement_residual,
        maximum_velocity_residual,
    })
}

fn validate_view(
    dof_count: usize,
    constraints: StructuralInitialConstraintView<'_>,
) -> Result<(), StructuralInitialStateError> {
    let direct_is_empty = constraints.direct_constrained_dofs.is_empty()
        && constraints.direct_prescribed_displacement.is_empty();
    if !direct_is_empty
        && (constraints.direct_constrained_dofs.len() != dof_count
            || constraints.direct_prescribed_displacement.len() != dof_count)
    {
        return Err(invalid_constraints(
            "direct constraint vectors do not match the structural DOF count",
        ));
    }
    if let Some(matrix) = constraints.constraint_matrix {
        matrix.validate().map_err(invalid_constraints)?;
        if matrix.full_dof_count != dof_count {
            return Err(invalid_constraints(
                "constraint matrix does not match the structural DOF count",
            ));
        }
    }
    if let Some(reduction) = constraints.affine_reduction {
        reduction.validate().map_err(invalid_constraints)?;
        if reduction.full_dof_count != dof_count {
            return Err(invalid_constraints(
                "affine reduction does not match the structural DOF count",
            ));
        }
        if constraints.constraint_matrix.is_none() {
            return Err(invalid_constraints(
                "affine reduction requires its resolved constraint matrix",
            ));
        }
    }
    Ok(())
}

fn apply_direct_defaults(
    displacement: &mut [Option<Assignment>],
    velocity: &mut [Option<Assignment>],
    constraints: StructuralInitialConstraintView<'_>,
) -> Result<(), StructuralInitialStateError> {
    if constraints.direct_constrained_dofs.is_empty() {
        return Ok(());
    }
    for row in 0..displacement.len() {
        if !constraints.direct_constrained_dofs[row] {
            continue;
        }
        let prescribed = constraints.direct_prescribed_displacement[row];
        if !prescribed.is_finite() {
            return Err(invalid_constraints(
                "direct prescribed displacement is non-finite",
            ));
        }
        if let Some(authored) = &displacement[row] {
            if authored.value != prescribed {
                return Err(
                    StructuralInitialStateError::PrescribedDisplacementConflict {
                        initial_condition_id: authored.initial_condition_id().to_string(),
                        row,
                    },
                );
            }
        } else {
            displacement[row] = Some(Assignment::constraint_default(prescribed));
        }
        if let Some(authored) = &velocity[row] {
            if authored.value != 0.0 {
                return Err(StructuralInitialStateError::PrescribedVelocityConflict {
                    initial_condition_id: authored.initial_condition_id().to_string(),
                    row,
                });
            }
        } else {
            velocity[row] = Some(Assignment::constraint_default(0.0));
        }
    }
    Ok(())
}

fn apply_affine_defaults(
    assignments: &mut [Option<Assignment>],
    reduction: &AffineConstraintReduction,
    homogeneous: bool,
) -> Result<(), StructuralInitialStateError> {
    let reduced = reduction
        .free_dof_indices
        .iter()
        .map(|&row| assignments[row].as_ref().map_or(0.0, |value| value.value))
        .collect::<Vec<_>>();
    let default = if homogeneous {
        reduction.expand_linear(&reduced)
    } else {
        reduction.expand(&reduced)
    }
    .map_err(invalid_constraints)?;
    for &pivot in &reduction.pivot_dof_indices {
        if assignments[pivot].is_none() {
            assignments[pivot] = Some(Assignment::constraint_default(default[pivot]));
        }
    }
    Ok(())
}

fn validate_equations(
    matrix: &ResolvedConstraintMatrix,
    displacement_assignments: &[Option<Assignment>],
    velocity_assignments: &[Option<Assignment>],
    displacement: &[f64],
    velocity: &[f64],
) -> Result<(f64, f64), StructuralInitialStateError> {
    let mut maximum_displacement_residual = 0.0_f64;
    let mut maximum_velocity_residual = 0.0_f64;
    for row in 0..matrix.row_count() {
        let start = matrix.row_offsets[row];
        let end = matrix.row_offsets[row + 1];
        let displacement_lhs = row_value(matrix, start, end, displacement);
        let velocity_lhs = row_value(matrix, start, end, velocity);
        let scale = matrix.right_hand_side[row]
            .abs()
            .max(displacement_lhs.abs())
            .max(1.0);
        let displacement_residual = (displacement_lhs - matrix.right_hand_side[row]).abs();
        let velocity_scale = velocity_lhs.abs().max(1.0);
        let velocity_residual = velocity_lhs.abs();
        maximum_displacement_residual = maximum_displacement_residual.max(displacement_residual);
        maximum_velocity_residual = maximum_velocity_residual.max(velocity_residual);
        if displacement_residual > RELATIVE_RESIDUAL_TOLERANCE * scale {
            return Err(
                StructuralInitialStateError::DisplacementConstraintViolation {
                    constraint_id: matrix.constraint_ids[row].clone(),
                    equation_row: row,
                    authored_initial_condition_ids: authored_ids(
                        matrix,
                        start,
                        end,
                        displacement_assignments,
                    ),
                },
            );
        }
        if velocity_residual > RELATIVE_RESIDUAL_TOLERANCE * velocity_scale {
            return Err(StructuralInitialStateError::VelocityConstraintViolation {
                constraint_id: matrix.constraint_ids[row].clone(),
                equation_row: row,
                authored_initial_condition_ids: authored_ids(
                    matrix,
                    start,
                    end,
                    velocity_assignments,
                ),
            });
        }
    }
    Ok((maximum_displacement_residual, maximum_velocity_residual))
}

fn authored_ids(
    matrix: &ResolvedConstraintMatrix,
    start: usize,
    end: usize,
    assignments: &[Option<Assignment>],
) -> Vec<String> {
    let mut ids = (start..end)
        .filter_map(|entry| assignments[matrix.column_indices[entry]].as_ref())
        .filter_map(Assignment::authored_initial_condition_id)
        .map(str::to_string)
        .collect::<Vec<_>>();
    ids.sort();
    ids.dedup();
    ids
}

fn row_value(matrix: &ResolvedConstraintMatrix, start: usize, end: usize, state: &[f64]) -> f64 {
    (start..end)
        .map(|entry| matrix.coefficients[entry] * state[matrix.column_indices[entry]])
        .sum()
}

fn values(assignments: &[Option<Assignment>]) -> Vec<f64> {
    assignments
        .iter()
        .map(|entry| entry.as_ref().map_or(0.0, |assignment| assignment.value))
        .collect()
}

fn invalid_constraints(error: impl ToString) -> StructuralInitialStateError {
    StructuralInitialStateError::InvalidConstraintInputs {
        detail: error.to_string(),
    }
}
