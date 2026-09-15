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
    let reduction = match (
        constraints.affine_reduction,
        constraints.displacement_constraints,
    ) {
        (Some(reduction), _) => Some(reduction),
        (None, Some(matrix)) => {
            derived_reduction = matrix.affine_reduction().map_err(invalid_constraints)?;
            Some(&derived_reduction)
        }
        (None, None) => None,
    };
    let velocity_reduction = velocity_reduction(constraints, reduction)?;
    if let Some(reduction) = reduction {
        apply_affine_defaults(displacement, reduction, false)?;
    }
    if let Some(reduction) = velocity_reduction.as_ref() {
        apply_affine_defaults(velocity, reduction, false)?;
    }

    let displacement_values = values(displacement);
    let velocity_values = values(velocity);
    let (maximum_displacement_residual, maximum_velocity_residual) = constraints
        .displacement_constraints
        .map(|matrix| {
            validate_equations(
                matrix,
                constraints.velocity_constraints,
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
            .displacement_constraints
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
    if let Some(matrix) = constraints.displacement_constraints {
        matrix.validate().map_err(invalid_constraints)?;
        if matrix.full_dof_count != dof_count {
            return Err(invalid_constraints(
                "constraint matrix does not match the structural DOF count",
            ));
        }
    }
    if let Some(matrix) = constraints.velocity_constraints {
        matrix.validate().map_err(invalid_constraints)?;
        let Some(displacement) = constraints.displacement_constraints else {
            return Err(invalid_constraints(
                "velocity constraints require the same structural DOF space as displacement constraints",
            ));
        };
        if matrix.full_dof_count != dof_count
            || matrix.row_offsets != displacement.row_offsets
            || matrix.column_indices != displacement.column_indices
            || matrix.coefficients != displacement.coefficients
            || matrix.row_identities != displacement.row_identities
            || matrix.constraint_ids != displacement.constraint_ids
        {
            return Err(invalid_constraints(
                "velocity constraints differ from the displacement constraint row space",
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
        if constraints.displacement_constraints.is_none() {
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

fn velocity_reduction(
    constraints: StructuralInitialConstraintView<'_>,
    displacement: Option<&AffineConstraintReduction>,
) -> Result<Option<AffineConstraintReduction>, StructuralInitialStateError> {
    let Some(displacement) = displacement else {
        return Ok(None);
    };
    let mut matrix = constraints
        .velocity_constraints
        .cloned()
        .or_else(|| constraints.displacement_constraints.cloned())
        .ok_or_else(|| invalid_constraints("affine constraints have no resolved row space"))?;
    if constraints.velocity_constraints.is_none() {
        matrix.right_hand_side.fill(0.0);
    }
    let velocity = matrix.affine_reduction().map_err(invalid_constraints)?;
    if velocity.full_dof_count != displacement.full_dof_count
        || velocity.reduced_dof_count != displacement.reduced_dof_count
        || velocity.free_dof_indices != displacement.free_dof_indices
        || velocity.pivot_dof_indices != displacement.pivot_dof_indices
        || velocity.recovery_row_offsets != displacement.recovery_row_offsets
        || velocity.recovery_column_indices != displacement.recovery_column_indices
        || velocity.recovery_coefficients != displacement.recovery_coefficients
    {
        return Err(invalid_constraints(
            "displacement and velocity constraints have different homogeneous transforms",
        ));
    }
    Ok(Some(velocity))
}

fn validate_equations(
    displacement_matrix: &ResolvedConstraintMatrix,
    velocity_matrix: Option<&ResolvedConstraintMatrix>,
    displacement_assignments: &[Option<Assignment>],
    velocity_assignments: &[Option<Assignment>],
    displacement: &[f64],
    velocity: &[f64],
) -> Result<(f64, f64), StructuralInitialStateError> {
    let mut maximum_displacement_residual = 0.0_f64;
    let mut maximum_velocity_residual = 0.0_f64;
    for row in 0..displacement_matrix.row_count() {
        let start = displacement_matrix.row_offsets[row];
        let end = displacement_matrix.row_offsets[row + 1];
        let displacement_lhs = row_value(displacement_matrix, start, end, displacement);
        let expected_velocity = velocity_matrix.map_or(0.0, |matrix| matrix.right_hand_side[row]);
        let velocity_rows = velocity_matrix.unwrap_or(displacement_matrix);
        let velocity_start = velocity_rows.row_offsets[row];
        let velocity_end = velocity_rows.row_offsets[row + 1];
        let velocity_lhs = row_value(velocity_rows, velocity_start, velocity_end, velocity);
        let scale = displacement_matrix.right_hand_side[row]
            .abs()
            .max(displacement_lhs.abs())
            .max(1.0);
        let displacement_residual =
            (displacement_lhs - displacement_matrix.right_hand_side[row]).abs();
        let velocity_scale = velocity_lhs.abs().max(expected_velocity.abs()).max(1.0);
        let velocity_residual = (velocity_lhs - expected_velocity).abs();
        maximum_displacement_residual = maximum_displacement_residual.max(displacement_residual);
        maximum_velocity_residual = maximum_velocity_residual.max(velocity_residual);
        if displacement_residual > RELATIVE_RESIDUAL_TOLERANCE * scale {
            return Err(
                StructuralInitialStateError::DisplacementConstraintViolation {
                    constraint_id: displacement_matrix.constraint_ids[row].clone(),
                    equation_row: row,
                    authored_initial_condition_ids: authored_ids(
                        displacement_matrix,
                        start,
                        end,
                        displacement_assignments,
                    ),
                },
            );
        }
        if velocity_residual > RELATIVE_RESIDUAL_TOLERANCE * velocity_scale {
            return Err(StructuralInitialStateError::VelocityConstraintViolation {
                constraint_id: displacement_matrix.constraint_ids[row].clone(),
                equation_row: row,
                authored_initial_condition_ids: authored_ids(
                    velocity_rows,
                    velocity_start,
                    velocity_end,
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
