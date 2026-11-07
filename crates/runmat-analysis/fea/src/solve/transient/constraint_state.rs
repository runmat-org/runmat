use crate::{
    assembly::AssemblySummary,
    constraints::{AffineConstraintReduction, ConstraintAlgebraError, ResolvedConstraintMatrix},
    solve::affine::affine_expand_work,
};

use super::{
    effective_solve::{measured, measured_affine, EffectiveDynamicsSolveError},
    newmark::count,
    state::{SecondOrderState, SecondOrderStateError},
};

const KINEMATIC_RELATIVE_TOLERANCE: f64 = 1.0e-10;

/// Recovers dependent accelerations through the exact homogeneous constraint transform after an
/// iterative solve. Direct prescriptions are the degenerate case and have zero acceleration.
pub(super) fn canonicalize_homogeneous_solution(
    summary: &crate::solve::linear::StructuralLinearSystemView<'_>,
    solution: &mut Vec<f64>,
) -> Result<(), EffectiveDynamicsSolveError> {
    if let Some(matrix) = summary.constraint_matrix.as_ref() {
        let reduction = matrix
            .affine_reduction()
            .map_err(EffectiveDynamicsSolveError::InvalidConstraintState)?;
        let independent = reduction
            .free_dof_indices
            .iter()
            .map(|dof| solution[*dof])
            .collect::<Vec<_>>();
        *solution = expand_homogeneous_solution(&reduction, &independent)?;
    } else {
        for (value, constrained) in solution.iter_mut().zip(summary.operator.constrained_dofs()) {
            if *constrained {
                *value = 0.0;
            }
        }
    }
    Ok(())
}

pub(super) fn expand_homogeneous_solution(
    reduction: &AffineConstraintReduction,
    independent: &[f64],
) -> Result<Vec<f64>, EffectiveDynamicsSolveError> {
    measured_affine(
        || {
            affine_expand_work(reduction, None)?
                .checked_add(
                    count(reduction.full_dof_count, 1)
                        .map_err(crate::algebra::ScopedSpmvError::Usage)?,
                )
                .map_err(|error| crate::algebra::ScopedSpmvError::Usage(error.into()))
        },
        || {
            reduction
                .expand_linear(independent)
                .map_err(EffectiveDynamicsSolveError::InvalidConstraintState)
        },
    )
}

pub(super) fn homogeneous_constraints(
    summary: &AssemblySummary,
) -> Result<
    (
        Option<ResolvedConstraintMatrix>,
        Option<AffineConstraintReduction>,
    ),
    EffectiveDynamicsSolveError,
> {
    let Some(source_matrix) = summary.constraint_matrix.as_ref() else {
        if summary.affine_constraint_reduction.is_some() {
            return Err(EffectiveDynamicsSolveError::InvalidConstraintState(
                ConstraintAlgebraError::InvalidMatrix(
                    "affine reduction has no resolved constraint matrix".into(),
                ),
            ));
        }
        return Ok((None, None));
    };
    source_matrix
        .validate()
        .map_err(EffectiveDynamicsSolveError::InvalidConstraintState)?;
    if let Some(source_reduction) = summary.affine_constraint_reduction.as_ref() {
        let expected = source_matrix
            .affine_reduction()
            .map_err(EffectiveDynamicsSolveError::InvalidConstraintState)?;
        if source_reduction != &expected {
            return Err(EffectiveDynamicsSolveError::InvalidConstraintState(
                ConstraintAlgebraError::InvalidMatrix(
                    "affine reduction differs from its resolved constraint matrix".into(),
                ),
            ));
        }
    }
    let mut matrix = source_matrix.clone();
    matrix.right_hand_side.fill(0.0);
    let reduction = summary
        .affine_constraint_reduction
        .as_ref()
        .map(|_| matrix.affine_reduction())
        .transpose()
        .map_err(EffectiveDynamicsSolveError::InvalidConstraintState)?;
    Ok((Some(matrix), reduction))
}

pub(super) fn validate_constant_constraint_kinematics(
    summary: &AssemblySummary,
    state: &SecondOrderState,
    validate_acceleration: bool,
) -> Result<(), EffectiveDynamicsSolveError> {
    if state.dof_count() != summary.dof_count {
        return Err(EffectiveDynamicsSolveError::InvalidState(
            SecondOrderStateError::DimensionMismatch,
        ));
    }
    if let Some(matrix) = summary.constraint_matrix.as_ref() {
        let displacement = matrix
            .apply(state.displacement())
            .map_err(EffectiveDynamicsSolveError::InvalidConstraintState)?;
        let velocity = matrix
            .apply(state.velocity())
            .map_err(EffectiveDynamicsSolveError::InvalidConstraintState)?;
        let acceleration = validate_acceleration
            .then(|| matrix.apply(state.acceleration()))
            .transpose()
            .map_err(EffectiveDynamicsSolveError::InvalidConstraintState)?;
        if !approximately_equal(&displacement, &matrix.right_hand_side) {
            return Err(inconsistent_kinematics(
                "displacement",
                &displacement,
                &matrix.right_hand_side,
            ));
        }
        if !approximately_zero(&velocity) {
            return Err(inconsistent_kinematics(
                "velocity",
                &velocity,
                &vec![0.0; velocity.len()],
            ));
        }
        if let Some(acceleration) = acceleration.filter(|value| !approximately_zero(value)) {
            return Err(inconsistent_kinematics(
                "acceleration",
                &acceleration,
                &vec![0.0; acceleration.len()],
            ));
        }
        return Ok(());
    }
    for (dof, constrained) in summary
        .operator
        .constrained_dofs()
        .iter()
        .copied()
        .enumerate()
    {
        if constrained
            && (!approximately_scalar(state.displacement()[dof], summary.operator.load()[dof])
                || !approximately_scalar(state.velocity()[dof], 0.0)
                || (validate_acceleration && !approximately_scalar(state.acceleration()[dof], 0.0)))
        {
            let (quantity, maximum_violation) =
                if !approximately_scalar(state.displacement()[dof], summary.operator.load()[dof]) {
                    (
                        "displacement",
                        (state.displacement()[dof] - summary.operator.load()[dof]).abs(),
                    )
                } else if !approximately_scalar(state.velocity()[dof], 0.0) {
                    ("velocity", state.velocity()[dof].abs())
                } else {
                    ("acceleration", state.acceleration()[dof].abs())
                };
            return Err(
                EffectiveDynamicsSolveError::InconsistentConstraintKinematics {
                    quantity,
                    maximum_violation,
                },
            );
        }
    }
    Ok(())
}

pub(super) fn project_vector(
    summary: &AssemblySummary,
    full: &[f64],
) -> Result<Vec<f64>, EffectiveDynamicsSolveError> {
    if let Some(matrix) = summary.constraint_matrix.as_ref() {
        let reduction = matrix
            .affine_reduction()
            .map_err(EffectiveDynamicsSolveError::InvalidConstraintState)?;
        return measured(
            || count(reduction.recovery_coefficients.len(), 2),
            || {
                reduction
                    .transpose_apply(full)
                    .map_err(EffectiveDynamicsSolveError::InvalidConstraintState)
            },
        );
    }
    Ok(full
        .iter()
        .zip(summary.operator.constrained_dofs())
        .filter_map(|(value, constrained)| (!constrained).then_some(*value))
        .collect())
}

pub(super) fn physical_load(summary: &AssemblySummary) -> Vec<f64> {
    summary.physical_load.clone()
}

fn inconsistent_kinematics(
    quantity: &'static str,
    actual: &[f64],
    expected: &[f64],
) -> EffectiveDynamicsSolveError {
    let maximum_violation = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    EffectiveDynamicsSolveError::InconsistentConstraintKinematics {
        quantity,
        maximum_violation,
    }
}

fn approximately_equal(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| approximately_scalar(*left, *right))
}

fn approximately_zero(values: &[f64]) -> bool {
    values.iter().all(|value| approximately_scalar(*value, 0.0))
}

fn approximately_scalar(left: f64, right: f64) -> bool {
    (left - right).abs() <= KINEMATIC_RELATIVE_TOLERANCE * left.abs().max(right.abs()).max(1.0)
}

#[cfg(test)]
mod tests {
    use runmat_execution::Digest;

    use super::*;

    #[test]
    fn homogeneous_mpc_preserves_its_transform_and_removes_only_the_offset() {
        let matrix = ResolvedConstraintMatrix {
            full_dof_count: 2,
            row_offsets: vec![0, 2],
            column_indices: vec![0, 1],
            coefficients: vec![1.0, -1.0],
            right_hand_side: vec![0.25],
            row_identities: vec![Digest::sha256(b"mpc-row")],
            constraint_ids: vec!["relative_displacement".into()],
        };
        let original = matrix.affine_reduction().unwrap();
        let mut homogeneous = matrix.clone();
        homogeneous.right_hand_side.fill(0.0);
        let homogeneous = homogeneous.affine_reduction().unwrap();

        assert_eq!(homogeneous.free_dof_indices, original.free_dof_indices);
        assert_eq!(homogeneous.pivot_dof_indices, original.pivot_dof_indices);
        assert_eq!(
            homogeneous.recovery_row_offsets,
            original.recovery_row_offsets
        );
        assert_eq!(
            homogeneous.recovery_column_indices,
            original.recovery_column_indices
        );
        assert_eq!(
            homogeneous.recovery_coefficients,
            original.recovery_coefficients
        );
        assert_eq!(original.offset, vec![0.25, 0.0]);
        assert_eq!(homogeneous.offset, vec![0.0, 0.0]);
    }
}
