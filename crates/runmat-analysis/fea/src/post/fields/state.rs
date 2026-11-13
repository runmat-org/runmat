use crate::{
    algebra::apply_k,
    assembly::{dofs::StructuralDofKind, AssemblySummary},
};

use super::VECTOR_COMPONENT_COUNT;
use crate::algebra::{capacity::measured, coercivity::NumericalCapacityError};
use crate::post::error::StructuralFieldRecoveryError;
use runmat_simulation_usage::{AsyncCapacityOwner, LedgerError, OperationCount, UsageCategory};

pub(super) fn recover_reaction_force(
    summary: &AssemblySummary,
    internal_force: &[f64],
    node_count: usize,
) -> Vec<f64> {
    let mut reactions = vec![0.0; node_count * VECTOR_COMPONENT_COUNT];
    for (dof, is_constrained) in summary.operator.constrained_dofs().iter().enumerate() {
        if !*is_constrained && !is_affine_constraint_dof(summary, dof) {
            continue;
        }
        let Some(address) = summary.structural_dof_layout.address(dof) else {
            continue;
        };
        let Some(component) = translational_component(address.kind) else {
            continue;
        };
        let target = address.node_index * VECTOR_COMPONENT_COUNT + component;
        if target >= reactions.len() {
            continue;
        }
        let rhs = summary.physical_load.get(dof).copied().unwrap_or(0.0);
        let internal = internal_force.get(dof).copied().unwrap_or(0.0);
        reactions[target] = internal - rhs;
    }
    reactions
}

pub(super) fn reaction_reference_work(
    summary: &AssemblySummary,
    node_count: usize,
) -> Result<OperationCount, StructuralFieldRecoveryError> {
    let mut operations = 0_u64;
    for (dof, constrained) in summary.operator.constrained_dofs().iter().enumerate() {
        if !*constrained && !is_affine_constraint_dof(summary, dof) {
            continue;
        }
        let Some(address) = summary.structural_dof_layout.address(dof) else {
            continue;
        };
        let translation = translational_component(address.kind).is_some_and(|component| {
            address.node_index * VECTOR_COMPONENT_COUNT + component
                < node_count * VECTOR_COMPONENT_COUNT
        });
        let rotation = summary.structural_rotational_dof_count != 0
            && rotational_component(address.kind).is_some_and(|component| {
                address.node_index * VECTOR_COMPONENT_COUNT + component
                    < summary.structural_dof_layout.node_count() * VECTOR_COMPONENT_COUNT
            });
        if translation || rotation {
            operations = operations
                .checked_add(1)
                .ok_or(LedgerError::CountOverflow)?;
        }
    }
    Ok(OperationCount::from_operations(operations))
}

pub(super) fn recover_rotation(summary: &AssemblySummary, solution: &[f64]) -> Vec<f64> {
    let mut rotation = vec![0.0; rotation_shape(summary).iter().product()];
    if rotation.is_empty() {
        return rotation;
    }
    for row in 0..summary.structural_dof_layout.total_dof_count() {
        let Some(address) = summary.structural_dof_layout.address(row) else {
            continue;
        };
        let Some(component) = rotational_component(address.kind) else {
            continue;
        };
        let target = address.node_index * VECTOR_COMPONENT_COUNT + component;
        if target < rotation.len() {
            rotation[target] = solution.get(row).copied().unwrap_or(0.0);
        }
    }
    rotation
}

pub(super) fn recover_displacement(
    summary: &AssemblySummary,
    solution: &[f64],
    node_count: usize,
) -> Vec<f64> {
    if summary.structural_rotational_dof_count == 0 {
        let mut displacement = solution.to_vec();
        displacement.resize(node_count * VECTOR_COMPONENT_COUNT, 0.0);
        return displacement;
    }

    let mut displacement = vec![0.0; node_count * VECTOR_COMPONENT_COUNT];
    for row in 0..summary.structural_dof_layout.total_dof_count() {
        let Some(address) = summary.structural_dof_layout.address(row) else {
            continue;
        };
        let Some(component) = translational_component(address.kind) else {
            continue;
        };
        let target = address.node_index * VECTOR_COMPONENT_COUNT + component;
        if target < displacement.len() {
            displacement[target] = solution.get(row).copied().unwrap_or(0.0);
        }
    }
    displacement
}

fn translational_component(kind: StructuralDofKind) -> Option<usize> {
    match kind {
        StructuralDofKind::Ux => Some(0),
        StructuralDofKind::Uy => Some(1),
        StructuralDofKind::Uz => Some(2),
        _ => None,
    }
}

pub(super) fn recover_reaction_moment(
    summary: &AssemblySummary,
    internal_force: &[f64],
) -> Vec<f64> {
    let mut reactions = vec![0.0; rotation_shape(summary).iter().product()];
    if reactions.is_empty() {
        return reactions;
    }
    for (dof, is_constrained) in summary.operator.constrained_dofs().iter().enumerate() {
        if !*is_constrained && !is_affine_constraint_dof(summary, dof) {
            continue;
        }
        let Some(address) = summary.structural_dof_layout.address(dof) else {
            continue;
        };
        let Some(component) = rotational_component(address.kind) else {
            continue;
        };
        let target = address.node_index * VECTOR_COMPONENT_COUNT + component;
        if target < reactions.len() {
            let rhs = summary.physical_load.get(dof).copied().unwrap_or(0.0);
            let internal = internal_force.get(dof).copied().unwrap_or(0.0);
            reactions[target] = internal - rhs;
        }
    }
    reactions
}

fn rotational_component(kind: StructuralDofKind) -> Option<usize> {
    match kind {
        StructuralDofKind::Rx => Some(0),
        StructuralDofKind::Ry => Some(1),
        StructuralDofKind::Rz => Some(2),
        _ => None,
    }
}

pub(super) fn rotation_shape(summary: &AssemblySummary) -> Vec<usize> {
    if summary.structural_rotational_dof_count == 0 {
        return vec![0, VECTOR_COMPONENT_COUNT];
    }
    vec![
        summary.structural_dof_layout.node_count(),
        VECTOR_COMPONENT_COUNT,
    ]
}

pub(super) fn structural_displacement_node_count(
    summary: &AssemblySummary,
    dof_count: usize,
) -> usize {
    if summary.structural_rotational_dof_count > 0 {
        summary.structural_node_count.max(1)
    } else {
        dof_count.div_ceil(VECTOR_COMPONENT_COUNT).max(1)
    }
}

pub(super) struct StructuralResidualMetrics {
    pub(super) normalized_residual_norm: f64,
    pub(super) equation_scale: f64,
}

pub(super) async fn recover_residual_metrics_with_capacity<A>(
    summary: &AssemblySummary,
    displacement: &[f64],
    owner: Option<&dyn AsyncCapacityOwner<A>>,
) -> Result<StructuralResidualMetrics, NumericalCapacityError<StructuralFieldRecoveryError, A>> {
    if displacement.len() != summary.dof_count {
        return Err(StructuralFieldRecoveryError::DisplacementShapeMismatch {
            expected: summary.dof_count,
            actual: displacement.len(),
        }
        .into());
    }
    if summary.physical_load.len() != summary.dof_count {
        return Err(StructuralFieldRecoveryError::InvalidPhysicalLayout {
            reason: "residual load does not match the structural DOF count".into(),
        }
        .into());
    }
    let matrix = summary.operator.operators().stiffness();
    let count = |length: usize| {
        u64::try_from(length)
            .map_err(|_| StructuralFieldRecoveryError::Accounting(LedgerError::CountOverflow))
    };
    let mut work = OperationCount::ZERO;
    for row in 0..matrix.rows() {
        work = work
            .checked_add(
                OperationCount::direct_real_dot(count(matrix.pattern().row_range(row).len())?)
                    .map_err(LedgerError::from)
                    .map_err(StructuralFieldRecoveryError::from)?,
            )
            .map_err(LedgerError::from)
            .map_err(StructuralFieldRecoveryError::from)?;
    }
    work = work
        .checked_add(OperationCount::from_operations(count(summary.dof_count)?))
        .map_err(LedgerError::from)
        .map_err(StructuralFieldRecoveryError::from)?;
    let norm_length = if let Some(reduction) = &summary.affine_constraint_reduction {
        reduction
            .validate()
            .map_err(StructuralFieldRecoveryError::from)?;
        if reduction.full_dof_count != summary.dof_count {
            return Err(StructuralFieldRecoveryError::ConstraintProjection(
                crate::constraints::ConstraintAlgebraError::OperatorDimensionMismatch,
            )
            .into());
        }
        work = work
            .checked_add(
                OperationCount::from_operations(count(reduction.recovery_coefficients.len())?)
                    .checked_mul(4)
                    .map_err(LedgerError::from)
                    .map_err(StructuralFieldRecoveryError::from)?,
            )
            .map_err(LedgerError::from)
            .map_err(StructuralFieldRecoveryError::from)?;
        reduction.reduced_dof_count
    } else {
        summary.dof_count
    };
    let norms = OperationCount::direct_real_dot(count(norm_length)?)
        .and_then(|work| work.checked_mul(2))
        .and_then(|work| work.checked_add(OperationCount::from_operations(3)))
        .map_err(LedgerError::from)
        .map_err(StructuralFieldRecoveryError::from)?;
    work = work
        .checked_add(norms)
        .map_err(LedgerError::from)
        .map_err(StructuralFieldRecoveryError::from)?;
    measured(owner, UsageCategory::Solve, work, || {
        recover_residual_metrics_unmetered(summary, displacement)
    })
    .await
}

fn recover_residual_metrics_unmetered(
    summary: &AssemblySummary,
    displacement: &[f64],
) -> Result<StructuralResidualMetrics, StructuralFieldRecoveryError> {
    let applied = apply_k(&summary.operator, displacement);
    let mut full_residual = applied
        .iter()
        .zip(summary.physical_load.iter())
        .map(|(lhs, rhs)| lhs - rhs)
        .collect::<Vec<_>>();
    if summary.constraint_matrix.is_none() && summary.affine_constraint_reduction.is_none() {
        for (value, constrained) in full_residual
            .iter_mut()
            .zip(summary.operator.constrained_dofs())
        {
            if *constrained {
                *value = 0.0;
            }
        }
    }
    let (residual, scaled_rhs) = match &summary.affine_constraint_reduction {
        Some(reduction) => (
            reduction.transpose_apply(&full_residual)?,
            reduction.transpose_apply(&summary.physical_load)?,
        ),
        None => (full_residual, summary.physical_load.clone()),
    };
    let residual_norm = residual
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let equation_scale = scaled_rhs
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
        .max(1.0);
    Ok(StructuralResidualMetrics {
        normalized_residual_norm: residual_norm / equation_scale,
        equation_scale,
    })
}

fn is_affine_constraint_dof(summary: &AssemblySummary, dof: usize) -> bool {
    summary
        .constraint_matrix
        .as_ref()
        .is_some_and(|matrix| matrix.column_indices.contains(&dof))
}
