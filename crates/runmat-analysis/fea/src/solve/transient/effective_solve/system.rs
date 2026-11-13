//! Effective dynamics owns only solver matrices and constraints, never physical recovery metadata.
use super::{homogeneous_constraints, EffectiveDynamicsSolveError};
use crate::{
    algebra::{AssembledRealSystem, CanonicalRealOperators, RealSparseMatrix},
    assembly::AssemblySummary,
    constraints::{AffineConstraintReduction, ResolvedConstraintMatrix},
    solve::linear::StructuralLinearSystemView,
};

pub(super) struct EffectiveLinearSystem<'a> {
    physical_layout: crate::assembly::physical_fields::StructuralSolveLayout<'a>,
    dof_count: usize,
    operator: AssembledRealSystem,
    constraint_matrix: Option<ResolvedConstraintMatrix>,
    affine_constraint_reduction: Option<AffineConstraintReduction>,
}

impl EffectiveLinearSystem<'_> {
    pub(super) fn view(&self) -> StructuralLinearSystemView<'_> {
        StructuralLinearSystemView {
            dof_count: self.dof_count,
            physical_layout: self.physical_layout.clone(),
            operator: &self.operator,
            constraint_matrix: self.constraint_matrix.as_ref(),
            affine_constraint_reduction: self.affine_constraint_reduction.as_ref(),
            planned_reduced_support: None,
        }
    }
}

pub(super) fn effective_system<'a>(
    source: &'a AssemblySummary,
    effective_operator: RealSparseMatrix,
    right_hand_side: Vec<f64>,
) -> Result<EffectiveLinearSystem<'a>, EffectiveDynamicsSolveError> {
    let (constraint_matrix, affine_constraint_reduction) = homogeneous_constraints(source)?;
    let mut constrained_load = right_hand_side;
    for (dof, constrained) in source
        .operator
        .constrained_dofs()
        .iter()
        .copied()
        .enumerate()
    {
        if constrained {
            constrained_load[dof] = 0.0;
        }
    }
    let operators = CanonicalRealOperators::try_new(
        effective_operator,
        source.operator.operators().mass().clone(),
        source.operator.operators().damping().clone(),
    )
    .map_err(EffectiveDynamicsSolveError::InvalidEffectiveSystem)?;
    let operator = AssembledRealSystem::try_new(
        operators,
        source.operator.constrained_dofs().to_vec(),
        constrained_load,
    )
    .map_err(EffectiveDynamicsSolveError::InvalidEffectiveSystem)?;
    Ok(EffectiveLinearSystem {
        physical_layout: source.structural_solve_layout(),
        dof_count: source.dof_count,
        operator,
        constraint_matrix,
        affine_constraint_reduction,
    })
}

pub(super) fn reduced_effective_system<'a>(
    source: &'a AssemblySummary,
    reduction: std::sync::Arc<AffineConstraintReduction>,
    effective_operator: RealSparseMatrix,
    right_hand_side: Vec<f64>,
) -> Result<EffectiveLinearSystem<'a>, EffectiveDynamicsSolveError> {
    let mass = reduction
        .reduce_sparse_operator(source.operator.operators().mass())
        .map_err(EffectiveDynamicsSolveError::InvalidConstraintState)?;
    let damping = reduction
        .reduce_sparse_operator(source.operator.operators().damping())
        .map_err(EffectiveDynamicsSolveError::InvalidConstraintState)?;
    let operators = CanonicalRealOperators::try_new(effective_operator, mass, damping)
        .map_err(EffectiveDynamicsSolveError::InvalidEffectiveSystem)?;
    let operator = AssembledRealSystem::try_new(
        operators,
        vec![false; reduction.reduced_dof_count],
        right_hand_side,
    )
    .map_err(EffectiveDynamicsSolveError::InvalidEffectiveSystem)?;
    let dof_count = reduction.reduced_dof_count;
    Ok(EffectiveLinearSystem {
        physical_layout: crate::assembly::physical_fields::StructuralSolveLayout::Affine {
            full: &source.structural_dof_layout,
            reduction,
        },
        dof_count,
        operator,
        constraint_matrix: None,
        affine_constraint_reduction: None,
    })
}
