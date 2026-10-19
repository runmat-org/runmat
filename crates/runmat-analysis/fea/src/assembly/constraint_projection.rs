use runmat_meshing_core::StableDigest;
use std::collections::BTreeMap;

use super::{
    direct_constraints::DirectConstraintAssignments,
    dofs::{StructuralDofKind, StructuralDofLayout},
    StructuralAssemblyError,
};
use crate::{
    constraints::{
        CanonicalStructuralConstraintPlan, CanonicalStructuralDof, CanonicalStructuralDofOwner,
        ConstraintAlgebraError, ReactionRecovery, ResolvedConstraintMatrix,
    },
    physics::structural::topology::node_space::CanonicalStructuralNodeSpace,
};

pub(super) fn resolve_constraint_matrix(
    plan: &CanonicalStructuralConstraintPlan,
    topology: &CanonicalStructuralNodeSpace,
    layout: &StructuralDofLayout,
) -> Result<ResolvedConstraintMatrix, ConstraintAlgebraError> {
    let nodes = node_indices(topology);
    let reference_points = plan
        .reference_points
        .iter()
        .enumerate()
        .map(|(index, point)| (point.identity, index))
        .collect::<BTreeMap<_, _>>();
    ResolvedConstraintMatrix::from_plan(plan, layout.total_dof_count(), |owner, dof| match owner {
        CanonicalStructuralDofOwner::StructuralNode(identity) => nodes
            .get(&identity)
            .and_then(|node| layout.index(*node, StructuralDofKind::from_constraint(dof))),
        CanonicalStructuralDofOwner::ReferencePoint(identity) => {
            reference_points.get(&identity).and_then(|index| {
                layout.reference_point_index(*index, StructuralDofKind::from_constraint(dof))
            })
        }
    })
}

pub(super) fn apply_direct_constraints(
    plan: &CanonicalStructuralConstraintPlan,
    topology: &CanonicalStructuralNodeSpace,
    assignments: &mut DirectConstraintAssignments<'_>,
) -> Result<(), StructuralAssemblyError> {
    let nodes = node_indices(topology);
    for row in &plan.rows {
        if row.reaction_recovery == ReactionRecovery::RedundantEquation {
            continue;
        }
        let [term] = row.terms.as_slice() else {
            return Err(StructuralAssemblyError::UnsupportedStructuralConstraint {
                constraint_id: row.constraint_id.clone(),
                constraint_kind: "general_constraint_equation",
            });
        };
        let CanonicalStructuralDofOwner::StructuralNode(identity) = term.owner else {
            return Err(StructuralAssemblyError::ConstraintPlanNodeAbsent);
        };
        let node = nodes
            .get(&identity)
            .copied()
            .ok_or(StructuralAssemblyError::ConstraintPlanNodeAbsent)?;
        assignments.assign(
            &row.constraint_id,
            node,
            StructuralDofKind::from_constraint(term.dof),
            row.right_hand_side / term.coefficient,
        )?;
    }
    Ok(())
}

pub(super) fn rotational_row_count(plan: &CanonicalStructuralConstraintPlan) -> usize {
    plan.rows
        .iter()
        .filter(|row| {
            row.reaction_recovery != ReactionRecovery::RedundantEquation
                && row
                    .terms
                    .iter()
                    .any(|term| matches!(term.dof, CanonicalStructuralDof::Rotation(_)))
        })
        .count()
}

fn node_indices(topology: &CanonicalStructuralNodeSpace) -> BTreeMap<StableDigest, usize> {
    topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.identity, index))
        .collect()
}
