use std::collections::BTreeMap;

use runmat_execution::Digest;
use runmat_meshing_core::StableDigest;

use crate::{
    assembly::dofs::{StructuralDofKind, StructuralDofLayout},
    constraints::{CanonicalStructuralDofOwner, CanonicalStructuralReferencePoint},
    physics::structural::topology::node_space::CanonicalStructuralNodeSpace,
};

use super::{CanonicalStructuralLoadPlan, StructuralLoadPlanError};

pub(crate) fn apply(
    plan: &CanonicalStructuralLoadPlan,
    topology: &CanonicalStructuralNodeSpace,
    reference_points: &[CanonicalStructuralReferencePoint],
    layout: &StructuralDofLayout,
    rhs: &mut [f64],
) -> Result<(), StructuralLoadPlanError> {
    plan.validate()?;
    if plan.topology != topology.identity
        || layout.node_count() != topology.nodes.len()
        || layout.reference_point_count() != reference_points.len()
        || rhs.len() != layout.total_dof_count()
    {
        return Err(StructuralLoadPlanError::TopologyMismatch);
    }
    let nodes = topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.identity, index))
        .collect::<BTreeMap<StableDigest, usize>>();
    let points = reference_points
        .iter()
        .enumerate()
        .map(|(index, point)| (point.identity, index))
        .collect::<BTreeMap<Digest, usize>>();
    rhs.fill(0.0);
    for entry in &plan.entries {
        for term in &entry.terms {
            let kind = StructuralDofKind::from_constraint(term.dof);
            let row = match term.owner {
                CanonicalStructuralDofOwner::StructuralNode(identity) => nodes
                    .get(&identity)
                    .and_then(|index| layout.index(*index, kind)),
                CanonicalStructuralDofOwner::ReferencePoint(identity) => points
                    .get(&identity)
                    .and_then(|index| layout.reference_point_index(*index, kind)),
            }
            .ok_or_else(|| StructuralLoadPlanError::UnresolvedTarget {
                excitation_id: entry.excitation_id.clone(),
                reason: "canonical load owner has no solver row".into(),
            })?;
            rhs[row] += term.value;
        }
    }
    Ok(())
}
