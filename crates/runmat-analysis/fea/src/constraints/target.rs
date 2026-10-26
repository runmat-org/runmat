use runmat_analysis_core::{EntityDimension, ExpectedTopology, RegionTarget, TargetCardinality};
use runmat_meshing_core::{SolverMeshArtifact, StableDigest};

use super::contract::{ConstraintPlanError, ResolvedConstraintTarget};

pub(super) mod identity;
pub(super) mod profile;
mod selection;
pub use profile::{constraint_target_workspace_profile, ConstraintTargetWorkspaceProfile};

#[cfg(test)]
mod tests;

pub(crate) fn resolve_nodal_target(
    mesh: &SolverMeshArtifact,
    target: &RegionTarget,
    constraint_id: &str,
) -> Result<ResolvedConstraintTarget, ConstraintPlanError> {
    let selection = selection::Selection::derive(mesh, target, constraint_id)?;
    let (entities, mut nodes) = selection.into_parts();
    nodes.sort_unstable();
    nodes.dedup();
    if nodes.is_empty() {
        return Err(ConstraintPlanError::EmptyTarget {
            constraint_id: constraint_id.into(),
            region_id: target.region_id.clone(),
        });
    }
    let matched_entity_count = if entities.is_empty() {
        nodes.len()
    } else {
        entities.len()
    };
    if target.cardinality == TargetCardinality::One && matched_entity_count != 1 {
        return Err(ConstraintPlanError::Cardinality {
            constraint_id: constraint_id.into(),
            actual: matched_entity_count,
        });
    }
    let selection_digest = identity::digest(target, &entities, &nodes)?;
    Ok(ResolvedConstraintTarget {
        authored: target.clone(),
        matched_entities: selection::clone_entities(&entities)?,
        matched_node_identities: nodes,
        selection_digest,
    })
}

pub(super) fn validate_resolved_target(
    constraint_id: &str,
    target: &ResolvedConstraintTarget,
) -> Result<(), ConstraintPlanError> {
    if target.matched_node_identities.is_empty()
        || target.matched_node_identities.contains(&StableDigest::ZERO)
        || !target
            .matched_node_identities
            .windows(2)
            .all(|pair| pair[0] < pair[1])
        || !target
            .matched_entities
            .windows(2)
            .all(|pair| pair[0] < pair[1])
        || target.selection_digest
            != identity::digest(
                &target.authored,
                &target.matched_entities,
                &target.matched_node_identities,
            )?
    {
        return Err(ConstraintPlanError::Invalid(format!(
            "resolved target for constraint `{constraint_id}` is inconsistent"
        )));
    }
    Ok(())
}

pub(super) fn node_coordinates(
    mesh: &SolverMeshArtifact,
    identity: StableDigest,
    constraint_id: &str,
) -> Result<[f64; 3], ConstraintPlanError> {
    mesh.topology
        .nodes
        .iter()
        .find(|node| node.stable_identity == identity)
        .map(|node| node.coordinates_m.canonical_value())
        .ok_or_else(|| {
            ConstraintPlanError::Invalid(format!(
                "constraint `{constraint_id}` references a node absent from solver topology"
            ))
        })
}
