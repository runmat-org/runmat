use std::collections::BTreeSet;

use runmat_analysis_core::{EntityDimension, ExpectedTopology, RegionTarget, TargetCardinality};
use runmat_meshing_core::{SolverMeshArtifact, StableDigest};

use super::contract::{ConstraintPlanError, ResolvedConstraintTarget};
use super::plan::canonical_digest;

pub(super) fn resolve_solid_target(
    mesh: &SolverMeshArtifact,
    target: &RegionTarget,
    constraint_id: &str,
) -> Result<ResolvedConstraintTarget, ConstraintPlanError> {
    let (mut entities, mut nodes) = match (target.entity_dimension, target.expected_topology) {
        (EntityDimension::Face, ExpectedTopology::BoundaryFace) => {
            let faces = mesh
                .topology
                .boundary_faces
                .iter()
                .filter(|face| {
                    face.provenance
                        .iter()
                        .any(|entity| entity.matches_analysis_selection(&target.region_id))
                })
                .collect::<Vec<_>>();
            let entities = faces
                .iter()
                .flat_map(|face| face.provenance.iter())
                .filter(|entity| entity.matches_analysis_selection(&target.region_id))
                .cloned()
                .collect::<Vec<_>>();
            let node_ids = faces
                .iter()
                .flat_map(|face| face.node_ids.iter().copied())
                .collect::<BTreeSet<_>>();
            let nodes = mesh
                .topology
                .nodes
                .iter()
                .filter(|node| node_ids.contains(&node.node_id))
                .map(|node| node.stable_identity)
                .collect::<Vec<_>>();
            (entities, nodes)
        }
        (EntityDimension::Point, ExpectedTopology::MeshNode) => {
            let node_id = target
                .region_id
                .strip_prefix("node:")
                .and_then(|value| value.parse::<u64>().ok())
                .ok_or_else(|| ConstraintPlanError::UnsupportedTarget {
                    constraint_id: constraint_id.into(),
                    reason: "mesh-node targets require a `node:<id>` selector",
                })?;
            let nodes = mesh
                .topology
                .nodes
                .iter()
                .filter(|node| node.node_id == node_id)
                .map(|node| node.stable_identity)
                .collect::<Vec<_>>();
            (Vec::new(), nodes)
        }
        _ => {
            return Err(ConstraintPlanError::UnsupportedTarget {
                constraint_id: constraint_id.into(),
                reason: "solid essential constraints require a boundary-face or mesh-node target",
            });
        }
    };
    entities.sort();
    entities.dedup();
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
    let selection_digest =
        canonical_digest("resolved-constraint-target", &(target, &entities, &nodes))?;
    Ok(ResolvedConstraintTarget {
        authored: target.clone(),
        matched_entities: entities,
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
            != canonical_digest(
                "resolved-constraint-target",
                &(
                    &target.authored,
                    &target.matched_entities,
                    &target.matched_node_identities,
                ),
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
        .map(|node| node.coordinates_m)
        .ok_or_else(|| {
            ConstraintPlanError::Invalid(format!(
                "constraint `{constraint_id}` references a node absent from solver topology"
            ))
        })
}
