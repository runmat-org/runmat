//! Borrowed topology selection. Candidate references/IDs are counted before allocation.
use super::super::workspace::{check_cancelled, reserve};
use super::*;
use runmat_meshing_core::PersistentEntityId;

pub(super) enum Selector {
    Face,
    Point(u64),
    ExactPoint,
}
impl Selector {
    pub(super) fn derive(
        target: &RegionTarget,
        constraint_id: &str,
    ) -> Result<Self, ConstraintPlanError> {
        match (target.entity_dimension,target.expected_topology) {
            (EntityDimension::Face,ExpectedTopology::BoundaryFace)=>Ok(Self::Face),
            (EntityDimension::Point,ExpectedTopology::MeshNode) if !target.resolved_entities().is_empty()=>Ok(Self::ExactPoint),
            // An exact edge (a beam member or a shell boundary) constrains every mesh node
            // whose provenance carries that edge, like an exact vertex constrains its node.
            (EntityDimension::Edge,_) if !target.resolved_entities().is_empty()=>Ok(Self::ExactPoint),
            (EntityDimension::Point,ExpectedTopology::MeshNode)=>target.region_id.strip_prefix("node:")
                .and_then(|s|s.parse().ok()).map(Self::Point).ok_or_else(||ConstraintPlanError::UnsupportedTarget {
                    constraint_id:constraint_id.into(),reason:"mesh-node targets require a `node:<id>` selector or exact vertex resolution" }),
            _=>Err(ConstraintPlanError::UnsupportedTarget { constraint_id:constraint_id.into(),reason:"essential constraints require a boundary-face, exact edge or mesh-node target" })
        }
    }
}
pub(super) struct Selection<'a> {
    entities: Vec<&'a PersistentEntityId>,
    nodes: Vec<StableDigest>,
}
impl<'a> Selection<'a> {
    pub(super) fn derive(
        mesh: &'a SolverMeshArtifact,
        target: &'a RegionTarget,
        id: &str,
    ) -> Result<Self, ConstraintPlanError> {
        let p = profile::derive(mesh, target, id)?;
        let selector = Selector::derive(target, id)?;
        let mut entities = reserve(p.entity_reference_slots)?;
        let mut nodes;
        match selector {
            Selector::Face => {
                let mut ids = reserve(p.raw_node_id_slots)?;
                for face in &mesh.topology.boundary_faces {
                    check_cancelled()?;
                    if face
                        .provenance
                        .iter()
                        .any(|e| target.matches_persistent_entity(e))
                    {
                        entities.extend(
                            face.provenance
                                .iter()
                                .filter(|e| target.matches_persistent_entity(e)),
                        );
                        ids.extend(face.node_ids.iter().copied());
                    }
                }
                ids.sort_unstable();
                ids.dedup();
                let mut count = 0;
                for (ordinal, node) in mesh.topology.nodes.iter().enumerate() {
                    if ordinal % 1024 == 0 {
                        check_cancelled()?;
                    }
                    if ids.binary_search(&node.node_id).is_ok() {
                        count = super::super::workspace::count_add(count, 1)?;
                    }
                }
                nodes = reserve(count)?;
                for (ordinal, node) in mesh.topology.nodes.iter().enumerate() {
                    if ordinal % 1024 == 0 {
                        check_cancelled()?;
                    }
                    if ids.binary_search(&node.node_id).is_ok() {
                        nodes.push(node.stable_identity);
                    }
                }
            }
            Selector::Point(id) => {
                nodes = reserve(p.node_identity_slots)?;
                for (ordinal, node) in mesh.topology.nodes.iter().enumerate() {
                    if ordinal % 1024 == 0 {
                        check_cancelled()?;
                    }
                    if node.node_id == id {
                        nodes.push(node.stable_identity);
                    }
                }
            }
            Selector::ExactPoint => {
                nodes = reserve(p.node_identity_slots)?;
                entities.extend(target.resolved_entities().iter());
                for (ordinal, node) in mesh.topology.nodes.iter().enumerate() {
                    if ordinal % 1024 == 0 {
                        check_cancelled()?;
                    }
                    if node
                        .provenance
                        .iter()
                        .any(|e| target.matches_persistent_entity(e))
                    {
                        nodes.push(node.stable_identity);
                    }
                }
            }
        }
        entities.sort_unstable();
        entities.dedup();
        Ok(Self { entities, nodes })
    }
    pub(super) fn into_parts(self) -> (Vec<&'a PersistentEntityId>, Vec<StableDigest>) {
        (self.entities, self.nodes)
    }
}
pub(super) fn clone_entities(
    entities: &[&PersistentEntityId],
) -> Result<Vec<PersistentEntityId>, ConstraintPlanError> {
    let mut owned = reserve(entities.len())?;
    for entity in entities {
        check_cancelled()?;
        owned.push((*entity).clone());
    }
    Ok(owned)
}
