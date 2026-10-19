//! Count-only envelope for target selection and typed target/entity copies.
use super::super::workspace::{add, check_cancelled, count_add, slots};
use super::selection::Selector;
use super::*;
use runmat_meshing_core::PersistentEntityId;
mod entities;
use entities::EntityInventory;

/// Additional requested selection vectors and typed metadata copies. Candidate CAD
/// references may repeat; copy storage is an upper bound before deduplication.
/// Includes shared codec selection-identity materialization; borrowed mesh/target,
/// allocator overhead and RSS remain excluded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstraintTargetWorkspaceProfile {
    pub entity_reference_slots: usize,
    pub raw_node_id_slots: usize,
    pub node_identity_slots: usize,
    pub owned_metadata_bytes: u64,
    pub identity_materialization_bytes: u64,
    pub total_requested_bytes: u64,
}
pub fn constraint_target_workspace_profile(
    mesh: &SolverMeshArtifact,
    target: &RegionTarget,
) -> Result<ConstraintTargetWorkspaceProfile, ConstraintPlanError> {
    derive(mesh, target, "target workspace")
}
pub(in crate::constraints) fn derive(
    mesh: &SolverMeshArtifact,
    target: &RegionTarget,
    id: &str,
) -> Result<ConstraintTargetWorkspaceProfile, ConstraintPlanError> {
    check_cancelled()?;
    let selector = Selector::derive(target, id)?;
    let mut refs = 0;
    let mut ids = 0;
    let mut metadata = authored_target_metadata_bytes(target)?;
    let node_slots = match &selector {
        Selector::Face => {
            for face in &mesh.topology.boundary_faces {
                check_cancelled()?;
                if face
                    .provenance
                    .iter()
                    .any(|e| target.matches_persistent_entity(e))
                {
                    ids = count_add(ids, face.node_ids.len())?;
                    for entity in face
                        .provenance
                        .iter()
                        .filter(|e| target.matches_persistent_entity(e))
                    {
                        refs = count_add(refs, 1)?;
                        metadata = add(
                            metadata,
                            add(
                                slots::<PersistentEntityId>(1)?,
                                entity_payload_bytes(entity)?,
                            )?,
                        )?;
                    }
                }
            }
            mesh.topology.nodes.len()
        }
        Selector::Point(id) => {
            let mut count = 0;
            for (ordinal, node) in mesh.topology.nodes.iter().enumerate() {
                if ordinal % 1024 == 0 {
                    check_cancelled()?;
                }
                if node.node_id == *id {
                    count = count_add(count, 1)?;
                }
            }
            count
        }
        Selector::ExactPoint => {
            refs = target.resolved_entities().len();
            for entity in target.resolved_entities() {
                check_cancelled()?;
                metadata = add(
                    metadata,
                    add(
                        slots::<PersistentEntityId>(1)?,
                        entity_payload_bytes(entity)?,
                    )?,
                )?;
            }
            let mut count = 0;
            for (ordinal, node) in mesh.topology.nodes.iter().enumerate() {
                if ordinal % 1024 == 0 {
                    check_cancelled()?;
                }
                if node
                    .provenance
                    .iter()
                    .any(|e| target.matches_persistent_entity(e))
                {
                    count = count_add(count, 1)?;
                }
            }
            count
        }
    };
    let identity_materialization_bytes = super::identity::workspace_from_count(
        target,
        &EntityInventory {
            mesh,
            target,
            selector: &selector,
            count: refs,
        },
        node_slots,
    )?;
    let total = add(
        identity_materialization_bytes,
        add(
            metadata,
            add(
                slots::<&PersistentEntityId>(refs)?,
                add(slots::<u64>(ids)?, slots::<StableDigest>(node_slots)?)?,
            )?,
        )?,
    )?;
    usize::try_from(total).map_err(|_| super::super::workspace::overflow())?;
    Ok(ConstraintTargetWorkspaceProfile {
        entity_reference_slots: refs,
        raw_node_id_slots: ids,
        node_identity_slots: node_slots,
        owned_metadata_bytes: metadata,
        identity_materialization_bytes,
        total_requested_bytes: total,
    })
}
fn entity_payload_bytes(e: &PersistentEntityId) -> Result<u64, ConstraintPlanError> {
    let mut total = add(
        e.source_topology_id.len() as u64,
        slots::<String>(e.assembly_path.len())?,
    )?;
    for part in &e.assembly_path {
        check_cancelled()?;
        total = add(total, part.len() as u64)?;
    }
    Ok(total)
}
#[cfg(test)]
mod tests;

pub(in crate::constraints) fn authored_target_metadata_bytes(
    target: &RegionTarget,
) -> Result<u64, ConstraintPlanError> {
    check_cancelled()?;
    let mut metadata = add(
        target.region_id.len() as u64,
        slots::<PersistentEntityId>(target.resolved_entities().len())?,
    )?;
    for entity in target.resolved_entities() {
        check_cancelled()?;
        metadata = add(metadata, entity_payload_bytes(entity)?)?;
    }
    Ok(metadata)
}
