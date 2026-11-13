use crate::topology::storage::{self, SortedValues, TopologyStorageError};

use runmat_analysis_core::{EntityDimension, ExpectedTopology, RegionTarget, StructuralModel};
use runmat_meshing_core::SolverMeshArtifact;
use thiserror::Error;

use super::{explicit_target_node_indices, node_space::CanonicalStructuralNodeSpace};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MixedStructuralTargetOwnership {
    Solid { canonical_nodes: Vec<usize> },
    Explicit { canonical_nodes: Vec<usize> },
    Shared { canonical_nodes: Vec<usize> },
}

impl MixedStructuralTargetOwnership {
    pub fn into_canonical_nodes(self) -> Vec<usize> {
        match self {
            Self::Solid { canonical_nodes }
            | Self::Explicit { canonical_nodes }
            | Self::Shared { canonical_nodes } => canonical_nodes,
        }
    }
    pub fn canonical_nodes(&self) -> &[usize] {
        match self {
            Self::Solid { canonical_nodes }
            | Self::Explicit { canonical_nodes }
            | Self::Shared { canonical_nodes } => canonical_nodes,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub(crate) enum MixedStructuralTargetError {
    #[error(transparent)]
    Storage(#[from] TopologyStorageError),
    #[error("target `{selector}` does not resolve to solid or explicit structural topology")]
    Unmapped { selector: String },
    #[error(
        "target `{selector}` resolves to different solid and explicit node sets ({solid_node_count} solid, {explicit_node_count} explicit)"
    )]
    Ambiguous {
        selector: String,
        solid_node_count: usize,
        explicit_node_count: usize,
    },
    #[error("mixed structural topology is invalid: {0}")]
    InvalidTopology(String),
}

pub(crate) fn resolve_mixed_structural_target(
    target: &RegionTarget,
    mesh: &SolverMeshArtifact,
    structural: &StructuralModel,
    topology: &CanonicalStructuralNodeSpace,
) -> Result<MixedStructuralTargetOwnership, MixedStructuralTargetError> {
    let _profile = mixed_target_workspace_profile(target, mesh, structural)?;
    let selector = &target.region_id;
    let solid = solid_nodes(mesh, target, topology)?;
    let mut explicit = explicit_target_node_indices(structural, selector)?;
    for node in &mut explicit {
        *node = *topology.explicit_node_indices.get(*node).ok_or_else(|| {
            MixedStructuralTargetError::InvalidTopology(
                "explicit target index is absent from canonical topology".into(),
            )
        })?;
    }
    explicit.sort_unstable();
    explicit.dedup();
    match (solid.is_empty(), explicit.is_empty()) {
        (true, true) => Err(MixedStructuralTargetError::Unmapped {
            selector: selector.into(),
        }),
        (false, true) => Ok(MixedStructuralTargetOwnership::Solid {
            canonical_nodes: solid,
        }),
        (true, false) => Ok(MixedStructuralTargetOwnership::Explicit {
            canonical_nodes: explicit,
        }),
        (false, false) if solid == explicit => Ok(MixedStructuralTargetOwnership::Shared {
            canonical_nodes: solid,
        }),
        (false, false) => Err(MixedStructuralTargetError::Ambiguous {
            selector: selector.into(),
            solid_node_count: solid.len(),
            explicit_node_count: explicit.len(),
        }),
    }
}

fn solid_nodes(
    mesh: &SolverMeshArtifact,
    target: &RegionTarget,
    topology: &CanonicalStructuralNodeSpace,
) -> Result<Vec<usize>, MixedStructuralTargetError> {
    storage::check_cancelled()?;
    if let Some(node_id) = exact_mesh_node_id(target) {
        // The node-space constructor validates canonical mesh ordering and retains
        // solid nodes in mesh order. The lookup therefore needs no per-target index.
        if topology.identity.mesh_digest() != Some(mesh.canonical_digest)
            || topology.solid_node_count != mesh.topology.nodes.len()
        {
            return Err(MixedStructuralTargetError::InvalidTopology(
                "solid node space does not match the solver mesh".into(),
            ));
        }
        storage::slots::<usize>(1)?;
        let mut nodes = storage::reserve(1)?;
        if let Ok(index) = mesh
            .topology
            .nodes
            .binary_search_by_key(&node_id, |node| node.node_id)
        {
            if topology.nodes.get(index).map(|node| node.identity)
                != Some(mesh.topology.nodes[index].stable_identity)
            {
                return Err(MixedStructuralTargetError::InvalidTopology(
                    "solid node identity does not match the solver mesh".into(),
                ));
            }
            nodes.push(index);
        }
        return Ok(nodes);
    }
    let node_by_id = SortedValues::derive(
        mesh.topology
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.node_id, i)),
        mesh.topology.nodes.len(),
    )?;
    let count = solid_candidate_count(mesh, target)?;
    storage::slots::<usize>(count)?;
    let mut nodes = storage::reserve(count)?;
    if target.resolved_entities().is_empty() {
        if let Some(node_id) = target
            .region_id
            .strip_prefix("node:")
            .and_then(|value| value.parse::<u64>().ok())
        {
            if let Some(node) = node_by_id.get(&node_id) {
                nodes.push(*node);
            }
        }
    }
    for (index, node) in mesh.topology.nodes.iter().enumerate() {
        if node
            .provenance
            .iter()
            .any(|entity| target.matches_persistent_entity(entity))
        {
            nodes.push(index);
        }
    }
    for face in &mesh.topology.boundary_faces {
        storage::check_cancelled()?;
        if face
            .provenance
            .iter()
            .any(|entity| target.matches_persistent_entity(entity))
        {
            for node_id in &face.node_ids {
                nodes.push(*node_by_id.get(node_id).ok_or_else(|| {
                    MixedStructuralTargetError::InvalidTopology(format!(
                        "boundary face {} references absent node {}",
                        face.face_id, node_id
                    ))
                })?);
            }
        }
    }
    for element in &mesh.topology.volume_elements {
        storage::check_cancelled()?;
        if target.selects_volume_element(&element.region_id, &element.provenance) {
            for node_id in &element.node_ids {
                nodes.push(*node_by_id.get(node_id).ok_or_else(|| {
                    MixedStructuralTargetError::InvalidTopology(format!(
                        "volume element {} references absent node {}",
                        element.element_id, node_id
                    ))
                })?);
            }
        }
    }
    nodes.sort_unstable();
    nodes.dedup();
    Ok(nodes)
}

fn solid_candidate_count(
    mesh: &SolverMeshArtifact,
    target: &RegionTarget,
) -> Result<usize, TopologyStorageError> {
    storage::check_cancelled()?;
    if exact_mesh_node_id(target).is_some() {
        // Count-only upper bound; no ordering assumption is needed before the
        // canonical node-space has been constructed by the resolver.
        return Ok(1);
    }
    let selected_point = if target.resolved_entities().is_empty() {
        target
            .region_id
            .strip_prefix("node:")
            .and_then(|v| v.parse::<u64>().ok())
            .map_or(0, |id| {
                usize::from(mesh.topology.nodes.iter().any(|node| node.node_id == id))
            })
    } else {
        0
    };
    let mut selected_provenance = 0;
    for (ordinal, node) in mesh.topology.nodes.iter().enumerate() {
        if ordinal % 1024 == 0 {
            storage::check_cancelled()?;
        }
        if node
            .provenance
            .iter()
            .any(|entity| target.matches_persistent_entity(entity))
        {
            selected_provenance = storage::count_add(selected_provenance, 1)?;
        }
    }
    let mut count = storage::count_add(selected_point, selected_provenance)?;
    for face in &mesh.topology.boundary_faces {
        storage::check_cancelled()?;
        if face
            .provenance
            .iter()
            .any(|e| target.matches_persistent_entity(e))
        {
            count = storage::count_add(count, face.node_ids.len())?;
        }
    }
    for element in &mesh.topology.volume_elements {
        storage::check_cancelled()?;
        if target.selects_volume_element(&element.region_id, &element.provenance) {
            count = storage::count_add(count, element.node_ids.len())?;
        }
    }
    Ok(count)
}

fn exact_mesh_node_id(target: &RegionTarget) -> Option<u64> {
    if target.entity_dimension != EntityDimension::Point
        || target.expected_topology != ExpectedTopology::MeshNode
        || !target.resolved_entities().is_empty()
    {
        return None;
    }
    target.region_id.strip_prefix("node:")?.parse().ok()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MixedTargetWorkspaceProfile {
    pub solid_candidate_slots: usize,
    pub explicit_candidate_slots: usize,
    pub total_requested_bytes: u64,
}

/// Additional index/selection buffers only; target metadata/identity and canonical
/// node-space construction remain their owners. No allocator overhead/RSS claim.
pub(crate) fn mixed_target_workspace_profile(
    target: &RegionTarget,
    mesh: &SolverMeshArtifact,
    structural: &StructuralModel,
) -> Result<MixedTargetWorkspaceProfile, TopologyStorageError> {
    let solid_candidate_slots = solid_candidate_count(mesh, target)?;
    let explicit = super::explicit_target_workspace_profile(structural, &target.region_id)?;
    let node_index_bytes = if exact_mesh_node_id(target).is_some() {
        0
    } else {
        storage::slots::<(u64, usize)>(mesh.topology.nodes.len())?
    };
    let total_requested_bytes = storage::add(
        explicit.total_requested_bytes,
        storage::add(
            node_index_bytes,
            storage::slots::<usize>(solid_candidate_slots)?,
        )?,
    )?;
    usize::try_from(total_requested_bytes).map_err(|_| TopologyStorageError::Capacity)?;
    Ok(MixedTargetWorkspaceProfile {
        solid_candidate_slots,
        explicit_candidate_slots: explicit.candidate_slots,
        total_requested_bytes,
    })
}
