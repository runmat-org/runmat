use crate::topology::storage::{self, SortedValues, TopologyStorageError};

use runmat_analysis_core::{RegionTarget, StructuralModel};
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
    let selector = &target.region_id;
    let solid = solid_nodes(mesh, target)?;
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
) -> Result<Vec<usize>, MixedStructuralTargetError> {
    storage::check_cancelled()?;
    let node_by_id = SortedValues::derive(
        mesh.topology
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.node_id, i)),
        mesh.topology.nodes.len(),
    )?;
    let selected_point = if target.resolved_entities().is_empty() {
        target
            .region_id
            .strip_prefix("node:")
            .and_then(|v| v.parse::<u64>().ok())
            .and_then(|id| node_by_id.get(&id))
            .map_or(0, |_| 1)
    } else {
        0
    };
    let selected_provenance = mesh
        .topology
        .nodes
        .iter()
        .filter(|n| {
            n.provenance
                .iter()
                .any(|e| target.matches_persistent_entity(e))
        })
        .count();
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
        if target.matches_persistent_entity(&element.region_id)
            || element
                .provenance
                .iter()
                .any(|e| target.matches_persistent_entity(e))
        {
            count = storage::count_add(count, element.node_ids.len())?;
        }
    }
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
        if target.matches_persistent_entity(&element.region_id)
            || element
                .provenance
                .iter()
                .any(|entity| target.matches_persistent_entity(entity))
        {
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
