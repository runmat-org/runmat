use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_core::StructuralModel;
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
    selector: &str,
    mesh: &SolverMeshArtifact,
    structural: &StructuralModel,
    topology: &CanonicalStructuralNodeSpace,
) -> Result<MixedStructuralTargetOwnership, MixedStructuralTargetError> {
    let solid = solid_nodes(mesh, selector)?;
    let explicit = explicit_target_node_indices(structural, selector)
        .into_iter()
        .map(|node| topology.explicit_node_indices[node])
        .collect::<BTreeSet<_>>();
    match (solid.is_empty(), explicit.is_empty()) {
        (true, true) => Err(MixedStructuralTargetError::Unmapped {
            selector: selector.into(),
        }),
        (false, true) => Ok(MixedStructuralTargetOwnership::Solid {
            canonical_nodes: solid.into_iter().collect(),
        }),
        (true, false) => Ok(MixedStructuralTargetOwnership::Explicit {
            canonical_nodes: explicit.into_iter().collect(),
        }),
        (false, false) if solid == explicit => Ok(MixedStructuralTargetOwnership::Shared {
            canonical_nodes: solid.into_iter().collect(),
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
    selector: &str,
) -> Result<BTreeSet<usize>, MixedStructuralTargetError> {
    let node_by_id = mesh
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.node_id, index))
        .collect::<BTreeMap<_, _>>();
    let mut nodes = BTreeSet::new();
    if let Some(node_id) = selector
        .strip_prefix("node:")
        .and_then(|value| value.parse::<u64>().ok())
    {
        if let Some(node) = node_by_id.get(&node_id) {
            nodes.insert(*node);
        }
    }
    for (index, node) in mesh.topology.nodes.iter().enumerate() {
        if node
            .provenance
            .iter()
            .any(|entity| entity.matches_analysis_selection(selector))
        {
            nodes.insert(index);
        }
    }
    for face in &mesh.topology.boundary_faces {
        if face
            .provenance
            .iter()
            .any(|entity| entity.matches_analysis_selection(selector))
        {
            for node_id in &face.node_ids {
                nodes.insert(*node_by_id.get(node_id).ok_or_else(|| {
                    MixedStructuralTargetError::InvalidTopology(format!(
                        "boundary face {} references absent node {}",
                        face.face_id, node_id
                    ))
                })?);
            }
        }
    }
    for element in &mesh.topology.volume_elements {
        if element.region_id.matches_analysis_selection(selector)
            || element
                .provenance
                .iter()
                .any(|entity| entity.matches_analysis_selection(selector))
        {
            for node_id in &element.node_ids {
                nodes.insert(*node_by_id.get(node_id).ok_or_else(|| {
                    MixedStructuralTargetError::InvalidTopology(format!(
                        "volume element {} references absent node {}",
                        element.element_id, node_id
                    ))
                })?);
            }
        }
    }
    Ok(nodes)
}
