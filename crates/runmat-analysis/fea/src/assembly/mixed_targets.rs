use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_core::StructuralModel;
use runmat_meshing_core::SolverMeshArtifact;

use super::{structural_target_nodes, StructuralAssemblyError};
use crate::physics::structural::topology::mixed::CanonicalMixedStructuralTopology;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum MixedTargetOwnership {
    Solid { canonical_nodes: Vec<usize> },
    Explicit { canonical_nodes: Vec<usize> },
    Shared { canonical_nodes: Vec<usize> },
}

pub(super) fn resolve_mixed_target(
    target_id: &str,
    selector: &str,
    mesh: &SolverMeshArtifact,
    structural: &StructuralModel,
    topology: &CanonicalMixedStructuralTopology,
) -> Result<MixedTargetOwnership, StructuralAssemblyError> {
    let solid = solid_nodes(mesh, selector)?;
    let explicit = structural_target_nodes(structural, selector)
        .into_iter()
        .map(|node| topology.explicit_node_indices[node])
        .collect::<BTreeSet<_>>();
    match (solid.is_empty(), explicit.is_empty()) {
        (true, true) => Err(StructuralAssemblyError::UnmappedMixedTarget {
            target_id: target_id.into(),
            selector: selector.into(),
        }),
        (false, true) => Ok(MixedTargetOwnership::Solid {
            canonical_nodes: solid.into_iter().collect(),
        }),
        (true, false) => Ok(MixedTargetOwnership::Explicit {
            canonical_nodes: explicit.into_iter().collect(),
        }),
        (false, false) if solid == explicit => Ok(MixedTargetOwnership::Shared {
            canonical_nodes: solid.into_iter().collect(),
        }),
        (false, false) => Err(StructuralAssemblyError::AmbiguousMixedTarget {
            target_id: target_id.into(),
            selector: selector.into(),
            solid_node_count: solid.len(),
            explicit_node_count: explicit.len(),
        }),
    }
}

fn solid_nodes(
    mesh: &SolverMeshArtifact,
    selector: &str,
) -> Result<BTreeSet<usize>, StructuralAssemblyError> {
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
                    StructuralAssemblyError::MixedTopology(format!(
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
                    StructuralAssemblyError::MixedTopology(format!(
                        "volume element {} references absent node {}",
                        element.element_id, node_id
                    ))
                })?);
            }
        }
    }
    Ok(nodes)
}

#[cfg(test)]
mod tests {
    use runmat_analysis_core::{ConformalSolidAttachment, StructuralNode};
    use runmat_meshing_core::{
        ElementOrder, PersistentEntityId, PersistentEntityKind, StableDigest,
    };

    use super::*;

    #[test]
    fn same_attached_node_is_exactly_shared() {
        let (mesh, structural, topology) = topology(true);
        assert_eq!(
            resolve_mixed_target("constraint", "node:1", &mesh, &structural, &topology)
                .expect("same canonical node"),
            MixedTargetOwnership::Shared {
                canonical_nodes: vec![0]
            }
        );
    }

    #[test]
    fn colliding_node_ids_on_different_nodes_are_ambiguous() {
        let (mesh, structural, topology) = topology(false);
        assert!(matches!(
            resolve_mixed_target("constraint", "node:1", &mesh, &structural, &topology),
            Err(StructuralAssemblyError::AmbiguousMixedTarget {
                solid_node_count: 1,
                explicit_node_count: 1,
                ..
            })
        ));
    }

    fn topology(
        attached: bool,
    ) -> (
        SolverMeshArtifact,
        StructuralModel,
        CanonicalMixedStructuralTopology,
    ) {
        let mut mesh = crate::assembly::solver_solid::tests::artifact(ElementOrder::Tet4);
        mesh.topology.nodes[0].provenance = vec![PersistentEntityId {
            kind: PersistentEntityKind::Vertex,
            source_topology_id: "vertex_1".into(),
            assembly_path: vec!["root".into()],
        }];
        mesh.canonical_digest = StableDigest::ZERO;
        mesh.seal_canonical_digest().expect("canonical mesh");
        let structural = StructuralModel {
            nodes: vec![StructuralNode {
                node_id: 1,
                coordinates_m: if attached {
                    mesh.topology.nodes[0].coordinates_m
                } else {
                    [2.0, 0.0, 0.0]
                },
                solid_attachment: attached.then(|| ConformalSolidAttachment {
                    geometry_vertex_id: "vertex_1".into(),
                }),
            }],
            elements: Vec::new(),
            beam_sections: Vec::new(),
            shell_sections: Vec::new(),
        };
        let topology =
            CanonicalMixedStructuralTopology::derive(&mesh, &structural).expect("mixed topology");
        (mesh, structural, topology)
    }
}
