//! Small conforming multi-element tetrahedron for physical solver evidence.

use crate::{
    solver_volume_element_identity, ElementOrder, FieldTopologyLocation, MeshNeighbor,
    SolverMeshArtifact, SolverMeshNode, StableDigest,
};

use super::solver_mesh::canonical_tetrahedron_solver_mesh;

/// Four conforming Tet4 children formed by splitting the canonical tetrahedron at its centroid.
pub fn connected_tetrahedron_solver_mesh() -> SolverMeshArtifact {
    let mut artifact = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let parent = artifact.topology.volume_elements[0].clone();
    let centroid_id = 5;
    artifact.topology.nodes.push(SolverMeshNode {
        node_id: centroid_id,
        stable_identity: StableDigest::from_bytes([42; 32]),
        coordinates_m: [0.25, 0.25, 0.25],
        provenance: parent.provenance.clone(),
        exact_parameters: Vec::new(),
    });
    let children = [[5, 2, 3, 4], [1, 5, 3, 4], [1, 2, 5, 4], [1, 2, 3, 5]];
    artifact.topology.volume_elements = children
        .into_iter()
        .enumerate()
        .map(|(index, node_ids)| {
            let mut child = parent.clone();
            child.element_id = index as u64 + 1;
            child.node_ids = node_ids.to_vec();
            child.stable_identity = solver_volume_element_identity(
                node_ids
                    .map(|node_id| artifact.topology.nodes[node_id as usize - 1].stable_identity),
            );
            child
        })
        .collect();
    for face in &mut artifact.topology.boundary_faces {
        let corners = &face.node_ids[..3];
        face.adjacent_volume_element_ids = artifact
            .topology
            .volume_elements
            .iter()
            .filter(|element| corners.iter().all(|node| element.node_ids.contains(node)))
            .map(|element| element.element_id)
            .collect();
    }
    artifact.topology.neighbors = artifact
        .topology
        .volume_elements
        .iter()
        .flat_map(|element| {
            (0..4).map(|local_face_index| {
                let shared = element
                    .node_ids
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| *index != local_face_index)
                    .map(|(_, node)| node)
                    .collect::<Vec<_>>();
                MeshNeighbor {
                    element_id: element.element_id,
                    local_face_index: local_face_index as u8,
                    adjacent_element_id: artifact
                        .topology
                        .volume_elements
                        .iter()
                        .find(|other| {
                            other.element_id != element.element_id
                                && shared.iter().all(|node| other.node_ids.contains(node))
                        })
                        .map(|other| other.element_id),
                }
            })
        })
        .collect();
    artifact.topology.regions[0].element_ids = (1..=4).collect();
    for topology in &mut artifact.topology.field_topologies {
        match topology.location {
            FieldTopologyLocation::Node => topology.ordered_entity_ids.push(centroid_id),
            FieldTopologyLocation::VolumeElement => {
                topology.ordered_entity_ids = (1..=4).collect();
            }
            FieldTopologyLocation::BoundaryFace | FieldTopologyLocation::BoundaryEdge => {}
        }
    }
    artifact.canonical_digest = StableDigest::ZERO;
    artifact.seal_canonical_digest().unwrap();
    artifact
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CanonicalMeshingContract;

    #[test]
    fn connected_tet4_fixture_preserves_exterior_and_reciprocal_interior_faces() {
        let mesh = connected_tetrahedron_solver_mesh();
        mesh.validate_canonical().unwrap();
        assert_eq!(mesh.topology.nodes.len(), 5);
        assert_eq!(mesh.topology.volume_elements.len(), 4);
        assert_eq!(mesh.topology.boundary_faces.len(), 4);
        assert_eq!(mesh.topology.regions[0].element_ids, vec![1, 2, 3, 4]);
        assert_eq!(
            mesh.topology
                .neighbors
                .iter()
                .filter(|neighbor| neighbor.adjacent_element_id.is_some())
                .count(),
            12
        );
        assert!(mesh
            .topology
            .boundary_faces
            .iter()
            .all(|face| face.adjacent_volume_element_ids.len() == 1));
    }
}
