//! Small conforming multi-element tetrahedron for physical solver evidence.

use std::collections::BTreeMap;

use crate::{
    solver_midside_node_identity, solver_volume_element_identity,
    sort_solver_node_exact_parameters, BoundaryEdgeOrder, BoundaryTriangleOrder, ElementOrder,
    FieldTopologyLocation, MeshNeighbor, PersistentEntityId, PersistentEntityKind,
    SolverMeshArtifact, SolverMeshNode, SolverNodeExactParameter, StableDigest,
    TETRAHEDRON_MIDSIDE_EDGE_CORNERS,
};

use super::solver_mesh::canonical_tetrahedron_solver_mesh;

/// Four conforming Tet4 children formed by splitting the canonical tetrahedron at its centroid.
pub fn connected_tetrahedron_solver_mesh() -> SolverMeshArtifact {
    let mut artifact = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let topo = &mut *artifact.topology;
    let parent = topo.volume_elements[0].clone();
    let centroid_id = 5;
    topo.nodes.push(SolverMeshNode {
        node_id: centroid_id,
        stable_identity: StableDigest::from_bytes([42; 32]),
        coordinates_m:
            runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new([
                0.25, 0.25, 0.25,
            ]),
        provenance: parent.provenance.clone(),
        exact_parameters: Vec::new(),
    });
    let children = [[5, 2, 3, 4], [1, 5, 3, 4], [1, 2, 5, 4], [1, 2, 3, 5]];
    topo.volume_elements = children
        .into_iter()
        .enumerate()
        .map(|(index, node_ids)| {
            let mut child = parent.clone();
            child.element_id = index as u64 + 1;
            child.node_ids = node_ids.to_vec();
            child.stable_identity = solver_volume_element_identity(
                node_ids.map(|node_id| topo.nodes[node_id as usize - 1].stable_identity),
            );
            child
        })
        .collect();
    for face in &mut topo.boundary_faces {
        let corners = &face.node_ids[..3];
        face.adjacent_volume_element_ids = topo
            .volume_elements
            .iter()
            .filter(|element| corners.iter().all(|node| element.node_ids.contains(node)))
            .map(|element| element.element_id)
            .collect();
    }
    topo.neighbors = topo
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
                    adjacent_element_id: topo
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
    topo.regions[0].element_ids = (1..=4).collect();
    for topology in &mut topo.field_topologies {
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

/// Straight-sided Tet10 elevation of the conforming four-element fixture.
/// Every shared volume edge receives exactly one midside node.
pub fn connected_quadratic_tetrahedron_solver_mesh() -> SolverMeshArtifact {
    elevate_connected_tet4_fixture(connected_tetrahedron_solver_mesh())
}

/// Elevate a refined canonical connected-Tet4 fixture, retaining its exterior
/// four source faces and their unit-tetrahedron parameter charts.
pub fn elevate_connected_tet4_fixture(mut artifact: SolverMeshArtifact) -> SolverMeshArtifact {
    let topo = &mut *artifact.topology;
    assert_eq!(artifact.resolved_request.element_order, ElementOrder::Tet4);
    assert_eq!(topo.boundary_faces.len(), 4);
    assert!(topo
        .volume_elements
        .iter()
        .all(|element| element.order == ElementOrder::Tet4));
    let mut source_faces_by_edge: BTreeMap<[u64; 2], Vec<(usize, PersistentEntityId)>> =
        BTreeMap::new();
    for (face_index, face) in topo.boundary_faces.iter().enumerate() {
        for local_edge in [[0, 1], [1, 2], [2, 0]] {
            let mut edge = [face.node_ids[local_edge[0]], face.node_ids[local_edge[1]]];
            edge.sort_unstable();
            source_faces_by_edge.entry(edge).or_default().extend(
                face.provenance
                    .iter()
                    .filter(|source| source.kind == PersistentEntityKind::Face)
                    .map(|source| (face_index, source.clone())),
            );
        }
    }
    for source_faces in source_faces_by_edge.values_mut() {
        source_faces.sort();
        source_faces.dedup();
    }

    let solid = topo.volume_elements[0].provenance[0].clone();
    let mut midpoint_by_edge = BTreeMap::new();
    for element in &topo.volume_elements {
        for local_edge in TETRAHEDRON_MIDSIDE_EDGE_CORNERS {
            let mut edge = [
                element.node_ids[local_edge[0]],
                element.node_ids[local_edge[1]],
            ];
            edge.sort_unstable();
            if midpoint_by_edge.contains_key(&edge) {
                continue;
            }
            let endpoints = edge.map(|id| &topo.nodes[id as usize - 1]);
            let mut provenance = vec![solid.clone()];
            let mut exact_parameters = Vec::new();
            if let Some(source_faces) = source_faces_by_edge.get(&edge) {
                for (face_index, source_face_id) in source_faces {
                    provenance.push(source_face_id.clone());
                    let midpoint = std::array::from_fn::<_, 3, _>(|axis| {
                        (endpoints[0].coordinates_m.canonical_value()[axis]
                            + endpoints[1].coordinates_m.canonical_value()[axis])
                            * 0.5
                    });
                    let evaluator_uv = match face_index {
                        0 | 3 => [midpoint[0], midpoint[1]],
                        1 => [midpoint[0], midpoint[2]],
                        2 => [midpoint[1], midpoint[2]],
                        _ => unreachable!("canonical tetrahedron has four source faces"),
                    };
                    exact_parameters.push(SolverNodeExactParameter::Surface {
                        source_face_id: source_face_id.clone(),
                        chart_id: StableDigest::from_bytes([*face_index as u8 + 10; 32]),
                        evaluator_uv,
                    });
                }
            }
            provenance.sort();
            provenance.dedup();
            sort_solver_node_exact_parameters(&mut exact_parameters);
            let node_id = topo.nodes.len() as u64 + 1;
            topo.nodes.push(SolverMeshNode {
                node_id,
                stable_identity: solver_midside_node_identity(
                    endpoints.map(|node| node.stable_identity),
                ),
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(std::array::from_fn(|axis| {
                    (endpoints[0].coordinates_m.canonical_value()[axis]
                        + endpoints[1].coordinates_m.canonical_value()[axis])
                        * 0.5
                })),
                provenance,
                exact_parameters,
            });
            midpoint_by_edge.insert(edge, node_id);
        }
    }
    let midpoint = |a: u64, b: u64| {
        let mut edge = [a, b];
        edge.sort_unstable();
        midpoint_by_edge[&edge]
    };
    for element in &mut topo.volume_elements {
        let corners = element.node_ids.clone();
        element.node_ids.extend(
            TETRAHEDRON_MIDSIDE_EDGE_CORNERS.map(|[a, b]| midpoint(corners[a], corners[b])),
        );
        element.order = ElementOrder::Tet10;
    }
    for face in &mut topo.boundary_faces {
        let corners = face.node_ids.clone();
        face.node_ids
            .extend([[0, 1], [1, 2], [2, 0]].map(|[a, b]| midpoint(corners[a], corners[b])));
        face.order = BoundaryTriangleOrder::Tri6;
    }
    for edge in &mut topo.boundary_edges {
        edge.node_ids
            .push(midpoint(edge.node_ids[0], edge.node_ids[1]));
        edge.order = BoundaryEdgeOrder::Line3;
    }
    artifact.resolved_request.element_order = ElementOrder::Tet10;
    artifact.resolved_request.limits.maximum_nodes = artifact
        .resolved_request
        .limits
        .maximum_nodes
        .max(topo.nodes.len() as u64);
    for topology in &mut topo.field_topologies {
        if topology.location == FieldTopologyLocation::Node {
            topology.ordered_entity_ids = (1..=topo.nodes.len() as u64).collect();
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

    #[test]
    fn connected_tet10_fixture_shares_midsides_and_preserves_exterior() {
        let mesh = connected_quadratic_tetrahedron_solver_mesh();
        mesh.validate_canonical().unwrap();
        assert_eq!(mesh.topology.nodes.len(), 15);
        assert_eq!(mesh.topology.volume_elements.len(), 4);
        assert_eq!(mesh.topology.boundary_faces.len(), 4);
        assert!(mesh
            .topology
            .volume_elements
            .iter()
            .all(|element| element.node_ids.len() == 10));
    }
}
