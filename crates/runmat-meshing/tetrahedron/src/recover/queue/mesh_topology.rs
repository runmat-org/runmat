use std::collections::BTreeMap;

use runmat_meshing_core::contracts::{
    ProtectedBoundaryComplex, TetrahedronBoundaryFace, TetrahedronMesh, TopologyEntityId,
};

use super::super::topology::sorted_topology_ids;

pub(super) fn boundary_face_is_exterior(
    boundary_face: &TetrahedronBoundaryFace,
    element_face_counts: &BTreeMap<[TopologyEntityId; 3], usize>,
) -> bool {
    element_face_counts
        .get(&sorted_topology_ids(boundary_face.node_ids.clone()))
        .copied()
        == Some(1)
}

pub(super) fn plc_facets_adjacent_to_edge_have_exterior_face(
    plc: &ProtectedBoundaryComplex,
    edge_key: &[TopologyEntityId; 2],
    element_face_counts: &BTreeMap<[TopologyEntityId; 3], usize>,
) -> bool {
    plc.facets
        .iter()
        .filter(|facet| {
            let face_node_ids = sorted_topology_ids(facet.node_ids.clone());
            face_node_ids.contains(&edge_key[0]) && face_node_ids.contains(&edge_key[1])
        })
        .any(|facet| {
            element_face_counts
                .get(&sorted_topology_ids(facet.node_ids.clone()))
                .copied()
                == Some(1)
        })
}

pub(super) fn element_face_counts(
    tetrahedron_mesh: &TetrahedronMesh,
) -> BTreeMap<[TopologyEntityId; 3], usize> {
    tetrahedron_mesh
        .elements
        .iter()
        .flat_map(|element| tetrahedron_faces(element.node_ids.clone()))
        .fold(
            BTreeMap::<[TopologyEntityId; 3], usize>::new(),
            |mut counts, face| {
                *counts.entry(face).or_default() += 1;
                counts
            },
        )
}

pub(super) fn tetrahedron_edges(node_ids: [TopologyEntityId; 4]) -> [[TopologyEntityId; 2]; 6] {
    [
        sorted_topology_ids([node_ids[0].clone(), node_ids[1].clone()]),
        sorted_topology_ids([node_ids[0].clone(), node_ids[2].clone()]),
        sorted_topology_ids([node_ids[0].clone(), node_ids[3].clone()]),
        sorted_topology_ids([node_ids[1].clone(), node_ids[2].clone()]),
        sorted_topology_ids([node_ids[1].clone(), node_ids[3].clone()]),
        sorted_topology_ids([node_ids[2].clone(), node_ids[3].clone()]),
    ]
}

pub(super) fn tetrahedron_faces(node_ids: [TopologyEntityId; 4]) -> [[TopologyEntityId; 3]; 4] {
    [
        sorted_topology_ids([
            node_ids[0].clone(),
            node_ids[1].clone(),
            node_ids[2].clone(),
        ]),
        sorted_topology_ids([
            node_ids[0].clone(),
            node_ids[1].clone(),
            node_ids[3].clone(),
        ]),
        sorted_topology_ids([
            node_ids[0].clone(),
            node_ids[2].clone(),
            node_ids[3].clone(),
        ]),
        sorted_topology_ids([
            node_ids[1].clone(),
            node_ids[2].clone(),
            node_ids[3].clone(),
        ]),
    ]
}
