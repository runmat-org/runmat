use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    contracts::{TetrahedronMesh, TopologyEntityId},
    quality::predicate::{tetrahedron_scaled_jacobian, tetrahedron_signed_volume},
};

use crate::recover::topology::sorted_topology_ids;

const MIN_PARTITION_TETRAHEDRON_VOLUME_M3: f64 = 1.0e-18;
const MIN_PARTITION_TETRAHEDRON_SCALED_JACOBIAN: f64 = 1.0e-8;

pub(super) fn volume_face_counts(
    tetrahedron_mesh: &TetrahedronMesh,
) -> BTreeMap<[TopologyEntityId; 3], usize> {
    tetrahedron_mesh
        .elements
        .iter()
        .flat_map(|element| tetrahedron_faces(element.node_ids.clone()))
        .fold(
            BTreeMap::<[TopologyEntityId; 3], usize>::new(),
            |mut counts, face_key| {
                *counts.entry(face_key).or_default() += 1;
                counts
            },
        )
}

pub(super) fn element_exists(
    tetrahedron_mesh: &TetrahedronMesh,
    node_ids: &[TopologyEntityId; 4],
) -> bool {
    let mut candidate_node_ids = node_ids.clone();
    candidate_node_ids.sort();
    tetrahedron_mesh.elements.iter().any(|element| {
        let mut element_node_ids = element.node_ids.clone();
        element_node_ids.sort();
        element_node_ids == candidate_node_ids
    })
}

pub(super) fn orient_partition_tetrahedron(
    mut node_ids: [TopologyEntityId; 4],
    node_coordinates: &BTreeMap<TopologyEntityId, [f64; 3]>,
) -> Option<[TopologyEntityId; 4]> {
    let mut points = points_for_nodes(node_ids.clone(), node_coordinates)?;
    let mut signed_volume = tetrahedron_signed_volume(points);
    if signed_volume < 0.0 {
        node_ids.swap(1, 2);
        points = points_for_nodes(node_ids.clone(), node_coordinates)?;
        signed_volume = -signed_volume;
    }
    if !signed_volume.is_finite() || signed_volume <= MIN_PARTITION_TETRAHEDRON_VOLUME_M3 {
        return None;
    }
    let scaled_jacobian = tetrahedron_scaled_jacobian(points);
    if !scaled_jacobian.is_finite() || scaled_jacobian < MIN_PARTITION_TETRAHEDRON_SCALED_JACOBIAN {
        return None;
    }
    Some(node_ids)
}

pub(super) fn node_coordinates(
    tetrahedron_mesh: &TetrahedronMesh,
) -> BTreeMap<TopologyEntityId, [f64; 3]> {
    tetrahedron_mesh
        .nodes
        .iter()
        .map(|node| (node.node_id.clone(), node.coordinates_m))
        .collect()
}

pub(super) fn tetrahedron_faces(
    node_ids: [TopologyEntityId; 4],
) -> BTreeSet<[TopologyEntityId; 3]> {
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
    .into_iter()
    .collect()
}

fn points_for_nodes(
    node_ids: [TopologyEntityId; 4],
    node_coordinates: &BTreeMap<TopologyEntityId, [f64; 3]>,
) -> Option<[[f64; 3]; 4]> {
    Some([
        *node_coordinates.get(&node_ids[0])?,
        *node_coordinates.get(&node_ids[1])?,
        *node_coordinates.get(&node_ids[2])?,
        *node_coordinates.get(&node_ids[3])?,
    ])
}
