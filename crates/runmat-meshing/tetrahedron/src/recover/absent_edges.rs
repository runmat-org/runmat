use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    contracts::{
        PlcFacet, ProtectedBoundaryComplex, Tetrahedron4Element, TetrahedronBoundaryFace,
        TetrahedronMesh, TopologyEntityId,
    },
    quality::predicate::{tetrahedron_scaled_jacobian, tetrahedron_signed_volume},
};

use crate::protected_edges::{face_edges, sorted_edge, source_edge_ids_for_face_edges};

use super::{
    topology::sorted_topology_ids, TetrahedronProtectedEdgeTopology, TetrahedronRecoveryKind,
    TetrahedronRecoveryQueue, TetrahedronRecoveryStatus,
};

const MIN_RECOVERED_TETRAHEDRON_VOLUME_M3: f64 = 1.0e-18;
const MIN_RECOVERED_TETRAHEDRON_SCALED_JACOBIAN: f64 = 1.0e-8;

pub(super) struct AbsentSourceEdgeRecovery {
    pub source_edge_count: usize,
    pub boundary_face_count: usize,
}

pub(super) fn recover_absent_protected_edges_by_boundary_diagonal_flip(
    plc: &ProtectedBoundaryComplex,
    initial_recovery_queue: &TetrahedronRecoveryQueue,
    tetrahedron_mesh: &mut TetrahedronMesh,
) -> AbsentSourceEdgeRecovery {
    let recoverable_source_edges = initial_recovery_queue
        .items
        .iter()
        .filter(|item| {
            item.kind == TetrahedronRecoveryKind::SourceEdge
                && item.status == TetrahedronRecoveryStatus::Missing
                && item.protected_edge_topology == Some(TetrahedronProtectedEdgeTopology::Absent)
        })
        .filter_map(|item| {
            Some((
                item.protected_edge_node_ids.clone()?,
                item.source_entity_id.clone()?,
            ))
        })
        .collect::<BTreeSet<_>>();

    let mut recovered = AbsentSourceEdgeRecovery {
        source_edge_count: 0,
        boundary_face_count: 0,
    };
    for protected_edge in plc.protected_edges.iter().filter(|protected_edge| {
        recoverable_source_edges.contains(&(
            sorted_edge(protected_edge.node_ids.clone()),
            protected_edge.source_edge_id.clone(),
        ))
    }) {
        if let Some(boundary_face_count) = recover_absent_protected_edge_by_boundary_diagonal_flip(
            plc,
            tetrahedron_mesh,
            sorted_edge(protected_edge.node_ids.clone()),
        ) {
            recovered.source_edge_count += 1;
            recovered.boundary_face_count += boundary_face_count;
        }
    }

    recovered
}

fn recover_absent_protected_edge_by_boundary_diagonal_flip(
    plc: &ProtectedBoundaryComplex,
    tetrahedron_mesh: &mut TetrahedronMesh,
    protected_edge: [TopologyEntityId; 2],
) -> Option<usize> {
    let adjacent_facets = plc
        .facets
        .iter()
        .filter(|facet| face_contains_edge(facet.node_ids.clone(), protected_edge.clone()))
        .collect::<Vec<_>>();
    if adjacent_facets.len() != 2 {
        return None;
    }

    let opposite_nodes = adjacent_facets
        .iter()
        .filter_map(|facet| {
            facet
                .node_ids
                .iter()
                .find(|node_id| !protected_edge.contains(node_id))
                .cloned()
        })
        .collect::<Vec<_>>();
    if opposite_nodes.len() != 2 || opposite_nodes[0] == opposite_nodes[1] {
        return None;
    }

    let current_boundary_face_keys = current_boundary_face_keys(&protected_edge, &opposite_nodes);
    if !current_boundary_face_keys
        .iter()
        .all(|face_key| boundary_face_exists(tetrahedron_mesh, face_key))
    {
        return None;
    }

    let element_face_index = element_index_by_face(tetrahedron_mesh);
    let left_element_index = *element_face_index.get(&current_boundary_face_keys[0])?;
    let right_element_index = *element_face_index.get(&current_boundary_face_keys[1])?;
    if left_element_index == right_element_index {
        return None;
    }

    let left_element = tetrahedron_mesh.elements[left_element_index].clone();
    let right_element = tetrahedron_mesh.elements[right_element_index].clone();
    if left_element.material_region_id != right_element.material_region_id {
        return None;
    }

    let support_node = shared_support_node(
        &left_element,
        &right_element,
        &protected_edge,
        &opposite_nodes,
    )?;
    let node_coordinates = node_coordinates(tetrahedron_mesh);
    let recovered_left = recovered_element_for_facet(
        adjacent_facets[0],
        support_node.clone(),
        &left_element,
        &node_coordinates,
    )?;
    let recovered_right = recovered_element_for_facet(
        adjacent_facets[1],
        support_node,
        &right_element,
        &node_coordinates,
    )?;

    tetrahedron_mesh.elements[left_element_index] = recovered_left;
    tetrahedron_mesh.elements[right_element_index] = recovered_right;
    tetrahedron_mesh.boundary_faces.retain(|face| {
        !current_boundary_face_keys.contains(&sorted_topology_ids(face.node_ids.clone()))
    });
    let mut inserted_boundary_faces = 0;
    let mut boundary_face_keys = tetrahedron_mesh
        .boundary_faces
        .iter()
        .map(|face| sorted_topology_ids(face.node_ids.clone()))
        .collect::<BTreeSet<_>>();
    for facet in adjacent_facets {
        let face_key = sorted_topology_ids(facet.node_ids.clone());
        if !boundary_face_keys.insert(face_key) {
            continue;
        }
        tetrahedron_mesh
            .boundary_faces
            .push(TetrahedronBoundaryFace {
                face_id: facet.facet_id.clone(),
                node_ids: facet.node_ids.clone(),
                source_face_id: facet.source_face_id.clone(),
                source_edge_ids: source_edge_ids_for_face_edges(
                    &plc.protected_edges,
                    facet.node_ids.clone(),
                ),
            });
        inserted_boundary_faces += 1;
    }

    Some(inserted_boundary_faces)
}

fn current_boundary_face_keys(
    protected_edge: &[TopologyEntityId; 2],
    opposite_nodes: &[TopologyEntityId],
) -> [[TopologyEntityId; 3]; 2] {
    [
        sorted_topology_ids([
            protected_edge[0].clone(),
            opposite_nodes[0].clone(),
            opposite_nodes[1].clone(),
        ]),
        sorted_topology_ids([
            protected_edge[1].clone(),
            opposite_nodes[0].clone(),
            opposite_nodes[1].clone(),
        ]),
    ]
}

fn boundary_face_exists(
    tetrahedron_mesh: &TetrahedronMesh,
    face_key: &[TopologyEntityId; 3],
) -> bool {
    tetrahedron_mesh
        .boundary_faces
        .iter()
        .any(|face| sorted_topology_ids(face.node_ids.clone()) == *face_key)
}

fn element_index_by_face(
    tetrahedron_mesh: &TetrahedronMesh,
) -> BTreeMap<[TopologyEntityId; 3], usize> {
    let mut index_by_face = BTreeMap::<[TopologyEntityId; 3], usize>::new();
    let mut duplicate_faces = BTreeSet::<[TopologyEntityId; 3]>::new();
    for (element_index, element) in tetrahedron_mesh.elements.iter().enumerate() {
        for face in tetrahedron_element_faces(element.node_ids.clone()) {
            if index_by_face.insert(face.clone(), element_index).is_some() {
                duplicate_faces.insert(face);
            }
        }
    }
    for duplicate_face in duplicate_faces {
        index_by_face.remove(&duplicate_face);
    }
    index_by_face
}

fn shared_support_node(
    left_element: &Tetrahedron4Element,
    right_element: &Tetrahedron4Element,
    protected_edge: &[TopologyEntityId; 2],
    opposite_nodes: &[TopologyEntityId],
) -> Option<TopologyEntityId> {
    let excluded = protected_edge
        .iter()
        .chain(opposite_nodes.iter())
        .collect::<BTreeSet<_>>();
    let left_support = left_element
        .node_ids
        .iter()
        .find(|node_id| !excluded.contains(node_id))?;
    let right_support = right_element
        .node_ids
        .iter()
        .find(|node_id| !excluded.contains(node_id))?;
    (left_support == right_support).then(|| left_support.clone())
}

fn recovered_element_for_facet(
    facet: &PlcFacet,
    support_node: TopologyEntityId,
    original_element: &Tetrahedron4Element,
    node_coordinates: &BTreeMap<TopologyEntityId, [f64; 3]>,
) -> Option<Tetrahedron4Element> {
    let node_ids = [
        facet.node_ids[0].clone(),
        facet.node_ids[1].clone(),
        facet.node_ids[2].clone(),
        support_node,
    ];
    let oriented_node_ids = orient_recovered_tetrahedron(node_ids, node_coordinates)?;
    Some(Tetrahedron4Element {
        element_id: original_element.element_id.clone(),
        node_ids: oriented_node_ids,
        material_region_id: original_element.material_region_id.clone(),
    })
}

fn orient_recovered_tetrahedron(
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
    if !signed_volume.is_finite() || signed_volume <= MIN_RECOVERED_TETRAHEDRON_VOLUME_M3 {
        return None;
    }
    let scaled_jacobian = tetrahedron_scaled_jacobian(points);
    if !scaled_jacobian.is_finite() || scaled_jacobian < MIN_RECOVERED_TETRAHEDRON_SCALED_JACOBIAN {
        return None;
    }
    Some(node_ids)
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

fn node_coordinates(tetrahedron_mesh: &TetrahedronMesh) -> BTreeMap<TopologyEntityId, [f64; 3]> {
    tetrahedron_mesh
        .nodes
        .iter()
        .map(|node| (node.node_id.clone(), node.coordinates_m))
        .collect()
}

fn face_contains_edge(node_ids: [TopologyEntityId; 3], edge_key: [TopologyEntityId; 2]) -> bool {
    face_edges(node_ids).contains(&edge_key)
}

fn tetrahedron_element_faces(node_ids: [TopologyEntityId; 4]) -> [[TopologyEntityId; 3]; 4] {
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
