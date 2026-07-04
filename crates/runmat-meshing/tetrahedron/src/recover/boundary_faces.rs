use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::contracts::{
    ProtectedBoundaryComplex, TetrahedronBoundaryFace, TetrahedronMesh, TopologyEntityId,
};

use crate::protected_edges::{face_edges, sorted_edge, source_edge_ids_for_face_edges};

use super::{
    topology::sorted_topology_ids, TetrahedronProtectedEdgeTopology, TetrahedronRecoveryKind,
    TetrahedronRecoveryQueue, TetrahedronRecoveryStatus, TetrahedronSourceFaceTopology,
};

pub(super) fn recover_missing_protected_edge_boundary_faces(
    plc: &ProtectedBoundaryComplex,
    initial_recovery_queue: &TetrahedronRecoveryQueue,
    tetrahedron_mesh: &mut TetrahedronMesh,
) -> usize {
    let element_face_counts = element_face_counts(tetrahedron_mesh);
    let mut boundary_face_keys = boundary_face_keys(tetrahedron_mesh);
    let recoverable_source_edges = initial_recovery_queue
        .items
        .iter()
        .filter(|item| {
            item.kind == TetrahedronRecoveryKind::SourceEdge
                && item.status == TetrahedronRecoveryStatus::Missing
                && item.protected_edge_topology
                    == Some(TetrahedronProtectedEdgeTopology::VolumeEdge)
        })
        .filter_map(|item| {
            Some((
                item.protected_edge_node_ids.clone()?,
                item.source_entity_id.clone()?,
            ))
        })
        .collect::<BTreeSet<_>>();

    let mut recovered_count = 0;
    for protected_edge in plc.protected_edges.iter().filter(|protected_edge| {
        recoverable_source_edges.contains(&(
            sorted_edge(protected_edge.node_ids.clone()),
            protected_edge.source_edge_id.clone(),
        ))
    }) {
        let protected_edge_key = sorted_edge(protected_edge.node_ids.clone());
        for facet in &plc.facets {
            if !facet_contains_edge(facet.node_ids.clone(), protected_edge_key.clone()) {
                continue;
            }
            if recover_facet_boundary_face(
                plc,
                &element_face_counts,
                &mut boundary_face_keys,
                tetrahedron_mesh,
                facet,
            ) {
                recovered_count += 1;
            }
        }
    }

    recovered_count
}

pub(super) fn recover_volume_face_source_face_boundary_faces(
    plc: &ProtectedBoundaryComplex,
    initial_recovery_queue: &TetrahedronRecoveryQueue,
    tetrahedron_mesh: &mut TetrahedronMesh,
) -> usize {
    let element_face_counts = element_face_counts(tetrahedron_mesh);
    let mut boundary_face_keys = boundary_face_keys(tetrahedron_mesh);
    let recoverable_source_faces = initial_recovery_queue
        .items
        .iter()
        .filter(|item| {
            item.kind == TetrahedronRecoveryKind::SourceFace
                && item.status == TetrahedronRecoveryStatus::Missing
                && item.source_face_topology == Some(TetrahedronSourceFaceTopology::VolumeFace)
        })
        .filter_map(|item| {
            Some((
                item.source_entity_id.clone()?,
                item.source_face_node_ids.clone()?,
            ))
        })
        .collect::<BTreeSet<_>>();
    let mut recovered_count = 0;

    for facet in plc.facets.iter().filter(|facet| {
        recoverable_source_faces.contains(&(
            facet.source_face_id.clone(),
            sorted_topology_ids(facet.node_ids.clone()),
        ))
    }) {
        if recover_facet_boundary_face(
            plc,
            &element_face_counts,
            &mut boundary_face_keys,
            tetrahedron_mesh,
            facet,
        ) {
            recovered_count += 1;
        }
    }

    recovered_count
}

pub(super) fn repair_boundary_source_face_provenance(
    initial_recovery_queue: &TetrahedronRecoveryQueue,
    tetrahedron_mesh: &mut TetrahedronMesh,
) -> usize {
    let recoverable_source_faces = initial_recovery_queue
        .items
        .iter()
        .filter(|item| {
            item.kind == TetrahedronRecoveryKind::SourceFace
                && item.status == TetrahedronRecoveryStatus::Missing
                && item.source_face_topology == Some(TetrahedronSourceFaceTopology::BoundaryFace)
        })
        .filter_map(|item| {
            Some((
                item.source_face_node_ids.clone()?,
                item.source_entity_id.clone()?,
            ))
        })
        .collect::<BTreeMap<_, _>>();

    let mut repaired_count = 0;
    for boundary_face in &mut tetrahedron_mesh.boundary_faces {
        let face_key = sorted_topology_ids(boundary_face.node_ids.clone());
        let Some(expected_source_face_id) = recoverable_source_faces.get(&face_key) else {
            continue;
        };
        if &boundary_face.source_face_id != expected_source_face_id {
            boundary_face.source_face_id = expected_source_face_id.clone();
            repaired_count += 1;
        }
    }
    repaired_count
}

pub(super) fn repair_boundary_face_identity(
    plc: &ProtectedBoundaryComplex,
    tetrahedron_mesh: &mut TetrahedronMesh,
) -> usize {
    let expected_face_id_by_nodes = plc
        .facets
        .iter()
        .map(|facet| {
            (
                sorted_topology_ids(facet.node_ids.clone()),
                facet.facet_id.clone(),
            )
        })
        .collect::<BTreeMap<_, _>>();

    let mut repaired_count = 0;
    for boundary_face in &mut tetrahedron_mesh.boundary_faces {
        let face_key = sorted_topology_ids(boundary_face.node_ids.clone());
        let Some(expected_face_id) = expected_face_id_by_nodes.get(&face_key) else {
            continue;
        };
        if &boundary_face.face_id != expected_face_id {
            boundary_face.face_id = expected_face_id.clone();
            repaired_count += 1;
        }
    }
    repaired_count
}

pub(super) fn repair_boundary_source_edge_provenance(
    plc: &ProtectedBoundaryComplex,
    tetrahedron_mesh: &mut TetrahedronMesh,
) -> usize {
    let mut repaired_count = 0;
    for boundary_face in &mut tetrahedron_mesh.boundary_faces {
        let expected_source_edge_ids =
            source_edge_ids_for_face_edges(&plc.protected_edges, boundary_face.node_ids.clone());
        for (edge_index, expected_source_edge_id) in
            expected_source_edge_ids.into_iter().enumerate()
        {
            if boundary_face.source_edge_ids[edge_index] != expected_source_edge_id {
                boundary_face.source_edge_ids[edge_index] = expected_source_edge_id;
                repaired_count += 1;
            }
        }
    }
    repaired_count
}

pub(super) fn boundary_face_source_edges(
    face: &TetrahedronBoundaryFace,
) -> Vec<([TopologyEntityId; 2], Option<TopologyEntityId>)> {
    face_edges(face.node_ids.clone())
        .into_iter()
        .zip(face.source_edge_ids.clone())
        .collect()
}

fn recover_facet_boundary_face(
    plc: &ProtectedBoundaryComplex,
    element_face_counts: &BTreeMap<[TopologyEntityId; 3], usize>,
    boundary_face_keys: &mut BTreeSet<[TopologyEntityId; 3]>,
    tetrahedron_mesh: &mut TetrahedronMesh,
    facet: &runmat_meshing_core::contracts::PlcFacet,
) -> bool {
    let face_key = sorted_topology_ids(facet.node_ids.clone());
    if boundary_face_keys.contains(&face_key) {
        return false;
    }
    if element_face_counts.get(&face_key).copied() != Some(1) {
        return false;
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
    boundary_face_keys.insert(face_key);
    true
}

fn facet_contains_edge(node_ids: [TopologyEntityId; 3], edge_key: [TopologyEntityId; 2]) -> bool {
    face_edges(node_ids).contains(&edge_key)
}

fn boundary_face_keys(tetrahedron_mesh: &TetrahedronMesh) -> BTreeSet<[TopologyEntityId; 3]> {
    tetrahedron_mesh
        .boundary_faces
        .iter()
        .map(|face| sorted_topology_ids(face.node_ids.clone()))
        .collect()
}

fn element_face_counts(
    tetrahedron_mesh: &TetrahedronMesh,
) -> BTreeMap<[TopologyEntityId; 3], usize> {
    tetrahedron_mesh
        .elements
        .iter()
        .flat_map(|element| tetrahedron_element_faces(element.node_ids.clone()))
        .fold(
            BTreeMap::<[TopologyEntityId; 3], usize>::new(),
            |mut counts, face| {
                *counts.entry(face).or_default() += 1;
                counts
            },
        )
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
