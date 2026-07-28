use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::contracts::{
    ProtectedBoundaryComplex, TetrahedronBoundaryFace, TetrahedronMesh, TopologyEntityId,
};
use runmat_meshing_plc::validate::validate_protected_boundary_complex;

mod evidence;
mod mesh_topology;

use evidence::build_queue_evidence;
use mesh_topology::{
    boundary_face_is_exterior, element_face_counts, plc_facets_adjacent_to_edge_have_exterior_face,
    tetrahedron_edges, tetrahedron_faces,
};

use super::{
    boundary_faces::boundary_face_source_edges,
    input_validation::validate_tetrahedron_recovery_input_mesh,
    material_interfaces::material_interface_recovery_topology,
    source_face_coverage::{
        boundary_source_face_area_coverage_complete,
        boundary_source_face_group_area_coverage_complete, recovery_geometry_tolerance,
        tetrahedron_node_coordinates,
    },
    topology::sorted_topology_ids,
    TetrahedronMaterialInterfaceTopology, TetrahedronProtectedEdgeTopology,
    TetrahedronRecoveryError, TetrahedronRecoveryKind, TetrahedronRecoveryQueue,
    TetrahedronRecoveryQueueItem, TetrahedronRecoveryStatus, TetrahedronSourceFaceTopology,
};
pub fn build_recovery_queue_from_plc(
    plc: &ProtectedBoundaryComplex,
    tetrahedron_mesh: &TetrahedronMesh,
) -> Result<TetrahedronRecoveryQueue, TetrahedronRecoveryError> {
    validate_protected_boundary_complex(plc)
        .map_err(|error| TetrahedronRecoveryError::InvalidProtectedBoundaryComplex { error })?;
    if tetrahedron_mesh.nodes.is_empty() || tetrahedron_mesh.elements.is_empty() {
        return Err(TetrahedronRecoveryError::EmptyTetrahedronMesh);
    }
    validate_tetrahedron_recovery_input_mesh(tetrahedron_mesh)?;

    let element_face_counts = element_face_counts(tetrahedron_mesh);
    let exterior_boundary_faces = tetrahedron_mesh
        .boundary_faces
        .iter()
        .filter(|face| boundary_face_is_exterior(face, &element_face_counts))
        .collect::<Vec<_>>();
    let node_coordinates = tetrahedron_node_coordinates(tetrahedron_mesh);
    let tolerance = recovery_geometry_tolerance(&node_coordinates);
    let recovered_boundary_faces = exterior_boundary_faces
        .iter()
        .map(|face| sorted_topology_ids(face.node_ids.clone()))
        .collect::<BTreeSet<_>>();
    let recovered_volume_faces = tetrahedron_mesh
        .elements
        .iter()
        .flat_map(|element| tetrahedron_faces(element.node_ids.clone()))
        .collect::<BTreeSet<_>>();
    let recovered_boundary_edges = exterior_boundary_faces
        .iter()
        .flat_map(|face| boundary_face_source_edges(face))
        .map(|(edge_key, _)| edge_key)
        .collect::<BTreeSet<_>>();
    let recovered_volume_edges = tetrahedron_mesh
        .elements
        .iter()
        .flat_map(|element| tetrahedron_edges(element.node_ids.clone()))
        .collect::<BTreeSet<_>>();
    let recovered_material_interfaces = tetrahedron_mesh
        .elements
        .iter()
        .map(|element| element.material_region_id.clone())
        .collect::<BTreeSet<_>>();
    let cad_curve_source_edge_ids = plc
        .protected_edges
        .iter()
        .filter(|edge| edge.cad_curve_boundary.is_some())
        .map(|edge| edge.source_edge_id.clone())
        .collect::<BTreeSet<_>>();

    let mut items = Vec::<TetrahedronRecoveryQueueItem>::new();
    for facet in &plc.facets {
        let face_node_ids = sorted_topology_ids(facet.node_ids.clone());
        let face_key = (facet.source_face_id.clone(), face_node_ids.clone());
        let source_face_boundary_complete = boundary_source_face_provenance_complete(
            &exterior_boundary_faces,
            &face_key.1,
            &face_key.0,
        ) || boundary_source_face_split_provenance_complete(
            &exterior_boundary_faces,
            &plc.protected_edges,
            facet,
        ) || boundary_source_face_area_coverage_complete(
            &exterior_boundary_faces,
            facet,
            &node_coordinates,
            tolerance,
        ) || (!recovered_boundary_faces.contains(&face_key.1)
            && boundary_source_face_group_area_coverage_complete(
                &exterior_boundary_faces,
                &plc.facets,
                &facet.source_face_id,
                &node_coordinates,
                tolerance,
            ));
        let source_face_topology =
            if recovered_boundary_faces.contains(&face_node_ids) || source_face_boundary_complete {
                TetrahedronSourceFaceTopology::BoundaryFace
            } else if element_face_counts.get(&face_node_ids).copied() == Some(1) {
                TetrahedronSourceFaceTopology::VolumeFace
            } else if recovered_volume_faces.contains(&face_node_ids) {
                TetrahedronSourceFaceTopology::InteriorFace
            } else {
                TetrahedronSourceFaceTopology::Absent
            };
        items.push(TetrahedronRecoveryQueueItem {
            item_id: format!("source_face:{}", facet.facet_id.id),
            kind: TetrahedronRecoveryKind::SourceFace,
            status: if source_face_boundary_complete {
                TetrahedronRecoveryStatus::Recovered
            } else {
                TetrahedronRecoveryStatus::Missing
            },
            source_entity_id: Some(facet.source_face_id.clone()),
            source_face_node_ids: Some(face_node_ids),
            source_face_topology: Some(source_face_topology),
            protected_edge_node_ids: None,
            protected_edge_topology: None,
            material_interface_topology: None,
            material_interface_id: None,
        });
    }

    for protected_edge in &plc.protected_edges {
        let edge_key = sorted_topology_ids(protected_edge.node_ids.clone());
        let source_edge_boundary_complete = protected_boundary_edge_provenance_complete(
            &exterior_boundary_faces,
            &edge_key,
            &protected_edge.source_edge_id,
        );
        let protected_edge_topology = if recovered_boundary_edges.contains(&edge_key)
            || source_edge_boundary_complete
        {
            TetrahedronProtectedEdgeTopology::BoundaryEdge
        } else if recovered_volume_edges.contains(&edge_key)
            && plc_facets_adjacent_to_edge_have_exterior_face(plc, &edge_key, &element_face_counts)
        {
            TetrahedronProtectedEdgeTopology::VolumeEdge
        } else if recovered_volume_edges.contains(&edge_key) {
            TetrahedronProtectedEdgeTopology::InteriorEdge
        } else {
            TetrahedronProtectedEdgeTopology::Absent
        };
        let status = if source_edge_boundary_complete {
            TetrahedronRecoveryStatus::Recovered
        } else {
            TetrahedronRecoveryStatus::Missing
        };
        items.push(TetrahedronRecoveryQueueItem {
            item_id: format!("source_edge:{}", protected_edge.edge_id.id),
            kind: TetrahedronRecoveryKind::SourceEdge,
            status,
            source_entity_id: Some(protected_edge.source_edge_id.clone()),
            source_face_node_ids: None,
            source_face_topology: None,
            protected_edge_node_ids: Some(edge_key),
            protected_edge_topology: Some(protected_edge_topology),
            material_interface_topology: None,
            material_interface_id: None,
        });
    }

    let material_interfaces = plc
        .facets
        .iter()
        .flat_map(|facet| facet.material_interface_ids.iter().cloned())
        .collect::<BTreeSet<_>>();
    for material_interface_id in &material_interfaces {
        let material_interface_topology = material_interface_recovery_topology(
            plc,
            tetrahedron_mesh,
            material_interface_id,
            &recovered_material_interfaces,
            &material_interfaces,
        );
        let status = if recovered_material_interfaces.contains(material_interface_id)
            && material_interface_topology == TetrahedronMaterialInterfaceTopology::AbsentPartition
        {
            TetrahedronRecoveryStatus::Recovered
        } else {
            TetrahedronRecoveryStatus::Missing
        };
        items.push(TetrahedronRecoveryQueueItem {
            item_id: format!("material_interface:{material_interface_id}"),
            kind: TetrahedronRecoveryKind::MaterialInterface,
            status,
            source_entity_id: None,
            source_face_node_ids: None,
            source_face_topology: None,
            protected_edge_node_ids: None,
            protected_edge_topology: None,
            material_interface_topology: if status == TetrahedronRecoveryStatus::Missing {
                Some(material_interface_topology)
            } else {
                None
            },
            material_interface_id: Some(material_interface_id.clone()),
        });
    }

    let evidence = build_queue_evidence(&items, &cad_curve_source_edge_ids);

    Ok(TetrahedronRecoveryQueue { items, evidence })
}

fn boundary_source_face_provenance_complete(
    exterior_boundary_faces: &[&TetrahedronBoundaryFace],
    face_key: &[TopologyEntityId; 3],
    source_face_id: &TopologyEntityId,
) -> bool {
    let matching_source_faces = exterior_boundary_faces
        .iter()
        .filter_map(|boundary_face| {
            (sorted_topology_ids(boundary_face.node_ids.clone()) == *face_key)
                .then_some(&boundary_face.source_face_id)
        })
        .collect::<Vec<_>>();

    !matching_source_faces.is_empty()
        && matching_source_faces
            .iter()
            .all(|boundary_source_face_id| *boundary_source_face_id == source_face_id)
}

fn protected_boundary_edge_provenance_complete(
    exterior_boundary_faces: &[&TetrahedronBoundaryFace],
    edge_key: &[TopologyEntityId; 2],
    source_edge_id: &TopologyEntityId,
) -> bool {
    let matching_edge_sources = exterior_boundary_faces
        .iter()
        .flat_map(|face| boundary_face_source_edges(face))
        .filter_map(|(boundary_edge_key, boundary_source_edge_id)| {
            (boundary_edge_key == *edge_key).then_some(boundary_source_edge_id)
        })
        .collect::<Vec<_>>();

    let exact_edge_complete = !matching_edge_sources.is_empty()
        && matching_edge_sources.iter().all(|boundary_source_edge_id| {
            boundary_source_edge_id.as_ref() == Some(source_edge_id)
        });
    exact_edge_complete
        || boundary_source_edge_chain(exterior_boundary_faces, edge_key, source_edge_id)
            .is_some_and(|chain| chain.len() >= 3)
}

fn boundary_source_face_split_provenance_complete(
    exterior_boundary_faces: &[&TetrahedronBoundaryFace],
    protected_edges: &[runmat_meshing_core::contracts::PlcProtectedEdge],
    facet: &runmat_meshing_core::contracts::PlcFacet,
) -> bool {
    protected_edges
        .iter()
        .filter(|protected_edge| {
            facet.node_ids.contains(&protected_edge.node_ids[0])
                && facet.node_ids.contains(&protected_edge.node_ids[1])
        })
        .any(|protected_edge| {
            let edge_key = sorted_topology_ids(protected_edge.node_ids.clone());
            let Some(chain) = boundary_source_edge_chain(
                exterior_boundary_faces,
                &edge_key,
                &protected_edge.source_edge_id,
            ) else {
                return false;
            };
            if chain.len() < 3 {
                return false;
            }
            let Some(opposite_node_id) = facet
                .node_ids
                .iter()
                .find(|node_id| !edge_key.contains(node_id))
            else {
                return false;
            };
            chain.windows(2).all(|segment| {
                let child_face = sorted_topology_ids([
                    segment[0].clone(),
                    segment[1].clone(),
                    opposite_node_id.clone(),
                ]);
                exterior_boundary_faces.iter().any(|boundary_face| {
                    boundary_face.source_face_id == facet.source_face_id
                        && sorted_topology_ids(boundary_face.node_ids.clone()) == child_face
                })
            })
        })
}

fn boundary_source_edge_chain(
    exterior_boundary_faces: &[&TetrahedronBoundaryFace],
    edge_key: &[TopologyEntityId; 2],
    source_edge_id: &TopologyEntityId,
) -> Option<Vec<TopologyEntityId>> {
    let mut adjacency = BTreeMap::<TopologyEntityId, BTreeSet<TopologyEntityId>>::new();
    for boundary_face in exterior_boundary_faces {
        for (face_edge, boundary_source_edge_id) in boundary_face_source_edges(boundary_face) {
            if boundary_source_edge_id.as_ref() != Some(source_edge_id) {
                continue;
            }
            adjacency
                .entry(face_edge[0].clone())
                .or_default()
                .insert(face_edge[1].clone());
            adjacency
                .entry(face_edge[1].clone())
                .or_default()
                .insert(face_edge[0].clone());
        }
    }
    boundary_source_edge_chain_path(&adjacency, edge_key)
}

fn boundary_source_edge_chain_path(
    adjacency: &BTreeMap<TopologyEntityId, BTreeSet<TopologyEntityId>>,
    edge_key: &[TopologyEntityId; 2],
) -> Option<Vec<TopologyEntityId>> {
    let mut queue = Vec::<Vec<TopologyEntityId>>::new();
    let mut visited = BTreeSet::<TopologyEntityId>::new();
    queue.push(vec![edge_key[0].clone()]);
    visited.insert(edge_key[0].clone());

    while let Some(path) = queue.pop() {
        let current = path.last()?;
        if current == &edge_key[1] {
            return (path.len() >= 2).then_some(path);
        }
        for next in adjacency
            .get(current)
            .into_iter()
            .flat_map(|nodes| nodes.iter())
        {
            if visited.insert(next.clone()) {
                let mut next_path = path.clone();
                next_path.push(next.clone());
                queue.insert(0, next_path);
            }
        }
    }
    None
}
