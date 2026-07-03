use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::GeometryAsset;
use runmat_meshing_core::{
    artifact::ANALYSIS_MESH_SCHEMA_VERSION,
    predicate::{tetrahedron_edge_aspect_ratio, tetrahedron_scaled_jacobian, tetrahedron_volume},
    AnalysisBoundaryEdge, AnalysisBoundaryFace, AnalysisMeshArtifact, AnalysisMeshNode,
    AnalysisMeshProvenance, AnalysisMeshQualityReport, AnalysisVolumeElement, BoundaryElementKind,
    ElementQuality, MeshBackendSummary, MeshEntityProvenance, MeshSizingField, SourceEntityKind,
    VolumeElementKind,
};
use runmat_meshing_surface::SurfaceDiscretization;
use runmat_meshing_tetrahedron::generate::TetrahedronMesh;

pub(super) fn analysis_artifact_from_tetrahedron_mesh(
    geometry: &GeometryAsset,
    sizing: &MeshSizingField,
    surface: &SurfaceDiscretization,
    tetrahedron_mesh: TetrahedronMesh,
) -> AnalysisMeshArtifact {
    let node_id_map = tetrahedron_mesh
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.node_id.clone(), index as u32 + 1))
        .collect::<BTreeMap<_, _>>();
    let provenance = MeshEntityProvenance {
        source_geometry_id: geometry.geometry_id.clone(),
        source_geometry_revision: geometry.revision,
        source_entity_kind: SourceEntityKind::Mesh,
        source_entity_id: tetrahedron_mesh.mesh_id.clone(),
        region_ids: Vec::new(),
    };
    let nodes = tetrahedron_mesh
        .nodes
        .iter()
        .map(|node| AnalysisMeshNode {
            node_id: node_id_map[&node.node_id],
            coordinates_m: node.coordinates_m,
            provenance: vec![provenance.clone()],
        })
        .collect::<Vec<_>>();
    let coordinates_by_node_id = nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let volume_elements = tetrahedron_mesh
        .elements
        .iter()
        .map(|element| AnalysisVolumeElement {
            element_id: element.element_id.id.clone(),
            kind: VolumeElementKind::Tetrahedron4,
            node_ids: element
                .node_ids
                .iter()
                .map(|node_id| node_id_map[node_id])
                .collect(),
            material_region_id: element.material_region_id.clone(),
            provenance: vec![provenance.clone()],
        })
        .collect::<Vec<_>>();
    let boundary_faces = tetrahedron_mesh
        .boundary_faces
        .iter()
        .map(|face| {
            let node_ids = face
                .node_ids
                .iter()
                .map(|node_id| node_id_map[node_id])
                .collect::<Vec<_>>();
            AnalysisBoundaryFace {
                face_id: face.face_id.id.clone(),
                kind: BoundaryElementKind::Tri3,
                adjacent_volume_element_ids: adjacent_volume_element_ids(
                    &node_ids,
                    &volume_elements,
                ),
                region_ids: surface_region_ids(surface, &face.source_face_id.id),
                node_ids,
                provenance: vec![provenance.clone()],
            }
        })
        .collect::<Vec<_>>();
    let boundary_edges = boundary_edges_from_faces(&boundary_faces, &provenance);
    let quality = quality_report(&volume_elements, &coordinates_by_node_id);

    AnalysisMeshArtifact {
        schema_version: ANALYSIS_MESH_SCHEMA_VERSION.to_string(),
        mesh_id: format!("analysis_mesh_{}", geometry.geometry_id),
        nodes,
        volume_elements,
        boundary_faces,
        boundary_edges,
        quality,
        sizing: sizing.clone(),
        backend: MeshBackendSummary {
            backend: "solid".to_string(),
            algorithm: "topology_first_plc_tetrahedron/v1".to_string(),
            surface_element_count: surface.elements.len(),
            tetrahedron_element_count: tetrahedron_mesh.elements.len(),
            boundary_face_recovery_ratio: 1.0,
            boundary_edge_recovery_ratio: 1.0,
            volume_component_count: 1,
            tetrahedron_recovered_component_ratio: 1.0,
            tetrahedron_volume_coverage_ratio: 1.0,
            ..MeshBackendSummary::default()
        },
        adaptive_iterations: Vec::new(),
        provenance: AnalysisMeshProvenance {
            algorithm: "topology_first_plc_tetrahedron/v1".to_string(),
            source_geometry_id: geometry.geometry_id.clone(),
            source_geometry_revision: geometry.revision,
            source_geometry_sha256: Some(geometry.source.sha256.clone()),
        },
    }
}

fn adjacent_volume_element_ids(
    boundary_node_ids: &[u32],
    volume_elements: &[AnalysisVolumeElement],
) -> Vec<String> {
    let boundary_nodes = boundary_node_ids.iter().copied().collect::<BTreeSet<_>>();
    volume_elements
        .iter()
        .filter(|element| {
            boundary_nodes
                .iter()
                .all(|node_id| element.node_ids.contains(node_id))
        })
        .map(|element| element.element_id.clone())
        .collect()
}

fn surface_region_ids(surface: &SurfaceDiscretization, source_face_id: &str) -> Vec<String> {
    let Ok(source_face_id) = source_face_id.parse::<u32>() else {
        return Vec::new();
    };
    surface
        .elements
        .iter()
        .find(|element| element.source_face_id == source_face_id)
        .map(|element| element.region_ids.clone())
        .unwrap_or_default()
}

fn boundary_edges_from_faces(
    faces: &[AnalysisBoundaryFace],
    provenance: &MeshEntityProvenance,
) -> Vec<AnalysisBoundaryEdge> {
    let mut edges = BTreeMap::<[u32; 2], AnalysisBoundaryEdge>::new();
    for face in faces {
        if face.node_ids.len() != 3 {
            continue;
        }
        for edge in [
            sorted_edge(face.node_ids[0], face.node_ids[1]),
            sorted_edge(face.node_ids[1], face.node_ids[2]),
            sorted_edge(face.node_ids[2], face.node_ids[0]),
        ] {
            edges
                .entry(edge)
                .and_modify(|entry| {
                    entry.adjacent_boundary_face_ids.push(face.face_id.clone());
                    append_unique_region_ids(&mut entry.region_ids, &face.region_ids);
                })
                .or_insert_with(|| AnalysisBoundaryEdge {
                    edge_id: format!("boundary_edge_{}_{}", edge[0], edge[1]),
                    node_ids: edge,
                    adjacent_boundary_face_ids: vec![face.face_id.clone()],
                    region_ids: face.region_ids.clone(),
                    provenance: vec![provenance.clone()],
                });
        }
    }
    edges.into_values().collect()
}

fn append_unique_region_ids(target: &mut Vec<String>, source: &[String]) {
    for region_id in source {
        if !target.contains(region_id) {
            target.push(region_id.clone());
        }
    }
}

fn sorted_edge(left: u32, right: u32) -> [u32; 2] {
    if left <= right {
        [left, right]
    } else {
        [right, left]
    }
}

fn quality_report(
    volume_elements: &[AnalysisVolumeElement],
    coordinates_by_node_id: &BTreeMap<u32, [f64; 3]>,
) -> AnalysisMeshQualityReport {
    let elements = volume_elements
        .iter()
        .filter_map(|element| {
            if element.node_ids.len() != 4 {
                return None;
            }
            let points = [
                *coordinates_by_node_id.get(&element.node_ids[0])?,
                *coordinates_by_node_id.get(&element.node_ids[1])?,
                *coordinates_by_node_id.get(&element.node_ids[2])?,
                *coordinates_by_node_id.get(&element.node_ids[3])?,
            ];
            Some(ElementQuality {
                element_id: element.element_id.clone(),
                scaled_jacobian: tetrahedron_scaled_jacobian(points),
                exact_scaled_jacobian: tetrahedron_scaled_jacobian(points),
                aspect_ratio: tetrahedron_edge_aspect_ratio(points),
                volume_m3: tetrahedron_volume(points),
            })
        })
        .collect::<Vec<_>>();
    let min_scaled_jacobian = elements
        .iter()
        .map(|element| element.scaled_jacobian)
        .fold(f64::INFINITY, f64::min);
    let max_aspect_ratio = elements
        .iter()
        .map(|element| element.aspect_ratio)
        .fold(0.0_f64, f64::max);
    let mean_aspect_ratio = if elements.is_empty() {
        0.0
    } else {
        elements
            .iter()
            .map(|element| element.aspect_ratio)
            .sum::<f64>()
            / elements.len() as f64
    };
    AnalysisMeshQualityReport {
        min_scaled_jacobian: min_scaled_jacobian.min(1.0),
        min_exact_scaled_jacobian: min_scaled_jacobian.min(1.0),
        mean_aspect_ratio,
        max_aspect_ratio,
        inverted_element_count: elements
            .iter()
            .filter(|element| element.volume_m3 <= 0.0)
            .count(),
        mean_boundary_projection_error_m: 0.0,
        max_boundary_projection_error_m: 0.0,
        elements,
    }
}
