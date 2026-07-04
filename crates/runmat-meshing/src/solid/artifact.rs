use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::GeometryAsset;
use runmat_meshing_core::{
    contracts::{
        artifact::ANALYSIS_MESH_SCHEMA_VERSION, AnalysisBoundaryEdge, AnalysisBoundaryFace,
        AnalysisMeshArtifact, AnalysisMeshNode, AnalysisMeshProvenance, AnalysisVolumeElement,
        BoundaryElementKind, MeshBackendSummary, MeshEntityProvenance, MeshingStage,
        SourceEntityKind, TopologyEntityId, VolumeElementKind,
    },
    quality::{
        predicate::{
            tetrahedron_edge_aspect_ratio, tetrahedron_scaled_jacobian, tetrahedron_volume,
        },
        AnalysisMeshQualityReport, ElementQuality,
    },
    size::field::MeshSizingField,
};
use runmat_meshing_surface::{SurfaceDiscretization, INTERNAL_SOURCE_EDGE_ID};
use runmat_meshing_tetrahedron::{generate::TetrahedronMesh, recover::TetrahedronRecoveryQueue};

const SOLID_PLC_TETRAHEDRON_ALGORITHM: &str = "plc_tetrahedron/v1";

pub(super) fn analysis_artifact_from_tetrahedron_mesh(
    geometry: &GeometryAsset,
    sizing: &MeshSizingField,
    surface: &SurfaceDiscretization,
    recovery_queue: &TetrahedronRecoveryQueue,
    tetrahedron_mesh: TetrahedronMesh,
) -> AnalysisMeshArtifact {
    let node_id_map = tetrahedron_mesh
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.node_id.clone(), index as u32 + 1))
        .collect::<BTreeMap<_, _>>();
    let mesh_provenance = MeshEntityProvenance {
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
            provenance: vec![mesh_provenance.clone()],
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
            provenance: vec![mesh_provenance.clone()],
        })
        .collect::<Vec<_>>();
    let source_edge_provenance_by_edge =
        source_edge_provenance_by_boundary_edge(geometry, surface, &node_id_map);
    let boundary_faces = tetrahedron_mesh
        .boundary_faces
        .iter()
        .map(|face| {
            let node_ids = face
                .node_ids
                .iter()
                .map(|node_id| node_id_map[node_id])
                .collect::<Vec<_>>();
            let region_ids = surface_region_ids(surface, &face.source_face_id.id);
            AnalysisBoundaryFace {
                face_id: face.face_id.id.clone(),
                kind: BoundaryElementKind::Tri3,
                adjacent_volume_element_ids: adjacent_volume_element_ids(
                    &node_ids,
                    &volume_elements,
                ),
                region_ids: region_ids.clone(),
                node_ids,
                provenance: vec![
                    mesh_provenance.clone(),
                    MeshEntityProvenance {
                        source_geometry_id: geometry.geometry_id.clone(),
                        source_geometry_revision: geometry.revision,
                        source_entity_kind: SourceEntityKind::Face,
                        source_entity_id: face.source_face_id.id.clone(),
                        region_ids,
                    },
                ],
            }
        })
        .collect::<Vec<_>>();
    let boundary_edges = boundary_edges_from_faces(
        &boundary_faces,
        &mesh_provenance,
        &source_edge_provenance_by_edge,
    );
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
            algorithm: SOLID_PLC_TETRAHEDRON_ALGORITHM.to_string(),
            surface_element_count: surface.elements.len(),
            plc_input_node_count: tetrahedron_entity_count(&tetrahedron_mesh, "input_plc_nodes"),
            plc_input_facet_count: tetrahedron_entity_count(&tetrahedron_mesh, "input_plc_facets"),
            plc_input_protected_edge_count: tetrahedron_entity_count(
                &tetrahedron_mesh,
                "input_plc_protected_edges",
            ),
            plc_input_boundary_component_count: tetrahedron_entity_count(
                &tetrahedron_mesh,
                "input_plc_boundary_components",
            ),
            plc_input_boundary_component_node_count: tetrahedron_entity_count(
                &tetrahedron_mesh,
                "input_plc_boundary_component_nodes",
            ),
            plc_input_max_boundary_component_node_count: tetrahedron_entity_count(
                &tetrahedron_mesh,
                "input_plc_max_boundary_component_nodes",
            ),
            plc_input_shell_nesting_classified: tetrahedron_entity_count(
                &tetrahedron_mesh,
                "input_plc_shell_nesting_classified",
            ) > 0,
            plc_input_outer_shell_count: tetrahedron_entity_count(
                &tetrahedron_mesh,
                "input_plc_outer_shells",
            ),
            plc_input_nested_shell_count: tetrahedron_entity_count(
                &tetrahedron_mesh,
                "input_plc_nested_shells",
            ),
            plc_input_max_shell_nesting_depth: tetrahedron_entity_count(
                &tetrahedron_mesh,
                "input_plc_max_shell_nesting_depth",
            ),
            tetrahedron_element_count: tetrahedron_mesh.elements.len(),
            boundary_face_recovery_ratio: 1.0,
            boundary_edge_recovery_ratio: 1.0,
            volume_component_count: 1,
            tetrahedron_recovered_component_ratio: 1.0,
            tetrahedron_volume_coverage_ratio: 1.0,
            tetrahedron_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "recovery_items",
            ),
            tetrahedron_recovered_item_count: recovery_entity_count(
                recovery_queue,
                "recovered_items",
            ),
            tetrahedron_missing_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "missing_items",
            ),
            tetrahedron_source_face_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "source_face_items",
            ),
            tetrahedron_missing_source_face_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "missing_source_face_items",
            ),
            tetrahedron_source_edge_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "source_edge_items",
            ),
            tetrahedron_missing_source_edge_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "missing_source_edge_items",
            ),
            tetrahedron_material_interface_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "material_interface_items",
            ),
            tetrahedron_missing_material_interface_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "missing_material_interface_items",
            ),
            ..MeshBackendSummary::default()
        },
        adaptive_iterations: Vec::new(),
        provenance: AnalysisMeshProvenance {
            algorithm: SOLID_PLC_TETRAHEDRON_ALGORITHM.to_string(),
            source_geometry_id: geometry.geometry_id.clone(),
            source_geometry_revision: geometry.revision,
            source_geometry_sha256: Some(geometry.source.sha256.clone()),
        },
    }
}

fn tetrahedron_entity_count(tetrahedron_mesh: &TetrahedronMesh, key: &str) -> usize {
    tetrahedron_mesh
        .evidence
        .entity_counts
        .get(key)
        .copied()
        .unwrap_or_default()
}

fn recovery_entity_count(recovery_queue: &TetrahedronRecoveryQueue, key: &str) -> usize {
    recovery_queue
        .evidence
        .entity_counts
        .get(key)
        .copied()
        .unwrap_or_default()
}

fn source_edge_provenance_by_boundary_edge(
    geometry: &GeometryAsset,
    surface: &SurfaceDiscretization,
    node_id_map: &BTreeMap<TopologyEntityId, u32>,
) -> BTreeMap<[u32; 2], MeshEntityProvenance> {
    let mut provenance_by_edge = BTreeMap::<[u32; 2], MeshEntityProvenance>::new();
    for element in &surface.elements {
        for (source_edge_id, edge) in element.source_edge_ids.into_iter().zip([
            sorted_edge(element.node_ids[0], element.node_ids[1]),
            sorted_edge(element.node_ids[1], element.node_ids[2]),
            sorted_edge(element.node_ids[2], element.node_ids[0]),
        ]) {
            if source_edge_id == INTERNAL_SOURCE_EDGE_ID {
                continue;
            }
            let Some(edge) = analysis_edge_from_surface_edge(edge, node_id_map) else {
                continue;
            };
            provenance_by_edge
                .entry(edge)
                .and_modify(|entry| {
                    append_unique_region_ids(&mut entry.region_ids, &element.region_ids)
                })
                .or_insert_with(|| MeshEntityProvenance {
                    source_geometry_id: geometry.geometry_id.clone(),
                    source_geometry_revision: geometry.revision,
                    source_entity_kind: SourceEntityKind::Edge,
                    source_entity_id: source_edge_id.to_string(),
                    region_ids: element.region_ids.clone(),
                });
        }
    }
    provenance_by_edge
}

fn analysis_edge_from_surface_edge(
    edge: [u32; 2],
    node_id_map: &BTreeMap<TopologyEntityId, u32>,
) -> Option<[u32; 2]> {
    let left = node_id_map.get(&surface_node_plc_id(edge[0]))?;
    let right = node_id_map.get(&surface_node_plc_id(edge[1]))?;
    Some(sorted_edge(*left, *right))
}

fn surface_node_plc_id(node_id: u32) -> TopologyEntityId {
    TopologyEntityId {
        stage: MeshingStage::ProtectedBoundaryComplex,
        id: node_id.to_string(),
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
    mesh_provenance: &MeshEntityProvenance,
    source_edge_provenance_by_edge: &BTreeMap<[u32; 2], MeshEntityProvenance>,
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
                .or_insert_with(|| {
                    let mut provenance = vec![mesh_provenance.clone()];
                    if let Some(source_edge_provenance) = source_edge_provenance_by_edge.get(&edge)
                    {
                        provenance.push(source_edge_provenance.clone());
                    }
                    AnalysisBoundaryEdge {
                        edge_id: format!("boundary_edge_{}_{}", edge[0], edge[1]),
                        node_ids: edge,
                        adjacent_boundary_face_ids: vec![face.face_id.clone()],
                        region_ids: face.region_ids.clone(),
                        provenance,
                    }
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
