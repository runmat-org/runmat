use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::GeometryAsset;
use runmat_meshing_core::{
    contracts::{
        artifact::ANALYSIS_MESH_SCHEMA_VERSION, AnalysisBoundaryEdge, AnalysisBoundaryFace,
        AnalysisMeshArtifact, AnalysisMeshNode, AnalysisMeshProvenance, AnalysisVolumeElement,
        BoundaryElementKind, MeshBackendSummary, MeshEntityProvenance, MeshingStage,
        SourceEntityKind, TopologyEntityId, VolumeElementKind,
        TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_ACCEPTED_COUNT,
        TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_ATTEMPT_COUNT,
        TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_BUDGET_LIMIT_COUNT,
        TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_REJECTED_COUNT,
        TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_REJECTION_PREFIX,
        TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_ACCEPTED_COUNT,
        TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_ATTEMPT_COUNT,
        TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_BUDGET_LIMIT_COUNT,
        TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_REJECTED_COUNT,
        TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_REJECTION_PREFIX,
        TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_ACCEPTED_COUNT,
        TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_ATTEMPT_COUNT,
        TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_BUDGET_LIMIT_COUNT,
        TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_REJECTED_COUNT,
        TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_REJECTION_PREFIX,
        UNCLASSIFIED_MATERIAL_REGION_ID,
    },
    quality::{
        predicate::{
            tetrahedron_edge_aspect_ratio, tetrahedron_scaled_jacobian, tetrahedron_volume,
        },
        AnalysisMeshQualityReport, ElementQuality,
    },
    size::field::MeshSizingField,
};
use runmat_meshing_opt::sliver::{
    classify_sliver_tetrahedra, SliverRecoveryOptions, SliverTetrahedronQuality,
};
use runmat_meshing_surface::{SurfaceDiscretization, INTERNAL_SOURCE_EDGE_ID};
use runmat_meshing_tetrahedron::{
    generate::TetrahedronMesh,
    recover::{TetrahedronRecoveryKind, TetrahedronRecoveryQueue, TetrahedronRecoveryStatus},
};

const SOLID_PLC_TETRAHEDRON_ALGORITHM: &str = "plc_tetrahedron/v1";
const MAX_REPORTED_RECOVERY_IDS: usize = 64;

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
    let mut source_edge_provenance_by_edge = tetrahedron_source_edge_provenance_by_boundary_edge(
        geometry,
        surface,
        &node_id_map,
        &tetrahedron_mesh,
    );
    merge_surface_source_edge_provenance(
        &mut source_edge_provenance_by_edge,
        source_edge_provenance_by_boundary_edge(geometry, surface, &node_id_map),
    );
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
    let backend_quality = backend_quality_evidence(&quality);
    let missing_source_face_recovery =
        bounded_missing_recovery_ids(recovery_queue, TetrahedronRecoveryKind::SourceFace);
    let missing_source_edge_recovery =
        bounded_missing_recovery_ids(recovery_queue, TetrahedronRecoveryKind::SourceEdge);
    let missing_material_interface_recovery =
        bounded_missing_recovery_ids(recovery_queue, TetrahedronRecoveryKind::MaterialInterface);

    let mut artifact = AnalysisMeshArtifact {
        schema_version: ANALYSIS_MESH_SCHEMA_VERSION.to_string(),
        mesh_id: format!("analysis_mesh_{}", geometry.geometry_id),
        nodes,
        volume_elements,
        boundary_faces,
        boundary_edges,
        quality,
        sizing: sizing.clone(),
        field_topology: Vec::new(),
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
            plc_input_material_region_count: tetrahedron_entity_count(
                &tetrahedron_mesh,
                "input_plc_material_regions",
            ),
            plc_input_material_region_facet_count: tetrahedron_entity_count(
                &tetrahedron_mesh,
                "input_plc_material_region_facets",
            ),
            plc_input_cad_curve_boundary_source_edge_count: tetrahedron_entity_count(
                &tetrahedron_mesh,
                "input_plc_cad_curve_boundary_source_edges",
            ),
            plc_input_cad_curve_boundary_segment_count: tetrahedron_entity_count(
                &tetrahedron_mesh,
                "input_plc_cad_curve_boundary_segments",
            ),
            plc_input_cad_curve_imported_edge_count: tetrahedron_entity_count(
                &tetrahedron_mesh,
                "input_plc_cad_curve_imported_edges",
            ),
            plc_input_cad_curve_evaluator_edge_count: tetrahedron_entity_count(
                &tetrahedron_mesh,
                "input_plc_cad_curve_evaluator_edges",
            ),
            plc_input_cad_curve_evaluator_sample_count: tetrahedron_entity_count(
                &tetrahedron_mesh,
                "input_plc_cad_curve_evaluator_samples",
            ),
            plc_input_cad_curve_live_query_edge_count: tetrahedron_entity_count(
                &tetrahedron_mesh,
                "input_plc_cad_curve_live_query_edges",
            ),
            plc_input_cad_curve_live_query_sample_count: tetrahedron_entity_count(
                &tetrahedron_mesh,
                "input_plc_cad_curve_live_query_samples",
            ),
            plc_input_cad_curve_rejected_evaluator_sample_count: tetrahedron_entity_count(
                &tetrahedron_mesh,
                "input_plc_cad_curve_rejected_evaluator_samples",
            ),
            plc_input_cad_curve_curvature_sized_edge_count: tetrahedron_entity_count(
                &tetrahedron_mesh,
                "input_plc_cad_curve_curvature_sized_edges",
            ),
            plc_input_cad_curve_curvature_sample_count: tetrahedron_entity_count(
                &tetrahedron_mesh,
                "input_plc_cad_curve_curvature_samples",
            ),
            tetrahedron_element_count: tetrahedron_mesh.elements.len(),
            tetrahedron_material_region_count: tetrahedron_material_region_count(
                &tetrahedron_mesh,
            ),
            tetrahedron_unclassified_material_element_count:
                tetrahedron_unclassified_material_element_count(&tetrahedron_mesh),
            tetrahedron_min_exact_scaled_jacobian: backend_quality.min_exact_scaled_jacobian,
            tetrahedron_exact_scaled_jacobian_below_threshold_count: backend_quality
                .exact_scaled_jacobian_below_threshold_count,
            tetrahedron_exact_scaled_jacobian_bins: backend_quality.exact_scaled_jacobian_bins,
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
            tetrahedron_recovered_boundary_face_count: recovery_entity_count(
                recovery_queue,
                "recovered_missing_boundary_faces",
            ),
            tetrahedron_recovered_protected_edge_boundary_face_count: recovery_entity_count(
                recovery_queue,
                "recovered_protected_edge_boundary_faces",
            ),
            tetrahedron_recovered_cad_curve_protected_edge_boundary_face_count:
                recovery_entity_count(
                    recovery_queue,
                    "recovered_cad_curve_protected_edge_boundary_faces",
                ),
            tetrahedron_attempted_protected_edge_boundary_face_restoration_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "attempted_protected_edge_boundary_face_restoration_items",
                ),
            tetrahedron_attempted_cad_curve_protected_edge_boundary_face_restoration_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "attempted_cad_curve_protected_edge_boundary_face_restoration_items",
                ),
            tetrahedron_rejected_protected_edge_boundary_face_restoration_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "rejected_protected_edge_boundary_face_restoration_items",
                ),
            tetrahedron_rejected_cad_curve_protected_edge_boundary_face_restoration_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "rejected_cad_curve_protected_edge_boundary_face_restoration_items",
                ),
            tetrahedron_rejected_protected_edge_boundary_face_restoration_volume_face_topology_count:
                recovery_entity_count(
                    recovery_queue,
                    "protected_edge_rejected_boundary_face_restoration_volume_face_topology",
                ),
            tetrahedron_volume_edge_source_edge_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "volume_edge_source_edge_recovery_items",
            ),
            tetrahedron_recovered_volume_edge_source_edge_recovery_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "recovered_volume_edge_source_edge_items",
                ),
            tetrahedron_boundary_edge_source_edge_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "boundary_edge_source_edge_recovery_items",
            ),
            tetrahedron_recovered_boundary_edge_source_edge_recovery_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "recovered_boundary_edge_source_edge_items",
                ),
            tetrahedron_interior_edge_source_edge_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "interior_edge_source_edge_recovery_items",
            ),
            tetrahedron_recovered_interior_edge_source_edge_recovery_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "recovered_interior_edge_source_edge_items",
                ),
            tetrahedron_cad_curve_interior_edge_source_edge_recovery_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "cad_curve_interior_edge_source_edge_recovery_items",
                ),
            tetrahedron_recovered_cad_curve_interior_edge_source_edge_recovery_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "recovered_cad_curve_interior_edge_source_edge_items",
                ),
            tetrahedron_attempted_source_edge_split_refill_item_count: recovery_entity_count(
                recovery_queue,
                "attempted_source_edge_split_refill_items",
            ),
            tetrahedron_attempted_cad_curve_source_edge_split_refill_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "attempted_cad_curve_source_edge_split_refill_items",
                ),
            tetrahedron_accepted_source_edge_split_refill_candidate_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "accepted_source_edge_split_refill_candidate_items",
                ),
            tetrahedron_accepted_cad_curve_source_edge_split_refill_candidate_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "accepted_cad_curve_source_edge_split_refill_candidate_items",
                ),
            tetrahedron_applied_source_edge_split_refill_item_count: recovery_entity_count(
                recovery_queue,
                "applied_source_edge_split_refill_items",
            ),
            tetrahedron_applied_cad_curve_source_edge_split_refill_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "applied_cad_curve_source_edge_split_refill_items",
                ),
            tetrahedron_rejected_source_edge_split_refill_item_count: recovery_entity_count(
                recovery_queue,
                "rejected_source_edge_split_refill_items",
            ),
            tetrahedron_rejected_cad_curve_source_edge_split_refill_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "rejected_cad_curve_source_edge_split_refill_items",
                ),
            tetrahedron_absent_edge_source_edge_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "absent_edge_source_edge_recovery_items",
            ),
            tetrahedron_recovered_absent_edge_source_edge_recovery_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "recovered_absent_edge_source_edge_items",
                ),
            tetrahedron_boundary_face_source_face_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "boundary_face_source_face_recovery_items",
            ),
            tetrahedron_recovered_boundary_face_source_face_recovery_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "recovered_boundary_face_source_face_items",
                ),
            tetrahedron_interior_face_source_face_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "interior_face_source_face_recovery_items",
            ),
            tetrahedron_recovered_interior_face_source_face_recovery_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "recovered_interior_face_source_face_items",
                ),
            tetrahedron_volume_face_source_face_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "volume_face_source_face_recovery_items",
            ),
            tetrahedron_recovered_volume_face_source_face_recovery_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "recovered_volume_face_source_face_items",
                ),
            tetrahedron_attempted_volume_face_source_face_boundary_restoration_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "attempted_volume_face_source_face_boundary_restoration_items",
                ),
            tetrahedron_rejected_volume_face_source_face_boundary_restoration_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "rejected_volume_face_source_face_boundary_restoration_items",
                ),
            tetrahedron_rejected_volume_face_source_face_boundary_restoration_volume_face_topology_count:
                recovery_entity_count(
                    recovery_queue,
                    "source_face_rejected_boundary_face_restoration_volume_face_topology",
                ),
            tetrahedron_absent_face_source_face_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "absent_face_source_face_recovery_items",
            ),
            tetrahedron_recovered_absent_face_source_face_recovery_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "recovered_absent_face_source_face_items",
                ),
            tetrahedron_deferred_absent_source_edge_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "deferred_absent_source_edge_recovery_items",
            ),
            tetrahedron_attempted_absent_source_edge_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "attempted_absent_source_edge_recovery_items",
            ),
            tetrahedron_attempted_cad_curve_absent_source_edge_recovery_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "attempted_cad_curve_absent_source_edge_recovery_items",
                ),
            tetrahedron_reconnected_absent_source_edge_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "reconnected_absent_source_edge_items",
            ),
            tetrahedron_reconnected_cad_curve_absent_source_edge_recovery_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "reconnected_cad_curve_absent_source_edge_items",
                ),
            tetrahedron_rejected_absent_source_edge_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "rejected_absent_source_edge_recovery_items",
            ),
            tetrahedron_rejected_cad_curve_absent_source_edge_recovery_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "rejected_cad_curve_absent_source_edge_recovery_items",
                ),
            tetrahedron_rejected_absent_source_edge_adjacent_facet_count: recovery_entity_count(
                recovery_queue,
                "rejected_absent_source_edge_recovery_adjacent_facet_count",
            ),
            tetrahedron_rejected_absent_source_edge_adjacent_facet_topology_count:
                recovery_entity_count(
                    recovery_queue,
                    "rejected_absent_source_edge_recovery_adjacent_facet_topology",
                ),
            tetrahedron_rejected_absent_source_edge_current_boundary_face_count:
                recovery_entity_count(
                    recovery_queue,
                    "rejected_absent_source_edge_recovery_current_boundary_faces",
                ),
            tetrahedron_rejected_absent_source_edge_element_topology_count: recovery_entity_count(
                recovery_queue,
                "rejected_absent_source_edge_recovery_element_topology",
            ),
            tetrahedron_rejected_absent_source_edge_material_region_mismatch_count:
                recovery_entity_count(
                    recovery_queue,
                    "rejected_absent_source_edge_recovery_material_region_mismatch",
                ),
            tetrahedron_rejected_absent_source_edge_quality_gate_count: recovery_entity_count(
                recovery_queue,
                "rejected_absent_source_edge_recovery_quality_gate",
            ),
            tetrahedron_recovered_absent_source_edge_boundary_face_count: recovery_entity_count(
                recovery_queue,
                "recovered_absent_source_edge_boundary_faces",
            ),
            tetrahedron_attempted_source_face_diagonal_recovery_pair_count: recovery_entity_count(
                recovery_queue,
                "attempted_source_face_diagonal_recovery_pairs",
            ),
            tetrahedron_recovered_source_face_diagonal_pair_count: recovery_entity_count(
                recovery_queue,
                "recovered_source_face_diagonal_pairs",
            ),
            tetrahedron_recovered_source_face_diagonal_boundary_face_count: recovery_entity_count(
                recovery_queue,
                "recovered_source_face_diagonal_boundary_faces",
            ),
            tetrahedron_rejected_source_face_diagonal_recovery_pair_count: recovery_entity_count(
                recovery_queue,
                "rejected_source_face_diagonal_recovery_pairs",
            ),
            tetrahedron_rejected_source_face_diagonal_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "rejected_source_face_diagonal_recovery_items",
            ),
            tetrahedron_rejected_source_face_diagonal_adjacent_facet_count: recovery_entity_count(
                recovery_queue,
                "rejected_source_face_diagonal_recovery_adjacent_facet_count",
            ),
            tetrahedron_rejected_source_face_diagonal_adjacent_facet_topology_count:
                recovery_entity_count(
                    recovery_queue,
                    "rejected_source_face_diagonal_recovery_adjacent_facet_topology",
                ),
            tetrahedron_rejected_source_face_diagonal_current_boundary_face_count:
                recovery_entity_count(
                    recovery_queue,
                    "rejected_source_face_diagonal_recovery_current_boundary_faces",
                ),
            tetrahedron_rejected_source_face_diagonal_element_topology_count: recovery_entity_count(
                recovery_queue,
                "rejected_source_face_diagonal_recovery_element_topology",
            ),
            tetrahedron_rejected_source_face_diagonal_material_region_mismatch_count:
                recovery_entity_count(
                    recovery_queue,
                    "rejected_source_face_diagonal_recovery_material_region_mismatch",
                ),
            tetrahedron_rejected_source_face_diagonal_quality_gate_count: recovery_entity_count(
                recovery_queue,
                "rejected_source_face_diagonal_recovery_quality_gate",
            ),
            tetrahedron_rejected_source_face_diagonal_unpaired_source_face_count:
                recovery_entity_count(
                    recovery_queue,
                    "rejected_source_face_diagonal_recovery_unpaired_source_face",
                ),
            tetrahedron_repaired_boundary_face_identity_count: recovery_entity_count(
                recovery_queue,
                "repaired_boundary_face_identity_items",
            ),
            tetrahedron_removed_redundant_boundary_face_count: recovery_entity_count(
                recovery_queue,
                "removed_redundant_boundary_faces",
            ),
            tetrahedron_removed_unsupported_boundary_face_count: recovery_entity_count(
                recovery_queue,
                "removed_unsupported_boundary_faces",
            ),
            tetrahedron_attempted_boundary_leak_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "attempted_boundary_leak_recovery_items",
            ),
            tetrahedron_removed_exterior_leaked_element_count: recovery_entity_count(
                recovery_queue,
                "removed_exterior_leaked_elements",
            ),
            tetrahedron_exposed_interior_source_face_count: recovery_entity_count(
                recovery_queue,
                "exposed_interior_source_faces",
            ),
            tetrahedron_inserted_exposed_interior_boundary_face_count: recovery_entity_count(
                recovery_queue,
                "inserted_exposed_interior_boundary_faces",
            ),
            tetrahedron_rejected_boundary_leak_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "rejected_boundary_leak_recovery_items",
            ),
            tetrahedron_rejected_boundary_leak_adjacent_element_count: recovery_entity_count(
                recovery_queue,
                "rejected_boundary_leak_adjacent_element_count",
            ),
            tetrahedron_rejected_boundary_leak_material_region_mismatch_count:
                recovery_entity_count(
                    recovery_queue,
                    "rejected_boundary_leak_material_region_mismatch",
                ),
            tetrahedron_rejected_boundary_leak_outside_classification_count: recovery_entity_count(
                recovery_queue,
                "rejected_boundary_leak_outside_classification",
            ),
            tetrahedron_rejected_boundary_leak_closed_surface_coordinate_count:
                recovery_entity_count(
                    recovery_queue,
                    "rejected_boundary_leak_closed_surface_coordinates",
                ),
            tetrahedron_repaired_source_face_provenance_count: recovery_entity_count(
                recovery_queue,
                "repaired_source_face_provenance_items",
            ),
            tetrahedron_repaired_source_edge_provenance_count: recovery_entity_count(
                recovery_queue,
                "repaired_source_edge_provenance_items",
            ),
            tetrahedron_repaired_cad_curve_source_edge_provenance_count: recovery_entity_count(
                recovery_queue,
                "repaired_cad_curve_source_edge_provenance_items",
            ),
            tetrahedron_repaired_material_interface_element_count: recovery_entity_count(
                recovery_queue,
                "repaired_material_interface_elements",
            ),
            tetrahedron_attempted_material_interface_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "attempted_material_interface_recovery_items",
            ),
            tetrahedron_rejected_material_interface_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "rejected_material_interface_recovery_items",
            ),
            tetrahedron_global_material_interface_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "global_material_interface_recovery_items",
            ),
            tetrahedron_boundary_owned_material_interface_recovery_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "boundary_owned_material_interface_recovery_items",
                ),
            tetrahedron_recovered_boundary_owned_material_interface_recovery_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "recovered_boundary_owned_material_interface_items",
                ),
            tetrahedron_interior_material_interface_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "interior_material_interface_recovery_items",
            ),
            tetrahedron_recovered_interior_face_material_interface_recovery_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "recovered_interior_face_material_interface_items",
                ),
            tetrahedron_recovered_absent_partition_material_interface_recovery_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "recovered_absent_partition_material_interface_items",
                ),
            tetrahedron_rejected_material_interface_missing_boundary_ownership_count:
                recovery_entity_count(
                    recovery_queue,
                    "rejected_material_interface_missing_boundary_ownership",
                ),
            tetrahedron_rejected_material_interface_ambiguous_boundary_ownership_count:
                recovery_entity_count(
                    recovery_queue,
                    "rejected_material_interface_ambiguous_boundary_ownership",
                ),
            tetrahedron_attempted_absent_material_partition_recovery_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "attempted_absent_material_partition_recovery_items",
                ),
            tetrahedron_inserted_absent_material_partition_recovery_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "inserted_absent_material_partition_recovery_items",
                ),
            tetrahedron_inserted_absent_material_partition_element_count: recovery_entity_count(
                recovery_queue,
                "inserted_absent_material_partition_elements",
            ),
            tetrahedron_inserted_absent_material_partition_boundary_face_count:
                recovery_entity_count(
                    recovery_queue,
                    "inserted_absent_material_partition_boundary_faces",
                ),
            tetrahedron_rejected_absent_material_partition_recovery_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "rejected_absent_material_partition_recovery_items",
                ),
            tetrahedron_rolled_back_absent_material_partition_recovery_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "rolled_back_absent_material_partition_recovery_items",
                ),
            tetrahedron_rolled_back_absent_material_partition_element_count: recovery_entity_count(
                recovery_queue,
                "rolled_back_absent_material_partition_elements",
            ),
            tetrahedron_rolled_back_absent_material_partition_boundary_face_count:
                recovery_entity_count(
                    recovery_queue,
                    "rolled_back_absent_material_partition_boundary_faces",
                ),
            tetrahedron_rejected_absent_material_partition_facet_count: recovery_entity_count(
                recovery_queue,
                "rejected_absent_material_partition_facet_count",
            ),
            tetrahedron_rejected_absent_material_partition_facet_topology_count:
                recovery_entity_count(
                    recovery_queue,
                    "rejected_absent_material_partition_facet_topology",
                ),
            tetrahedron_rejected_absent_material_partition_element_exists_count:
                recovery_entity_count(
                    recovery_queue,
                    "rejected_absent_material_partition_element_exists",
                ),
            tetrahedron_rejected_absent_material_partition_interior_face_topology_count:
                recovery_entity_count(
                    recovery_queue,
                    "rejected_absent_material_partition_interior_face_topology",
                ),
            tetrahedron_rejected_absent_material_partition_quality_gate_count:
                recovery_entity_count(
                    recovery_queue,
                    "rejected_absent_material_partition_quality_gate",
                ),
            tetrahedron_rejected_absent_material_partition_post_insertion_audit_count:
                recovery_entity_count(
                    recovery_queue,
                    "rejected_absent_material_partition_post_insertion_audit",
                ),
            tetrahedron_source_face_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "source_face_items",
            ),
            tetrahedron_recovered_source_face_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "recovered_source_face_items",
            ),
            tetrahedron_missing_source_face_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "missing_source_face_items",
            ),
            tetrahedron_missing_source_face_topology_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "missing_source_face_topology_items",
            ),
            tetrahedron_missing_source_face_provenance_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "missing_source_face_provenance_items",
            ),
            tetrahedron_missing_source_face_boundary_face_recovery_item_count:
                recovery_entity_count(recovery_queue, "missing_source_face_boundary_face_items"),
            tetrahedron_missing_source_face_volume_face_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "missing_source_face_volume_face_items",
            ),
            tetrahedron_missing_source_face_interior_face_recovery_item_count:
                recovery_entity_count(recovery_queue, "missing_source_face_interior_face_items"),
            tetrahedron_missing_source_face_absent_face_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "missing_source_face_absent_face_items",
            ),
            tetrahedron_missing_source_face_recovery_ids: missing_source_face_recovery.ids,
            tetrahedron_omitted_missing_source_face_recovery_id_count: missing_source_face_recovery
                .omitted_count,
            tetrahedron_source_edge_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "source_edge_items",
            ),
            tetrahedron_recovered_source_edge_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "recovered_source_edge_items",
            ),
            tetrahedron_missing_source_edge_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "missing_source_edge_items",
            ),
            tetrahedron_missing_source_edge_topology_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "missing_source_edge_topology_items",
            ),
            tetrahedron_missing_source_edge_provenance_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "missing_source_edge_provenance_items",
            ),
            tetrahedron_missing_source_edge_volume_edge_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "missing_source_edge_volume_edge_items",
            ),
            tetrahedron_missing_source_edge_interior_edge_recovery_item_count:
                recovery_entity_count(recovery_queue, "missing_source_edge_interior_edge_items"),
            tetrahedron_missing_source_edge_absent_edge_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "missing_source_edge_absent_edge_items",
            ),
            tetrahedron_missing_source_edge_recovery_ids: missing_source_edge_recovery.ids,
            tetrahedron_omitted_missing_source_edge_recovery_id_count: missing_source_edge_recovery
                .omitted_count,
            tetrahedron_cad_curve_source_edge_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "cad_curve_source_edge_items",
            ),
            tetrahedron_recovered_cad_curve_source_edge_recovery_item_count:
                recovery_entity_count(recovery_queue, "recovered_cad_curve_source_edge_items"),
            tetrahedron_missing_cad_curve_source_edge_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "missing_cad_curve_source_edge_items",
            ),
            tetrahedron_missing_cad_curve_source_edge_topology_recovery_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "missing_cad_curve_source_edge_topology_items",
                ),
            tetrahedron_missing_cad_curve_source_edge_provenance_recovery_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "missing_cad_curve_source_edge_provenance_items",
                ),
            tetrahedron_material_interface_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "material_interface_items",
            ),
            tetrahedron_recovered_material_interface_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "recovered_material_interface_items",
            ),
            tetrahedron_missing_material_interface_recovery_item_count: recovery_entity_count(
                recovery_queue,
                "missing_material_interface_items",
            ),
            tetrahedron_missing_material_interface_boundary_owned_recovery_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "missing_material_interface_boundary_owned_items",
                ),
            tetrahedron_missing_material_interface_interior_face_recovery_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "missing_material_interface_interior_face_items",
                ),
            tetrahedron_missing_material_interface_absent_partition_recovery_item_count:
                recovery_entity_count(
                    recovery_queue,
                    "missing_material_interface_absent_partition_items",
                ),
            tetrahedron_missing_material_interface_recovery_ids:
                missing_material_interface_recovery.ids,
            tetrahedron_omitted_missing_material_interface_recovery_id_count:
                missing_material_interface_recovery.omitted_count,
            tetrahedron_optimization_pass_count: usize::from(tetrahedron_mesh.quality_optimized),
            tetrahedron_optimization_budget_limited_count: tetrahedron_entity_count(
                &tetrahedron_mesh,
                TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_BUDGET_LIMIT_COUNT,
            ) + tetrahedron_entity_count(
                &tetrahedron_mesh,
                TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_BUDGET_LIMIT_COUNT,
            ) + tetrahedron_entity_count(
                &tetrahedron_mesh,
                TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_BUDGET_LIMIT_COUNT,
            ),
            tetrahedron_smoothed_point_count: tetrahedron_entity_count(
                &tetrahedron_mesh,
                TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_ACCEPTED_COUNT,
            ) + tetrahedron_entity_count(
                &tetrahedron_mesh,
                TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_ACCEPTED_COUNT,
            ),
            tetrahedron_sliver_count: backend_quality.sliver_count,
            tetrahedron_optimization_target_seed_count: backend_quality.quality_repair_target_count,
            tetrahedron_optimization_skipped_target_seed_count: backend_quality
                .quality_repair_target_count,
            tetrahedron_optimization_interior_smoothing_attempt_count: tetrahedron_entity_count(
                &tetrahedron_mesh,
                TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_ATTEMPT_COUNT,
            ),
            tetrahedron_optimization_interior_smoothing_accepted_count: tetrahedron_entity_count(
                &tetrahedron_mesh,
                TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_ACCEPTED_COUNT,
            ),
            tetrahedron_optimization_interior_smoothing_rejected_count: tetrahedron_entity_count(
                &tetrahedron_mesh,
                TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_REJECTED_COUNT,
            ),
            tetrahedron_optimization_interior_smoothing_budget_limited_count:
                tetrahedron_entity_count(
                    &tetrahedron_mesh,
                    TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_BUDGET_LIMIT_COUNT,
                ),
            tetrahedron_optimization_interior_smoothing_rejected_by_reason:
                tetrahedron_rejection_counts_by_prefix(
                    &tetrahedron_mesh,
                    TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_REJECTION_PREFIX,
                ),
            tetrahedron_optimization_boundary_smoothing_attempt_count: tetrahedron_entity_count(
                &tetrahedron_mesh,
                TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_ATTEMPT_COUNT,
            ),
            tetrahedron_optimization_boundary_smoothing_accepted_count: tetrahedron_entity_count(
                &tetrahedron_mesh,
                TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_ACCEPTED_COUNT,
            ),
            tetrahedron_optimization_boundary_smoothing_rejected_count: tetrahedron_entity_count(
                &tetrahedron_mesh,
                TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_REJECTED_COUNT,
            ),
            tetrahedron_optimization_boundary_smoothing_budget_limited_count:
                tetrahedron_entity_count(
                    &tetrahedron_mesh,
                    TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_BUDGET_LIMIT_COUNT,
                ),
            tetrahedron_optimization_boundary_smoothing_rejected_by_reason:
                tetrahedron_rejection_counts_by_prefix(
                    &tetrahedron_mesh,
                    TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_REJECTION_PREFIX,
                ),
            tetrahedron_optimization_local_reconnection_attempt_count: tetrahedron_entity_count(
                &tetrahedron_mesh,
                TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_ATTEMPT_COUNT,
            ),
            tetrahedron_optimization_local_reconnection_accepted_count: tetrahedron_entity_count(
                &tetrahedron_mesh,
                TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_ACCEPTED_COUNT,
            ),
            tetrahedron_optimization_local_reconnection_rejected_count: tetrahedron_entity_count(
                &tetrahedron_mesh,
                TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_REJECTED_COUNT,
            ),
            tetrahedron_optimization_local_reconnection_budget_limited_count:
                tetrahedron_entity_count(
                    &tetrahedron_mesh,
                    TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_BUDGET_LIMIT_COUNT,
                ),
            tetrahedron_optimization_local_reconnection_rejected_by_reason:
                tetrahedron_rejection_counts_by_prefix(
                    &tetrahedron_mesh,
                    TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_REJECTION_PREFIX,
                ),
            tetrahedron_optimization_initial_max_aspect_ratio: backend_quality.max_aspect_ratio,
            tetrahedron_optimization_final_max_aspect_ratio: backend_quality.max_aspect_ratio,
            tetrahedron_optimization_initial_min_exact_scaled_jacobian: backend_quality
                .min_exact_scaled_jacobian,
            tetrahedron_optimization_final_min_exact_scaled_jacobian: backend_quality
                .min_exact_scaled_jacobian,
            ..MeshBackendSummary::default()
        },
        adaptive_iterations: Vec::new(),
        provenance: AnalysisMeshProvenance {
            algorithm: SOLID_PLC_TETRAHEDRON_ALGORITHM.to_string(),
            source_geometry_id: geometry.geometry_id.clone(),
            source_geometry_revision: geometry.revision,
            source_geometry_sha256: Some(geometry.source.sha256.clone()),
        },
    };
    artifact.refresh_field_topology();
    artifact
}

struct BackendQualityEvidence {
    min_exact_scaled_jacobian: f64,
    exact_scaled_jacobian_below_threshold_count: usize,
    exact_scaled_jacobian_bins: BTreeMap<String, usize>,
    sliver_count: usize,
    quality_repair_target_count: usize,
    max_aspect_ratio: f64,
}

fn backend_quality_evidence(quality: &AnalysisMeshQualityReport) -> BackendQualityEvidence {
    let options = SliverRecoveryOptions::default();
    let min_exact_scaled_jacobian = if quality.elements.is_empty() {
        0.0
    } else {
        quality
            .elements
            .iter()
            .map(|element| element.exact_scaled_jacobian)
            .fold(f64::INFINITY, f64::min)
    };
    let exact_scaled_jacobian_below_threshold_count = quality
        .elements
        .iter()
        .filter(|element| element.exact_scaled_jacobian < options.min_exact_scaled_jacobian)
        .count();
    let mut exact_scaled_jacobian_bins = BTreeMap::<String, usize>::new();
    for element in &quality.elements {
        *exact_scaled_jacobian_bins
            .entry(scaled_jacobian_bin(element.exact_scaled_jacobian))
            .or_default() += 1;
    }
    let sliver_inputs = quality
        .elements
        .iter()
        .enumerate()
        .map(|(index, element)| SliverTetrahedronQuality {
            tetrahedron_id: index as u32 + 1,
            aspect_ratio: element.aspect_ratio,
            exact_scaled_jacobian: element.exact_scaled_jacobian,
        })
        .collect::<Vec<_>>();
    let sliver_count =
        classify_sliver_tetrahedra(&sliver_inputs, options).map_or(0, |slivers| slivers.len());

    BackendQualityEvidence {
        min_exact_scaled_jacobian,
        exact_scaled_jacobian_below_threshold_count,
        exact_scaled_jacobian_bins,
        sliver_count,
        quality_repair_target_count: sliver_count.max(exact_scaled_jacobian_below_threshold_count),
        max_aspect_ratio: quality.max_aspect_ratio,
    }
}

fn scaled_jacobian_bin(value: f64) -> String {
    if !value.is_finite() {
        return "non_finite".to_string();
    }
    match value {
        value if value < 0.0 => "negative".to_string(),
        value if value < 0.15 => "0_00_to_0_15".to_string(),
        value if value < 0.35 => "0_15_to_0_35".to_string(),
        value if value < 0.65 => "0_35_to_0_65".to_string(),
        _ => "0_65_to_1_00".to_string(),
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

fn tetrahedron_rejection_counts_by_prefix(
    tetrahedron_mesh: &TetrahedronMesh,
    prefix: &str,
) -> BTreeMap<String, usize> {
    tetrahedron_mesh
        .evidence
        .rejection_counts
        .iter()
        .filter_map(|(reason, count)| {
            reason
                .strip_prefix(prefix)
                .map(|stripped| (stripped.to_string(), *count))
        })
        .collect()
}

fn tetrahedron_material_region_count(tetrahedron_mesh: &TetrahedronMesh) -> usize {
    tetrahedron_mesh
        .elements
        .iter()
        .map(|element| element.material_region_id.as_str())
        .collect::<BTreeSet<_>>()
        .len()
}

fn tetrahedron_unclassified_material_element_count(tetrahedron_mesh: &TetrahedronMesh) -> usize {
    tetrahedron_mesh
        .elements
        .iter()
        .filter(|element| element.material_region_id == UNCLASSIFIED_MATERIAL_REGION_ID)
        .count()
}

fn recovery_entity_count(recovery_queue: &TetrahedronRecoveryQueue, key: &str) -> usize {
    recovery_queue
        .evidence
        .entity_counts
        .get(key)
        .copied()
        .unwrap_or_default()
}

struct BoundedRecoveryIds {
    ids: Vec<String>,
    omitted_count: usize,
}

fn bounded_missing_recovery_ids(
    recovery_queue: &TetrahedronRecoveryQueue,
    kind: TetrahedronRecoveryKind,
) -> BoundedRecoveryIds {
    let all_ids = recovery_queue
        .items
        .iter()
        .filter(|item| item.kind == kind && item.status == TetrahedronRecoveryStatus::Missing)
        .filter_map(|item| match kind {
            TetrahedronRecoveryKind::SourceFace | TetrahedronRecoveryKind::SourceEdge => item
                .source_entity_id
                .as_ref()
                .map(|source_id| source_id.id.clone()),
            TetrahedronRecoveryKind::MaterialInterface => item.material_interface_id.clone(),
        })
        .collect::<BTreeSet<_>>();
    let total_count = all_ids.len();
    let ids = all_ids
        .into_iter()
        .take(MAX_REPORTED_RECOVERY_IDS)
        .collect::<Vec<_>>();
    BoundedRecoveryIds {
        omitted_count: total_count.saturating_sub(ids.len()),
        ids,
    }
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

fn tetrahedron_source_edge_provenance_by_boundary_edge(
    geometry: &GeometryAsset,
    surface: &SurfaceDiscretization,
    node_id_map: &BTreeMap<TopologyEntityId, u32>,
    tetrahedron_mesh: &TetrahedronMesh,
) -> BTreeMap<[u32; 2], MeshEntityProvenance> {
    let mut provenance_by_edge = BTreeMap::<[u32; 2], MeshEntityProvenance>::new();
    for face in &tetrahedron_mesh.boundary_faces {
        for (source_edge_id, edge) in face.source_edge_ids.clone().into_iter().zip([
            sorted_topology_edge(face.node_ids[0].clone(), face.node_ids[1].clone()),
            sorted_topology_edge(face.node_ids[1].clone(), face.node_ids[2].clone()),
            sorted_topology_edge(face.node_ids[2].clone(), face.node_ids[0].clone()),
        ]) {
            let Some(source_edge_id) = source_edge_id else {
                continue;
            };
            let Some(edge) = analysis_edge_from_tetrahedron_edge(edge, node_id_map) else {
                continue;
            };
            let region_ids = surface_region_ids(surface, &face.source_face_id.id);
            provenance_by_edge
                .entry(edge)
                .and_modify(|entry| append_unique_region_ids(&mut entry.region_ids, &region_ids))
                .or_insert_with(|| MeshEntityProvenance {
                    source_geometry_id: geometry.geometry_id.clone(),
                    source_geometry_revision: geometry.revision,
                    source_entity_kind: SourceEntityKind::Edge,
                    source_entity_id: source_edge_id.id,
                    region_ids,
                });
        }
    }
    provenance_by_edge
}

fn merge_surface_source_edge_provenance(
    target: &mut BTreeMap<[u32; 2], MeshEntityProvenance>,
    source: BTreeMap<[u32; 2], MeshEntityProvenance>,
) {
    for (edge, provenance) in source {
        target
            .entry(edge)
            .and_modify(|entry| {
                append_unique_region_ids(&mut entry.region_ids, &provenance.region_ids)
            })
            .or_insert(provenance);
    }
}

fn analysis_edge_from_surface_edge(
    edge: [u32; 2],
    node_id_map: &BTreeMap<TopologyEntityId, u32>,
) -> Option<[u32; 2]> {
    let left = node_id_map.get(&surface_node_plc_id(edge[0]))?;
    let right = node_id_map.get(&surface_node_plc_id(edge[1]))?;
    Some(sorted_edge(*left, *right))
}

fn analysis_edge_from_tetrahedron_edge(
    edge: [TopologyEntityId; 2],
    node_id_map: &BTreeMap<TopologyEntityId, u32>,
) -> Option<[u32; 2]> {
    let left = node_id_map.get(&edge[0])?;
    let right = node_id_map.get(&edge[1])?;
    Some(sorted_edge(*left, *right))
}

fn sorted_topology_edge(left: TopologyEntityId, right: TopologyEntityId) -> [TopologyEntityId; 2] {
    let mut edge = [left, right];
    edge.sort();
    edge
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
