use runmat_meshing_core::{
    MeshingStage, StageEvidence, SurfaceCadCurveBoundaryEdgeProvenance,
    SurfaceCadCurveBoundaryProvenance, SurfaceCurveBoundaryValidation, SurfaceLoopCoverage,
    SurfaceMesh, SurfaceMeshNode, SurfaceMeshTriangle, TopologyEntityId,
};

use crate::{
    SurfaceCadCurveBoundaryEdgeProvenance as SurfaceStageCadCurveBoundaryEdgeProvenance,
    SurfaceCadCurveBoundaryProvenanceReport, SurfaceDiscretization, SurfaceLoopCoverageReport,
    SurfaceValidationReport, INTERNAL_SOURCE_EDGE_ID,
};

pub fn build_surface_mesh_contract(
    mesh_id: impl Into<String>,
    surface: &SurfaceDiscretization,
    validation: &SurfaceValidationReport,
) -> SurfaceMesh {
    let node_provenance = surface_node_provenance(surface);
    let mut evidence = StageEvidence::complete(MeshingStage::SurfaceMesh);
    evidence
        .entity_counts
        .insert("source_faces".to_string(), validation.source_face_count);
    evidence
        .entity_counts
        .insert("nodes".to_string(), surface.nodes.len());
    evidence
        .entity_counts
        .insert("triangles".to_string(), validation.surface_element_count);
    evidence.entity_counts.insert(
        "source_edge_loops".to_string(),
        validation.source_edge_loop_count,
    );
    evidence.entity_counts.insert(
        "closed_source_edge_loops".to_string(),
        validation.closed_source_edge_loop_count,
    );
    evidence.entity_counts.insert(
        "conforming_source_edges".to_string(),
        validation.conforming_source_edge_count,
    );
    evidence.entity_counts.insert(
        "missing_source_edges".to_string(),
        validation.missing_source_edge_count,
    );
    evidence.entity_counts.insert(
        "material_regions".to_string(),
        surface
            .elements
            .iter()
            .flat_map(|element| element.material_region_ids.iter())
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
    );
    evidence.max_projection_error_m = Some(validation.max_projection_error_m);

    SurfaceMesh {
        mesh_id: mesh_id.into(),
        nodes: surface
            .nodes
            .iter()
            .map(|node| SurfaceMeshNode {
                node_id: surface_entity_id(node.node_id),
                coordinates_m: node.coordinates_m,
                source_edge_id: node_provenance
                    .get(&node.node_id)
                    .and_then(|provenance| provenance.source_edge_id.map(curve_entity_id)),
                source_face_id: surface_entity_id(
                    node_provenance
                        .get(&node.node_id)
                        .map(|provenance| provenance.source_face_id)
                        .unwrap_or_default(),
                ),
            })
            .collect(),
        triangles: surface
            .elements
            .iter()
            .map(|element| SurfaceMeshTriangle {
                triangle_id: surface_entity_id(element.element_id),
                source_face_id: surface_entity_id(element.source_face_id),
                source_edge_ids: element.source_edge_ids.map(surface_source_edge_id),
                node_ids: [
                    surface_entity_id(element.node_ids[0]),
                    surface_entity_id(element.node_ids[1]),
                    surface_entity_id(element.node_ids[2]),
                ],
                region_ids: element.region_ids.clone(),
                material_region_ids: element.material_region_ids.clone(),
                max_projection_error_m: element.max_projection_error_m,
                area_m2: element.area_m2,
            })
            .collect(),
        curve_boundary_validation: surface
            .curve_boundary_validation
            .as_ref()
            .map(surface_curve_boundary_validation),
        loop_coverage: surface.loop_coverage.as_ref().map(surface_loop_coverage),
        cad_curve_boundary_provenance: surface
            .cad_curve_boundary_provenance
            .as_ref()
            .map(surface_cad_curve_boundary_provenance),
        evidence,
    }
}

#[derive(Debug, Clone, Copy)]
struct SurfaceNodeProvenance {
    source_face_id: u32,
    source_edge_id: Option<u32>,
}

fn surface_node_provenance(
    surface: &SurfaceDiscretization,
) -> std::collections::BTreeMap<u32, SurfaceNodeProvenance> {
    let mut provenance = std::collections::BTreeMap::<u32, SurfaceNodeProvenance>::new();
    for element in &surface.elements {
        for node_id in element.node_ids {
            provenance.entry(node_id).or_insert(SurfaceNodeProvenance {
                source_face_id: element.source_face_id,
                source_edge_id: None,
            });
        }
        for (edge_index, source_edge_id) in element.source_edge_ids.into_iter().enumerate() {
            if source_edge_id == INTERNAL_SOURCE_EDGE_ID {
                continue;
            }
            let left = element.node_ids[edge_index];
            let right = element.node_ids[(edge_index + 1) % element.node_ids.len()];
            for node_id in [left, right] {
                let entry = provenance.entry(node_id).or_insert(SurfaceNodeProvenance {
                    source_face_id: element.source_face_id,
                    source_edge_id: None,
                });
                if entry.source_edge_id.is_none() {
                    entry.source_edge_id = Some(source_edge_id);
                }
            }
        }
    }
    provenance
}

fn surface_entity_id(id: impl ToString) -> TopologyEntityId {
    TopologyEntityId {
        stage: MeshingStage::SurfaceMesh,
        id: id.to_string(),
    }
}

fn curve_entity_id(id: impl ToString) -> TopologyEntityId {
    TopologyEntityId {
        stage: MeshingStage::CurveMesh,
        id: id.to_string(),
    }
}

fn surface_source_edge_id(id: u32) -> Option<TopologyEntityId> {
    (id != INTERNAL_SOURCE_EDGE_ID).then(|| curve_entity_id(id))
}

fn surface_curve_boundary_validation(
    report: &runmat_meshing_curve::CurveValidationReport,
) -> SurfaceCurveBoundaryValidation {
    SurfaceCurveBoundaryValidation {
        source_edge_count: report.source_edge_count,
        curve_node_count: report.curve_node_count,
        curve_element_count: report.curve_element_count,
        max_endpoint_error_m: report.max_endpoint_error_m,
        max_projection_error_m: report.max_projection_error_m,
        max_length_error_m: report.max_length_error_m,
        max_segment_length_m: report.max_segment_length_m,
        max_parameter_gap: report.max_parameter_gap,
        max_adjacent_length_ratio: report.max_adjacent_length_ratio,
    }
}

fn surface_loop_coverage(report: &SurfaceLoopCoverageReport) -> SurfaceLoopCoverage {
    SurfaceLoopCoverage {
        source_face_count: report.source_face_count,
        recovered_face_count: report.recovered_face_count,
        boundary_loop_count: report.boundary_loop_count,
        boundary_node_count: report.boundary_node_count,
        recovered_source_edge_count: report.recovered_source_edge_count,
        boundary_segment_count: report.boundary_segment_count,
        max_loops_per_face: report.max_loops_per_face,
    }
}

fn surface_cad_curve_boundary_provenance(
    report: &SurfaceCadCurveBoundaryProvenanceReport,
) -> SurfaceCadCurveBoundaryProvenance {
    SurfaceCadCurveBoundaryProvenance {
        recovered_source_edge_count: report.recovered_source_edge_count,
        boundary_segment_count: report.boundary_segment_count,
        imported_curve_edge_count: report.imported_curve_edge_count,
        evaluator_curve_edge_count: report.evaluator_curve_edge_count,
        evaluator_sample_count: report.evaluator_sample_count,
        live_query_edge_count: report.live_query_edge_count,
        live_query_sample_count: report.live_query_sample_count,
        rejected_evaluator_sample_count: report.rejected_evaluator_sample_count,
        curvature_sized_edge_count: report.curvature_sized_edge_count,
        curvature_sample_count: report.curvature_sample_count,
        edges: report
            .edges
            .iter()
            .map(surface_cad_curve_boundary_edge_provenance)
            .collect(),
    }
}

fn surface_cad_curve_boundary_edge_provenance(
    provenance: &SurfaceStageCadCurveBoundaryEdgeProvenance,
) -> SurfaceCadCurveBoundaryEdgeProvenance {
    SurfaceCadCurveBoundaryEdgeProvenance {
        source_edge_id: curve_entity_id(provenance.source_edge_id),
        cad_edge_id: provenance.cad_edge_id.clone(),
        imported_curve_id: provenance.imported_curve_id,
        evaluator_id: provenance.evaluator_id.clone(),
        evaluator_supports_point_evaluation: provenance.evaluator_supports_point_evaluation,
        evaluator_supports_projection: provenance.evaluator_supports_projection,
        evaluator_supports_tangent: provenance.evaluator_supports_tangent,
        evaluator_supports_curvature: provenance.evaluator_supports_curvature,
        evaluator_sample_count: provenance.evaluator_sample_count,
        live_query_backed: provenance.live_query_backed,
        live_query_sample_count: provenance.live_query_sample_count,
        rejected_evaluator_sample_count: provenance.rejected_evaluator_sample_count,
        curvature_sample_count: provenance.curvature_sample_count,
        curvature_limited_target_size_m: provenance.curvature_limited_target_size_m,
        boundary_segment_count: provenance.boundary_segment_count,
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        SurfaceDiscretization, SurfaceElement, SurfaceNode, SurfaceValidationReport,
        INTERNAL_SOURCE_EDGE_ID,
    };
    use runmat_meshing_core::{MeshingStage, StageEvidenceStatus};

    use super::build_surface_mesh_contract;

    #[test]
    fn surface_contract_preserves_source_and_material_regions() {
        let surface = SurfaceDiscretization {
            nodes: vec![
                node(0, [0.0, 0.0, 0.0]),
                node(1, [1.0, 0.0, 0.0]),
                node(2, [0.0, 1.0, 0.0]),
            ],
            elements: vec![SurfaceElement {
                element_id: 7,
                source_face_id: 3,
                cad_face_id: Some("cad_face_3".to_string()),
                source_edge_ids: [11, INTERNAL_SOURCE_EDGE_ID, 12],
                node_ids: [0, 1, 2],
                parametric_node_uv: [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                max_projection_error_m: 2.0e-10,
                region_ids: vec!["fixed_face".to_string(), "load_face".to_string()],
                material_region_ids: vec!["body".to_string()],
                area_m2: 0.5,
                unit_normal: [0.0, 0.0, 1.0],
            }],
            curve_boundary_validation: None,
            loop_coverage: None,
            cad_curve_boundary_provenance: None,
            exact_cad_sample_node_count: 0,
            rejected_exact_cad_sample_count: 0,
        };
        let validation = SurfaceValidationReport {
            source_face_count: 1,
            surface_element_count: 1,
            source_edge_loop_count: 1,
            closed_source_edge_loop_count: 1,
            conforming_source_edge_count: 2,
            missing_source_edge_count: 0,
            max_projection_error_m: 2.0e-10,
            min_orientation_alignment: 1.0,
            face_coverage_ratio: 1.0,
        };

        let contract = build_surface_mesh_contract("surface", &surface, &validation);

        assert_eq!(contract.mesh_id, "surface");
        assert_eq!(contract.evidence.stage, MeshingStage::SurfaceMesh);
        assert_eq!(contract.evidence.status, StageEvidenceStatus::Complete);
        assert_eq!(contract.evidence.entity_counts["source_faces"], 1);
        assert_eq!(contract.evidence.entity_counts["nodes"], 3);
        assert_eq!(contract.evidence.entity_counts["triangles"], 1);
        assert_eq!(contract.evidence.entity_counts["material_regions"], 1);
        assert_eq!(contract.evidence.max_projection_error_m, Some(2.0e-10));
        assert!(contract
            .nodes
            .iter()
            .all(|node| node.node_id.stage == MeshingStage::SurfaceMesh));
        let triangle = &contract.triangles[0];
        assert_eq!(triangle.triangle_id.stage, MeshingStage::SurfaceMesh);
        assert_eq!(triangle.source_face_id.stage, MeshingStage::SurfaceMesh);
        assert_eq!(
            triangle
                .source_edge_ids
                .iter()
                .map(|source_edge_id| {
                    source_edge_id
                        .as_ref()
                        .map(|source_edge_id| (source_edge_id.stage, source_edge_id.id.as_str()))
                })
                .collect::<Vec<_>>(),
            vec![
                Some((MeshingStage::CurveMesh, "11")),
                None,
                Some((MeshingStage::CurveMesh, "12"))
            ]
        );
        assert_eq!(
            triangle.region_ids,
            vec!["fixed_face".to_string(), "load_face".to_string()]
        );
        assert_eq!(triangle.material_region_ids, vec!["body".to_string()]);
    }

    fn node(node_id: u32, coordinates_m: [f64; 3]) -> SurfaceNode {
        SurfaceNode {
            node_id,
            source_vertex_id: node_id,
            coordinates_m,
        }
    }
}
