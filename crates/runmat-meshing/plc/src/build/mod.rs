use std::collections::{BTreeMap, BTreeSet};

mod errors;
pub use errors::PlcBuildError;
use runmat_meshing_core::contracts::{
    MeshingStage, StageEvidence, SurfaceCadCurveBoundaryEdgeProvenance,
    SurfaceCadCurveBoundaryProvenance, SurfaceMesh, TopologyEntityId,
};
pub use runmat_meshing_core::contracts::{
    PlcFacet, PlcNode, PlcProtectedEdge, PlcProtectedEdgeCadCurveBoundary, ProtectedBoundaryComplex,
};

use crate::validate::{
    classify_boundary_components, classify_shell_nesting, validate_protected_boundary_complex,
};

pub const MODULE_PURPOSE: &str = "oriented protected boundary complex construction";

#[cfg(test)]
mod tests;

pub fn build_protected_boundary_complex(
    surface: &SurfaceMesh,
) -> Result<ProtectedBoundaryComplex, PlcBuildError> {
    if surface.triangles.is_empty() {
        return Err(PlcBuildError::EmptySurface);
    }
    let surface_source_face_count = surface
        .triangles
        .iter()
        .map(|element| element.source_face_id.clone())
        .collect::<BTreeSet<_>>()
        .len();
    let protected_source_edge_count = surface
        .triangles
        .iter()
        .flat_map(|element| element.source_edge_ids.iter().filter_map(|id| id.as_ref()))
        .collect::<BTreeSet<_>>()
        .len();
    let has_protected_source_edges = protected_source_edge_count > 0;
    if has_protected_source_edges && surface.curve_boundary_validation.is_none() {
        return Err(PlcBuildError::MissingCurveBoundaryValidation);
    }
    if has_protected_source_edges && surface.loop_coverage.is_none() {
        return Err(PlcBuildError::MissingSurfaceLoopCoverage);
    }
    let cad_curve_boundary_by_source_edge = surface
        .cad_curve_boundary_provenance
        .as_ref()
        .map(|report| {
            report
                .edges
                .iter()
                .map(|edge| (edge.source_edge_id.clone(), edge))
                .collect::<BTreeMap<_, _>>()
        })
        .unwrap_or_default();

    let surface_nodes = surface
        .nodes
        .iter()
        .map(|node| Ok((numeric_entity_id(&node.node_id)?, node.coordinates_m)))
        .collect::<Result<BTreeMap<_, _>, PlcBuildError>>()?;
    for node in &surface.nodes {
        if !node
            .coordinates_m
            .iter()
            .all(|coordinate| coordinate.is_finite())
        {
            return Err(PlcBuildError::NonFiniteSurfaceNode {
                node_id: numeric_entity_id(&node.node_id)?,
            });
        }
    }

    let mut facets = Vec::<PlcFacet>::with_capacity(surface.triangles.len());
    let mut protected_edges = BTreeMap::<(TopologyEntityId, u32, u32), PlcProtectedEdge>::new();
    let mut source_edge_marker_by_segment = BTreeMap::<[u32; 2], Option<u32>>::new();
    let mut edge_incidence = BTreeMap::<[u32; 2], usize>::new();
    let mut facet_keys = BTreeSet::<[u32; 3]>::new();
    for element in &surface.triangles {
        let element_id = numeric_entity_id(&element.triangle_id)?;
        let element_node_ids = element
            .node_ids
            .iter()
            .map(numeric_entity_id)
            .collect::<Result<Vec<_>, _>>()?;
        let element_node_ids: [u32; 3] = element_node_ids
            .try_into()
            .expect("surface triangles always carry three node IDs");
        if !element.area_m2.is_finite() || !element.max_projection_error_m.is_finite() {
            return Err(PlcBuildError::NonFiniteSurfaceTriangle {
                triangle_id: element_id,
            });
        }
        if element.area_m2 <= 0.0 {
            return Err(PlcBuildError::NonPositiveSurfaceTriangleArea {
                triangle_id: element_id,
            });
        }
        for node_id in element_node_ids {
            if !surface_nodes.contains_key(&node_id) {
                return Err(PlcBuildError::MissingSurfaceNode {
                    triangle_id: element_id,
                    node_id,
                });
            }
        }
        let mut facet_key = element_node_ids;
        facet_key.sort_unstable();
        if !facet_keys.insert(facet_key) {
            return Err(PlcBuildError::DuplicateFacet { element_id });
        }

        for edge_index in 0..3 {
            let left = element_node_ids[edge_index];
            let right = element_node_ids[(edge_index + 1) % 3];
            let edge = sorted_edge(left, right);
            *edge_incidence.entry(edge).or_insert(0) += 1;

            let source_edge_id = element.source_edge_ids[edge_index].as_ref();
            let source_edge_marker = source_edge_id.map(numeric_entity_id).transpose()?;
            if let Some(first_source_edge_marker) =
                source_edge_marker_by_segment.insert(edge, source_edge_marker)
            {
                match (first_source_edge_marker, source_edge_marker) {
                    (Some(first_source_edge_id), Some(second_source_edge_id))
                        if first_source_edge_id != second_source_edge_id =>
                    {
                        return Err(PlcBuildError::AmbiguousProtectedBoundarySegment {
                            node_ids: edge,
                            first_source_edge_id,
                            second_source_edge_id,
                        });
                    }
                    (Some(source_edge_id), None) | (None, Some(source_edge_id)) => {
                        return Err(PlcBuildError::PartiallyProtectedBoundarySegment {
                            node_ids: edge,
                            source_edge_id,
                        });
                    }
                    _ => {}
                }
            }
            if let Some(source_edge_id) = source_edge_id {
                let source_edge_marker =
                    source_edge_marker.expect("source-edge marker exists when ID exists");
                protected_edges
                    .entry((source_edge_id.clone(), edge[0], edge[1]))
                    .or_insert_with(|| PlcProtectedEdge {
                        edge_id: topology_entity_id(
                            MeshingStage::ProtectedBoundaryComplex,
                            format!(
                                "plc_protected_edge_{source_edge_marker}_{}_{}",
                                edge[0], edge[1]
                            ),
                        ),
                        node_ids: [
                            topology_entity_id(MeshingStage::ProtectedBoundaryComplex, edge[0]),
                            topology_entity_id(MeshingStage::ProtectedBoundaryComplex, edge[1]),
                        ],
                        source_edge_id: source_edge_id.clone(),
                        cad_curve_boundary: cad_curve_boundary_by_source_edge
                            .get(source_edge_id)
                            .map(|provenance| plc_cad_curve_boundary(provenance)),
                    });
            }
        }

        facets.push(PlcFacet {
            facet_id: topology_entity_id(
                MeshingStage::ProtectedBoundaryComplex,
                element.triangle_id.id.clone(),
            ),
            node_ids: [
                topology_entity_id(
                    MeshingStage::ProtectedBoundaryComplex,
                    element.node_ids[0].id.clone(),
                ),
                topology_entity_id(
                    MeshingStage::ProtectedBoundaryComplex,
                    element.node_ids[1].id.clone(),
                ),
                topology_entity_id(
                    MeshingStage::ProtectedBoundaryComplex,
                    element.node_ids[2].id.clone(),
                ),
            ],
            source_face_id: element.source_face_id.clone(),
            material_interface_ids: element.material_region_ids.clone(),
        });
    }

    for (edge, incidence_count) in edge_incidence {
        if incidence_count < 2 {
            return Err(PlcBuildError::OpenBoundaryEdge {
                node_ids: edge,
                incidence_count,
            });
        }
        if incidence_count > 2 {
            return Err(PlcBuildError::NonManifoldBoundaryEdge {
                node_ids: edge,
                incidence_count,
            });
        }
    }
    if let Some(loop_coverage) = &surface.loop_coverage {
        if loop_coverage.recovered_face_count != surface_source_face_count
            || loop_coverage.boundary_loop_count < surface_source_face_count
            || loop_coverage.hole_loop_count
                != loop_coverage
                    .boundary_loop_count
                    .saturating_sub(loop_coverage.recovered_face_count)
            || loop_coverage.max_loops_per_face == 0
            || loop_coverage.boundary_node_count == 0
            || loop_coverage.boundary_node_count > surface.nodes.len()
            || loop_coverage.recovered_source_edge_count < protected_source_edge_count
            || loop_coverage.boundary_segment_count < protected_source_edge_count
        {
            return Err(PlcBuildError::InconsistentSurfaceLoopCoverage {
                recovered_face_count: loop_coverage.recovered_face_count,
                surface_source_face_count,
                boundary_loop_count: loop_coverage.boundary_loop_count,
                hole_loop_count: loop_coverage.hole_loop_count,
                max_loops_per_face: loop_coverage.max_loops_per_face,
                boundary_node_count: loop_coverage.boundary_node_count,
                recovered_source_edge_count: loop_coverage.recovered_source_edge_count,
                protected_source_edge_count,
                boundary_segment_count: loop_coverage.boundary_segment_count,
            });
        }
    }
    if let Some(cad_curve_boundary_provenance) = &surface.cad_curve_boundary_provenance {
        validate_cad_curve_boundary_provenance(
            cad_curve_boundary_provenance,
            protected_source_edge_count,
        )?;
    }

    let mut evidence = StageEvidence::complete(MeshingStage::ProtectedBoundaryComplex);
    evidence
        .entity_counts
        .insert("nodes".to_string(), surface.nodes.len());
    evidence
        .entity_counts
        .insert("facets".to_string(), facets.len());
    evidence
        .entity_counts
        .insert("protected_edges".to_string(), protected_edges.len());
    if let Some(curve_boundary_validation) = &surface.curve_boundary_validation {
        evidence.entity_counts.insert(
            "validated_curve_source_edges".to_string(),
            curve_boundary_validation.source_edge_count,
        );
        evidence.entity_counts.insert(
            "validated_curve_nodes".to_string(),
            curve_boundary_validation.curve_node_count,
        );
        evidence.entity_counts.insert(
            "validated_curve_elements".to_string(),
            curve_boundary_validation.curve_element_count,
        );
    }
    if let Some(loop_coverage) = &surface.loop_coverage {
        evidence.entity_counts.insert(
            "surface_loop_faces".to_string(),
            loop_coverage.recovered_face_count,
        );
        evidence.entity_counts.insert(
            "surface_boundary_loops".to_string(),
            loop_coverage.boundary_loop_count,
        );
        evidence.entity_counts.insert(
            "surface_hole_loops".to_string(),
            loop_coverage.hole_loop_count,
        );
        evidence.entity_counts.insert(
            "surface_boundary_segments".to_string(),
            loop_coverage.boundary_segment_count,
        );
        evidence.entity_counts.insert(
            "surface_boundary_nodes".to_string(),
            loop_coverage.boundary_node_count,
        );
        evidence.entity_counts.insert(
            "recovered_surface_source_edges".to_string(),
            loop_coverage.recovered_source_edge_count,
        );
    }
    if let Some(cad_curve_boundary_provenance) = &surface.cad_curve_boundary_provenance {
        evidence.entity_counts.insert(
            "cad_curve_boundary_source_edges".to_string(),
            cad_curve_boundary_provenance.recovered_source_edge_count,
        );
        evidence.entity_counts.insert(
            "cad_curve_boundary_segments".to_string(),
            cad_curve_boundary_provenance.boundary_segment_count,
        );
        evidence.entity_counts.insert(
            "cad_curve_imported_edges".to_string(),
            cad_curve_boundary_provenance.imported_curve_edge_count,
        );
        evidence.entity_counts.insert(
            "cad_curve_evaluator_edges".to_string(),
            cad_curve_boundary_provenance.evaluator_curve_edge_count,
        );
        evidence.entity_counts.insert(
            "cad_curve_evaluator_samples".to_string(),
            cad_curve_boundary_provenance.evaluator_sample_count,
        );
        evidence.entity_counts.insert(
            "cad_curve_live_query_edges".to_string(),
            cad_curve_boundary_provenance.live_query_edge_count,
        );
        evidence.entity_counts.insert(
            "cad_curve_live_query_samples".to_string(),
            cad_curve_boundary_provenance.live_query_sample_count,
        );
        evidence.entity_counts.insert(
            "cad_curve_rejected_evaluator_samples".to_string(),
            cad_curve_boundary_provenance.rejected_evaluator_sample_count,
        );
        evidence.entity_counts.insert(
            "cad_curve_curvature_sized_edges".to_string(),
            cad_curve_boundary_provenance.curvature_sized_edge_count,
        );
        evidence.entity_counts.insert(
            "cad_curve_curvature_samples".to_string(),
            cad_curve_boundary_provenance.curvature_sample_count,
        );
    }

    let mut plc = ProtectedBoundaryComplex {
        complex_id: "plc_surface_boundary".to_string(),
        nodes: surface
            .nodes
            .iter()
            .map(|node| PlcNode {
                node_id: topology_entity_id(
                    MeshingStage::ProtectedBoundaryComplex,
                    node.node_id.id.clone(),
                ),
                coordinates_m: node.coordinates_m,
            })
            .collect(),
        facets,
        protected_edges: protected_edges.into_values().collect(),
        validation: runmat_meshing_core::contracts::PlcValidationSummary {
            watertight: true,
            manifold: true,
            shell_nesting_classified: true,
            material_interfaces_classified: true,
        },
        evidence,
    };
    let component_report = classify_boundary_components(&plc);
    plc.evidence.entity_counts.insert(
        "boundary_components".to_string(),
        component_report.component_count,
    );
    plc.evidence.entity_counts.insert(
        "boundary_component_nodes".to_string(),
        component_report.referenced_node_count,
    );
    plc.evidence.entity_counts.insert(
        "min_boundary_component_nodes".to_string(),
        component_report.min_component_node_count,
    );
    plc.evidence.entity_counts.insert(
        "max_boundary_component_nodes".to_string(),
        component_report.max_component_node_count,
    );
    let shell_classification = classify_shell_nesting(&plc, &component_report);
    plc.evidence.entity_counts.insert(
        "shell_nesting_classified".to_string(),
        usize::from(shell_classification.shell_nesting_classified),
    );
    plc.evidence.entity_counts.insert(
        "outer_shells".to_string(),
        shell_classification.outer_shell_count,
    );
    plc.evidence.entity_counts.insert(
        "nested_shells".to_string(),
        shell_classification.nested_shell_count,
    );
    plc.evidence.entity_counts.insert(
        "max_shell_nesting_depth".to_string(),
        shell_classification.max_nesting_depth,
    );
    plc.validation = validate_protected_boundary_complex(&plc)
        .map_err(PlcBuildError::ProtectedBoundaryValidation)?;
    Ok(plc)
}

fn sorted_edge(left: u32, right: u32) -> [u32; 2] {
    if left <= right {
        [left, right]
    } else {
        [right, left]
    }
}

fn topology_entity_id(stage: MeshingStage, id: impl ToString) -> TopologyEntityId {
    TopologyEntityId {
        stage,
        id: id.to_string(),
    }
}

fn numeric_entity_id(entity_id: &TopologyEntityId) -> Result<u32, PlcBuildError> {
    entity_id
        .id
        .parse()
        .map_err(|_| PlcBuildError::InvalidSurfaceEntityId {
            entity_id: entity_id.clone(),
        })
}

fn validate_cad_curve_boundary_provenance(
    report: &SurfaceCadCurveBoundaryProvenance,
    protected_source_edge_count: usize,
) -> Result<(), PlcBuildError> {
    let computed_boundary_segment_count = report
        .edges
        .iter()
        .map(|edge| edge.boundary_segment_count)
        .sum::<usize>();
    let computed_imported_edge_count = report
        .edges
        .iter()
        .filter(|edge| edge.imported_curve_id.is_some())
        .count();
    let computed_evaluator_edge_count = report
        .edges
        .iter()
        .filter(|edge| edge.evaluator_id.is_some())
        .count();
    let computed_evaluator_sample_count = report
        .edges
        .iter()
        .map(|edge| edge.evaluator_sample_count)
        .sum::<usize>();
    let computed_live_query_edge_count = report
        .edges
        .iter()
        .filter(|edge| edge.live_query_backed)
        .count();
    let computed_live_query_sample_count = report
        .edges
        .iter()
        .map(|edge| edge.live_query_sample_count)
        .sum::<usize>();
    let computed_rejected_evaluator_sample_count = report
        .edges
        .iter()
        .map(|edge| edge.rejected_evaluator_sample_count)
        .sum::<usize>();
    let computed_curvature_sized_edge_count = report
        .edges
        .iter()
        .filter(|edge| edge.curvature_limited_target_size_m.is_some())
        .count();
    let computed_curvature_sample_count = report
        .edges
        .iter()
        .map(|edge| edge.curvature_sample_count)
        .sum::<usize>();

    let reason = if report.recovered_source_edge_count != report.edges.len() {
        Some("source_edge_count_mismatch")
    } else if report.recovered_source_edge_count > protected_source_edge_count {
        Some("source_edge_count_exceeds_protected_edges")
    } else if report.boundary_segment_count != computed_boundary_segment_count {
        Some("boundary_segment_count_mismatch")
    } else if report.recovered_source_edge_count > 0
        && report.boundary_segment_count < report.recovered_source_edge_count
    {
        Some("boundary_segment_count_below_source_edges")
    } else if report.imported_curve_edge_count != computed_imported_edge_count {
        Some("imported_curve_edge_count_mismatch")
    } else if report.evaluator_curve_edge_count != computed_evaluator_edge_count {
        Some("evaluator_curve_edge_count_mismatch")
    } else if report.evaluator_sample_count != computed_evaluator_sample_count {
        Some("evaluator_sample_count_mismatch")
    } else if report.live_query_edge_count != computed_live_query_edge_count {
        Some("live_query_edge_count_mismatch")
    } else if report.live_query_edge_count > report.evaluator_curve_edge_count {
        Some("live_query_edge_count_exceeds_evaluator_edges")
    } else if report.live_query_sample_count != computed_live_query_sample_count {
        Some("live_query_sample_count_mismatch")
    } else if report.rejected_evaluator_sample_count != computed_rejected_evaluator_sample_count {
        Some("rejected_evaluator_sample_count_mismatch")
    } else if report.curvature_sized_edge_count != computed_curvature_sized_edge_count {
        Some("curvature_sized_edge_count_mismatch")
    } else if report.curvature_sample_count != computed_curvature_sample_count {
        Some("curvature_sample_count_mismatch")
    } else {
        None
    };

    if let Some(reason) = reason {
        return Err(PlcBuildError::InconsistentCadCurveBoundaryProvenance {
            reason,
            recovered_source_edge_count: report.recovered_source_edge_count,
            protected_source_edge_count,
            boundary_segment_count: report.boundary_segment_count,
            edge_report_count: report.edges.len(),
        });
    }

    Ok(())
}

fn plc_cad_curve_boundary(
    provenance: &SurfaceCadCurveBoundaryEdgeProvenance,
) -> PlcProtectedEdgeCadCurveBoundary {
    PlcProtectedEdgeCadCurveBoundary {
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
