use std::collections::BTreeMap;

use runmat_meshing_cad::{
    project_to_face, CadEvaluationModel, CadFace, CadLoop, CadTopologyModel, SourceTopologyEdge,
    SourceTopologyFace, SourceTopologyModel,
};
use runmat_meshing_curve::{
    validate_curve_discretization, CadCurveDiscretization, CadCurveEdgeProvenance,
    CurveDiscretization, CurveValidationOptions,
};

mod boundary;
mod coverage;
mod elements;
mod geometry;
mod sampling;
mod subdivision;
mod triangulation;
mod types;

use boundary::{
    curve_nodes_by_source_edge, curve_segments_for_source_edges, face_curve_segment_loops,
    oriented_face_curve_segments,
};
use coverage::{SurfaceCadCurveBoundaryProvenanceAccumulator, SurfaceLoopCoverageAccumulator};
use elements::append_curve_driven_face_elements;
use subdivision::{append_centroid_subdivision, triangle_centroid};
pub use types::{
    SurfaceCadCurveBoundaryEdgeProvenance, SurfaceCadCurveBoundaryProvenanceReport,
    SurfaceDiscretization, SurfaceDiscretizationError, SurfaceDiscretizationOptions,
    SurfaceElement, SurfaceLoopCoverageReport, SurfaceNode,
};

pub const MODULE_PURPOSE: &str = "face-domain triangulation from recovered curve boundaries";
pub const INTERNAL_SOURCE_EDGE_ID: u32 = u32::MAX;

pub fn discretize_topology_surfaces(
    topology: &SourceTopologyModel,
    _options: SurfaceDiscretizationOptions,
) -> Result<SurfaceDiscretization, SurfaceDiscretizationError> {
    let nodes = topology
        .vertices
        .iter()
        .map(|vertex| SurfaceNode {
            node_id: vertex.vertex_id,
            source_vertex_id: vertex.vertex_id,
            coordinates_m: vertex.coordinates_m,
        })
        .collect::<Vec<_>>();

    let mut elements = Vec::<SurfaceElement>::with_capacity(topology.faces.len());
    for face in &topology.faces {
        validate_face_vertices(topology, face)?;
        elements.push(SurfaceElement {
            element_id: elements.len() as u32,
            source_face_id: face.face_id,
            cad_face_id: None,
            source_edge_ids: face.edge_ids,
            node_ids: face.node_ids,
            parametric_node_uv: [[0.0, 0.0]; 3],
            max_projection_error_m: 0.0,
            region_ids: face.region_ids.clone(),
            material_region_ids: face.material_region_ids.clone(),
            area_m2: face.area_m2,
            unit_normal: face.unit_normal,
        });
    }

    Ok(SurfaceDiscretization {
        nodes,
        elements,
        curve_boundary_validation: None,
        loop_coverage: None,
        cad_curve_boundary_provenance: None,
        exact_cad_sample_node_count: 0,
        rejected_exact_cad_sample_count: 0,
    })
}

pub fn discretize_cad_surfaces(
    topology: &SourceTopologyModel,
    cad_evaluation: &CadEvaluationModel,
    options: SurfaceDiscretizationOptions,
) -> Result<SurfaceDiscretization, SurfaceDiscretizationError> {
    let mut nodes = topology
        .vertices
        .iter()
        .map(|vertex| SurfaceNode {
            node_id: vertex.vertex_id,
            source_vertex_id: vertex.vertex_id,
            coordinates_m: vertex.coordinates_m,
        })
        .collect::<Vec<_>>();
    let frames_by_source_face = cad_evaluation
        .face_frames
        .iter()
        .map(|frame| (frame.source_face_id, frame))
        .collect::<BTreeMap<_, _>>();

    let element_capacity = if options.centroid_subdivision {
        topology.faces.len() * 3
    } else {
        topology.faces.len()
    };
    let mut elements = Vec::<SurfaceElement>::with_capacity(element_capacity);
    for face in &topology.faces {
        validate_face_vertices(topology, face)?;
        let frame = frames_by_source_face.get(&face.face_id).ok_or(
            SurfaceDiscretizationError::MissingCadFaceFrame {
                source_face_id: face.face_id,
            },
        )?;
        let mut parametric_node_uv = [[0.0_f64, 0.0_f64]; 3];
        let mut max_projection_error_m = 0.0_f64;
        let mut corner_points = [[0.0_f64, 0.0_f64, 0.0_f64]; 3];
        for (index, node_id) in face.node_ids.into_iter().enumerate() {
            let point = topology
                .vertices
                .get(node_id as usize)
                .filter(|vertex| vertex.vertex_id == node_id)
                .map(|vertex| vertex.coordinates_m)
                .ok_or(SurfaceDiscretizationError::MissingFaceVertex {
                    face_id: face.face_id,
                    node_id,
                })?;
            corner_points[index] = point;
            let projection = project_to_face(frame, point);
            if !projection.uv_in_bounds {
                return Err(SurfaceDiscretizationError::CadProjectionOutsideFaceDomain {
                    face_id: face.face_id,
                    node_id,
                });
            }
            parametric_node_uv[index] = projection.uv;
            max_projection_error_m = max_projection_error_m.max(projection.distance_m);
        }

        if options.centroid_subdivision {
            let centroid = triangle_centroid(corner_points);
            let centroid_projection = project_to_face(frame, centroid);
            if !centroid_projection.uv_in_bounds {
                return Err(SurfaceDiscretizationError::CadProjectionOutsideFaceDomain {
                    face_id: face.face_id,
                    node_id: u32::MAX,
                });
            }
            max_projection_error_m = max_projection_error_m.max(centroid_projection.distance_m);
            append_centroid_subdivision(
                face,
                frame,
                parametric_node_uv,
                centroid,
                centroid_projection.uv,
                max_projection_error_m,
                &mut nodes,
                &mut elements,
            );
        } else {
            elements.push(SurfaceElement {
                element_id: elements.len() as u32,
                source_face_id: face.face_id,
                cad_face_id: Some(frame.face_id.clone()),
                source_edge_ids: face.edge_ids,
                node_ids: face.node_ids,
                parametric_node_uv,
                max_projection_error_m,
                region_ids: face.region_ids.clone(),
                material_region_ids: face.material_region_ids.clone(),
                area_m2: face.area_m2,
                unit_normal: frame.unit_normal,
            });
        }
    }

    Ok(SurfaceDiscretization {
        nodes,
        elements,
        curve_boundary_validation: None,
        loop_coverage: None,
        cad_curve_boundary_provenance: None,
        exact_cad_sample_node_count: 0,
        rejected_exact_cad_sample_count: 0,
    })
}

pub fn discretize_cad_surfaces_with_curves(
    topology: &SourceTopologyModel,
    cad_evaluation: &CadEvaluationModel,
    curves: &CurveDiscretization,
    options: SurfaceDiscretizationOptions,
) -> Result<SurfaceDiscretization, SurfaceDiscretizationError> {
    let curve_boundary_validation =
        validate_curve_discretization(topology, curves, cad_curve_validation_options())
            .map_err(SurfaceDiscretizationError::InvalidCurveBoundary)?;
    let mut nodes = topology
        .vertices
        .iter()
        .map(|vertex| SurfaceNode {
            node_id: vertex.vertex_id,
            source_vertex_id: vertex.vertex_id,
            coordinates_m: vertex.coordinates_m,
        })
        .collect::<Vec<_>>();
    let frames_by_source_face = cad_evaluation
        .face_frames
        .iter()
        .map(|frame| (frame.source_face_id, frame))
        .collect::<BTreeMap<_, _>>();
    let topology_edges = topology
        .edges
        .iter()
        .map(|edge| (edge.edge_id, edge))
        .collect::<BTreeMap<_, _>>();
    let curve_nodes_by_edge = curve_nodes_by_source_edge(curves);
    let mut curve_node_to_surface_node = BTreeMap::<u32, u32>::new();

    let mut elements = Vec::<SurfaceElement>::new();
    let mut loop_coverage = SurfaceLoopCoverageAccumulator::new(topology.faces.len());
    let mut exact_cad_sample_node_count = 0_usize;
    let mut rejected_exact_cad_sample_count = 0_usize;
    for face in &topology.faces {
        validate_face_vertices(topology, face)?;
        let frame = frames_by_source_face.get(&face.face_id).ok_or(
            SurfaceDiscretizationError::MissingCadFaceFrame {
                source_face_id: face.face_id,
            },
        )?;
        let segments = oriented_face_curve_segments(
            &topology_edges,
            &curve_nodes_by_edge,
            face,
            options.max_curve_segments_per_edge.max(1),
            &mut nodes,
            &mut curve_node_to_surface_node,
        )?;
        let segment_loops = face_curve_segment_loops(face.face_id, &segments)?;
        loop_coverage.record_face(face, &segment_loops);
        let sample_report = append_curve_driven_face_elements(
            face,
            frame,
            &segment_loops,
            &mut nodes,
            &mut elements,
        );
        exact_cad_sample_node_count += sample_report.accepted_count;
        rejected_exact_cad_sample_count += sample_report.rejected_count;
    }

    Ok(SurfaceDiscretization {
        nodes,
        elements,
        curve_boundary_validation: Some(curve_boundary_validation),
        loop_coverage: Some(loop_coverage.finish()),
        cad_curve_boundary_provenance: None,
        exact_cad_sample_node_count,
        rejected_exact_cad_sample_count,
    })
}

pub fn discretize_cad_topology_surfaces_with_curves(
    cad_topology: &CadTopologyModel,
    topology: &SourceTopologyModel,
    cad_evaluation: &CadEvaluationModel,
    curves: &CurveDiscretization,
    options: SurfaceDiscretizationOptions,
) -> Result<SurfaceDiscretization, SurfaceDiscretizationError> {
    discretize_cad_topology_surfaces_with_curve_inputs(
        cad_topology,
        topology,
        cad_evaluation,
        curves,
        None,
        options,
    )
}

pub fn discretize_cad_topology_surfaces_with_cad_curves(
    cad_topology: &CadTopologyModel,
    topology: &SourceTopologyModel,
    cad_evaluation: &CadEvaluationModel,
    cad_curves: &CadCurveDiscretization,
    options: SurfaceDiscretizationOptions,
) -> Result<SurfaceDiscretization, SurfaceDiscretizationError> {
    discretize_cad_topology_surfaces_with_curve_inputs(
        cad_topology,
        topology,
        cad_evaluation,
        &cad_curves.curves,
        Some(&cad_curves.edge_provenance),
        options,
    )
}

fn discretize_cad_topology_surfaces_with_curve_inputs(
    cad_topology: &CadTopologyModel,
    topology: &SourceTopologyModel,
    cad_evaluation: &CadEvaluationModel,
    curves: &CurveDiscretization,
    cad_curve_edge_provenance: Option<&[CadCurveEdgeProvenance]>,
    options: SurfaceDiscretizationOptions,
) -> Result<SurfaceDiscretization, SurfaceDiscretizationError> {
    let curve_boundary_validation =
        validate_curve_discretization(topology, curves, cad_curve_validation_options())
            .map_err(SurfaceDiscretizationError::InvalidCurveBoundary)?;
    let mut nodes = topology
        .vertices
        .iter()
        .map(|vertex| SurfaceNode {
            node_id: vertex.vertex_id,
            source_vertex_id: vertex.vertex_id,
            coordinates_m: vertex.coordinates_m,
        })
        .collect::<Vec<_>>();
    let frames_by_cad_face = cad_evaluation
        .face_frames
        .iter()
        .map(|frame| (frame.face_id.clone(), frame))
        .collect::<BTreeMap<_, _>>();
    let topology_edges = topology
        .edges
        .iter()
        .map(|edge| (edge.edge_id, edge))
        .collect::<BTreeMap<_, _>>();
    let curve_nodes_by_edge = curve_nodes_by_source_edge(curves);
    let mut curve_node_to_surface_node = BTreeMap::<u32, u32>::new();
    let cad_curve_provenance_by_source_edge = cad_curve_edge_provenance.map(|edge_provenance| {
        edge_provenance
            .iter()
            .map(|provenance| (provenance.source_edge_id, provenance))
            .collect::<BTreeMap<_, _>>()
    });
    let cad_loops_by_id = cad_topology
        .loops
        .iter()
        .map(|cad_loop| (cad_loop.entity_id.id.as_str(), cad_loop))
        .collect::<BTreeMap<_, _>>();

    let mut elements = Vec::<SurfaceElement>::new();
    let mut loop_coverage = SurfaceLoopCoverageAccumulator::new(cad_topology.faces.len());
    let mut cad_curve_boundary_provenance = cad_curve_provenance_by_source_edge
        .as_ref()
        .map(|_| SurfaceCadCurveBoundaryProvenanceAccumulator::new());
    let mut exact_cad_sample_node_count = 0_usize;
    let mut rejected_exact_cad_sample_count = 0_usize;
    for cad_face in &cad_topology.faces {
        let face = cad_face_surface_seed(cad_face, &cad_loops_by_id, &topology_edges)?;
        validate_face_vertices(topology, &face)?;
        let frame = frames_by_cad_face.get(&cad_face.entity_id.id).ok_or(
            SurfaceDiscretizationError::MissingCadFaceFrame {
                source_face_id: face.face_id,
            },
        )?;
        let source_edge_ids = cad_face_loop_source_edge_ids(cad_face, &cad_loops_by_id)?;
        let segments = curve_segments_for_source_edges(
            &topology_edges,
            &curve_nodes_by_edge,
            face.face_id,
            &source_edge_ids,
            options.max_curve_segments_per_edge.max(1),
            &mut nodes,
            &mut curve_node_to_surface_node,
        )?;
        let segment_loops = face_curve_segment_loops(face.face_id, &segments)?;
        loop_coverage.record_face_edges(&source_edge_ids, &segment_loops);
        if let (Some(accumulator), Some(provenance_by_source_edge)) = (
            cad_curve_boundary_provenance.as_mut(),
            cad_curve_provenance_by_source_edge.as_ref(),
        ) {
            accumulator.record_segments(&segment_loops, provenance_by_source_edge)?;
        }
        let sample_report = append_curve_driven_face_elements(
            &face,
            frame,
            &segment_loops,
            &mut nodes,
            &mut elements,
        );
        exact_cad_sample_node_count += sample_report.accepted_count;
        rejected_exact_cad_sample_count += sample_report.rejected_count;
    }

    Ok(SurfaceDiscretization {
        nodes,
        elements,
        curve_boundary_validation: Some(curve_boundary_validation),
        loop_coverage: Some(loop_coverage.finish()),
        cad_curve_boundary_provenance: cad_curve_boundary_provenance
            .map(SurfaceCadCurveBoundaryProvenanceAccumulator::finish),
        exact_cad_sample_node_count,
        rejected_exact_cad_sample_count,
    })
}

fn cad_face_surface_seed(
    cad_face: &CadFace,
    cad_loops_by_id: &BTreeMap<&str, &CadLoop>,
    topology_edges: &BTreeMap<u32, &SourceTopologyEdge>,
) -> Result<SourceTopologyFace, SurfaceDiscretizationError> {
    let face_id = cad_face.source_face_ids.first().copied().ok_or_else(|| {
        SurfaceDiscretizationError::CadFaceWithoutSourceFaces {
            cad_face_id: cad_face.entity_id.id.clone(),
        }
    })?;
    let source_edge_ids = cad_face_loop_source_edge_ids(cad_face, cad_loops_by_id)?;
    if source_edge_ids.len() < 3 {
        return Err(SurfaceDiscretizationError::EmptyFaceLoop { face_id });
    }
    let mut node_ids = Vec::<u32>::new();
    for source_edge_id in &source_edge_ids {
        let edge = topology_edges.get(source_edge_id).ok_or(
            SurfaceDiscretizationError::MissingFaceEdge {
                face_id,
                edge_id: *source_edge_id,
            },
        )?;
        for node_id in edge.node_ids {
            if !node_ids.contains(&node_id) {
                node_ids.push(node_id);
            }
        }
    }
    if node_ids.len() < 3 {
        return Err(SurfaceDiscretizationError::EmptyFaceLoop { face_id });
    }
    Ok(SourceTopologyFace {
        face_id,
        source_triangle_id: face_id,
        node_ids: [node_ids[0], node_ids[1], node_ids[2]],
        edge_ids: [source_edge_ids[0], source_edge_ids[1], source_edge_ids[2]],
        region_ids: cad_face.region_ids.clone(),
        material_region_ids: cad_face.material_region_ids.clone(),
        area_m2: cad_face.area_m2,
        unit_normal: cad_face.unit_normal,
    })
}

fn cad_curve_validation_options() -> CurveValidationOptions {
    CurveValidationOptions {
        require_source_edge_projection: false,
        ..CurveValidationOptions::default()
    }
}

fn cad_face_loop_source_edge_ids(
    cad_face: &CadFace,
    cad_loops_by_id: &BTreeMap<&str, &CadLoop>,
) -> Result<Vec<u32>, SurfaceDiscretizationError> {
    if cad_face.loop_ids.is_empty() {
        return Err(SurfaceDiscretizationError::CadFaceWithoutLoops {
            cad_face_id: cad_face.entity_id.id.clone(),
        });
    }
    let mut source_edge_ids = Vec::<u32>::new();
    for loop_id in &cad_face.loop_ids {
        let cad_loop = cad_loops_by_id.get(loop_id.as_str()).ok_or_else(|| {
            SurfaceDiscretizationError::MissingCadFaceLoop {
                cad_face_id: cad_face.entity_id.id.clone(),
                loop_id: loop_id.clone(),
            }
        })?;
        if cad_loop.face_id != cad_face.entity_id.id {
            return Err(SurfaceDiscretizationError::CadLoopFaceMismatch {
                cad_face_id: cad_face.entity_id.id.clone(),
                loop_id: loop_id.clone(),
                actual_face_id: cad_loop.face_id.clone(),
            });
        }
        source_edge_ids.extend(
            cad_loop
                .edge_ids
                .iter()
                .map(|loop_edge_id| {
                    loop_edge_id
                        .strip_prefix("cad_edge_")
                        .and_then(|edge_id| edge_id.parse::<u32>().ok())
                        .ok_or_else(|| SurfaceDiscretizationError::InvalidCadLoopEdgeId {
                            cad_face_id: cad_face.entity_id.id.clone(),
                            loop_edge_id: loop_edge_id.clone(),
                        })
                })
                .collect::<Result<Vec<_>, _>>()?,
        );
    }
    Ok(source_edge_ids)
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct FaceTriangulationPoint {
    node_id: u32,
    uv: [f64; 2],
}

fn validate_face_vertices(
    topology: &SourceTopologyModel,
    face: &SourceTopologyFace,
) -> Result<(), SurfaceDiscretizationError> {
    for node_id in face.node_ids {
        if topology
            .vertices
            .get(node_id as usize)
            .is_none_or(|vertex| vertex.vertex_id != node_id)
        {
            return Err(SurfaceDiscretizationError::MissingFaceVertex {
                face_id: face.face_id,
                node_id,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
