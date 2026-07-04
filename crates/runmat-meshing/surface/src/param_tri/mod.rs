use std::collections::BTreeMap;

use runmat_meshing_cad::{
    project_to_face, CadEvaluationModel, SourceTopologyFace, SourceTopologyModel,
};
use runmat_meshing_curve::CurveDiscretization;

mod boundary;
mod elements;
mod geometry;
mod sampling;
mod subdivision;
mod triangulation;
mod types;

use boundary::{
    curve_nodes_by_source_edge, face_curve_segment_loops, oriented_face_curve_segments,
};
use elements::append_curve_driven_face_elements;
use subdivision::{append_centroid_subdivision, triangle_centroid};
pub use types::{
    SurfaceDiscretization, SurfaceDiscretizationError, SurfaceDiscretizationOptions,
    SurfaceElement, SurfaceNode,
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
            area_m2: face.area_m2,
            unit_normal: face.unit_normal,
        });
    }

    Ok(SurfaceDiscretization {
        nodes,
        elements,
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
                area_m2: face.area_m2,
                unit_normal: frame.unit_normal,
            });
        }
    }

    Ok(SurfaceDiscretization {
        nodes,
        elements,
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
        exact_cad_sample_node_count,
        rejected_exact_cad_sample_count,
    })
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
