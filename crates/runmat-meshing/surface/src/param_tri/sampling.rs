use std::collections::BTreeMap;

use runmat_geometry_core::{CadFaceEvaluationSample, CadFaceEvaluationSampleSource};
use runmat_meshing_cad::{project_to_face, CadFaceEvaluationFrame, SourceTopologyFace};

use crate::math::{dot, sub};

use super::{
    geometry::{
        distance2_2d, finite_point2, finite_point3, point_in_triangle_3d,
        point_in_trimmed_domain_2d, sorted_node_pair,
    },
    FaceCurveSegment, FaceTriangulationPoint, SurfaceElement, SurfaceNode,
};

const FACE_AREA_RECOVERY_TOLERANCE: f64 = 1.0e-8;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct ExactCadSampleSurfaceReport {
    pub(super) accepted_count: usize,
    pub(super) rejected_count: usize,
}

impl ExactCadSampleSurfaceReport {
    pub(super) fn rejected_after_area_guard(self) -> Self {
        Self {
            accepted_count: 0,
            rejected_count: self.rejected_count + self.accepted_count,
        }
    }
}

pub(super) fn face_area_is_recovered(
    face: &SourceTopologyFace,
    elements: &[SurfaceElement],
) -> bool {
    if elements.is_empty() {
        return false;
    }
    let recovered_area_m2 = elements
        .iter()
        .filter(|element| element.source_face_id == face.face_id)
        .map(|element| element.area_m2)
        .sum::<f64>();
    if !recovered_area_m2.is_finite() || !face.area_m2.is_finite() || face.area_m2 <= 0.0 {
        return false;
    }
    ((recovered_area_m2 - face.area_m2).abs() / face.area_m2) <= FACE_AREA_RECOVERY_TOLERANCE
}

pub(super) fn face_edges_are_recovered(
    elements: &[SurfaceElement],
    boundary_edge_ids: &BTreeMap<[u32; 2], u32>,
) -> bool {
    let mut edge_counts = BTreeMap::<[u32; 2], usize>::new();
    for element in elements {
        for edge in [
            sorted_node_pair(element.node_ids[0], element.node_ids[1]),
            sorted_node_pair(element.node_ids[1], element.node_ids[2]),
            sorted_node_pair(element.node_ids[2], element.node_ids[0]),
        ] {
            *edge_counts.entry(edge).or_default() += 1;
        }
    }
    edge_counts.into_iter().all(|(edge, count)| {
        if boundary_edge_ids.contains_key(&edge) {
            count == 1
        } else {
            count == 2
        }
    })
}

pub(super) fn append_exact_face_domain_sample_points(
    face: &SourceTopologyFace,
    frame: &CadFaceEvaluationFrame,
    boundary_polygons: &[Vec<[f64; 2]>],
    nodes: &mut Vec<SurfaceNode>,
    points: &mut Vec<FaceTriangulationPoint>,
) -> ExactCadSampleSurfaceReport {
    let mut report = ExactCadSampleSurfaceReport::default();
    let face_points = face
        .node_ids
        .map(|node_id| nodes[node_id as usize].coordinates_m);
    for sample in &frame.evaluator_samples {
        if !is_usable_exact_face_domain_sample(sample) {
            report.rejected_count += 1;
            continue;
        }
        let coordinates = sample
            .projected_point_m
            .filter(|point| finite_point3(*point))
            .unwrap_or(sample.point_m);
        let projection = project_to_face(frame, coordinates);
        if !point_in_triangle_3d(projection.point_m, face_points) {
            report.rejected_count += 1;
            continue;
        }
        let local_uv = frame_local_uv(frame, projection.point_m);
        if !point_in_trimmed_domain_2d(local_uv, boundary_polygons) {
            report.rejected_count += 1;
            continue;
        }
        if points
            .iter()
            .any(|point| distance2_2d(point.uv, local_uv) <= 1.0e-24)
        {
            report.rejected_count += 1;
            continue;
        }
        let node_id = nodes.len() as u32;
        nodes.push(SurfaceNode {
            node_id,
            source_vertex_id: u32::MAX,
            coordinates_m: projection.point_m,
        });
        points.push(FaceTriangulationPoint {
            node_id,
            uv: local_uv,
        });
        report.accepted_count += 1;
    }
    report
}

fn frame_local_uv(frame: &CadFaceEvaluationFrame, point_m: [f64; 3]) -> [f64; 2] {
    let relative = sub(point_m, frame.origin_m);
    [dot(relative, frame.u_axis), dot(relative, frame.v_axis)]
}

pub(super) fn has_exact_face_domain_samples(frame: &CadFaceEvaluationFrame) -> bool {
    frame
        .evaluator_samples
        .iter()
        .any(is_usable_exact_face_domain_sample)
}

fn is_usable_exact_face_domain_sample(sample: &CadFaceEvaluationSample) -> bool {
    sample.source == CadFaceEvaluationSampleSource::BackendQuery
        && finite_point3(sample.point_m)
        && sample.uv.is_some_and(finite_point2)
        && sample.projected_point_m.is_none_or(finite_point3)
}

pub(super) fn append_face_lattice_points(
    face: &SourceTopologyFace,
    frame: &CadFaceEvaluationFrame,
    boundary_polygons: &[Vec<[f64; 2]>],
    segments: &[FaceCurveSegment],
    nodes: &mut Vec<SurfaceNode>,
    points: &mut Vec<FaceTriangulationPoint>,
) {
    let segment_count = segments_per_source_edge(segments)
        .into_values()
        .max()
        .unwrap_or(1)
        .max(2);
    if segment_count <= 2 {
        return;
    }
    let corners = face
        .node_ids
        .map(|node_id| nodes[node_id as usize].coordinates_m);
    for i in 1..segment_count {
        for j in 1..(segment_count - i) {
            let u = i as f64 / segment_count as f64;
            let v = j as f64 / segment_count as f64;
            let w = 1.0 - u - v;
            if w <= f64::EPSILON {
                continue;
            }
            let coordinates = [
                corners[0][0] * w + corners[1][0] * u + corners[2][0] * v,
                corners[0][1] * w + corners[1][1] * u + corners[2][1] * v,
                corners[0][2] * w + corners[1][2] * u + corners[2][2] * v,
            ];
            let projection = project_to_face(frame, coordinates);
            if !projection.uv_in_bounds {
                continue;
            }
            if !point_in_trimmed_domain_2d(projection.uv, boundary_polygons) {
                continue;
            }
            if points
                .iter()
                .any(|point| distance2_2d(point.uv, projection.uv) <= 1.0e-24)
            {
                continue;
            }
            let node_id = nodes.len() as u32;
            nodes.push(SurfaceNode {
                node_id,
                source_vertex_id: u32::MAX,
                coordinates_m: projection.point_m,
            });
            points.push(FaceTriangulationPoint {
                node_id,
                uv: projection.uv,
            });
        }
    }
}

fn segments_per_source_edge(segments: &[FaceCurveSegment]) -> BTreeMap<u32, usize> {
    let mut counts = BTreeMap::<u32, usize>::new();
    for segment in segments {
        *counts.entry(segment.source_edge_id).or_default() += 1;
    }
    counts
}
