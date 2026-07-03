use std::collections::BTreeMap;

use crate::{
    math::{cross, dot, norm, scale, sub, triangle_centroid, Point3, Triangle3},
    topology::{CadTopologyModel, SourceTopologyModel},
};
use runmat_geometry_core::{CadFaceEvaluationSample, CadFaceEvaluationSampleSource};

use super::report::build_cad_evaluation_report;
use super::{
    samples::{
        estimate_max_curvature, estimate_uv_derivatives, evaluator_max_projection_error,
        exact_backend_sample, exact_backend_sample_point, live_evaluator_samples,
        merged_bounded_evaluator_samples,
    },
    types::{
        CadEvaluationError, CadEvaluationModel, CadFaceEvaluationFrame, CadFaceEvaluatorProvider,
        NoopCadFaceEvaluatorProvider,
    },
};

pub fn build_cad_evaluation_model(
    cad_topology: &CadTopologyModel,
    topology: &SourceTopologyModel,
) -> Result<CadEvaluationModel, CadEvaluationError> {
    build_cad_evaluation_model_with_provider(cad_topology, topology, &NoopCadFaceEvaluatorProvider)
}

pub fn build_cad_evaluation_model_with_provider(
    cad_topology: &CadTopologyModel,
    topology: &SourceTopologyModel,
    evaluator_provider: &dyn CadFaceEvaluatorProvider,
) -> Result<CadEvaluationModel, CadEvaluationError> {
    if cad_topology.faces.is_empty() {
        return Err(CadEvaluationError::EmptyFaces);
    }
    let source_faces = topology
        .faces
        .iter()
        .map(|face| (face.face_id, face))
        .collect::<BTreeMap<_, _>>();
    let frame_capacity = cad_topology
        .faces
        .iter()
        .map(|face| face.source_face_ids.len().max(1))
        .sum();
    let mut frames = Vec::<CadFaceEvaluationFrame>::with_capacity(frame_capacity);
    for face in &cad_topology.faces {
        if face.source_face_ids.is_empty() {
            return Err(CadEvaluationError::MissingSourceFace { source_face_id: 0 });
        }
        for source_face_id in &face.source_face_ids {
            let source_face =
                source_faces
                    .get(source_face_id)
                    .ok_or(CadEvaluationError::MissingSourceFace {
                        source_face_id: *source_face_id,
                    })?;
            let points = [
                topology_vertex(topology, source_face.node_ids[0])?,
                topology_vertex(topology, source_face.node_ids[1])?,
                topology_vertex(topology, source_face.node_ids[2])?,
            ];
            let fallback_reference_point_m = face
                .evaluator_reference_point_m
                .unwrap_or_else(|| triangle_centroid(points));
            let fallback_unit_normal = face
                .evaluator_unit_normal
                .unwrap_or(source_face.unit_normal);
            let live_samples = live_evaluator_samples(
                evaluator_provider,
                face,
                *source_face_id,
                fallback_reference_point_m,
                fallback_unit_normal,
            );
            let live_query_backed = !live_samples.samples.is_empty();
            let evaluator_samples = merged_bounded_evaluator_samples(face, live_samples, points);
            let exact_sample = exact_backend_sample(&evaluator_samples.samples);
            let evaluator_max_projection_error_m =
                evaluator_max_projection_error(&evaluator_samples.samples);
            let frame = face_frame(
                face.entity_id.id.clone(),
                *source_face_id,
                points,
                exact_sample
                    .and_then(|sample| sample.unit_normal)
                    .or(face.evaluator_unit_normal)
                    .unwrap_or(source_face.unit_normal),
                source_face.area_m2,
                exact_sample
                    .map(exact_backend_sample_point)
                    .or(face.evaluator_reference_point_m),
                face.evaluator_id.is_some()
                    || face.evaluator_unit_normal.is_some()
                    || !evaluator_samples.samples.is_empty(),
                exact_sample.is_some(),
                live_query_backed,
                evaluator_samples.samples.len(),
                evaluator_samples.rejected_count,
                evaluator_max_projection_error_m,
                evaluator_samples.samples,
            )?;
            frames.push(frame);
        }
    }
    let report = build_cad_evaluation_report(cad_topology, &frames);
    Ok(CadEvaluationModel {
        source_geometry_id: cad_topology.source_geometry_id.clone(),
        source_geometry_revision: cad_topology.source_geometry_revision,
        source: report.source,
        face_frames: frames,
        report,
    })
}

fn face_frame(
    face_id: String,
    source_face_id: u32,
    points: Triangle3,
    unit_normal: Point3,
    area_m2: f64,
    evaluator_reference_point_m: Option<Point3>,
    evaluator_backed: bool,
    exact_query_backed: bool,
    live_query_backed: bool,
    evaluator_sample_count: usize,
    evaluator_rejected_sample_count: usize,
    evaluator_max_projection_error_m: f64,
    evaluator_samples: Vec<CadFaceEvaluationSample>,
) -> Result<CadFaceEvaluationFrame, CadEvaluationError> {
    let edge = sub(points[1], points[0]);
    let edge_length = norm(edge);
    let normal_length = norm(unit_normal);
    if edge_length <= f64::EPSILON || normal_length <= f64::EPSILON {
        return Err(CadEvaluationError::DegenerateFace { source_face_id });
    }
    let unit_normal =
        orient_unit_normal_to_source_triangle(scale(unit_normal, 1.0 / normal_length), points);
    let u_axis = scale(edge, 1.0 / edge_length);
    let v_axis = cross(unit_normal, u_axis);
    let v_length = norm(v_axis);
    if v_length <= f64::EPSILON {
        return Err(CadEvaluationError::DegenerateFace { source_face_id });
    }
    let origin_m = evaluator_reference_point_m.unwrap_or_else(|| triangle_centroid(points));
    let (u_derivative_m_per_uv, v_derivative_m_per_uv) =
        estimate_uv_derivatives(&evaluator_samples);
    let max_curvature_estimate_1_per_m = estimate_max_curvature(&evaluator_samples, unit_normal);
    let (uv_bounds, uv_bounds_sample_count, uv_domain_source) = cad_uv_domain_summary(
        &evaluator_samples,
        points,
        origin_m,
        u_axis,
        scale(v_axis, 1.0 / v_length),
    );
    Ok(CadFaceEvaluationFrame {
        face_id,
        source_face_id,
        origin_m,
        u_axis,
        v_axis: scale(v_axis, 1.0 / v_length),
        unit_normal,
        area_m2,
        evaluator_backed,
        exact_query_backed,
        live_query_backed,
        evaluator_sample_count,
        evaluator_rejected_sample_count,
        evaluator_max_projection_error_m,
        evaluator_samples,
        u_derivative_m_per_uv,
        v_derivative_m_per_uv,
        max_curvature_estimate_1_per_m,
        uv_bounds,
        uv_bounds_sample_count,
        uv_domain_source,
    })
}

fn cad_uv_domain_summary(
    evaluator_samples: &[CadFaceEvaluationSample],
    source_points: Triangle3,
    origin: Point3,
    u_axis: Point3,
    v_axis: Point3,
) -> (Option<[[f64; 2]; 2]>, usize, Option<String>) {
    let exact_sample_uvs = evaluator_samples
        .iter()
        .filter(|sample| sample.source == CadFaceEvaluationSampleSource::BackendQuery)
        .filter_map(|sample| sample.uv)
        .filter(|uv| uv.iter().all(|value| value.is_finite()))
        .collect::<Vec<_>>();
    if exact_sample_uvs.len() >= 3 {
        return (
            uv_bounds_from_points(exact_sample_uvs.as_slice()),
            exact_sample_uvs.len(),
            Some("exact_samples".to_string()),
        );
    }

    let fallback_uvs = source_points
        .iter()
        .map(|point| {
            let relative = sub(*point, origin);
            [dot(relative, u_axis), dot(relative, v_axis)]
        })
        .collect::<Vec<_>>();
    (
        uv_bounds_from_points(fallback_uvs.as_slice()),
        fallback_uvs.len(),
        Some("source_face_projection".to_string()),
    )
}

fn uv_bounds_from_points(points: &[[f64; 2]]) -> Option<[[f64; 2]; 2]> {
    let mut finite_points = points
        .iter()
        .copied()
        .filter(|uv| uv.iter().all(|value| value.is_finite()));
    let first = finite_points.next()?;
    let mut min = first;
    let mut max = first;
    for uv in finite_points {
        min[0] = min[0].min(uv[0]);
        min[1] = min[1].min(uv[1]);
        max[0] = max[0].max(uv[0]);
        max[1] = max[1].max(uv[1]);
    }
    Some([min, max])
}

fn orient_unit_normal_to_source_triangle(unit_normal: Point3, points: Triangle3) -> Point3 {
    let source_normal = cross(sub(points[1], points[0]), sub(points[2], points[0]));
    let source_normal_length = norm(source_normal);
    if source_normal_length <= f64::EPSILON {
        return unit_normal;
    }
    let source_unit_normal = scale(source_normal, 1.0 / source_normal_length);
    if dot(unit_normal, source_unit_normal) < 0.0 {
        scale(unit_normal, -1.0)
    } else {
        unit_normal
    }
}

fn topology_vertex(
    topology: &SourceTopologyModel,
    vertex_id: u32,
) -> Result<Point3, CadEvaluationError> {
    topology
        .vertices
        .get(vertex_id as usize)
        .filter(|vertex| vertex.vertex_id == vertex_id)
        .map(|vertex| vertex.coordinates_m)
        .ok_or(CadEvaluationError::MissingSourceVertex { vertex_id })
}

#[cfg(test)]
mod tests;
