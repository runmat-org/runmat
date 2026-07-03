use std::collections::BTreeMap;

use crate::{
    math::{cross, dot, norm, scale, sub, triangle_centroid, Point3, Triangle3},
    topology::{CadFace, CadTopologyModel, SourceTopologyModel},
};
use runmat_geometry_core::{CadFaceEvaluationSample, CadFaceEvaluationSampleSource};

use super::projection::project_to_face;
use super::{
    samples::{
        estimate_max_curvature, estimate_uv_derivatives, evaluator_max_projection_error,
        exact_backend_sample, exact_backend_sample_point, live_evaluator_samples,
        max_optional_curvature, merged_bounded_evaluator_samples,
    },
    types::{
        CadEvaluationError, CadEvaluationModel, CadEvaluationReport, CadEvaluationSource,
        CadFaceEvaluationFrame, CadFaceEvaluatorProvider, NoopCadFaceEvaluatorProvider,
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
    let evaluator_backed_frame_count = frames.iter().filter(|frame| frame.evaluator_backed).count();
    let live_query_face_count = frames
        .iter()
        .filter(|frame| frame.live_query_backed)
        .count();
    let exact_query_face_count = frames
        .iter()
        .filter(|frame| frame.exact_query_backed)
        .count();
    let point_evaluation_supported_face_count =
        evaluator_supported_source_face_count(cad_topology, |face| {
            face.evaluator_supports_point_evaluation
        });
    let projection_supported_face_count =
        evaluator_supported_source_face_count(cad_topology, |face| {
            face.evaluator_supports_projection
        });
    let normal_supported_face_count =
        evaluator_supported_source_face_count(cad_topology, |face| face.evaluator_supports_normal);
    let derivative_supported_face_count =
        evaluator_supported_source_face_count(cad_topology, |face| {
            face.evaluator_supports_derivatives
        });
    let curvature_supported_face_count =
        evaluator_supported_source_face_count(cad_topology, |face| {
            face.evaluator_supports_curvature
        });
    let missing_exact_query_face_count =
        evaluator_backed_frame_count.saturating_sub(exact_query_face_count);
    let evaluator_sample_count = frames
        .iter()
        .map(|frame| frame.evaluator_sample_count)
        .sum();
    let evaluator_rejected_sample_count = frames
        .iter()
        .map(|frame| frame.evaluator_rejected_sample_count)
        .sum();
    let max_evaluator_projection_error_m = frames
        .iter()
        .map(|frame| frame.evaluator_max_projection_error_m)
        .fold(0.0_f64, f64::max);
    let derivative_query_count = frames
        .iter()
        .filter(|frame| {
            frame.u_derivative_m_per_uv.is_some() && frame.v_derivative_m_per_uv.is_some()
        })
        .count();
    let curvature_query_count = frames
        .iter()
        .filter(|frame| frame.max_curvature_estimate_1_per_m.is_some())
        .count();
    let missing_derivative_query_face_count =
        derivative_supported_face_count.saturating_sub(derivative_query_count);
    let missing_curvature_query_face_count =
        curvature_supported_face_count.saturating_sub(curvature_query_count);
    let uv_domain_face_count = frames
        .iter()
        .filter(|frame| frame.uv_bounds.is_some())
        .count();
    let max_curvature_estimate_1_per_m = max_optional_curvature(
        frames
            .iter()
            .filter_map(|frame| frame.max_curvature_estimate_1_per_m),
    );
    let report = CadEvaluationReport {
        source: evaluation_source(
            frames.len(),
            evaluator_backed_frame_count,
            live_query_face_count,
            exact_query_face_count,
        ),
        face_frame_count: frames.len(),
        evaluator_face_count: cad_topology.report.evaluator_face_count,
        live_query_face_count,
        exact_query_face_count,
        point_evaluation_supported_face_count,
        projection_supported_face_count,
        normal_supported_face_count,
        derivative_supported_face_count,
        curvature_supported_face_count,
        missing_exact_query_face_count,
        missing_derivative_query_face_count,
        missing_curvature_query_face_count,
        evaluator_sample_count,
        evaluator_rejected_sample_count,
        normal_query_count: frames.len(),
        projection_query_count: frames.len(),
        derivative_query_count,
        curvature_query_count,
        max_projection_error_m: max_evaluator_projection_error_m,
        max_normal_deviation: 0.0,
        uv_domain_face_count,
        uv_projection_out_of_bounds_count: 0,
        max_curvature_estimate_1_per_m,
    };
    Ok(CadEvaluationModel {
        source_geometry_id: cad_topology.source_geometry_id.clone(),
        source_geometry_revision: cad_topology.source_geometry_revision,
        source: report.source,
        face_frames: frames,
        report,
    })
}

pub fn summarize_cad_evaluation(
    model: &CadEvaluationModel,
    topology: &SourceTopologyModel,
) -> Result<CadEvaluationReport, CadEvaluationError> {
    let mut projection_query_count = 0_usize;
    let mut max_projection_error_m = 0.0_f64;
    let mut max_normal_deviation = 0.0_f64;
    let mut uv_domain_face_count = 0_usize;
    let mut uv_projection_out_of_bounds_count = 0_usize;
    let source_faces = topology
        .faces
        .iter()
        .map(|face| (face.face_id, face))
        .collect::<BTreeMap<_, _>>();
    for frame in &model.face_frames {
        let source_face = source_faces.get(&frame.source_face_id).ok_or(
            CadEvaluationError::MissingSourceFace {
                source_face_id: frame.source_face_id,
            },
        )?;
        let points = [
            topology_vertex(topology, source_face.node_ids[0])?,
            topology_vertex(topology, source_face.node_ids[1])?,
            topology_vertex(topology, source_face.node_ids[2])?,
        ];
        for point in points {
            let projection = project_to_face(frame, point);
            projection_query_count += 1;
            max_projection_error_m = max_projection_error_m.max(projection.distance_m);
            if !projection.uv_in_bounds {
                uv_projection_out_of_bounds_count += 1;
            }
        }
        if frame.uv_bounds.is_some() {
            uv_domain_face_count += 1;
        }
        max_projection_error_m = max_projection_error_m.max(frame.evaluator_max_projection_error_m);
        max_normal_deviation =
            max_normal_deviation.max(1.0 - dot(frame.unit_normal, source_face.unit_normal).abs());
    }
    Ok(CadEvaluationReport {
        source: model.source,
        face_frame_count: model.face_frames.len(),
        evaluator_face_count: model.report.evaluator_face_count,
        live_query_face_count: model.report.live_query_face_count,
        exact_query_face_count: model.report.exact_query_face_count,
        point_evaluation_supported_face_count: model.report.point_evaluation_supported_face_count,
        projection_supported_face_count: model.report.projection_supported_face_count,
        normal_supported_face_count: model.report.normal_supported_face_count,
        derivative_supported_face_count: model.report.derivative_supported_face_count,
        curvature_supported_face_count: model.report.curvature_supported_face_count,
        missing_exact_query_face_count: model.report.missing_exact_query_face_count,
        missing_derivative_query_face_count: model.report.missing_derivative_query_face_count,
        missing_curvature_query_face_count: model.report.missing_curvature_query_face_count,
        evaluator_sample_count: model.report.evaluator_sample_count,
        evaluator_rejected_sample_count: model.report.evaluator_rejected_sample_count,
        normal_query_count: model.face_frames.len(),
        projection_query_count,
        derivative_query_count: model.report.derivative_query_count,
        curvature_query_count: model.report.curvature_query_count,
        max_projection_error_m,
        max_normal_deviation,
        uv_domain_face_count,
        uv_projection_out_of_bounds_count,
        max_curvature_estimate_1_per_m: model.report.max_curvature_estimate_1_per_m,
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

fn evaluation_source(
    _face_frame_count: usize,
    evaluator_backed_frame_count: usize,
    live_query_face_count: usize,
    exact_query_face_count: usize,
) -> CadEvaluationSource {
    if live_query_face_count > 0 {
        CadEvaluationSource::ParametricCad
    } else if evaluator_backed_frame_count > 0 || exact_query_face_count > 0 {
        CadEvaluationSource::ImportedEvaluatorSamples
    } else {
        CadEvaluationSource::PlanarFacetApproximation
    }
}

fn evaluator_supported_source_face_count(
    cad_topology: &CadTopologyModel,
    predicate: impl Fn(&CadFace) -> bool,
) -> usize {
    cad_topology
        .faces
        .iter()
        .filter(|face| predicate(face))
        .map(|face| face.source_face_ids.len().max(1))
        .sum()
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
