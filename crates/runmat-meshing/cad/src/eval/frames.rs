use std::collections::BTreeMap;

use crate::{
    math::{cross, dot, norm, scale, sub, triangle_centroid, Point3, Triangle3},
    topology::{CadFace, CadTopologyModel, SourceTopologyModel},
};
use runmat_geometry_core::{CadFaceEvaluationSample, CadFaceEvaluationSampleSource};

#[cfg(test)]
use super::types::CadFaceEvaluationRequest;
use super::{
    samples::{
        estimate_max_curvature, estimate_uv_derivatives, evaluator_max_projection_error,
        exact_backend_sample, exact_backend_sample_is_valid, exact_backend_sample_point,
        live_evaluator_samples, max_optional_curvature, merged_bounded_evaluator_samples,
        normalized_sample_normal, orient_sample_normal,
    },
    types::{
        CadEvaluationError, CadEvaluationModel, CadEvaluationReport, CadEvaluationSource,
        CadFaceEvaluationFrame, CadFaceEvaluatorProvider, CadFaceProjection,
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

pub fn project_to_face(frame: &CadFaceEvaluationFrame, point: Point3) -> CadFaceProjection {
    if let Some(projection) = sample_backed_projection(frame, point) {
        return projection;
    }

    let relative = sub(point, frame.origin_m);
    let normal_distance = dot(relative, frame.unit_normal);
    let projected = sub(point, scale(frame.unit_normal, normal_distance));
    let projected_relative = sub(projected, frame.origin_m);
    CadFaceProjection {
        point_m: projected,
        uv: [
            dot(projected_relative, frame.u_axis),
            dot(projected_relative, frame.v_axis),
        ],
        distance_m: normal_distance.abs(),
        unit_normal: frame.unit_normal,
        uv_in_bounds: face_uv_contains(
            frame,
            [
                dot(projected_relative, frame.u_axis),
                dot(projected_relative, frame.v_axis),
            ],
        ),
    }
}

pub fn face_uv_contains(frame: &CadFaceEvaluationFrame, uv: [f64; 2]) -> bool {
    let Some(bounds) = frame.uv_bounds else {
        return true;
    };
    if !uv.iter().all(|value| value.is_finite()) {
        return false;
    }
    let tolerance = 1.0e-9;
    uv[0] + tolerance >= bounds[0][0]
        && uv[0] <= bounds[1][0] + tolerance
        && uv[1] + tolerance >= bounds[0][1]
        && uv[1] <= bounds[1][1] + tolerance
}

fn sample_backed_projection(
    frame: &CadFaceEvaluationFrame,
    point: Point3,
) -> Option<CadFaceProjection> {
    frame
        .evaluator_samples
        .iter()
        .filter(|sample| exact_backend_sample_is_valid(sample))
        .filter_map(|sample| projection_from_matching_sample(frame, point, sample))
        .min_by(|left, right| left.distance_m.total_cmp(&right.distance_m))
}

fn projection_from_matching_sample(
    frame: &CadFaceEvaluationFrame,
    point: Point3,
    sample: &CadFaceEvaluationSample,
) -> Option<CadFaceProjection> {
    let uv = sample.uv?;
    if !uv.iter().all(|value| value.is_finite()) {
        return None;
    }
    let projected = exact_backend_sample_point(sample);
    let projection_error_m = sample.projection_error_m.unwrap_or(0.0);
    let match_tolerance_m = projection_error_m.max(1.0e-10);
    let point_to_query_m = norm(sub(point, sample.point_m));
    let point_to_projected_m = norm(sub(point, projected));
    if point_to_query_m > match_tolerance_m && point_to_projected_m > match_tolerance_m {
        return None;
    }
    let unit_normal = sample
        .unit_normal
        .and_then(normalized_sample_normal)
        .map(|normal| orient_sample_normal(normal, frame.unit_normal))
        .unwrap_or(frame.unit_normal);
    Some(CadFaceProjection {
        point_m: projected,
        uv,
        distance_m: point_to_projected_m,
        unit_normal,
        uv_in_bounds: face_uv_contains(frame, uv),
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
mod tests {
    use super::*;
    use crate::build_cad_topology;
    use runmat_geometry_core::{
        CadEvaluatorSet, CadFaceEvaluationSample, CadFaceEvaluationSampleSource, CadFaceEvaluator,
        CadLabelRef, CadRegionOwnership, CadSemanticKind, EntityIdRange, EntityKind, Region,
        RegionEntityMapping,
    };

    #[test]
    fn builds_planar_face_evaluation_frames() {
        let topology = cube_topology();
        let geometry = geometry_for_topology();
        let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

        let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");
        let report = summarize_cad_evaluation(&model, &topology).expect("summary");

        assert_eq!(model.face_frames.len(), topology.faces.len());
        assert_eq!(report.face_frame_count, topology.faces.len());
        assert_eq!(report.source, CadEvaluationSource::PlanarFacetApproximation);
        assert_eq!(report.evaluator_face_count, 0);
        assert_eq!(report.projection_query_count, topology.faces.len() * 3);
        assert_eq!(report.max_projection_error_m, 0.0);
        assert_eq!(report.max_normal_deviation, 0.0);
    }

    #[test]
    fn projects_points_to_face_frame() {
        let topology = cube_topology();
        let geometry = geometry_for_topology();
        let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");
        let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");

        let projection = project_to_face(&model.face_frames[0], [0.25, 0.25, 0.5]);
        let outside_projection = project_to_face(&model.face_frames[0], [10.0, 10.0, 0.5]);

        assert!(projection.distance_m > 0.0);
        assert!(projection.uv_in_bounds);
        assert!(!outside_projection.uv_in_bounds);
        assert!(dot(projection.unit_normal, model.face_frames[0].unit_normal) > 0.999);
        assert_eq!(
            model.face_frames[0].uv_domain_source.as_deref(),
            Some("source_face_projection")
        );
    }

    #[test]
    fn uses_imported_evaluator_face_samples_when_available() {
        let topology = cube_topology();
        let geometry = geometry_with_face_evaluator();
        let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

        let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");

        assert_eq!(model.source, CadEvaluationSource::ImportedEvaluatorSamples);
        assert_eq!(model.report.evaluator_face_count, 1);
        assert_eq!(model.report.point_evaluation_supported_face_count, 2);
        assert_eq!(model.report.projection_supported_face_count, 2);
        assert_eq!(model.report.normal_supported_face_count, 2);
        assert_eq!(model.report.derivative_supported_face_count, 2);
        assert_eq!(model.report.curvature_supported_face_count, 2);
        assert_eq!(model.report.exact_query_face_count, 0);
        assert_eq!(model.report.missing_exact_query_face_count, 2);
        assert_eq!(model.report.missing_derivative_query_face_count, 2);
        assert_eq!(model.report.missing_curvature_query_face_count, 2);
        assert_eq!(model.report.evaluator_sample_count, 0);
        assert!(model.face_frames.iter().any(|frame| frame.evaluator_backed
            && frame.origin_m == [0.25, 0.25, 0.75]
            && frame.unit_normal == [0.0, 0.0, 1.0]));
    }

    #[test]
    fn no_op_provider_keeps_imported_evaluator_metadata_sample_based() {
        let topology = cube_topology();
        let geometry = geometry_with_face_evaluator();
        let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

        let model = build_cad_evaluation_model_with_provider(
            &cad_topology,
            &topology,
            &NoopCadFaceEvaluatorProvider,
        )
        .expect("evaluation model");

        assert_eq!(model.source, CadEvaluationSource::ImportedEvaluatorSamples);
        assert_eq!(model.report.live_query_face_count, 0);
        assert_eq!(model.report.exact_query_face_count, 0);
        assert_eq!(model.report.missing_exact_query_face_count, 2);
        assert_eq!(model.report.missing_derivative_query_face_count, 2);
        assert_eq!(model.report.missing_curvature_query_face_count, 2);
        assert!(model
            .face_frames
            .iter()
            .all(|frame| !frame.live_query_backed));
    }

    #[test]
    fn live_evaluator_provider_samples_drive_parametric_cad_frames() {
        #[derive(Debug)]
        struct LiveProvider;

        impl CadFaceEvaluatorProvider for LiveProvider {
            fn evaluate_face(
                &self,
                request: &CadFaceEvaluationRequest<'_>,
            ) -> Vec<CadFaceEvaluationSample> {
                assert_eq!(request.imported_face_id, Some(1));
                assert_eq!(request.evaluator_id, Some("cad_face_1"));
                assert!(request.supports_projection);
                assert!(request.supports_normal);
                assert_eq!(request.reference_point_m, [0.25, 0.25, 0.75]);
                assert_eq!(request.reference_unit_normal, [0.0, 0.0, 1.0]);
                vec![CadFaceEvaluationSample {
                    source: CadFaceEvaluationSampleSource::BackendQuery,
                    point_m: [0.5, 0.5, 1.01],
                    uv: Some([0.5, 0.5]),
                    projected_point_m: Some([0.5, 0.5, 1.0]),
                    unit_normal: Some([0.0, 0.0, 1.0]),
                    projection_error_m: Some(0.01),
                }]
            }
        }

        let topology = cube_topology();
        let geometry = geometry_with_face_evaluator();
        let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

        let model =
            build_cad_evaluation_model_with_provider(&cad_topology, &topology, &LiveProvider)
                .expect("evaluation model");
        let report = summarize_cad_evaluation(&model, &topology).expect("summary");
        let frame = model
            .face_frames
            .iter()
            .find(|frame| frame.live_query_backed)
            .expect("live-query frame");

        assert_eq!(model.source, CadEvaluationSource::ParametricCad);
        assert_eq!(model.report.live_query_face_count, 2);
        assert_eq!(model.report.exact_query_face_count, 2);
        assert_eq!(model.report.projection_supported_face_count, 2);
        assert_eq!(model.report.normal_supported_face_count, 2);
        assert_eq!(report.projection_supported_face_count, 2);
        assert_eq!(report.normal_supported_face_count, 2);
        assert_eq!(model.report.missing_exact_query_face_count, 0);
        assert_eq!(model.report.missing_derivative_query_face_count, 2);
        assert_eq!(model.report.missing_curvature_query_face_count, 2);
        assert_eq!(model.report.evaluator_sample_count, 2);
        assert_eq!(report.live_query_face_count, 2);
        assert_eq!(report.missing_exact_query_face_count, 0);
        assert_eq!(report.source, CadEvaluationSource::ParametricCad);
        assert_eq!(frame.origin_m, [0.5, 0.5, 1.0]);
        assert_eq!(frame.evaluator_samples.len(), 1);
        assert_eq!(frame.evaluator_max_projection_error_m, 0.01);
    }

    #[test]
    fn exact_backend_query_samples_drive_parametric_cad_frames() {
        let topology = cube_topology();
        let mut geometry = geometry_with_face_evaluator();
        geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples =
            vec![CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [0.5, 0.5, 1.0],
                uv: Some([0.5, 0.5]),
                projected_point_m: Some([0.5, 0.5, 1.0]),
                unit_normal: Some([0.0, 0.0, 1.0]),
                projection_error_m: Some(2.0e-6),
            }];
        let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

        let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");
        let report = summarize_cad_evaluation(&model, &topology).expect("summary");

        assert_eq!(model.source, CadEvaluationSource::ImportedEvaluatorSamples);
        assert_eq!(model.report.evaluator_sample_count, 2);
        assert_eq!(model.report.exact_query_face_count, 2);
        assert_eq!(model.report.max_projection_error_m, 2.0e-6);
        assert_eq!(report.max_projection_error_m, 2.0e-6);
        let frame = model
            .face_frames
            .iter()
            .find(|frame| frame.exact_query_backed)
            .expect("one frame should be exact-query backed");
        assert_eq!(frame.origin_m, [0.5, 0.5, 1.0]);
        assert_eq!(frame.unit_normal, [0.0, 0.0, 1.0]);
        assert_eq!(frame.evaluator_max_projection_error_m, 2.0e-6);
        assert_eq!(frame.evaluator_samples.len(), 1);
        assert_eq!(frame.evaluator_samples[0].uv, Some([0.5, 0.5]));
        assert!(frame.uv_bounds.is_some());
        assert!(face_uv_contains(frame, [0.5, 0.5]));
    }

    #[test]
    fn exact_backend_samples_define_uv_domain_when_sufficient() {
        let topology = cube_topology();
        let mut geometry = geometry_with_face_evaluator();
        geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples = vec![
            CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [0.0, 0.0, 1.0],
                uv: Some([2.0, 4.0]),
                projected_point_m: Some([0.0, 0.0, 1.0]),
                unit_normal: Some([0.0, 0.0, 1.0]),
                projection_error_m: Some(0.0),
            },
            CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [1.0, 0.0, 1.0],
                uv: Some([5.0, 4.0]),
                projected_point_m: Some([1.0, 0.0, 1.0]),
                unit_normal: Some([0.0, 0.0, 1.0]),
                projection_error_m: Some(0.0),
            },
            CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [1.0, 1.0, 1.0],
                uv: Some([5.0, 7.0]),
                projected_point_m: Some([1.0, 1.0, 1.0]),
                unit_normal: Some([0.0, 0.0, 1.0]),
                projection_error_m: Some(0.0),
            },
        ];
        let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

        let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");
        let report = summarize_cad_evaluation(&model, &topology).expect("summary");
        let frame = model
            .face_frames
            .iter()
            .find(|frame| frame.evaluator_samples.len() == 3)
            .expect("sample-backed frame");

        assert_eq!(frame.uv_bounds, Some([[2.0, 4.0], [5.0, 7.0]]));
        assert_eq!(frame.uv_bounds_sample_count, 3);
        assert_eq!(frame.uv_domain_source.as_deref(), Some("exact_samples"));
        assert!(face_uv_contains(frame, [3.0, 6.0]));
        assert!(!face_uv_contains(frame, [6.0, 6.0]));
        assert!(model.report.uv_domain_face_count > 0);
        assert!(report.uv_domain_face_count > 0);
        assert!(report.uv_projection_out_of_bounds_count > 0);
    }

    #[test]
    fn all_backend_query_samples_remain_imported_sample_source() {
        assert_eq!(
            evaluation_source(2, 2, 0, 2),
            CadEvaluationSource::ImportedEvaluatorSamples
        );
    }

    #[test]
    fn cad_face_frames_normalize_backend_normals() {
        let topology = cube_topology();
        let mut geometry = geometry_with_face_evaluator();
        geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples =
            vec![CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [0.5, 0.5, 1.0],
                uv: Some([0.5, 0.5]),
                projected_point_m: Some([0.5, 0.5, 1.0]),
                unit_normal: Some([0.0, 0.0, 2.0]),
                projection_error_m: Some(0.0),
            }];
        let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

        let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");
        let frame = model
            .face_frames
            .iter()
            .find(|frame| frame.exact_query_backed)
            .expect("one frame should be exact-query backed");
        let projection = project_to_face(frame, [0.5, 0.5, 1.25]);

        assert_eq!(frame.unit_normal, [0.0, 0.0, 1.0]);
        assert!((norm(frame.v_axis) - 1.0).abs() <= 1.0e-12);
        assert!((projection.distance_m - 0.25).abs() <= 1.0e-12);
    }

    #[test]
    fn cad_face_frames_orient_backend_normals_to_source_face() {
        let topology = cube_topology();
        let mut geometry = geometry_with_face_evaluator();
        geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples =
            vec![CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [0.5, 0.5, 1.0],
                uv: Some([0.5, 0.5]),
                projected_point_m: Some([0.5, 0.5, 1.0]),
                unit_normal: Some([0.0, 0.0, -1.0]),
                projection_error_m: Some(0.0),
            }];
        let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

        let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");
        let frame = model
            .face_frames
            .iter()
            .find(|frame| frame.exact_query_backed)
            .expect("one frame should be exact-query backed");

        assert_eq!(frame.unit_normal, [0.0, 0.0, 1.0]);
        assert_eq!(frame.v_axis, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn exact_backend_query_uses_projected_point_for_frame_origin() {
        let topology = cube_topology();
        let mut geometry = geometry_with_face_evaluator();
        geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples =
            vec![CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [0.5, 0.5, 1.02],
                uv: Some([0.5, 0.5]),
                projected_point_m: Some([0.5, 0.5, 1.0]),
                unit_normal: Some([0.0, 0.0, 1.0]),
                projection_error_m: Some(0.02),
            }];
        let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

        let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");
        let frame = model
            .face_frames
            .iter()
            .find(|frame| frame.exact_query_backed)
            .expect("one frame should be exact-query backed");

        assert_eq!(frame.origin_m, [0.5, 0.5, 1.0]);
        assert_eq!(frame.evaluator_samples[0].point_m, [0.5, 0.5, 1.02]);
        assert_eq!(
            frame.evaluator_samples[0].projected_point_m,
            Some([0.5, 0.5, 1.0])
        );
        assert_eq!(frame.evaluator_max_projection_error_m, 0.02);
    }

    #[test]
    fn exact_backend_query_samples_drive_matching_projection() {
        let topology = cube_topology();
        let mut geometry = geometry_with_face_evaluator();
        geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples =
            vec![CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [0.5, 0.5, 1.02],
                uv: Some([0.5, 0.5]),
                projected_point_m: Some([0.5, 0.5, 1.0]),
                unit_normal: Some([0.0, 0.0, 2.0]),
                projection_error_m: Some(0.02),
            }];
        let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

        let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");
        let frame = model
            .face_frames
            .iter()
            .find(|frame| frame.exact_query_backed)
            .expect("one frame should be exact-query backed");
        let query_projection = project_to_face(frame, [0.5, 0.5, 1.02]);
        let projected_point_projection = project_to_face(frame, [0.5, 0.5, 1.0]);
        let fallback_projection = project_to_face(frame, [0.25, 0.25, 1.02]);

        assert_eq!(query_projection.point_m, [0.5, 0.5, 1.0]);
        assert_eq!(query_projection.uv, [0.5, 0.5]);
        assert!((query_projection.distance_m - 0.02).abs() <= 1.0e-12);
        assert_eq!(query_projection.unit_normal, [0.0, 0.0, 1.0]);
        assert_eq!(projected_point_projection.point_m, [0.5, 0.5, 1.0]);
        assert_eq!(projected_point_projection.uv, [0.5, 0.5]);
        assert_ne!(fallback_projection.uv, [0.5, 0.5]);
    }

    #[test]
    fn merged_cad_face_samples_are_filtered_to_source_triangle_frames() {
        let topology = cube_topology();
        let mut geometry = geometry_with_face_evaluator();
        geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples = vec![
            CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [0.75, 0.25, 1.0],
                uv: Some([0.75, 0.25]),
                projected_point_m: Some([0.75, 0.25, 1.0]),
                unit_normal: Some([0.0, 0.0, 1.0]),
                projection_error_m: Some(0.0),
            },
            CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [0.25, 0.75, 1.0],
                uv: Some([0.25, 0.75]),
                projected_point_m: Some([0.25, 0.75, 1.0]),
                unit_normal: Some([0.0, 0.0, 1.0]),
                projection_error_m: Some(0.0),
            },
        ];
        let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

        let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");
        let lower_triangle_frame = model
            .face_frames
            .iter()
            .find(|frame| frame.source_face_id == 2)
            .expect("lower source triangle frame");
        let upper_triangle_frame = model
            .face_frames
            .iter()
            .find(|frame| frame.source_face_id == 3)
            .expect("upper source triangle frame");

        assert_eq!(model.report.exact_query_face_count, 2);
        assert_eq!(model.report.evaluator_sample_count, 2);
        assert_eq!(model.report.evaluator_rejected_sample_count, 2);
        assert_eq!(lower_triangle_frame.evaluator_samples.len(), 1);
        assert_eq!(upper_triangle_frame.evaluator_samples.len(), 1);
        assert_eq!(lower_triangle_frame.origin_m, [0.75, 0.25, 1.0]);
        assert_eq!(upper_triangle_frame.origin_m, [0.25, 0.75, 1.0]);
    }

    #[test]
    fn exact_backend_query_prefers_lowest_projection_error_sample() {
        let topology = cube_topology();
        let mut geometry = geometry_with_face_evaluator();
        geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples = vec![
            CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [0.7, 0.5, 1.01],
                uv: Some([0.7, 0.5]),
                projected_point_m: Some([0.7, 0.5, 1.0]),
                unit_normal: Some([0.0, 0.0, 1.0]),
                projection_error_m: Some(0.01),
            },
            CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [0.8, 0.5, 1.001],
                uv: Some([0.8, 0.5]),
                projected_point_m: Some([0.8, 0.5, 1.0]),
                unit_normal: Some([0.0, 0.0, 1.0]),
                projection_error_m: Some(0.001),
            },
        ];
        let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

        let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");
        let frame = model
            .face_frames
            .iter()
            .find(|frame| frame.exact_query_backed)
            .expect("one frame should be exact-query backed");

        assert_eq!(frame.origin_m, [0.8, 0.5, 1.0]);
        assert_eq!(frame.unit_normal, [0.0, 0.0, 1.0]);
        assert_eq!(frame.evaluator_max_projection_error_m, 0.01);
        assert_eq!(frame.evaluator_samples.len(), 2);
    }

    #[test]
    fn exact_backend_query_prefers_measured_projection_error_over_unknown() {
        let topology = cube_topology();
        let mut geometry = geometry_with_face_evaluator();
        geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples = vec![
            CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [0.6, 0.5, 1.0],
                uv: Some([0.6, 0.5]),
                projected_point_m: Some([0.6, 0.5, 1.0]),
                unit_normal: Some([0.0, 0.0, 1.0]),
                projection_error_m: None,
            },
            CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [0.8, 0.5, 1.002],
                uv: Some([0.8, 0.5]),
                projected_point_m: Some([0.8, 0.5, 1.0]),
                unit_normal: Some([0.0, 0.0, 1.0]),
                projection_error_m: Some(0.002),
            },
        ];
        let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

        let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");
        let frame = model
            .face_frames
            .iter()
            .find(|frame| frame.exact_query_backed)
            .expect("one frame should be exact-query backed");

        assert_eq!(frame.origin_m, [0.8, 0.5, 1.0]);
        assert_eq!(frame.evaluator_max_projection_error_m, 0.002);
        assert_eq!(frame.evaluator_samples.len(), 2);
    }

    #[test]
    fn evaluator_sample_report_counts_invalid_and_over_budget_samples() {
        let topology = cube_topology();
        let mut geometry = geometry_with_face_evaluator();
        geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples = (0..10)
            .map(|index| CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [index as f64 * 0.01, index as f64 * 0.01, 1.0],
                uv: Some([index as f64 * 0.01, index as f64 * 0.01]),
                projected_point_m: Some([index as f64 * 0.01, index as f64 * 0.01, 1.0]),
                unit_normal: Some([0.0, 0.0, 1.0]),
                projection_error_m: Some(0.0),
            })
            .chain(std::iter::once(CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [f64::NAN, 0.5, 1.0],
                uv: Some([0.5, 0.5]),
                projected_point_m: Some([0.5, 0.5, 1.0]),
                unit_normal: Some([0.0, 0.0, 1.0]),
                projection_error_m: Some(0.0),
            }))
            .collect();
        let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

        let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");
        let frame = model
            .face_frames
            .iter()
            .find(|frame| frame.evaluator_samples.len() == 8)
            .expect("sample-backed frame should retain bounded valid samples");

        assert_eq!(frame.evaluator_sample_count, 8);
        assert_eq!(frame.evaluator_rejected_sample_count, 3);
        assert_eq!(model.report.evaluator_sample_count, 16);
        assert_eq!(model.report.evaluator_rejected_sample_count, 6);
    }

    #[test]
    fn backend_samples_expose_derivative_and_curvature_estimates() {
        let topology = cube_topology();
        let mut geometry = geometry_with_face_evaluator();
        geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples = vec![
            CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [0.0, 0.0, 1.0],
                uv: Some([0.0, 0.0]),
                projected_point_m: Some([0.0, 0.0, 1.0]),
                unit_normal: Some([0.0, 0.0, 1.0]),
                projection_error_m: Some(0.0),
            },
            CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [1.0, 0.0, 1.0],
                uv: Some([1.0, 0.0]),
                projected_point_m: Some([1.0, 0.0, 1.0]),
                unit_normal: Some([0.0, 0.05, 0.998749217771909]),
                projection_error_m: Some(0.0),
            },
            CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [1.0, 1.0, 1.0],
                uv: Some([1.0, 1.0]),
                projected_point_m: Some([1.0, 1.0, 1.0]),
                unit_normal: Some([0.04, 0.0, 0.9991996797437437]),
                projection_error_m: Some(0.0),
            },
            CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [0.0, 1.0, 1.0],
                uv: Some([0.0, 1.0]),
                projected_point_m: Some([0.0, 1.0, 1.0]),
                unit_normal: Some([0.0, 0.04, 0.9991996797437437]),
                projection_error_m: Some(0.0),
            },
        ];
        let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

        let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");
        let report = summarize_cad_evaluation(&model, &topology).expect("summary");
        let frame = model
            .face_frames
            .iter()
            .find(|frame| frame.evaluator_samples.len() == 3)
            .expect("sample-backed frame");

        assert_eq!(model.report.derivative_query_count, 2);
        assert_eq!(model.report.curvature_query_count, 2);
        assert_eq!(report.derivative_query_count, 2);
        assert_eq!(report.curvature_query_count, 2);
        assert_eq!(report.missing_derivative_query_face_count, 0);
        assert_eq!(report.missing_curvature_query_face_count, 0);
        assert_eq!(frame.u_derivative_m_per_uv, Some([1.0, 0.0, 0.0]));
        assert_eq!(frame.v_derivative_m_per_uv, Some([0.0, 1.0, 0.0]));
        assert!(frame.max_curvature_estimate_1_per_m.unwrap_or(0.0) > 0.0);
        assert_eq!(
            report.max_curvature_estimate_1_per_m,
            model.report.max_curvature_estimate_1_per_m
        );
    }

    #[test]
    fn cad_derivative_estimates_use_projected_backend_points() {
        let topology = cube_topology();
        let mut geometry = geometry_with_face_evaluator();
        geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples = vec![
            CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [0.0, 0.0, 1.0],
                uv: Some([0.0, 0.0]),
                projected_point_m: Some([0.0, 0.0, 1.0]),
                unit_normal: Some([0.0, 0.0, 1.0]),
                projection_error_m: Some(0.0),
            },
            CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [1.0, 0.0, 1.4],
                uv: Some([1.0, 0.0]),
                projected_point_m: Some([1.0, 0.0, 1.0]),
                unit_normal: Some([0.0, 0.0, 1.0]),
                projection_error_m: Some(0.4),
            },
            CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [1.0, 1.0, 0.6],
                uv: Some([1.0, 1.0]),
                projected_point_m: Some([1.0, 1.0, 1.0]),
                unit_normal: Some([0.0, 0.0, 1.0]),
                projection_error_m: Some(0.4),
            },
        ];
        let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

        let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");
        let frame = model
            .face_frames
            .iter()
            .find(|frame| frame.evaluator_samples.len() == 3)
            .expect("sample-backed frame");

        assert_eq!(frame.u_derivative_m_per_uv, Some([1.0, 0.0, 0.0]));
        assert_eq!(frame.v_derivative_m_per_uv, Some([0.0, 1.0, 0.0]));
    }

    #[test]
    fn cad_curvature_estimates_orient_backend_normals_to_face_frame() {
        let topology = cube_topology();
        let mut geometry = geometry_with_face_evaluator();
        geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples = vec![
            CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [0.0, 0.0, 1.0],
                uv: Some([0.0, 0.0]),
                projected_point_m: Some([0.0, 0.0, 1.0]),
                unit_normal: Some([0.0, 0.0, 1.0]),
                projection_error_m: Some(0.0),
            },
            CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [1.0, 0.0, 1.0],
                uv: Some([1.0, 0.0]),
                projected_point_m: Some([1.0, 0.0, 1.0]),
                unit_normal: Some([0.0, 0.0, -1.0]),
                projection_error_m: Some(0.0),
            },
        ];
        let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

        let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");
        let frame = model
            .face_frames
            .iter()
            .find(|frame| frame.evaluator_samples.len() == 2)
            .expect("sample-backed frame");

        assert_eq!(frame.unit_normal, [0.0, 0.0, 1.0]);
        assert_eq!(frame.max_curvature_estimate_1_per_m, Some(0.0));
    }

    fn cube_topology() -> SourceTopologyModel {
        crate::topology::source_mesh::source_topology_from_boundary_input(
            &crate::topology::source_mesh::SourceTopologyInput {
                mesh_id: "cube_surface".to_string(),
                source_geometry_id: "geo_eval_cube".to_string(),
                source_geometry_revision: 1,
                source_geometry_sha256: None,
                vertices: vec![
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                ],
                triangles: vec![
                    crate::topology::source_mesh::SourceTopologyTriangle {
                        triangle_id: 0,
                        node_ids: [0, 2, 1],
                        region_ids: Vec::new(),
                    },
                    crate::topology::source_mesh::SourceTopologyTriangle {
                        triangle_id: 1,
                        node_ids: [0, 3, 2],
                        region_ids: Vec::new(),
                    },
                    crate::topology::source_mesh::SourceTopologyTriangle {
                        triangle_id: 2,
                        node_ids: [4, 5, 6],
                        region_ids: Vec::new(),
                    },
                    crate::topology::source_mesh::SourceTopologyTriangle {
                        triangle_id: 3,
                        node_ids: [4, 6, 7],
                        region_ids: Vec::new(),
                    },
                ],
                bounds_min_m: [0.0, 0.0, 0.0],
                bounds_max_m: [1.0, 1.0, 1.0],
                region_ids: Vec::new(),
            },
        )
    }

    fn geometry_for_topology() -> runmat_geometry_core::GeometryAsset {
        runmat_geometry_core::GeometryAsset {
            geometry_id: "geo_eval_cube".to_string(),
            source: runmat_geometry_core::GeometrySource {
                path: "/fixtures/eval.step".to_string(),
                sha256: "eval".to_string(),
                importer_version: "test".to_string(),
            },
            source_geometry: runmat_geometry_core::SourceGeometry {
                kind: runmat_geometry_core::SourceGeometryKind::Cad,
                assembly: None,
                material_evidence: Vec::new(),
                cad_evaluators: Vec::new(),
            },
            tessellation_profile: runmat_geometry_core::TessellationProfile::default(),
            units: runmat_geometry_core::UnitSystem::Meter,
            revision: 1,
            meshes: Vec::new(),
            surface_meshes: Vec::new(),
            regions: Vec::new(),
            region_entity_mappings: Vec::new(),
            diagnostics: Vec::new(),
        }
    }

    fn geometry_with_face_evaluator() -> runmat_geometry_core::GeometryAsset {
        let mut geometry = geometry_for_topology();
        geometry.regions = vec![Region {
            region_id: "face_000001".to_string(),
            name: "face".to_string(),
            tag: Some("cad_face".to_string()),
            cad_ownership: Some(CadRegionOwnership {
                face_id: Some(1),
                label: Some(CadLabelRef {
                    label_entry: "0:1:1".to_string(),
                    name: "face".to_string(),
                    kind: CadSemanticKind::Face,
                }),
                owner_path: Vec::new(),
                layers: Vec::new(),
                color: None,
                material: None,
            }),
        }];
        geometry.region_entity_mappings = vec![RegionEntityMapping::new(
            "face_000001",
            "mesh_1",
            EntityKind::Face,
            vec![EntityIdRange::new(2, 2)],
        )];
        geometry.source_geometry.cad_evaluators = vec![CadEvaluatorSet {
            evaluator_id: "cad_evaluator_test".to_string(),
            backend: "test".to_string(),
            format_name: "step".to_string(),
            requires_source_geometry: true,
            faces: vec![CadFaceEvaluator {
                evaluator_id: "cad_face_1".to_string(),
                imported_face_id: 1,
                name: "face".to_string(),
                supports_point_evaluation: true,
                supports_projection: true,
                supports_normal: true,
                supports_derivatives: true,
                supports_curvature: true,
                reference_point_m: Some([0.25, 0.25, 0.75]),
                reference_unit_normal: Some([0.0, 0.0, 1.0]),
                evaluation_samples: Vec::new(),
            }],
            curves: Vec::new(),
        }];
        geometry
    }
}
