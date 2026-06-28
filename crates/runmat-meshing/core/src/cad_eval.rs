use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{
    cad_topology::{CadFace, CadTopologyModel},
    predicate::{dot, norm, scale, sub, triangle_centroid, Point3, Triangle3},
    source_topology::SourceTopologyModel,
};
use runmat_geometry_core::{CadFaceEvaluationSample, CadFaceEvaluationSampleSource};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CadEvaluationSource {
    ParametricCad,
    ImportedEvaluatorSamples,
    PlanarFacetApproximation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadFaceEvaluationFrame {
    pub face_id: String,
    pub source_face_id: u32,
    pub origin_m: Point3,
    pub u_axis: Point3,
    pub v_axis: Point3,
    pub unit_normal: Point3,
    pub area_m2: f64,
    #[serde(default)]
    pub evaluator_backed: bool,
    #[serde(default)]
    pub exact_query_backed: bool,
    #[serde(default)]
    pub live_query_backed: bool,
    #[serde(default)]
    pub evaluator_sample_count: usize,
    #[serde(default)]
    pub evaluator_max_projection_error_m: f64,
    #[serde(default)]
    pub evaluator_samples: Vec<CadFaceEvaluationSample>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub u_derivative_m_per_uv: Option<Point3>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub v_derivative_m_per_uv: Option<Point3>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_curvature_estimate_1_per_m: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uv_bounds: Option<[[f64; 2]; 2]>,
    #[serde(default)]
    pub uv_bounds_sample_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uv_domain_source: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CadFaceProjection {
    pub point_m: Point3,
    pub uv: [f64; 2],
    pub distance_m: f64,
    pub unit_normal: Point3,
    #[serde(default)]
    pub uv_in_bounds: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadEvaluationReport {
    pub source: CadEvaluationSource,
    pub face_frame_count: usize,
    pub evaluator_face_count: usize,
    #[serde(default)]
    pub live_query_face_count: usize,
    #[serde(default)]
    pub exact_query_face_count: usize,
    #[serde(default)]
    pub evaluator_sample_count: usize,
    pub normal_query_count: usize,
    pub projection_query_count: usize,
    #[serde(default)]
    pub derivative_query_count: usize,
    #[serde(default)]
    pub curvature_query_count: usize,
    pub max_projection_error_m: f64,
    pub max_normal_deviation: f64,
    #[serde(default)]
    pub uv_domain_face_count: usize,
    #[serde(default)]
    pub uv_projection_out_of_bounds_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_curvature_estimate_1_per_m: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadEvaluationModel {
    pub source_geometry_id: String,
    pub source_geometry_revision: u32,
    pub source: CadEvaluationSource,
    pub face_frames: Vec<CadFaceEvaluationFrame>,
    pub report: CadEvaluationReport,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CadEvaluationError {
    EmptyFaces,
    MissingSourceFace { source_face_id: u32 },
    MissingSourceVertex { vertex_id: u32 },
    DegenerateFace { source_face_id: u32 },
}

impl std::fmt::Display for CadEvaluationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyFaces => write!(formatter, "CAD evaluation model has no faces"),
            Self::MissingSourceFace { source_face_id } => {
                write!(formatter, "source face {source_face_id} is missing")
            }
            Self::MissingSourceVertex { vertex_id } => {
                write!(formatter, "source vertex {vertex_id} is missing")
            }
            Self::DegenerateFace { source_face_id } => {
                write!(formatter, "source face {source_face_id} is degenerate")
            }
        }
    }
}

impl std::error::Error for CadEvaluationError {}

#[derive(Debug, Clone, PartialEq)]
pub struct CadFaceEvaluationRequest<'a> {
    pub face_id: &'a str,
    pub source_face_id: u32,
    pub imported_face_id: Option<u64>,
    pub evaluator_id: Option<&'a str>,
    pub supports_point_evaluation: bool,
    pub supports_projection: bool,
    pub supports_normal: bool,
    pub supports_derivatives: bool,
    pub supports_curvature: bool,
    pub reference_point_m: Point3,
    pub reference_unit_normal: Point3,
}

pub trait CadFaceEvaluatorProvider {
    fn evaluate_face(&self, request: &CadFaceEvaluationRequest<'_>)
        -> Vec<CadFaceEvaluationSample>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NoopCadFaceEvaluatorProvider;

impl CadFaceEvaluatorProvider for NoopCadFaceEvaluatorProvider {
    fn evaluate_face(
        &self,
        _request: &CadFaceEvaluationRequest<'_>,
    ) -> Vec<CadFaceEvaluationSample> {
        Vec::new()
    }
}

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
    let mut frames = Vec::<CadFaceEvaluationFrame>::with_capacity(cad_topology.faces.len());
    for face in &cad_topology.faces {
        let source_face_id = face
            .source_face_ids
            .first()
            .copied()
            .ok_or(CadEvaluationError::MissingSourceFace { source_face_id: 0 })?;
        let source_face = source_faces
            .get(&source_face_id)
            .ok_or(CadEvaluationError::MissingSourceFace { source_face_id })?;
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
            source_face_id,
            fallback_reference_point_m,
            fallback_unit_normal,
        );
        let live_query_backed = !live_samples.is_empty();
        let evaluator_samples = merged_bounded_evaluator_samples(face, live_samples);
        let exact_sample = exact_backend_sample(&evaluator_samples);
        let evaluator_max_projection_error_m = evaluator_max_projection_error(&evaluator_samples);
        let frame = face_frame(
            face.entity_id.id.clone(),
            source_face_id,
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
                || !evaluator_samples.is_empty(),
            exact_sample.is_some(),
            live_query_backed,
            evaluator_samples.len(),
            evaluator_max_projection_error_m,
            evaluator_samples,
        )?;
        frames.push(frame);
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
    let evaluator_sample_count = frames
        .iter()
        .map(|frame| frame.evaluator_sample_count)
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
        evaluator_sample_count,
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
        evaluator_sample_count: model.report.evaluator_sample_count,
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
    let v_axis = crate::predicate::cross(unit_normal, u_axis);
    let v_length = norm(v_axis);
    if v_length <= f64::EPSILON {
        return Err(CadEvaluationError::DegenerateFace { source_face_id });
    }
    let (u_derivative_m_per_uv, v_derivative_m_per_uv) =
        estimate_uv_derivatives(&evaluator_samples);
    let max_curvature_estimate_1_per_m = estimate_max_curvature(&evaluator_samples, unit_normal);
    let (uv_bounds, uv_bounds_sample_count, uv_domain_source) = cad_uv_domain_summary(
        &evaluator_samples,
        points,
        evaluator_reference_point_m,
        u_axis,
        scale(v_axis, 1.0 / v_length),
    );
    Ok(CadFaceEvaluationFrame {
        face_id,
        source_face_id,
        origin_m: evaluator_reference_point_m.unwrap_or_else(|| triangle_centroid(points)),
        u_axis,
        v_axis: scale(v_axis, 1.0 / v_length),
        unit_normal,
        area_m2,
        evaluator_backed,
        exact_query_backed,
        live_query_backed,
        evaluator_sample_count,
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

fn cad_uv_domain_summary(
    evaluator_samples: &[CadFaceEvaluationSample],
    source_points: Triangle3,
    evaluator_reference_point_m: Option<Point3>,
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

    let origin = evaluator_reference_point_m.unwrap_or_else(|| triangle_centroid(source_points));
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
    let source_normal =
        crate::predicate::cross(sub(points[1], points[0]), sub(points[2], points[0]));
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

fn exact_backend_sample(samples: &[CadFaceEvaluationSample]) -> Option<&CadFaceEvaluationSample> {
    samples
        .iter()
        .filter(|sample| exact_backend_sample_is_valid(sample))
        .min_by(|left, right| compare_exact_backend_samples(left, right))
}

fn exact_backend_sample_is_valid(sample: &CadFaceEvaluationSample) -> bool {
    sample.source == CadFaceEvaluationSampleSource::BackendQuery
        && finite_point(sample.point_m)
        && sample
            .unit_normal
            .is_some_and(|normal| finite_point(normal) && norm(normal) > 0.0)
        && sample
            .projection_error_m
            .is_none_or(|error| error.is_finite() && error >= 0.0)
}

fn compare_exact_backend_samples(
    left: &CadFaceEvaluationSample,
    right: &CadFaceEvaluationSample,
) -> std::cmp::Ordering {
    sample_projection_error(left)
        .total_cmp(&sample_projection_error(right))
        .then_with(|| compare_points_lexicographically(left.point_m, right.point_m))
}

fn sample_projection_error(sample: &CadFaceEvaluationSample) -> f64 {
    sample.projection_error_m.unwrap_or(f64::INFINITY)
}

fn exact_backend_sample_point(sample: &CadFaceEvaluationSample) -> Point3 {
    sample
        .projected_point_m
        .filter(|point| finite_point(*point))
        .unwrap_or(sample.point_m)
}

fn normalized_sample_normal(unit_normal: Point3) -> Option<Point3> {
    let normal_length = norm(unit_normal);
    if normal_length.is_finite() && normal_length > 0.0 {
        Some(scale(unit_normal, 1.0 / normal_length))
    } else {
        None
    }
}

fn orient_sample_normal(unit_normal: Point3, frame_unit_normal: Point3) -> Point3 {
    if dot(unit_normal, frame_unit_normal) < 0.0 {
        scale(unit_normal, -1.0)
    } else {
        unit_normal
    }
}

fn evaluator_max_projection_error(samples: &[CadFaceEvaluationSample]) -> f64 {
    samples
        .iter()
        .filter_map(|sample| sample.projection_error_m)
        .filter(|error| error.is_finite() && *error >= 0.0)
        .fold(0.0_f64, f64::max)
}

fn live_evaluator_samples(
    evaluator_provider: &dyn CadFaceEvaluatorProvider,
    face: &CadFace,
    source_face_id: u32,
    reference_point_m: Point3,
    reference_unit_normal: Point3,
) -> Vec<CadFaceEvaluationSample> {
    if face.evaluator_id.is_none()
        || !(face.evaluator_supports_point_evaluation
            || face.evaluator_supports_projection
            || face.evaluator_supports_normal
            || face.evaluator_supports_derivatives
            || face.evaluator_supports_curvature)
    {
        return Vec::new();
    }
    let request = CadFaceEvaluationRequest {
        face_id: &face.entity_id.id,
        source_face_id,
        imported_face_id: face.imported_face_id,
        evaluator_id: face.evaluator_id.as_deref(),
        supports_point_evaluation: face.evaluator_supports_point_evaluation,
        supports_projection: face.evaluator_supports_projection,
        supports_normal: face.evaluator_supports_normal,
        supports_derivatives: face.evaluator_supports_derivatives,
        supports_curvature: face.evaluator_supports_curvature,
        reference_point_m,
        reference_unit_normal,
    };
    evaluator_provider
        .evaluate_face(&request)
        .into_iter()
        .filter(bounded_sample_is_valid)
        .take(8)
        .collect()
}

fn merged_bounded_evaluator_samples(
    face: &CadFace,
    live_samples: Vec<CadFaceEvaluationSample>,
) -> Vec<CadFaceEvaluationSample> {
    live_samples
        .into_iter()
        .chain(face.evaluator_samples.iter().cloned())
        .filter(|sample| bounded_sample_is_valid(sample))
        .take(8)
        .collect()
}

fn bounded_sample_is_valid(sample: &CadFaceEvaluationSample) -> bool {
    finite_point(sample.point_m)
        && sample
            .projected_point_m
            .is_none_or(|point| finite_point(point))
        && sample
            .uv
            .is_none_or(|uv| uv.iter().all(|value| value.is_finite()))
        && sample
            .unit_normal
            .is_none_or(|normal| finite_point(normal) && norm(normal) > 0.0)
        && sample
            .projection_error_m
            .is_none_or(|error| error.is_finite() && error >= 0.0)
}

fn estimate_uv_derivatives(
    samples: &[CadFaceEvaluationSample],
) -> (Option<Point3>, Option<Point3>) {
    let samples = samples
        .iter()
        .filter_map(|sample| {
            let uv = sample.uv?;
            let point_m = exact_backend_sample_point(sample);
            (finite_point(point_m) && uv.iter().all(|value| value.is_finite()))
                .then_some((uv, point_m))
        })
        .collect::<Vec<_>>();
    for base_index in 0..samples.len() {
        for u_index in 0..samples.len() {
            for v_index in 0..samples.len() {
                if base_index == u_index || base_index == v_index || u_index == v_index {
                    continue;
                }
                let (base_uv, base_point) = samples[base_index];
                let (u_uv, u_point) = samples[u_index];
                let (v_uv, v_point) = samples[v_index];
                let du = [u_uv[0] - base_uv[0], u_uv[1] - base_uv[1]];
                let dv = [v_uv[0] - base_uv[0], v_uv[1] - base_uv[1]];
                let determinant = du[0] * dv[1] - du[1] * dv[0];
                if !determinant.is_finite() || determinant.abs() <= 1.0e-12 {
                    continue;
                }
                let dp_u = sub(u_point, base_point);
                let dp_v = sub(v_point, base_point);
                let inv_det = 1.0 / determinant;
                let derivative_u = [
                    (dp_u[0] * dv[1] - dp_v[0] * du[1]) * inv_det,
                    (dp_u[1] * dv[1] - dp_v[1] * du[1]) * inv_det,
                    (dp_u[2] * dv[1] - dp_v[2] * du[1]) * inv_det,
                ];
                let derivative_v = [
                    (dp_v[0] * du[0] - dp_u[0] * dv[0]) * inv_det,
                    (dp_v[1] * du[0] - dp_u[1] * dv[0]) * inv_det,
                    (dp_v[2] * du[0] - dp_u[2] * dv[0]) * inv_det,
                ];
                if finite_point(derivative_u) && finite_point(derivative_v) {
                    return (Some(derivative_u), Some(derivative_v));
                }
            }
        }
    }
    (None, None)
}

fn estimate_max_curvature(
    samples: &[CadFaceEvaluationSample],
    frame_unit_normal: Point3,
) -> Option<f64> {
    let samples = samples
        .iter()
        .filter_map(|sample| {
            let normal = sample.unit_normal?;
            let normal_length = norm(normal);
            let point_m = exact_backend_sample_point(sample);
            if finite_point(point_m) && finite_point(normal) && normal_length > 0.0 {
                let mut unit_normal = scale(normal, 1.0 / normal_length);
                if dot(unit_normal, frame_unit_normal) < 0.0 {
                    unit_normal = scale(unit_normal, -1.0);
                }
                Some((point_m, unit_normal))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let mut max_curvature = None::<f64>;
    for left_index in 0..samples.len() {
        for right_index in (left_index + 1)..samples.len() {
            let distance_m = norm(sub(samples[left_index].0, samples[right_index].0));
            if !distance_m.is_finite() || distance_m <= 1.0e-12 {
                continue;
            }
            let normal_delta = norm(sub(samples[left_index].1, samples[right_index].1));
            if !normal_delta.is_finite() {
                continue;
            }
            let curvature = normal_delta / distance_m;
            if curvature.is_finite() {
                max_curvature =
                    Some(max_curvature.map_or(curvature, |current| current.max(curvature)));
            }
        }
    }
    max_curvature
}

fn max_optional_curvature(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut max_value = None::<f64>;
    for value in values {
        if value.is_finite() && value >= 0.0 {
            max_value = Some(max_value.map_or(value, |current| current.max(value)));
        }
    }
    max_value
}

fn finite_point(point: Point3) -> bool {
    point.iter().all(|coordinate| coordinate.is_finite())
}

fn compare_points_lexicographically(left: Point3, right: Point3) -> std::cmp::Ordering {
    left[0]
        .total_cmp(&right[0])
        .then_with(|| left[1].total_cmp(&right[1]))
        .then_with(|| left[2].total_cmp(&right[2]))
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
    use crate::{build_cad_topology, source_topology_from_boundary_input, BoundaryMeshInput};
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
        assert_eq!(model.report.evaluator_face_count, 2);
        assert_eq!(model.report.exact_query_face_count, 0);
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
        assert_eq!(model.report.evaluator_sample_count, 2);
        assert_eq!(report.live_query_face_count, 2);
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
                point_m: [0.0, 1.0, 1.0],
                uv: Some([2.0, 7.0]),
                projected_point_m: Some([0.0, 1.0, 1.0]),
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
                point_m: [0.3, 0.5, 1.001],
                uv: Some([0.3, 0.5]),
                projected_point_m: Some([0.3, 0.5, 1.0]),
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

        assert_eq!(frame.origin_m, [0.3, 0.5, 1.0]);
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
                point_m: [0.2, 0.5, 1.0],
                uv: Some([0.2, 0.5]),
                projected_point_m: Some([0.2, 0.5, 1.0]),
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
                point_m: [0.0, 1.0, 1.0],
                uv: Some([0.0, 1.0]),
                projected_point_m: Some([0.0, 1.0, 1.0]),
                unit_normal: Some([0.04, 0.0, 0.9991996797437437]),
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
                point_m: [0.0, 1.0, 0.6],
                uv: Some([0.0, 1.0]),
                projected_point_m: Some([0.0, 1.0, 1.0]),
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
        source_topology_from_boundary_input(&BoundaryMeshInput {
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
                crate::BoundaryMeshTriangle {
                    triangle_id: 0,
                    node_ids: [0, 2, 1],
                    region_ids: Vec::new(),
                    provenance: Vec::new(),
                },
                crate::BoundaryMeshTriangle {
                    triangle_id: 1,
                    node_ids: [0, 3, 2],
                    region_ids: Vec::new(),
                    provenance: Vec::new(),
                },
                crate::BoundaryMeshTriangle {
                    triangle_id: 2,
                    node_ids: [4, 5, 6],
                    region_ids: Vec::new(),
                    provenance: Vec::new(),
                },
                crate::BoundaryMeshTriangle {
                    triangle_id: 3,
                    node_ids: [4, 6, 7],
                    region_ids: Vec::new(),
                    provenance: Vec::new(),
                },
            ],
            bounds_min_m: [0.0, 0.0, 0.0],
            bounds_max_m: [1.0, 1.0, 1.0],
            region_ids: Vec::new(),
        })
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
