use crate::{
    math::{cross, dot, norm, scale, sub, Point3, Triangle3},
    topology::CadFace,
};
use runmat_geometry_core::{CadFaceEvaluationSample, CadFaceEvaluationSampleSource};

use super::types::{CadFaceEvaluationRequest, CadFaceEvaluatorProvider};

#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct BoundedCadFaceEvaluationSamples {
    pub(super) samples: Vec<CadFaceEvaluationSample>,
    pub(super) rejected_count: usize,
}

pub(super) fn exact_backend_sample(
    samples: &[CadFaceEvaluationSample],
) -> Option<&CadFaceEvaluationSample> {
    samples
        .iter()
        .filter(|sample| exact_backend_sample_is_valid(sample))
        .min_by(|left, right| compare_exact_backend_samples(left, right))
}

pub(super) fn exact_backend_sample_is_valid(sample: &CadFaceEvaluationSample) -> bool {
    sample.source == CadFaceEvaluationSampleSource::BackendQuery
        && finite_point(sample.point_m)
        && sample
            .unit_normal
            .is_some_and(|normal| finite_point(normal) && norm(normal) > 0.0)
        && sample
            .projection_error_m
            .is_none_or(|error| error.is_finite() && error >= 0.0)
}

pub(super) fn exact_backend_sample_point(sample: &CadFaceEvaluationSample) -> Point3 {
    sample
        .projected_point_m
        .filter(|point| finite_point(*point))
        .unwrap_or(sample.point_m)
}

pub(super) fn normalized_sample_normal(unit_normal: Point3) -> Option<Point3> {
    let normal_length = norm(unit_normal);
    if normal_length.is_finite() && normal_length > 0.0 {
        Some(scale(unit_normal, 1.0 / normal_length))
    } else {
        None
    }
}

pub(super) fn orient_sample_normal(unit_normal: Point3, frame_unit_normal: Point3) -> Point3 {
    if dot(unit_normal, frame_unit_normal) < 0.0 {
        scale(unit_normal, -1.0)
    } else {
        unit_normal
    }
}

pub(super) fn evaluator_max_projection_error(samples: &[CadFaceEvaluationSample]) -> f64 {
    samples
        .iter()
        .filter_map(|sample| sample.projection_error_m)
        .filter(|error| error.is_finite() && *error >= 0.0)
        .fold(0.0_f64, f64::max)
}

pub(super) fn live_evaluator_samples(
    evaluator_provider: &dyn CadFaceEvaluatorProvider,
    face: &CadFace,
    source_face_id: u32,
    reference_point_m: Point3,
    reference_unit_normal: Point3,
) -> BoundedCadFaceEvaluationSamples {
    if face.evaluator_id.is_none()
        || !(face.evaluator_supports_point_evaluation
            || face.evaluator_supports_projection
            || face.evaluator_supports_normal
            || face.evaluator_supports_derivatives
            || face.evaluator_supports_curvature)
    {
        return BoundedCadFaceEvaluationSamples::default();
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
    bounded_cad_face_evaluation_samples(evaluator_provider.evaluate_face(&request))
}

pub(super) fn merged_bounded_evaluator_samples(
    face: &CadFace,
    live_samples: BoundedCadFaceEvaluationSamples,
    source_points: Triangle3,
) -> BoundedCadFaceEvaluationSamples {
    let imported = bounded_cad_face_evaluation_samples(face.evaluator_samples.clone());
    let mut samples = live_samples
        .samples
        .into_iter()
        .chain(imported.samples)
        .collect::<Vec<_>>();
    let rejected_count = live_samples
        .rejected_count
        .saturating_add(imported.rejected_count)
        .saturating_add(filter_samples_to_source_face(&mut samples, source_points))
        .saturating_add(samples.len().saturating_sub(8));
    samples.truncate(8);
    BoundedCadFaceEvaluationSamples {
        samples,
        rejected_count,
    }
}

pub(super) fn estimate_uv_derivatives(
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

pub(super) fn estimate_max_curvature(
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

pub(super) fn max_optional_curvature(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut max_value = None::<f64>;
    for value in values {
        if value.is_finite() && value >= 0.0 {
            max_value = Some(max_value.map_or(value, |current| current.max(value)));
        }
    }
    max_value
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

fn filter_samples_to_source_face(
    samples: &mut Vec<CadFaceEvaluationSample>,
    source_points: Triangle3,
) -> usize {
    let original_len = samples.len();
    samples.retain(|sample| {
        let sample_point = exact_backend_sample_point(sample);
        point_in_source_triangle(sample_point, source_points)
    });
    original_len.saturating_sub(samples.len())
}

fn bounded_cad_face_evaluation_samples(
    samples: Vec<CadFaceEvaluationSample>,
) -> BoundedCadFaceEvaluationSamples {
    let mut accepted = Vec::<CadFaceEvaluationSample>::new();
    let mut rejected_count = 0_usize;
    for sample in samples {
        if bounded_sample_is_valid(&sample) {
            accepted.push(sample);
        } else {
            rejected_count += 1;
        }
    }
    rejected_count = rejected_count.saturating_add(accepted.len().saturating_sub(8));
    accepted.truncate(8);
    BoundedCadFaceEvaluationSamples {
        samples: accepted,
        rejected_count,
    }
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

fn finite_point(point: Point3) -> bool {
    point.iter().all(|coordinate| coordinate.is_finite())
}

fn point_in_source_triangle(point: Point3, triangle: Triangle3) -> bool {
    let v0 = sub(triangle[1], triangle[0]);
    let v1 = sub(triangle[2], triangle[0]);
    let v2 = sub(point, triangle[0]);
    let dot00 = dot(v0, v0);
    let dot01 = dot(v0, v1);
    let dot02 = dot(v0, v2);
    let dot11 = dot(v1, v1);
    let dot12 = dot(v1, v2);
    let denominator = dot00 * dot11 - dot01 * dot01;
    if !denominator.is_finite() || denominator.abs() <= f64::EPSILON {
        return false;
    }
    let inv_denominator = 1.0 / denominator;
    let u = (dot11 * dot02 - dot01 * dot12) * inv_denominator;
    let v = (dot00 * dot12 - dot01 * dot02) * inv_denominator;
    let normal = cross(v0, v1);
    let normal_length = norm(normal);
    if !normal_length.is_finite() || normal_length <= f64::EPSILON {
        return false;
    }
    let plane_distance = dot(v2, scale(normal, 1.0 / normal_length)).abs();
    plane_distance <= 1.0e-8 && u >= -1.0e-10 && v >= -1.0e-10 && u + v <= 1.0 + 1.0e-10
}

fn compare_points_lexicographically(left: Point3, right: Point3) -> std::cmp::Ordering {
    left[0]
        .total_cmp(&right[0])
        .then_with(|| left[1].total_cmp(&right[1]))
        .then_with(|| left[2].total_cmp(&right[2]))
}
