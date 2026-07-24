use crate::math::{dot, norm, scale, sub, Point3};
use runmat_geometry_core::CadFaceEvaluationSample;

use super::{
    samples::{
        exact_backend_sample_is_valid, exact_backend_sample_point, normalized_sample_normal,
        orient_sample_normal,
    },
    types::{CadFaceEvaluationFrame, CadFaceProjection},
};

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
