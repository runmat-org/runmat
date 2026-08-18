use runmat_meshing_core::SurfaceQualityTargets;

use crate::{ExactFaceGeometry, ExactFacePslg, ExactFaceTriangleGeometry, ParametricMetricTensor};

use super::{
    validate_exact_face_feature_collars, ExactFaceFeatureCollars, ExactFaceRefinementCandidate,
    ExactFaceRefinementError, ExactFaceRefinementErrorKind, ExactFaceRefinementReason,
};

const MAXIMUM_NORMALIZED_METRIC_EDGE_LENGTH: f64 = 1.0;

pub fn select_exact_face_refinement_candidate(
    geometry: &ExactFaceGeometry,
    pslg: &ExactFacePslg,
    collars: &ExactFaceFeatureCollars,
    quality: SurfaceQualityTargets,
) -> Result<Option<ExactFaceRefinementCandidate>, ExactFaceRefinementError> {
    quality.validate().map_err(|error| {
        ExactFaceRefinementError::new(
            ExactFaceRefinementErrorKind::InvalidQuality,
            &geometry.source_face_id,
            error.to_string(),
        )
    })?;
    validate_exact_face_feature_collars(collars, pslg, geometry, quality)?;
    if geometry.triangles.is_empty() || geometry.vertices.is_empty() {
        return Err(invalid(geometry, "face geometry inventory is empty"));
    }
    for (triangle_index, triangle) in geometry.triangles.iter().enumerate() {
        let reason = violation(triangle, collars, quality);
        let Some(reason) = reason else {
            continue;
        };
        let uv = match reason {
            ExactFaceRefinementReason::MetricEdgeLength
            | ExactFaceRefinementReason::MetricAngle => metric_circumcenter(geometry, triangle)?,
            ExactFaceRefinementReason::ChordalDeviation
            | ExactFaceRefinementReason::NormalDeviation
            | ExactFaceRefinementReason::PhysicalAspectRatio => triangle.centroid.uv,
        };
        if uv.iter().any(|value| !value.is_finite()) {
            return Err(invalid(geometry, "refinement candidate UV is not finite"));
        }
        return Ok(Some(ExactFaceRefinementCandidate {
            source_face_id: geometry.source_face_id.clone(),
            triangle_index: triangle_index as u32,
            triangle: triangle.triangle,
            reason,
            uv,
        }));
    }
    Ok(None)
}

fn violation(
    triangle: &ExactFaceTriangleGeometry,
    collars: &ExactFaceFeatureCollars,
    quality: SurfaceQualityTargets,
) -> Option<ExactFaceRefinementReason> {
    let minimum_angle = quality.minimum_metric_angle_degrees.to_radians();
    let maximum_normal = quality.maximum_normal_deviation_degrees.to_radians();
    let protected_feature_triangle = collars.collars.iter().any(|collar| {
        triangle
            .triangle
            .vertex_indices
            .contains(&collar.pslg_vertex_index)
    });
    if triangle.chordal_deviation_m > quality.maximum_chordal_deviation_m {
        Some(ExactFaceRefinementReason::ChordalDeviation)
    } else if triangle.normal_deviation_rad > maximum_normal {
        Some(ExactFaceRefinementReason::NormalDeviation)
    } else if triangle
        .metric_edge_lengths
        .into_iter()
        .any(|length| length > MAXIMUM_NORMALIZED_METRIC_EDGE_LENGTH)
    {
        Some(ExactFaceRefinementReason::MetricEdgeLength)
    } else if triangle.minimum_metric_angle_rad < minimum_angle && !protected_feature_triangle {
        Some(ExactFaceRefinementReason::MetricAngle)
    } else if triangle.physical_aspect_ratio > quality.maximum_physical_aspect_ratio
        && !protected_feature_triangle
    {
        Some(ExactFaceRefinementReason::PhysicalAspectRatio)
    } else {
        None
    }
}

fn metric_circumcenter(
    geometry: &ExactFaceGeometry,
    triangle: &ExactFaceTriangleGeometry,
) -> Result<[f64; 2], ExactFaceRefinementError> {
    let corners = triangle
        .triangle
        .vertex_indices
        .map(|index| geometry.vertices.get(index as usize));
    let [Some(first), Some(second), Some(third)] = corners else {
        return Err(invalid(
            geometry,
            "face triangle references an absent geometry vertex",
        ));
    };
    let points = [
        first.evaluation.uv,
        second.evaluation.uv,
        third.evaluation.uv,
    ];
    let metric = triangle.centroid.sizing_metric;
    metric
        .validate()
        .map_err(|reason| invalid(geometry, reason))?;
    let rows = [
        metric_row(points[0], points[1], metric),
        metric_row(points[0], points[2], metric),
    ];
    let right = [
        0.5 * (metric_norm(points[1], metric) - metric_norm(points[0], metric)),
        0.5 * (metric_norm(points[2], metric) - metric_norm(points[0], metric)),
    ];
    let determinant = rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0];
    if !determinant.is_finite() || determinant == 0.0 {
        return Err(invalid(geometry, "metric circumcenter system is singular"));
    }
    Ok([
        (right[0] * rows[1][1] - rows[0][1] * right[1]) / determinant,
        (rows[0][0] * right[1] - right[0] * rows[1][0]) / determinant,
    ])
}

fn metric_row(origin: [f64; 2], point: [f64; 2], metric: ParametricMetricTensor) -> [f64; 2] {
    let delta = [point[0] - origin[0], point[1] - origin[1]];
    [
        metric.uu * delta[0] + metric.uv * delta[1],
        metric.uv * delta[0] + metric.vv * delta[1],
    ]
}

fn metric_norm(point: [f64; 2], metric: ParametricMetricTensor) -> f64 {
    metric.uu * point[0] * point[0]
        + 2.0 * metric.uv * point[0] * point[1]
        + metric.vv * point[1] * point[1]
}

fn invalid(geometry: &ExactFaceGeometry, reason: impl Into<String>) -> ExactFaceRefinementError {
    ExactFaceRefinementError::new(
        ExactFaceRefinementErrorKind::InvalidGeometry,
        &geometry.source_face_id,
        reason,
    )
}
