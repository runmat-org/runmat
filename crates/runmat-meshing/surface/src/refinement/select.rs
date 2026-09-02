use runmat_meshing_core::{predicate::orient2d, PredicateSign, SurfaceQualityTargets};

use crate::{ExactFaceGeometry, ExactFacePslg, ExactFaceTriangleGeometry, ParametricMetricTensor};

use super::{
    encroachment::metric_diametral_disk_contains, validate_exact_face_feature_collars,
    ExactFaceFeatureCollars, ExactFaceRefinementCandidate, ExactFaceRefinementError,
    ExactFaceRefinementErrorKind, ExactFaceRefinementReason,
};

const MAXIMUM_NORMALIZED_METRIC_EDGE_LENGTH: f64 = 1.0;

pub fn select_exact_face_refinement_candidate(
    geometry: &ExactFaceGeometry,
    pslg: &ExactFacePslg,
    collars: &ExactFaceFeatureCollars,
    quality: SurfaceQualityTargets,
) -> Result<Option<ExactFaceRefinementCandidate>, ExactFaceRefinementError> {
    Ok(
        select_exact_face_refinement_candidates(geometry, pslg, collars, quality)?
            .into_iter()
            .next(),
    )
}

pub(crate) fn select_exact_face_refinement_candidates(
    geometry: &ExactFaceGeometry,
    pslg: &ExactFacePslg,
    collars: &ExactFaceFeatureCollars,
    quality: SurfaceQualityTargets,
) -> Result<Vec<ExactFaceRefinementCandidate>, ExactFaceRefinementError> {
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
    let mut candidates = Vec::new();
    for (triangle_index, triangle) in geometry.triangles.iter().enumerate() {
        if let Some((edge, length)) =
            short_unprotected_edge(triangle, pslg, quality.minimum_metric_edge_length)
        {
            return Err(ExactFaceRefinementError::new(
                ExactFaceRefinementErrorKind::UnsatisfiedQuality,
                &geometry.source_face_id,
                format!(
                    "surface topology contains unprotected edge {edge:?} at {:?} -> {:?} with metric length {length}, below the {} minimum",
                    pslg.vertices[edge[0] as usize].uv,
                    pslg.vertices[edge[1] as usize].uv,
                    quality.minimum_metric_edge_length,
                ),
            ));
        }
        let reason = violation(triangle, pslg, collars, quality);
        let Some(reason) = reason else {
            continue;
        };
        let mut uv = match reason {
            ExactFaceRefinementReason::MetricEdgeLength => {
                longest_metric_edge_midpoint(geometry, triangle)?
            }
            ExactFaceRefinementReason::MetricAngle => metric_circumcenter(geometry, triangle)?,
            ExactFaceRefinementReason::ChordalDeviation => triangle.chordal_deviation_uv,
            ExactFaceRefinementReason::NormalDeviation => triangle.normal_deviation_uv,
            ExactFaceRefinementReason::PhysicalAspectRatio => triangle.deviation_samples[3].uv,
        };
        if pslg.vertices.iter().any(|vertex| vertex.uv == uv)
            || !lies_in_trimmed_domain(geometry, uv)?
        {
            // Cocircular neighboring triangles can select a circumcenter that a
            // previous refinement step already inserted. The triangle centroid
            // is a deterministic interior fallback that still reduces its
            // metric edge lengths; encroachment classification below decides
            // whether it must split a protected segment instead.
            uv = triangle.deviation_samples[3].uv;
        }
        if uv.iter().any(|value| !value.is_finite()) {
            return Err(invalid(geometry, "refinement candidate UV is not finite"));
        }
        if pslg.vertices.iter().any(|vertex| vertex.uv == uv) {
            return Err(invalid(
                geometry,
                "refinement triangle has no distinct insertion candidate",
            ));
        }
        if matches!(
            reason,
            ExactFaceRefinementReason::MetricAngle | ExactFaceRefinementReason::PhysicalAspectRatio
        ) && shape_candidate_encroaches_sized_protected_segment(
            uv,
            pslg,
            triangle.deviation_samples[3].sizing_metric,
        ) {
            continue;
        }
        candidates.push(ExactFaceRefinementCandidate {
            source_face_id: geometry.source_face_id.clone(),
            triangle_index: triangle_index as u32,
            triangle: triangle.triangle,
            reason,
            uv,
        });
    }
    Ok(candidates)
}

pub(super) fn shape_candidate_encroaches_sized_protected_segment(
    candidate: [f64; 2],
    pslg: &ExactFacePslg,
    metric: ParametricMetricTensor,
) -> bool {
    pslg.segments.iter().any(|segment| {
        let [Some(first), Some(second)] = segment
            .vertex_indices
            .map(|index| pslg.vertices.get(index as usize))
        else {
            return false;
        };
        let segment_length = metric_distance(first.uv, second.uv, metric);
        // Curve discretization already enforces the sizing field. Shape-only refinement must not
        // drive a sizing-compliant shared boundary toward the much smaller emergency minimum.
        if segment_length > MAXIMUM_NORMALIZED_METRIC_EDGE_LENGTH {
            return false;
        }
        metric_diametral_disk_contains(candidate, first.uv, second.uv, metric).unwrap_or(false)
    })
}

/// Moves an independently discovered geometric-error witness toward its
/// triangle centroid only as far as needed to respect the declared minimum
/// metric spacing. The witness remains in the same convex triangle, so trim
/// membership is preserved; the normal candidate-classification path still
/// decides whether the relocated point requires a protected-boundary split.
pub(crate) fn separate_geometric_witness(
    witness: [f64; 2],
    triangle_index: usize,
    geometry: &ExactFaceGeometry,
    pslg: &ExactFacePslg,
    minimum_separation: f64,
) -> Option<[f64; 2]> {
    let triangle = geometry.triangles.get(triangle_index)?;
    let centroid = triangle.deviation_samples[3].uv;
    let metric = triangle.deviation_samples[3].sizing_metric;
    [0.0, 0.25, 0.5, 0.75, 1.0]
        .into_iter()
        .map(|toward_centroid| {
            std::array::from_fn(|axis| {
                witness[axis] * (1.0 - toward_centroid) + centroid[axis] * toward_centroid
            })
        })
        .find(|candidate| {
            !pslg.vertices.iter().any(|vertex| vertex.uv == *candidate)
                && pslg.vertices.iter().all(|vertex| {
                    metric_distance(*candidate, vertex.uv, metric) + f64::EPSILON
                        >= minimum_separation
                })
        })
        .or_else(|| {
            let corners = triangle
                .triangle
                .vertex_indices
                .map(|index| geometry.vertices[index as usize].evaluation.uv);
            let mut feasible = Vec::new();
            const DENOMINATOR: u32 = 8;
            for first in 0..=DENOMINATOR {
                for second in 0..=(DENOMINATOR - first) {
                    let third = DENOMINATOR - first - second;
                    if [first, second, third].contains(&DENOMINATOR) {
                        continue;
                    }
                    let weights =
                        [first, second, third].map(|value| value as f64 / DENOMINATOR as f64);
                    let candidate = std::array::from_fn(|axis| {
                        corners[0][axis] * weights[0]
                            + corners[1][axis] * weights[1]
                            + corners[2][axis] * weights[2]
                    });
                    if pslg.vertices.iter().any(|vertex| vertex.uv == candidate)
                        || pslg.vertices.iter().any(|vertex| {
                            metric_distance(candidate, vertex.uv, metric) + f64::EPSILON
                                < minimum_separation
                        })
                    {
                        continue;
                    }
                    feasible.push(candidate);
                }
            }
            feasible.sort_by(|left, right| {
                metric_distance(*left, witness, metric)
                    .total_cmp(&metric_distance(*right, witness, metric))
                    .then_with(|| left[0].total_cmp(&right[0]))
                    .then_with(|| left[1].total_cmp(&right[1]))
            });
            feasible.into_iter().next()
        })
}

pub(crate) fn short_unprotected_edge(
    triangle: &ExactFaceTriangleGeometry,
    pslg: &ExactFacePslg,
    minimum_metric_edge_length: f64,
) -> Option<([u32; 2], f64)> {
    let vertices = triangle.triangle.vertex_indices;
    let edges = [
        [vertices[0], vertices[1]],
        [vertices[1], vertices[2]],
        [vertices[2], vertices[0]],
    ];
    triangle
        .metric_edge_lengths
        .iter()
        .zip(edges)
        .find_map(|(length, mut edge)| {
            if *length + f64::EPSILON >= minimum_metric_edge_length {
                return None;
            }
            edge.sort_unstable();
            (!pslg.segments.iter().any(|segment| {
                let mut protected = segment.vertex_indices;
                protected.sort_unstable();
                protected == edge
            }))
            .then_some((edge, *length))
        })
}

/// Selects a deterministic maximal set of refinement cavities that cannot touch through a
/// triangle vertex. Inserting circumcenters from adjacent triangles against one stale topology
/// can manufacture arbitrarily short edges even when every individual candidate is valid.
pub(crate) fn independent_candidate_batch(
    candidates: Vec<ExactFaceRefinementCandidate>,
    geometry: &ExactFaceGeometry,
    pslg: &ExactFacePslg,
    minimum_candidate_separation: f64,
) -> Result<Vec<ExactFaceRefinementCandidate>, ExactFaceRefinementError> {
    let mut selected = Vec::<ExactFaceRefinementCandidate>::new();
    for candidate in candidates {
        let metric = geometry.triangles[candidate.triangle_index as usize].deviation_samples[3]
            .sizing_metric;
        let nearest_existing_distance = pslg
            .vertices
            .iter()
            .map(|vertex| metric_distance(candidate.uv, vertex.uv, metric))
            .fold(f64::INFINITY, f64::min);
        if nearest_existing_distance < minimum_candidate_separation {
            if matches!(
                candidate.reason,
                ExactFaceRefinementReason::ChordalDeviation
                    | ExactFaceRefinementReason::NormalDeviation
                    | ExactFaceRefinementReason::MetricEdgeLength
            ) {
                return Err(ExactFaceRefinementError::new(
                    ExactFaceRefinementErrorKind::UnsatisfiedQuality,
                    &candidate.source_face_id,
                    format!(
                        "surface {:?} target requires candidate {:?} at metric distance {nearest_existing_distance} from the nearest node, below the {minimum_candidate_separation} insertion separation",
                        candidate.reason, candidate.uv
                    ),
                ));
            }
            continue;
        }
        if selected.iter().all(|existing| {
            candidate
                .triangle
                .vertex_indices
                .iter()
                .all(|vertex| !existing.triangle.vertex_indices.contains(vertex))
                && metric_distance(candidate.uv, existing.uv, metric)
                    >= minimum_candidate_separation
        }) {
            selected.push(candidate);
        }
    }
    Ok(selected)
}

fn longest_metric_edge_midpoint(
    geometry: &ExactFaceGeometry,
    triangle: &ExactFaceTriangleGeometry,
) -> Result<[f64; 2], ExactFaceRefinementError> {
    let edge = triangle
        .metric_edge_lengths
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1).then_with(|| right.0.cmp(&left.0)))
        .map(|(index, _)| index)
        .ok_or_else(|| invalid(geometry, "refinement triangle has no metric edges"))?;
    let vertices = triangle.triangle.vertex_indices;
    let [first, second] = match edge {
        0 => [vertices[0], vertices[1]],
        1 => [vertices[1], vertices[2]],
        _ => [vertices[2], vertices[0]],
    };
    let first = geometry.vertices[first as usize].evaluation.uv;
    let second = geometry.vertices[second as usize].evaluation.uv;
    Ok([
        first[0] * 0.5 + second[0] * 0.5,
        first[1] * 0.5 + second[1] * 0.5,
    ])
}

fn metric_distance(left: [f64; 2], right: [f64; 2], metric: ParametricMetricTensor) -> f64 {
    let delta = [left[0] - right[0], left[1] - right[1]];
    (metric.uu * delta[0] * delta[0]
        + 2.0 * metric.uv * delta[0] * delta[1]
        + metric.vv * delta[1] * delta[1])
        .max(0.0)
        .sqrt()
}

fn lies_in_trimmed_domain(
    geometry: &ExactFaceGeometry,
    uv: [f64; 2],
) -> Result<bool, ExactFaceRefinementError> {
    for triangle in &geometry.triangles {
        let vertices = triangle
            .triangle
            .vertex_indices
            .map(|index| geometry.vertices[index as usize].evaluation.uv);
        let signs = [
            orient2d([vertices[0], vertices[1], uv]),
            orient2d([vertices[1], vertices[2], uv]),
            orient2d([vertices[2], vertices[0], uv]),
        ]
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| invalid(geometry, "refinement candidate has invalid predicate input"))?;
        let nonnegative = signs.iter().all(|sign| *sign != PredicateSign::Negative);
        let nonpositive = signs.iter().all(|sign| *sign != PredicateSign::Positive);
        if nonnegative || nonpositive {
            return Ok(true);
        }
    }
    Ok(false)
}

fn violation(
    triangle: &ExactFaceTriangleGeometry,
    pslg: &ExactFacePslg,
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
    // Shape-only refinement stops once a protected edge reaches the declared minimum. Geometric
    // deviation and metric-size requirements remain hard because suppressing those would change
    // the represented CAD boundary. Angle and physical-aspect candidates approaching a nearby
    // sizing-compliant boundary use the protected sizing collar above: neither may refine shared
    // curves below the requested local size merely to improve triangle shape.
    let vertices = triangle.triangle.vertex_indices;
    let protected_edges = [
        [vertices[0], vertices[1]],
        [vertices[1], vertices[2]],
        [vertices[2], vertices[0]],
    ];
    let protected_minimum_edge_triangle = triangle
        .metric_edge_lengths
        .into_iter()
        .zip(protected_edges)
        .any(|(length, mut edge)| {
            edge.sort_unstable();
            length <= quality.minimum_metric_edge_length
                && pslg.segments.iter().any(|segment| {
                    let mut protected = segment.vertex_indices;
                    protected.sort_unstable();
                    protected == edge
                })
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
    } else if triangle.minimum_metric_angle_rad < minimum_angle
        && !protected_feature_triangle
        && !protected_minimum_edge_triangle
    {
        Some(ExactFaceRefinementReason::MetricAngle)
    } else if triangle.physical_aspect_ratio > quality.maximum_physical_aspect_ratio
        && !protected_feature_triangle
        && !protected_minimum_edge_triangle
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
    let metric = triangle.deviation_samples[3].sizing_metric;
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
