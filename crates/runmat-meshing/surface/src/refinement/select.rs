use runmat_geometry_core::GeometryEvaluationControl;
use runmat_meshing_core::{MetricTensor3, SurfaceQualityTargets};

mod batch;
mod domain;
mod protected_edges;
mod separation;
mod vertices;
use super::control::checkpoint;
#[cfg(test)]
pub(crate) use batch::independent_candidate_batch;
pub(crate) use batch::independent_candidate_batch_in_metric;
use domain::DomainQueries;
use protected_edges::ProtectedEdges;
use runmat_geometry_core::GeometryAllocationScope;
pub(crate) use separation::separate_refinement_candidate;
use vertices::VertexQueries;

use crate::{ExactFaceGeometry, ExactFacePslg, ExactFaceTriangleGeometry, ParametricMetricTensor};

use super::{
    encroachment::metric_diametral_disk_contains, validate_exact_face_feature_collars,
    ExactFaceFeatureCollars, ExactFaceRefinementCandidate, ExactFaceRefinementError,
    ExactFaceRefinementErrorKind, ExactFaceRefinementReason,
};

const MAXIMUM_NORMALIZED_METRIC_EDGE_LENGTH: f64 = MetricTensor3::NORMALIZED_TARGET_LENGTH;

pub fn select_exact_face_refinement_candidate(
    geometry: &ExactFaceGeometry,
    pslg: &ExactFacePslg,
    collars: &ExactFaceFeatureCollars,
    quality: SurfaceQualityTargets,
) -> Result<Option<ExactFaceRefinementCandidate>, ExactFaceRefinementError> {
    Ok(
        select_exact_face_refinement_candidates(geometry, pslg, collars, quality, None)?
            .into_iter()
            .next(),
    )
}

pub(crate) fn select_exact_face_refinement_candidates(
    geometry: &ExactFaceGeometry,
    pslg: &ExactFacePslg,
    collars: &ExactFaceFeatureCollars,
    quality: SurfaceQualityTargets,
    control: Option<&dyn GeometryEvaluationControl>,
) -> Result<Vec<ExactFaceRefinementCandidate>, ExactFaceRefinementError> {
    checkpoint(control, &geometry.source_face_id)?;
    quality.validate().map_err(|error| {
        ExactFaceRefinementError::new(
            ExactFaceRefinementErrorKind::InvalidQuality,
            &geometry.source_face_id,
            error.to_string(),
        )
    })?;
    validate_exact_face_feature_collars(collars, pslg, geometry, quality)?;
    checkpoint(control, &geometry.source_face_id)?;
    if geometry.triangles.is_empty() || geometry.vertices.is_empty() {
        return Err(invalid(geometry, "face geometry inventory is empty"));
    }
    let protected_edges = ProtectedEdges::prepare(pslg, geometry, control)?;
    let storage = control.map(GeometryAllocationScope::new);
    let admitted = storage
        .as_ref()
        .map(|scope| scope as &dyn GeometryEvaluationControl);
    let mut queries = None;
    let mut candidates = Vec::new();
    for (triangle_index, triangle) in geometry.triangles.iter().enumerate() {
        checkpoint(control, &geometry.source_face_id)?;
        if let Some((edge, length)) = short_unprotected_edge(
            triangle,
            &protected_edges,
            quality.minimum_metric_edge_length,
        ) {
            return Err(ExactFaceRefinementError::new(
                ExactFaceRefinementErrorKind::UnsatisfiedQuality,
                &geometry.source_face_id,
                format!(
                    "surface topology contains unprotected edge {edge:?} at {:?} -> {:?} with metric length {length}, below the {} minimum; incident segments {}",
                    pslg.vertices[edge[0] as usize].uv,
                    pslg.vertices[edge[1] as usize].uv,
                    quality.minimum_metric_edge_length,
                    incident_segments(pslg, edge),
                ),
            ));
        }
        let reason = violation(triangle, &protected_edges, collars, quality);
        let Some(reason) = reason else {
            continue;
        };
        if queries.is_none() {
            queries = Some((
                VertexQueries::prepare(pslg, admitted)?,
                DomainQueries::prepare(geometry, admitted)?,
            ));
        }
        let (vertices, domain) = queries
            .as_ref()
            .expect("prepared only for a violated triangle");
        let mut uv = match reason {
            ExactFaceRefinementReason::MetricEdgeLength => {
                longest_metric_edge_midpoint(geometry, triangle)?
            }
            ExactFaceRefinementReason::BoundaryMetricSupport => {
                unreachable!("boundary support has its own geometric proposal")
            }
            ExactFaceRefinementReason::MetricAngle => metric_circumcenter(geometry, triangle)?,
            ExactFaceRefinementReason::ChordalDeviation => triangle.chordal_deviation_uv,
            ExactFaceRefinementReason::NormalDeviation => triangle.normal_deviation_uv,
            ExactFaceRefinementReason::PhysicalAspectRatio => triangle.deviation_samples[3].uv,
        };
        if vertices.contains(uv, control)? || !domain.contains(uv, control)? {
            // Cocircular neighboring triangles can select a circumcenter that a
            // previous refinement step already inserted. The triangle centroid
            // is a deterministic interior fallback that still reduces its
            // metric edge lengths; encroachment classification below decides
            // whether it must split a protected segment instead.
            uv = triangle.deviation_samples[3].uv;
        }
        if reason == ExactFaceRefinementReason::MetricEdgeLength {
            let metric = triangle.deviation_samples[3].sizing_metric;
            if vertices.nearest_within(uv, metric, quality.minimum_metric_edge_length, control)?
                < quality.minimum_metric_edge_length
            {
                // A nonincident node can lie close to the longest-edge midpoint in
                // the local metric. Construct a separated point in this triangle
                // before protected-segment classification; never waive spacing.
                if let Some(separated) = separation::separate_with_queries(
                    uv,
                    triangle_index,
                    geometry,
                    vertices,
                    quality.minimum_metric_edge_length,
                    control,
                )? {
                    uv = separated;
                }
            }
        }
        if uv.iter().any(|value| !value.is_finite()) {
            return Err(invalid(geometry, "refinement candidate UV is not finite"));
        }
        if vertices.contains(uv, control)? {
            return Err(invalid(
                geometry,
                "refinement triangle has no distinct insertion candidate",
            ));
        }
        if matches!(
            reason,
            ExactFaceRefinementReason::MetricAngle | ExactFaceRefinementReason::PhysicalAspectRatio
        ) && (shape_candidate_encroaches_sized_protected_segment(
            uv,
            pslg,
            triangle.deviation_samples[3].sizing_metric,
        ) || shape_candidate_crowds_a_shared_curve(
            uv,
            pslg,
            triangle.deviation_samples[3].sizing_metric,
            quality.minimum_metric_edge_length,
        )) {
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

/// Shape refinement never splits a shared curve, so a shape point closer to one than the minimum
/// metric spacing would make a sliver against it whose own repair point lies closer still; the
/// cascade ends with points within roundoff of the curve. Such a point is not proposed, as with
/// the vertex spacing collar.
fn shape_candidate_crowds_a_shared_curve(
    candidate: [f64; 2],
    pslg: &ExactFacePslg,
    metric: ParametricMetricTensor,
    minimum_metric_edge_length: f64,
) -> bool {
    pslg.segments.iter().any(|segment| {
        if !matches!(
            segment.source,
            crate::ExactFacePslgSegmentSource::ExactTrim { .. }
        ) {
            return false;
        }
        let [Some(first), Some(second)] = segment
            .vertex_indices
            .map(|index| pslg.vertices.get(index as usize))
        else {
            return false;
        };
        let along = [second.uv[0] - first.uv[0], second.uv[1] - first.uv[1]];
        let offset = [candidate[0] - first.uv[0], candidate[1] - first.uv[1]];
        let inner = |left: [f64; 2], right: [f64; 2]| {
            metric.uu * left[0] * right[0]
                + metric.uv * (left[0] * right[1] + left[1] * right[0])
                + metric.vv * left[1] * right[1]
        };
        let length_squared = inner(along, along);
        if length_squared.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return false;
        }
        let t = (inner(offset, along) / length_squared).clamp(0.0, 1.0);
        let gap = [offset[0] - t * along[0], offset[1] - t * along[1]];
        let distance_squared = inner(gap, gap);
        distance_squared.is_finite()
            && distance_squared < minimum_metric_edge_length * minimum_metric_edge_length
    })
}

/// The PSLG segments at each end of a short unprotected edge, with their trim edge and far end.
fn incident_segments(pslg: &ExactFacePslg, edge: [u32; 2]) -> String {
    edge.iter()
        .map(|vertex| {
            let segments = pslg
                .segments
                .iter()
                .filter(|segment| segment.vertex_indices.contains(vertex))
                .map(|segment| {
                    let other =
                        segment.vertex_indices[usize::from(segment.vertex_indices[0] == *vertex)];
                    let source = match &segment.source {
                        crate::ExactFacePslgSegmentSource::ExactTrim { source_edge_id, .. } => {
                            source_edge_id.source_topology_id.clone()
                        }
                        crate::ExactFacePslgSegmentSource::ChartCut { .. } => {
                            "chart cut".to_owned()
                        }
                    };
                    format!(
                        "{source} to {other} at {:?}",
                        pslg.vertices[other as usize].uv
                    )
                })
                .collect::<Vec<_>>();
            format!("{vertex}: [{}]", segments.join("; "))
        })
        .collect::<Vec<_>>()
        .join(" / ")
}

fn short_unprotected_edge(
    triangle: &ExactFaceTriangleGeometry,
    protected_edges: &ProtectedEdges,
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
            (!protected_edges.contains(edge)).then_some((edge, *length))
        })
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

pub(super) fn metric_distance(
    left: [f64; 2],
    right: [f64; 2],
    metric: ParametricMetricTensor,
) -> f64 {
    let delta = [left[0] - right[0], left[1] - right[1]];
    (metric.uu * delta[0] * delta[0]
        + 2.0 * metric.uv * delta[0] * delta[1]
        + metric.vv * delta[1] * delta[1])
        .max(0.0)
        .sqrt()
}

fn violation(
    triangle: &ExactFaceTriangleGeometry,
    protected_edges: &ProtectedEdges,
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
    let triangle_edges = [
        [vertices[0], vertices[1]],
        [vertices[1], vertices[2]],
        [vertices[2], vertices[0]],
    ];
    let protected_minimum_edge_triangle = triangle
        .metric_edge_lengths
        .into_iter()
        .zip(triangle_edges)
        .any(|(length, edge)| {
            length <= quality.minimum_metric_edge_length && protected_edges.contains(edge)
        });
    if triangle.chordal_deviation_m > quality.maximum_chordal_deviation_m.canonical_value() {
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
