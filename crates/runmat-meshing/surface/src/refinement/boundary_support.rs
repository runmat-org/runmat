//! Face-owned boundary support in each adjacent region's edge-local metric.
//! This is a generation strategy, not a proof that volume quality is feasible.
//! It preserves exact curve samples and every original acceptance threshold.
use super::{
    ExactFaceRefinementCandidate, ExactFaceRefinementError, ExactFaceRefinementErrorKind,
    ExactFaceRefinementReason,
};
use crate::{
    ExactFaceGeometry, ExactFacePslg, ExactFacePslgSegmentSource, ResolvedFaceMetricField,
};
use runmat_geometry_core::{GeometryAllocationReservation, GeometryEvaluationControl};
use runmat_meshing_core::MetricTensor3;
pub(super) fn candidates(
    field: &ResolvedFaceMetricField,
    geometry: &ExactFaceGeometry,
    pslg: &ExactFacePslg,
    control: &dyn GeometryEvaluationControl,
) -> Result<Vec<ExactFaceRefinementCandidate>, ExactFaceRefinementError> {
    if !field.has_volume_edge_metrics(&pslg.source_face_id) {
        return Ok(Vec::new());
    }
    let failure = |reason| {
        ExactFaceRefinementError::new(
            ExactFaceRefinementErrorKind::UnsatisfiedQuality,
            &pslg.source_face_id,
            reason,
        )
    };
    let map_control = |error: runmat_geometry_core::GeometryEvaluationError| {
        ExactFaceRefinementError::new(
            ExactFaceRefinementErrorKind::Metric(
                crate::ExactFaceMetricErrorKind::GeometryEvaluation(error.kind),
            ),
            &pslg.source_face_id,
            error.reason,
        )
    };
    let bytes = pslg
        .segments
        .len()
        .checked_mul(std::mem::size_of::<(
            [u32; 2],
            &runmat_geometry_core::PersistentEntityId,
        )>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| failure("boundary support index overflow"))?;
    let _index_storage = GeometryAllocationReservation::new(control, bytes).map_err(map_control)?;
    let mut segments = Vec::with_capacity(pslg.segments.len());
    for segment in &pslg.segments {
        let ExactFacePslgSegmentSource::ExactTrim { source_edge_id, .. } = &segment.source else {
            continue;
        };
        let mut edge = segment.vertex_indices;
        edge.sort_unstable();
        segments.push((edge, source_edge_id));
    }
    segments.sort_unstable_by_key(|entry| entry.0);
    let mut result = Vec::new();
    for (triangle_index, triangle) in geometry.triangles.iter().enumerate() {
        control.checkpoint().map_err(map_control)?;
        let corners = triangle.triangle.vertex_indices;
        let mut selected = None;
        for (first, second, opposite) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)] {
            let mut edge = [corners[first], corners[second]];
            edge.sort_unstable();
            let Ok(index) = segments.binary_search_by_key(&edge, |entry| entry.0) else {
                continue;
            };
            let points = [first, second, opposite]
                .map(|slot| geometry.vertices[corners[slot] as usize].point_m);
            for metric in field.volume_edge_metrics(&pslg.source_face_id, segments[index].1) {
                if let Some((ratio, fraction)) = proposal(points, metric) {
                    if selected
                        .as_ref()
                        .is_none_or(|(best, _, _, _)| ratio > *best)
                    {
                        selected = Some((ratio, fraction, [first, second], opposite));
                    }
                }
            }
        }
        if let Some((_, fraction, edge, opposite)) = selected {
            let uv = [0, 1].map(|axis| {
                let midpoint = 0.5
                    * (pslg.vertices[corners[edge[0]] as usize].uv[axis]
                        + pslg.vertices[corners[edge[1]] as usize].uv[axis]);
                midpoint
                    + fraction * (pslg.vertices[corners[opposite] as usize].uv[axis] - midpoint)
            });
            if uv.iter().any(|value| !value.is_finite()) {
                return Err(failure("nonfinite boundary support proposal"));
            }
            storage::reserve_candidate(&mut result, segments.len(), control, &pslg.source_face_id)?;
            let owned = crate::ExactFaceMetricProvenance::storage_bytes(&pslg.source_face_id, &[])
                .and_then(|bytes| {
                    bytes
                        .checked_sub(std::mem::size_of::<crate::ExactFaceMetricProvenance>() as u64)
                })
                .ok_or_else(|| failure("boundary support candidate storage overflow"))?;
            control
                .consume_allocation_bytes(owned)
                .map_err(map_control)?;
            result.push(ExactFaceRefinementCandidate {
                source_face_id: pslg.source_face_id.clone(),
                triangle_index: triangle_index as u32,
                triangle: triangle.triangle,
                reason: ExactFaceRefinementReason::BoundaryMetricSupport,
                uv,
            });
        }
    }
    Ok(result)
}
/// A transformed altitude beyond twice the edge length creates a poorly
/// supported boundary collar even when a stronger face envelope accepts it.
/// Place the support one edge length from the edge in that local metric, using
/// a convex chart-space blend; no global curve split or whole-face tightening.
fn proposal(points: [[f64; 3]; 3], metric: MetricTensor3) -> Option<(f64, f64)> {
    let edge = std::array::from_fn(|axis| points[1][axis] - points[0][axis]);
    let radial =
        std::array::from_fn(|axis| points[2][axis] - 0.5 * (points[0][axis] + points[1][axis]));
    let squared = dot(edge, metric, edge);
    let axial = dot(edge, metric, radial);
    let radial_squared = dot(radial, metric, radial) - axial * axial / squared;
    let ratio = radial_squared / squared;
    if squared > 0. && ratio.is_finite() && ratio > 4. {
        Some((ratio, 1. / ratio.sqrt()))
    } else {
        None
    }
}
fn dot(a: [f64; 3], metric: MetricTensor3, b: [f64; 3]) -> f64 {
    a[0] * (metric.xx * b[0] + metric.xy * b[1] + metric.xz * b[2])
        + a[1] * (metric.xy * b[0] + metric.yy * b[1] + metric.yz * b[2])
        + a[2] * (metric.xz * b[0] + metric.yz * b[1] + metric.zz * b[2])
}
mod storage;
#[cfg(test)]
mod tests;
