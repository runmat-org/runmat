//! Face-local broad phase. Original metric/predicate tests remain authoritative.
//! Packed storage is admitted before construction and released with the caller's scope.
use super::{ExactFaceRefinementError, ExactFaceRefinementErrorKind};
use crate::ParametricMetricTensor;
use runmat_geometry_core::{GeometryEvaluationControl, PersistentEntityId};
use runmat_meshing_core::quality::spatial_index::{Aabb3, PackedBoundsSpatialIndex, SpatialEntry};

pub(super) fn build(
    count: usize,
    entries: impl Iterator<Item = SpatialEntry<usize>>,
    control: Option<&dyn GeometryEvaluationControl>,
    face: &PersistentEntityId,
) -> Result<PackedBoundsSpatialIndex<usize>, ExactFaceRefinementError> {
    let profile = PackedBoundsSpatialIndex::<usize>::storage_profile(count)
        .ok_or_else(|| invalid(face, "face query storage overflow"))?;
    let input = count
        .checked_mul(std::mem::size_of::<SpatialEntry<usize>>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| invalid(face, "face query entry storage overflow"))?;
    let temporary = input
        .checked_add(profile.construction_scratch_bytes)
        .ok_or_else(|| invalid(face, "face query scratch storage overflow"))?;
    let total = temporary
        .checked_add(profile.retained_bytes)
        .ok_or_else(|| invalid(face, "face query storage overflow"))?;
    if let Some(control) = control {
        control
            .consume_allocation_bytes(total)
            .map_err(|error| failure(face, error))?;
    }
    let result =
        PackedBoundsSpatialIndex::from_entries_with_checkpoint(entries.collect(), &mut || {
            super::control::checkpoint(control, face)
        })?
        .ok_or_else(|| invalid(face, "face query bounds are not finite and ordered"))?;
    if let Some(control) = control {
        control.release_allocation_bytes(temporary);
    }
    Ok(result)
}

pub(super) fn point(uv: [f64; 2]) -> Aabb3 {
    Aabb3 {
        min_m: [uv[0], uv[1], 0.0],
        max_m: [uv[0], uv[1], 0.0],
    }
}

pub(super) fn bounds(points: impl IntoIterator<Item = [f64; 2]>) -> Aabb3 {
    let mut result = Aabb3 {
        min_m: [f64::MAX, f64::MAX, 0.0],
        max_m: [-f64::MAX, -f64::MAX, 0.0],
    };
    for uv in points {
        for (axis, value) in uv.into_iter().enumerate() {
            result.min_m[axis] = result.min_m[axis].min(value);
            result.max_m[axis] = result.max_m[axis].max(value);
        }
    }
    result
}

/// Gershgorin gives a lower bound on metric distance in chart coordinates.
/// Subtract a roundoff allowance for the original floating quadratic expression.
/// If that bound is inconclusive, retain the complete finite plane: broad phase
/// uncertainty must never change an exact decision, including clamped distances.
pub(super) fn metric_bounds(
    uv: [f64; 2],
    metric: ParametricMetricTensor,
    distance: f64,
    coordinate_extent: f64,
) -> Aabb3 {
    let scale = metric.uu.abs().max(metric.vv.abs()).max(metric.uv.abs());
    let lower = metric.uu.min(metric.vv) - metric.uv.abs() - 64.0 * f64::EPSILON * scale;
    // Keep the complete inventory when the direct expression could overflow.
    // Its existing clamped arithmetic must not be changed by spatial pruning.
    let safe_extent = (f64::MAX / (16.0 * scale.max(1.0))).sqrt();
    let radius = if lower > 0.0 && coordinate_extent <= safe_extent {
        (distance.max(f64::MIN_POSITIVE.sqrt()) / lower.sqrt()).next_up()
    } else {
        f64::INFINITY
    };
    let mut result = point(uv);
    for (axis, value) in uv.into_iter().enumerate() {
        result.min_m[axis] = (value - radius).next_down().max(-f64::MAX);
        result.max_m[axis] = (value + radius).next_up().min(f64::MAX);
    }
    result
}

fn invalid(face: &PersistentEntityId, reason: &str) -> ExactFaceRefinementError {
    ExactFaceRefinementError::new(ExactFaceRefinementErrorKind::InvalidGeometry, face, reason)
}
fn failure(
    face: &PersistentEntityId,
    error: runmat_geometry_core::GeometryEvaluationError,
) -> ExactFaceRefinementError {
    ExactFaceRefinementError::new(
        ExactFaceRefinementErrorKind::Metric(crate::ExactFaceMetricErrorKind::GeometryEvaluation(
            error.kind,
        )),
        face,
        error.reason,
    )
}
