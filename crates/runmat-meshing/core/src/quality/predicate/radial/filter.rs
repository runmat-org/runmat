//! Outward binary64 enclosure. An uncertain interval always falls back to exact
//! arithmetic; this filter introduces no tolerance and no construction rounding.
use super::super::{Point3Construction, PredicateSign, SpatialPredicatePoint};

pub(super) fn explicit_sign(points: [SpatialPredicatePoint; 3]) -> Option<PredicateSign> {
    if points
        .iter()
        .any(|point| point.construction != Point3Construction::Explicit)
    {
        return None;
    }
    let [center, reference, query] = points.map(|point| point.coordinates);
    let mut difference = [0.0; 2];
    for axis in 0..3 {
        let query_delta = [
            (query[axis] - center[axis]).next_down(),
            (query[axis] - center[axis]).next_up(),
        ];
        let reference_delta = [
            (reference[axis] - center[axis]).next_down(),
            (reference[axis] - center[axis]).next_up(),
        ];
        let query_squared = square(query_delta);
        let reference_squared = square(reference_delta);
        difference = [
            ((difference[0] + query_squared[0]).next_down() - reference_squared[1]).next_down(),
            ((difference[1] + query_squared[1]).next_up() - reference_squared[0]).next_up(),
        ];
    }
    if !difference.into_iter().all(f64::is_finite) {
        return None;
    }
    if difference[0] > 0.0 {
        Some(PredicateSign::Positive)
    } else if difference[1] < 0.0 {
        Some(PredicateSign::Negative)
    } else {
        None
    }
}
fn square(range: [f64; 2]) -> [f64; 2] {
    let values = range.map(|value| value * value);
    let lower = if range[0] <= 0.0 && range[1] >= 0.0 {
        0.0
    } else {
        values[0].min(values[1]).next_down().max(0.0)
    };
    [lower, values[0].max(values[1]).next_up()]
}
