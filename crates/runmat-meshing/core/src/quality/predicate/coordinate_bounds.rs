//! Outward bounds enclose exact affine constructions and stored coordinates.
//! They supply conservative broad-phase geometry, never predicate admission.
use super::adaptive_spatial::{affine_terms, validate_points};
use super::{Point3Construction, SpatialPredicateError, SpatialPredicatePoint};
use crate::Aabb3;

pub fn constructed_point_bounds(
    point: SpatialPredicatePoint,
) -> Result<Aabb3, SpatialPredicateError> {
    validate_points(&[point])?;
    let mut result = Aabb3 {
        min_m: point.coordinates,
        max_m: point.coordinates,
    };
    if point.construction == Point3Construction::Explicit {
        return Ok(result);
    }
    let terms = affine_terms(point).expect("validated affine construction");
    for axis in 0..3 {
        let mut range = [0.0, 0.0];
        let mut hull = [f64::INFINITY, f64::NEG_INFINITY];
        for (source, numerator, exponent) in &terms {
            hull[0] = hull[0].min(source[axis]);
            hull[1] = hull[1].max(source[axis]);
            if *numerator == 0 {
                continue;
            }
            let rounded = *numerator as f64;
            let mut weight = [rounded, rounded];
            if rounded as u64 > *numerator {
                weight[0] = rounded.next_down();
            }
            if (rounded as u64) < *numerator {
                weight[1] = rounded.next_up();
            }
            let denominator = 2.0_f64.powi(i32::from(*exponent));
            weight = weight.map(|value| value / denominator);
            let products = weight.map(|value| source[axis] * value);
            let low = products[0].min(products[1]).next_down();
            let high = products[0].max(products[1]).next_up();
            range = [(range[0] + low).next_down(), (range[1] + high).next_up()];
        }
        // Validated nonnegative affine weights sum exactly to one. Their source
        // hull is an independent finite enclosure even if interval sums overflow.
        result.min_m[axis] = result.min_m[axis].min(range[0].max(hull[0]));
        result.max_m[axis] = result.max_m[axis].max(range[1].min(hull[1]));
    }
    Ok(result)
}

#[cfg(test)]
mod tests;
