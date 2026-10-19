//! Exact radius ordering for affine points. Placement hints never confer admission.
mod filter;
use super::{
    adaptive_spatial::{exact_coordinates, validate_points},
    ExactDyadic, PredicateSign, SpatialPredicateError, SpatialPredicatePoint,
};

/// Sign of |query-center|² - |reference-center|², without a tolerance.
/// Negative means the query is inside the reference radius. Finite binary64
/// coordinates and dyadic constructions are evaluated as exact rationals.
pub fn compare_radial_distances_constructed(
    center: SpatialPredicatePoint,
    reference: SpatialPredicatePoint,
    query: SpatialPredicatePoint,
) -> Result<PredicateSign, SpatialPredicateError> {
    let points = [center, reference, query];
    validate_points(&points)?;
    if let Some(sign) = filter::explicit_sign(points) {
        return Ok(sign);
    }
    Ok(exact_sign(points))
}

fn exact_sign(points: [SpatialPredicatePoint; 3]) -> PredicateSign {
    let [center, reference, query] = points.map(exact_coordinates);
    let mut difference = ExactDyadic::zero();
    for axis in 0..3 {
        let query_delta = query[axis].clone().subtract_exact(center[axis].clone());
        let reference_delta = reference[axis].clone().subtract_exact(center[axis].clone());
        difference = difference
            .add_exact(query_delta.clone().multiply_exact(query_delta))
            .subtract_exact(reference_delta.clone().multiply_exact(reference_delta));
    }
    difference.sign()
}

#[cfg(test)]
mod tests;
