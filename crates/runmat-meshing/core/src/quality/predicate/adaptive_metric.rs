use super::{
    adaptive_spatial::{exact_coordinates, validate_points},
    exact_dyadic::ExactDyadic,
    Point3Construction, PredicateSign, SpatialPredicateError, SpatialPredicatePoint,
};

/// Exact classification against the closed sphere whose diameter is `endpoints`.
///
/// Negative is inside, zero is on the sphere, and positive is outside. The filtered path handles
/// ordinary inputs without allocation. Borderline inputs fall back to exact arithmetic over the
/// dyadic rational represented by each finite `f64` coordinate.
pub fn diametral_sphere_position(
    endpoints: [[f64; 3]; 2],
    point: [f64; 3],
) -> Result<PredicateSign, SpatialPredicateError> {
    validate(endpoints, point)?;
    let scale = endpoints
        .into_iter()
        .chain([point])
        .flatten()
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Ok(PredicateSign::Zero);
    }
    let encoded_exponent = (scale.to_bits() >> 52) & 0x7ff;
    if encoded_exponent == 0 {
        return Ok(exact_dot_sign(endpoints, point));
    }
    let power_of_two_scale = f64::from_bits(encoded_exponent << 52);
    let normalized_endpoints =
        endpoints.map(|endpoint| endpoint.map(|value| value / power_of_two_scale));
    let normalized_point = point.map(|value| value / power_of_two_scale);
    if endpoints
        .into_iter()
        .chain([point])
        .flatten()
        .zip(
            normalized_endpoints
                .into_iter()
                .chain([normalized_point])
                .flatten(),
        )
        .any(|(original, normalized)| original != 0.0 && normalized == 0.0)
    {
        return Ok(exact_dot_sign(endpoints, point));
    }
    let products = std::array::from_fn::<_, 3, _>(|axis| {
        (normalized_point[axis] - normalized_endpoints[0][axis])
            * (normalized_point[axis] - normalized_endpoints[1][axis])
    });
    let estimate = products.into_iter().sum::<f64>();
    let magnitude = products.into_iter().map(f64::abs).sum::<f64>();
    let error_bound = 32.0 * f64::EPSILON * magnitude;
    if estimate.abs() > error_bound {
        return Ok(PredicateSign::from_value(estimate));
    }

    Ok(exact_dot_sign(endpoints, point))
}

/// Exact diametral-sphere classification for explicit and dyadic segment-constructed points.
pub fn diametral_sphere_position_constructed(
    endpoints: [SpatialPredicatePoint; 2],
    point: SpatialPredicatePoint,
) -> Result<PredicateSign, SpatialPredicateError> {
    let points = [endpoints[0], endpoints[1], point];
    validate_points(&points)?;
    if points
        .iter()
        .all(|point| point.construction == Point3Construction::Explicit)
    {
        return diametral_sphere_position(
            endpoints.map(|point| point.coordinates),
            point.coordinates,
        );
    }
    let [left, right, query] = points.map(exact_coordinates);
    let terms = std::array::from_fn::<_, 3, _>(|axis| {
        query[axis]
            .clone()
            .subtract_exact(left[axis].clone())
            .multiply_exact(query[axis].clone().subtract_exact(right[axis].clone()))
    });
    Ok(terms
        .into_iter()
        .reduce(ExactDyadic::add_exact)
        .unwrap_or_else(ExactDyadic::zero)
        .sign())
}

fn validate(endpoints: [[f64; 3]; 2], point: [f64; 3]) -> Result<(), SpatialPredicateError> {
    for (point_index, coordinates) in endpoints.into_iter().chain([point]).enumerate() {
        for (axis, coordinate) in coordinates.into_iter().enumerate() {
            if !coordinate.is_finite() {
                return Err(SpatialPredicateError::NonFiniteCoordinate { point_index, axis });
            }
        }
    }
    Ok(())
}

fn exact_dot_sign(endpoints: [[f64; 3]; 2], point: [f64; 3]) -> PredicateSign {
    let terms = std::array::from_fn::<_, 3, _>(|axis| {
        ExactDyadic::from_f64(point[axis])
            .subtract_exact(ExactDyadic::from_f64(endpoints[0][axis]))
            .multiply_exact(
                ExactDyadic::from_f64(point[axis])
                    .subtract_exact(ExactDyadic::from_f64(endpoints[1][axis])),
            )
    });
    let sum = terms
        .into_iter()
        .reduce(ExactDyadic::add_exact)
        .unwrap_or_else(ExactDyadic::zero);
    sum.sign()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_inside_boundary_and_outside_without_a_tolerance() {
        let endpoints = [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        assert_eq!(
            diametral_sphere_position(endpoints, [0.0, 0.0, 0.0]),
            Ok(PredicateSign::Negative)
        );
        assert_eq!(
            diametral_sphere_position(endpoints, [0.0, 1.0, 0.0]),
            Ok(PredicateSign::Zero)
        );
        assert_eq!(
            diametral_sphere_position(endpoints, [0.0, 1.0 + f64::EPSILON, 0.0]),
            Ok(PredicateSign::Positive)
        );
    }

    #[test]
    fn exact_fallback_resolves_large_translated_borderline_input() {
        let origin = 2.0_f64.powi(500);
        let step = 2.0_f64.powi(449);
        let endpoints = [[origin - step, origin, 0.0], [origin + step, origin, 0.0]];
        assert_eq!(
            diametral_sphere_position(endpoints, [origin, origin + step, 0.0]),
            Ok(PredicateSign::Zero)
        );
        assert_eq!(
            diametral_sphere_position(endpoints, [origin, origin + step + 2.0_f64.powi(448), 0.0]),
            Ok(PredicateSign::Positive)
        );
    }

    #[test]
    fn exact_fallback_preserves_subnormal_geometry() {
        let step = f64::from_bits(4);
        let endpoints = [[0.0, 0.0, 0.0], [step, 0.0, 0.0]];
        assert_eq!(
            diametral_sphere_position(endpoints, [f64::from_bits(2), 0.0, 0.0]),
            Ok(PredicateSign::Negative)
        );
        assert_eq!(
            diametral_sphere_position(endpoints, [0.0, step, 0.0]),
            Ok(PredicateSign::Positive)
        );
    }
}
