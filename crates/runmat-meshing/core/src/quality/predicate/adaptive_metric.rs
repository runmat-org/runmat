use std::cmp::Ordering;

use num_bigint::BigInt;

use super::{PredicateSign, SpatialPredicateError};

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
        Dyadic::from_f64(point[axis])
            .subtract(Dyadic::from_f64(endpoints[0][axis]))
            .multiply(Dyadic::from_f64(point[axis]).subtract(Dyadic::from_f64(endpoints[1][axis])))
    });
    let sum = terms
        .into_iter()
        .reduce(Dyadic::add)
        .unwrap_or_else(Dyadic::zero);
    match sum.significand.cmp(&BigInt::from(0)) {
        Ordering::Less => PredicateSign::Negative,
        Ordering::Equal => PredicateSign::Zero,
        Ordering::Greater => PredicateSign::Positive,
    }
}

struct Dyadic {
    significand: BigInt,
    exponent: i16,
}

impl Dyadic {
    fn zero() -> Self {
        Self {
            significand: BigInt::from(0),
            exponent: 0,
        }
    }

    fn from_f64(value: f64) -> Self {
        if value == 0.0 {
            return Self::zero();
        }
        let bits = value.to_bits();
        let negative = bits >> 63 != 0;
        let encoded_exponent = ((bits >> 52) & 0x7ff) as i16;
        let fraction = bits & ((1_u64 << 52) - 1);
        let (mantissa, exponent) = if encoded_exponent == 0 {
            (fraction, -1074)
        } else {
            (fraction | (1_u64 << 52), encoded_exponent - 1023 - 52)
        };
        let significand = if negative {
            -BigInt::from(mantissa)
        } else {
            BigInt::from(mantissa)
        };
        Self {
            significand,
            exponent,
        }
    }

    fn subtract(self, right: Self) -> Self {
        self.add(Self {
            significand: -right.significand,
            exponent: right.exponent,
        })
    }

    fn add(self, right: Self) -> Self {
        let exponent = self.exponent.min(right.exponent);
        let left_shift = usize::from((self.exponent - exponent) as u16);
        let right_shift = usize::from((right.exponent - exponent) as u16);
        Self {
            significand: (self.significand << left_shift) + (right.significand << right_shift),
            exponent,
        }
    }

    fn multiply(self, right: Self) -> Self {
        Self {
            significand: self.significand * right.significand,
            exponent: self.exponent + right.exponent,
        }
    }
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
