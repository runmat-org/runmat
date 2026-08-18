use robust::Coord;

use crate::StableDigest;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredicateSign {
    Negative,
    Zero,
    Positive,
}

impl PredicateSign {
    fn from_value(value: f64) -> Self {
        if value > 0.0 {
            Self::Positive
        } else if value < 0.0 {
            Self::Negative
        } else {
            Self::Zero
        }
    }

    fn from_odd_permutation(odd: bool) -> Self {
        if odd {
            Self::Negative
        } else {
            Self::Positive
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanarPredicatePoint {
    pub identity: StableDigest,
    pub coordinates: [f64; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanarPredicateError {
    NonFiniteCoordinate {
        point_index: usize,
        axis: usize,
    },
    ZeroIdentity {
        point_index: usize,
    },
    DuplicateIdentity {
        left_index: usize,
        right_index: usize,
    },
}

/// Adaptive-precision orientation without tolerance or symbolic perturbation.
pub fn orient2d(points: [[f64; 2]; 3]) -> Result<PredicateSign, PlanarPredicateError> {
    validate_coordinates(&points)?;
    Ok(PredicateSign::from_value(robust::orient2d(
        coord(points[0]),
        coord(points[1]),
        coord(points[2]),
    )))
}

/// Adaptive-precision in-circle test. Positive means inside for a counterclockwise triangle.
pub fn incircle2d(points: [[f64; 2]; 4]) -> Result<PredicateSign, PlanarPredicateError> {
    validate_coordinates(&points)?;
    Ok(PredicateSign::from_value(robust::incircle(
        coord(points[0]),
        coord(points[1]),
        coord(points[2]),
        coord(points[3]),
    )))
}

/// Orientation with a stable alternating tie-break for distinct persistent point identities.
pub fn orient2d_symbolic(
    points: [PlanarPredicatePoint; 3],
) -> Result<PredicateSign, PlanarPredicateError> {
    validate_points(&points)?;
    match orient2d(points.map(|point| point.coordinates))? {
        PredicateSign::Zero => Ok(PredicateSign::from_odd_permutation(permutation_is_odd(
            points.map(|point| point.identity),
        ))),
        sign => Ok(sign),
    }
}

/// In-circle test with a stable alternating tie-break for cocircular points.
pub fn incircle2d_symbolic(
    points: [PlanarPredicatePoint; 4],
) -> Result<PredicateSign, PlanarPredicateError> {
    validate_points(&points)?;
    match incircle2d(points.map(|point| point.coordinates))? {
        PredicateSign::Zero => Ok(PredicateSign::from_odd_permutation(permutation_is_odd(
            points.map(|point| point.identity),
        ))),
        sign => Ok(sign),
    }
}

fn coord(point: [f64; 2]) -> Coord<f64> {
    Coord {
        x: point[0],
        y: point[1],
    }
}

fn validate_coordinates<const N: usize>(
    points: &[[f64; 2]; N],
) -> Result<(), PlanarPredicateError> {
    for (point_index, point) in points.iter().enumerate() {
        for (axis, value) in point.iter().enumerate() {
            if !value.is_finite() {
                return Err(PlanarPredicateError::NonFiniteCoordinate { point_index, axis });
            }
        }
    }
    Ok(())
}

fn validate_points<const N: usize>(
    points: &[PlanarPredicatePoint; N],
) -> Result<(), PlanarPredicateError> {
    validate_coordinates(&points.map(|point| point.coordinates))?;
    for (point_index, point) in points.iter().enumerate() {
        if point.identity == StableDigest::ZERO {
            return Err(PlanarPredicateError::ZeroIdentity { point_index });
        }
        if let Some(left_index) = points[..point_index]
            .iter()
            .position(|left| left.identity == point.identity)
        {
            return Err(PlanarPredicateError::DuplicateIdentity {
                left_index,
                right_index: point_index,
            });
        }
    }
    Ok(())
}

fn permutation_is_odd<const N: usize>(identities: [StableDigest; N]) -> bool {
    let inversion_count = (0..N)
        .flat_map(|left| (left + 1..N).map(move |right| (left, right)))
        .filter(|(left, right)| identities[*left] > identities[*right])
        .count();
    inversion_count % 2 == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_predicates_resolve_near_degenerate_inputs() {
        let epsilon = f64::EPSILON;
        assert_eq!(
            orient2d([[0.0, 0.0], [1.0, epsilon], [2.0, 0.0]]).unwrap(),
            PredicateSign::Negative
        );
        assert_eq!(
            incircle2d([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]).unwrap(),
            PredicateSign::Positive
        );
    }

    #[test]
    fn symbolic_ties_are_alternating_and_never_zero() {
        let point = |identity, coordinates| PlanarPredicatePoint {
            identity: StableDigest::from_bytes([identity; 32]),
            coordinates,
        };
        let a = point(1, [0.0, 0.0]);
        let b = point(2, [1.0, 0.0]);
        let c = point(3, [2.0, 0.0]);
        assert_eq!(
            orient2d_symbolic([a, b, c]).unwrap(),
            PredicateSign::Positive
        );
        assert_eq!(
            orient2d_symbolic([b, a, c]).unwrap(),
            PredicateSign::Negative
        );

        let d = point(4, [0.0, 1.0]);
        let e = point(5, [1.0, 0.0]);
        let f = point(6, [0.0, -1.0]);
        let g = point(7, [-1.0, 0.0]);
        assert_eq!(
            incircle2d_symbolic([d, e, f, g]).unwrap(),
            PredicateSign::Positive
        );
        assert_eq!(
            incircle2d_symbolic([e, d, f, g]).unwrap(),
            PredicateSign::Negative
        );
    }

    #[test]
    fn symbolic_predicates_reject_ambiguous_identity_and_invalid_coordinates() {
        let identity = StableDigest::from_bytes([1; 32]);
        let duplicate = PlanarPredicatePoint {
            identity,
            coordinates: [0.0, 0.0],
        };
        assert_eq!(
            orient2d_symbolic([duplicate; 3]),
            Err(PlanarPredicateError::DuplicateIdentity {
                left_index: 0,
                right_index: 1,
            })
        );
        assert_eq!(
            orient2d([[0.0, 0.0], [f64::NAN, 0.0], [1.0, 1.0]]),
            Err(PlanarPredicateError::NonFiniteCoordinate {
                point_index: 1,
                axis: 0,
            })
        );
    }
}
