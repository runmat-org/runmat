use robust::Coord3D;

use crate::StableDigest;

use super::adaptive_planar::{permutation_is_odd, PredicateSign};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpatialPredicatePoint {
    pub identity: StableDigest,
    pub coordinates: [f64; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpatialPredicateError {
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

/// Adaptive-precision tetrahedron orientation without a tolerance or tie-break.
pub fn orient3d(points: [[f64; 3]; 4]) -> Result<PredicateSign, SpatialPredicateError> {
    validate_coordinates(&points)?;
    Ok(PredicateSign::from_value(robust::orient3d(
        coord(points[0]),
        coord(points[1]),
        coord(points[2]),
        coord(points[3]),
    )))
}

/// Adaptive-precision in-sphere test. Positive means inside when the first four
/// points have positive [`orient3d`] orientation.
pub fn insphere3d(points: [[f64; 3]; 5]) -> Result<PredicateSign, SpatialPredicateError> {
    validate_coordinates(&points)?;
    Ok(PredicateSign::from_value(robust::insphere(
        coord(points[0]),
        coord(points[1]),
        coord(points[2]),
        coord(points[3]),
        coord(points[4]),
    )))
}

/// Tetrahedron orientation with a stable alternating tie-break for distinct identities.
pub fn orient3d_symbolic(
    points: [SpatialPredicatePoint; 4],
) -> Result<PredicateSign, SpatialPredicateError> {
    validate_points(&points)?;
    match orient3d(points.map(|point| point.coordinates))? {
        PredicateSign::Zero => Ok(PredicateSign::from_odd_permutation(permutation_is_odd(
            points.map(|point| point.identity),
        ))),
        sign => Ok(sign),
    }
}

/// In-sphere test with a stable alternating tie-break for cospherical points.
pub fn insphere3d_symbolic(
    points: [SpatialPredicatePoint; 5],
) -> Result<PredicateSign, SpatialPredicateError> {
    validate_points(&points)?;
    match insphere3d(points.map(|point| point.coordinates))? {
        PredicateSign::Zero => Ok(PredicateSign::from_odd_permutation(permutation_is_odd(
            points.map(|point| point.identity),
        ))),
        sign => Ok(sign),
    }
}

fn coord(point: [f64; 3]) -> Coord3D<f64> {
    Coord3D {
        x: point[0],
        y: point[1],
        z: point[2],
    }
}

fn validate_coordinates<const N: usize>(
    points: &[[f64; 3]; N],
) -> Result<(), SpatialPredicateError> {
    for (point_index, point) in points.iter().enumerate() {
        for (axis, value) in point.iter().enumerate() {
            if !value.is_finite() {
                return Err(SpatialPredicateError::NonFiniteCoordinate { point_index, axis });
            }
        }
    }
    Ok(())
}

fn validate_points<const N: usize>(
    points: &[SpatialPredicatePoint; N],
) -> Result<(), SpatialPredicateError> {
    validate_coordinates(&points.map(|point| point.coordinates))?;
    for (point_index, point) in points.iter().enumerate() {
        if point.identity == StableDigest::ZERO {
            return Err(SpatialPredicateError::ZeroIdentity { point_index });
        }
        if let Some(left_index) = points[..point_index]
            .iter()
            .position(|left| left.identity == point.identity)
        {
            return Err(SpatialPredicateError::DuplicateIdentity {
                left_index,
                right_index: point_index,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_spatial_predicates_resolve_near_degenerate_inputs() {
        let tetrahedron = [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, f64::EPSILON],
        ];
        assert_eq!(orient3d(tetrahedron).unwrap(), PredicateSign::Positive);

        let enclosing = [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.25, 0.25, 0.25],
        ];
        assert_eq!(insphere3d(enclosing).unwrap(), PredicateSign::Positive);
    }

    #[test]
    fn symbolic_spatial_ties_are_alternating_and_never_zero() {
        let point = |identity, coordinates| SpatialPredicatePoint {
            identity: StableDigest::from_bytes([identity; 32]),
            coordinates,
        };
        let coplanar = [
            point(1, [0.0, 0.0, 0.0]),
            point(2, [1.0, 0.0, 0.0]),
            point(3, [0.0, 1.0, 0.0]),
            point(4, [1.0, 1.0, 0.0]),
        ];
        assert_eq!(
            orient3d_symbolic(coplanar).unwrap(),
            PredicateSign::Positive
        );
        assert_eq!(
            orient3d_symbolic([coplanar[1], coplanar[0], coplanar[2], coplanar[3]]).unwrap(),
            PredicateSign::Negative
        );

        let cospherical = [
            point(5, [1.0, 0.0, 0.0]),
            point(6, [0.0, 1.0, 0.0]),
            point(7, [0.0, 0.0, 1.0]),
            point(8, [-1.0, 0.0, 0.0]),
            point(9, [0.0, -1.0, 0.0]),
        ];
        assert_eq!(
            insphere3d_symbolic(cospherical).unwrap(),
            PredicateSign::Positive
        );
        assert_eq!(
            insphere3d_symbolic([
                cospherical[1],
                cospherical[0],
                cospherical[2],
                cospherical[3],
                cospherical[4],
            ])
            .unwrap(),
            PredicateSign::Negative
        );
    }

    #[test]
    fn spatial_predicates_reject_invalid_coordinates_and_identities() {
        assert_eq!(
            orient3d([
                [0.0; 3],
                [1.0, 0.0, 0.0],
                [0.0, f64::INFINITY, 0.0],
                [0.0, 0.0, 1.0],
            ]),
            Err(SpatialPredicateError::NonFiniteCoordinate {
                point_index: 2,
                axis: 1,
            })
        );
        let duplicate = SpatialPredicatePoint {
            identity: StableDigest::from_bytes([1; 32]),
            coordinates: [0.0; 3],
        };
        assert_eq!(
            orient3d_symbolic([duplicate; 4]),
            Err(SpatialPredicateError::DuplicateIdentity {
                left_index: 0,
                right_index: 1,
            })
        );
    }
}
