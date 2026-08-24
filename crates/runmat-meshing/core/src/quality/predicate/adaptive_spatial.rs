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

    fn exact_sign(value: i128) -> PredicateSign {
        match value.cmp(&0) {
            std::cmp::Ordering::Less => PredicateSign::Negative,
            std::cmp::Ordering::Equal => PredicateSign::Zero,
            std::cmp::Ordering::Greater => PredicateSign::Positive,
        }
    }

    fn exact_orient3d(points: [[i64; 3]; 4]) -> PredicateSign {
        let [a, b, c, d] = points.map(|point| point.map(i128::from));
        let [adx, ady, adz] = subtract(a, d);
        let [bdx, bdy, bdz] = subtract(b, d);
        let [cdx, cdy, cdz] = subtract(c, d);
        exact_sign(
            adz * (bdx * cdy - cdx * bdy)
                + bdz * (cdx * ady - adx * cdy)
                + cdz * (adx * bdy - bdx * ady),
        )
    }

    fn exact_insphere3d(points: [[i64; 3]; 5]) -> PredicateSign {
        let [a, b, c, d, e] = points.map(|point| point.map(i128::from));
        let rows = [a, b, c, d].map(|point| {
            let delta = subtract(point, e);
            [
                delta[0],
                delta[1],
                delta[2],
                delta.iter().map(|value| value * value).sum(),
            ]
        });
        exact_sign(determinant4(rows))
    }

    fn subtract(left: [i128; 3], right: [i128; 3]) -> [i128; 3] {
        std::array::from_fn(|axis| left[axis] - right[axis])
    }

    fn determinant3(matrix: [[i128; 3]; 3]) -> i128 {
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
            - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
            + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    }

    fn determinant4(matrix: [[i128; 4]; 4]) -> i128 {
        (0..4)
            .map(|column| {
                let minor = std::array::from_fn(|minor_row| {
                    std::array::from_fn(|minor_column| {
                        let source_column = if minor_column < column {
                            minor_column
                        } else {
                            minor_column + 1
                        };
                        matrix[minor_row + 1][source_column]
                    })
                });
                let term = matrix[0][column] * determinant3(minor);
                if column % 2 == 0 {
                    term
                } else {
                    -term
                }
            })
            .sum()
    }

    fn next_coordinate(state: &mut u64) -> i64 {
        *state = state
            .wrapping_mul(2_862_933_555_777_941_757)
            .wrapping_add(3_037_000_493);
        ((*state >> 32) % 401) as i64 - 200
    }

    fn integer_points<const N: usize>(state: &mut u64) -> [[i64; 3]; N] {
        std::array::from_fn(|_| {
            [
                next_coordinate(state),
                next_coordinate(state),
                next_coordinate(state),
            ]
        })
    }

    fn floating<const N: usize>(points: [[i64; 3]; N]) -> [[f64; 3]; N] {
        points.map(|point| point.map(|coordinate| coordinate as f64))
    }

    #[test]
    fn spatial_predicates_match_exact_integer_oracles_and_transform_invariantly() {
        let mut state = 0xd1ff_e2e5_5eed_cafe;
        for _ in 0..10_000 {
            let oriented = integer_points::<4>(&mut state);
            let expected = exact_orient3d(oriented);
            assert_eq!(orient3d(floating(oriented)).unwrap(), expected);
            let transformed =
                oriented.map(|[x, y, z]| [(x + 512) * 8, (y - 1_024) * 8, (z + 2_048) * 8]);
            assert_eq!(orient3d(floating(transformed)).unwrap(), expected);

            let spherical = integer_points::<5>(&mut state);
            let expected = exact_insphere3d(spherical);
            assert_eq!(insphere3d(floating(spherical)).unwrap(), expected);
            let transformed =
                spherical.map(|[x, y, z]| [(x - 256) * 4, (y + 128) * 4, (z - 64) * 4]);
            assert_eq!(insphere3d(floating(transformed)).unwrap(), expected);
        }

        let coplanar = [[0, 0, 0], [2, 0, 0], [0, 3, 0], [1, 1, 0]];
        assert_eq!(orient3d(floating(coplanar)).unwrap(), PredicateSign::Zero);
        let cospherical = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0]];
        assert_eq!(
            insphere3d(floating(cospherical)).unwrap(),
            PredicateSign::Zero
        );
    }

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
