mod projection;
pub use projection::{
    projected_orient2d_constructed, triangle_projection_axes, triangle_projection_axes_constructed,
};
use robust::Coord3D;
use serde::{Deserialize, Serialize};

use crate::StableDigest;

use super::{
    adaptive_planar::{permutation_is_odd, PredicateSign},
    exact_dyadic::ExactDyadic,
};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Point3Construction {
    Explicit,
    Segment {
        endpoints: [[f64; 3]; 2],
        parameter_numerator: u64,
        parameter_exponent: u8,
    },
    TriangleDyadic {
        vertices: [[f64; 3]; 3],
        weights_numerator: [u64; 3],
        weight_exponent: u8,
    },
    /// A flattened dyadic affine construction over at most four explicit source vertices.
    /// Flattening makes repeated segment and facet refinement exact without a recursive
    /// construction tree.
    SimplexDyadic {
        vertices: [[f64; 3]; 4],
        weights_numerator: [u64; 4],
        vertex_count: u8,
        weight_exponent: u8,
    },
    TetrahedronCentroid {
        vertices: [[f64; 3]; 4],
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpatialPredicatePoint {
    pub identity: StableDigest,
    pub coordinates: [f64; 3],
    pub construction: Point3Construction,
}

/// Composes a dyadic affine combination of already constructed points into one flat exact
/// construction. Meshing constructions originate on edges, triangles, or tetrahedra, so four
/// explicit source vertices are sufficient; incompatible constructions return `None`.
pub fn compose_dyadic_point<const N: usize>(
    points: [SpatialPredicatePoint; N],
    weights_numerator: [u64; N],
    weight_exponent: u8,
) -> Option<([f64; 3], Point3Construction)> {
    if weight_exponent >= 63
        || points
            .iter()
            .flat_map(|point| point.coordinates)
            .any(|coordinate| !coordinate.is_finite())
        || weights_numerator
            .into_iter()
            .try_fold(0_u64, u64::checked_add)
            != Some(1_u64 << weight_exponent)
    {
        return None;
    }
    let coordinates = std::array::from_fn(|axis| {
        points
            .iter()
            .zip(weights_numerator)
            .map(|(point, weight)| point.coordinates[axis] * weight as f64)
            .sum::<f64>()
            / (1_u64 << weight_exponent) as f64
    });
    if coordinates.iter().any(|coordinate| !coordinate.is_finite()) {
        return None;
    }
    let mut terms = Vec::<([f64; 3], u128, u8)>::new();
    for (point, outer_weight) in points.into_iter().zip(weights_numerator) {
        for (vertex, numerator, exponent) in affine_terms(point)? {
            let numerator = u128::from(numerator).checked_mul(u128::from(outer_weight))?;
            terms.push((vertex, numerator, exponent.checked_add(weight_exponent)?));
        }
    }
    let common_exponent = terms.iter().map(|(_, _, exponent)| *exponent).max()?;
    if common_exponent >= 127 {
        return None;
    }
    let mut merged = Vec::<([f64; 3], u128)>::new();
    for (vertex, numerator, exponent) in terms {
        let numerator = numerator.checked_shl(u32::from(common_exponent - exponent))?;
        if let Some((_, existing)) = merged
            .iter_mut()
            .find(|(candidate, _)| *candidate == vertex)
        {
            *existing = existing.checked_add(numerator)?;
        } else {
            merged.push((vertex, numerator));
        }
    }
    merged.retain(|(_, numerator)| *numerator != 0);
    merged.sort_by(|left, right| coordinate_order(left.0, right.0));
    if merged.is_empty() || merged.len() > 4 {
        return None;
    }
    let common_shift = merged
        .iter()
        .map(|(_, numerator)| numerator.trailing_zeros())
        .min()
        .unwrap_or(0)
        .min(u32::from(common_exponent));
    let exponent = common_exponent - common_shift as u8;
    if exponent >= 63 {
        return None;
    }
    let mut vertices = [[0.0; 3]; 4];
    let mut weights = [0_u64; 4];
    for (index, (vertex, numerator)) in merged.iter().enumerate() {
        vertices[index] = *vertex;
        weights[index] = u64::try_from(*numerator >> common_shift).ok()?;
    }
    if weights.into_iter().try_fold(0_u64, u64::checked_add) != Some(1_u64 << exponent) {
        return None;
    }
    Some((
        coordinates,
        Point3Construction::SimplexDyadic {
            vertices,
            weights_numerator: weights,
            vertex_count: merged.len() as u8,
            weight_exponent: exponent,
        },
    ))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpatialPredicateError {
    InvalidProjectionAxes {
        axes: [usize; 2],
    },
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
    InvalidConstruction {
        point_index: usize,
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

/// Exact orientation for explicit points and dyadic points constructed on segments.
pub fn orient3d_constructed(
    points: [SpatialPredicatePoint; 4],
) -> Result<PredicateSign, SpatialPredicateError> {
    validate_points(&points)?;
    if points
        .iter()
        .all(|point| point.construction == Point3Construction::Explicit)
    {
        return orient3d(points.map(|point| point.coordinates));
    }
    let [a, b, c, d] = points.map(exact_coordinates);
    let [adx, ady, adz] = subtract_exact(a, d.clone());
    let [bdx, bdy, bdz] = subtract_exact(b, d.clone());
    let [cdx, cdy, cdz] = subtract_exact(c, d);
    Ok(adz
        .multiply_exact(
            bdx.clone()
                .multiply_exact(cdy.clone())
                .subtract_exact(cdx.clone().multiply_exact(bdy.clone())),
        )
        .add_exact(
            bdz.multiply_exact(
                cdx.multiply_exact(ady.clone())
                    .subtract_exact(adx.clone().multiply_exact(cdy)),
            ),
        )
        .add_exact(
            cdz.multiply_exact(
                adx.multiply_exact(bdy)
                    .subtract_exact(bdx.multiply_exact(ady)),
            ),
        )
        .sign())
}

/// Exact in-sphere classification for explicit and dyadic segment-constructed points.
pub fn insphere3d_constructed(
    points: [SpatialPredicatePoint; 5],
) -> Result<PredicateSign, SpatialPredicateError> {
    validate_points(&points)?;
    if points
        .iter()
        .all(|point| point.construction == Point3Construction::Explicit)
    {
        return insphere3d(points.map(|point| point.coordinates));
    }
    let [a, b, c, d, e] = points.map(exact_coordinates);
    let rows = [a, b, c, d].map(|point| {
        let delta = subtract_exact(point, e.clone());
        let norm = delta
            .iter()
            .cloned()
            .map(|value| value.clone().multiply_exact(value))
            .reduce(ExactDyadic::add_exact)
            .unwrap_or_else(ExactDyadic::zero);
        [delta[0].clone(), delta[1].clone(), delta[2].clone(), norm]
    });
    Ok(determinant4(rows).sign())
}

/// Tetrahedron orientation with a stable alternating tie-break for distinct identities.
pub fn orient3d_symbolic(
    points: [SpatialPredicatePoint; 4],
) -> Result<PredicateSign, SpatialPredicateError> {
    validate_points(&points)?;
    match orient3d_constructed(points)? {
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
    match insphere3d_constructed(points)? {
        PredicateSign::Zero => Ok(PredicateSign::from_odd_permutation(permutation_is_odd(
            points.map(|point| point.identity),
        ))),
        sign => Ok(sign),
    }
}

/// Exact alignment of two constructed triangle normals. Positive means the triangles have the
/// same orientation, negative means opposite orientation, and zero means at least one triangle is
/// degenerate or their normals are orthogonal.
pub fn triangle_normal_alignment_constructed(
    reference: [SpatialPredicatePoint; 3],
    triangle: [SpatialPredicatePoint; 3],
) -> Result<PredicateSign, SpatialPredicateError> {
    validate_points(&reference)?;
    validate_points(&triangle)?;
    let [reference_first, reference_second, reference_third] = reference.map(exact_coordinates);
    let [triangle_first, triangle_second, triangle_third] = triangle.map(exact_coordinates);
    let reference_normal = cross_exact(
        subtract_exact(reference_second, reference_first.clone()),
        subtract_exact(reference_third, reference_first),
    );
    let triangle_normal = cross_exact(
        subtract_exact(triangle_second, triangle_first.clone()),
        subtract_exact(triangle_third, triangle_first),
    );
    Ok(reference_normal
        .into_iter()
        .zip(triangle_normal)
        .map(|(left, right)| left.multiply_exact(right))
        .reduce(ExactDyadic::add_exact)
        .unwrap_or_else(ExactDyadic::zero)
        .sign())
}

/// Exact collinearity for explicit and supported constructed 3D points.
pub fn points_collinear_constructed(
    points: [SpatialPredicatePoint; 3],
) -> Result<bool, SpatialPredicateError> {
    validate_points(&points)?;
    let [first, second, third] = points.map(exact_coordinates);
    Ok(cross_exact(
        subtract_exact(second, first.clone()),
        subtract_exact(third, first),
    )
    .into_iter()
    .all(|component| component.sign() == PredicateSign::Zero))
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

pub(super) fn validate_points<const N: usize>(
    points: &[SpatialPredicatePoint; N],
) -> Result<(), SpatialPredicateError> {
    validate_coordinates(&points.map(|point| point.coordinates))?;
    for (point_index, point) in points.iter().enumerate() {
        if let Point3Construction::Segment {
            endpoints,
            parameter_numerator,
            parameter_exponent,
        } = point.construction
        {
            validate_coordinates(&endpoints)?;
            if parameter_exponent >= 63 || parameter_numerator > (1_u64 << parameter_exponent) {
                return Err(SpatialPredicateError::InvalidConstruction { point_index });
            }
        }
        if let Point3Construction::TetrahedronCentroid { vertices } = point.construction {
            validate_coordinates(&vertices)?;
        }
        if let Point3Construction::TriangleDyadic {
            vertices,
            weights_numerator,
            weight_exponent,
        } = point.construction
        {
            validate_coordinates(&vertices)?;
            if weight_exponent >= 63
                || weights_numerator
                    .into_iter()
                    .try_fold(0_u64, u64::checked_add)
                    != Some(1_u64 << weight_exponent)
                || weights_numerator.contains(&0)
            {
                return Err(SpatialPredicateError::InvalidConstruction { point_index });
            }
        }
        if let Point3Construction::SimplexDyadic {
            vertices,
            weights_numerator,
            vertex_count,
            weight_exponent,
        } = point.construction
        {
            let active = usize::from(vertex_count);
            if active == 0
                || active > vertices.len()
                || vertices[..active]
                    .iter()
                    .flatten()
                    .any(|coordinate| !coordinate.is_finite())
                || weight_exponent >= 63
                || weights_numerator[..active]
                    .iter()
                    .copied()
                    .try_fold(0_u64, u64::checked_add)
                    != Some(1_u64 << weight_exponent)
                || weights_numerator[active..]
                    .iter()
                    .any(|weight| *weight != 0)
            {
                return Err(SpatialPredicateError::InvalidConstruction { point_index });
            }
        }
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

pub(super) fn exact_coordinates(point: SpatialPredicatePoint) -> [ExactDyadic; 3] {
    match point.construction {
        Point3Construction::Explicit => point.coordinates.map(ExactDyadic::from_f64),
        Point3Construction::Segment {
            endpoints,
            parameter_numerator,
            parameter_exponent,
        } => std::array::from_fn(|axis| {
            ExactDyadic::linear(
                endpoints[0][axis],
                endpoints[1][axis],
                parameter_numerator,
                parameter_exponent,
            )
        }),
        Point3Construction::TriangleDyadic {
            vertices,
            weights_numerator,
            weight_exponent,
        } => std::array::from_fn(|axis| {
            vertices
                .into_iter()
                .zip(weights_numerator)
                .map(|(vertex, weight)| {
                    ExactDyadic::from_f64(vertex[axis]).multiply_integer(weight)
                })
                .reduce(ExactDyadic::add_exact)
                .unwrap_or_else(ExactDyadic::zero)
                .divide_by_power_of_two(weight_exponent)
        }),
        Point3Construction::SimplexDyadic {
            vertices,
            weights_numerator,
            vertex_count,
            weight_exponent,
        } => std::array::from_fn(|axis| {
            vertices[..usize::from(vertex_count)]
                .iter()
                .zip(weights_numerator)
                .map(|(vertex, weight)| {
                    ExactDyadic::from_f64(vertex[axis]).multiply_integer(weight)
                })
                .reduce(ExactDyadic::add_exact)
                .unwrap_or_else(ExactDyadic::zero)
                .divide_by_power_of_two(weight_exponent)
        }),
        Point3Construction::TetrahedronCentroid { vertices } => std::array::from_fn(|axis| {
            vertices
                .into_iter()
                .map(|vertex| ExactDyadic::from_f64(vertex[axis]))
                .reduce(ExactDyadic::add_exact)
                .unwrap_or_else(ExactDyadic::zero)
                .divide_by_power_of_two(2)
        }),
    }
}

pub(super) fn affine_terms(point: SpatialPredicatePoint) -> Option<Vec<([f64; 3], u64, u8)>> {
    Some(match point.construction {
        Point3Construction::Explicit => vec![(point.coordinates, 1, 0)],
        Point3Construction::Segment {
            endpoints,
            parameter_numerator,
            parameter_exponent,
        } => {
            if endpoints
                .iter()
                .flatten()
                .any(|coordinate| !coordinate.is_finite())
            {
                return None;
            }
            let denominator = 1_u64.checked_shl(u32::from(parameter_exponent))?;
            vec![
                (
                    endpoints[0],
                    denominator.checked_sub(parameter_numerator)?,
                    parameter_exponent,
                ),
                (endpoints[1], parameter_numerator, parameter_exponent),
            ]
        }
        Point3Construction::TriangleDyadic {
            vertices,
            weights_numerator,
            weight_exponent,
        } => {
            if vertices
                .iter()
                .flatten()
                .any(|coordinate| !coordinate.is_finite())
                || weight_exponent >= 63
                || weights_numerator
                    .into_iter()
                    .try_fold(0_u64, u64::checked_add)
                    != Some(1_u64 << weight_exponent)
            {
                return None;
            }
            vertices
                .into_iter()
                .zip(weights_numerator)
                .map(|(vertex, weight)| (vertex, weight, weight_exponent))
                .collect()
        }
        Point3Construction::SimplexDyadic {
            vertices,
            weights_numerator,
            vertex_count,
            weight_exponent,
        } => {
            let active = usize::from(vertex_count);
            if active == 0
                || active > vertices.len()
                || vertices[..active]
                    .iter()
                    .flatten()
                    .any(|coordinate| !coordinate.is_finite())
                || weight_exponent >= 63
                || weights_numerator[..active]
                    .iter()
                    .copied()
                    .try_fold(0_u64, u64::checked_add)
                    != Some(1_u64 << weight_exponent)
                || weights_numerator[active..]
                    .iter()
                    .any(|weight| *weight != 0)
            {
                return None;
            }
            vertices[..active]
                .iter()
                .copied()
                .zip(weights_numerator)
                .map(|(vertex, weight)| (vertex, weight, weight_exponent))
                .collect()
        }
        Point3Construction::TetrahedronCentroid { vertices } => {
            if vertices
                .iter()
                .flatten()
                .any(|coordinate| !coordinate.is_finite())
            {
                return None;
            }
            vertices.into_iter().map(|vertex| (vertex, 1, 2)).collect()
        }
    })
}

fn coordinate_order(left: [f64; 3], right: [f64; 3]) -> std::cmp::Ordering {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| left.total_cmp(&right))
        .find(|ordering| *ordering != std::cmp::Ordering::Equal)
        .unwrap_or(std::cmp::Ordering::Equal)
}

fn subtract_exact(left: [ExactDyadic; 3], right: [ExactDyadic; 3]) -> [ExactDyadic; 3] {
    let mut left = left.into_iter();
    let mut right = right.into_iter();
    std::array::from_fn(|_| {
        left.next()
            .expect("three coordinates")
            .subtract_exact(right.next().expect("three coordinates"))
    })
}

fn cross_exact(left: [ExactDyadic; 3], right: [ExactDyadic; 3]) -> [ExactDyadic; 3] {
    [
        left[1]
            .clone()
            .multiply_exact(right[2].clone())
            .subtract_exact(left[2].clone().multiply_exact(right[1].clone())),
        left[2]
            .clone()
            .multiply_exact(right[0].clone())
            .subtract_exact(left[0].clone().multiply_exact(right[2].clone())),
        left[0]
            .clone()
            .multiply_exact(right[1].clone())
            .subtract_exact(left[1].clone().multiply_exact(right[0].clone())),
    ]
}

pub(super) fn determinant3(matrix: [[ExactDyadic; 3]; 3]) -> ExactDyadic {
    matrix[0][0]
        .clone()
        .multiply_exact(
            matrix[1][1]
                .clone()
                .multiply_exact(matrix[2][2].clone())
                .subtract_exact(matrix[1][2].clone().multiply_exact(matrix[2][1].clone())),
        )
        .subtract_exact(
            matrix[0][1].clone().multiply_exact(
                matrix[1][0]
                    .clone()
                    .multiply_exact(matrix[2][2].clone())
                    .subtract_exact(matrix[1][2].clone().multiply_exact(matrix[2][0].clone())),
            ),
        )
        .add_exact(
            matrix[0][2].clone().multiply_exact(
                matrix[1][0]
                    .clone()
                    .multiply_exact(matrix[2][1].clone())
                    .subtract_exact(matrix[1][1].clone().multiply_exact(matrix[2][0].clone())),
            ),
        )
}

fn determinant4(matrix: [[ExactDyadic; 4]; 4]) -> ExactDyadic {
    (0..4)
        .map(|column| {
            let minor = std::array::from_fn(|minor_row| {
                std::array::from_fn(|minor_column| {
                    let source_column = if minor_column < column {
                        minor_column
                    } else {
                        minor_column + 1
                    };
                    matrix[minor_row + 1][source_column].clone()
                })
            });
            let term = matrix[0][column]
                .clone()
                .multiply_exact(determinant3(minor));
            if column % 2 == 0 {
                term
            } else {
                ExactDyadic::zero().subtract_exact(term)
            }
        })
        .reduce(ExactDyadic::add_exact)
        .unwrap_or_else(ExactDyadic::zero)
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
    fn composed_dyadic_point_flattens_nested_facet_construction_exactly() {
        let vertices = [
            [2.380896857755243, 0.7533057313016712, 0.03495121082094581],
            [2.38085681630771, 0.7534792408036042, 0.034809126648409974],
            [2.380885189085703, 0.7533447406117164, 0.03493475896938903],
        ];
        let explicit = |identity: u8, coordinates| SpatialPredicatePoint {
            identity: StableDigest::from_bytes([identity; 32]),
            coordinates,
            construction: Point3Construction::Explicit,
        };
        let first_coordinates = std::array::from_fn(|axis| {
            (2.0 * vertices[0][axis] + vertices[1][axis] + vertices[2][axis]) * 0.25
        });
        let first = SpatialPredicatePoint {
            identity: StableDigest::from_bytes([4; 32]),
            coordinates: first_coordinates,
            construction: Point3Construction::TriangleDyadic {
                vertices,
                weights_numerator: [2, 1, 1],
                weight_exponent: 2,
            },
        };
        let (coordinates, construction) =
            compose_dyadic_point([first, explicit(2, vertices[1])], [1, 1], 1).unwrap();
        let composed = SpatialPredicatePoint {
            identity: StableDigest::from_bytes([5; 32]),
            coordinates,
            construction,
        };

        assert_eq!(
            orient3d_constructed([
                explicit(1, vertices[0]),
                explicit(2, vertices[1]),
                composed,
                explicit(3, vertices[2]),
            ])
            .unwrap(),
            PredicateSign::Zero
        );
        assert!(matches!(
            construction,
            Point3Construction::SimplexDyadic {
                vertex_count: 3,
                ..
            }
        ));
    }

    #[test]
    fn constructed_triangle_normal_alignment_is_exact_and_orientation_sensitive() {
        let explicit = |identity, coordinates| SpatialPredicatePoint {
            identity: StableDigest::from_bytes([identity; 32]),
            coordinates,
            construction: Point3Construction::Explicit,
        };
        let reference = [
            explicit(1, [0.0, 0.0, 0.0]),
            explicit(2, [1.0, 0.0, 0.0]),
            explicit(3, [0.0, 1.0, 0.0]),
        ];
        let midpoint = SpatialPredicatePoint {
            identity: StableDigest::from_bytes([4; 32]),
            coordinates: [0.5, 0.0, 0.0],
            construction: Point3Construction::Segment {
                endpoints: [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                parameter_numerator: 1,
                parameter_exponent: 1,
            },
        };
        let aligned = [midpoint, reference[1], reference[2]];
        assert_eq!(
            triangle_normal_alignment_constructed(reference, aligned).unwrap(),
            PredicateSign::Positive
        );
        assert_eq!(
            triangle_normal_alignment_constructed(reference, [aligned[1], aligned[0], aligned[2]],)
                .unwrap(),
            PredicateSign::Negative
        );
        assert!(points_collinear_constructed([reference[0], midpoint, reference[1]]).unwrap());
        assert!(!points_collinear_constructed([reference[0], midpoint, reference[2]]).unwrap());
    }

    #[test]
    fn dyadic_triangle_point_remains_exactly_in_its_support_plane() {
        let vertices = [
            [2.380896857755243, 0.7533057313016712, 0.03495121082094581],
            [2.38085681630771, 0.7534792408036042, 0.034809126648409974],
            [2.380885189085703, 0.7533447406117164, 0.03493475896938903],
        ];
        let explicit = |identity, coordinates| SpatialPredicatePoint {
            identity: StableDigest::from_bytes([identity; 32]),
            coordinates,
            construction: Point3Construction::Explicit,
        };
        let interior = SpatialPredicatePoint {
            identity: StableDigest::from_bytes([4; 32]),
            coordinates: std::array::from_fn(|axis| {
                (2.0 * vertices[0][axis] + vertices[1][axis] + vertices[2][axis]) * 0.25
            }),
            construction: Point3Construction::TriangleDyadic {
                vertices,
                weights_numerator: [2, 1, 1],
                weight_exponent: 2,
            },
        };

        assert_eq!(
            orient3d_constructed([
                explicit(1, vertices[0]),
                explicit(2, vertices[1]),
                explicit(3, vertices[2]),
                interior,
            ])
            .unwrap(),
            PredicateSign::Zero
        );
    }

    #[test]
    fn symbolic_spatial_ties_are_alternating_and_never_zero() {
        let point = |identity, coordinates| SpatialPredicatePoint {
            identity: StableDigest::from_bytes([identity; 32]),
            coordinates,
            construction: Point3Construction::Explicit,
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
            construction: Point3Construction::Explicit,
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
