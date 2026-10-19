//! Coordinate projection preserves original affine point construction.
use super::super::{
    adaptive_planar::{orient2d, PlanarPredicateError, PredicateSign},
    exact_dyadic::ExactDyadic,
};
use super::{
    exact_coordinates, validate_coordinates, validate_points, Point3Construction,
    SpatialPredicateError, SpatialPredicatePoint,
};

/// Orientation in a coordinate plane, using the original dyadic construction.
/// Materialized split points must not turn exact collinearity into a rounded corner.
pub fn projected_orient2d_constructed(
    points: [SpatialPredicatePoint; 3],
    axes: [usize; 2],
) -> Result<PredicateSign, SpatialPredicateError> {
    if axes[0] >= 3 || axes[1] >= 3 || axes[0] == axes[1] {
        return Err(SpatialPredicateError::InvalidProjectionAxes { axes });
    }
    validate_points(&points)?;
    projected_sign_validated(points, axes)
}

// Callers validate their own authority: constructed points require identities,
// while coordinate-only physical queries have no persistent point identity.
fn projected_sign_validated(
    points: [SpatialPredicatePoint; 3],
    axes: [usize; 2],
) -> Result<PredicateSign, SpatialPredicateError> {
    if points
        .iter()
        .all(|point| point.construction == Point3Construction::Explicit)
    {
        return orient2d(
            points.map(|point| [point.coordinates[axes[0]], point.coordinates[axes[1]]]),
        )
        .map_err(|error| match error {
            PlanarPredicateError::NonFiniteCoordinate { point_index, axis } => {
                SpatialPredicateError::NonFiniteCoordinate {
                    point_index,
                    axis: axes[axis],
                }
            }
            PlanarPredicateError::ZeroIdentity { point_index } => {
                SpatialPredicateError::ZeroIdentity { point_index }
            }
            PlanarPredicateError::DuplicateIdentity {
                left_index,
                right_index,
            } => SpatialPredicateError::DuplicateIdentity {
                left_index,
                right_index,
            },
        });
    }
    let [a, b, c] = points.map(exact_coordinates);
    let delta =
        |point: &[ExactDyadic; 3], axis: usize| point[axis].clone().subtract_exact(a[axis].clone());
    Ok(delta(&b, axes[0])
        .multiply_exact(delta(&c, axes[1]))
        .subtract_exact(delta(&b, axes[1]).multiply_exact(delta(&c, axes[0])))
        .sign())
}

/// Prefer the largest scaled normal component, then verify that plane with the
/// original point authority. Approximate ranking never decides a predicate sign.
pub fn triangle_projection_axes_constructed(
    points: [SpatialPredicatePoint; 3],
) -> Result<Option<[usize; 2]>, SpatialPredicateError> {
    validate_points(&points)?;
    triangle_projection_axes_validated(points)
}

fn triangle_projection_axes_validated(
    points: [SpatialPredicatePoint; 3],
) -> Result<Option<[usize; 2]>, SpatialPredicateError> {
    let scale = points
        .iter()
        .flat_map(|point| point.coordinates)
        .map(f64::abs)
        .fold(0.0, f64::max)
        .max(f64::MIN_POSITIVE);
    let coordinates = points.map(|point| point.coordinates.map(|value| value / scale));
    let first: [f64; 3] = std::array::from_fn(|axis| coordinates[1][axis] - coordinates[0][axis]);
    let second: [f64; 3] = std::array::from_fn(|axis| coordinates[2][axis] - coordinates[0][axis]);
    let mut candidates = [[0, 1], [0, 2], [1, 2]].map(|axes| {
        let area = first[axes[0]] * second[axes[1]] - first[axes[1]] * second[axes[0]];
        (axes, area.abs())
    });
    candidates.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
    for (axes, _) in candidates {
        if projected_sign_validated(points, axes)? != PredicateSign::Zero {
            return Ok(Some(axes));
        }
    }
    Ok(None)
}

/// The same stable projection for explicitly materialized physical geometry.
pub fn triangle_projection_axes(
    points: [[f64; 3]; 3],
) -> Result<Option<[usize; 2]>, SpatialPredicateError> {
    validate_coordinates(&points)?;
    triangle_projection_axes_validated(points.map(|coordinates| SpatialPredicatePoint {
        identity: crate::StableDigest::ZERO,
        coordinates,
        construction: Point3Construction::Explicit,
    }))
}

#[cfg(test)]
mod tests;
