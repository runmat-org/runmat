//! Exact comparison of piecewise affine parabolic lifting over tetrahedra.
//! The parabolic functional is minimized by Delaunay triangulations:
//! O. R. Musin, https://arxiv.org/abs/math/0312050.
use super::{
    adaptive_spatial::determinant3, ExactDyadic, PredicateSign, SpatialPredicateError, Tetrahedron3,
};

/// Sign of the created minus removed parabolic lifting integral.
///
/// The integral is `sum(volume(t) * sum(norm_squared(vertex))) / 4`.
/// Its positive factor 1/24 is omitted here. For reconnections of the same
/// embedded cavity, a negative change establishes strict geometric progress
/// toward a Delaunay triangulation, independently of the violation count.
/// Callers must establish equal domain, vertex inventory and admissibility;
/// this scalar comparison establishes none of those contracts.
pub fn tetrahedral_lifting_change(
    removed: &[Tetrahedron3],
    created: &[Tetrahedron3],
) -> Result<PredicateSign, SpatialPredicateError> {
    Ok(potential(created)?
        .subtract_exact(potential(removed)?)
        .sign())
}

fn potential(tetrahedra: &[Tetrahedron3]) -> Result<ExactDyadic, SpatialPredicateError> {
    let mut total = ExactDyadic::zero();
    for tetrahedron in tetrahedra {
        for (point_index, point) in tetrahedron.iter().enumerate() {
            for (axis, value) in point.iter().enumerate() {
                if !value.is_finite() {
                    return Err(SpatialPredicateError::NonFiniteCoordinate { point_index, axis });
                }
            }
        }
        let points = tetrahedron.map(|point| point.map(ExactDyadic::from_f64));
        let mut determinant = determinant3(std::array::from_fn(|row| {
            std::array::from_fn(|axis| {
                points[row][axis]
                    .clone()
                    .subtract_exact(points[3][axis].clone())
            })
        }));
        if determinant.sign() == PredicateSign::Negative {
            determinant = ExactDyadic::zero().subtract_exact(determinant);
        }
        let lifted_sum = points
            .into_iter()
            .flatten()
            .fold(ExactDyadic::zero(), |sum, coordinate| {
                sum.add_exact(coordinate.clone().multiply_exact(coordinate))
            });
        total = total.add_exact(determinant.multiply_exact(lifted_sum));
    }
    Ok(total)
}

#[cfg(test)]
mod tests;
