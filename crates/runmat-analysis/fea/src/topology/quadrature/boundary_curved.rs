use super::{
    boundary_shape, weighted_derivative, weighted_position, GAUSS_LEGENDRE_EIGHT_UNIT,
    GAUSS_LEGENDRE_SIXTEEN_UNIT, GAUSS_LEGENDRE_TWELVE_UNIT,
};
use crate::topology::{
    contract::BoundaryQuadraturePoint,
    element::{cross, dot, norm, subtract},
};
use runmat_meshing_core::BoundaryTriangleOrder;

const MOMENT_RELATIVE_TOLERANCE: f64 = 1.0e-10;

pub(super) fn is_curved(coordinates: &[[f64; 3]]) -> bool {
    let scale = (0..3)
        .map(|index| norm(subtract(coordinates[(index + 1) % 3], coordinates[index])))
        .fold(0.0_f64, f64::max);
    (0..3).any(|index| {
        let edge = [(0, 1), (1, 2), (2, 0)][index];
        let midpoint = std::array::from_fn(|axis| {
            0.5 * (coordinates[edge.0][axis] + coordinates[edge.1][axis])
        });
        norm(subtract(coordinates[index + 3], midpoint)) > 1.0e-12 * scale
    })
}

pub(super) fn converged_points(
    coordinates: &[[f64; 3]],
    corner_normal: [f64; 3],
) -> Option<Vec<BoundaryQuadraturePoint>> {
    let (eight, _) = integrate(coordinates, corner_normal, &GAUSS_LEGENDRE_EIGHT_UNIT)?;
    let (twelve, twelve_points) =
        integrate(coordinates, corner_normal, &GAUSS_LEGENDRE_TWELVE_UNIT)?;
    if moments_converged(&eight, &twelve, coordinates) {
        return Some(twelve_points);
    }
    let (sixteen, sixteen_points) =
        integrate(coordinates, corner_normal, &GAUSS_LEGENDRE_SIXTEEN_UNIT)?;
    moments_converged(&twelve, &sixteen, coordinates).then_some(sixteen_points)
}

#[derive(Default)]
struct BoundaryMoments {
    area: f64,
    nodal_area: [f64; 6],
    first_moment: [f64; 3],
    area_vector: [f64; 3],
}

fn integrate(
    coordinates: &[[f64; 3]],
    corner_normal: [f64; 3],
    rule: &[(f64, f64)],
) -> Option<(BoundaryMoments, Vec<BoundaryQuadraturePoint>)> {
    let mut moments = BoundaryMoments::default();
    let mut points = Vec::with_capacity(rule.len() * rule.len());
    for &(r, r_weight) in rule {
        for &(s, s_weight) in rule {
            let barycentric = [(1.0 - r) * (1.0 - s), r, (1.0 - r) * s];
            let (shape_values, derivatives) =
                boundary_shape(BoundaryTriangleOrder::Tri6, barycentric);
            let position_m = weighted_position(coordinates, &shape_values);
            let tangent_r = weighted_derivative(coordinates, &derivatives, 0);
            let tangent_s = weighted_derivative(coordinates, &derivatives, 1);
            let area_vector = cross(tangent_r, tangent_s);
            let jacobian = norm(area_vector);
            if !jacobian.is_finite() || jacobian <= 0.0 || dot(area_vector, corner_normal) <= 0.0 {
                return None;
            }
            let weight = r_weight * s_weight * (1.0 - r);
            let area_weight_m2 = jacobian * weight;
            let area_vector_weight_m2 = area_vector.map(|component| component * weight);
            moments.area += area_weight_m2;
            for (sum, shape) in moments.nodal_area.iter_mut().zip(shape_values) {
                *sum += shape * area_weight_m2;
            }
            for axis in 0..3 {
                moments.first_moment[axis] += position_m[axis] * area_weight_m2;
                moments.area_vector[axis] += area_vector_weight_m2[axis];
            }
            points.push(BoundaryQuadraturePoint {
                position_m,
                area_vector_weight_m2,
                area_weight_m2,
                shape_values,
                node_count: 6,
            });
        }
    }
    moments.area.is_finite().then_some((moments, points))
}

fn moments_converged(
    previous: &BoundaryMoments,
    refined: &BoundaryMoments,
    coordinates: &[[f64; 3]],
) -> bool {
    let area_scale = previous.area.abs().max(refined.area.abs());
    let length_scale = coordinates
        .iter()
        .map(|point| norm(subtract(*point, coordinates[0])))
        .fold(0.0_f64, f64::max);
    let close = |left: f64, right: f64, scale: f64| {
        left.is_finite()
            && right.is_finite()
            && (left - right).abs() <= MOMENT_RELATIVE_TOLERANCE * scale
    };
    close(previous.area, refined.area, area_scale)
        && previous
            .nodal_area
            .iter()
            .zip(refined.nodal_area)
            .all(|(left, right)| close(*left, right, area_scale))
        && previous
            .first_moment
            .iter()
            .zip(refined.first_moment)
            .all(|(left, right)| close(*left, right, area_scale * length_scale))
        && previous
            .area_vector
            .iter()
            .zip(refined.area_vector)
            .all(|(left, right)| close(*left, right, area_scale))
}
