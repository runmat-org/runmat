use super::{
    boundary_tensor, GAUSS_LEGENDRE_EIGHT_UNIT, GAUSS_LEGENDRE_SIXTEEN_UNIT,
    GAUSS_LEGENDRE_TWELVE_UNIT,
};
use crate::topology::{
    contract::BoundaryQuadraturePoint,
    element::{norm, subtract},
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

pub(super) fn converged_points(coordinates: &[[f64; 3]]) -> Option<Vec<BoundaryQuadraturePoint>> {
    let (eight, _) = integrate(coordinates, &GAUSS_LEGENDRE_EIGHT_UNIT)?;
    let (twelve, twelve_points) = integrate(coordinates, &GAUSS_LEGENDRE_TWELVE_UNIT)?;
    if moments_converged(&eight, &twelve, coordinates) {
        return Some(twelve_points);
    }
    let (sixteen, sixteen_points) = integrate(coordinates, &GAUSS_LEGENDRE_SIXTEEN_UNIT)?;
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
    rule: &[(f64, f64)],
) -> Option<(BoundaryMoments, Vec<BoundaryQuadraturePoint>)> {
    let mut moments = BoundaryMoments::default();
    let mut points = Vec::with_capacity(rule.len() * rule.len());
    boundary_tensor::visit(BoundaryTriangleOrder::Tri6, coordinates, rule, |point| {
        moments.area += point.area_weight_m2;
        for (sum, shape) in moments.nodal_area.iter_mut().zip(point.shape_values) {
            *sum += shape * point.area_weight_m2;
        }
        for axis in 0..3 {
            moments.first_moment[axis] += point.position_m[axis] * point.area_weight_m2;
            moments.area_vector[axis] += point.area_vector_weight_m2[axis];
        }
        points.push(point);
    })?;
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
