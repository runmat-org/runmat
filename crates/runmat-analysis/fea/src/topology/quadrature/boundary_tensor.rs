//! Explicit high-order face integration. Geometry validity is checked over the
//! whole quadratic triangle, not merely at integration points. The caller must
//! establish convergence of its actual integrand on curved faces.
use super::{boundary_shape, weighted_derivative, weighted_position};
use crate::topology::{
    contract::BoundaryQuadraturePoint,
    element::{cross, norm},
};
use runmat_meshing_core::BoundaryTriangleOrder;

pub(crate) fn visit(
    order: BoundaryTriangleOrder,
    coordinates: &[[f64; 3]],
    rule: &[(f64, f64)],
    mut visit: impl FnMut(BoundaryQuadraturePoint),
) -> Option<()> {
    let node_count = match order {
        BoundaryTriangleOrder::Tri3 => 3,
        BoundaryTriangleOrder::Tri6 => 6,
    };
    if coordinates.len() != node_count {
        return None;
    }
    if !runmat_meshing_core::is_valid_boundary_face(order, coordinates) {
        return None;
    }
    for &(r, wr) in rule {
        for &(s, ws) in rule {
            let barycentric = [(1.0 - r) * (1.0 - s), r, (1.0 - r) * s];
            let (shape_values, derivatives) = boundary_shape(order, barycentric);
            let position_m = weighted_position(coordinates, &shape_values[..node_count]);
            let tangent_u = weighted_derivative(coordinates, &derivatives[..node_count], 0);
            let tangent_v = weighted_derivative(coordinates, &derivatives[..node_count], 1);
            let area_vector = cross(tangent_u, tangent_v);
            let jacobian = norm(area_vector);
            let weight = wr * ws * (1.0 - r);
            visit(BoundaryQuadraturePoint {
                reference_barycentric: barycentric,
                reference_weight: weight,
                position_m,
                area_vector_weight_m2: area_vector.map(|value| value * weight),
                area_weight_m2: jacobian * weight,
                shape_values,
                node_count,
            });
        }
    }
    Some(())
}
