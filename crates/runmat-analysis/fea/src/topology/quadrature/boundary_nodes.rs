use runmat_meshing_core::BoundaryTriangleOrder;

use super::{boundary_shape, weighted_derivative};
use crate::topology::element::cross;

pub(crate) fn boundary_nodal_area_vectors(
    order: BoundaryTriangleOrder,
    coordinates: &[[f64; 3]],
) -> Option<Vec<[f64; 3]>> {
    let reference_nodes: &[[f64; 3]] = match order {
        BoundaryTriangleOrder::Tri3 => &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        BoundaryTriangleOrder::Tri6 => &[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
        ],
    };
    if coordinates.len() != reference_nodes.len() {
        return None;
    }
    reference_nodes
        .iter()
        .map(|&barycentric| {
            let (_, derivatives) = boundary_shape(order, barycentric);
            let tangent_xi = weighted_derivative(coordinates, &derivatives[..coordinates.len()], 0);
            let tangent_eta =
                weighted_derivative(coordinates, &derivatives[..coordinates.len()], 1);
            let vector = cross(tangent_xi, tangent_eta);
            vector
                .iter()
                .all(|component| component.is_finite())
                .then_some(vector)
        })
        .collect()
}
