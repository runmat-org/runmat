//! Shared isoparametric volume quadrature and temperature/gradient interpolation.

use super::{weighted_position, KEAST_FIFTEEN, TETRAHEDRON_DERIVATIVES};
use crate::topology::{
    contract::VolumeQuadraturePoint,
    element::{cross, dot, invert},
};
use runmat_meshing_core::{ElementOrder, TETRAHEDRON_MIDSIDE_EDGE_CORNERS};

pub(crate) fn visit_volume_quadrature(
    order: ElementOrder,
    coordinates: &[[f64; 3]],
    visit: impl FnMut(VolumeQuadraturePoint),
) -> Option<()> {
    visit_volume_rule(order, coordinates, KEAST_FIFTEEN, visit)
}

pub(crate) fn visit_volume_rule(
    order: ElementOrder,
    coordinates: &[[f64; 3]],
    rule: impl IntoIterator<Item = ([f64; 4], f64)>,
    mut visit: impl FnMut(VolumeQuadraturePoint),
) -> Option<()> {
    let node_count = match order {
        ElementOrder::Tet4 => 4,
        ElementOrder::Tet10 => 10,
    };
    if coordinates.len() != node_count {
        return None;
    }
    for (barycentric, reference_weight) in rule {
        let (shape_values, derivatives) = volume_shape(order, barycentric);
        let jacobian = std::array::from_fn::<_, 3, _>(|reference_axis| {
            std::array::from_fn(|physical_axis| {
                coordinates
                    .iter()
                    .zip(&derivatives[..node_count])
                    .map(|(node, derivative)| node[physical_axis] * derivative[reference_axis])
                    .sum::<f64>()
            })
        });
        let determinant = dot(jacobian[0], cross(jacobian[1], jacobian[2]));
        if !determinant.is_finite() || determinant <= 0.0 {
            return None;
        }
        let inverse = invert(jacobian)?;
        let mut physical_shape_gradients = [[0.0; 3]; 10];
        for node in 0..node_count {
            physical_shape_gradients[node] = std::array::from_fn(|physical_axis| {
                (0..3)
                    .map(|reference_axis| {
                        inverse[physical_axis][reference_axis] * derivatives[node][reference_axis]
                    })
                    .sum()
            });
        }
        visit(VolumeQuadraturePoint {
            position_m: weighted_position(coordinates, &shape_values[..node_count]),
            volume_weight_m3: reference_weight * determinant,
            barycentric,
            physical_barycentric_gradients: TETRAHEDRON_DERIVATIVES.map(|derivative| {
                std::array::from_fn(|physical_axis| {
                    (0..3)
                        .map(|reference_axis| {
                            inverse[physical_axis][reference_axis] * derivative[reference_axis]
                        })
                        .sum()
                })
            }),
            shape_values,
            physical_shape_gradients,
            node_count,
        });
    }
    Some(())
}

pub(crate) fn volume_shape(
    order: ElementOrder,
    barycentric: [f64; 4],
) -> ([f64; 10], [[f64; 3]; 10]) {
    let mut values = [0.0; 10];
    let mut derivatives = [[0.0; 3]; 10];
    if order == ElementOrder::Tet4 {
        values[..4].copy_from_slice(&barycentric);
        derivatives[..4].copy_from_slice(&TETRAHEDRON_DERIVATIVES);
        return (values, derivatives);
    }
    values = runmat_meshing_core::interpolation::tetrahedron10_lagrange_weights(barycentric);
    for node in 0..4 {
        derivatives[node] = TETRAHEDRON_DERIVATIVES[node]
            .map(|derivative| (4.0 * barycentric[node] - 1.0) * derivative);
    }
    for (edge, [left, right]) in TETRAHEDRON_MIDSIDE_EDGE_CORNERS.into_iter().enumerate() {
        derivatives[4 + edge] = std::array::from_fn(|axis| {
            4.0 * (barycentric[right] * TETRAHEDRON_DERIVATIVES[left][axis]
                + barycentric[left] * TETRAHEDRON_DERIVATIVES[right][axis])
        });
    }
    (values, derivatives)
}

// The collapsed tensor rule is owned by meshing core with the Tet10 element-validity rule.
pub(crate) use runmat_meshing_core::tensor_tetrahedron_rule;
