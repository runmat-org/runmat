//! Shared isoparametric volume quadrature and temperature/gradient interpolation.

use super::{weighted_position, KEAST_FIFTEEN, TETRAHEDRON_DERIVATIVES};
use crate::topology::{
    contract::VolumeQuadraturePoint,
    element::{cross, dot, invert},
};
use runmat_meshing_core::{ElementOrder, TETRAHEDRON_MIDSIDE_EDGE_CORNERS};
use runmat_simulation_usage::{LedgerError, OperationCount, UsageCategory, UsageContext};

/// The same reference batches reserved by `visit_volume_rule` and `volume_shape`.
pub(crate) fn volume_point_work(order: ElementOrder) -> OperationCount {
    let nodes = match order {
        ElementOrder::Tet4 => 4,
        ElementOrder::Tet10 => 10,
    };
    OperationCount::from_operations(39 * nodes + 105)
}

pub(crate) fn volume_shape_work(order: ElementOrder) -> OperationCount {
    OperationCount::from_operations(if order == ElementOrder::Tet10 { 116 } else { 0 })
}

pub(crate) fn volume_rule_work(
    order: ElementOrder,
    points: usize,
) -> Result<OperationCount, LedgerError> {
    let points = u64::try_from(points).map_err(|_| LedgerError::CountOverflow)?;
    volume_point_work(order)
        .checked_add(volume_shape_work(order))
        .and_then(|per_point| per_point.checked_mul(points))
        .map_err(LedgerError::from)
}

pub(crate) fn visit_volume_quadrature(
    order: ElementOrder,
    coordinates: &[[f64; 3]],
    visit: impl FnMut(VolumeQuadraturePoint),
) -> Result<Option<()>, LedgerError> {
    visit_volume_rule(order, coordinates, KEAST_FIFTEEN, visit)
}

pub(crate) fn visit_volume_rule(
    order: ElementOrder,
    coordinates: &[[f64; 3]],
    rule: impl IntoIterator<Item = ([f64; 4], f64)>,
    mut visit: impl FnMut(VolumeQuadraturePoint),
) -> Result<Option<()>, LedgerError> {
    let node_count = match order {
        ElementOrder::Tet4 => 4,
        ElementOrder::Tet10 => 10,
    };
    if coordinates.len() != node_count {
        return Ok(None);
    }
    for (barycentric, reference_weight) in rule {
        let (shape_values, derivatives) = volume_shape(order, barycentric)?;
        // Nine n-term Jacobian dots, cross/dot determinant (14), inverse
        // cofactors and normalization (42), n physical three-term gradients,
        // one weighted position, volume scaling, and four barycentric gradients.
        let full = volume_point_work(order);
        let determinant_prefix = 9 * (2 * node_count as u64 - 1) + 14;
        let usage = UsageContext::capture();
        let mut batch = usage
            .as_ref()
            .map(|usage| usage.ledger().reserve(UsageCategory::Assembly, full))
            .transpose()?;
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
            if let Some(batch) = &mut batch {
                batch.complete(OperationCount::from_operations(determinant_prefix))?;
            }
            return Ok(None);
        }
        let Some(inverse) = invert(jacobian) else {
            if let Some(batch) = &mut batch {
                batch.complete(OperationCount::from_operations(determinant_prefix + 32))?;
            }
            return Ok(None);
        };
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
        let point = VolumeQuadraturePoint {
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
        };
        if let Some(batch) = &mut batch {
            batch.complete(full)?;
        }
        visit(point);
    }
    Ok(Some(()))
}

pub(crate) fn volume_shape(
    order: ElementOrder,
    barycentric: [f64; 4],
) -> Result<([f64; 10], [[f64; 3]; 10]), LedgerError> {
    let mut values = [0.0; 10];
    let mut derivatives = [[0.0; 3]; 10];
    if order == ElementOrder::Tet4 {
        values[..4].copy_from_slice(&barycentric);
        derivatives[..4].copy_from_slice(&TETRAHEDRON_DERIVATIVES);
        return Ok((values, derivatives));
    }
    // Ten quadratic shape values (24) and ten reference derivatives (92).
    let usage = UsageContext::capture();
    let mut batch = usage
        .as_ref()
        .map(|usage| {
            usage
                .ledger()
                .reserve(UsageCategory::Assembly, volume_shape_work(order))
        })
        .transpose()?;
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
    if let Some(batch) = &mut batch {
        batch.complete(volume_shape_work(order))?;
    }
    Ok((values, derivatives))
}

// The collapsed tensor rule is owned by meshing core with the Tet10 element-validity rule.
pub(crate) use runmat_meshing_core::tensor_tetrahedron_rule;
