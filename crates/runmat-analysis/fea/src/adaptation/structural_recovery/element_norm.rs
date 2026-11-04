//! Integrate the squared discrepancy before summing; averaging first loses variance.
//! Tet4 needs degree two. For curved Tet10, squared P2 recovery times cubic det(J)
//! has degree at most seven; the shared five-point tensor rule integrates that density.
use super::*;
use crate::topology::quadrature::{
    tensor_tetrahedron_rule, visit_volume_rule, GAUSS_LEGENDRE_FIVE_UNIT,
};

pub(super) fn integrate(
    order: ElementOrder,
    coordinates: &[[f64; 3]],
    nodal: &[[f64; STRESS_COMPONENT_COUNT]],
    original: [f64; STRESS_COMPONENT_COUNT],
    work: &mut u64,
    options: StructuralRecoveryEstimatorOptions,
) -> Result<(f64, f64), StructuralRecoveryEstimatorError> {
    let expected = match order {
        ElementOrder::Tet4 => 4,
        ElementOrder::Tet10 => 10,
    };
    if nodal.len() != expected || coordinates.len() != expected {
        return Err(StructuralRecoveryEstimatorError::InvalidElementGeometry);
    }
    let mut squared = [0.0; 2];
    let mut failure = None;
    let mut visit = |point: crate::topology::VolumeQuadraturePoint| {
        if failure.is_some() {
            return;
        }
        if let Err(error) = checkpoint(work, options) {
            failure = Some(error);
            return;
        }
        let recovered: [f64; STRESS_COMPONENT_COUNT] = std::array::from_fn(|component| {
            nodal
                .iter()
                .zip(point.shape_values())
                .map(|(value, shape)| value[component] * shape)
                .sum::<f64>()
        });
        squared[0] += point.volume_weight_m3
            * stress_norm_squared(std::array::from_fn(|i| original[i] - recovered[i]));
        squared[1] += point.volume_weight_m3 * stress_norm_squared(original);
    };
    let geometry = match order {
        ElementOrder::Tet4 => visit_volume_rule(
            order,
            coordinates,
            QUADRATURE.map(|p| (p, 1.0 / 24.0)),
            &mut visit,
        ),
        ElementOrder::Tet10 => visit_volume_rule(
            order,
            coordinates,
            tensor_tetrahedron_rule(&GAUSS_LEGENDRE_FIVE_UNIT),
            &mut visit,
        ),
    };
    if let Some(error) = failure {
        return Err(error);
    }
    geometry
        .map_err(StructuralRecoveryEstimatorError::Accounting)?
        .ok_or(StructuralRecoveryEstimatorError::InvalidElementGeometry)?;
    if squared.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(StructuralRecoveryEstimatorError::InvalidStressField);
    }
    Ok((squared[0], squared[1]))
}

#[cfg(test)]
mod tests;
