//! Volume integrals use retained nodal temperatures, never a fit to four recovery samples.

use super::{stress::response_error, StructuralFieldRecoveryError};
use crate::{
    assembly::{
        elements::solid::response::response_at_volume_point,
        elements::solid::{tetrahedron10_is_affine, Tetrahedron10ElementGeometry},
        SolidRecoveryElementSummary, SolidRecoveryGeometry, SolvedThermalRecoveryQuadrature,
    },
    topology::quadrature::{
        tensor_tetrahedron_rule, visit_volume_rule, GAUSS_LEGENDRE_EIGHT_UNIT,
        GAUSS_LEGENDRE_FIVE_UNIT, GAUSS_LEGENDRE_SIXTEEN_UNIT, GAUSS_LEGENDRE_TWELVE_UNIT,
    },
};
use runmat_meshing_core::ElementOrder;

type Averages = ([f64; 6], [f64; 6], f64);

pub(super) fn integrate(
    element: &SolidRecoveryElementSummary,
    thermal: &SolvedThermalRecoveryQuadrature,
    displacement: &[f64],
) -> Result<Averages, StructuralFieldRecoveryError> {
    let affine = match &element.geometry {
        SolidRecoveryGeometry::Tetrahedron4 { .. } => true,
        SolidRecoveryGeometry::Tetrahedron10 { coordinates_m, .. } => {
            tetrahedron10_is_affine(Tetrahedron10ElementGeometry {
                nodes_m: *coordinates_m,
            })
        }
    };
    if affine {
        return integrate_rule(element, thermal, displacement, &GAUSS_LEGENDRE_FIVE_UNIT);
    }
    let eight = integrate_rule(element, thermal, displacement, &GAUSS_LEGENDRE_EIGHT_UNIT)?;
    let twelve = integrate_rule(element, thermal, displacement, &GAUSS_LEGENDRE_TWELVE_UNIT)?;
    if converged(&eight, &twelve) {
        return Ok(twelve);
    }
    let sixteen = integrate_rule(element, thermal, displacement, &GAUSS_LEGENDRE_SIXTEEN_UNIT)?;
    if converged(&twelve, &sixteen) {
        return Ok(sixteen);
    }
    Err(StructuralFieldRecoveryError::UnconvergedThermalQuadrature {
        element_id: element.element_id.clone(),
    })
}

fn converged(left: &Averages, right: &Averages) -> bool {
    let close = |a: f64, b: f64, floor: f64| (a - b).abs() <= floor + 1.0e-8 * a.abs().max(b.abs());
    left.0
        .iter()
        .zip(right.0)
        .all(|(&a, b)| close(a, b, 1.0e-14))
        && left
            .1
            .iter()
            .zip(right.1)
            .all(|(&a, b)| close(a, b, 1.0e-6))
        && close(left.2, right.2, 1.0e-12)
}

fn integrate_rule(
    element: &SolidRecoveryElementSummary,
    thermal: &SolvedThermalRecoveryQuadrature,
    displacement: &[f64],
    rule: &[(f64, f64)],
) -> Result<Averages, StructuralFieldRecoveryError> {
    let invalid = || StructuralFieldRecoveryError::NonFiniteResult {
        element_id: element.element_id.clone(),
    };
    let (order, coordinates): (ElementOrder, &[[f64; 3]]) = match &element.geometry {
        SolidRecoveryGeometry::Tetrahedron4 { coordinates_m, .. } => {
            (ElementOrder::Tet4, coordinates_m)
        }
        SolidRecoveryGeometry::Tetrahedron10 { coordinates_m, .. } => {
            (ElementOrder::Tet10, coordinates_m)
        }
    };
    let indices = element.geometry.node_indices();
    if thermal.nodal_temperature_k.len() != indices.len()
        || thermal
            .nodal_temperature_k
            .iter()
            .any(|x| !x.is_finite() || *x <= 0.0)
    {
        return Err(invalid());
    }
    let nodal = indices
        .iter()
        .map(|&index| {
            let start = index.checked_mul(3).ok_or_else(invalid)?;
            let end = start.checked_add(3).ok_or_else(invalid)?;
            let values = displacement
                .get(start..end)
                .filter(|v| v.iter().all(|x| x.is_finite()))
                .ok_or_else(invalid)?;
            Ok([values[0], values[1], values[2]])
        })
        .collect::<Result<Vec<_>, StructuralFieldRecoveryError>>()?;
    let mut integrals = ([0.0; 6], [0.0; 6], 0.0);
    let mut volume = 0.0;
    let mut error = None;
    visit_volume_rule(order, coordinates, tensor_tetrahedron_rule(rule), |point| {
        if error.is_some() {
            return;
        }
        match response_at_volume_point(element, &nodal, &point) {
            Err(failure) => error = Some(response_error(element, failure)),
            Ok(response) => {
                let weight = point.volume_weight_m3;
                volume += weight;
                for component in 0..6 {
                    integrals.0[component] += weight * response.total_strain[component];
                    integrals.1[component] += weight * response.stress[component];
                    integrals.2 += 0.5
                        * weight
                        * response.elastic_strain[component]
                        * response.stress[component];
                }
            }
        }
    })
    .ok_or_else(invalid)?;
    if let Some(error) = error {
        return Err(error);
    }
    if !volume.is_finite() || volume <= 0.0 || !integrals.2.is_finite() {
        return Err(invalid());
    }
    Ok((
        integrals.0.map(|x| x / volume),
        integrals.1.map(|x| x / volume),
        integrals.2 / volume,
    ))
}
