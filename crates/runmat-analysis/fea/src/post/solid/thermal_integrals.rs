//! Volume integrals use retained nodal temperatures, never a fit to four recovery samples.

use super::{stress::response_error, StructuralFieldRecoveryError};
use crate::algebra::coercivity::NumericalCapacityError;
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
use runmat_simulation_usage::{AsyncCapacityOwner, OperationCount, UsageCategory, UsageContext};

type Averages = ([f64; 6], [f64; 6], f64);

pub(super) async fn integrate_with_capacity<A>(
    element: &SolidRecoveryElementSummary,
    thermal: &SolvedThermalRecoveryQuadrature,
    displacement: &[f64],
    owner: Option<&dyn AsyncCapacityOwner<A>>,
) -> Result<Averages, NumericalCapacityError<StructuralFieldRecoveryError, A>> {
    let affine = match &element.geometry {
        SolidRecoveryGeometry::Tetrahedron4 { .. } => true,
        SolidRecoveryGeometry::Tetrahedron10 { coordinates_m, .. } => {
            tetrahedron10_is_affine(Tetrahedron10ElementGeometry {
                nodes_m: coordinates_m.canonical_value(),
            })
        }
    };
    if affine {
        return integrate_rule(
            element,
            thermal,
            displacement,
            &GAUSS_LEGENDRE_FIVE_UNIT,
            owner,
        )
        .await;
    }
    let eight = integrate_rule(
        element,
        thermal,
        displacement,
        &GAUSS_LEGENDRE_EIGHT_UNIT,
        owner,
    )
    .await?;
    let twelve = integrate_rule(
        element,
        thermal,
        displacement,
        &GAUSS_LEGENDRE_TWELVE_UNIT,
        owner,
    )
    .await?;
    if converged_with_capacity(&eight, &twelve, owner).await? {
        return Ok(twelve);
    }
    let sixteen = integrate_rule(
        element,
        thermal,
        displacement,
        &GAUSS_LEGENDRE_SIXTEEN_UNIT,
        owner,
    )
    .await?;
    if converged_with_capacity(&twelve, &sixteen, owner).await? {
        return Ok(sixteen);
    }
    Err(StructuralFieldRecoveryError::UnconvergedThermalQuadrature {
        element_id: element.element_id.clone(),
    }
    .into())
}

async fn converged_with_capacity<A>(
    left: &Averages,
    right: &Averages,
    owner: Option<&dyn AsyncCapacityOwner<A>>,
) -> Result<bool, NumericalCapacityError<StructuralFieldRecoveryError, A>> {
    let close = |a: f64, b: f64, floor: f64| (a - b).abs() <= floor + 1.0e-8 * a.abs().max(b.abs());
    for (&a, &b) in left.0.iter().zip(right.0.iter()) {
        if !crate::algebra::capacity::measured(
            owner,
            UsageCategory::Solve,
            OperationCount::from_operations(3),
            || Ok::<_, StructuralFieldRecoveryError>(close(a, b, 1.0e-14)),
        )
        .await?
        {
            return Ok(false);
        }
    }
    for (&a, &b) in left.1.iter().zip(right.1.iter()) {
        if !crate::algebra::capacity::measured(
            owner,
            UsageCategory::Solve,
            OperationCount::from_operations(3),
            || Ok::<_, StructuralFieldRecoveryError>(close(a, b, 1.0e-6)),
        )
        .await?
        {
            return Ok(false);
        }
    }
    crate::algebra::capacity::measured(
        owner,
        UsageCategory::Solve,
        OperationCount::from_operations(3),
        || Ok::<_, StructuralFieldRecoveryError>(close(left.2, right.2, 1.0e-12)),
    )
    .await
}

async fn integrate_rule<A>(
    element: &SolidRecoveryElementSummary,
    thermal: &SolvedThermalRecoveryQuadrature,
    displacement: &[f64],
    rule: &[(f64, f64)],
    owner: Option<&dyn AsyncCapacityOwner<A>>,
) -> Result<Averages, NumericalCapacityError<StructuralFieldRecoveryError, A>> {
    let invalid = || StructuralFieldRecoveryError::NonFiniteResult {
        element_id: element.element_id.clone(),
    };
    let (order, coordinates): (ElementOrder, &[[f64; 3]]) = match &element.geometry {
        SolidRecoveryGeometry::Tetrahedron4 { coordinates_m, .. } => {
            (ElementOrder::Tet4, coordinates_m.canonical_values())
        }
        SolidRecoveryGeometry::Tetrahedron10 { coordinates_m, .. } => {
            (ElementOrder::Tet10, coordinates_m.canonical_values())
        }
    };
    let indices = element.geometry.node_indices();
    if thermal.nodal_temperature_k.canonical_values().len() != indices.len()
        || thermal
            .nodal_temperature_k
            .canonical_values()
            .iter()
            .any(|x| !x.is_finite() || *x <= 0.0)
    {
        return Err(invalid().into());
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
    let nodes = indices.len() as u64;
    // One selected point: Duffy coordinates (13), displacement gradient (18n),
    // engineering shear (3), thermal interpolation/eigenstrain (2n+1),
    // constitutive response excluding its separately reserved material scalars (20),
    // volume accumulation (1), and six weighted strain/stress/energy updates (48).
    let outer = OperationCount::from_operations(20 * nodes + 86);
    let nested = OperationCount::from_operations(match order {
        ElementOrder::Tet4 => 261 + 9,
        ElementOrder::Tet10 => 116 + 495 + 9,
    });
    let required = outer.checked_add(nested).map_err(|error| {
        NumericalCapacityError::Numerical(StructuralFieldRecoveryError::Accounting(error.into()))
    })?;
    let mut points = tensor_tetrahedron_rule(rule);
    for _ in 0..rule.len().pow(3) {
        if let Some(owner) = owner {
            owner
                .ensure_capacity(required)
                .await
                .map_err(NumericalCapacityError::Admission)?;
        }
        let usage = UsageContext::capture();
        let mut batch = usage
            .as_ref()
            .map(|usage| usage.ledger().reserve(UsageCategory::Solve, outer))
            .transpose()
            .map_err(StructuralFieldRecoveryError::Accounting)?;
        let point = points
            .next()
            .expect("selected tensor rule has N cubed points");
        let mut error = None;
        visit_volume_rule(order, coordinates, std::iter::once(point), |point| {
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
        .map_err(StructuralFieldRecoveryError::Accounting)?
        .ok_or_else(invalid)?;
        if let Some(error) = error {
            return Err(error.into());
        }
        if let Some(batch) = &mut batch {
            batch
                .complete(outer)
                .map_err(StructuralFieldRecoveryError::Accounting)?;
        }
    }
    if !volume.is_finite() || volume <= 0.0 || !integrals.2.is_finite() {
        return Err(invalid().into());
    }
    crate::algebra::capacity::measured(
        owner,
        UsageCategory::Solve,
        OperationCount::from_operations(13),
        || {
            Ok::<_, StructuralFieldRecoveryError>((
                integrals.0.map(|x| x / volume),
                integrals.1.map(|x| x / volume),
                integrals.2 / volume,
            ))
        },
    )
    .await
}
