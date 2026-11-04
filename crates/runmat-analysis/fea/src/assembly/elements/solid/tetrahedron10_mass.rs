//! Exact-degree consistent mass for straight and curved isoparametric Tet10 solids.

use crate::topology::quadrature::{tensor_tetrahedron_rule, GAUSS_LEGENDRE_FIVE_UNIT};
use runmat_simulation_usage::{OperationCount, UsageCategory, UsageContext};

use super::{
    tetrahedron10::{
        jacobian_determinant, shape_values, validate_nodes, Tetrahedron10CapacityError,
        Tetrahedron10ElementError, Tetrahedron10ElementGeometry, Tetrahedron10Matrix30,
        TETRAHEDRON10_ELEMENT_DOF_COUNT, TETRAHEDRON10_ELEMENT_NODE_COUNT,
        TETRAHEDRON10_NODE_DOF_COUNT,
    },
    SolidMaterial,
};

pub fn consistent_mass_matrix(
    material: SolidMaterial,
    geometry: Tetrahedron10ElementGeometry,
) -> Result<Tetrahedron10Matrix30, Tetrahedron10ElementError> {
    let future = consistent_mass_matrix_with_optional_capacity::<std::convert::Infallible>(
        material, geometry, None,
    );
    let mut future = std::pin::pin!(future);
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    match std::future::Future::poll(future.as_mut(), &mut context) {
        std::task::Poll::Ready(result) => result.map_err(|error| match error {
            Tetrahedron10CapacityError::Numerical(error) => error,
            Tetrahedron10CapacityError::Admission(never) => match never {},
        }),
        std::task::Poll::Pending => unreachable!("immediate Tet10 mass owner cannot suspend"),
    }
}

pub async fn consistent_mass_matrix_with_capacity<E>(
    material: SolidMaterial,
    geometry: Tetrahedron10ElementGeometry,
    owner: &dyn runmat_simulation_usage::AsyncCapacityOwner<E>,
) -> Result<Tetrahedron10Matrix30, Tetrahedron10CapacityError<E>> {
    consistent_mass_matrix_with_optional_capacity(material, geometry, Some(owner)).await
}

async fn consistent_mass_matrix_with_optional_capacity<E>(
    material: SolidMaterial,
    geometry: Tetrahedron10ElementGeometry,
    owner: Option<&dyn runmat_simulation_usage::AsyncCapacityOwner<E>>,
) -> Result<Tetrahedron10Matrix30, Tetrahedron10CapacityError<E>> {
    if let Some(owner) = owner {
        super::tetrahedron10_geometry::validate_nodes_with_capacity(geometry.nodes_m, owner)
            .await?;
    } else {
        validate_nodes(geometry.nodes_m)?;
    }
    material.validate()?;
    if super::tetrahedron10_geometry::is_affine(geometry) {
        if let Some(owner) = owner {
            owner
                .ensure_capacity(OperationCount::from_operations(1_015))
                .await
                .map_err(Tetrahedron10CapacityError::Admission)?;
        }
        return affine_consistent_mass(material, geometry).map_err(Into::into);
    }
    // Five-point tensor rule has 125 points. At each point density/weight/
    // Jacobian scaling is two products; each of 100 node pairs forms a value
    // with two products and adds it to three displacement-axis entries.
    // The shared rule iterator and shape helper are pure, so this caller owns
    // their work: 13 Duffy-rule operations and 24 quadratic shape operations
    // at each point. Their reservation precedes the first iterator advance.
    let usage = UsageContext::capture();
    let mut mass = [[0.0; TETRAHEDRON10_ELEMENT_DOF_COUNT]; TETRAHEDRON10_ELEMENT_DOF_COUNT];
    let mut points = tensor_tetrahedron_rule(&GAUSS_LEGENDRE_FIVE_UNIT);
    for _ in 0..125 {
        if let Some(owner) = owner {
            owner
                .ensure_capacity(OperationCount::from_operations(37 + 502 + 92 + 185))
                .await
                .map_err(Tetrahedron10CapacityError::Admission)?;
        }
        let mut geometry_batch = usage
            .as_ref()
            .map(|usage| {
                usage
                    .ledger()
                    .reserve(UsageCategory::Assembly, OperationCount::from_operations(37))
            })
            .transpose()?;
        let mut batch = usage
            .as_ref()
            .map(|usage| {
                usage.ledger().reserve(
                    UsageCategory::Assembly,
                    OperationCount::from_operations(502),
                )
            })
            .transpose()?;
        let (barycentric, reference_weight) = points.next().expect("fixed five-point tensor rule");
        if let Some(batch) = &mut geometry_batch {
            batch.complete(OperationCount::from_operations(13))?;
        }
        let shape = shape_values(barycentric);
        if let Some(batch) = &mut geometry_batch {
            batch.complete(OperationCount::from_operations(24))?;
        }
        let scale = material.density_kg_per_m3.canonical_value()
            * reference_weight
            * jacobian_determinant(geometry.nodes_m, barycentric)?;
        for left_node in 0..TETRAHEDRON10_ELEMENT_NODE_COUNT {
            for right_node in 0..TETRAHEDRON10_ELEMENT_NODE_COUNT {
                let value = scale * shape[left_node] * shape[right_node];
                for axis in 0..TETRAHEDRON10_NODE_DOF_COUNT {
                    mass[left_node * TETRAHEDRON10_NODE_DOF_COUNT + axis]
                        [right_node * TETRAHEDRON10_NODE_DOF_COUNT + axis] += value;
                }
            }
        }
        if let Some(batch) = &mut batch {
            batch.complete(OperationCount::from_operations(502))?;
        }
    }
    Ok(mass)
}

/// Exact barycentric moments: integral(prod L_i^a_i) = 6 V prod(a_i!)/(3+sum a_i)!.
/// Integrating quadratic corner/edge shapes gives integer coefficients times rho V/420.
/// Negative corner/edge entries are consistent mass, never row-sum lumping.
fn affine_consistent_mass(
    material: SolidMaterial,
    geometry: Tetrahedron10ElementGeometry,
) -> Result<Tetrahedron10Matrix30, Tetrahedron10ElementError> {
    use runmat_meshing_core::TETRAHEDRON_MIDSIDE_EDGE_CORNERS as EDGES;
    let determinant = jacobian_determinant(geometry.nodes_m, [0.25; 4])?;
    // Apply the dimensionless coefficient to the larger factor first. Dividing a
    // subnormal determinant before multiplying by a compensating density loses mass.
    let density = material.density_kg_per_m3.canonical_value();
    let (large, small) = if density >= determinant {
        (density, determinant)
    } else {
        (determinant, density)
    };
    // One coefficient division and two products for each of the 100 node
    // pairs and three displacement axes. Jacobian evaluation is separate.
    let usage = UsageContext::capture();
    let mut batch = usage
        .as_ref()
        .map(|usage| {
            usage.ledger().reserve(
                UsageCategory::Assembly,
                OperationCount::from_operations(900),
            )
        })
        .transpose()?;
    let mut mass = [[0.0; TETRAHEDRON10_ELEMENT_DOF_COUNT]; TETRAHEDRON10_ELEMENT_DOF_COUNT];
    for left in 0..TETRAHEDRON10_ELEMENT_NODE_COUNT {
        for right in 0..TETRAHEDRON10_ELEMENT_NODE_COUNT {
            let coefficient = match (left < 4, right < 4) {
                (true, true) => {
                    if left == right {
                        6.0
                    } else {
                        1.0
                    }
                }
                (true, false) => {
                    if EDGES[right - 4].contains(&left) {
                        -4.0
                    } else {
                        -6.0
                    }
                }
                (false, true) => {
                    if EDGES[left - 4].contains(&right) {
                        -4.0
                    } else {
                        -6.0
                    }
                }
                (false, false) => {
                    if left == right {
                        32.0
                    } else if EDGES[left - 4]
                        .iter()
                        .any(|corner| EDGES[right - 4].contains(corner))
                    {
                        16.0
                    } else {
                        8.0
                    }
                }
            };
            for axis in 0..TETRAHEDRON10_NODE_DOF_COUNT {
                mass[left * 3 + axis][right * 3 + axis] = (large * (coefficient / 2520.0)) * small;
            }
        }
    }
    if let Some(batch) = &mut batch {
        batch.complete(OperationCount::from_operations(900))?;
    }
    Ok(mass)
}

#[cfg(test)]
mod tests;
