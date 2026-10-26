//! Exact-degree consistent mass for straight and curved isoparametric Tet10 solids.

use crate::topology::quadrature::{tensor_tetrahedron_rule, GAUSS_LEGENDRE_FIVE_UNIT};

use super::{
    tetrahedron10::{
        jacobian_determinant, shape_values, validate_nodes, Tetrahedron10ElementError,
        Tetrahedron10ElementGeometry, Tetrahedron10Matrix30, TETRAHEDRON10_ELEMENT_DOF_COUNT,
        TETRAHEDRON10_ELEMENT_NODE_COUNT, TETRAHEDRON10_NODE_DOF_COUNT,
    },
    SolidMaterial,
};

pub fn consistent_mass_matrix(
    material: SolidMaterial,
    geometry: Tetrahedron10ElementGeometry,
) -> Result<Tetrahedron10Matrix30, Tetrahedron10ElementError> {
    validate_nodes(geometry.nodes_m)?;
    material.validate()?;
    if super::tetrahedron10_geometry::is_affine(geometry) {
        return affine_consistent_mass(material, geometry);
    }
    let mut mass = [[0.0; TETRAHEDRON10_ELEMENT_DOF_COUNT]; TETRAHEDRON10_ELEMENT_DOF_COUNT];
    for (barycentric, reference_weight) in tensor_tetrahedron_rule(&GAUSS_LEGENDRE_FIVE_UNIT) {
        let shape = shape_values(barycentric);
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
    Ok(mass)
}

#[cfg(test)]
mod tests;
