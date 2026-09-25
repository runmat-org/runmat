//! Exact-degree consistent mass for straight and curved isoparametric Tet10 solids.

use crate::topology::quadrature::GAUSS_LEGENDRE_FIVE_UNIT;

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
    let mut mass = [[0.0; TETRAHEDRON10_ELEMENT_DOF_COUNT]; TETRAHEDRON10_ELEMENT_DOF_COUNT];
    for (r, r_weight) in GAUSS_LEGENDRE_FIVE_UNIT {
        let remaining_r = 1.0 - r;
        for (s, s_weight) in GAUSS_LEGENDRE_FIVE_UNIT {
            let remaining_s = 1.0 - s;
            for (t, t_weight) in GAUSS_LEGENDRE_FIVE_UNIT {
                let barycentric = [
                    remaining_r * remaining_s * (1.0 - t),
                    r,
                    remaining_r * s,
                    remaining_r * remaining_s * t,
                ];
                let reference_weight =
                    r_weight * s_weight * t_weight * remaining_r * remaining_r * remaining_s;
                let shape = shape_values(barycentric);
                let scale = material.density_kg_per_m3
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
        }
    }
    Ok(mass)
}
