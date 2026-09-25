//! Material stiffness integration for affine and curved isoparametric Tet10 solids.

use runmat_meshing_core::TETRAHEDRON_MIDSIDE_EDGE_CORNERS;

use crate::topology::quadrature::{GAUSS_LEGENDRE_EIGHT_UNIT, TETRAHEDRON_FOUR_POINT_RULE};

use super::{
    material::elasticity_matrix,
    tetrahedron10::{
        jacobian_determinant, physical_shape_gradients, strain_displacement_matrix, validate_nodes,
        Tetrahedron10ElementError, Tetrahedron10ElementGeometry, Tetrahedron10Matrix30,
        TETRAHEDRON10_ELEMENT_DOF_COUNT,
    },
    SolidMaterial,
};

pub fn global_stiffness_matrix(
    material: SolidMaterial,
    geometry: Tetrahedron10ElementGeometry,
) -> Result<Tetrahedron10Matrix30, Tetrahedron10ElementError> {
    validate_nodes(geometry.nodes_m)?;
    let elasticity = elasticity_matrix(material)?;
    let mut stiffness = [[0.0; TETRAHEDRON10_ELEMENT_DOF_COUNT]; TETRAHEDRON10_ELEMENT_DOF_COUNT];
    if is_affine(geometry) {
        for barycentric in TETRAHEDRON_FOUR_POINT_RULE {
            add_point(
                &mut stiffness,
                geometry,
                &elasticity,
                barycentric,
                1.0 / 24.0,
            )?;
        }
    } else {
        // Curved Tet10 BᵀDB det(J) is rational, not a degree-two polynomial. The
        // straight-element four-point rule measurably underintegrates even mild warps.
        for (r, r_weight) in GAUSS_LEGENDRE_EIGHT_UNIT {
            let remaining_r = 1.0 - r;
            for (s, s_weight) in GAUSS_LEGENDRE_EIGHT_UNIT {
                let remaining_s = 1.0 - s;
                for (t, t_weight) in GAUSS_LEGENDRE_EIGHT_UNIT {
                    let barycentric = [
                        remaining_r * remaining_s * (1.0 - t),
                        r,
                        remaining_r * s,
                        remaining_r * remaining_s * t,
                    ];
                    let reference_weight =
                        r_weight * s_weight * t_weight * remaining_r * remaining_r * remaining_s;
                    add_point(
                        &mut stiffness,
                        geometry,
                        &elasticity,
                        barycentric,
                        reference_weight,
                    )?;
                }
            }
        }
    }
    if stiffness.iter().flatten().all(|value| value.is_finite()) {
        Ok(stiffness)
    } else {
        Err(Tetrahedron10ElementError::DegenerateOrInverted)
    }
}

fn is_affine(geometry: Tetrahedron10ElementGeometry) -> bool {
    TETRAHEDRON_MIDSIDE_EDGE_CORNERS
        .iter()
        .enumerate()
        .all(|(edge, [left, right])| {
            (0..3).all(|axis| {
                geometry.nodes_m[4 + edge][axis]
                    == 0.5 * (geometry.nodes_m[*left][axis] + geometry.nodes_m[*right][axis])
            })
        })
}

fn add_point(
    stiffness: &mut Tetrahedron10Matrix30,
    geometry: Tetrahedron10ElementGeometry,
    elasticity: &[[f64; 6]; 6],
    barycentric: [f64; 4],
    reference_weight: f64,
) -> Result<(), Tetrahedron10ElementError> {
    let gradients = physical_shape_gradients(geometry.nodes_m, barycentric)?;
    let b = strain_displacement_matrix(gradients);
    let scale = reference_weight * jacobian_determinant(geometry.nodes_m, barycentric)?;
    let mut elasticity_b = [[0.0; TETRAHEDRON10_ELEMENT_DOF_COUNT]; 6];
    for row in 0..6 {
        for column in 0..TETRAHEDRON10_ELEMENT_DOF_COUNT {
            elasticity_b[row][column] = (0..6)
                .map(|inner| elasticity[row][inner] * b[inner][column])
                .sum();
        }
    }
    for row in 0..TETRAHEDRON10_ELEMENT_DOF_COUNT {
        for column in row..TETRAHEDRON10_ELEMENT_DOF_COUNT {
            let contribution = scale
                * (0..6)
                    .map(|inner| b[inner][row] * elasticity_b[inner][column])
                    .sum::<f64>();
            stiffness[row][column] += contribution;
            if row != column {
                stiffness[column][row] += contribution;
            }
        }
    }
    Ok(())
}
