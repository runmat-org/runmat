//! Material stiffness integration for affine and curved isoparametric Tet10 solids.

use runmat_meshing_core::TETRAHEDRON_MIDSIDE_EDGE_CORNERS;

use crate::topology::quadrature::{
    tensor_tetrahedron_rule, GAUSS_LEGENDRE_EIGHT_UNIT, GAUSS_LEGENDRE_SIXTEEN_UNIT,
    GAUSS_LEGENDRE_TWELVE_UNIT, TETRAHEDRON_FOUR_POINT_RULE,
};

use super::{
    material::elasticity_matrix,
    tetrahedron10::{
        physical_shape_gradients_and_measure, validate_nodes, Tetrahedron10ElementError,
        Tetrahedron10ElementGeometry, Tetrahedron10Matrix30, TETRAHEDRON10_ELEMENT_DOF_COUNT,
    },
    SolidMaterial,
};

pub fn global_stiffness_matrix(
    material: SolidMaterial,
    geometry: Tetrahedron10ElementGeometry,
) -> Result<Tetrahedron10Matrix30, Tetrahedron10ElementError> {
    validate_nodes(geometry.nodes_m)?;
    let elasticity = elasticity_matrix(material)?;
    let stiffness = if is_affine(geometry) {
        let mut stiffness =
            [[0.0; TETRAHEDRON10_ELEMENT_DOF_COUNT]; TETRAHEDRON10_ELEMENT_DOF_COUNT];
        for barycentric in TETRAHEDRON_FOUR_POINT_RULE {
            add_point(
                &mut stiffness,
                geometry,
                &elasticity,
                barycentric,
                1.0 / 24.0,
            )?;
        }
        stiffness
    } else {
        // Curved BᵀDB det(J) is rational. Compare full matrices at successive
        // orders; the difference is a numerical convergence check, not an error bound.
        let eight = integrate_curved(geometry, &elasticity, &GAUSS_LEGENDRE_EIGHT_UNIT)?;
        let twelve = integrate_curved(geometry, &elasticity, &GAUSS_LEGENDRE_TWELVE_UNIT)?;
        if converged(&eight, &twelve) {
            twelve
        } else {
            let sixteen = integrate_curved(geometry, &elasticity, &GAUSS_LEGENDRE_SIXTEEN_UNIT)?;
            if !converged(&twelve, &sixteen) {
                return Err(Tetrahedron10ElementError::UnconvergedStiffnessQuadrature);
            }
            sixteen
        }
    };
    if stiffness.iter().flatten().all(|value| value.is_finite()) {
        Ok(stiffness)
    } else {
        Err(Tetrahedron10ElementError::DegenerateOrInverted)
    }
}

fn integrate_curved(
    geometry: Tetrahedron10ElementGeometry,
    elasticity: &[[f64; 6]; 6],
    rule: &[(f64, f64)],
) -> Result<Tetrahedron10Matrix30, Tetrahedron10ElementError> {
    let mut stiffness = [[0.0; TETRAHEDRON10_ELEMENT_DOF_COUNT]; TETRAHEDRON10_ELEMENT_DOF_COUNT];
    for (barycentric, reference_weight) in tensor_tetrahedron_rule(rule) {
        add_point(
            &mut stiffness,
            geometry,
            elasticity,
            barycentric,
            reference_weight,
        )?;
    }
    Ok(stiffness)
}

fn converged(previous: &Tetrahedron10Matrix30, next: &Tetrahedron10Matrix30) -> bool {
    let scale = next
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    scale.is_finite()
        && scale > 0.0
        && previous
            .iter()
            .flatten()
            .zip(next.iter().flatten())
            .all(|(left, right)| (left - right).abs() <= 1.0e-8 * scale)
}

pub(crate) fn is_affine(geometry: Tetrahedron10ElementGeometry) -> bool {
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
    let (gradients, determinant) =
        physical_shape_gradients_and_measure(geometry.nodes_m, barycentric)?;
    let scale = reference_weight * determinant;
    let lambda = elasticity[0][1];
    let mu = elasticity[3][3];
    for (left_node, left_gradient) in gradients.iter().enumerate() {
        for (right_node, right_gradient) in gradients.iter().enumerate().skip(left_node) {
            let gradient_dot = left_gradient
                .iter()
                .zip(right_gradient)
                .map(|(left, right)| left * right)
                .sum::<f64>();
            for left_axis in 0..3 {
                for right_axis in 0..3 {
                    let row = 3 * left_node + left_axis;
                    let column = 3 * right_node + right_axis;
                    if row > column {
                        continue;
                    }
                    // The isotropic BᵀDB block is λ∂ₐNᵢ∂ᵦNⱼ +
                    // μ(δₐᵦ∇Nᵢ·∇Nⱼ + ∂ᵦNᵢ∂ₐNⱼ).
                    let contribution = scale
                        * (lambda * left_gradient[left_axis] * right_gradient[right_axis]
                            + mu * left_gradient[right_axis] * right_gradient[left_axis]
                            + if left_axis == right_axis {
                                mu * gradient_dot
                            } else {
                                0.0
                            });
                    stiffness[row][column] += contribution;
                    if row != column {
                        stiffness[column][row] += contribution;
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

    use super::*;
    use crate::assembly::elements::solid::tetrahedron10::{
        jacobian_determinant, strain_displacement_matrix,
    };

    #[test]
    fn isotropic_gradient_form_matches_full_b_transpose_d_b_at_every_entry() {
        let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
        let base_nodes_m = std::array::from_fn(|index| mesh.topology.nodes[index].coordinates_m);
        let material = SolidMaterial {
            youngs_modulus_pa: 100.0,
            poisson_ratio: 0.25,
            density_kg_per_m3: 1.0,
        };
        let elasticity = elasticity_matrix(material).unwrap();
        for warp in [0.0, 0.1, 0.2] {
            let mut nodes_m = base_nodes_m;
            nodes_m[4][0] += warp;
            let geometry = Tetrahedron10ElementGeometry { nodes_m };
            for barycentric in [[0.25; 4], [0.1, 0.2, 0.3, 0.4]] {
                let mut actual =
                    [[0.0; TETRAHEDRON10_ELEMENT_DOF_COUNT]; TETRAHEDRON10_ELEMENT_DOF_COUNT];
                add_point(&mut actual, geometry, &elasticity, barycentric, 1.0).unwrap();
                let gradients = physical_shape_gradients_and_measure(nodes_m, barycentric)
                    .unwrap()
                    .0;
                let b = strain_displacement_matrix(gradients);
                let determinant = jacobian_determinant(nodes_m, barycentric).unwrap();
                for (row, actual_row) in actual.iter().enumerate() {
                    for (column, actual_value) in actual_row.iter().enumerate() {
                        let reference = determinant
                            * (0..6)
                                .flat_map(|stress| {
                                    (0..6).map(move |strain| {
                                        b[stress][row]
                                            * elasticity[stress][strain]
                                            * b[strain][column]
                                    })
                                })
                                .sum::<f64>();
                        assert!(
                            (actual_value - reference).abs()
                                <= 1.0e-11 * reference.abs().max(1.0),
                            "warp {warp}, point {barycentric:?}, entry ({row}, {column}): {} vs {reference}",
                            actual_value
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn strongly_curved_element_fails_closed_when_quadrature_does_not_converge() {
        let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
        let mut nodes_m = std::array::from_fn(|index| mesh.topology.nodes[index].coordinates_m);
        nodes_m[4][0] += 0.24;
        let geometry = Tetrahedron10ElementGeometry { nodes_m };
        let material = SolidMaterial {
            youngs_modulus_pa: 100.0,
            poisson_ratio: 0.25,
            density_kg_per_m3: 1.0,
        };
        assert_eq!(
            global_stiffness_matrix(material, geometry),
            Err(Tetrahedron10ElementError::UnconvergedStiffnessQuadrature)
        );
    }
}
