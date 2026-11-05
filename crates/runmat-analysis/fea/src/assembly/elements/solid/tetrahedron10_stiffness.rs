//! Material stiffness of isoparametric Tet10 solids, formed from the shape-gradient moments that
//! the shared Tet10 validity rule integrates (`runmat_meshing_core::tetrahedron10_stiffness_moments`).
//! The rule chooses the quadrature (exact four-point for affine elements, the converged tensor
//! Gauss rule for curved ones) and rejects elements whose quadrature does not converge, so the
//! mesher and this assembly accept the same elements. Forming K from the moments is valid only
//! because this element's elasticity tensor is constant over it; consumers with per-point
//! material state integrate on the shared rule's converged points instead.

use runmat_simulation_usage::{OperationCount, UsageCategory, UsageContext};

use super::{
    material::elasticity_matrix,
    tetrahedron10::{
        validate_nodes, Tetrahedron10CapacityError, Tetrahedron10ElementError,
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
    let moments = runmat_meshing_core::tetrahedron10_stiffness_moments_for_category(
        &geometry.nodes_m,
        UsageCategory::Assembly,
    )
    .map_err(map_moments_error)?;
    finish_stiffness(elasticity, &moments)
}

fn map_moments_error(
    invalidity: runmat_meshing_core::Tet10ElementInvalidity,
) -> Tetrahedron10ElementError {
    match invalidity {
        runmat_meshing_core::Tet10ElementInvalidity::InvertedOrDegenerate => {
            Tetrahedron10ElementError::DegenerateOrInverted
        }
        runmat_meshing_core::Tet10ElementInvalidity::InsufficientJacobianMargin => {
            Tetrahedron10ElementError::UncertifiedJacobian
        }
        runmat_meshing_core::Tet10ElementInvalidity::UnconvergedStiffnessQuadrature => {
            Tetrahedron10ElementError::UnconvergedStiffnessQuadrature
        }
        runmat_meshing_core::Tet10ElementInvalidity::Accounting(error) => {
            Tetrahedron10ElementError::Accounting(error)
        }
    }
}

pub async fn global_stiffness_matrix_with_capacity<E>(
    material: SolidMaterial,
    geometry: Tetrahedron10ElementGeometry,
    owner: &dyn runmat_simulation_usage::AsyncCapacityOwner<E>,
) -> Result<Tetrahedron10Matrix30, Tetrahedron10CapacityError<E>> {
    super::tetrahedron10_geometry::validate_nodes_with_capacity(geometry.nodes_m, owner).await?;
    owner
        .ensure_capacity(OperationCount::from_operations(15))
        .await
        .map_err(Tetrahedron10CapacityError::Admission)?;
    let elasticity = elasticity_matrix(material)?;
    let moments = runmat_meshing_core::tetrahedron10_stiffness_moments_with_capacity(
        &geometry.nodes_m,
        UsageCategory::Assembly,
        Some(owner),
    )
    .await
    .map_err(|error| match error {
        runmat_meshing_core::Tet10CapacityError::Invalidity(error) => {
            Tetrahedron10CapacityError::Numerical(map_moments_error(error))
        }
        runmat_meshing_core::Tet10CapacityError::Admission(error) => {
            Tetrahedron10CapacityError::Admission(error)
        }
    })?;
    owner
        .ensure_capacity(OperationCount::from_operations(4_200))
        .await
        .map_err(Tetrahedron10CapacityError::Admission)?;
    finish_stiffness(elasticity, &moments).map_err(Tetrahedron10CapacityError::Numerical)
}

fn finish_stiffness(
    elasticity: super::material::ElasticityMatrix,
    moments: &runmat_meshing_core::Tet10StiffnessMoments,
) -> Result<Tetrahedron10Matrix30, Tetrahedron10ElementError> {
    let lambda = elasticity[0][1];
    let mu = elasticity[3][3];
    // Final 30x30 contraction only. Each cell has two products and two
    // additions; 300 diagonal-component cells additionally sum three moments
    // with two additions. Shared moment integration is a separate obligation.
    let usage = UsageContext::capture();
    let mut batch = usage
        .as_ref()
        .map(|usage| {
            usage.ledger().reserve(
                UsageCategory::Assembly,
                OperationCount::from_operations(4_200),
            )
        })
        .transpose()?;
    // The isotropic BᵀDB block is λ∂ₐNᵢ∂ᵦNⱼ + μ(δₐᵦ∇Nᵢ·∇Nⱼ + ∂ᵦNᵢ∂ₐNⱼ), integrated term by term.
    let stiffness: Tetrahedron10Matrix30 = std::array::from_fn(|row| {
        let (i, a) = (row / 3, row % 3);
        std::array::from_fn(|column| {
            let (j, b) = (column / 3, column % 3);
            let gradient_dot = if a == b {
                (0..3).map(|c| moments.moment(i, c, j, c)).sum::<f64>()
            } else {
                0.0
            };
            lambda * moments.moment(i, a, j, b) + mu * (moments.moment(i, b, j, a) + gradient_dot)
        })
    });
    debug_assert_eq!(stiffness.len(), TETRAHEDRON10_ELEMENT_DOF_COUNT);
    if let Some(batch) = &mut batch {
        batch.complete(OperationCount::from_operations(4_200))?;
    }
    if stiffness.iter().flatten().all(|value| value.is_finite()) {
        Ok(stiffness)
    } else {
        Err(Tetrahedron10ElementError::DegenerateOrInverted)
    }
}

#[cfg(test)]
mod tests {
    use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

    use super::*;
    use crate::assembly::elements::solid::tetrahedron10::{
        jacobian_determinant, physical_shape_gradients_and_measure, strain_displacement_matrix,
    };

    /// Full BᵀDB det(J) summed over the rule the shared validity rule chose.
    fn full_b_transpose_d_b(
        nodes_m: [[f64; 3]; 10],
        elasticity: &[[f64; 6]; 6],
        rule: runmat_meshing_core::Tet10StiffnessRule,
    ) -> Tetrahedron10Matrix30 {
        use crate::topology::quadrature::{
            tensor_tetrahedron_rule, GAUSS_LEGENDRE_SIXTEEN_UNIT, GAUSS_LEGENDRE_TWELVE_UNIT,
            TETRAHEDRON_FOUR_POINT_RULE,
        };
        let points: Vec<([f64; 4], f64)> = match rule {
            runmat_meshing_core::Tet10StiffnessRule::Affine => TETRAHEDRON_FOUR_POINT_RULE
                .iter()
                .map(|point| (*point, 1.0 / 24.0))
                .collect(),
            runmat_meshing_core::Tet10StiffnessRule::GaussTwelve => {
                tensor_tetrahedron_rule(&GAUSS_LEGENDRE_TWELVE_UNIT).collect()
            }
            runmat_meshing_core::Tet10StiffnessRule::GaussSixteen => {
                tensor_tetrahedron_rule(&GAUSS_LEGENDRE_SIXTEEN_UNIT).collect()
            }
        };
        let mut stiffness =
            [[0.0; TETRAHEDRON10_ELEMENT_DOF_COUNT]; TETRAHEDRON10_ELEMENT_DOF_COUNT];
        for (barycentric, weight) in points {
            let gradients = physical_shape_gradients_and_measure(nodes_m, barycentric)
                .unwrap()
                .0;
            let b = strain_displacement_matrix(gradients);
            let scale = weight * jacobian_determinant(nodes_m, barycentric).unwrap();
            for (row, stiffness_row) in stiffness.iter_mut().enumerate() {
                for (column, value) in stiffness_row.iter_mut().enumerate() {
                    *value += scale
                        * (0..6)
                            .flat_map(|stress| {
                                (0..6).map(move |strain| {
                                    b[stress][row] * elasticity[stress][strain] * b[strain][column]
                                })
                            })
                            .sum::<f64>();
                }
            }
        }
        stiffness
    }

    #[test]
    fn stiffness_from_shared_moments_matches_full_b_transpose_d_b_at_every_entry() {
        let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
        let base_nodes_m =
            std::array::from_fn(|index| mesh.topology.nodes[index].coordinates_m.canonical_value());
        let material = SolidMaterial {
            youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(100.0),
            poisson_ratio: runmat_types::physical::PoissonRatio::new(0.25),
            density_kg_per_m3: runmat_types::physical::MassDensity::new(1.0),
        };
        let elasticity = elasticity_matrix(material).unwrap();
        for warp in [0.0, 0.05, 0.1] {
            let mut nodes_m: [[f64; 3]; 10] = base_nodes_m;
            nodes_m[4][0] += warp;
            let rule = runmat_meshing_core::tetrahedron10_stiffness_moments(&nodes_m)
                .unwrap()
                .rule;
            let actual =
                global_stiffness_matrix(material, Tetrahedron10ElementGeometry { nodes_m })
                    .unwrap();
            let reference = full_b_transpose_d_b(nodes_m, &elasticity, rule);
            let scale = reference
                .iter()
                .flatten()
                .fold(0.0_f64, |m, v| m.max(v.abs()));
            for (row, actual_row) in actual.iter().enumerate() {
                for (column, actual_value) in actual_row.iter().enumerate() {
                    assert!(
                        (actual_value - reference[row][column]).abs() <= 1.0e-12 * scale,
                        "warp {warp}, entry ({row}, {column}): {actual_value} vs {}",
                        reference[row][column]
                    );
                }
            }
        }
    }

    #[test]
    fn strongly_curved_element_fails_closed_under_the_shared_element_rule() {
        let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
        let mut nodes_m =
            std::array::from_fn(|index| mesh.topology.nodes[index].coordinates_m.canonical_value());
        nodes_m[4][0] += 0.24;
        let geometry = Tetrahedron10ElementGeometry { nodes_m };
        let material = SolidMaterial {
            youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(100.0),
            poisson_ratio: runmat_types::physical::PoissonRatio::new(0.25),
            density_kg_per_m3: runmat_types::physical::MassDensity::new(1.0),
        };
        assert!(!runmat_meshing_core::is_valid_tetrahedron10_element(&nodes_m).unwrap());
        assert!(matches!(
            global_stiffness_matrix(material, geometry),
            Err(Tetrahedron10ElementError::UnconvergedStiffnessQuadrature
                | Tetrahedron10ElementError::UncertifiedJacobian)
        ));
    }
}
