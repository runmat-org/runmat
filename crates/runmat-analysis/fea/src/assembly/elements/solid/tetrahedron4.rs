use runmat_simulation_usage::{LedgerError, OperationCount, UsageCategory, UsageContext};
use thiserror::Error;

use super::{
    material::{elasticity_matrix as material_elasticity_matrix, SolidMaterialError},
    quality::SolidElementQuality,
    SolidMaterial,
};

pub const TETRAHEDRON4_NODE_DOF_COUNT: usize = 3;
pub const TETRAHEDRON4_ELEMENT_NODE_COUNT: usize = 4;
pub const TETRAHEDRON4_ELEMENT_DOF_COUNT: usize =
    TETRAHEDRON4_NODE_DOF_COUNT * TETRAHEDRON4_ELEMENT_NODE_COUNT;

pub type Tetrahedron4Matrix12 =
    [[f64; TETRAHEDRON4_ELEMENT_DOF_COUNT]; TETRAHEDRON4_ELEMENT_DOF_COUNT];
pub type Tetrahedron4BMatrix = [[f64; TETRAHEDRON4_ELEMENT_DOF_COUNT]; 6];
pub type ElasticityMatrix = super::material::ElasticityMatrix;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tetrahedron4ElementGeometry {
    pub nodes_m: [[f64; 3]; TETRAHEDRON4_ELEMENT_NODE_COUNT],
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum Tetrahedron4ElementError {
    #[error("Tetrahedron4 node coordinates must be finite")]
    NonFiniteCoordinate,
    #[error("Tetrahedron4 element volume must be positive and finite")]
    DegenerateOrInverted,
    #[error("Tetrahedron4 material is invalid: {0}")]
    InvalidMaterial(#[from] SolidMaterialError),
    #[error("Tetrahedron4 numerical accounting failed: {0}")]
    Accounting(#[from] LedgerError),
}

impl Tetrahedron4ElementGeometry {
    pub fn volume_m3(self) -> Result<f64, Tetrahedron4ElementError> {
        validate_nodes(self.nodes_m)?;
        // Three vector differences (9), cross product (9), dot product (5), divide (1).
        let usage = UsageContext::capture();
        let mut batch = usage
            .as_ref()
            .map(|usage| {
                usage
                    .ledger()
                    .reserve(UsageCategory::Assembly, OperationCount::from_operations(24))
            })
            .transpose()?;
        let volume = signed_volume(self.nodes_m) / 6.0;
        if let Some(batch) = &mut batch {
            batch.complete(OperationCount::from_operations(24))?;
        }
        if volume.is_finite() && volume > 0.0 {
            Ok(volume)
        } else {
            Err(Tetrahedron4ElementError::DegenerateOrInverted)
        }
    }

    pub fn shape_function_gradients(self) -> Result<[[f64; 3]; 4], Tetrahedron4ElementError> {
        validate_nodes(self.nodes_m)?;
        // The inverse's determinant needs 23 operations; a valid inverse needs 37
        // more. Four transformed gradients then need 4 * 3 * 5 operations.
        let usage = UsageContext::capture();
        let mut batch = usage
            .as_ref()
            .map(|usage| {
                usage.ledger().reserve(
                    UsageCategory::Assembly,
                    OperationCount::from_operations(120),
                )
            })
            .transpose()?;
        let inverse = match inverse_jacobian(self.nodes_m) {
            Ok(inverse) => inverse,
            Err(error) => {
                if let Some(batch) = &mut batch {
                    batch.complete(OperationCount::from_operations(23))?;
                }
                return Err(error);
            }
        };
        let reference_gradients = [
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let mut gradients = [[0.0_f64; 3]; 4];
        for (node, reference) in reference_gradients.into_iter().enumerate() {
            for (axis, inverse_row) in inverse.iter().enumerate() {
                gradients[node][axis] = inverse_row[0] * reference[0]
                    + inverse_row[1] * reference[1]
                    + inverse_row[2] * reference[2];
            }
        }
        if let Some(batch) = &mut batch {
            batch.complete(OperationCount::from_operations(120))?;
        }
        Ok(gradients)
    }

    pub fn quality(self) -> Result<SolidElementQuality, Tetrahedron4ElementError> {
        let volume = self.volume_m3()?;
        // Six edges: three differences, three squares, two additions and one
        // square root each (54). The final aspect ratio adds one division.
        let usage = UsageContext::capture();
        let mut batch = usage
            .as_ref()
            .map(|usage| {
                usage
                    .ledger()
                    .reserve(UsageCategory::Assembly, OperationCount::from_operations(55))
            })
            .transpose()?;
        let quality = SolidElementQuality::from_tetrahedron4_nodes(self.nodes_m, volume);
        if let Some(batch) = &mut batch {
            batch.complete(OperationCount::from_operations(55))?;
        }
        Ok(quality)
    }
}

pub fn strain_displacement_matrix(
    geometry: Tetrahedron4ElementGeometry,
) -> Result<Tetrahedron4BMatrix, Tetrahedron4ElementError> {
    let gradients = geometry.shape_function_gradients()?;
    let mut b = [[0.0_f64; TETRAHEDRON4_ELEMENT_DOF_COUNT]; 6];
    for (node, gradient) in gradients.into_iter().enumerate() {
        let col = node * TETRAHEDRON4_NODE_DOF_COUNT;
        let [dn_dx, dn_dy, dn_dz] = gradient;
        b[0][col] = dn_dx;
        b[1][col + 1] = dn_dy;
        b[2][col + 2] = dn_dz;
        b[3][col + 1] = dn_dz;
        b[3][col + 2] = dn_dy;
        b[4][col] = dn_dz;
        b[4][col + 2] = dn_dx;
        b[5][col] = dn_dy;
        b[5][col + 1] = dn_dx;
    }
    Ok(b)
}

pub fn elasticity_matrix(
    material: SolidMaterial,
) -> Result<ElasticityMatrix, Tetrahedron4ElementError> {
    material_elasticity_matrix(material).map_err(Tetrahedron4ElementError::InvalidMaterial)
}

pub fn global_stiffness_matrix(
    material: SolidMaterial,
    geometry: Tetrahedron4ElementGeometry,
) -> Result<Tetrahedron4Matrix12, Tetrahedron4ElementError> {
    let volume = geometry.volume_m3()?;
    let b = strain_displacement_matrix(geometry)?;
    let d = elasticity_matrix(material)?;
    // DB: 6 * 12 cells, each six products and five reference sum additions.
    // Symmetric upper K: 78 cells, each six products, five sum additions,
    // and one volume product. Assigning the reflected value is free.
    let usage = UsageContext::capture();
    let mut batch = usage
        .as_ref()
        .map(|usage| {
            usage.ledger().reserve(
                UsageCategory::Assembly,
                OperationCount::from_operations(1_728),
            )
        })
        .transpose()?;
    let mut db = [[0.0_f64; TETRAHEDRON4_ELEMENT_DOF_COUNT]; 6];
    for row in 0..6 {
        for col in 0..TETRAHEDRON4_ELEMENT_DOF_COUNT {
            db[row][col] = (0..6).map(|idx| d[row][idx] * b[idx][col]).sum();
        }
    }

    let mut k = [[0.0_f64; TETRAHEDRON4_ELEMENT_DOF_COUNT]; TETRAHEDRON4_ELEMENT_DOF_COUNT];
    for row in 0..TETRAHEDRON4_ELEMENT_DOF_COUNT {
        for col in row..TETRAHEDRON4_ELEMENT_DOF_COUNT {
            let value = volume * (0..6).map(|idx| b[idx][row] * db[idx][col]).sum::<f64>();
            k[row][col] = value;
            k[col][row] = value;
        }
    }
    if let Some(batch) = &mut batch {
        batch.complete(OperationCount::from_operations(1_728))?;
    }
    Ok(k)
}

pub fn consistent_mass_matrix(
    material: SolidMaterial,
    geometry: Tetrahedron4ElementGeometry,
) -> Result<Tetrahedron4Matrix12, Tetrahedron4ElementError> {
    material.validate()?;
    let volume = geometry.volume_m3()?;
    // Density scaling and division (2), then one diagonal product per DOF (12).
    let usage = UsageContext::capture();
    let mut batch = usage
        .as_ref()
        .map(|usage| {
            usage
                .ledger()
                .reserve(UsageCategory::Assembly, OperationCount::from_operations(14))
        })
        .transpose()?;
    let scalar = material.density_kg_per_m3.canonical_value() * volume / 20.0;
    let mass = std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            if row % TETRAHEDRON4_NODE_DOF_COUNT != column % TETRAHEDRON4_NODE_DOF_COUNT {
                0.0
            } else if row / TETRAHEDRON4_NODE_DOF_COUNT == column / TETRAHEDRON4_NODE_DOF_COUNT {
                2.0 * scalar
            } else {
                scalar
            }
        })
    });
    if let Some(batch) = &mut batch {
        batch.complete(OperationCount::from_operations(14))?;
    }
    Ok(mass)
}

fn validate_nodes(nodes_m: [[f64; 3]; 4]) -> Result<(), Tetrahedron4ElementError> {
    if nodes_m.iter().flatten().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(Tetrahedron4ElementError::NonFiniteCoordinate)
    }
}

fn inverse_jacobian(nodes_m: [[f64; 3]; 4]) -> Result<[[f64; 3]; 3], Tetrahedron4ElementError> {
    let j = [
        sub(nodes_m[1], nodes_m[0]),
        sub(nodes_m[2], nodes_m[0]),
        sub(nodes_m[3], nodes_m[0]),
    ];
    let det = dot(j[0], cross(j[1], j[2]));
    if !det.is_finite() || det <= 0.0 {
        return Err(Tetrahedron4ElementError::DegenerateOrInverted);
    }
    let inv_det = 1.0 / det;
    Ok([
        [
            (j[1][1] * j[2][2] - j[1][2] * j[2][1]) * inv_det,
            (j[0][2] * j[2][1] - j[0][1] * j[2][2]) * inv_det,
            (j[0][1] * j[1][2] - j[0][2] * j[1][1]) * inv_det,
        ],
        [
            (j[1][2] * j[2][0] - j[1][0] * j[2][2]) * inv_det,
            (j[0][0] * j[2][2] - j[0][2] * j[2][0]) * inv_det,
            (j[0][2] * j[1][0] - j[0][0] * j[1][2]) * inv_det,
        ],
        [
            (j[1][0] * j[2][1] - j[1][1] * j[2][0]) * inv_det,
            (j[0][1] * j[2][0] - j[0][0] * j[2][1]) * inv_det,
            (j[0][0] * j[1][1] - j[0][1] * j[1][0]) * inv_det,
        ],
    ])
}

fn signed_volume(nodes_m: [[f64; 3]; 4]) -> f64 {
    dot(
        sub(nodes_m[1], nodes_m[0]),
        cross(sub(nodes_m[2], nodes_m[0]), sub(nodes_m[3], nodes_m[0])),
    )
}

fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_simulation_usage::UsageLedger;
    use std::sync::Arc;

    fn unit_tetrahedron() -> Tetrahedron4ElementGeometry {
        Tetrahedron4ElementGeometry {
            nodes_m: [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        }
    }

    fn steel() -> SolidMaterial {
        SolidMaterial {
            youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(200.0e9),
            poisson_ratio: runmat_types::physical::PoissonRatio::new(0.3),
            density_kg_per_m3: runmat_types::physical::MassDensity::new(7_850.0),
        }
    }

    #[test]
    fn tetrahedron4_volume_for_unit_tetrahedron() {
        let volume = unit_tetrahedron()
            .volume_m3()
            .expect("unit Tetrahedron volume");
        assert!((volume - 1.0 / 6.0).abs() < 1.0e-14);
    }

    #[test]
    fn tetrahedron4_gradients_are_constant_and_partition_unity() {
        let gradients = unit_tetrahedron()
            .shape_function_gradients()
            .expect("unit Tetrahedron gradients");
        assert_eq!(gradients[0], [-1.0, -1.0, -1.0]);
        assert_eq!(gradients[1], [1.0, 0.0, 0.0]);
        assert_eq!(gradients[2], [0.0, 1.0, 0.0]);
        assert_eq!(gradients[3], [0.0, 0.0, 1.0]);
        for axis in 0..3 {
            let sum = gradients.iter().map(|gradient| gradient[axis]).sum::<f64>();
            assert!(sum.abs() < 1.0e-14);
        }
    }

    #[test]
    fn tetrahedron4_b_matrix_rejects_rigid_translation_and_rotation() {
        let b = strain_displacement_matrix(unit_tetrahedron()).expect("b matrix");
        let translation = [
            2.0, -3.0, 4.0, 2.0, -3.0, 4.0, 2.0, -3.0, 4.0, 2.0, -3.0, 4.0,
        ];
        let rotation_z = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        for displacement in [translation, rotation_z] {
            for strain_row in b {
                let strain = strain_row
                    .iter()
                    .zip(displacement)
                    .map(|(lhs, rhs)| lhs * rhs)
                    .sum::<f64>();
                assert!(strain.abs() < 1.0e-14);
            }
        }
    }

    #[test]
    fn skew_tetrahedron4_gradients_reproduce_physical_coordinates() {
        let geometry = Tetrahedron4ElementGeometry {
            nodes_m: [
                [0.2, -0.4, 0.7],
                [2.1, 0.3, 0.9],
                [0.8, 1.7, 0.5],
                [0.5, 0.2, 2.4],
            ],
        };
        let gradients = geometry.shape_function_gradients().unwrap();
        for coordinate in 0..3 {
            for derivative in 0..3 {
                let value = geometry
                    .nodes_m
                    .iter()
                    .zip(gradients)
                    .map(|(node, gradient)| node[coordinate] * gradient[derivative])
                    .sum::<f64>();
                let expected = f64::from(coordinate == derivative);
                assert!((value - expected).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn tetrahedron4_stiffness_is_symmetric_and_positive_semidefinite_for_samples() {
        let stiffness = global_stiffness_matrix(steel(), unit_tetrahedron()).expect("stiffness");
        for row in 0..TETRAHEDRON4_ELEMENT_DOF_COUNT {
            for col in 0..TETRAHEDRON4_ELEMENT_DOF_COUNT {
                assert!((stiffness[row][col] - stiffness[col][row]).abs() < 1.0e-5);
            }
        }

        for displacement in [
            [1.0; TETRAHEDRON4_ELEMENT_DOF_COUNT],
            [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, -0.2, 0.4, 0.1, 0.3, -0.1, 0.2],
            [
                1.0, -2.0, 3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0, 10.0, -11.0, 12.0,
            ],
        ] {
            let energy = quadratic_form(&stiffness, displacement);
            assert!(energy >= -1.0e-3, "energy={energy}");
        }
    }

    #[test]
    fn tetrahedron4_stiffness_has_near_zero_rigid_body_residual() {
        let stiffness = global_stiffness_matrix(steel(), unit_tetrahedron()).expect("stiffness");
        let translation_x = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let residual = mat_vec(&stiffness, translation_x);
        assert!(residual.iter().all(|value| value.abs() < 1.0e-3));
    }

    #[test]
    fn inverted_tetrahedron4_is_rejected() {
        let inverted = Tetrahedron4ElementGeometry {
            nodes_m: [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        };
        assert_eq!(
            inverted
                .volume_m3()
                .expect_err("inverted Tetrahedron should fail"),
            Tetrahedron4ElementError::DegenerateOrInverted
        );
    }

    #[test]
    fn admitted_tetrahedron4_stiffness_and_mass_conserve_reference_operations() {
        // Stiffness: volume 24 + gradients 120 + material 15 + matrix 1728.
        // Mass: volume 24 + scale/diagonal 14.
        let limit = OperationCount::from_operations(1_925);
        let ledger = Arc::new(UsageLedger::new(limit, limit).unwrap());
        let context = UsageContext::new(Arc::clone(&ledger));
        context.run_sync(|| {
            global_stiffness_matrix(steel(), unit_tetrahedron()).unwrap();
            consistent_mass_matrix(steel(), unit_tetrahedron()).unwrap();
        });
        let snapshot = ledger.snapshot().unwrap();
        assert_eq!(snapshot.completed.assembly, limit);
        assert_eq!(snapshot.reserved, OperationCount::ZERO);
    }

    #[test]
    fn tetrahedron4_denial_precedes_numerical_work() {
        let zero = OperationCount::ZERO;
        let ledger = Arc::new(UsageLedger::new(zero, zero).unwrap());
        let context = UsageContext::new(Arc::clone(&ledger));
        let failure = context.run_sync(|| unit_tetrahedron().volume_m3().unwrap_err());
        assert!(matches!(
            failure,
            Tetrahedron4ElementError::Accounting(LedgerError::BudgetExceeded { .. })
        ));
        assert_eq!(ledger.snapshot().unwrap().completed.assembly, zero);
    }

    #[test]
    fn rejected_tetrahedron4_gradient_reports_completed_determinant_prefix() {
        let inverted = Tetrahedron4ElementGeometry {
            nodes_m: [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        };
        let limit = OperationCount::from_operations(120);
        let ledger = Arc::new(UsageLedger::new(limit, limit).unwrap());
        let context = UsageContext::new(Arc::clone(&ledger));
        assert_eq!(
            context.run_sync(|| inverted.shape_function_gradients().unwrap_err()),
            Tetrahedron4ElementError::DegenerateOrInverted
        );
        let snapshot = ledger.snapshot().unwrap();
        assert_eq!(
            snapshot.completed.assembly,
            OperationCount::from_operations(23)
        );
        assert_eq!(snapshot.reserved, OperationCount::ZERO);
    }

    fn quadratic_form(
        matrix: &Tetrahedron4Matrix12,
        displacement: [f64; TETRAHEDRON4_ELEMENT_DOF_COUNT],
    ) -> f64 {
        let product = mat_vec(matrix, displacement);
        product
            .into_iter()
            .zip(displacement)
            .map(|(lhs, rhs)| lhs * rhs)
            .sum()
    }

    fn mat_vec(
        matrix: &Tetrahedron4Matrix12,
        displacement: [f64; TETRAHEDRON4_ELEMENT_DOF_COUNT],
    ) -> [f64; TETRAHEDRON4_ELEMENT_DOF_COUNT] {
        let mut result = [0.0_f64; TETRAHEDRON4_ELEMENT_DOF_COUNT];
        for row in 0..TETRAHEDRON4_ELEMENT_DOF_COUNT {
            result[row] = matrix[row]
                .iter()
                .zip(displacement)
                .map(|(lhs, rhs)| lhs * rhs)
                .sum();
        }
        result
    }
}
