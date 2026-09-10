use thiserror::Error;

use runmat_meshing_core::TETRAHEDRON_MIDSIDE_EDGE_CORNERS;

use super::{
    material::{elasticity_matrix, SolidMaterialError},
    SolidMaterial,
};

pub const TETRAHEDRON10_ELEMENT_NODE_COUNT: usize = 10;
pub const TETRAHEDRON10_NODE_DOF_COUNT: usize = 3;
pub const TETRAHEDRON10_ELEMENT_DOF_COUNT: usize =
    TETRAHEDRON10_ELEMENT_NODE_COUNT * TETRAHEDRON10_NODE_DOF_COUNT;

pub type Tetrahedron10Matrix30 =
    [[f64; TETRAHEDRON10_ELEMENT_DOF_COUNT]; TETRAHEDRON10_ELEMENT_DOF_COUNT];

const BARYCENTRIC_DERIVATIVES: [[f64; 3]; 4] = [
    [-1.0, -1.0, -1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
];
const QUADRATURE_A: f64 = 0.585_410_196_624_968_5;
const QUADRATURE_B: f64 = 0.138_196_601_125_010_5;
const QUADRATURE_WEIGHT: f64 = 1.0 / 24.0;
const QUADRATURE: [[f64; 4]; 4] = [
    [QUADRATURE_A, QUADRATURE_B, QUADRATURE_B, QUADRATURE_B],
    [QUADRATURE_B, QUADRATURE_A, QUADRATURE_B, QUADRATURE_B],
    [QUADRATURE_B, QUADRATURE_B, QUADRATURE_A, QUADRATURE_B],
    [QUADRATURE_B, QUADRATURE_B, QUADRATURE_B, QUADRATURE_A],
];
const MASS_QUADRATURE: [([f64; 4], f64); 11] = [
    ([0.25, 0.25, 0.25, 0.25], -0.013_155_555_555_555_556),
    (
        [
            0.785_714_285_714_285_7,
            0.071_428_571_428_571_42,
            0.071_428_571_428_571_42,
            0.071_428_571_428_571_42,
        ],
        0.007_622_222_222_222_222,
    ),
    (
        [
            0.071_428_571_428_571_42,
            0.785_714_285_714_285_7,
            0.071_428_571_428_571_42,
            0.071_428_571_428_571_42,
        ],
        0.007_622_222_222_222_222,
    ),
    (
        [
            0.071_428_571_428_571_42,
            0.071_428_571_428_571_42,
            0.785_714_285_714_285_7,
            0.071_428_571_428_571_42,
        ],
        0.007_622_222_222_222_222,
    ),
    (
        [
            0.071_428_571_428_571_42,
            0.071_428_571_428_571_42,
            0.071_428_571_428_571_42,
            0.785_714_285_714_285_7,
        ],
        0.007_622_222_222_222_222,
    ),
    (
        [
            0.399_403_576_166_799_2,
            0.399_403_576_166_799_2,
            0.100_596_423_833_200_8,
            0.100_596_423_833_200_8,
        ],
        0.024_888_888_888_888_89,
    ),
    (
        [
            0.399_403_576_166_799_2,
            0.100_596_423_833_200_8,
            0.399_403_576_166_799_2,
            0.100_596_423_833_200_8,
        ],
        0.024_888_888_888_888_89,
    ),
    (
        [
            0.399_403_576_166_799_2,
            0.100_596_423_833_200_8,
            0.100_596_423_833_200_8,
            0.399_403_576_166_799_2,
        ],
        0.024_888_888_888_888_89,
    ),
    (
        [
            0.100_596_423_833_200_8,
            0.399_403_576_166_799_2,
            0.399_403_576_166_799_2,
            0.100_596_423_833_200_8,
        ],
        0.024_888_888_888_888_89,
    ),
    (
        [
            0.100_596_423_833_200_8,
            0.399_403_576_166_799_2,
            0.100_596_423_833_200_8,
            0.399_403_576_166_799_2,
        ],
        0.024_888_888_888_888_89,
    ),
    (
        [
            0.100_596_423_833_200_8,
            0.100_596_423_833_200_8,
            0.399_403_576_166_799_2,
            0.399_403_576_166_799_2,
        ],
        0.024_888_888_888_89,
    ),
];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tetrahedron10ElementGeometry {
    /// Corners 0..4 followed by edges 01, 12, 20, 03, 13, and 23.
    pub nodes_m: [[f64; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT],
}

impl Tetrahedron10ElementGeometry {
    pub fn volume_m3(self) -> Result<f64, Tetrahedron10ElementError> {
        validate_nodes(self.nodes_m)?;
        let volume =
            MASS_QUADRATURE
                .into_iter()
                .try_fold(0.0, |volume, (barycentric, weight)| {
                    Ok::<_, Tetrahedron10ElementError>(
                        volume + weight * jacobian_determinant(self.nodes_m, barycentric)?,
                    )
                })?;
        if volume.is_finite() && volume > 0.0 {
            Ok(volume)
        } else {
            Err(Tetrahedron10ElementError::DegenerateOrInverted)
        }
    }

    pub fn volume_average_strain(
        self,
        displacement: [f64; TETRAHEDRON10_ELEMENT_DOF_COUNT],
    ) -> Result<[f64; 6], Tetrahedron10ElementError> {
        validate_nodes(self.nodes_m)?;
        if displacement.iter().any(|value| !value.is_finite()) {
            return Err(Tetrahedron10ElementError::NonFiniteDisplacement);
        }
        let mut integrated = [0.0; 6];
        let mut volume = 0.0;
        for (barycentric, weight) in MASS_QUADRATURE {
            let gradients = physical_shape_gradients(self.nodes_m, barycentric)?;
            let b = strain_displacement_matrix(gradients);
            let weighted_volume = weight * jacobian_determinant(self.nodes_m, barycentric)?;
            for component in 0..6 {
                integrated[component] += weighted_volume
                    * b[component]
                        .iter()
                        .zip(displacement)
                        .map(|(shape, value)| shape * value)
                        .sum::<f64>();
            }
            volume += weighted_volume;
        }
        if !volume.is_finite() || volume <= 0.0 || integrated.iter().any(|value| !value.is_finite())
        {
            return Err(Tetrahedron10ElementError::DegenerateOrInverted);
        }
        Ok(integrated.map(|value| value / volume))
    }
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum Tetrahedron10ElementError {
    #[error("Tetrahedron10 node coordinates must be finite")]
    NonFiniteCoordinate,
    #[error("Tetrahedron10 nodal displacement must be finite")]
    NonFiniteDisplacement,
    #[error("Tetrahedron10 quadrature Jacobian must be positive and finite")]
    DegenerateOrInverted,
    #[error("Tetrahedron10 material is invalid: {0}")]
    InvalidMaterial(#[from] SolidMaterialError),
}

pub fn global_stiffness_matrix(
    material: SolidMaterial,
    geometry: Tetrahedron10ElementGeometry,
) -> Result<Tetrahedron10Matrix30, Tetrahedron10ElementError> {
    validate_nodes(geometry.nodes_m)?;
    let elasticity = elasticity_matrix(material)?;
    let mut stiffness = [[0.0; TETRAHEDRON10_ELEMENT_DOF_COUNT]; TETRAHEDRON10_ELEMENT_DOF_COUNT];
    for barycentric in QUADRATURE {
        let gradients = physical_shape_gradients(geometry.nodes_m, barycentric)?;
        let b = strain_displacement_matrix(gradients);
        let determinant = jacobian_determinant(geometry.nodes_m, barycentric)?;
        let scale = QUADRATURE_WEIGHT * determinant;
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
    }
    if stiffness.iter().flatten().all(|value| value.is_finite()) {
        Ok(stiffness)
    } else {
        Err(Tetrahedron10ElementError::DegenerateOrInverted)
    }
}

pub fn consistent_mass_matrix(
    material: SolidMaterial,
    geometry: Tetrahedron10ElementGeometry,
) -> Result<Tetrahedron10Matrix30, Tetrahedron10ElementError> {
    validate_nodes(geometry.nodes_m)?;
    material.validate()?;
    let mut mass = [[0.0; TETRAHEDRON10_ELEMENT_DOF_COUNT]; TETRAHEDRON10_ELEMENT_DOF_COUNT];
    for (barycentric, weight) in MASS_QUADRATURE {
        let shape = shape_values(barycentric);
        let scale = material.density_kg_per_m3
            * weight
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

fn shape_values(barycentric: [f64; 4]) -> [f64; TETRAHEDRON10_ELEMENT_NODE_COUNT] {
    let mut values = [0.0; TETRAHEDRON10_ELEMENT_NODE_COUNT];
    for node in 0..4 {
        values[node] = barycentric[node] * (2.0 * barycentric[node] - 1.0);
    }
    for (local_edge, [left, right]) in TETRAHEDRON_MIDSIDE_EDGE_CORNERS.into_iter().enumerate() {
        values[4 + local_edge] = 4.0 * barycentric[left] * barycentric[right];
    }
    values
}

fn physical_shape_gradients(
    nodes_m: [[f64; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT],
    barycentric: [f64; 4],
) -> Result<[[f64; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT], Tetrahedron10ElementError> {
    let reference = reference_shape_gradients(barycentric);
    let inverse = inverse_jacobian(nodes_m, reference)?;
    Ok(reference.map(|gradient| {
        std::array::from_fn(|axis| {
            inverse[0][axis] * gradient[0]
                + inverse[1][axis] * gradient[1]
                + inverse[2][axis] * gradient[2]
        })
    }))
}

fn reference_shape_gradients(
    barycentric: [f64; 4],
) -> [[f64; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT] {
    let mut gradients = [[0.0; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT];
    for node in 0..4 {
        let factor = 4.0 * barycentric[node] - 1.0;
        gradients[node] = BARYCENTRIC_DERIVATIVES[node].map(|derivative| factor * derivative);
    }
    for (local_edge, [left, right]) in TETRAHEDRON_MIDSIDE_EDGE_CORNERS.into_iter().enumerate() {
        gradients[4 + local_edge] = std::array::from_fn(|axis| {
            4.0 * (barycentric[right] * BARYCENTRIC_DERIVATIVES[left][axis]
                + barycentric[left] * BARYCENTRIC_DERIVATIVES[right][axis])
        });
    }
    gradients
}

fn strain_displacement_matrix(
    gradients: [[f64; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT],
) -> [[f64; TETRAHEDRON10_ELEMENT_DOF_COUNT]; 6] {
    let mut b = [[0.0; TETRAHEDRON10_ELEMENT_DOF_COUNT]; 6];
    for (node, [dn_dx, dn_dy, dn_dz]) in gradients.into_iter().enumerate() {
        let column = node * TETRAHEDRON10_NODE_DOF_COUNT;
        b[0][column] = dn_dx;
        b[1][column + 1] = dn_dy;
        b[2][column + 2] = dn_dz;
        b[3][column + 1] = dn_dz;
        b[3][column + 2] = dn_dy;
        b[4][column] = dn_dz;
        b[4][column + 2] = dn_dx;
        b[5][column] = dn_dy;
        b[5][column + 1] = dn_dx;
    }
    b
}

fn jacobian_determinant(
    nodes_m: [[f64; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT],
    barycentric: [f64; 4],
) -> Result<f64, Tetrahedron10ElementError> {
    let reference = reference_shape_gradients(barycentric);
    let jacobian = jacobian(nodes_m, reference);
    let determinant = dot(jacobian[0], cross(jacobian[1], jacobian[2]));
    if determinant.is_finite() && determinant > 0.0 {
        Ok(determinant)
    } else {
        Err(Tetrahedron10ElementError::DegenerateOrInverted)
    }
}

fn inverse_jacobian(
    nodes_m: [[f64; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT],
    reference: [[f64; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT],
) -> Result<[[f64; 3]; 3], Tetrahedron10ElementError> {
    let matrix = jacobian(nodes_m, reference);
    let determinant = dot(matrix[0], cross(matrix[1], matrix[2]));
    if !determinant.is_finite() || determinant <= 0.0 {
        return Err(Tetrahedron10ElementError::DegenerateOrInverted);
    }
    let inverse_determinant = 1.0 / determinant;
    Ok([
        [
            (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) * inverse_determinant,
            (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) * inverse_determinant,
            (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) * inverse_determinant,
        ],
        [
            (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) * inverse_determinant,
            (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) * inverse_determinant,
            (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]) * inverse_determinant,
        ],
        [
            (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) * inverse_determinant,
            (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]) * inverse_determinant,
            (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) * inverse_determinant,
        ],
    ])
}

fn jacobian(
    nodes_m: [[f64; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT],
    reference: [[f64; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT],
) -> [[f64; 3]; 3] {
    std::array::from_fn(|reference_axis| {
        std::array::from_fn(|physical_axis| {
            (0..TETRAHEDRON10_ELEMENT_NODE_COUNT)
                .map(|node| nodes_m[node][physical_axis] * reference[node][reference_axis])
                .sum()
        })
    })
}

fn validate_nodes(
    nodes_m: [[f64; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT],
) -> Result<(), Tetrahedron10ElementError> {
    if nodes_m.iter().flatten().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(Tetrahedron10ElementError::NonFiniteCoordinate)
    }
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

    fn straight_unit_tetrahedron() -> Tetrahedron10ElementGeometry {
        let corners = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        Tetrahedron10ElementGeometry {
            nodes_m: [
                corners[0],
                corners[1],
                corners[2],
                corners[3],
                midpoint(corners[0], corners[1]),
                midpoint(corners[1], corners[2]),
                midpoint(corners[2], corners[0]),
                midpoint(corners[0], corners[3]),
                midpoint(corners[1], corners[3]),
                midpoint(corners[2], corners[3]),
            ],
        }
    }

    fn steel() -> SolidMaterial {
        SolidMaterial {
            youngs_modulus_pa: 200.0e9,
            poisson_ratio: 0.3,
            density_kg_per_m3: 7_850.0,
        }
    }

    #[test]
    fn quadratic_shape_gradients_preserve_partition_of_unity() {
        for barycentric in QUADRATURE {
            let gradients = reference_shape_gradients(barycentric);
            for axis in 0..3 {
                assert!(
                    gradients
                        .iter()
                        .map(|gradient| gradient[axis])
                        .sum::<f64>()
                        .abs()
                        < 1.0e-14
                );
            }
        }
    }

    #[test]
    fn straight_tetrahedron10_stiffness_is_symmetric_and_rejects_rigid_translation() {
        let stiffness = global_stiffness_matrix(steel(), straight_unit_tetrahedron()).unwrap();
        for row in 0..TETRAHEDRON10_ELEMENT_DOF_COUNT {
            for column in 0..TETRAHEDRON10_ELEMENT_DOF_COUNT {
                assert!((stiffness[row][column] - stiffness[column][row]).abs() < 1.0e-4);
            }
        }
        let mut translation = [0.0; TETRAHEDRON10_ELEMENT_DOF_COUNT];
        for displacement in translation.chunks_exact_mut(3) {
            displacement.copy_from_slice(&[1.0, -2.0, 3.0]);
        }
        let residual = std::array::from_fn::<_, TETRAHEDRON10_ELEMENT_DOF_COUNT, _>(|row| {
            stiffness[row]
                .iter()
                .zip(translation)
                .map(|(left, right)| left * right)
                .sum::<f64>()
        });
        assert!(residual.iter().all(|value| value.abs() < 2.0e-3));
    }

    #[test]
    fn volume_average_strain_recovers_a_quadratic_displacement_field() {
        let geometry = straight_unit_tetrahedron();
        let displacement = std::array::from_fn(|dof| {
            let node = dof / TETRAHEDRON10_NODE_DOF_COUNT;
            let axis = dof % TETRAHEDRON10_NODE_DOF_COUNT;
            if axis == 0 {
                geometry.nodes_m[node][0].powi(2)
            } else {
                0.0
            }
        });
        let strain = geometry.volume_average_strain(displacement).unwrap();
        assert!((geometry.volume_m3().unwrap() - 1.0 / 6.0).abs() < 1.0e-14);
        assert!((strain[0] - 0.5).abs() < 1.0e-13);
        assert!(strain[1..].iter().all(|value| value.abs() < 1.0e-13));
    }

    #[test]
    fn tetrahedron10_rejects_nonfinite_and_inverted_geometry() {
        let mut nonfinite = straight_unit_tetrahedron();
        nonfinite.nodes_m[4][0] = f64::NAN;
        assert_eq!(
            global_stiffness_matrix(steel(), nonfinite).unwrap_err(),
            Tetrahedron10ElementError::NonFiniteCoordinate
        );

        let mut inverted = straight_unit_tetrahedron();
        inverted.nodes_m.swap(1, 2);
        inverted.nodes_m.swap(4, 6);
        inverted.nodes_m.swap(8, 9);
        assert_eq!(
            global_stiffness_matrix(steel(), inverted).unwrap_err(),
            Tetrahedron10ElementError::DegenerateOrInverted
        );
    }

    fn midpoint(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|axis| (left[axis] + right[axis]) * 0.5)
    }
}
