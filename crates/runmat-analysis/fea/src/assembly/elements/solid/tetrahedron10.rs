use thiserror::Error;

use runmat_meshing_core::TETRAHEDRON_MIDSIDE_EDGE_CORNERS;

use crate::topology::quadrature::{
    TETRAHEDRON_ELEVEN_POINT_RULE as VOLUME_QUADRATURE, TETRAHEDRON_FOUR_POINT_RULE as QUADRATURE,
};

use super::{
    jacobian_certificate::{certify_positive_jacobian, CertificateFailure},
    material::{elasticity_matrix, SolidMaterialError},
    SolidMaterial,
};

pub const TETRAHEDRON10_ELEMENT_NODE_COUNT: usize = 10;
pub const TETRAHEDRON10_NODE_DOF_COUNT: usize = 3;
pub const TETRAHEDRON10_ELEMENT_DOF_COUNT: usize =
    TETRAHEDRON10_ELEMENT_NODE_COUNT * TETRAHEDRON10_NODE_DOF_COUNT;

pub type Tetrahedron10Matrix30 =
    [[f64; TETRAHEDRON10_ELEMENT_DOF_COUNT]; TETRAHEDRON10_ELEMENT_DOF_COUNT];
pub type SymmetricStressTensor = [[f64; 3]; 3];

const BARYCENTRIC_DERIVATIVES: [[f64; 3]; 4] = [
    [-1.0, -1.0, -1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
];
const QUADRATURE_WEIGHT: f64 = 1.0 / 24.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tetrahedron10ElementGeometry {
    /// Corners 0..4 followed by edges 01, 12, 20, 03, 13, and 23.
    pub nodes_m: [[f64; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT],
}

impl Tetrahedron10ElementGeometry {
    pub fn volume_m3(self) -> Result<f64, Tetrahedron10ElementError> {
        validate_nodes(self.nodes_m)?;
        let volume =
            VOLUME_QUADRATURE
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

    /// Strain and physical integration weight at each stiffness quadrature point.
    ///
    /// The ordering is the canonical four-point tetrahedral rule used by stiffness assembly.
    /// Keeping the Jacobian-weighted measure beside each value prevents recovery and error
    /// estimation from silently treating curved isoparametric elements as affine tetrahedra.
    pub(crate) fn quadrature_strain(
        self,
        displacement: [f64; TETRAHEDRON10_ELEMENT_DOF_COUNT],
    ) -> Result<[Tetrahedron10RecoveryPoint; 4], Tetrahedron10ElementError> {
        validate_nodes(self.nodes_m)?;
        if displacement.iter().any(|value| !value.is_finite()) {
            return Err(Tetrahedron10ElementError::NonFiniteDisplacement);
        }
        let mut recovered = [Tetrahedron10RecoveryPoint {
            barycentric: [0.0; 4],
            integration_weight_m3: 0.0,
            strain: [0.0; 6],
        }; 4];
        for (index, point) in recovered.iter_mut().enumerate() {
            let barycentric = QUADRATURE[index];
            let gradients = physical_shape_gradients(self.nodes_m, barycentric)?;
            let b = strain_displacement_matrix(gradients);
            let strain: [f64; 6] = std::array::from_fn(|component| {
                b[component]
                    .iter()
                    .zip(displacement)
                    .map(|(shape, value)| shape * value)
                    .sum()
            });
            let integration_weight_m3 =
                QUADRATURE_WEIGHT * jacobian_determinant(self.nodes_m, barycentric)?;
            if strain.iter().all(|value| value.is_finite())
                && integration_weight_m3.is_finite()
                && integration_weight_m3 > 0.0
            {
                *point = Tetrahedron10RecoveryPoint {
                    barycentric,
                    integration_weight_m3,
                    strain,
                };
            } else {
                return Err(Tetrahedron10ElementError::DegenerateOrInverted);
            }
        }
        Ok(recovered)
    }

    pub(crate) fn quadrature_integration_data(
        self,
    ) -> Result<[([f64; 4], f64); 4], Tetrahedron10ElementError> {
        validate_nodes(self.nodes_m)?;
        let mut data = [([0.0; 4], 0.0); 4];
        for (index, entry) in data.iter_mut().enumerate() {
            let barycentric = QUADRATURE[index];
            let weight = QUADRATURE_WEIGHT * jacobian_determinant(self.nodes_m, barycentric)?;
            if !weight.is_finite() || weight <= 0.0 {
                return Err(Tetrahedron10ElementError::DegenerateOrInverted);
            }
            *entry = (barycentric, weight);
        }
        Ok(data)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Tetrahedron10RecoveryPoint {
    pub barycentric: [f64; 4],
    pub integration_weight_m3: f64,
    pub strain: [f64; 6],
}

/// Extrapolates the canonical four integration-point samples to the ten element nodes.
///
/// A linear field in barycentric coordinates is fitted exactly at the four symmetric points;
/// midside values are then evaluated from that same field. This is deliberately an element-local
/// operation. Region-side smoothing, if requested, is performed by the estimator so material
/// interfaces cannot be averaged together accidentally.
pub(crate) fn extrapolate_quadrature_to_nodes(
    samples: [[f64; 6]; 4],
) -> [[f64; 6]; TETRAHEDRON10_ELEMENT_NODE_COUNT] {
    let diagonal = QUADRATURE[0][0];
    let off_diagonal = QUADRATURE[0][1];
    let scale = diagonal - off_diagonal;
    let corners = std::array::from_fn::<_, 4, _>(|corner| {
        std::array::from_fn(|component| {
            let sum = samples.iter().map(|sample| sample[component]).sum::<f64>();
            (samples[corner][component] - off_diagonal * sum) / scale
        })
    });
    std::array::from_fn(|node| {
        if node < 4 {
            corners[node]
        } else {
            let [left, right] = TETRAHEDRON_MIDSIDE_EDGE_CORNERS[node - 4];
            std::array::from_fn(|component| {
                0.5 * (corners[left][component] + corners[right][component])
            })
        }
    })
}

pub(crate) fn interpolate_nodal_samples(
    nodal: [[f64; 6]; TETRAHEDRON10_ELEMENT_NODE_COUNT],
    barycentric: [f64; 4],
) -> [f64; 6] {
    let shape = shape_values(barycentric);
    std::array::from_fn(|component| {
        shape
            .iter()
            .zip(nodal)
            .map(|(shape, value)| shape * value[component])
            .sum()
    })
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum Tetrahedron10ElementError {
    #[error("Tetrahedron10 node coordinates must be finite")]
    NonFiniteCoordinate,
    #[error("Tetrahedron10 nodal displacement must be finite")]
    NonFiniteDisplacement,
    #[error("Tetrahedron10 Jacobian must be positive and finite throughout the element")]
    DegenerateOrInverted,
    #[error("Tetrahedron10 Jacobian could not be certified positive throughout the element")]
    UncertifiedJacobian,
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

/// Integrates the initial-stress stiffness using the same isoparametric geometry and quadrature
/// as the material stiffness. Stress is tension-positive and is supplied at each quadrature point.
pub fn geometric_stiffness_matrix(
    geometry: Tetrahedron10ElementGeometry,
    stresses: [SymmetricStressTensor; 4],
) -> Result<Tetrahedron10Matrix30, Tetrahedron10ElementError> {
    validate_nodes(geometry.nodes_m)?;
    let mut matrix = [[0.0; TETRAHEDRON10_ELEMENT_DOF_COUNT]; TETRAHEDRON10_ELEMENT_DOF_COUNT];
    for (quadrature_index, barycentric) in QUADRATURE.into_iter().enumerate() {
        let gradients = physical_shape_gradients(geometry.nodes_m, barycentric)?;
        let scale = QUADRATURE_WEIGHT * jacobian_determinant(geometry.nodes_m, barycentric)?;
        for left_node in 0..TETRAHEDRON10_ELEMENT_NODE_COUNT {
            for right_node in 0..TETRAHEDRON10_ELEMENT_NODE_COUNT {
                let scalar = scale
                    * quadratic_form(
                        gradients[left_node],
                        stresses[quadrature_index],
                        gradients[right_node],
                    );
                for axis in 0..TETRAHEDRON10_NODE_DOF_COUNT {
                    matrix[left_node * 3 + axis][right_node * 3 + axis] += scalar;
                }
            }
        }
    }
    if matrix.iter().flatten().all(|value| value.is_finite()) {
        Ok(matrix)
    } else {
        Err(Tetrahedron10ElementError::DegenerateOrInverted)
    }
}

/// Full signed stress at the stiffness quadrature points for a nodal displacement state.
pub fn quadrature_stress(
    material: SolidMaterial,
    geometry: Tetrahedron10ElementGeometry,
    displacement: [f64; TETRAHEDRON10_ELEMENT_DOF_COUNT],
) -> Result<[SymmetricStressTensor; 4], Tetrahedron10ElementError> {
    validate_nodes(geometry.nodes_m)?;
    if displacement.iter().any(|value| !value.is_finite()) {
        return Err(Tetrahedron10ElementError::NonFiniteDisplacement);
    }
    let elasticity = elasticity_matrix(material)?;
    let mut stresses = [[[0.0; 3]; 3]; 4];
    for (index, barycentric) in QUADRATURE.into_iter().enumerate() {
        let gradients = physical_shape_gradients(geometry.nodes_m, barycentric)?;
        let b = strain_displacement_matrix(gradients);
        let strain = std::array::from_fn::<_, 6, _>(|row| {
            b[row]
                .iter()
                .zip(displacement)
                .map(|(left, right)| left * right)
                .sum::<f64>()
        });
        let stress = std::array::from_fn::<_, 6, _>(|row| {
            elasticity[row]
                .iter()
                .zip(strain)
                .map(|(left, right)| left * right)
                .sum::<f64>()
        });
        stresses[index] = voigt_stress_tensor(stress);
    }
    Ok(stresses)
}

fn voigt_stress_tensor(stress: [f64; 6]) -> SymmetricStressTensor {
    [
        [stress[0], stress[5], stress[4]],
        [stress[5], stress[1], stress[3]],
        [stress[4], stress[3], stress[2]],
    ]
}

fn quadratic_form(left: [f64; 3], matrix: [[f64; 3]; 3], right: [f64; 3]) -> f64 {
    (0..3)
        .map(|row| {
            left[row]
                * (0..3)
                    .map(|column| matrix[row][column] * right[column])
                    .sum::<f64>()
        })
        .sum()
}

pub(super) fn shape_values(barycentric: [f64; 4]) -> [f64; TETRAHEDRON10_ELEMENT_NODE_COUNT] {
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
            inverse[axis][0] * gradient[0]
                + inverse[axis][1] * gradient[1]
                + inverse[axis][2] * gradient[2]
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

pub(super) fn jacobian_determinant(
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

pub(super) fn validate_nodes(
    nodes_m: [[f64; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT],
) -> Result<(), Tetrahedron10ElementError> {
    if !nodes_m.iter().flatten().all(|value| value.is_finite()) {
        return Err(Tetrahedron10ElementError::NonFiniteCoordinate);
    }
    let vertex_jacobians = std::array::from_fn(|vertex| {
        let mut barycentric = [0.0; 4];
        barycentric[vertex] = 1.0;
        jacobian(nodes_m, reference_shape_gradients(barycentric))
    });
    certify_positive_jacobian(vertex_jacobians).map_err(|failure| match failure {
        CertificateFailure::InvertedOrDegenerate => Tetrahedron10ElementError::DegenerateOrInverted,
        CertificateFailure::Inconclusive => Tetrahedron10ElementError::UncertifiedJacobian,
    })
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
    use super::super::tetrahedron10_mass::consistent_mass_matrix;
    use super::*;
    use crate::assembly::elements::solid::Tetrahedron4ElementGeometry;

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
    fn skew_tetrahedron10_gradients_reproduce_physical_coordinates() {
        let corners = [
            [0.2, -0.4, 0.7],
            [2.1, 0.3, 0.9],
            [0.8, 1.7, 0.5],
            [0.5, 0.2, 2.4],
        ];
        let geometry = Tetrahedron10ElementGeometry {
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
        };
        for barycentric in QUADRATURE {
            let gradients = physical_shape_gradients(geometry.nodes_m, barycentric).unwrap();
            for coordinate in 0..3 {
                for derivative in 0..3 {
                    let value = geometry
                        .nodes_m
                        .iter()
                        .zip(gradients)
                        .map(|(node, gradient)| node[coordinate] * gradient[derivative])
                        .sum::<f64>();
                    let expected = f64::from(coordinate == derivative);
                    assert!((value - expected).abs() < 1.0e-11);
                }
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
    fn tetrahedron10_geometric_stiffness_is_symmetric_and_load_linear() {
        let geometry = straight_unit_tetrahedron();
        let unit = [[3.0, 0.2, 0.1], [0.2, 2.0, 0.3], [0.1, 0.3, 1.0]];
        let base = geometric_stiffness_matrix(geometry, [unit; 4]).unwrap();
        let doubled =
            geometric_stiffness_matrix(geometry, [unit.map(|row| row.map(|value| 2.0 * value)); 4])
                .unwrap();
        for row in 0..TETRAHEDRON10_ELEMENT_DOF_COUNT {
            for column in 0..TETRAHEDRON10_ELEMENT_DOF_COUNT {
                assert!((base[row][column] - base[column][row]).abs() < 1.0e-13);
                assert!((doubled[row][column] - 2.0 * base[row][column]).abs() < 1.0e-13);
            }
        }
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

    #[test]
    fn tetrahedron10_rejects_a_corner_fold_missed_by_all_quadrature_points() {
        let mut folded = straight_unit_tetrahedron();
        // The 01 midside node moves from x=0.5 to x=0.8. Along that edge,
        // det(J)=1+1.2(1-2*x), so it is negative at corner 1 even though every
        // current stiffness and volume quadrature point still has a positive determinant.
        folded.nodes_m[4][0] += 0.3;
        for barycentric in QUADRATURE {
            assert!(jacobian_determinant(folded.nodes_m, barycentric).is_ok());
        }
        for (barycentric, _) in VOLUME_QUADRATURE {
            assert!(jacobian_determinant(folded.nodes_m, barycentric).is_ok());
        }
        assert_eq!(
            validate_nodes(folded.nodes_m),
            Err(Tetrahedron10ElementError::DegenerateOrInverted)
        );
        assert_eq!(
            global_stiffness_matrix(steel(), folded),
            Err(Tetrahedron10ElementError::DegenerateOrInverted)
        );
        assert_eq!(
            consistent_mass_matrix(steel(), folded),
            Err(Tetrahedron10ElementError::DegenerateOrInverted)
        );
        assert_eq!(
            folded.volume_m3(),
            Err(Tetrahedron10ElementError::DegenerateOrInverted)
        );
    }

    #[test]
    fn curved_tetrahedron10_mass_matches_exact_barycentric_moments() {
        let mut geometry = straight_unit_tetrahedron();
        let warp = 0.1;
        geometry.nodes_m[4][0] += warp;
        let mass = consistent_mass_matrix(steel(), geometry).unwrap();
        let density = steel().density_kg_per_m3;
        // For this warp det(J)=1+4δ(L0-L1). Exact simplex moments give
        // ∫N0² dV = 1/420 + δ/315 and ∫N4² dV = 4/315.
        assert!((mass[0][0] - density * (1.0 / 420.0 + warp / 315.0)).abs() < 1.0e-10);
        assert!((mass[12][12] - density * 4.0 / 315.0).abs() < 1.0e-10);
        let rigid_translation_mass = (0..10)
            .flat_map(|left| (0..10).map(move |right| mass[left * 3][right * 3]))
            .sum::<f64>();
        assert!((rigid_translation_mass - density * geometry.volume_m3().unwrap()).abs() < 1.0e-9);
    }

    #[test]
    fn quadratic_recovery_exhibits_expected_h_and_p_convergence() {
        let tet4_coarse = manufactured_quadratic_recovery_error(1, false);
        let tet4_medium = manufactured_quadratic_recovery_error(2, false);
        let tet4_fine = manufactured_quadratic_recovery_error(4, false);
        let tet10 = manufactured_quadratic_recovery_error(1, true);

        assert!((tet4_coarse / tet4_medium - 2.0).abs() < 1.0e-12);
        assert!((tet4_medium / tet4_fine - 2.0).abs() < 1.0e-12);
        assert!(tet10 < 1.0e-13 * tet4_coarse);
    }

    fn manufactured_quadratic_recovery_error(divisions: usize, quadratic: bool) -> f64 {
        let h = 1.0 / divisions as f64;
        let mut error_squared = 0.0;
        for i in 0..divisions {
            for j in 0..divisions {
                for k in 0..divisions {
                    let origin = [i as f64 * h, j as f64 * h, k as f64 * h];
                    let vertex = |dx: f64, dy: f64, dz: f64| {
                        [origin[0] + dx * h, origin[1] + dy * h, origin[2] + dz * h]
                    };
                    let v000 = vertex(0.0, 0.0, 0.0);
                    let v100 = vertex(1.0, 0.0, 0.0);
                    let v010 = vertex(0.0, 1.0, 0.0);
                    let v110 = vertex(1.0, 1.0, 0.0);
                    let v001 = vertex(0.0, 0.0, 1.0);
                    let v101 = vertex(1.0, 0.0, 1.0);
                    let v011 = vertex(0.0, 1.0, 1.0);
                    let v111 = vertex(1.0, 1.0, 1.0);
                    for mut corners in [
                        [v000, v100, v110, v111],
                        [v000, v110, v010, v111],
                        [v000, v010, v011, v111],
                        [v000, v011, v001, v111],
                        [v000, v001, v101, v111],
                        [v000, v101, v100, v111],
                    ] {
                        if (Tetrahedron4ElementGeometry { nodes_m: corners })
                            .volume_m3()
                            .is_err()
                        {
                            corners.swap(1, 2);
                        }
                        error_squared += if quadratic {
                            quadratic_element_recovery_error(corners)
                        } else {
                            linear_element_recovery_error(corners)
                        };
                    }
                }
            }
        }
        error_squared.sqrt()
    }

    fn linear_element_recovery_error(corners: [[f64; 3]; 4]) -> f64 {
        let geometry = Tetrahedron4ElementGeometry { nodes_m: corners };
        let b = super::super::tetrahedron4::strain_displacement_matrix(geometry).unwrap();
        let displacement = corners
            .into_iter()
            .flat_map(|node| [node[0].powi(2), 0.0, 0.0])
            .collect::<Vec<_>>();
        let recovered = b[0]
            .iter()
            .zip(displacement)
            .map(|(shape, value)| shape * value)
            .sum::<f64>();
        let volume = geometry.volume_m3().unwrap();
        QUADRATURE
            .iter()
            .map(|barycentric| {
                let x = (0..4)
                    .map(|node| barycentric[node] * corners[node][0])
                    .sum::<f64>();
                (recovered - 2.0 * x).powi(2) * volume / 4.0
            })
            .sum()
    }

    fn quadratic_element_recovery_error(corners: [[f64; 3]; 4]) -> f64 {
        let nodes_m = [
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
        ];
        let geometry = Tetrahedron10ElementGeometry { nodes_m };
        let displacement = std::array::from_fn(|dof| {
            let node = dof / 3;
            if dof % 3 == 0 {
                nodes_m[node][0].powi(2)
            } else {
                0.0
            }
        });
        geometry
            .quadrature_strain(displacement)
            .unwrap()
            .iter()
            .map(|point| {
                let x = (0..10)
                    .map(|node| shape_values(point.barycentric)[node] * nodes_m[node][0])
                    .sum::<f64>();
                (point.strain[0] - 2.0 * x).powi(2) * point.integration_weight_m3
            })
            .sum()
    }

    fn midpoint(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|axis| (left[axis] + right[axis]) * 0.5)
    }
}
