use runmat_simulation_usage::{OperationCount, UsageCategory, UsageContext};
use thiserror::Error;

use runmat_meshing_core::TETRAHEDRON_MIDSIDE_EDGE_CORNERS;

use crate::topology::quadrature::{
    TETRAHEDRON_ELEVEN_POINT_RULE as VOLUME_QUADRATURE, TETRAHEDRON_FOUR_POINT_RULE as QUADRATURE,
};

use super::material::SolidMaterialError;
#[cfg(test)]
use super::tetrahedron10_geometry::reference_shape_gradients;
pub(super) use super::tetrahedron10_geometry::{
    jacobian_determinant, physical_shape_gradients_and_measure, validate_nodes,
};
#[cfg(test)]
use super::SolidMaterial;

pub const TETRAHEDRON10_ELEMENT_NODE_COUNT: usize = 10;
pub const TETRAHEDRON10_NODE_DOF_COUNT: usize = 3;
pub const TETRAHEDRON10_ELEMENT_DOF_COUNT: usize =
    TETRAHEDRON10_ELEMENT_NODE_COUNT * TETRAHEDRON10_NODE_DOF_COUNT;

pub type Tetrahedron10Matrix30 =
    [[f64; TETRAHEDRON10_ELEMENT_DOF_COUNT]; TETRAHEDRON10_ELEMENT_DOF_COUNT];

fn immediate<R>(
    future: impl std::future::Future<
        Output = Result<R, Tetrahedron10CapacityError<std::convert::Infallible>>,
    >,
) -> Result<R, Tetrahedron10ElementError> {
    let mut future = std::pin::pin!(future);
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    match std::future::Future::poll(future.as_mut(), &mut context) {
        std::task::Poll::Ready(result) => result.map_err(|error| match error {
            Tetrahedron10CapacityError::Numerical(error) => error,
            Tetrahedron10CapacityError::Admission(never) => match never {},
        }),
        std::task::Poll::Pending => unreachable!("immediate Tet10 geometry cannot suspend"),
    }
}

const QUADRATURE_WEIGHT: f64 = 1.0 / 24.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tetrahedron10ElementGeometry {
    /// Corners 0..4 followed by edges 01, 12, 20, 03, 13, and 23.
    pub nodes_m: [[f64; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT],
}

impl Tetrahedron10ElementGeometry {
    pub fn volume_m3(self) -> Result<f64, Tetrahedron10ElementError> {
        immediate(self.volume_m3_with_optional_capacity::<std::convert::Infallible>(None))
    }

    pub async fn volume_m3_with_capacity<E>(
        self,
        owner: &dyn runmat_simulation_usage::AsyncCapacityOwner<E>,
    ) -> Result<f64, Tetrahedron10CapacityError<E>> {
        self.volume_m3_with_optional_capacity(Some(owner)).await
    }

    async fn volume_m3_with_optional_capacity<E>(
        self,
        owner: Option<&dyn runmat_simulation_usage::AsyncCapacityOwner<E>>,
    ) -> Result<f64, Tetrahedron10CapacityError<E>> {
        if let Some(owner) = owner {
            super::tetrahedron10_geometry::validate_nodes_with_capacity(self.nodes_m, owner)
                .await?;
            let jacobian = if super::tetrahedron10_geometry::is_affine(self) {
                23
            } else {
                185
            };
            owner
                .ensure_capacity(OperationCount::from_operations(22 + 11 * (92 + jacobian)))
                .await
                .map_err(Tetrahedron10CapacityError::Admission)?;
        } else {
            validate_nodes(self.nodes_m)?;
        }
        let usage = UsageContext::capture();
        let mut batch = usage
            .as_ref()
            .map(|usage| {
                usage
                    .ledger()
                    .reserve(UsageCategory::Assembly, OperationCount::from_operations(22))
            })
            .transpose()?;
        let volume =
            VOLUME_QUADRATURE
                .into_iter()
                .try_fold(0.0, |volume, (barycentric, weight)| {
                    let determinant = jacobian_determinant(self.nodes_m, barycentric)?;
                    let next = volume + weight * determinant;
                    if let Some(batch) = &mut batch {
                        batch.complete(OperationCount::from_operations(2))?;
                    }
                    Ok::<_, Tetrahedron10ElementError>(next)
                })?;
        if volume.is_finite() && volume > 0.0 {
            Ok(volume)
        } else {
            Err(Tetrahedron10ElementError::DegenerateOrInverted.into())
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
        immediate(
            self.quadrature_strain_with_optional_capacity::<std::convert::Infallible>(
                displacement,
                None,
            ),
        )
    }

    pub(crate) async fn quadrature_strain_with_capacity<E>(
        self,
        displacement: [f64; TETRAHEDRON10_ELEMENT_DOF_COUNT],
        owner: &dyn runmat_simulation_usage::AsyncCapacityOwner<E>,
    ) -> Result<[Tetrahedron10RecoveryPoint; 4], Tetrahedron10CapacityError<E>> {
        self.quadrature_strain_with_optional_capacity(displacement, Some(owner))
            .await
    }

    async fn quadrature_strain_with_optional_capacity<E>(
        self,
        displacement: [f64; TETRAHEDRON10_ELEMENT_DOF_COUNT],
        owner: Option<&dyn runmat_simulation_usage::AsyncCapacityOwner<E>>,
    ) -> Result<[Tetrahedron10RecoveryPoint; 4], Tetrahedron10CapacityError<E>> {
        if let Some(owner) = owner {
            super::tetrahedron10_geometry::validate_nodes_with_capacity(self.nodes_m, owner)
                .await?;
        } else {
            validate_nodes(self.nodes_m)?;
        }
        if displacement.iter().any(|value| !value.is_finite()) {
            return Err(Tetrahedron10ElementError::NonFiniteDisplacement.into());
        }
        let mut recovered = [Tetrahedron10RecoveryPoint {
            barycentric: [0.0; 4],
            integration_weight_m3: 0.0,
            strain: [0.0; 6],
        }; 4];
        for (index, point) in recovered.iter_mut().enumerate() {
            if let Some(owner) = owner {
                let jacobian = if super::tetrahedron10_geometry::is_affine(self) {
                    23
                } else {
                    185
                };
                owner
                    .ensure_capacity(OperationCount::from_operations(
                        92 + jacobian + 37 + 150 + 355,
                    ))
                    .await
                    .map_err(Tetrahedron10CapacityError::Admission)?;
            }
            let barycentric = QUADRATURE[index];
            let (gradients, determinant) =
                physical_shape_gradients_and_measure(self.nodes_m, barycentric)?;
            let b = strain_displacement_matrix(gradients);
            let usage = UsageContext::capture();
            let mut batch = usage
                .as_ref()
                .map(|usage| {
                    usage.ledger().reserve(
                        UsageCategory::Assembly,
                        OperationCount::from_operations(355),
                    )
                })
                .transpose()?;
            let strain: [f64; 6] = std::array::from_fn(|component| {
                b[component]
                    .iter()
                    .zip(displacement)
                    .map(|(shape, value)| shape * value)
                    .sum()
            });
            let integration_weight_m3 = QUADRATURE_WEIGHT * determinant;
            if let Some(batch) = &mut batch {
                batch.complete(OperationCount::from_operations(355))?;
            }
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
                return Err(Tetrahedron10ElementError::DegenerateOrInverted.into());
            }
        }
        Ok(recovered)
    }

    pub(crate) fn quadrature_integration_data(
        self,
    ) -> Result<[([f64; 4], f64); 4], Tetrahedron10ElementError> {
        validate_nodes(self.nodes_m)?;
        let usage = UsageContext::capture();
        let mut batch = usage
            .as_ref()
            .map(|usage| {
                usage
                    .ledger()
                    .reserve(UsageCategory::Assembly, OperationCount::from_operations(4))
            })
            .transpose()?;
        let mut data = [([0.0; 4], 0.0); 4];
        for (index, entry) in data.iter_mut().enumerate() {
            let barycentric = QUADRATURE[index];
            let weight = QUADRATURE_WEIGHT * jacobian_determinant(self.nodes_m, barycentric)?;
            if let Some(batch) = &mut batch {
                batch.complete(OperationCount::from_operations(1))?;
            }
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
    #[error("Tetrahedron10 curved material-stiffness quadrature did not converge")]
    UnconvergedStiffnessQuadrature,
    #[error("Tetrahedron10 material is invalid: {0}")]
    InvalidMaterial(#[from] SolidMaterialError),
    #[error("Tetrahedron10 numerical accounting failed: {0}")]
    Accounting(#[from] runmat_simulation_usage::LedgerError),
}

pub enum Tetrahedron10CapacityError<E> {
    Numerical(Tetrahedron10ElementError),
    Admission(E),
}

impl<E> From<Tetrahedron10ElementError> for Tetrahedron10CapacityError<E> {
    fn from(error: Tetrahedron10ElementError) -> Self {
        Self::Numerical(error)
    }
}

impl<E> From<runmat_simulation_usage::LedgerError> for Tetrahedron10CapacityError<E> {
    fn from(error: runmat_simulation_usage::LedgerError) -> Self {
        Self::Numerical(Tetrahedron10ElementError::Accounting(error))
    }
}

impl<E> From<SolidMaterialError> for Tetrahedron10CapacityError<E> {
    fn from(error: SolidMaterialError) -> Self {
        Self::Numerical(Tetrahedron10ElementError::InvalidMaterial(error))
    }
}

pub(super) fn shape_values(barycentric: [f64; 4]) -> [f64; TETRAHEDRON10_ELEMENT_NODE_COUNT] {
    runmat_meshing_core::interpolation::tetrahedron10_lagrange_weights(barycentric)
}

pub(super) fn strain_displacement_matrix(
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

#[cfg(test)]
mod tests {
    use super::super::tetrahedron10_mass::consistent_mass_matrix;
    use super::super::tetrahedron10_stiffness::global_stiffness_matrix;
    use super::*;
    use crate::assembly::elements::solid::Tetrahedron4ElementGeometry;
    use runmat_simulation_usage::UsageLedger;
    use std::sync::Arc;

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
            youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(200.0e9),
            poisson_ratio: runmat_types::physical::PoissonRatio::new(0.3),
            density_kg_per_m3: runmat_types::physical::MassDensity::new(7_850.0),
        }
    }

    #[test]
    fn admitted_affine_tetrahedron10_geometry_conserves_reference_work() {
        // A reference gradient plus affine determinant (92 + 23), followed
        // by a reference gradient, affine inverse (60), and 150 transforms.
        let limit = OperationCount::from_operations(92 + 23 + 92 + 60 + 150);
        let ledger = Arc::new(UsageLedger::new(limit, limit).unwrap());
        let context = UsageContext::new(Arc::clone(&ledger));
        context.run_sync(|| {
            jacobian_determinant(straight_unit_tetrahedron().nodes_m, [0.25; 4]).unwrap();
            physical_shape_gradients_and_measure(straight_unit_tetrahedron().nodes_m, [0.25; 4])
                .unwrap();
        });
        let snapshot = ledger.snapshot().unwrap();
        assert_eq!(snapshot.completed.assembly, limit);
        assert_eq!(snapshot.reserved, OperationCount::ZERO);
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
            let gradients = physical_shape_gradients_and_measure(geometry.nodes_m, barycentric)
                .unwrap()
                .0;
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
        let density = steel().density_kg_per_m3.canonical_value();
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
