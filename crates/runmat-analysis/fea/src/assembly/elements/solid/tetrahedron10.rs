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
mod tests;
