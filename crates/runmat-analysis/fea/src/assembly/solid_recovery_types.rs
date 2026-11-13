//! Geometry and thermal integration-point data retained for physical solid recovery.

use super::elements::solid::SolidMaterial;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolidRecoveryElementSummary {
    pub element_id: String,
    pub region_id: String,
    pub material: SolidMaterial,
    pub thermal_eigenstrain: Option<f64>,
    #[serde(with = "runmat_types::physical::canonical_scalar::optional")]
    pub prescribed_temperature_k: Option<runmat_types::physical::Temperature>,
    pub solved_thermal_quadrature: Option<SolvedThermalRecoveryQuadrature>,
    pub geometry: SolidRecoveryGeometry,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolvedThermalRecoveryQuadrature {
    /// Complete local nodal solution, retained for higher-order energy integration.
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub nodal_temperature_k:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Temperature, Vec<f64>>,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub expansion_coefficient_per_k: runmat_types::physical::ThermalExpansion,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub reference_temperature_k: runmat_types::physical::Temperature,
    /// Interpolated retained nodal solution in the shared four-point recovery rule's order.
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub temperature_k:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Temperature, [f64; 4]>,
    /// Material-owned isotropic thermal strain α(T − Tref) at those same points.
    pub eigenstrain: [f64; 4],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "order", rename_all = "snake_case", deny_unknown_fields)]
pub enum SolidRecoveryGeometry {
    Tetrahedron4 {
        node_indices: [usize; 4],
        #[serde(with = "runmat_types::physical::canonical_storage")]
        coordinates_m:
            runmat_types::physical::Quantity<runmat_types::physical::kind::Position, [[f64; 3]; 4]>,
    },
    Tetrahedron10 {
        node_indices: [usize; 10],
        #[serde(with = "runmat_types::physical::canonical_storage")]
        coordinates_m: runmat_types::physical::Quantity<
            runmat_types::physical::kind::Position,
            [[f64; 3]; 10],
        >,
    },
}

impl SolidRecoveryGeometry {
    pub fn node_indices(&self) -> &[usize] {
        match self {
            Self::Tetrahedron4 { node_indices, .. } => node_indices,
            Self::Tetrahedron10 { node_indices, .. } => node_indices,
        }
    }

    /// Exact physical volume used by topology-changing recovery operations.
    pub fn volume_m3(&self) -> Result<f64, SolidRecoveryGeometryError> {
        use super::elements::solid::{Tetrahedron10ElementGeometry, Tetrahedron4ElementGeometry};

        match self {
            Self::Tetrahedron4 { coordinates_m, .. } => Tetrahedron4ElementGeometry {
                nodes_m: coordinates_m.canonical_value(),
            }
            .volume_m3()
            .map_err(SolidRecoveryGeometryError::Tetrahedron4),
            Self::Tetrahedron10 { coordinates_m, .. } => Tetrahedron10ElementGeometry {
                nodes_m: coordinates_m.canonical_value(),
            }
            .volume_m3()
            .map_err(SolidRecoveryGeometryError::Tetrahedron10),
        }
    }

    pub(crate) async fn volume_m3_with_capacity<A>(
        &self,
        owner: Option<&dyn runmat_simulation_usage::AsyncCapacityOwner<A>>,
    ) -> Result<
        f64,
        crate::algebra::coercivity::NumericalCapacityError<SolidRecoveryGeometryError, A>,
    > {
        use super::elements::solid::tetrahedron10::Tetrahedron10CapacityError;
        use crate::algebra::coercivity::NumericalCapacityError;
        let Some(owner) = owner else {
            return self.volume_m3().map_err(NumericalCapacityError::Numerical);
        };
        match self {
            Self::Tetrahedron4 { .. } => {
                owner
                    .ensure_capacity(self.volume_reference_work())
                    .await
                    .map_err(NumericalCapacityError::Admission)?;
                self.volume_m3().map_err(NumericalCapacityError::Numerical)
            }
            Self::Tetrahedron10 { coordinates_m, .. } => {
                super::elements::solid::Tetrahedron10ElementGeometry {
                    nodes_m: coordinates_m.canonical_value(),
                }
                .volume_m3_with_capacity(owner)
                .await
                .map_err(|error| match error {
                    Tetrahedron10CapacityError::Numerical(error) => {
                        NumericalCapacityError::Numerical(
                            SolidRecoveryGeometryError::Tetrahedron10(error),
                        )
                    }
                    Tetrahedron10CapacityError::Admission(error) => {
                        NumericalCapacityError::Admission(error)
                    }
                })
            }
        }
    }

    /// Work reserved by the fixed physical-volume primitive for this geometry.
    pub(crate) fn volume_reference_work(&self) -> runmat_simulation_usage::OperationCount {
        use super::elements::solid::{tetrahedron10_is_affine, Tetrahedron10ElementGeometry};
        let operations = match self {
            Self::Tetrahedron4 { .. } => 24,
            Self::Tetrahedron10 { coordinates_m, .. } => {
                let determinant = if tetrahedron10_is_affine(Tetrahedron10ElementGeometry {
                    nodes_m: coordinates_m.canonical_value(),
                }) {
                    23
                } else {
                    185
                };
                // Eleven reference gradients and determinants, plus volume accumulation.
                11 * (92 + determinant) + 22
            }
        };
        runmat_simulation_usage::OperationCount::from_operations(operations)
    }
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SolidRecoveryGeometryError {
    #[error(transparent)]
    Tetrahedron4(#[from] super::elements::solid::Tetrahedron4ElementError),
    #[error(transparent)]
    Tetrahedron10(#[from] super::elements::solid::Tetrahedron10ElementError),
}
