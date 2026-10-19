//! Geometry and thermal integration-point data retained for physical solid recovery.

use super::elements::solid::SolidMaterial;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolidRecoveryElementSummary {
    pub element_id: String,
    pub region_id: String,
    pub material: SolidMaterial,
    pub thermal_eigenstrain: Option<f64>,
    pub prescribed_temperature_k: Option<f64>,
    pub solved_thermal_quadrature: Option<SolvedThermalRecoveryQuadrature>,
    pub geometry: SolidRecoveryGeometry,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolvedThermalRecoveryQuadrature {
    /// Complete local nodal solution, retained for higher-order energy integration.
    pub nodal_temperature_k: Vec<f64>,
    pub expansion_coefficient_per_k: f64,
    pub reference_temperature_k: f64,
    /// Interpolated retained nodal solution in the shared four-point recovery rule's order.
    pub temperature_k: [f64; 4],
    /// Material-owned isotropic thermal strain α(T − Tref) at those same points.
    pub eigenstrain: [f64; 4],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "order", rename_all = "snake_case", deny_unknown_fields)]
pub enum SolidRecoveryGeometry {
    Tetrahedron4 {
        node_indices: [usize; 4],
        coordinates_m: [[f64; 3]; 4],
    },
    Tetrahedron10 {
        node_indices: [usize; 10],
        coordinates_m: [[f64; 3]; 10],
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
    pub fn volume_m3(&self) -> Result<f64, String> {
        use super::elements::solid::{Tetrahedron10ElementGeometry, Tetrahedron4ElementGeometry};

        match self {
            Self::Tetrahedron4 { coordinates_m, .. } => Tetrahedron4ElementGeometry {
                nodes_m: *coordinates_m,
            }
            .volume_m3()
            .map_err(|source| source.to_string()),
            Self::Tetrahedron10 { coordinates_m, .. } => Tetrahedron10ElementGeometry {
                nodes_m: *coordinates_m,
            }
            .volume_m3()
            .map_err(|source| source.to_string()),
        }
    }
}
