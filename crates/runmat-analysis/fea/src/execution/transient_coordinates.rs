use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::pipeline::modal_basis::ModalBasisIdentity;

use super::{AnalysisExecutionError, AnalysisExecutionResult};

/// Recomputed evidence binding a reduced coordinate space to one physical operator and basis.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModalProjectionEvidence {
    pub basis: ModalBasisIdentity,
    pub operator_digest: Digest,
    pub full_dof_count: usize,
    pub mode_count: usize,
    pub maximum_mass_off_diagonal_ratio: f64,
    pub maximum_stiffness_off_diagonal_ratio: f64,
    pub maximum_damping_asymmetry_ratio: f64,
}

impl ModalProjectionEvidence {
    pub fn validate_for(
        &self,
        operator_digest: Digest,
        full_dof_count: usize,
    ) -> AnalysisExecutionResult<()> {
        if self.operator_digest != operator_digest
            || self.full_dof_count != full_dof_count
            || self.mode_count == 0
            || self.mode_count > full_dof_count
        {
            return Err(AnalysisExecutionError::Invalid(
                "modal projection evidence has inconsistent operator dimensions".into(),
            ));
        }
        if [
            self.maximum_mass_off_diagonal_ratio,
            self.maximum_stiffness_off_diagonal_ratio,
            self.maximum_damping_asymmetry_ratio,
        ]
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(AnalysisExecutionError::Invalid(
                "modal projection evidence contains an invalid normalized measure".into(),
            ));
        }
        Ok(())
    }
}

/// Durable generalized displacement, velocity, and acceleration at one accepted time boundary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModalTransientCoordinateState {
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub displacement:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Displacement, Vec<f64>>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub velocity:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Velocity, Vec<f64>>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub acceleration:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Acceleration, Vec<f64>>,
}
