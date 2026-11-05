use runmat_meshing_core::StableDigest;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralRecoveryEstimatorOptions {
    pub marking_fraction: f64,
    pub maximum_marked_elements: u64,
    pub cancellation_check_interval: u64,
}

impl Default for StructuralRecoveryEstimatorOptions {
    fn default() -> Self {
        Self {
            marking_fraction: 0.5,
            maximum_marked_elements: 1_000_000_000,
            cancellation_check_interval: 1_024,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralRecoveryIndicator {
    pub element_stable_identity: StableDigest,
    pub error: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralRecoveryStatistics {
    pub element_count: u64,
    pub minimum_error: f64,
    pub maximum_error: f64,
    pub mean_error: f64,
    pub root_mean_square_error: f64,
}

/// Rotation-invariant volume L2 norm of a physical symmetric stress tensor.
/// Indicators and total_error have units Pa * m^(3/2), not energy or relative error.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuralRecoveryNorm {
    StressTensorFrobeniusL2,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralRecoveryEstimate {
    pub norm: StructuralRecoveryNorm,
    pub solver_artifact_digest: StableDigest,
    pub stress_topology_id: String,
    pub total_error: f64,
    /// Original physical stress Frobenius volume-L2 norm, in the same units.
    pub reference_norm: f64,
    pub indicators: Vec<StructuralRecoveryIndicator>,
    pub marked_element_identities: Vec<StableDigest>,
    pub statistics: StructuralRecoveryStatistics,
}

impl StructuralRecoveryEstimate {
    /// Dimensionless discrepancy relative to the original solved stress field.
    pub fn relative_error(&self) -> Option<f64> {
        if self.reference_norm == 0.0 {
            (self.total_error == 0.0).then_some(0.0)
        } else {
            Some(self.total_error / self.reference_norm)
        }
    }

    pub fn validate(&self) -> Result<(), StructuralRecoveryEstimatorError> {
        if self.solver_artifact_digest == StableDigest::ZERO
            || self.stress_topology_id.is_empty()
            || self.stress_topology_id.len() > 256
            || !self.stress_topology_id.is_ascii()
            || self.stress_topology_id.chars().any(char::is_control)
            || self.indicators.is_empty()
            || self.statistics.element_count != self.indicators.len() as u64
            || !self.total_error.is_finite()
            || self.total_error < 0.0
            || !self.reference_norm.is_finite()
            || self.reference_norm < 0.0
            || (self.reference_norm == 0.0 && self.total_error != 0.0)
            || self
                .relative_error()
                .is_some_and(|value| !value.is_finite())
            || [
                self.statistics.minimum_error,
                self.statistics.maximum_error,
                self.statistics.mean_error,
                self.statistics.root_mean_square_error,
            ]
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(StructuralRecoveryEstimatorError::InvalidEstimate);
        }
        let mut previous = None;
        let mut sum = 0.0;
        let mut sum_squared = 0.0;
        for indicator in &self.indicators {
            if indicator.element_stable_identity == StableDigest::ZERO
                || !indicator.error.is_finite()
                || indicator.error < 0.0
                || previous.is_some_and(|identity| identity >= indicator.element_stable_identity)
            {
                return Err(StructuralRecoveryEstimatorError::InvalidEstimate);
            }
            sum += indicator.error;
            sum_squared += indicator.error * indicator.error;
            previous = Some(indicator.element_stable_identity);
        }
        let count = self.indicators.len() as f64;
        let expected = StructuralRecoveryStatistics {
            element_count: self.indicators.len() as u64,
            minimum_error: self
                .indicators
                .iter()
                .map(|indicator| indicator.error)
                .reduce(f64::min)
                .unwrap_or(0.0),
            maximum_error: self
                .indicators
                .iter()
                .map(|indicator| indicator.error)
                .reduce(f64::max)
                .unwrap_or(0.0),
            mean_error: sum / count,
            root_mean_square_error: (sum_squared / count).sqrt(),
        };
        let admitted = self
            .indicators
            .iter()
            .filter(|indicator| indicator.error > 0.0)
            .map(|indicator| indicator.element_stable_identity)
            .collect::<std::collections::BTreeSet<_>>();
        if self.total_error != sum_squared.sqrt()
            || self.statistics != expected
            || (self.total_error == 0.0) != self.marked_element_identities.is_empty()
            || !self
                .marked_element_identities
                .windows(2)
                .all(|pair| pair[0] < pair[1])
            || self
                .marked_element_identities
                .iter()
                .any(|identity| !admitted.contains(identity))
        {
            return Err(StructuralRecoveryEstimatorError::InvalidEstimate);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StructuralRecoveryEstimatorError {
    Accounting(runmat_simulation_usage::LedgerError),
    InvalidOptions,
    InvalidArtifact(String),
    MissingElementTopology,
    InvalidStressField,
    DeviceFieldRequiresHostTransfer,
    InvalidElementGeometry,
    InvalidEstimate,
    ResourceLimit,
    Cancelled,
}
