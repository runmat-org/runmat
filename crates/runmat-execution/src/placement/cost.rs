use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EstimateConfidence {
    Prior,
    Low,
    Medium,
    High,
    Exact,
}

impl EstimateConfidence {
    pub const fn uncertainty_basis_points(self) -> u32 {
        match self {
            Self::Prior => 2_500,
            Self::Low => 1_500,
            Self::Medium => 500,
            Self::High => 100,
            Self::Exact => 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EstimateSource {
    StaticPrior,
    Calibration,
    Observation,
    Compiler,
    Provider,
    Synthetic,
}

/// Complete additive cost decomposition for one legal execution candidate.
///
/// Components are deliberately executor-neutral. Providers retain ownership of
/// kernel scheduling, while placement retains ownership of comparing complete
/// candidates and residency transitions.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionCostComponents {
    pub compile_or_prepare_ns: u64,
    pub upload_ns: u64,
    pub allocation_ns: u64,
    pub queue_ns: u64,
    pub execution_ns: u64,
    pub synchronization_ns: u64,
    pub download_ns: u64,
    pub downstream_ns: u64,
}

impl ExecutionCostComponents {
    pub fn checked_total_ns(self) -> Option<u64> {
        [
            self.compile_or_prepare_ns,
            self.upload_ns,
            self.allocation_ns,
            self.queue_ns,
            self.execution_ns,
            self.synchronization_ns,
            self.download_ns,
            self.downstream_ns,
        ]
        .into_iter()
        .try_fold(0_u64, u64::checked_add)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionCostEstimate {
    pub components: ExecutionCostComponents,
    pub scratch_bytes: u64,
    pub confidence: EstimateConfidence,
    pub source: EstimateSource,
}

impl ExecutionCostEstimate {
    pub fn checked_total_ns(self) -> Option<u64> {
        self.components.checked_total_ns()
    }

    pub fn checked_risk_adjusted_ns(self) -> Option<u64> {
        let total = self.checked_total_ns()?;
        let uncertainty = total
            .checked_mul(u64::from(self.confidence.uncertainty_basis_points()))?
            .checked_add(9_999)?
            / 10_000;
        total.checked_add(uncertainty)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_totals_and_uncertainty_are_checked() {
        let estimate = ExecutionCostEstimate {
            components: ExecutionCostComponents {
                execution_ns: 100,
                upload_ns: 20,
                ..ExecutionCostComponents::default()
            },
            scratch_bytes: 0,
            confidence: EstimateConfidence::Medium,
            source: EstimateSource::Synthetic,
        };
        assert_eq!(estimate.checked_total_ns(), Some(120));
        assert_eq!(estimate.checked_risk_adjusted_ns(), Some(126));

        let overflow = ExecutionCostComponents {
            execution_ns: u64::MAX,
            upload_ns: 1,
            ..ExecutionCostComponents::default()
        };
        assert_eq!(overflow.checked_total_ns(), None);
    }
}
