use runmat_types::{RegionGuardId, RegionId};
use serde::{Deserialize, Serialize};

use super::ExecutionCostEstimate;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionCandidateKind {
    SharedRuntime,
    GenericNativeCpu,
    SpecializedNativeCpu,
    VectorizedNativeCpu,
    CpuLibrary,
    ProviderOperation,
    ProviderLibrary,
    ProviderGraph,
    ProviderFusion,
}

impl ExecutionCandidateKind {
    pub const fn is_provider(self) -> bool {
        matches!(
            self,
            Self::ProviderOperation
                | Self::ProviderLibrary
                | Self::ProviderGraph
                | Self::ProviderFusion
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum CandidateExecutionLocation {
    Host,
    Provider { device_id: u32 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidatePreparationState {
    Ready,
    Warm,
    Cold,
    Preparing,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum CandidateOutputResidency {
    Host,
    Provider { device_id: u32 },
    Mirrored { device_id: u32 },
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionCandidateDescriptor {
    pub identity: String,
    pub region: Option<RegionId>,
    pub kind: ExecutionCandidateKind,
    pub execution_location: CandidateExecutionLocation,
    pub preparation: CandidatePreparationState,
    pub cost: ExecutionCostEstimate,
    pub output_residency: CandidateOutputResidency,
    pub guards: Vec<RegionGuardId>,
}

impl ExecutionCandidateDescriptor {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.identity.is_empty()
            || self.identity.len() > 128
            || self.identity.chars().any(char::is_control)
        {
            return Err("candidate identity must be 1..=128 bytes without control characters");
        }
        if self.cost.checked_total_ns().is_none() {
            return Err("candidate cost components overflow u64");
        }
        if self.kind.is_provider()
            != matches!(
                self.execution_location,
                CandidateExecutionLocation::Provider { .. }
            )
        {
            return Err("candidate kind and execution location are inconsistent");
        }
        if self.guards.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err("candidate guards must be sorted and unique");
        }
        if self
            .region
            .is_some_and(|region| self.guards.iter().any(|guard| guard.region != region))
        {
            return Err("candidate guards must belong to the candidate region");
        }
        Ok(())
    }
}
