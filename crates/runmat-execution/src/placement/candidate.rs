use runmat_types::{RegionGuardId, RegionId};
use serde::{Deserialize, Serialize};

use super::{ExecutionCostEstimate, ResourceByteBound};

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
    /// The preparation owner cannot certify a cache phase for this route.
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum CandidateOutputResidency {
    Host,
    Provider { device_id: u32 },
    Mirrored { device_id: u32 },
    Unknown,
}

/// A complete route may compute on the host and then publish through an
/// admitted provider upload. This is a real presentation stage, not a change
/// to the computation location or a guessed transfer cost.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum CandidateOutputPresentation {
    Direct,
    ProviderUpload {
        device_id: u32,
        /// Bytes moved by the presentation stage, if physically established.
        /// Unknown is not a zero-cost observation or a memory bound.
        transfer_bytes: Option<u64>,
        /// Host-produced payload live during upload. When absent, legacy
        /// callers explicitly rely on retained bytes being the same bound.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        host_result_bytes: Option<ResourceByteBound>,
        /// Temporary host storage beyond the host-produced output itself.
        host_scratch_bytes: ResourceByteBound,
        /// Provider-owned scratch across the complete route. A host route that
        /// first reads back must include mapped or staged download buffers as
        /// well as upload scratch. Unknown is not zero even when the provider
        /// can separately bound its host workspace.
        provider_scratch_bytes: ResourceByteBound,
        queue_slots: u32,
        preparation: CandidatePreparationState,
    },
}

impl Default for CandidateOutputPresentation {
    fn default() -> Self {
        Self::Direct
    }
}

impl CandidateOutputPresentation {
    pub fn validate_for(
        self,
        execution: CandidateExecutionLocation,
        output: CandidateOutputResidency,
    ) -> Result<(), &'static str> {
        match (self, execution, output) {
            (Self::Direct, CandidateExecutionLocation::Host, CandidateOutputResidency::Provider { .. }
                | CandidateOutputResidency::Mirrored { .. }) => {
                Err("host execution needs an admitted provider presentation stage")
            }
            (
                Self::ProviderUpload {
                    device_id: upload,
                    queue_slots,
                    preparation,
                    ..
                },
                CandidateExecutionLocation::Host,
                CandidateOutputResidency::Provider { device_id: output }
                | CandidateOutputResidency::Mirrored { device_id: output },
            ) if upload == output && queue_slots > 0
                && !matches!(
                    preparation,
                    CandidatePreparationState::Preparing | CandidatePreparationState::Unknown
                ) => Ok(()),
            (Self::ProviderUpload { .. }, _, _) => {
                Err("provider upload presentation needs a ready stage, queue slot, and matching host route and output device")
            }
            (Self::Direct, _, _) => Ok(()),
        }
    }
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
    #[serde(default)]
    pub presentation: CandidateOutputPresentation,
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
        self.presentation
            .validate_for(self.execution_location, self.output_residency)?;
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
