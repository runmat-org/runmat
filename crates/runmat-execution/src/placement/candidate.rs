use runmat_types::{RegionGuardId, RegionId};
use serde::{Deserialize, Serialize};

use super::{ExecutionCostEstimate, ResourceByteBound};

/// One source owner's peak staging while a host route gathers its operands.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderReadbackDemand {
    pub device_id: u32,
    pub provider_scratch_bytes: ResourceByteBound,
    pub queue_slots: u32,
}

/// Placement snapshots admit at most sixteen provider namespaces. A fixed
/// inline set retains the candidate's Copy ABI and rejects excess owners,
/// never truncating an otherwise unbudgeted readback route.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderReadbackDemands {
    entries: [Option<ProviderReadbackDemand>; 16],
}

impl ProviderReadbackDemands {
    pub fn sequential(readbacks: impl IntoIterator<Item = ProviderReadbackDemand>) -> Option<Self> {
        let mut by_device = std::collections::BTreeMap::new();
        for readback in readbacks {
            if readback.queue_slots == 0 {
                return None;
            }
            let current: &mut ProviderReadbackDemand =
                by_device.entry(readback.device_id).or_insert(readback);
            current.queue_slots = current.queue_slots.max(readback.queue_slots);
            current.provider_scratch_bytes = match (
                current.provider_scratch_bytes,
                readback.provider_scratch_bytes,
            ) {
                (ResourceByteBound::Known(left), ResourceByteBound::Known(right)) => {
                    ResourceByteBound::Known(left.max(right))
                }
                (ResourceByteBound::Unavailable(reason), _)
                | (_, ResourceByteBound::Unavailable(reason)) => {
                    ResourceByteBound::Unavailable(reason)
                }
            };
        }
        if by_device.len() > 16 {
            return None;
        }
        let mut set = Self::default();
        for (slot, demand) in set.entries.iter_mut().zip(by_device.into_values()) {
            *slot = Some(demand);
        }
        Some(set)
    }

    pub fn iter(&self) -> impl Iterator<Item = ProviderReadbackDemand> + '_ {
        self.entries.iter().flatten().copied()
    }

    pub fn is_canonical(&self) -> bool {
        let mut previous = None;
        let mut empty = false;
        for entry in self.entries {
            match entry {
                Some(demand)
                    if empty
                        || demand.queue_slots == 0
                        || previous.is_some_and(|device| demand.device_id <= device) =>
                {
                    return false
                }
                Some(demand) => previous = Some(demand.device_id),
                None => empty = true,
            }
        }
        true
    }
}

#[cfg(test)]
mod readback_demand_tests {
    use super::*;

    #[test]
    fn sequential_readbacks_keep_each_owner_and_peak_same_owner_staging() {
        let demands = ProviderReadbackDemands::sequential([
            ProviderReadbackDemand {
                device_id: 8,
                provider_scratch_bytes: ResourceByteBound::Known(16),
                queue_slots: 1,
            },
            ProviderReadbackDemand {
                device_id: 7,
                provider_scratch_bytes: ResourceByteBound::Known(8),
                queue_slots: 1,
            },
            ProviderReadbackDemand {
                device_id: 8,
                provider_scratch_bytes: ResourceByteBound::Known(24),
                queue_slots: 1,
            },
        ])
        .unwrap();
        assert!(demands.is_canonical());
        assert_eq!(
            demands.iter().collect::<Vec<_>>(),
            vec![
                ProviderReadbackDemand {
                    device_id: 7,
                    provider_scratch_bytes: ResourceByteBound::Known(8),
                    queue_slots: 1,
                },
                ProviderReadbackDemand {
                    device_id: 8,
                    provider_scratch_bytes: ResourceByteBound::Known(24),
                    queue_slots: 1,
                },
            ]
        );
        assert!(
            ProviderReadbackDemands::sequential([ProviderReadbackDemand {
                device_id: 7,
                provider_scratch_bytes: ResourceByteBound::Known(0),
                queue_slots: 0,
            }])
            .is_none()
        );
    }
}

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

/// A complete host route may require a provider readback before computation
/// or an upload after computation. These are real transfer stages, not changes
/// to the computation location or guessed costs.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum CandidateOutputPresentation {
    Direct,
    /// A resident input is read back and the complete result is published on
    /// the host. Provider-owned mapped staging remains a provider resource.
    ProviderDownload {
        device_id: u32,
        provider_scratch_bytes: ResourceByteBound,
        queue_slots: u32,
    },
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
        /// Provider-owned scratch for this upload stage. Callers with resident
        /// gathered inputs must use `ProviderReadbackUpload` to budget each
        /// source owner's staging independently.
        provider_scratch_bytes: ResourceByteBound,
        queue_slots: u32,
        preparation: CandidatePreparationState,
    },
    /// One complete host gather/compute/upload route. Input providers may be
    /// different from the output owner and have independent staging quotas.
    /// Readback stages must complete sequentially before output publication;
    /// `ProviderReadbackDemands::sequential` records their per-owner peak.
    ProviderReadbackUpload {
        device_id: u32,
        transfer_bytes: Option<u64>,
        host_result_bytes: Option<ResourceByteBound>,
        host_scratch_bytes: ResourceByteBound,
        provider_scratch_bytes: ResourceByteBound,
        queue_slots: u32,
        preparation: CandidatePreparationState,
        readbacks: ProviderReadbackDemands,
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
            (
                Self::ProviderDownload { queue_slots, .. },
                CandidateExecutionLocation::Host,
                CandidateOutputResidency::Host,
            ) if queue_slots > 0 => Ok(()),
            (Self::ProviderDownload { .. }, _, _) => {
                Err("provider download presentation needs a host route, host output, and queue slot")
            }
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
            (
                Self::ProviderReadbackUpload {
                    device_id: upload,
                    queue_slots,
                    preparation,
                    readbacks,
                    ..
                },
                CandidateExecutionLocation::Host,
                CandidateOutputResidency::Provider { device_id: output }
                | CandidateOutputResidency::Mirrored { device_id: output },
            ) if upload == output && queue_slots > 0
                && !matches!(preparation, CandidatePreparationState::Preparing | CandidatePreparationState::Unknown)
                && readbacks.is_canonical() && readbacks.iter().next().is_some() => Ok(()),
            (Self::ProviderUpload { .. }, _, _) => {
                Err("provider upload presentation needs a ready stage, queue slot, and matching host route and output device")
            }
            (Self::ProviderReadbackUpload { .. }, _, _) => {
                Err("provider readback/upload needs canonical input stages and a ready matching host-to-provider output")
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
