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
/// inline set rejects excess owners,
/// never truncating an otherwise unbudgeted readback route.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderReadbackDemands {
    entries: [Option<ProviderReadbackDemand>; 16],
}

impl ProviderReadbackDemands {
    pub fn sequential(readbacks: impl IntoIterator<Item = ProviderReadbackDemand>) -> Option<Self> {
        let mut set = Self::default();
        let mut count = 0usize;
        for readback in readbacks {
            if readback.queue_slots == 0
                && readback.provider_scratch_bytes != ResourceByteBound::Known(0)
            {
                return None;
            }
            let slot = if let Some(index) = set.entries[..count]
                .iter()
                .position(|entry| entry.is_some_and(|entry| entry.device_id == readback.device_id))
            {
                index
            } else {
                if count == set.entries.len() {
                    return None;
                }
                let index = count;
                count += 1;
                set.entries[index] = Some(readback);
                index
            };
            let current = set.entries[slot]
                .as_mut()
                .expect("initialized inline readback");
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
        set.entries[..count]
            .sort_unstable_by_key(|entry| entry.expect("initialized inline readback").device_id);
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
                        || (demand.queue_slots == 0
                            && demand.provider_scratch_bytes != ResourceByteBound::Known(0))
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

/// A provider-executed route's sequential input-preparation stage. The peak
/// includes every operation-created allocation still live at this stage (for
/// example an adapted kernel uploaded before a second readback), plus mapped
/// staging. It excludes authenticated borrowed allocations already counted in
/// that provider's resource snapshot. No two stages in this set overlap.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderInputStageDemand {
    pub device_id: u32,
    pub incremental_peak_bytes: ResourceByteBound,
    pub queue_slots: u32,
}

/// Canonical per-owner peak of sequential provider input stages.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderInputStageDemands {
    entries: [Option<ProviderInputStageDemand>; 16],
}

impl ProviderInputStageDemands {
    pub fn sequential(stages: impl IntoIterator<Item = ProviderInputStageDemand>) -> Option<Self> {
        // The canonical wire carrier already bounds the provider set. Build
        // its sorted prefix directly, without an unadmitted temporary map.
        let mut set = Self::default();
        for stage in stages {
            if stage.queue_slots == 0 && stage.incremental_peak_bytes != ResourceByteBound::Known(0)
            {
                return None;
            }
            let index = set.entries.iter().position(|entry| {
                entry.is_none_or(|current| current.device_id >= stage.device_id)
            })?;
            if let Some(current) = set.entries[index]
                .as_mut()
                .filter(|current| current.device_id == stage.device_id)
            {
                current.queue_slots = current.queue_slots.max(stage.queue_slots);
                current.incremental_peak_bytes =
                    match (current.incremental_peak_bytes, stage.incremental_peak_bytes) {
                        (ResourceByteBound::Known(left), ResourceByteBound::Known(right)) => {
                            ResourceByteBound::Known(left.max(right))
                        }
                        (ResourceByteBound::Unavailable(reason), _)
                        | (_, ResourceByteBound::Unavailable(reason)) => {
                            ResourceByteBound::Unavailable(reason)
                        }
                    };
            } else {
                let last = set.entries.len() - 1;
                if set.entries[last].is_some() {
                    return None;
                }
                set.entries.copy_within(index..last, index + 1);
                set.entries[index] = Some(stage);
            }
        }
        Some(set)
    }

    pub fn iter(&self) -> impl Iterator<Item = ProviderInputStageDemand> + '_ {
        self.entries.iter().flatten().copied()
    }

    pub fn is_canonical(&self) -> bool {
        let mut previous = None;
        let mut empty = false;
        for entry in self.entries {
            match entry {
                Some(demand)
                    if empty
                        || (demand.queue_slots == 0
                            && demand.incremental_peak_bytes != ResourceByteBound::Known(0))
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
mod provider_input_stage_tests {
    use super::*;

    #[test]
    fn sequential_stages_retain_prior_live_peak_per_owner() {
        let stages = ProviderInputStageDemands::sequential([
            ProviderInputStageDemand {
                device_id: 8,
                incremental_peak_bytes: ResourceByteBound::Known(12),
                queue_slots: 1,
            },
            ProviderInputStageDemand {
                device_id: 7,
                incremental_peak_bytes: ResourceByteBound::Known(8),
                queue_slots: 0,
            },
            ProviderInputStageDemand {
                device_id: 8,
                incremental_peak_bytes: ResourceByteBound::Known(20),
                queue_slots: 2,
            },
        ]);
        assert!(
            stages.is_none(),
            "nonempty zero-queue stage is not admitted"
        );

        let stages = ProviderInputStageDemands::sequential([
            ProviderInputStageDemand {
                device_id: 8,
                incremental_peak_bytes: ResourceByteBound::Known(12),
                queue_slots: 1,
            },
            ProviderInputStageDemand {
                device_id: 7,
                incremental_peak_bytes: ResourceByteBound::Known(0),
                queue_slots: 0,
            },
            ProviderInputStageDemand {
                device_id: 8,
                incremental_peak_bytes: ResourceByteBound::Known(20),
                queue_slots: 2,
            },
        ])
        .unwrap();
        assert!(stages.is_canonical());
        assert_eq!(
            stages.iter().collect::<Vec<_>>(),
            vec![
                ProviderInputStageDemand {
                    device_id: 7,
                    incremental_peak_bytes: ResourceByteBound::Known(0),
                    queue_slots: 0,
                },
                ProviderInputStageDemand {
                    device_id: 8,
                    incremental_peak_bytes: ResourceByteBound::Known(20),
                    queue_slots: 2,
                },
            ]
        );
        let stage = |device_id| ProviderInputStageDemand {
            device_id,
            incremental_peak_bytes: ResourceByteBound::Known(u64::from(device_id)),
            queue_slots: 1,
        };
        let full =
            ProviderInputStageDemands::sequential((0..16).rev().map(stage).chain([stage(8)]))
                .expect("full bounded set still accepts a repeated owner");
        assert!(full.is_canonical());
        assert_eq!(
            full.iter().map(|entry| entry.device_id).collect::<Vec<_>>(),
            (0..16).collect::<Vec<_>>()
        );
        assert!(ProviderInputStageDemands::sequential((0..17).map(stage)).is_none());
        assert!(ProviderInputStageDemands::sequential((0..17).rev().map(stage)).is_none());
    }
}

#[cfg(test)]
mod readback_demand_tests {
    use super::*;

    #[test]
    fn presentation_keeps_inline_routes_compact_and_readback_wire_shape_unchanged() {
        assert!(std::mem::size_of::<CandidateOutputPresentation>() <= 128);
        let demands = ProviderReadbackDemands::sequential([ProviderReadbackDemand {
            device_id: 7,
            provider_scratch_bytes: ResourceByteBound::Known(8),
            queue_slots: 1,
        }])
        .unwrap();
        let presentation = CandidateOutputPresentation::ProviderReadbackUpload {
            device_id: 7,
            transfer_bytes: Some(16),
            host_result_bytes: Some(ResourceByteBound::Known(16)),
            host_scratch_bytes: ResourceByteBound::Known(0),
            provider_scratch_bytes: ResourceByteBound::Known(8),
            queue_slots: 1,
            preparation: CandidatePreparationState::Ready,
            readbacks: Box::new(demands),
        };
        let encoded = serde_json::to_value(&presentation).unwrap();
        assert_eq!(encoded["readbacks"], serde_json::to_value(demands).unwrap());
        assert_eq!(
            serde_json::from_value::<CandidateOutputPresentation>(encoded).unwrap(),
            presentation,
        );
        let host = CandidateOutputPresentation::ProviderReadbacksHost {
            readbacks: Box::new(demands),
        };
        let encoded = serde_json::to_value(&host).unwrap();
        assert_eq!(encoded["readbacks"], serde_json::to_value(demands).unwrap());
        assert_eq!(
            serde_json::from_value::<CandidateOutputPresentation>(encoded).unwrap(),
            host,
        );
        let owner = |device_id| ProviderReadbackDemand {
            device_id,
            provider_scratch_bytes: ResourceByteBound::Known(8),
            queue_slots: 1,
        };
        assert_eq!(
            ProviderReadbackDemands::sequential((0..16).map(owner))
                .unwrap()
                .iter()
                .count(),
            16,
        );
        assert!(ProviderReadbackDemands::sequential((0..17).map(owner)).is_none());
    }

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
                provider_scratch_bytes: ResourceByteBound::Known(8),
                queue_slots: 0,
            }])
            .is_none()
        );
        assert!(
            ProviderReadbackDemands::sequential([ProviderReadbackDemand {
                device_id: 7,
                provider_scratch_bytes: ResourceByteBound::Known(0),
                queue_slots: 0,
            }])
            .is_some()
        );
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionCandidateKind {
    SharedRuntime,
    /// The original, source-ordered VM instructions of one installed region.
    /// Unlike a host route, these instructions may invoke a provider.
    OriginalVmRegion,
    GenericNativeCpu,
    SpecializedNativeCpu,
    VectorizedNativeCpu,
    CpuLibrary,
    ProviderOperation,
    ProviderLibrary,
    ProviderGraph,
    ProviderFusion,
    /// Original source-ordered native MIR sites. These may invoke their exact
    /// provider; this is not a host CPU implementation or VM installation.
    OriginalNativeRegion,
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

    pub const fn matches_location(self, location: CandidateExecutionLocation) -> bool {
        match (self, location) {
            (Self::OriginalVmRegion, CandidateExecutionLocation::VmRegion { .. }) => true,
            (Self::OriginalNativeRegion, CandidateExecutionLocation::NativeRegion { .. }) => true,
            (Self::OriginalNativeRegion, CandidateExecutionLocation::NativeSource) => true,
            (_, CandidateExecutionLocation::NativeSource) => false,
            (Self::OriginalNativeRegion, _)
            | (_, CandidateExecutionLocation::NativeRegion { .. }) => false,
            (Self::OriginalVmRegion, _) | (_, CandidateExecutionLocation::VmRegion { .. }) => false,
            (kind, CandidateExecutionLocation::Provider { .. }) => kind.is_provider(),
            (kind, CandidateExecutionLocation::Host) => !kind.is_provider(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum CandidateExecutionLocation {
    Host,
    Provider {
        device_id: u32,
    },
    /// Execution remains at the original VM instruction boundary. The VM may
    /// run host operations and operations on this exact resident owner.
    VmRegion {
        device_id: u32,
    },
    /// Original native MIR operations retain per-operation execution owners.
    NativeRegion {
        device_id: u32,
    },
    /// Native source operations with per-operation owners but no proved
    /// single device/physical tail. Neither Host nor device completion.
    NativeSource,
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
    Provider {
        device_id: u32,
    },
    Mirrored {
        device_id: u32,
    },
    /// Distinct public outputs from one provider computation: at least one
    /// retained resident result and at least one result read back to host.
    ProviderAndHost {
        device_id: u32,
        host_bytes: ResourceByteBound,
    },
    /// One provider-first route may publish either a provider allocation or a
    /// host value after a genuine Unsupported hook result. `host_bytes` is the
    /// separately proven retained host payload; provider retained bytes remain
    /// in CandidateResourceDemand. The branches are alternatives, not mirrors.
    ConditionalHostOrProvider {
        device_id: u32,
        host_bytes: ResourceByteBound,
    },
    Unknown,
}

/// A complete host route may require a provider readback before computation
/// or an upload after computation. These are real transfer stages, not changes
/// to the computation location or guessed costs.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
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
    /// A provider hook first produces its physical outputs; one authenticated
    /// output is then read back for host publication. The readback is a real
    /// post-compute stage, not an original-input preparation or a free mirror.
    ProviderProducedReadback {
        device_id: u32,
        transfer_bytes: u64,
        provider_scratch_bytes: ResourceByteBound,
        queue_slots: u32,
    },
    /// Sequential gathers from one or more independently owned devices,
    /// followed by host computation and host publication. This does not
    /// imply a provider upload or a provider-executed kernel.
    ProviderReadbacksHost {
        readbacks: Box<ProviderReadbackDemands>,
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
        /// Physically required completion slots. Zero denotes a synchronous
        /// upload, not an unknown queue demand; scratch is admitted separately.
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
        /// Upload completion slots; synchronous uploads require none. Input
        /// readback stages retain their independently admitted queue demands.
        queue_slots: u32,
        preparation: CandidatePreparationState,
        readbacks: Box<ProviderReadbackDemands>,
    },
    /// A selected provider-first hook may, only on its typed Unsupported
    /// outcome, compute a host result and upload it to the same original
    /// output owner. This is a conditional stage, not a host candidate.
    ConditionalProviderUpload {
        device_id: u32,
        transfer_bytes: Option<u64>,
        host_result_bytes: Option<ResourceByteBound>,
        /// Conditional host workspace excluding its result payload.
        host_scratch_bytes: ResourceByteBound,
        provider_scratch_bytes: ResourceByteBound,
        queue_slots: u32,
    },
}

impl Default for CandidateOutputPresentation {
    fn default() -> Self {
        Self::Direct
    }
}

impl CandidateOutputPresentation {
    pub fn validate_for(
        &self,
        execution: CandidateExecutionLocation,
        output: CandidateOutputResidency,
    ) -> Result<(), &'static str> {
        match (self, execution, output) {
            (
                Self::ProviderProducedReadback { device_id: stage, queue_slots, .. },
                CandidateExecutionLocation::Provider { device_id: execution },
                CandidateOutputResidency::ProviderAndHost { device_id: output, .. },
            ) if stage == &execution && stage == &output && *queue_slots > 0 => Ok(()),
            (Self::ProviderProducedReadback { .. }, _, _) => {
                Err("provider-produced readback needs one provider owner, mixed public outputs, and a completion queue slot")
            }
            (
                Self::ProviderDownload { queue_slots, provider_scratch_bytes, .. },
                CandidateExecutionLocation::Host,
                CandidateOutputResidency::Host,
            ) if *queue_slots > 0 || *provider_scratch_bytes == ResourceByteBound::Known(0) => Ok(()),
            (Self::ProviderDownload { .. }, _, _) => {
                Err("provider download presentation needs a host route, host output, and a queue slot unless zero staging is proven")
            }
            (
                Self::ProviderReadbacksHost { readbacks },
                CandidateExecutionLocation::Host,
                CandidateOutputResidency::Host,
            ) if readbacks.is_canonical() && readbacks.iter().next().is_some() => Ok(()),
            (Self::ProviderReadbacksHost { .. }, _, _) => {
                Err("provider readbacks need canonical input stages and a host result")
            }
            (Self::Direct, CandidateExecutionLocation::Host, CandidateOutputResidency::Provider { .. }
                | CandidateOutputResidency::Mirrored { .. }
                | CandidateOutputResidency::ProviderAndHost { .. }
                | CandidateOutputResidency::ConditionalHostOrProvider { .. }) => {
                Err("host execution needs an admitted provider presentation stage")
            }
            (
                Self::ProviderUpload {
                    device_id: upload,
                    preparation,
                    ..
                },
                CandidateExecutionLocation::Host,
                CandidateOutputResidency::Provider { device_id: output }
                | CandidateOutputResidency::Mirrored { device_id: output },
            ) if *upload == output
                && !matches!(
                    preparation,
                    CandidatePreparationState::Preparing | CandidatePreparationState::Unknown
                ) => Ok(()),
            (
                Self::ProviderReadbackUpload {
                    device_id: upload,
                    preparation,
                    readbacks,
                    ..
                },
                CandidateExecutionLocation::Host,
                CandidateOutputResidency::Provider { device_id: output }
                | CandidateOutputResidency::Mirrored { device_id: output },
            ) if *upload == output
                && !matches!(preparation, CandidatePreparationState::Preparing | CandidatePreparationState::Unknown)
                && readbacks.is_canonical() && readbacks.iter().next().is_some() => Ok(()),
            (
                Self::ConditionalProviderUpload { device_id: upload, queue_slots, host_result_bytes: Some(_), .. },
                CandidateExecutionLocation::Provider { device_id: execution },
                CandidateOutputResidency::Provider { device_id: output },
            ) if *upload == execution && *upload == output && *queue_slots > 0 => Ok(()),
            (Self::ProviderUpload { .. }, _, _) => {
                Err("provider upload presentation needs a ready stage and matching host route and output device")
            }
            (Self::ProviderReadbackUpload { .. }, _, _) => {
                Err("provider readback/upload needs canonical input stages and a ready matching host-to-provider output")
            }
            (Self::ConditionalProviderUpload { .. }, _, _) => {
                Err("conditional provider upload needs the same selected provider/output owner and a completion queue slot")
            }
            (Self::Direct, _, CandidateOutputResidency::ProviderAndHost { .. }) => {
                Err("mixed provider and host outputs need an admitted produced-output readback")
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
        if !self.kind.matches_location(self.execution_location) {
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

#[cfg(test)]
mod produced_readback_tests {
    use super::*;

    #[test]
    fn mixed_output_requires_exact_selected_provider_and_real_readback_stage() {
        let stage = CandidateOutputPresentation::ProviderProducedReadback {
            device_id: 7,
            transfer_bytes: 12,
            provider_scratch_bytes: ResourceByteBound::Known(12),
            queue_slots: 1,
        };
        let output = CandidateOutputResidency::ProviderAndHost {
            device_id: 7,
            host_bytes: ResourceByteBound::Known(24),
        };
        assert_eq!(
            stage.validate_for(
                CandidateExecutionLocation::Provider { device_id: 7 },
                output
            ),
            Ok(())
        );
        assert!(stage
            .validate_for(
                CandidateExecutionLocation::Provider { device_id: 8 },
                output
            )
            .is_err());
        assert!(stage
            .validate_for(CandidateExecutionLocation::Host, output)
            .is_err());
        assert!(CandidateOutputPresentation::Direct
            .validate_for(CandidateExecutionLocation::Host, output)
            .is_err());
        let no_completion = CandidateOutputPresentation::ProviderProducedReadback {
            device_id: 7,
            transfer_bytes: 12,
            provider_scratch_bytes: ResourceByteBound::Known(12),
            queue_slots: 0,
        };
        assert!(no_completion
            .validate_for(
                CandidateExecutionLocation::Provider { device_id: 7 },
                output
            )
            .is_err());
    }
}
