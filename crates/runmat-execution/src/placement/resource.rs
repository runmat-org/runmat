use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderResourceSnapshot {
    pub device_id: u32,
    /// Total allocatable capacity when the provider exposes it. WebGPU, for
    /// example, exposes buffer limits but no trustworthy total-memory budget.
    pub capacity_bytes: Option<u64>,
    pub live_bytes: u64,
    pub reclaimable_bytes: u64,
    pub scratch_available_bytes: Option<u64>,
    /// Queue occupancy and limit are either both known or both unknown.
    pub queue_depth: Option<u32>,
    pub queue_limit: Option<u32>,
    pub lost: bool,
    pub epoch: u64,
}

impl ProviderResourceSnapshot {
    pub fn validate_typed(&self) -> Result<(), ResourceSnapshotError> {
        if self
            .capacity_bytes
            .is_some_and(|capacity| self.live_bytes > capacity)
            || self
                .scratch_available_bytes
                .zip(self.capacity_bytes)
                .is_some_and(|(scratch, capacity)| scratch > capacity)
            || self.reclaimable_bytes > self.live_bytes
            || self.queue_depth.is_some() != self.queue_limit.is_some()
            || self
                .queue_depth
                .zip(self.queue_limit)
                .is_some_and(|(depth, limit)| depth > limit)
        {
            return Err(ResourceSnapshotError::InconsistentProvider {
                device_id: self.device_id,
            });
        }
        Ok(())
    }
    pub fn immediately_available_bytes(&self) -> Option<u64> {
        self.capacity_bytes
            .map(|capacity| capacity.saturating_sub(self.live_bytes))
    }

    pub fn available_after_eviction_bytes(&self) -> Option<u64> {
        self.immediately_available_bytes()
            .map(|available| available.saturating_add(self.reclaimable_bytes.min(self.live_bytes)))
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementResourceSnapshot {
    pub cpu_millicores_available: u32,
    pub memory_available_bytes: Option<u64>,
    pub cancellation_requested: bool,
    pub providers: Vec<ProviderResourceSnapshot>,
    pub epoch: u64,
}

impl PlacementResourceSnapshot {
    pub fn validate(&self) -> Result<(), &'static str> {
        self.validate_typed()
            .map_err(ResourceSnapshotError::message)
    }

    pub fn validate_typed(&self) -> Result<(), ResourceSnapshotError> {
        if let [provider] = self.providers.as_slice() {
            return provider.validate_typed();
        }
        let mut provider_ids = BTreeSet::new();
        for provider in &self.providers {
            if !provider_ids.insert(provider.device_id) {
                return Err(ResourceSnapshotError::DuplicateProvider {
                    device_id: provider.device_id,
                });
            }
            provider.validate_typed()?;
        }
        Ok(())
    }

    pub fn provider(&self, device_id: u32) -> Option<&ProviderResourceSnapshot> {
        self.providers
            .iter()
            .find(|provider| provider.device_id == device_id)
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceSnapshotError {
    DuplicateProvider { device_id: u32 },
    InconsistentProvider { device_id: u32 },
}

impl ResourceSnapshotError {
    fn message(self) -> &'static str {
        match self {
            Self::DuplicateProvider { .. } => "provider resource device identities must be unique",
            Self::InconsistentProvider { .. } => "provider resource snapshot is inconsistent",
        }
    }
}
impl std::fmt::Display for ResourceSnapshotError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.message())
    }
}
impl std::error::Error for ResourceSnapshotError {}

mod account;
mod admission;
pub use account::{ResourceAccount, ResourceAdmissionError, ResourceBudgetKind};
pub use admission::{
    admit_candidate_resources, single_candidate_resources_admitted, AdmittedResourceDemand,
    CandidateResourceFacts, ResourceAdmissionContext,
};
