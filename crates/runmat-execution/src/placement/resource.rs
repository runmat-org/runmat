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
        let mut provider_ids = BTreeSet::new();
        for provider in &self.providers {
            if !provider_ids.insert(provider.device_id) {
                return Err("provider resource device identities must be unique");
            }
            if provider
                .capacity_bytes
                .is_some_and(|capacity| provider.live_bytes > capacity)
                || provider
                    .scratch_available_bytes
                    .zip(provider.capacity_bytes)
                    .is_some_and(|(scratch, capacity)| scratch > capacity)
                || provider.reclaimable_bytes > provider.live_bytes
                || provider.queue_depth.is_some() != provider.queue_limit.is_some()
                || provider
                    .queue_depth
                    .zip(provider.queue_limit)
                    .is_some_and(|(depth, limit)| depth > limit)
            {
                return Err("provider resource snapshot is inconsistent");
            }
        }
        Ok(())
    }

    pub fn provider(&self, device_id: u32) -> Option<&ProviderResourceSnapshot> {
        self.providers
            .iter()
            .find(|provider| provider.device_id == device_id)
    }
}
