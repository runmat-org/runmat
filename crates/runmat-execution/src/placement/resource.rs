use serde::{Deserialize, Serialize};

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
    pub fn provider(&self, device_id: u32) -> Option<&ProviderResourceSnapshot> {
        self.providers
            .iter()
            .find(|provider| provider.device_id == device_id)
    }
}
