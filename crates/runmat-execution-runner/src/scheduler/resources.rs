use std::collections::BTreeMap;

use runmat_execution::resource::{ResourceInventory, ResourceRequest};
use serde::{Deserialize, Serialize};

use crate::{RunnerError, RunnerResult};

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub cpu_millicores: u32,
    pub memory_bytes: u64,
    pub scratch_bytes: u64,
    pub accelerator_counts: BTreeMap<String, u16>,
}

impl ResourceAllocation {
    pub fn from_request(request: &ResourceRequest) -> Self {
        let mut accelerator_counts = BTreeMap::new();
        for accelerator in &request.accelerators {
            *accelerator_counts
                .entry(accelerator.class.clone())
                .or_default() += accelerator.count;
        }
        Self {
            cpu_millicores: request.cpu_millicores,
            memory_bytes: request.memory_bytes,
            scratch_bytes: request.scratch_bytes,
            accelerator_counts,
        }
    }
}

pub fn fits(
    inventory: &ResourceInventory,
    allocated: &ResourceAllocation,
    request: &ResourceRequest,
) -> bool {
    if !request
        .required_capabilities
        .is_subset(&inventory.capabilities)
    {
        return false;
    }
    if allocated
        .cpu_millicores
        .saturating_add(request.cpu_millicores)
        > inventory.cpu_millicores
        || allocated.memory_bytes.saturating_add(request.memory_bytes) > inventory.memory_bytes
        || allocated
            .scratch_bytes
            .saturating_add(request.scratch_bytes)
            > inventory.scratch_bytes
    {
        return false;
    }
    let mut requested_accelerators: BTreeMap<&str, (u16, u64)> = BTreeMap::new();
    for requested in &request.accelerators {
        let entry = requested_accelerators.entry(&requested.class).or_default();
        entry.0 = entry.0.saturating_add(requested.count);
        entry.1 = entry.1.max(requested.memory_bytes_each);
    }
    requested_accelerators
        .into_iter()
        .all(|(class, (requested_count, requested_memory))| {
            let total = inventory
                .accelerators
                .iter()
                .filter(|available| {
                    available.class == class && available.memory_bytes_each >= requested_memory
                })
                .map(|available| available.count)
                .sum::<u16>();
            let used = allocated
                .accelerator_counts
                .get(class)
                .copied()
                .unwrap_or_default();
            used.saturating_add(requested_count) <= total
        })
}

pub fn reserve(allocated: &mut ResourceAllocation, request: &ResourceRequest) -> RunnerResult<()> {
    allocated.cpu_millicores = allocated
        .cpu_millicores
        .checked_add(request.cpu_millicores)
        .ok_or_else(|| RunnerError::Invalid("CPU allocation overflow".into()))?;
    allocated.memory_bytes = allocated
        .memory_bytes
        .checked_add(request.memory_bytes)
        .ok_or_else(|| RunnerError::Invalid("memory allocation overflow".into()))?;
    allocated.scratch_bytes = allocated
        .scratch_bytes
        .checked_add(request.scratch_bytes)
        .ok_or_else(|| RunnerError::Invalid("scratch allocation overflow".into()))?;
    for accelerator in &request.accelerators {
        let count = allocated
            .accelerator_counts
            .entry(accelerator.class.clone())
            .or_default();
        *count = count
            .checked_add(accelerator.count)
            .ok_or_else(|| RunnerError::Invalid("accelerator allocation overflow".into()))?;
    }
    Ok(())
}

pub fn release(allocated: &mut ResourceAllocation, request: &ResourceRequest) {
    allocated.cpu_millicores = allocated
        .cpu_millicores
        .saturating_sub(request.cpu_millicores);
    allocated.memory_bytes = allocated.memory_bytes.saturating_sub(request.memory_bytes);
    allocated.scratch_bytes = allocated
        .scratch_bytes
        .saturating_sub(request.scratch_bytes);
    for accelerator in &request.accelerators {
        if let Some(count) = allocated.accelerator_counts.get_mut(&accelerator.class) {
            *count = count.saturating_sub(accelerator.count);
            if *count == 0 {
                allocated.accelerator_counts.remove(&accelerator.class);
            }
        }
    }
}
