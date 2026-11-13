use std::collections::BTreeSet;

use runmat_execution::resource::{
    accelerator_devices_exactly_satisfy, AcceleratorAllocationDomainId,
};
use runmat_execution_transport_native::control::{NodeAllocation, NodeInventory};

use crate::{AgentError, AgentResult};

pub fn validate_offer(
    allocation: &NodeAllocation,
    inventory: &NodeInventory,
    now_millis: i64,
) -> AgentResult<()> {
    if allocation.state != "offered"
        || allocation.fencing_token == 0
        || allocation.expires_at_millis <= now_millis
    {
        return Err(AgentError::AllocationRejected(
            "lease is stale, expired, or not offered".to_string(),
        ));
    }
    validate_resources(allocation, inventory)
}

pub fn validate_active(
    allocation: &NodeAllocation,
    inventory: &NodeInventory,
    now_millis: i64,
) -> AgentResult<()> {
    if allocation.state != "active"
        || allocation.fencing_token == 0
        || allocation.expires_at_millis <= now_millis
    {
        return Err(AgentError::AllocationRejected(
            "lease is stale, expired, or not active".to_string(),
        ));
    }
    validate_resources(allocation, inventory)
}

/// Verifies that the complete live allocation set fits the node inventory.
/// Individual offers are necessary but not sufficient when an operator permits
/// more than one concurrent allocation on a node.
pub fn validate_allocation_set(
    allocations: &[NodeAllocation],
    inventory: &NodeInventory,
    now_millis: i64,
) -> AgentResult<()> {
    let mut cpu_millicores = 0_u64;
    let mut memory_bytes = 0_u64;
    let mut scratch_bytes = 0_u64;
    let mut allocation_ids = BTreeSet::new();
    let mut allocation_domains = BTreeSet::<AcceleratorAllocationDomainId>::new();
    for allocation in allocations.iter().filter(|allocation| {
        matches!(allocation.state.as_str(), "offered" | "active")
            && allocation.expires_at_millis > now_millis
    }) {
        if !allocation_ids.insert(allocation.id.as_str()) {
            return Err(overcommitted());
        }
        match allocation.state.as_str() {
            "offered" => validate_offer(allocation, inventory, now_millis)?,
            "active" => validate_active(allocation, inventory, now_millis)?,
            _ => unreachable!("live allocation filter admits only offered and active states"),
        }
        cpu_millicores = cpu_millicores
            .checked_add(allocation.resources.cpu_millicores)
            .ok_or_else(overcommitted)?;
        memory_bytes = memory_bytes
            .checked_add(allocation.resources.memory_bytes)
            .ok_or_else(overcommitted)?;
        scratch_bytes = scratch_bytes
            .checked_add(allocation.resources.scratch_bytes)
            .ok_or_else(overcommitted)?;
        for device in &allocation.accelerator_devices {
            if !allocation_domains.insert(device.allocation_domain.clone()) {
                return Err(overcommitted());
            }
        }
    }
    if cpu_millicores > inventory.cpu_millicores
        || memory_bytes > inventory.memory_bytes
        || scratch_bytes > inventory.scratch_bytes
    {
        return Err(overcommitted());
    }
    Ok(())
}

fn validate_resources(allocation: &NodeAllocation, inventory: &NodeInventory) -> AgentResult<()> {
    let request = &allocation.resources;
    if request.cpu_millicores > inventory.cpu_millicores
        || request.memory_bytes > inventory.memory_bytes
        || request.scratch_bytes > inventory.scratch_bytes
        || !accelerator_devices_exactly_satisfy(
            &allocation.accelerator_devices,
            &request.accelerators,
        )
        || allocation.accelerator_devices.iter().any(|assigned| {
            inventory
                .accelerators
                .iter()
                .find(|available| available.id == assigned.id)
                != Some(assigned)
        })
    {
        return Err(AgentError::AllocationRejected(
            "inventory does not satisfy the allocation".to_string(),
        ));
    }
    Ok(())
}

fn overcommitted() -> AgentError {
    AgentError::AllocationRejected(
        "live allocations exceed the node's scalar or physical-device inventory".into(),
    )
}
