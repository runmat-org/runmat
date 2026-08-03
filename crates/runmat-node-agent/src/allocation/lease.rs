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

fn validate_resources(allocation: &NodeAllocation, inventory: &NodeInventory) -> AgentResult<()> {
    let request = &allocation.resources;
    if request.cpu_millicores > inventory.cpu_millicores
        || request.memory_bytes > inventory.memory_bytes
        || request.scratch_bytes > inventory.scratch_bytes
        || request.accelerator_count > inventory.accelerator_count
        || request.accelerator_memory_bytes > inventory.accelerator_memory_bytes
        || request
            .accelerator_class
            .as_ref()
            .is_some_and(|class| inventory.accelerator_class.as_ref() != Some(class))
    {
        return Err(AgentError::AllocationRejected(
            "inventory does not satisfy the allocation".to_string(),
        ));
    }
    Ok(())
}
