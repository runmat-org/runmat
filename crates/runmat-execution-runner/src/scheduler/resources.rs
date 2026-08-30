use std::collections::{BTreeMap, BTreeSet};

use runmat_execution::identity::AttemptId;
use runmat_execution::resource::{
    select_accelerator_devices, AcceleratorAllocationDomainId, AcceleratorClass, AcceleratorDevice,
    AcceleratorDeviceLease, ResourceAssignment, ResourceInventory, ResourceRequest,
};
use serde::{Deserialize, Serialize};

use crate::{RunnerError, RunnerResult};

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub cpu_millicores: u32,
    pub memory_bytes: u64,
    pub scratch_bytes: u64,
    pub accelerator_counts: BTreeMap<AcceleratorClass, u16>,
    pub leased_accelerator_domains: BTreeSet<AcceleratorAllocationDomainId>,
}

pub fn fits(
    inventory: &ResourceInventory,
    allocated: &ResourceAllocation,
    request: &ResourceRequest,
) -> bool {
    scalar_resources_fit(inventory, allocated, request)
        && select_devices(inventory, allocated, request).is_some()
}

pub fn scalar_resources_fit(
    inventory: &ResourceInventory,
    allocated: &ResourceAllocation,
    request: &ResourceRequest,
) -> bool {
    request
        .required_capabilities
        .is_subset(&inventory.capabilities)
        && allocated
            .cpu_millicores
            .saturating_add(request.cpu_millicores)
            <= inventory.cpu_millicores
        && allocated.memory_bytes.saturating_add(request.memory_bytes) <= inventory.memory_bytes
        && allocated
            .scratch_bytes
            .saturating_add(request.scratch_bytes)
            <= inventory.scratch_bytes
}

pub fn select_devices(
    inventory: &ResourceInventory,
    allocated: &ResourceAllocation,
    request: &ResourceRequest,
) -> Option<Vec<AcceleratorDevice>> {
    select_accelerator_devices(
        &inventory.accelerators,
        &request.accelerators,
        &allocated.leased_accelerator_domains,
    )
}

pub fn assignment_for(
    attempt_id: AttemptId,
    fencing_token: u64,
    devices: &[AcceleratorDevice],
) -> RunnerResult<ResourceAssignment> {
    if fencing_token == 0 {
        return Err(RunnerError::Invalid(
            "resource lease fencing token must be non-zero".into(),
        ));
    }
    let accelerator_leases = devices
        .iter()
        .map(|device| AcceleratorDeviceLease::for_attempt(attempt_id, fencing_token, device))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| RunnerError::Invalid(error.to_string()))?;
    let assignment = ResourceAssignment { accelerator_leases };
    assignment
        .validate()
        .map_err(|error| RunnerError::Invalid(error.to_string()))?;
    Ok(assignment)
}

pub fn reserve(
    allocated: &mut ResourceAllocation,
    request: &ResourceRequest,
    assignment: &ResourceAssignment,
) -> RunnerResult<()> {
    assignment
        .validate_for_request(request)
        .map_err(|error| RunnerError::Invalid(error.to_string()))?;
    let cpu_millicores = allocated
        .cpu_millicores
        .checked_add(request.cpu_millicores)
        .ok_or_else(|| RunnerError::Invalid("CPU allocation overflow".into()))?;
    let memory_bytes = allocated
        .memory_bytes
        .checked_add(request.memory_bytes)
        .ok_or_else(|| RunnerError::Invalid("memory allocation overflow".into()))?;
    let scratch_bytes = allocated
        .scratch_bytes
        .checked_add(request.scratch_bytes)
        .ok_or_else(|| RunnerError::Invalid("scratch allocation overflow".into()))?;
    if let Some(lease) = assignment.accelerator_leases.iter().find(|lease| {
        allocated
            .leased_accelerator_domains
            .contains(&lease.allocation_domain)
    }) {
        return Err(RunnerError::Invalid(format!(
            "accelerator device {} is already leased",
            lease.device_id
        )));
    }
    let mut accelerator_counts = allocated.accelerator_counts.clone();
    for lease in &assignment.accelerator_leases {
        let count = accelerator_counts.entry(lease.class.clone()).or_default();
        *count = count
            .checked_add(1)
            .ok_or_else(|| RunnerError::Invalid("accelerator allocation overflow".into()))?;
    }
    allocated.cpu_millicores = cpu_millicores;
    allocated.memory_bytes = memory_bytes;
    allocated.scratch_bytes = scratch_bytes;
    allocated.accelerator_counts = accelerator_counts;
    allocated.leased_accelerator_domains.extend(
        assignment
            .accelerator_leases
            .iter()
            .map(|lease| lease.allocation_domain.clone()),
    );
    Ok(())
}

pub fn reserve_scalar(
    allocated: &mut ResourceAllocation,
    request: &ResourceRequest,
) -> RunnerResult<()> {
    let cpu_millicores = allocated
        .cpu_millicores
        .checked_add(request.cpu_millicores)
        .ok_or_else(|| RunnerError::Invalid("CPU allocation overflow".into()))?;
    let memory_bytes = allocated
        .memory_bytes
        .checked_add(request.memory_bytes)
        .ok_or_else(|| RunnerError::Invalid("memory allocation overflow".into()))?;
    let scratch_bytes = allocated
        .scratch_bytes
        .checked_add(request.scratch_bytes)
        .ok_or_else(|| RunnerError::Invalid("scratch allocation overflow".into()))?;
    allocated.cpu_millicores = cpu_millicores;
    allocated.memory_bytes = memory_bytes;
    allocated.scratch_bytes = scratch_bytes;
    Ok(())
}

pub fn release(
    allocated: &mut ResourceAllocation,
    request: &ResourceRequest,
    assignment: &ResourceAssignment,
) {
    allocated.cpu_millicores = allocated
        .cpu_millicores
        .saturating_sub(request.cpu_millicores);
    allocated.memory_bytes = allocated.memory_bytes.saturating_sub(request.memory_bytes);
    allocated.scratch_bytes = allocated
        .scratch_bytes
        .saturating_sub(request.scratch_bytes);
    for lease in &assignment.accelerator_leases {
        allocated
            .leased_accelerator_domains
            .remove(&lease.allocation_domain);
        if let Some(count) = allocated.accelerator_counts.get_mut(&lease.class) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                allocated.accelerator_counts.remove(&lease.class);
            }
        }
    }
}

pub fn release_scalar(allocated: &mut ResourceAllocation, request: &ResourceRequest) {
    allocated.cpu_millicores = allocated
        .cpu_millicores
        .saturating_sub(request.cpu_millicores);
    allocated.memory_bytes = allocated.memory_bytes.saturating_sub(request.memory_bytes);
    allocated.scratch_bytes = allocated
        .scratch_bytes
        .saturating_sub(request.scratch_bytes);
}
