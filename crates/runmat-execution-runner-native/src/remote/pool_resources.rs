use runmat_execution::resource::{
    AcceleratorDevice, ResourceInventory, ResourceRequest as TaskResources,
};
use runmat_execution_transport_native::control::ResourceRequest as AllocationResources;

use crate::{NativeExecutionError, NativeExecutionResult};

pub(super) fn inventory(
    resources: &AllocationResources,
    accelerator_devices: &[AcceleratorDevice],
) -> NativeExecutionResult<ResourceInventory> {
    let inventory = ResourceInventory {
        cpu_millicores: u32::try_from(resources.cpu_millicores)
            .map_err(|_| protocol("worker CPU request overflows scheduler units"))?,
        memory_bytes: resources.memory_bytes,
        scratch_bytes: resources.scratch_bytes,
        accelerators: accelerator_devices.to_vec(),
        capabilities: Default::default(),
    };
    inventory
        .validate()
        .map_err(|error| protocol(format!("worker inventory is invalid: {error}")))?;
    Ok(inventory)
}

pub(super) fn pool_inventory(
    resources: &AllocationResources,
    workers: u32,
) -> NativeExecutionResult<ResourceInventory> {
    Ok(ResourceInventory {
        cpu_millicores: u32::try_from(resources.cpu_millicores)
            .map_err(|_| protocol("worker CPU request overflows scheduler units"))?
            .checked_mul(workers)
            .ok_or_else(|| protocol("worker pool CPU capacity overflows scheduler units"))?,
        memory_bytes: resources
            .memory_bytes
            .checked_mul(u64::from(workers))
            .ok_or_else(|| protocol("worker pool memory capacity overflows scheduler units"))?,
        scratch_bytes: resources
            .scratch_bytes
            .checked_mul(u64::from(workers))
            .ok_or_else(|| protocol("worker pool scratch capacity overflows scheduler units"))?,
        accelerators: Vec::new(),
        capabilities: Default::default(),
    })
}

pub(super) fn task_resources(
    resources: &AllocationResources,
) -> NativeExecutionResult<TaskResources> {
    Ok(TaskResources {
        cpu_millicores: u32::try_from(resources.cpu_millicores)
            .map_err(|_| protocol("worker CPU request overflows scheduler units"))?,
        memory_bytes: resources.memory_bytes,
        scratch_bytes: resources.scratch_bytes,
        max_wall_millis: resources.maximum_wall_millis,
        max_artifact_bytes: 64 * 1024 * 1024,
        max_egress_bytes: 64 * 1024 * 1024,
        max_relay_bytes: 4 * 1024 * 1024 * 1024,
        accelerators: resources.accelerators.clone(),
        required_capabilities: Default::default(),
    })
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
