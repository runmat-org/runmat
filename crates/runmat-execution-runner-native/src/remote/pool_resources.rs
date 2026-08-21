use runmat_execution::resource::{
    AcceleratorRequest, ResourceInventory, ResourceRequest as TaskResources,
};
use runmat_execution_transport_native::control::ResourceRequest as AllocationResources;

use crate::{NativeExecutionError, NativeExecutionResult};

pub(super) fn inventory(
    resources: &AllocationResources,
) -> NativeExecutionResult<ResourceInventory> {
    Ok(ResourceInventory {
        cpu_millicores: u32::try_from(resources.cpu_millicores)
            .map_err(|_| protocol("worker CPU request overflows scheduler units"))?,
        memory_bytes: resources.memory_bytes,
        scratch_bytes: resources.scratch_bytes,
        accelerators: accelerators(resources)?,
        capabilities: Default::default(),
    })
}

pub(super) fn pool_inventory(
    resources: &AllocationResources,
    workers: u32,
) -> NativeExecutionResult<ResourceInventory> {
    let mut inventory = inventory(resources)?;
    inventory.cpu_millicores = inventory
        .cpu_millicores
        .checked_mul(workers)
        .ok_or_else(|| protocol("worker pool CPU capacity overflows scheduler units"))?;
    inventory.memory_bytes = inventory
        .memory_bytes
        .checked_mul(u64::from(workers))
        .ok_or_else(|| protocol("worker pool memory capacity overflows scheduler units"))?;
    inventory.scratch_bytes = inventory
        .scratch_bytes
        .checked_mul(u64::from(workers))
        .ok_or_else(|| protocol("worker pool scratch capacity overflows scheduler units"))?;
    for accelerator in &mut inventory.accelerators {
        accelerator.count =
            accelerator
                .count
                .checked_mul(u16::try_from(workers).map_err(|_| {
                    protocol("worker pool size overflows accelerator scheduler units")
                })?)
                .ok_or_else(|| protocol("worker pool accelerators overflow scheduler units"))?;
    }
    Ok(inventory)
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
        accelerators: accelerators(resources)?,
        required_capabilities: Default::default(),
    })
}

fn accelerators(resources: &AllocationResources) -> NativeExecutionResult<Vec<AcceleratorRequest>> {
    match (
        resources.accelerator_count,
        resources.accelerator_class.as_ref(),
    ) {
        (0, _) => Ok(Vec::new()),
        (count, Some(class)) => Ok(vec![AcceleratorRequest {
            class: class.clone(),
            count: u16::try_from(count)
                .map_err(|_| protocol("worker accelerator count overflows scheduler units"))?,
            memory_bytes_each: resources.accelerator_memory_bytes / u64::from(count),
        }]),
        _ => Err(protocol("worker accelerator class is missing")),
    }
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
