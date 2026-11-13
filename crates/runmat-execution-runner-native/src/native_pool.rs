//! Register independently configured native capacity with the shared portable scheduler.
use crate::{NativeExecutionConfig, NativeExecutionError, NativeExecutionResult};
use runmat_execution::{
    identity::WorkerId, resource::ResourceInventory, state::PoolState, ExecutionScopeId, PoolId,
};
use runmat_execution_runner::{Driver, DriverCommand, PoolSpec, WorkerSpec};

pub(crate) fn register(
    driver: &mut Driver,
    config: &NativeExecutionConfig,
    scope_id: ExecutionScopeId,
) -> NativeExecutionResult<PoolId> {
    register_with_inventory(driver, config, scope_id, None)
}

/// A signed remote allocation replaces the local machine inventory for its
/// single owned worker; otherwise each slot receives the configured capacity.
pub(crate) fn register_with_inventory(
    driver: &mut Driver,
    config: &NativeExecutionConfig,
    scope_id: ExecutionScopeId,
    remote_inventory: Option<&ResourceInventory>,
) -> NativeExecutionResult<PoolId> {
    config.validate().map_err(invalid)?;
    let pool_id = PoolId::derive(&[scope_id.bytes(), b"local"]);
    let cpu = config
        .max_workers
        .checked_mul(config.worker_cpu_millicores)
        .ok_or_else(|| invalid("native pool CPU inventory overflowed"))?;
    let memory = u64::from(config.max_workers)
        .checked_mul(config.worker_memory_bytes)
        .ok_or_else(|| invalid("native pool memory inventory overflowed"))?;
    let scratch = u64::from(config.max_workers)
        .checked_mul(config.worker_scratch_bytes)
        .ok_or_else(|| invalid("native pool scratch inventory overflowed"))?;
    let resources = remote_inventory
        .cloned()
        .unwrap_or_else(|| ResourceInventory {
            cpu_millicores: config
                .local_cpu_capacity_millicores
                .map_or(cpu, |capacity| cpu.min(capacity)),
            memory_bytes: config
                .local_memory_capacity_bytes
                .map_or(memory, |capacity| memory.min(capacity)),
            scratch_bytes: scratch,
            accelerators: config.accelerator_devices.clone(),
            capabilities: config.worker_capabilities.clone(),
        });
    driver.handle(DriverCommand::RegisterScope {
        scope_id,
        parent: None,
    })?;
    driver.handle(DriverCommand::CreatePool(PoolSpec {
        id: pool_id,
        min_workers: 1,
        max_workers: config.max_workers,
        max_in_flight: config.max_workers,
        resource_limit: resources.clone(),
    }))?;
    driver.handle(DriverCommand::SetPoolState {
        pool_id,
        state: PoolState::Ready,
    })?;
    for index in 0..config.max_workers {
        let index_bytes = index.to_be_bytes();
        let worker_id = WorkerId::derive(&[pool_id.bytes().as_slice(), &index_bytes]);
        driver.handle(DriverCommand::RegisterWorker(WorkerSpec {
            id: worker_id,
            pool_id,
            resources: remote_inventory
                .cloned()
                .unwrap_or_else(|| ResourceInventory {
                    cpu_millicores: config.worker_cpu_millicores,
                    memory_bytes: config.worker_memory_bytes,
                    scratch_bytes: config.worker_scratch_bytes,
                    // Each physical device is leased exactly once: only the
                    // first slot advertises the local accelerator inventory.
                    accelerators: if index == 0 {
                        config.accelerator_devices.clone()
                    } else {
                        Vec::new()
                    },
                    capabilities: config.worker_capabilities.clone(),
                }),
            host: config.host_inventory.clone(),
        }))?;
    }
    Ok(pool_id)
}

fn invalid(message: impl Into<String>) -> NativeExecutionError {
    NativeExecutionError::Configuration(message.into())
}

#[cfg(test)]
mod tests;
