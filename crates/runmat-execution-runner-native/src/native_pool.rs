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
    let resources = ResourceInventory {
        cpu_millicores: cpu,
        memory_bytes: memory,
        scratch_bytes: scratch,
        accelerators: config.worker_accelerators.clone(),
        capabilities: config.worker_capabilities.clone(),
    };
    driver.handle(DriverCommand::RegisterScope {
        scope_id,
        parent: None,
    })?;
    driver.handle(DriverCommand::CreatePool(PoolSpec {
        id: pool_id,
        min_workers: config.max_workers,
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
            resources: ResourceInventory {
                cpu_millicores: config.worker_cpu_millicores,
                memory_bytes: config.worker_memory_bytes,
                scratch_bytes: config.worker_scratch_bytes,
                accelerators: config.worker_accelerators.clone(),
                capabilities: config.worker_capabilities.clone(),
            },
        }))?;
    }
    Ok(pool_id)
}

fn invalid(message: impl Into<String>) -> NativeExecutionError {
    NativeExecutionError::Configuration(message.into())
}

#[cfg(test)]
mod tests;
