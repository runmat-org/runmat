use runmat_execution::identity::WorkerId;
use runmat_execution::resource::ResourceInventory;
use runmat_execution::PoolId;
use serde::{Deserialize, Serialize};

use crate::{RunnerError, RunnerResult};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PoolSpec {
    pub id: PoolId,
    pub min_workers: u32,
    pub max_workers: u32,
    pub max_in_flight: u32,
    pub resource_limit: ResourceInventory,
}

impl PoolSpec {
    pub fn validate(&self) -> RunnerResult<()> {
        if self.max_workers == 0
            || self.max_in_flight == 0
            || self.min_workers > self.max_workers
            || self.resource_limit.cpu_millicores == 0
            || self.resource_limit.memory_bytes == 0
        {
            return Err(RunnerError::Invalid(
                "pool worker and in-flight bounds are inconsistent".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WorkerSpec {
    pub id: WorkerId,
    pub pool_id: PoolId,
    pub resources: ResourceInventory,
}
