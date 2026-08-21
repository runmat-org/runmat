use serde::{Deserialize, Serialize};

use crate::identity::{ExecutionScopeId, FutureId, JobId, PoolId, RunId, TaskId};

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct OutputContract {
    pub requested_outputs: u16,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct FutureHandle {
    pub id: FutureId,
    pub scope_id: ExecutionScopeId,
    pub outputs: OutputContract,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct TaskHandle {
    pub id: TaskId,
    pub scope_id: ExecutionScopeId,
    pub generation: u64,
    pub outputs: OutputContract,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct PoolHandle {
    pub id: PoolId,
    pub scope_id: ExecutionScopeId,
    pub generation: u64,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct JobHandle {
    pub id: JobId,
    pub run_id: RunId,
    pub generation: u64,
    pub outputs: OutputContract,
}
