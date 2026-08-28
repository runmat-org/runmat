use serde::{Deserialize, Serialize};

use crate::identity::{ExecutionScopeId, FutureId, JobId, PoolId, RunId, TaskId};
use crate::state::PoolState;

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

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PoolBackend {
    Serial,
    LocalProcesses,
    BrowserWorkers,
    Remote,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct PoolRequest {
    pub backend: Option<PoolBackend>,
    pub workers: Option<u32>,
}

impl PoolRequest {
    pub const fn automatic() -> Self {
        Self {
            backend: None,
            workers: None,
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct PoolSnapshot {
    pub handle: PoolHandle,
    pub backend: PoolBackend,
    pub workers: u32,
    pub state: PoolState,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionHandleState {
    Deferred,
    Queued,
    Running,
    Finished,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct ExecutionHandleSnapshot {
    pub state: ExecutionHandleState,
    pub outputs: OutputContract,
    pub read: bool,
}

/// Result of atomically attempting to reserve a completed task result.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskResultClaim {
    Pending,
    Exhausted,
    Claimed { index: usize },
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct JobHandle {
    pub id: JobId,
    pub run_id: RunId,
    pub generation: u64,
    pub outputs: OutputContract,
}
