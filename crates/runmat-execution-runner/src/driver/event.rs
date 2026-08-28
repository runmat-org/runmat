use runmat_execution::identity::{AttemptId, ResultCommitId, WorkerId};
use runmat_execution::state::{PoolState, TaskState};
use runmat_execution::{CancellationReason, ExecutionScopeId, PoolId, TaskId};
use serde::{Deserialize, Serialize};

use crate::task::AttemptFailureKind;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DriverEvent {
    pub sequence: u64,
    pub driver_fence: u64,
    pub kind: DriverEventKind,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum DriverEventKind {
    ScopeRegistered {
        scope_id: ExecutionScopeId,
    },
    PoolCreated {
        pool_id: PoolId,
    },
    PoolStateChanged {
        pool_id: PoolId,
        state: PoolState,
    },
    PoolResizeRequested {
        pool_id: PoolId,
        desired_workers: u32,
    },
    WorkerRegistered {
        worker_id: WorkerId,
        pool_id: PoolId,
    },
    WorkerDraining {
        worker_id: WorkerId,
    },
    WorkerLost {
        worker_id: WorkerId,
    },
    TaskSubmitted {
        task_id: TaskId,
        state: TaskState,
    },
    TaskStateChanged {
        task_id: TaskId,
        state: TaskState,
    },
    AttemptAssigned {
        task_id: TaskId,
        attempt_id: AttemptId,
        worker_id: WorkerId,
    },
    AttemptStarted {
        task_id: TaskId,
        attempt_id: AttemptId,
    },
    AttemptFailed {
        task_id: TaskId,
        attempt_id: AttemptId,
        kind: AttemptFailureKind,
    },
    AttemptLost {
        task_id: TaskId,
        attempt_id: AttemptId,
    },
    AttemptCancelled {
        task_id: TaskId,
        attempt_id: AttemptId,
    },
    ResultCommitted {
        task_id: TaskId,
        attempt_id: AttemptId,
        commit_id: ResultCommitId,
    },
    ReportDiscarded {
        task_id: TaskId,
        attempt_id: AttemptId,
        reason: String,
    },
    ScopeCancelled {
        scope_id: ExecutionScopeId,
        reason: CancellationReason,
    },
    DeadlineExpired {
        task_id: TaskId,
    },
    CheckpointRequested,
}
