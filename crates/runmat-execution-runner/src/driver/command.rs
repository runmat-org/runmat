use runmat_execution::identity::WorkerId;
use runmat_execution::state::PoolState;
use runmat_execution::value::ValueRef;
use runmat_execution::{CancellationReason, ExecutionScopeId, PoolId, TaskId};
use serde::{Deserialize, Serialize};

use crate::cancellation::EscalationPolicy;
use crate::pool::{PoolSpec, ResizeRequest, WorkerSpec};
use crate::port::BackendReport;
use crate::scheduler::FairnessPolicy;
use crate::task::{AttemptRequest, TaskSubmission};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DriverConfig {
    pub max_in_flight: u32,
    pub fairness: FairnessPolicy,
    pub cancellation_escalation: EscalationPolicy,
}

impl Default for DriverConfig {
    fn default() -> Self {
        Self {
            max_in_flight: 1024,
            fairness: FairnessPolicy::default(),
            cancellation_escalation: EscalationPolicy::default(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DriverCommand {
    RegisterScope {
        scope_id: ExecutionScopeId,
        parent: Option<ExecutionScopeId>,
    },
    CreatePool(PoolSpec),
    SetPoolState {
        pool_id: PoolId,
        state: PoolState,
    },
    ResizePool {
        pool_id: PoolId,
        request: ResizeRequest,
    },
    RegisterWorker(WorkerSpec),
    DrainWorker(WorkerId),
    WorkerLost(WorkerId),
    Submit(Box<TaskSubmission>),
    BackendReport(BackendReport),
    CancelScope {
        scope_id: ExecutionScopeId,
        reason: CancellationReason,
        now_millis: u64,
    },
    Tick {
        now_millis: u64,
    },
    Checkpoint,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DriverAction {
    Launch(AttemptRequest),
    Cancel(AttemptRequest),
    Terminate(AttemptRequest),
    ResizePool {
        pool_id: PoolId,
        desired_workers: u32,
    },
    Checkpoint,
    GarbageCollectResults {
        task_id: TaskId,
        objects: Vec<ValueRef>,
    },
}
