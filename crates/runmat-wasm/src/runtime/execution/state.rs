use std::collections::HashMap;

use runmat_execution::{ExecutionScopeId, FutureId, TaskId};
use runmat_execution_artifact::ProgramExecutionRequest;
use runmat_execution_runner::Driver;
use runmat_runtime::execution::{DeferredCall, ExecutionServiceError};
use runmat_value::Value;

pub(super) enum FutureState {
    Deferred(DeferredCall),
    ExecutingInCaller,
    Scheduled(TaskId),
    Completed(Result<Value, ExecutionServiceError>),
    Cancelled,
}

pub(super) struct TaskRecord {
    pub(super) future_id: FutureId,
    pub(super) generation: u64,
    pub(super) scope_id: ExecutionScopeId,
}

pub(super) struct State {
    pub(super) next_future: u64,
    pub(super) next_task: u64,
    pub(super) futures: HashMap<FutureId, FutureState>,
    pub(super) tasks: HashMap<TaskId, TaskRecord>,
    pub(super) requests: HashMap<TaskId, ProgramExecutionRequest>,
    pub(super) driver: Driver,
}
