use std::collections::HashMap;
use std::rc::Rc;

use super::request::BrowserWorkerRequest;
use runmat_execution::value::PayloadGraph;
use runmat_execution::{ExecutionScopeId, FutureId, TaskId};
use runmat_execution_runner::Driver;
use runmat_runtime::execution::{DeferredCall, ExecutionServiceError, GraphResultReceipt};

pub(super) enum FutureState {
    Deferred(Rc<DeferredCall>),
    Materializing {
        id: u64,
        call: Rc<DeferredCall>,
        cancelled: bool,
    },
    ExecutingInCaller,
    Scheduled {
        task_id: TaskId,
        receipt: Option<Rc<dyn GraphResultReceipt>>,
    },
    UnsettledCallback {
        task_id: TaskId,
        receipt: Option<Rc<dyn GraphResultReceipt>>,
        error: ExecutionServiceError,
    },
    CancellingCallback {
        task_id: TaskId,
        receipt: Option<Rc<dyn GraphResultReceipt>>,
    },
    GraphCompleted {
        graph: Rc<PayloadGraph>,
        receipt: Rc<dyn GraphResultReceipt>,
    },
    /// Keep the V8 failure wire and its exact selected receipt together until
    /// an explicit issuing RuntimeContext localizes the source frames.
    GraphFailed {
        failure: Rc<runmat_execution::ProgramRuntimeFailure>,
        receipt: Rc<dyn GraphResultReceipt>,
    },
    GraphResolved {
        result: Result<runmat_runtime::execution::RootedValueSequence, ExecutionServiceError>,
        receipt: Rc<dyn GraphResultReceipt>,
    },
    Completed(Result<runmat_runtime::execution::RootedValueSequence, ExecutionServiceError>),
    Cancelled,
}

pub(super) struct TaskRecord {
    pub(super) future_id: FutureId,
    pub(super) generation: u64,
    pub(super) scope_id: ExecutionScopeId,
    pub(super) read: bool,
    pub(super) claimed: bool,
    pub(super) completion_order: Option<u64>,
    pub(super) retention: runmat_runtime::execution::TaskRetention,
    pub(super) callback_exit: Option<super::service::PhysicalCallback>,
}

pub(super) struct State {
    pub(super) next_future: u64,
    pub(super) next_task: u64,
    pub(super) next_materialization: u64,
    pub(super) next_completion: u64,
    pub(super) futures: HashMap<FutureId, FutureState>,
    pub(super) tasks: HashMap<TaskId, TaskRecord>,
    pub(super) requests: HashMap<TaskId, BrowserWorkerRequest>,
    pub(super) driver: Driver,
    pub(super) pool_generation: u64,
    pub(super) pool_open: bool,
}
