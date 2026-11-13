use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::rc::{Rc, Weak};

use runmat_execution::identity::{ArtifactId, WorkerId};
use runmat_execution::state::PoolState;
use runmat_execution::task::{Callable, TaskRequest};
use runmat_execution::{
    CancellationReason, ExecutionHandleSnapshot, ExecutionHandleState, ExecutionScopeId,
    FutureHandle, FutureId, OutputContract, PoolBackend, PoolHandle, PoolId, PoolRequest,
    PoolSnapshot, TaskHandle, TaskId, TaskResultClaim,
};
use runmat_execution_artifact::{
    ProgramExecutionRequest, ProgramExecutionResponse, PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
};
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::{
    AttemptFailureKind, AttemptReport, AttemptRequest, AttemptSuccess, Driver, DriverAction,
    DriverCommand, DriverConfig, PoolSpec, TaskSubmission, WorkerSpec,
};
use runmat_runtime::execution::{
    AwaitAction, CpuCooperationPolicy, DeferredCall, ExecutionServiceError,
    RuntimeExecutionServices,
};
use runmat_value::Value;

use super::host::BrowserExecutionHost;
use super::model::BrowserExecutionCapabilities;
use super::request::BrowserWorkerRequest;
use super::resources::{
    browser_accelerators, browser_execution_host_inventory, browser_inventory, browser_request,
    browser_worker_inventory, driver_error,
};
use super::state::{FutureState, State, TaskRecord};

mod worker_batch;
mod program_preparation;
pub(super) use worker_batch::PhysicalCallback;
mod physical_exit;
mod owned_builtin;
mod timer;
mod reporting;
use owned_builtin::OwnedTask;

pub(crate) struct BrowserExecutionService {
    clock: timer::DriverClock,
    scope_id: ExecutionScopeId,
    pool_id: PoolId,
    host: Option<BrowserExecutionHost>,
    capabilities: BrowserExecutionCapabilities,
    state: RefCell<State>,
    owned: RefCell<HashMap<TaskId, OwnedTask>>,
    callback_waiters: RefCell<HashMap<TaskId, worker_batch::TaskWake>>,
    self_weak: RefCell<Weak<Self>>,
    driver_timer: RefCell<Option<timer::DriverTimer>>,
}

/// Keep the deferred source graph visible to GC while an arbitrary VM graph
/// snapshot runs outside the RefCell borrow. Errors and unwinds restore only
/// this exact reservation; a concurrent/reentrant cancellation wins.
struct MaterializationGuard<'a> {
    service: &'a BrowserExecutionService,
    future_id: FutureId,
    id: u64,
    armed: bool,
}

struct GraphDecodeGuard<'a> {
    service: &'a BrowserExecutionService,
    future_id: FutureId,
    graph: Rc<runmat_execution::value::PayloadGraph>,
    receipt: Rc<dyn runmat_runtime::execution::GraphResultReceipt>,
    armed: bool,
}

impl Drop for GraphDecodeGuard<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let mut state = self.service.state.borrow_mut();
        if matches!(
            state.futures.get(&self.future_id),
            Some(FutureState::ExecutingInCaller)
        ) {
            state.futures.insert(
                self.future_id,
                FutureState::GraphCompleted {
                    graph: Rc::clone(&self.graph),
                    receipt: Rc::clone(&self.receipt),
                },
            );
        }
    }
}

impl Drop for MaterializationGuard<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let retired = {
            let mut state = self.service.state.borrow_mut();
            let Some(FutureState::Materializing {
                id,
                call,
                cancelled,
            }) = state.futures.get(&self.future_id)
            else {
                return;
            };
            if *id != self.id {
                return;
            }
            let next = if *cancelled {
                FutureState::Cancelled
            } else {
                FutureState::Deferred(Rc::clone(call))
            };
            state.futures.insert(self.future_id, next)
        };
        drop(retired);
    }
}

impl BrowserExecutionService {
    pub(crate) fn new(
        host: Option<BrowserExecutionHost>,
    ) -> Result<Rc<Self>, ExecutionServiceError> {
        let clock = timer::DriverClock::new();
        let capabilities = host
            .as_ref()
            .map(BrowserExecutionHost::capabilities)
            .unwrap_or_default();
        let session_nonce = uuid::Uuid::new_v4();
        let scope_id = ExecutionScopeId::derive(&[b"browser-session", session_nonce.as_bytes()]);
        let pool_id = PoolId::derive(&[scope_id.bytes(), b"browser"]);
        let qualified = host
            .as_ref()
            .and_then(BrowserExecutionHost::worker_inventory);
        let accelerators = match qualified {
            Some(inventory) => inventory.accelerators.clone(),
            None => browser_accelerators()?,
        };
        let resources = browser_inventory(capabilities, accelerators.clone());
        let execution_host = match qualified {
            Some(inventory) => inventory.host.clone(),
            None => browser_execution_host_inventory(&accelerators)?,
        };
        let mut driver = Driver::new(DriverConfig::default(), 1)
            .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
        driver
            .handle(DriverCommand::RegisterScope {
                scope_id,
                parent: None,
            })
            .map_err(driver_error)?;
        driver
            .handle(DriverCommand::CreatePool(PoolSpec {
                id: pool_id,
                min_workers: 1,
                max_workers: capabilities.max_workers,
                max_in_flight: capabilities.max_workers,
                resource_limit: resources.clone(),
            }))
            .map_err(driver_error)?;
        driver
            .handle(DriverCommand::SetPoolState {
                pool_id,
                state: PoolState::Ready,
            })
            .map_err(driver_error)?;
        for index in 0..capabilities.max_workers {
            let worker_id = WorkerId::derive(&[pool_id.bytes(), &index.to_be_bytes()]);
            // Only qualified worker devices enter placement. In particular a
            // CPU worker's explicit empty inventory cannot borrow UI devices.
            // Keep one owner per physical allocation domain as before.
            let worker_accelerators = if index == 0 {
                accelerators.clone()
            } else {
                Vec::new()
            };
            driver
                .handle(DriverCommand::RegisterWorker(WorkerSpec {
                    id: worker_id,
                    pool_id,
                    resources: browser_worker_inventory(capabilities, worker_accelerators),
                    host: execution_host.clone(),
                }))
                .map_err(driver_error)?;
        }
        driver
            .handle(DriverCommand::Tick {
                now_millis: clock.now_millis(),
            })
            .map_err(driver_error)?;
        let service = Rc::new(Self {
            clock,
            scope_id,
            pool_id,
            host,
            capabilities,
            state: RefCell::new(State {
                next_future: 0,
                next_task: 0,
                next_materialization: 0,
                next_completion: 0,
                futures: Default::default(),
                tasks: Default::default(),
                requests: Default::default(),
                driver,
                pool_generation: 1,
                pool_open: false,
            }),
            owned: RefCell::new(HashMap::new()),
            callback_waiters: RefCell::new(HashMap::new()),
            self_weak: RefCell::new(Weak::new()),
            driver_timer: RefCell::new(None),
        });
        service.self_weak.replace(Rc::downgrade(&service));
        service
            .driver_timer
            .replace(Some(timer::DriverTimer::new(Rc::downgrade(&service))?));
        Ok(service)
    }

    fn validate_scope(&self, scope_id: ExecutionScopeId) -> Result<(), ExecutionServiceError> {
        if scope_id == self.scope_id {
            Ok(())
        } else {
            Err(ExecutionServiceError::ForeignScope)
        }
    }

    fn pool_backend(&self) -> PoolBackend {
        if self.capabilities.has_worker_isolation() {
            PoolBackend::BrowserWorkers
        } else {
            PoolBackend::Serial
        }
    }

    fn pool_snapshot(&self, generation: u64, workers: u32) -> PoolSnapshot {
        PoolSnapshot {
            handle: PoolHandle {
                id: self.pool_id,
                scope_id: self.scope_id,
                generation,
            },
            backend: self.pool_backend(),
            workers,
            state: PoolState::Ready,
        }
    }

    fn active_workers(state: &State, pool_id: PoolId) -> Result<u32, ExecutionServiceError> {
        let snapshot = state.driver.snapshot();
        let pool = snapshot
            .pools
            .get(&pool_id)
            .ok_or_else(|| ExecutionServiceError::Failed("browser pool disappeared".into()))?;
        u32::try_from(
            pool.workers
                .values()
                .filter(|worker| worker.accepts_work())
                .count(),
        )
        .map_err(|_| ExecutionServiceError::Failed("browser worker count exceeds u32".into()))
    }

    fn future_for_value(
        &self,
        value: Value,
    ) -> Result<
        Result<runmat_runtime::execution::FutureExecutionHandle, Value>,
        ExecutionServiceError,
    > {
        match value {
            Value::Future(handle) => Ok(Ok(handle.into())),
            Value::Task(handle) => {
                self.validate_scope(handle.scope_id)?;
                let state = self.state.borrow();
                let task = state
                    .tasks
                    .get(&handle.id)
                    .ok_or(ExecutionServiceError::UnknownHandle)?;
                if task.generation != handle.generation {
                    return Err(ExecutionServiceError::UnknownHandle);
                }
                Ok(Ok(
                    runmat_runtime::execution::FutureExecutionHandle::from_task(
                        handle,
                        task.future_id,
                    ),
                ))
            }
            value => Ok(Err(value)),
        }
    }

    fn dispatch(&self, actions: Vec<DriverAction>) {
        for action in actions {
            match action {
                DriverAction::Launch(request) => self.launch(request),
                DriverAction::Cancel(request) | DriverAction::Terminate(request) => {
                    let _ = self.cancel_attempt(&request);
                }
                DriverAction::ResizePool { .. }
                | DriverAction::Checkpoint
                | DriverAction::GarbageCollectResults { .. } => {}
            }
        }
        self.refresh_driver_timer();
    }

    fn launch(&self, attempt: AttemptRequest) {
        let eligible = self
            .state
            .borrow()
            .driver
            .snapshot_ref()
            .attempts
            .get(&attempt.id)
            .is_some_and(|record| {
                record.state == runmat_execution::state::AttemptState::Assigned
                    && record.terminal.is_none()
            });
        if !eligible {
            return;
        }
        // Admission may have waited behind other work or a browser task turn.
        // Apply real deadlines before acquiring any new physical worker.
        let actions = {
            let mut state = self.state.borrow_mut();
            let now_millis = self.clock.now_millis().max(state.driver.now_millis());
            state
                .driver
                .handle(DriverCommand::Tick { now_millis })
                .unwrap_or_default()
        };
        self.dispatch(actions);
        let cancelled_before_launch = self
            .state
            .borrow()
            .driver
            .snapshot_ref()
            .attempts
            .get(&attempt.id)
            .is_some_and(|record| record.cancellation_requested_at.is_some());
        if cancelled_before_launch {
            self.retire_unlaunched_attempt(&attempt);
            return;
        }
        let program = {
            let mut state = self.state.borrow_mut();
            let started = BackendReport::for_request(&attempt, AttemptReport::Started);
            let actions = state
                .driver
                .handle(DriverCommand::BackendReport(started))
                .unwrap_or_default();
            drop(state);
            self.dispatch(actions);
            self.state.borrow().requests.get(&attempt.task_id).cloned()
        };
        let Some(mut program) = program else {
            self.finish_attempt(
                attempt,
                Err("browser scheduler lost the exact program request".into()),
                1,
            );
            return;
        };
        let assignment = runmat_execution::ProgramExecutionAssignment {
            scope_id: attempt.scope_id,
            pool_id: attempt.task.pool_id,
            task_id: attempt.task_id,
            attempt_id: attempt.id,
            worker_id: attempt.worker_id,
            backend: self.pool_backend(),
            resources: attempt.resource_assignment.clone(),
        };
        let selected_assignment = assignment.clone();
        match &mut program {
            BrowserWorkerRequest::Direct(request) => request.assignment = Some(assignment),
            BrowserWorkerRequest::Graph(request) => request.assignment = Some(assignment),
            BrowserWorkerRequest::OwnedBuiltin(request) => request.assignment = Some(assignment),
        }
        let callback = {
            let mut state = self.state.borrow_mut();
            state.tasks.get_mut(&attempt.task_id).is_some_and(|task| {
                if self.capabilities.selected_program_work
                    && !matches!(&program, BrowserWorkerRequest::OwnedBuiltin(_))
                {
                    // Public Direct/Graph Futures keep their existing shapes
                    // and refetch policy, but use the SAME retirement owner
                    // on a qualified host. No legacy unqualified slot reuse.
                    task.callback_exit
                        .get_or_insert_with(PhysicalCallback::default);
                }
                task.callback_exit.is_some()
            })
        };
        if matches!(&program, BrowserWorkerRequest::OwnedBuiltin(_)) {
            self.mark_owned_launched(&attempt, selected_assignment);
        } else if callback {
            self.mark_callback_launched(&attempt, selected_assignment);
        }
        let weak = self.self_weak.borrow().clone();
        let retained_service = (callback
            || matches!(
                &program,
                BrowserWorkerRequest::OwnedBuiltin(_) | BrowserWorkerRequest::Graph(_)
            ))
        .then(|| weak.upgrade())
        .flatten();
        let host = self.host.clone();
        let started = runmat_time::Instant::now();
        wasm_bindgen_futures::spawn_local(async move {
            if callback {
                let terminal = match host.as_ref() {
                    Some(host) => {
                        host.launch_program(
                            &attempt.task_id.to_string(),
                            &attempt.worker_id.to_string(),
                            &program,
                        )
                        .await
                    }
                    None => Err("selected callback lost its isolated program host".into()),
                };
                if let Some(service) = retained_service.or_else(|| weak.upgrade()) {
                    let wall_millis = timer::duration_millis(started.elapsed()).max(1);
                    service.finish_program_attempt(attempt, terminal, wall_millis);
                }
                return;
            }
            let response = match host {
                Some(host) => {
                    host.launch(
                        &attempt.task_id.to_string(),
                        &attempt.worker_id.to_string(),
                        &program,
                    )
                    .await
                }
                None => match program {
                    // Complete V4 units run in a portable session that retains
                    // this service for nested execution.
                    BrowserWorkerRequest::Direct(request)
                        if request.artifact.form
                            == runmat_execution_artifact::ExecutableForm::ExecutableUnitV4 =>
                    {
                        match weak.upgrade() {
                            Some(service) => {
                                Ok(super::program::execute(request, Some(service)).await)
                            }
                            None => {
                                Err("browser execution authority ended before invocation".into())
                            }
                        }
                    }
                    program => Ok(super::admission::execute_local_worker_program(program).await),
                },
            };
            if let Some(service) = retained_service.or_else(|| weak.upgrade()) {
                let wall_millis = timer::duration_millis(started.elapsed()).max(1);
                service.finish_attempt(attempt, response, wall_millis);
            }
        });
    }

    fn finish_attempt(
        &self,
        attempt: AttemptRequest,
        response: Result<ProgramExecutionResponse, String>,
        wall_millis: u64,
    ) {
        if self.owned.borrow().contains_key(&attempt.task_id) {
            self.finish_owned_attempt(attempt, response, wall_millis);
            return;
        }
        if response.is_err() && self.callback_needs_exit(attempt.task_id) {
            self.callback_launch_failed(attempt, response.expect_err("failed callback launch"));
            return;
        }
        let unverified_host_launch = response.is_err() && self.host.is_some();
        self.advance_driver_now()
            .unwrap_or_else(|error| self.deadline_timer_failed(error));
        let (selected_request, graph_receipt) = {
            let state = self.state.borrow();
            let selected_request = state.requests.get(&attempt.task_id).cloned();
            let receipt = state
                .tasks
                .get(&attempt.task_id)
                .and_then(|task| state.futures.get(&task.future_id))
                .and_then(|future| match future {
                    FutureState::Scheduled { receipt, .. } => receipt.clone(),
                    FutureState::CancellingCallback { receipt, .. }
                    | FutureState::UnsettledCallback { receipt, .. } => receipt.clone(),
                    _ => None,
                });
            (selected_request, receipt)
        };
        let response = response.and_then(|response| {
            match selected_request.as_ref() {
                Some(BrowserWorkerRequest::Direct(request)) => response.validate_against(request),
                Some(BrowserWorkerRequest::Graph(request)) => request.validate_response(&response),
                Some(BrowserWorkerRequest::OwnedBuiltin(request)) => {
                    response.validate_owned_builtin(request)
                }
                None => return Err("browser scheduler lost the exact program request".into()),
            }
            .map_err(|error| error.to_string())?;
            Ok(response)
        });
        let (future_id, mut completion, mut report) = match response {
            Ok(ProgramExecutionResponse::Success { outputs }) => {
                let decoded = outputs
                    .iter()
                    .map(runmat_runtime::execution::value_codec::decode_inline_value)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|error| ExecutionServiceError::Failed(error.to_string()))
                    .and_then(|values| {
                        runmat_value::ValueSequence::comma_separated(values)
                            .map_err(|error| ExecutionServiceError::Failed(error.to_string()))
                    })
                    .and_then(runmat_runtime::execution::RootedValueSequence::new);
                let report =
                    match AttemptSuccess::new(&attempt, outputs, Vec::new(), wall_millis, 0, 0) {
                        Ok(result) => AttemptReport::Succeeded { result },
                        Err(message) => AttemptReport::Failed {
                            kind: AttemptFailureKind::Rejected,
                            message,
                        },
                    };
                (
                    self.future_id(attempt.task_id),
                    FutureState::Completed(decoded),
                    report,
                )
            }
            Ok(ProgramExecutionResponse::GraphSuccess { graph }) => {
                if let Some(receipt) = graph_receipt {
                    let report =
                        match AttemptSuccess::new_graph(&attempt, graph.clone(), wall_millis, 0, 0)
                        {
                            Ok(result) => AttemptReport::Succeeded { result },
                            Err(message) => AttemptReport::Failed {
                                kind: AttemptFailureKind::Rejected,
                                message,
                            },
                        };
                    (
                        self.future_id(attempt.task_id),
                        FutureState::GraphCompleted {
                            graph: Rc::new(graph),
                            receipt,
                        },
                        report,
                    )
                } else {
                    let message =
                        "graph worker response has no selected result receipt".to_string();
                    let result = Err(ExecutionServiceError::Infrastructure(message.clone()));
                    let report = AttemptReport::Failed {
                        kind: AttemptFailureKind::Infrastructure,
                        message,
                    };
                    (
                        self.future_id(attempt.task_id),
                        FutureState::Completed(result),
                        report,
                    )
                }
            }
            Ok(ProgramExecutionResponse::ExternalizedSuccess { .. }) => {
                let message =
                    "browser execution host cannot consume native externalized results".to_string();
                let result = Err(ExecutionServiceError::Failed(message.clone()));
                let report = AttemptReport::Failed {
                    kind: AttemptFailureKind::Rejected,
                    message,
                };
                (
                    self.future_id(attempt.task_id),
                    FutureState::Completed(result),
                    report,
                )
            }
            Ok(ProgramExecutionResponse::SpmdSuccess { .. }) => {
                let message =
                    "SPMD worker response requires the typed browser gang driver".to_string();
                let result = Err(ExecutionServiceError::Failed(message.clone()));
                let report = AttemptReport::Failed {
                    kind: AttemptFailureKind::Rejected,
                    message,
                };
                (
                    self.future_id(attempt.task_id),
                    FutureState::Completed(result),
                    report,
                )
            }
            Ok(ProgramExecutionResponse::Failure { message }) => {
                let result = Err(ExecutionServiceError::Failed(message.clone()));
                let report = AttemptReport::Failed {
                    kind: AttemptFailureKind::Execution,
                    message,
                };
                (
                    self.future_id(attempt.task_id),
                    FutureState::Completed(result),
                    report,
                )
            }
            Ok(ProgramExecutionResponse::RuntimeFailure { failure }) => {
                let report = AttemptReport::RuntimeFailed {
                    failure: failure.clone(),
                };
                let state = match graph_receipt {
                    Some(receipt) => FutureState::GraphFailed {
                        failure: Rc::new(failure),
                        receipt,
                    },
                    None => FutureState::Completed(Err(ExecutionServiceError::RuntimeFailure(
                        Box::new(failure),
                    ))),
                };
                (self.future_id(attempt.task_id), state, report)
            }
            Err(message) => {
                let result = Err(ExecutionServiceError::Failed(message.clone()));
                let report = AttemptReport::Failed {
                    kind: AttemptFailureKind::Infrastructure,
                    message,
                };
                (
                    self.future_id(attempt.task_id),
                    FutureState::Completed(result),
                    report,
                )
            }
        };
        let mut unsuccessful_resources = None;
        if unverified_host_launch {
            if let AttemptReport::Failed { message, .. } = report {
                report = AttemptReport::RetirementUnverified { message };
            }
        }
        if let AttemptReport::Succeeded { result } = &report {
            if let Err(message) = result.validate_against(&attempt) {
                unsuccessful_resources = result
                    .resources()
                    .filter(|resources| resources.validate_against(&attempt).is_ok())
                    .cloned()
                    .map(Box::new);
                report = AttemptReport::Failed {
                    kind: AttemptFailureKind::Rejected,
                    message,
                };
            }
        }
        if let AttemptReport::Failed { message, .. } = &report {
            if matches!(
                &completion,
                FutureState::Completed(Ok(_)) | FutureState::GraphCompleted { .. }
            ) {
                completion = FutureState::Completed(Err(ExecutionServiceError::Infrastructure(
                    message.clone(),
                )));
            }
        }
        let certified = self
            .state
            .borrow()
            .tasks
            .get(&attempt.task_id)
            .and_then(|task| task.callback_exit.as_ref())
            .is_some_and(PhysicalCallback::retirement_certified);
        let mut backend_report = BackendReport::for_request(&attempt, report);
        backend_report.unsuccessful_resources = unsuccessful_resources;
        let actions = self.apply_browser_report(backend_report, certified);
        let (task_state, unresolved) = {
            let state = self.state.borrow();
            let task_state = state
                .driver
                .task_record(attempt.task_id)
                .ok()
                .map(|task| task.state);
            let unresolved = state
                .driver
                .snapshot_ref()
                .attempts
                .get(&attempt.id)
                .and_then(|record| record.unverified_retirement.as_ref())
                .is_some_and(|evidence| evidence.retired_at_driver_millis.is_none());
            (task_state, unresolved)
        };
        if unresolved {
            if let Some(owner) = self.self_weak.borrow().upgrade() {
                // Legacy program hosts cannot prove a rejected or cancelled
                // invocation retired. Reuse the SAME physical retention table;
                // no replacement source/claim owner is invented.
                physical_exit::retain_until_physical_exit(attempt.task_id, owner);
            }
        }
        if task_state == Some(runmat_execution::state::TaskState::Cancelled) {
            completion = FutureState::Cancelled;
        } else if task_state != Some(runmat_execution::state::TaskState::Succeeded) {
            let error = ExecutionServiceError::WorkerLost(
                "browser operation has no accepted successful scheduler outcome".into(),
            );
            completion = match completion {
                FutureState::Completed(Ok(_)) => FutureState::UnsettledCallback {
                    task_id: attempt.task_id,
                    receipt: None,
                    error,
                },
                FutureState::GraphCompleted { receipt, .. } => FutureState::UnsettledCallback {
                    task_id: attempt.task_id,
                    receipt: Some(receipt),
                    error,
                },
                other => other,
            };
        }
        let mut state = self.state.borrow_mut();
        let mut retired = None;
        if let Some(future_id) = future_id {
            if !matches!(state.futures.get(&future_id), Some(FutureState::Cancelled))
                || task_state == Some(runmat_execution::state::TaskState::Indeterminate)
            {
                retired = state.futures.insert(future_id, completion);
                record_browser_completion(&mut state, attempt.task_id);
            }
        }
        if !unresolved {
            state.requests.remove(&attempt.task_id);
        }
        drop(state);
        drop(retired);
        self.wake_callback(attempt.task_id);
        self.retire_released_callback(attempt.task_id);
        self.dispatch(actions);
    }

    fn future_id(&self, task_id: TaskId) -> Option<FutureId> {
        self.state
            .borrow()
            .tasks
            .get(&task_id)
            .map(|task| task.future_id)
    }

    fn cancel_attempt(&self, attempt: &AttemptRequest) -> Result<(), ExecutionServiceError> {
        if self.owned.borrow().contains_key(&attempt.task_id) {
            return self.cancel_owned_attempt(attempt);
        }
        if self
            .state
            .borrow()
            .tasks
            .get(&attempt.task_id)
            .is_some_and(|t| t.callback_exit.is_some())
        {
            return self.cancel_callback_physical(attempt.task_id);
        }
        if let Some(host) = &self.host {
            host.cancel(&attempt.task_id.to_string())
                .map_err(ExecutionServiceError::Failed)?;
        }
        Ok(())
    }
}

impl RuntimeExecutionServices for BrowserExecutionService {
    fn cpu_cooperation_policy(&self) -> CpuCooperationPolicy {
        // Work units are charged by CPU kernels at their existing bounded
        // checkpoints. This is a scheduling quantum, not a placement estimate.
        CpuCooperationPolicy::WorkQuantum(std::num::NonZeroU64::new(16_384).unwrap())
    }

    fn cooperate_cpu(
        &self,
    ) -> runmat_runtime::context::RuntimeServiceFuture<Result<(), ExecutionServiceError>> {
        Box::pin(async { super::task_turn::BrowserTaskTurn::new()?.await })
    }

    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        let state = self.state.try_borrow().ok()?;
        for future in state.futures.values() {
            match future {
                FutureState::Deferred(call) | FutureState::Materializing { call, .. } => {
                    call.invocation.visit_retained_values(visitor)?;
                }
                FutureState::Completed(Ok(sequence))
                | FutureState::GraphResolved {
                    result: Ok(sequence),
                    ..
                } => {
                    for value in sequence.value().iter() {
                        visitor(value)?;
                    }
                }
                // Caller execution owns its moved invocation; scheduled worker
                // requests contain serialized payloads, not local Value handles.
                FutureState::ExecutingInCaller
                | FutureState::Scheduled { .. }
                | FutureState::CancellingCallback { .. }
                | FutureState::UnsettledCallback { .. }
                | FutureState::GraphCompleted { .. }
                | FutureState::GraphFailed { .. }
                | FutureState::GraphResolved { result: Err(_), .. }
                | FutureState::Completed(Err(_))
                | FutureState::Cancelled => {}
            }
        }
        Some(())
    }

    fn scope_id(&self) -> ExecutionScopeId {
        self.scope_id
    }

    fn current_pool(&self) -> Result<Option<PoolSnapshot>, ExecutionServiceError> {
        let state = self.state.borrow();
        if !state.pool_open {
            return Ok(None);
        }
        let workers = Self::active_workers(&state, self.pool_id)?;
        Ok(Some(self.pool_snapshot(state.pool_generation, workers)))
    }

    fn ensure_pool(&self, request: PoolRequest) -> Result<PoolSnapshot, ExecutionServiceError> {
        let backend = self.pool_backend();
        if request
            .backend
            .is_some_and(|requested| requested != backend)
        {
            return Err(ExecutionServiceError::Failed(format!(
                "this browser session supports the {backend:?} pool backend"
            )));
        }
        if request.workers.is_none() {
            let state = self.state.borrow();
            if state.pool_open {
                let workers = Self::active_workers(&state, self.pool_id)?;
                return Ok(self.pool_snapshot(state.pool_generation, workers));
            }
        }
        let workers = request.workers.unwrap_or(self.capabilities.max_workers);
        if workers == 0 || workers > self.capabilities.max_workers {
            return Err(ExecutionServiceError::Failed(format!(
                "browser pool worker count must be between 1 and {}",
                self.capabilities.max_workers
            )));
        }
        let mut state = self.state.borrow_mut();
        let previous_workers = if state.pool_open {
            Some(Self::active_workers(&state, self.pool_id)?)
        } else {
            None
        };
        let actions = state
            .driver
            .resize_registered_pool(self.pool_id, workers)
            .map_err(driver_error)?;
        if previous_workers.is_some_and(|previous| previous != workers) {
            state.pool_generation = state.pool_generation.wrapping_add(1);
        }
        state.pool_open = true;
        let generation = state.pool_generation;
        drop(state);
        self.dispatch(actions);
        Ok(self.pool_snapshot(generation, workers))
    }

    fn close_pool(&self, pool: &PoolHandle) -> Result<(), ExecutionServiceError> {
        self.validate_scope(pool.scope_id)?;
        let mut state = self.state.borrow_mut();
        if !state.pool_open || pool.id != self.pool_id || pool.generation != state.pool_generation {
            return Err(ExecutionServiceError::UnknownHandle);
        }
        state.pool_open = false;
        state.pool_generation = state.pool_generation.wrapping_add(1);
        drop(state);
        self.drain_scope(CancellationReason::User);
        Ok(())
    }

    fn requires_program_capture(&self) -> bool {
        true
    }

    fn supports_callback_batches(&self) -> bool {
        self.host.is_some()
            && self.capabilities.selected_program_work
            && self.capabilities.has_worker_isolation()
    }
    fn submit_callback_batch(
        &self,
        calls: Vec<DeferredCall>,
        pool: &PoolHandle,
        cancellation: runmat_execution::cancellation::CancellationToken,
    ) -> Result<Vec<TaskHandle>, ExecutionServiceError> {
        self.submit_selected_callbacks(calls, pool, cancellation)
    }
    fn release_callback_batch(&self, tasks: &[TaskHandle]) -> Result<(), ExecutionServiceError> {
        self.release_callbacks(tasks)
    }
    fn wait_pending<'a>(
        &'a self,
        value: &Value,
        runtime: &'a runmat_runtime::context::RuntimeContext,
    ) -> Result<
        Option<
            runmat_runtime::execution::AdmittedExecutionSupervisor<'a, (), ExecutionServiceError>,
        >,
        ExecutionServiceError,
    > {
        let waiting = (|| {
            let future = self
                .future_for_value(value.clone())?
                .map_err(|_| ExecutionServiceError::UnknownHandle)?;
            self.validate_scope(future.scope_id)?;
            let state = self.state.borrow();
            let task_id = match state.futures.get(&future.id) {
                Some(
                    FutureState::Scheduled { task_id, .. }
                    | FutureState::CancellingCallback { task_id, .. },
                ) => *task_id,
                Some(_) => return Ok(None),
                None => return Err(ExecutionServiceError::UnknownHandle),
            };
            self.callback_waiters
                .borrow()
                .get(&task_id)
                .map(|wake| Some(wake.receiver.clone()))
                .ok_or_else(|| {
                    ExecutionServiceError::Infrastructure(
                        "scheduled browser task has no completion event".into(),
                    )
                })
        })();
        Ok(Some(
            runmat_runtime::execution::admit_execution_supervisor_with(
                runtime,
                move || async move {
                    if let Some(waiting) = waiting? {
                        waiting.await?;
                    }
                    Ok(())
                },
            )?,
        ))
    }
    fn create_future(&self, call: DeferredCall) -> Result<FutureHandle, ExecutionServiceError> {
        let requested_outputs = u16::try_from(call.invocation.requested_outputs())
            .map_err(|_| ExecutionServiceError::InvalidOutputContract)?;
        let mut state = self.state.borrow_mut();
        let sequence = state.next_future;
        state.next_future = sequence.wrapping_add(1);
        let id = FutureId::derive(&[self.scope_id.bytes(), &sequence.to_be_bytes()]);
        state
            .futures
            .insert(id, FutureState::Deferred(Rc::new(call)));
        Ok(FutureHandle {
            id,
            scope_id: self.scope_id,
            outputs: OutputContract {
                requested_outputs,
                owned_work: None,
            },
        })
    }

    fn spawn(&self, future: &FutureHandle) -> Result<TaskHandle, ExecutionServiceError> {
        let (reservation_id, call) = self.reserve_worker_materialization(future)?;
        let mut guard = MaterializationGuard {
            service: self,
            future_id: future.id,
            id: reservation_id,
            armed: true,
        };
        // This can traverse managed values and call source/GC observers. It
        // must never run under the service's RefCell mutable borrow.
        let prepared = program_preparation::PreparedProgram::prepare(
            &call,
            future.outputs.clone(),
            self.capabilities,
        )?;
        self.advance_driver_now()?;
        prepared.admit_registered_worker(self)?;
        let mut state = self.state.borrow_mut();
        match state.futures.get(&future.id) {
            Some(FutureState::Materializing {
                id,
                cancelled: false,
                ..
            }) if *id == reservation_id => {}
            Some(FutureState::Materializing {
                id,
                cancelled: true,
                ..
            }) if *id == reservation_id => return Err(ExecutionServiceError::Cancelled),
            _ => return Err(ExecutionServiceError::UnknownHandle),
        }
        let sequence = state.next_task;
        state.next_task = sequence.wrapping_add(1);
        let task_id = TaskId::derive(&[self.scope_id.bytes(), &sequence.to_be_bytes()]);
        let task_scope_id =
            ExecutionScopeId::derive(&[self.scope_id.bytes(), task_id.bytes().as_slice()]);
        state
            .driver
            .handle(DriverCommand::RegisterScope {
                scope_id: task_scope_id,
                parent: Some(self.scope_id),
            })
            .map_err(driver_error)?;
        let (submission, request, receipt) = prepared.into_submission(
            task_id,
            task_scope_id,
            self.pool_id,
            future.outputs.clone(),
            call.retry,
        );
        let generation = 1;
        let actions = match state
            .driver
            .handle(DriverCommand::Submit(Box::new(submission)))
        {
            Ok(actions) => actions,
            Err(error) => {
                let _ = state.driver.handle(DriverCommand::CancelScope {
                    scope_id: task_scope_id,
                    reason: CancellationReason::DependencyFailed,
                    now_millis: self.clock.now_millis(),
                });
                return Err(driver_error(error));
            }
        };
        state.requests.insert(task_id, request);
        state.tasks.insert(
            task_id,
            TaskRecord {
                future_id: future.id,
                generation,
                scope_id: task_scope_id,
                read: false,
                claimed: false,
                completion_order: None,
                retention: runmat_runtime::execution::TaskRetention::Refetchable,
                callback_exit: None,
            },
        );
        let retired = state
            .futures
            .insert(future.id, FutureState::Scheduled { task_id, receipt });
        drop(state);
        drop(retired);
        guard.armed = false;
        self.callback_waiters
            .borrow_mut()
            .insert(task_id, worker_batch::TaskWake::new());
        self.dispatch(actions);
        Ok(TaskHandle {
            id: task_id,
            scope_id: self.scope_id,
            generation,
            outputs: future.outputs.clone(),
        })
    }

    fn inspect_future(
        &self,
        future: &FutureHandle,
    ) -> Result<ExecutionHandleSnapshot, ExecutionServiceError> {
        self.validate_scope(future.scope_id)?;
        let state = self.state.borrow();
        let stored_state = state
            .futures
            .get(&future.id)
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        Ok(ExecutionHandleSnapshot {
            state: browser_future_state(stored_state),
            outputs: future.outputs.clone(),
            read: state
                .tasks
                .values()
                .find(|task| task.future_id == future.id)
                .is_some_and(|task| task.read),
        })
    }

    fn inspect_task(
        &self,
        task: &TaskHandle,
    ) -> Result<ExecutionHandleSnapshot, ExecutionServiceError> {
        self.validate_scope(task.scope_id)?;
        let state = self.state.borrow();
        let record = state
            .tasks
            .get(&task.id)
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        if record.generation != task.generation {
            return Err(ExecutionServiceError::UnknownHandle);
        }
        let state = state
            .futures
            .get(&record.future_id)
            .map(browser_future_state)
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        Ok(ExecutionHandleSnapshot {
            state,
            outputs: task.outputs.clone(),
            read: record.read,
        })
    }

    fn claim_next_task_result(
        &self,
        tasks: &[runmat_value::SharedTaskHandle],
    ) -> Result<TaskResultClaim, ExecutionServiceError> {
        let mut state = self.state.borrow_mut();
        let mut selected: Option<(u64, usize, TaskId)> = None;
        let mut has_unread = false;
        for (index, task) in tasks.iter().enumerate() {
            self.validate_scope(task.scope_id)?;
            let record = state
                .tasks
                .get(&task.id)
                .ok_or(ExecutionServiceError::UnknownHandle)?;
            if record.generation != task.generation {
                return Err(ExecutionServiceError::UnknownHandle);
            }
            if record.read {
                continue;
            }
            has_unread = true;
            if record.claimed {
                continue;
            }
            if let Some(order) = record.completion_order {
                let candidate = (order, index, task.id);
                if selected.is_none_or(|current| candidate < current) {
                    selected = Some(candidate);
                }
            }
        }
        let Some((_, index, task_id)) = selected else {
            return Ok(if has_unread {
                TaskResultClaim::Pending
            } else {
                TaskResultClaim::Exhausted
            });
        };
        state
            .tasks
            .get_mut(&task_id)
            .expect("task identity was validated above")
            .claimed = true;
        Ok(TaskResultClaim::Claimed { index })
    }

    fn mark_task_result_read(&self, task: &TaskHandle) -> Result<(), ExecutionServiceError> {
        self.validate_scope(task.scope_id)?;
        let mut state = self.state.borrow_mut();
        let record = state
            .tasks
            .get_mut(&task.id)
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        if record.generation != task.generation {
            return Err(ExecutionServiceError::UnknownHandle);
        }
        record.claimed = false;
        record.read = true;
        Ok(())
    }

    fn begin_await(&self, value: Value) -> Result<AwaitAction, ExecutionServiceError> {
        let original = value.clone();
        let future = match self.future_for_value(value)? {
            Ok(future) => future,
            Err(value) => return Ok(AwaitAction::Passthrough(value)),
        };
        self.validate_scope(future.scope_id)?;
        let mut state = self.state.borrow_mut();
        let record = state
            .futures
            .get_mut(&future.id)
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        match record {
            FutureState::Deferred(_) => {
                let FutureState::Deferred(call) =
                    std::mem::replace(record, FutureState::ExecutingInCaller)
                else {
                    unreachable!("deferred state matched while holding its owner borrow")
                };
                // Move the original call when this future is its only owner.
                let call = Rc::try_unwrap(call).unwrap_or_else(|shared| (*shared).clone());
                Ok(AwaitAction::ExecuteFuture {
                    handle: future,
                    call: Box::new(call),
                })
            }
            FutureState::Materializing {
                cancelled: true, ..
            } => Err(ExecutionServiceError::Cancelled),
            FutureState::Scheduled { .. }
            | FutureState::CancellingCallback { .. }
            | FutureState::Materializing { .. } => Ok(AwaitAction::Pending(original)),
            FutureState::UnsettledCallback { error, .. } => Err(error.clone()),
            FutureState::ExecutingInCaller => Err(ExecutionServiceError::Failed(
                "future is already executing in its caller".into(),
            )),
            FutureState::Completed(result) => {
                let result = result.clone();
                result.map(AwaitAction::Completed)
            }
            FutureState::GraphCompleted { .. }
            | FutureState::GraphFailed { .. }
            | FutureState::GraphResolved { .. } => Err(ExecutionServiceError::Infrastructure(
                "graph result requires an explicit await context".into(),
            )),
            FutureState::Cancelled => Err(ExecutionServiceError::Cancelled),
        }
    }

    fn begin_await_in_context(
        &self,
        value: Value,
        runtime: &runmat_runtime::context::RuntimeContext,
    ) -> Result<AwaitAction, ExecutionServiceError> {
        let future = match self.future_for_value(value.clone())? {
            Ok(future) => future,
            Err(value) => return Ok(AwaitAction::Passthrough(value)),
        };
        self.validate_scope(future.scope_id)?;
        let (graph, receipt) = {
            let state = self.state.borrow();
            if let Some(FutureState::UnsettledCallback {
                error: ExecutionServiceError::RuntimeFailure(failure),
                receipt: Some(receipt),
                ..
            }) = state.futures.get(&future.id)
            {
                let failure = failure.clone();
                let receipt = receipt.clone();
                drop(state);
                receipt.check_issuing_context(runtime)?;
                // Preserve the issuing source while the same request/grant
                // remains quarantined; decoding an error is not retirement.
                return Err(receipt
                    .decode_failure_in_context(runtime, &failure)
                    .map(|error| ExecutionServiceError::LocalizedRuntimeFailure(error.into()))
                    .unwrap_or_else(|error| error));
            }
            if let Some(FutureState::GraphFailed { failure, receipt }) =
                state.futures.get(&future.id)
            {
                let failure = Rc::clone(failure);
                let receipt = Rc::clone(receipt);
                drop(state);
                receipt.check_issuing_context(runtime)?;
                let localized = receipt
                    .decode_failure_in_context(runtime, &failure)
                    .map(|error| ExecutionServiceError::LocalizedRuntimeFailure(error.into()))
                    .unwrap_or_else(|error| error);
                if matches!(&localized, ExecutionServiceError::ForeignScope) {
                    return Err(localized);
                }
                let mut state = self.state.borrow_mut();
                match state.futures.get(&future.id) {
                    Some(FutureState::GraphFailed {
                        failure: current_failure,
                        receipt: current_receipt,
                    }) if Rc::ptr_eq(current_failure, &failure)
                        && Rc::ptr_eq(current_receipt, &receipt) =>
                    {
                        let retired = state.futures.insert(
                            future.id,
                            FutureState::GraphResolved {
                                result: Err(localized.clone()),
                                receipt,
                            },
                        );
                        drop(state);
                        drop(retired);
                        return Err(localized);
                    }
                    Some(FutureState::Cancelled) => return Err(ExecutionServiceError::Cancelled),
                    _ => return Err(ExecutionServiceError::UnknownHandle),
                }
            }
            if let Some(FutureState::GraphResolved { result, receipt }) =
                state.futures.get(&future.id)
            {
                let result = result.clone();
                let receipt = Rc::clone(receipt);
                drop(state);
                receipt.check_issuing_context(runtime)?;
                return result.map(AwaitAction::Completed);
            }
            let Some(FutureState::GraphCompleted { graph, receipt }) =
                state.futures.get(&future.id)
            else {
                drop(state);
                return self.begin_await(value);
            };
            let graph = Rc::clone(graph);
            let receipt = Rc::clone(receipt);
            (graph, receipt)
        };
        receipt.check_issuing_context(runtime)?;
        {
            let mut state = self.state.borrow_mut();
            if !matches!(
                state.futures.get(&future.id),
                Some(FutureState::GraphCompleted { graph: current_graph, receipt: current_receipt })
                    if Rc::ptr_eq(current_graph, &graph) && Rc::ptr_eq(current_receipt, &receipt)
            ) {
                return Err(ExecutionServiceError::UnknownHandle);
            }
            state
                .futures
                .insert(future.id, FutureState::ExecutingInCaller);
        }
        let mut guard = GraphDecodeGuard {
            service: self,
            future_id: future.id,
            graph,
            receipt,
            armed: true,
        };
        let result = guard.receipt.decode_in_context(runtime, &guard.graph);
        if matches!(&result, Err(ExecutionServiceError::ForeignScope)) {
            // Dropping the armed guard restores GraphCompleted; another
            // session cannot consume the issuing session's result receipt.
            return Err(ExecutionServiceError::ForeignScope);
        }
        let mut state = self.state.borrow_mut();
        match state.futures.get(&future.id) {
            Some(FutureState::Cancelled) => Err(ExecutionServiceError::Cancelled),
            Some(FutureState::ExecutingInCaller) => {
                state.futures.insert(
                    future.id,
                    FutureState::GraphResolved {
                        result: result.clone(),
                        receipt: Rc::clone(&guard.receipt),
                    },
                );
                guard.armed = false;
                result.map(AwaitAction::Completed)
            }
            _ => Err(ExecutionServiceError::UnknownHandle),
        }
    }

    fn complete_future(
        &self,
        future: runmat_runtime::execution::FutureCompletionRef<'_>,
        result: Result<runmat_runtime::execution::RootedValueSequence, ExecutionServiceError>,
    ) -> Result<(), ExecutionServiceError> {
        self.validate_scope(future.scope_id)?;
        let mut state = self.state.borrow_mut();
        let record = state
            .futures
            .get_mut(&future.id)
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        match record {
            FutureState::ExecutingInCaller => {
                *record = FutureState::Completed(result);
                if let Some(task_id) = state
                    .tasks
                    .iter()
                    .find_map(|(task_id, task)| (task.future_id == future.id).then_some(*task_id))
                {
                    record_browser_completion(&mut state, task_id);
                }
                Ok(())
            }
            FutureState::Cancelled => Err(ExecutionServiceError::Cancelled),
            _ => Err(ExecutionServiceError::UnknownHandle),
        }
    }

    fn cancel(
        &self,
        value: &Value,
        reason: CancellationReason,
    ) -> Result<(), ExecutionServiceError> {
        let (future_id, task_id, cancellation_scope) = match value {
            Value::Future(handle) => {
                self.validate_scope(handle.scope_id)?;
                let state = self.state.borrow();
                let scheduled_task = match state.futures.get(&handle.id) {
                    Some(
                        FutureState::Scheduled { task_id, .. }
                        | FutureState::CancellingCallback { task_id, .. }
                        | FutureState::UnsettledCallback { task_id, .. },
                    ) => Some(*task_id),
                    _ => None,
                };
                let task_scope = scheduled_task
                    .and_then(|task_id| state.tasks.get(&task_id))
                    .map(|task| task.scope_id);
                (handle.id, scheduled_task, task_scope)
            }
            Value::Task(handle) => {
                self.validate_scope(handle.scope_id)?;
                let state = self.state.borrow();
                let task = state
                    .tasks
                    .get(&handle.id)
                    .ok_or(ExecutionServiceError::UnknownHandle)?;
                if task.generation != handle.generation {
                    return Err(ExecutionServiceError::UnknownHandle);
                }
                (task.future_id, Some(handle.id), Some(task.scope_id))
            }
            _ => return Err(ExecutionServiceError::UnknownHandle),
        };
        let retired = {
            let mut state = self.state.borrow_mut();
            let protected = task_id
                .and_then(|id| state.tasks.get(&id))
                .is_some_and(|task| task.callback_exit.is_some());
            if matches!(
                state.futures.get(&future_id),
                Some(FutureState::UnsettledCallback { .. })
            ) {
                None
            } else {
                let next = match state.futures.get(&future_id) {
                    Some(
                        FutureState::Scheduled { task_id, receipt }
                        | FutureState::CancellingCallback { task_id, receipt },
                    ) if protected => FutureState::CancellingCallback {
                        task_id: *task_id,
                        receipt: receipt.clone(),
                    },
                    _ => FutureState::Cancelled,
                };
                let pending_retirement = matches!(next, FutureState::CancellingCallback { .. });
                let retired = state.futures.insert(future_id, next);
                if let Some(task_id) = task_id.filter(|_| !pending_retirement) {
                    record_browser_completion(&mut state, task_id);
                }
                retired
            }
        };
        drop(retired);
        let actions = if let Some(scope_id) = cancellation_scope {
            self.state
                .borrow_mut()
                .driver
                .handle(DriverCommand::CancelScope {
                    scope_id,
                    reason,
                    now_millis: self.clock.now_millis(),
                })
                .map_err(driver_error)?
        } else {
            Vec::new()
        };
        if let Some(task_id) = task_id {
            let retained = {
                let state = self.state.borrow();
                matches!(
                    state.futures.get(&future_id),
                    Some(
                        FutureState::CancellingCallback { .. }
                            | FutureState::UnsettledCallback { .. }
                    )
                ) || state
                    .driver
                    .task_record(task_id)
                    .is_ok_and(|task| task.active_attempt.is_some())
            };
            if !retained {
                self.state.borrow_mut().requests.remove(&task_id);
                self.wake_callback(task_id);
            } else if let Some(owner) = self.self_weak.borrow().upgrade() {
                physical_exit::retain_until_physical_exit(task_id, owner);
            }
        }
        self.settle_callback_cancellations();
        self.dispatch(actions);
        Ok(())
    }

    fn drain_scope(&self, reason: CancellationReason) {
        let actions = self
            .state
            .borrow_mut()
            .driver
            .handle(DriverCommand::CancelScope {
                scope_id: self.scope_id,
                reason,
                now_millis: self.clock.now_millis(),
            })
            .unwrap_or_default();
        let mut state = self.state.borrow_mut();
        let protected = state
            .tasks
            .values()
            .filter(|task| task.callback_exit.is_some())
            .map(|task| task.future_id)
            .collect::<std::collections::HashSet<_>>();
        for (future_id, future) in &mut state.futures {
            match future {
                FutureState::Completed(_)
                | FutureState::GraphCompleted { .. }
                | FutureState::GraphFailed { .. }
                | FutureState::GraphResolved { .. } => {}
                FutureState::Scheduled { task_id, receipt } if protected.contains(future_id) => {
                    *future = FutureState::CancellingCallback {
                        task_id: *task_id,
                        receipt: receipt.clone(),
                    }
                }
                FutureState::CancellingCallback { .. } => {}
                FutureState::UnsettledCallback { .. } => {}
                FutureState::Materializing { cancelled, .. } => *cancelled = true,
                _ => *future = FutureState::Cancelled,
            }
        }
        let task_ids = state.tasks.keys().copied().collect::<Vec<_>>();
        for task_id in task_ids {
            if !state
                .tasks
                .get(&task_id)
                .and_then(|t| state.futures.get(&t.future_id))
                .is_some_and(|f| matches!(f, FutureState::CancellingCallback { .. }))
            {
                record_browser_completion(&mut state, task_id);
            }
        }
        self.retain_pending_requests(&mut state);
        drop(state);
        self.settle_callback_cancellations();
        self.dispatch(actions);
    }
}

fn record_browser_completion(state: &mut State, task_id: TaskId) {
    if state
        .tasks
        .get(&task_id)
        .is_none_or(|task| task.completion_order.is_some())
    {
        return;
    }
    let order = state.next_completion;
    state.next_completion = order.wrapping_add(1);
    state
        .tasks
        .get_mut(&task_id)
        .expect("task identity was validated above")
        .completion_order = Some(order);
}

fn browser_future_state(state: &FutureState) -> ExecutionHandleState {
    match state {
        FutureState::Deferred(_) => ExecutionHandleState::Deferred,
        FutureState::Materializing {
            cancelled: true, ..
        } => ExecutionHandleState::Cancelled,
        FutureState::ExecutingInCaller
        | FutureState::Materializing { .. }
        | FutureState::Scheduled { .. }
        | FutureState::CancellingCallback { .. } => ExecutionHandleState::Running,
        FutureState::UnsettledCallback { .. } => ExecutionHandleState::Failed,
        FutureState::GraphCompleted { .. } => ExecutionHandleState::Finished,
        FutureState::GraphFailed { .. } => ExecutionHandleState::Failed,
        FutureState::GraphResolved { result: Ok(_), .. } => ExecutionHandleState::Finished,
        FutureState::GraphResolved { result: Err(_), .. } => ExecutionHandleState::Failed,
        FutureState::Completed(Ok(_)) => ExecutionHandleState::Finished,
        FutureState::Completed(Err(_)) => ExecutionHandleState::Failed,
        FutureState::Cancelled => ExecutionHandleState::Cancelled,
    }
}

#[cfg(test)]
mod tests {
    use wasm_bindgen_test::*;

    use runmat_execution::task::RetryPolicy;

    use super::*;

    wasm_bindgen_test_configure!(run_in_browser);

    fn deferred_test_call(program: Vec<u8>) -> DeferredCall {
        DeferredCall {
            spawn_safety: runmat_types::SpawnSafetyFact::RequiresIsolation,
            invocation: runmat_runtime::execution::DeferredInvocation::callable(
                runmat_runtime::call::descriptor::CallableDescriptor::resolved(
                    runmat_hir::CallableIdentity::BoundFunction(runmat_hir::FunctionId(0)),
                    Vec::new(),
                    1,
                    runmat_hir::CallableFallbackPolicy::None,
                    runmat_runtime::call::descriptor::CallableCallKind::Direct,
                ),
            ),
            retry: RetryPolicy::Never,
            program_revision: None,
            environment: runmat_execution::EnvironmentSnapshot::default(),
            console: None,
            capabilities: Default::default(),
            program: runmat_runtime::execution::DeferredProgram::InterpreterV2(program.into()),
        }
    }

    fn original_deferred_failure() -> DeferredCall {
        runmat_runtime::builtins::wasm_registry::register_all();
        let program =
            runmat_vm::encode_interpreter_program_v2(&runmat_vm::FunctionRegistry::default())
                .expect("encode the actual builtin worker registry");
        let mut call = deferred_test_call(program);
        let digest = runmat_execution::Digest::sha256(match &call.program {
            runmat_runtime::execution::DeferredProgram::InterpreterV2(bytes) => bytes.as_ref(),
            _ => unreachable!("the fixture encoded an interpreter program"),
        });
        call.program_revision = Some(
            runmat_execution::ProgramRevision::new(
                digest,
                digest,
                runmat_core::program_environment(runmat_core::CompatMode::RunMat),
            )
            .expect("bind the fixture to its actual browser host environment"),
        );
        call.invocation = runmat_runtime::execution::DeferredInvocation::callable(
            runmat_runtime::call::descriptor::CallableDescriptor::resolved(
                runmat_hir::CallableIdentity::Builtin(runmat_hir::BuiltinId("error".into())),
                vec![Value::String("original browser worker failure".into())],
                1,
                runmat_hir::CallableFallbackPolicy::None,
                runmat_runtime::call::descriptor::CallableCallKind::Direct,
            ),
        );
        call
    }

    #[wasm_bindgen_test(async)]
    async fn serial_fallback_schedules_through_the_portable_driver() {
        let service = BrowserExecutionService::new(None).unwrap();
        let runtime = runmat_runtime::context::RuntimeContext::new(service.clone());
        assert_eq!(service.capabilities.max_workers, 1);
        assert!(!service.capabilities.has_worker_isolation());
        assert!(
            service.state.borrow().driver.now_millis()
                >= service.clock.now_millis().saturating_sub(1_000)
        );
        let future = service.create_future(original_deferred_failure()).unwrap();
        let task = service.spawn(&future).unwrap();
        assert!(matches!(
            service
                .begin_await(Value::Task(task.clone().into()))
                .unwrap(),
            AwaitAction::Pending(_)
        ));
        let value = Value::Task(task.into());
        let first = service
            .wait_pending(&value, &runtime)
            .unwrap()
            .expect("actual task event");
        let second = service
            .wait_pending(&value, &runtime)
            .unwrap()
            .expect("shared task event");
        let (first, second) = futures::join!(first, second);
        first.unwrap();
        second.unwrap();
        assert!(service.begin_await(value.clone()).is_err());
        // A later fetch observes the stored outcome, not a consumed event.
        service
            .wait_pending(&value, &runtime)
            .unwrap()
            .unwrap()
            .await
            .unwrap();
        assert!(service.begin_await(value).is_err());
    }

    #[wasm_bindgen_test]
    fn partial_registry_rejects_before_scheduling_child_work() {
        let service = BrowserExecutionService::new(None).unwrap();
        let future = service
            .create_future(deferred_test_call(b"invalid registry".to_vec()))
            .unwrap();
        assert!(service.spawn(&future).is_err());
        assert!(service.state.borrow().requests.is_empty());
    }

    #[wasm_bindgen_test]
    fn cancelling_a_scheduled_future_cancels_its_child_task() {
        let service = BrowserExecutionService::new(None).unwrap();
        let future = service.create_future(original_deferred_failure()).unwrap();
        let task = service.spawn(&future).unwrap();
        service
            .cancel(
                &Value::Future(future.into()),
                runmat_execution::CancellationReason::User,
            )
            .unwrap();
        assert_eq!(
            service.begin_await(Value::Task(task.into())),
            Err(ExecutionServiceError::Cancelled)
        );
    }

    #[wasm_bindgen_test]
    fn browser_pool_lifecycle_reports_exact_topology_and_fences_stale_handles() {
        let service = BrowserExecutionService::new(None).unwrap();
        assert!(service.current_pool().unwrap().is_none());
        service.ensure_pool(PoolRequest::automatic()).unwrap();
        let first = service.current_pool().unwrap().expect("browser pool");
        assert_eq!(first.backend, PoolBackend::Serial);
        assert_eq!(first.workers, 1);
        service.close_pool(&first.handle).unwrap();
        assert!(service.current_pool().unwrap().is_none());
        let reopened = service.ensure_pool(PoolRequest::automatic()).unwrap();
        assert_ne!(first.handle.generation, reopened.handle.generation);
    }
}
