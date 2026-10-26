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

mod owned_builtin;
mod selected_program;
use selected_program::SelectedTask;
use owned_builtin::OwnedTask;

pub(crate) struct BrowserExecutionService {
    scope_id: ExecutionScopeId,
    pool_id: PoolId,
    host: Option<BrowserExecutionHost>,
    capabilities: BrowserExecutionCapabilities,
    state: RefCell<State>,
    owned: RefCell<HashMap<TaskId, OwnedTask>>,
    selected: RefCell<HashMap<TaskId, SelectedTask>>,
    self_weak: RefCell<Weak<Self>>,
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
        let capabilities = host
            .as_ref()
            .map(BrowserExecutionHost::capabilities)
            .unwrap_or_default();
        let session_nonce = uuid::Uuid::new_v4();
        let scope_id = ExecutionScopeId::derive(&[b"browser-session", session_nonce.as_bytes()]);
        let pool_id = PoolId::derive(&[scope_id.bytes(), b"browser"]);
        let selected_inventory = host.as_ref().and_then(BrowserExecutionHost::worker_inventory);
        let accelerators = match selected_inventory {
            Some(inventory) => inventory.accelerators.clone(),
            None => browser_accelerators()?,
        };
        let resources = browser_inventory(capabilities, accelerators.clone());
        let execution_host = match selected_inventory {
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
            // Browser workers share one process-local WebGPU device. A single
            // scheduler worker owns its physical allocation domain so the
            // pool cannot lease provider views of that device concurrently.
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
        let service = Rc::new(Self {
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
            selected: RefCell::new(HashMap::new()),
            self_weak: RefCell::new(Weak::new()),
        });
        service.self_weak.replace(Rc::downgrade(&service));
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
    }

    fn launch(&self, attempt: AttemptRequest) {
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
        if matches!(&program, BrowserWorkerRequest::OwnedBuiltin(_)) {
            self.mark_owned_launched(&attempt, selected_assignment);
        }
        if self.capabilities.selected_program_work && !matches!(&program, BrowserWorkerRequest::OwnedBuiltin(_)) {
            self.launch_selected_program(attempt, program);
            return;
        }
        let weak = self.self_weak.borrow().clone();
        let retained_service = matches!(&program, BrowserWorkerRequest::OwnedBuiltin(_))
            .then(|| weak.upgrade())
            .flatten();
        let host = self.host.clone();
        let started_millis = js_sys::Date::now();
        wasm_bindgen_futures::spawn_local(async move {
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
                let wall_millis = (js_sys::Date::now() - started_millis).max(1.0) as u64;
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
        let (selected_request, graph_receipt) = {
            let state = self.state.borrow();
            let selected_request = state.requests.get(&attempt.task_id).cloned();
            let receipt = state
                .tasks
                .get(&attempt.task_id)
                .and_then(|task| state.futures.get(&task.future_id))
                .and_then(|future| match future {
                    FutureState::Scheduled { receipt, .. } => receipt.clone(),
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
        let (future_id, completion, report) = match response {
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
                let report = AttemptReport::Succeeded {
                    result: AttemptSuccess::new(&attempt, outputs, Vec::new(), wall_millis, 0, 0)
                        .expect("browser attempt resource accounting must remain bounded"),
                };
                (
                    self.future_id(attempt.task_id),
                    FutureState::Completed(decoded),
                    report,
                )
            }
            Ok(ProgramExecutionResponse::GraphSuccess { graph }) => {
                if let Some(receipt) = graph_receipt {
                    let report = AttemptReport::Succeeded {
                        result: AttemptSuccess::Graph {
                            graph: graph.clone(),
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
        let mut state = self.state.borrow_mut();
        let mut retired = None;
        if let Some(future_id) = future_id {
            if !matches!(state.futures.get(&future_id), Some(FutureState::Cancelled)) {
                retired = state.futures.insert(future_id, completion);
                record_browser_completion(&mut state, attempt.task_id);
            }
        }
        state.requests.remove(&attempt.task_id);
        let actions = state
            .driver
            .handle(DriverCommand::BackendReport(BackendReport::for_request(
                &attempt, report,
            )))
            .unwrap_or_default();
        drop(state);
        drop(retired);
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
        if self.selected.borrow().contains_key(&attempt.task_id) {
            return self.cancel_selected_program(attempt);
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
        self.validate_scope(future.scope_id)?;
        let (reservation_id, call) = {
            let mut state = self.state.borrow_mut();
            let call = match state.futures.get(&future.id) {
                Some(FutureState::Deferred(call)) => Rc::clone(call),
                Some(_) => {
                    return Err(ExecutionServiceError::Failed(
                        "future has already been scheduled".into(),
                    ));
                }
                None => return Err(ExecutionServiceError::UnknownHandle),
            };
            let id = state.next_materialization;
            state.next_materialization = id.checked_add(1).ok_or_else(|| {
                ExecutionServiceError::Infrastructure(
                    "browser materialization identity exhausted".into(),
                )
            })?;
            let retired = state.futures.insert(
                future.id,
                FutureState::Materializing {
                    id,
                    call: Rc::clone(&call),
                    cancelled: false,
                },
            );
            drop(state);
            drop(retired);
            (id, call)
        };
        let mut guard = MaterializationGuard {
            service: self,
            future_id: future.id,
            id: reservation_id,
            armed: true,
        };
        // This can traverse managed values and call source/GC observers. It
        // must never run under the service's RefCell mutable borrow.
        let target = runmat_execution_artifact::ProgramTarget::portable("wasm32-browser-interpreter-bytecode-v2");
        let (callable, request, receipt) = if matches!(&call.program, runmat_runtime::execution::DeferredProgram::SelectedGraph(_)) {
            let (callable, request, receipt) = runmat_vm::materialize_deferred_graph_call(
                &call, future.outputs.clone(), target, runmat_execution::value::ValueLimits::default(),
            )?;
            (callable, BrowserWorkerRequest::Graph(request), Some(receipt))
        } else {
            let (callable, context, recipe, artifact, arguments) = runmat_vm::materialize_deferred_call(&call, future.outputs.clone(), target)?;
            let request = ProgramExecutionRequest {
                schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
                recipe, artifact, callable: callable.clone(), context, assignment: None, job_id: None,
                arguments, requested_outputs: future.outputs.requested_outputs,
                environment: call.environment.clone(), console: call.console,
            };
            (callable, BrowserWorkerRequest::Direct(request), None)
        };
        let (recipe, artifact, invocation_context, arguments) = match &request {
            BrowserWorkerRequest::Direct(r) => (&r.recipe, &r.artifact, r.context.clone(), r.arguments.clone()),
            BrowserWorkerRequest::Graph(r) => (&r.recipe, &r.artifact, r.context.clone(), Vec::new()),
            BrowserWorkerRequest::OwnedBuiltin(_) => unreachable!(),
        };
        let artifact_id = ArtifactId::derive(&[artifact.id.0.bytes()]);
        let mut resources = browser_request(self.capabilities, recipe.accelerators.clone());
        let context_bytes =
            runmat_execution::console::invocation_memory_bytes(&call.environment, call.console)
                .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
        resources.memory_bytes = resources
            .memory_bytes
            .checked_add(context_bytes)
            .ok_or_else(|| {
                ExecutionServiceError::Failed("invocation memory reservation overflow".into())
            })?;
        if let BrowserWorkerRequest::Graph(graph) = &request {
            let wire_bytes = serde_json::to_vec(graph).map_err(|e| ExecutionServiceError::Failed(e.to_string()))?;
            resources.memory_bytes = resources.memory_bytes.max(graph.input_allocation_floor_bytes(wire_bytes.len()).map_err(|e| ExecutionServiceError::Failed(e.to_string()))?);
            resources.scratch_bytes = resources.scratch_bytes.max(u64::try_from(wire_bytes.len()).map_err(|_| ExecutionServiceError::Failed("graph wire size exceeds u64".into()))?);
        }
        recipe
            .validate_resource_request(&resources)
            .map_err(|error| {
                ExecutionServiceError::Failed(format!(
                    "browser scheduler resources do not satisfy the program: {error}"
                ))
            })?;
        let trust = BTreeSet::from([runmat_execution::security::ExecutionTrustTier::CustomerTrusted]);
        let host = match &request {
            BrowserWorkerRequest::Graph(graph) => graph.execution_host_requirement(trust),
            _ => artifact.execution_host_requirement(recipe, trust),
        }.map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
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
        let generation = 1;
        let actions = match state
            .driver
            .handle(DriverCommand::Submit(Box::new(TaskSubmission {
                request: TaskRequest {
                    id: task_id,
                    scope_id: task_scope_id,
                    pool_id: self.pool_id,
                    schema_version: runmat_execution::task::TASK_REQUEST_SCHEMA_VERSION,
                    executable: runmat_execution::TaskExecutableIdentity::ProgramArtifact(
                        artifact_id,
                    ),
                    callable: Callable::for_program("browser-session", &callable),
                    invocation_context,
                    inputs: arguments,
                    environment: call.environment.clone(),
                    console: call.console,
                    outputs: future.outputs.clone(),
                    resources,
                    host,
                    retry: call.retry,
                    deadline_unix_millis: None,
                },
                dependencies: BTreeSet::new(),
                priority: 0,
            }))) {
            Ok(actions) => actions,
            Err(error) => {
                let _ = state.driver.handle(DriverCommand::CancelScope {
                    scope_id: task_scope_id,
                    reason: CancellationReason::DependencyFailed,
                    now_millis: js_sys::Date::now() as u64,
                });
                return Err(driver_error(error));
            }
        };
        state
            .requests
            .insert(task_id, request);
        state.tasks.insert(
            task_id,
            TaskRecord {
                future_id: future.id,
                generation,
                scope_id: task_scope_id,
                read: false,
                claimed: false,
                completion_order: None,
            },
        );
        let retired = state.futures.insert(
            future.id,
            FutureState::Scheduled {
                task_id,
                receipt,
            },
        );
        drop(state);
        drop(retired);
        guard.armed = false;
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
            FutureState::Scheduled { .. } | FutureState::Materializing { .. } => {
                Ok(AwaitAction::Pending(original))
            }
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
                    Some(FutureState::Scheduled { task_id, .. }) => Some(*task_id),
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
                (task.future_id, Some(handle.id), Some(task.scope_id))
            }
            _ => return Err(ExecutionServiceError::UnknownHandle),
        };
        let retired = {
            let mut state = self.state.borrow_mut();
            let retired = state.futures.insert(future_id, FutureState::Cancelled);
            if let Some(task_id) = task_id {
                record_browser_completion(&mut state, task_id);
            }
            retired
        };
        drop(retired);
        let actions = if let Some(scope_id) = cancellation_scope {
            self.state
                .borrow_mut()
                .driver
                .handle(DriverCommand::CancelScope {
                    scope_id,
                    reason,
                    now_millis: js_sys::Date::now() as u64,
                })
                .map_err(driver_error)?
        } else {
            Vec::new()
        };
        if let Some(task_id) = task_id {
            if !self.selected.borrow().contains_key(&task_id) {
                self.state.borrow_mut().requests.remove(&task_id);
            }
        }
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
                now_millis: js_sys::Date::now() as u64,
            })
            .unwrap_or_default();
        let mut state = self.state.borrow_mut();
        for future in state.futures.values_mut() {
            match future {
                FutureState::Completed(_)
                | FutureState::GraphCompleted { .. }
                | FutureState::GraphFailed { .. }
                | FutureState::GraphResolved { .. } => {}
                FutureState::Materializing { cancelled, .. } => *cancelled = true,
                _ => *future = FutureState::Cancelled,
            }
        }
        let task_ids = state.tasks.keys().copied().collect::<Vec<_>>();
        for task_id in task_ids {
            record_browser_completion(&mut state, task_id);
        }
        self.retain_launched_owned_requests(&mut state);
        drop(state);
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
        | FutureState::Scheduled { .. } => ExecutionHandleState::Running,
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
    use runmat_execution_artifact::{ProgramExecutionRequest, PROGRAM_EXECUTION_REQUEST_SCHEMA_V3};
    use wasm_bindgen_test::*;

    use runmat_execution::task::RetryPolicy;

    use super::*;

    wasm_bindgen_test_configure!(run_in_browser);

    fn deferred_test_call(program: Vec<u8>) -> DeferredCall {
        DeferredCall {
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

    #[wasm_bindgen_test(async)]
    async fn serial_fallback_schedules_through_the_portable_driver() {
        let service = BrowserExecutionService::new(None).unwrap();
        assert_eq!(service.capabilities.max_workers, 1);
        assert!(!service.capabilities.has_worker_isolation());
        let future = service
            .create_future(deferred_test_call(b"invalid registry".to_vec()))
            .unwrap();
        let task = service.spawn(&future).unwrap();
        assert!(matches!(
            service
                .begin_await(Value::Task(task.clone().into()))
                .unwrap(),
            AwaitAction::Pending(_)
        ));
        for _ in 0..8 {
            wasm_bindgen_futures::JsFuture::from(js_sys::Promise::resolve(
                &wasm_bindgen::JsValue::UNDEFINED,
            ))
            .await
            .unwrap();
            if service
                .begin_await(Value::Task(task.clone().into()))
                .is_err()
            {
                return;
            }
        }
        panic!("serial browser execution did not report its worker failure");
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

    #[wasm_bindgen_test(async)]
    async fn cancelling_a_scheduled_future_cancels_its_child_task() {
        let service = BrowserExecutionService::new(None).unwrap();
        let future = service
            .create_future(original_deferred_failure().await)
            .unwrap();
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
        let first = service.current_pool().unwrap().expect("browser pool");
        assert_eq!(first.backend, PoolBackend::Serial);
        assert_eq!(first.workers, 1);
        service.close_pool(&first.handle).unwrap();
        assert!(service.current_pool().unwrap().is_none());
        let reopened = service.ensure_pool(PoolRequest::automatic()).unwrap();
        assert_ne!(first.handle.generation, reopened.handle.generation);
    }
}
