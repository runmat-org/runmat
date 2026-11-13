use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use runmat_execution::{
    CancellationReason, ExecutionHandleSnapshot, ExecutionHandleState, ExecutionScopeId,
    FutureHandle, FutureId, JobHandle, OutputContract, PoolBackend, PoolHandle, PoolRequest,
    PoolSnapshot, PoolState, TaskHandle, TaskId, TaskResultClaim,
};
use runmat_runtime::execution::{
    AwaitAction, DeferredCall, DurableJobOptions, ExecutionServiceError, GraphResultReceipt,
    RuntimeExecutionServices, SpmdExecutionMode,
};
use runmat_value::{Value, ValueSequence};

use crate::config::fresh_scope_id;
use crate::driver::{LocalDriver, TaskCompletion};
use crate::durable::DurableJobBridge;
use crate::{NativeExecutionConfig, NativeExecutionResult};

mod future_table;
mod materialization;
mod output;
mod job_output;
mod job_receipt;
mod spmd;
use future_table::FutureTable;
use job_receipt::{DurableJobReceipt, JobResultAuthority};

static NEXT_NATIVE_SCOPE: AtomicU64 = AtomicU64::new(1);

enum FutureState {
    Deferred(Rc<DeferredCall>),
    ExecutingInCaller,
    Materializing(materialization::Reservation),
    Running(RunningCompletion),
    // Graph cancellation is requested, but its selected receipt and host
    // resources remain owned until the process driver reports retirement.
    CancellingGraph(RunningCompletion),
    GraphResolved {
        result: Result<runmat_runtime::execution::RootedValueSequence, ExecutionServiceError>,
        receipt: Rc<dyn GraphResultReceipt>,
    },
    Completed(Result<runmat_runtime::execution::RootedValueSequence, ExecutionServiceError>),
    Cancelled,
}

enum RunningCompletion {
    Direct {
        completion: Arc<TaskCompletion>,
        output: Rc<RefCell<output::CallerOutput>>,
    },
    Graph {
        completion: Arc<TaskCompletion>,
        receipt: Rc<dyn GraphResultReceipt>,
        output: Rc<RefCell<output::CallerOutput>>,
    },
}

impl FutureState {
    /// Once an outcome exists, cancellation cannot rewrite it through another
    /// handle. A retained Running completion may already be terminal before
    /// its first await decodes the result. This check does not certify physical
    /// retirement: even an unverified terminal keeps its existing exact owner.
    fn has_terminal_outcome(&self) -> bool {
        match self {
            Self::Completed(_) | Self::GraphResolved { .. } | Self::Cancelled => true,
            Self::Running(running) => running.completion().verified_terminal().is_some(),
            Self::Deferred(_)
            | Self::ExecutingInCaller
            | Self::Materializing(_)
            | Self::CancellingGraph(_) => false,
        }
    }
}

impl RunningCompletion {
    fn new(completion: Arc<TaskCompletion>, receipt: Option<Rc<dyn GraphResultReceipt>>) -> Self {
        let output = Rc::new(RefCell::new(output::CallerOutput::default()));
        match receipt {
            Some(receipt) => Self::Graph {
                completion,
                receipt,
                output,
            },
            None => Self::Direct { completion, output },
        }
    }

    fn completion(&self) -> &Arc<TaskCompletion> {
        match self {
            Self::Direct { completion, .. } | Self::Graph { completion, .. } => completion,
        }
    }

    fn output(&self) -> &Rc<RefCell<output::CallerOutput>> {
        match self {
            Self::Direct { output, .. } | Self::Graph { output, .. } => output,
        }
    }
}

struct TaskRecord {
    future_id: FutureId,
    generation: u64,
    read: bool,
    claimed: bool,
    completion_order: Option<u64>,
    retention: runmat_runtime::execution::TaskRetention,
    callback_completion: Option<Arc<TaskCompletion>>,
}

struct State {
    next_future: u64,
    next_task: u64,
    next_materialization: u64,
    futures: FutureTable,
    cancelling_graphs: Vec<FutureId>,
    tasks: HashMap<TaskId, TaskRecord>,
    durable_jobs: HashMap<runmat_execution::JobId, Rc<DurableJobReceipt>>,
    pool_generation: u64,
    pool_open: bool,
}

pub struct NativeExecutionService {
    scope_id: ExecutionScopeId,
    driver: Arc<LocalDriver>,
    durable: DurableJobBridge,
    state: Mutex<State>,
}

impl NativeExecutionService {
    fn sweep_completed_graph_cancellations(&self) {
        let mut state = self.state.lock().expect("native service poisoned");
        let retired = retire_verified_graph_cancellations(&mut state);
        drop(state);
        for task in &retired.callbacks {
            self.driver.release_completed_callback(*task);
        }
        drop(retired);
    }

    pub fn new(mut config: NativeExecutionConfig) -> NativeExecutionResult<Self> {
        let nonce = NEXT_NATIVE_SCOPE.fetch_add(1, Ordering::Relaxed);
        let scope_id = fresh_scope_id(b"native-session", nonce);
        config.scope_store(scope_id);
        Ok(Self {
            scope_id,
            driver: LocalDriver::new(config, scope_id)?,
            durable: DurableJobBridge::start()
                .map_err(|error| crate::NativeExecutionError::Configuration(error.to_string()))?,
            state: Mutex::new(State {
                next_future: 0,
                next_task: 0,
                next_materialization: 0,
                futures: FutureTable::default(),
                cancelling_graphs: Vec::new(),
                tasks: HashMap::new(),
                durable_jobs: HashMap::new(),
                pool_generation: 1,
                pool_open: false,
            }),
        })
    }

    fn validate_scope(&self, scope_id: ExecutionScopeId) -> Result<(), ExecutionServiceError> {
        if scope_id == self.scope_id {
            Ok(())
        } else {
            Err(ExecutionServiceError::ForeignScope)
        }
    }

    fn future_for_value(
        &self,
        value: Value,
    ) -> Result<
        Result<runmat_runtime::execution::FutureExecutionHandle, Value>,
        ExecutionServiceError,
    > {
        match value {
            Value::Future(handle) => {
                self.validate_scope(handle.scope_id)?;
                self.state
                    .lock()
                    .expect("native service poisoned")
                    .futures
                    .validate(&handle.id, &handle.outputs)?;
                Ok(Ok(handle.into()))
            }
            Value::Task(handle) => {
                self.validate_scope(handle.scope_id)?;
                let state = self.state.lock().expect("native service poisoned");
                let task = state
                    .tasks
                    .get(&handle.id)
                    .ok_or(ExecutionServiceError::UnknownHandle)?;
                if task.generation != handle.generation {
                    return Err(ExecutionServiceError::UnknownHandle);
                }
                state.futures.validate(&task.future_id, &handle.outputs)?;
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
}

impl RuntimeExecutionServices for NativeExecutionService {
    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        let mut state = self.state.try_lock().ok()?;
        let retired = retire_verified_graph_cancellations(&mut state);
        drop(state);
        for task in &retired.callbacks {
            self.driver.release_completed_callback(*task);
        }
        drop(retired);
        let state = self.state.try_lock().ok()?;
        for future in state.futures.values() {
            match future {
                FutureState::Deferred(call) => call.invocation.visit_retained_values(visitor)?,
                FutureState::Materializing(reservation) => {
                    reservation.call.invocation.visit_retained_values(visitor)?;
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
                // Running workers retain serialized ValuePayload/AttemptSuccess,
                // never this process's resident Value handles.
                FutureState::Running(_)
                | FutureState::CancellingGraph(_)
                | FutureState::ExecutingInCaller
                | FutureState::Completed(Err(_))
                | FutureState::GraphResolved { result: Err(_), .. }
                | FutureState::Cancelled => {}
            }
        }
        for job in state.durable_jobs.values() {
            // Root leases protect collection; alias visibility also has to
            // expose these cached values before an ownership-sensitive move.
            let terminal = job.terminal.try_borrow().ok()?;
            if let Some(Ok(sequence)) = terminal.as_ref() {
                for value in sequence.value().iter() {
                    visitor(value)?;
                }
            }
        }
        Some(())
    }

    fn scope_id(&self) -> ExecutionScopeId {
        self.scope_id
    }

    fn spmd_execution_mode(&self) -> SpmdExecutionMode {
        SpmdExecutionMode::IsolatedWorkers
    }

    fn current_pool(&self) -> Result<Option<PoolSnapshot>, ExecutionServiceError> {
        self.sweep_completed_graph_cancellations();
        let state = self.state.lock().expect("native service poisoned");
        if !state.pool_open {
            return Ok(None);
        }
        let workers = self
            .driver
            .active_workers()
            .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
        Ok(Some(PoolSnapshot {
            handle: PoolHandle {
                id: self.driver.pool_id(),
                scope_id: self.scope_id,
                generation: state.pool_generation,
            },
            backend: PoolBackend::LocalProcesses,
            workers,
            state: PoolState::Ready,
        }))
    }

    fn ensure_pool(&self, request: PoolRequest) -> Result<PoolSnapshot, ExecutionServiceError> {
        if request
            .backend
            .is_some_and(|backend| backend != PoolBackend::LocalProcesses)
        {
            return Err(ExecutionServiceError::Failed(
                "this session supports the local process pool backend".into(),
            ));
        }
        if request.workers.is_none() {
            let state = self.state.lock().expect("native service poisoned");
            if state.pool_open {
                let generation = state.pool_generation;
                drop(state);
                let workers = self
                    .driver
                    .active_workers()
                    .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
                return Ok(PoolSnapshot {
                    handle: PoolHandle {
                        id: self.driver.pool_id(),
                        scope_id: self.scope_id,
                        generation,
                    },
                    backend: PoolBackend::LocalProcesses,
                    workers,
                    state: PoolState::Ready,
                });
            }
        }
        let workers = request.workers.unwrap_or_else(|| self.driver.max_workers());
        let previous_workers = {
            let state = self.state.lock().expect("native service poisoned");
            if state.pool_open {
                Some(
                    self.driver
                        .active_workers()
                        .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?,
                )
            } else {
                None
            }
        };
        self.driver
            .resize_pool(workers)
            .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
        let mut state = self.state.lock().expect("native service poisoned");
        if previous_workers.is_some_and(|previous| previous != workers) {
            state.pool_generation = state.pool_generation.wrapping_add(1);
        }
        state.pool_open = true;
        Ok(PoolSnapshot {
            handle: PoolHandle {
                id: self.driver.pool_id(),
                scope_id: self.scope_id,
                generation: state.pool_generation,
            },
            backend: PoolBackend::LocalProcesses,
            workers,
            state: PoolState::Ready,
        })
    }

    fn close_pool(&self, pool: &PoolHandle) -> Result<(), ExecutionServiceError> {
        self.validate_scope(pool.scope_id)?;
        let mut state = self.state.lock().expect("native service poisoned");
        if !state.pool_open
            || pool.id != self.driver.pool_id()
            || pool.generation != state.pool_generation
        {
            return Err(ExecutionServiceError::UnknownHandle);
        }
        state.pool_open = false;
        state.pool_generation = state.pool_generation.wrapping_add(1);
        for future in state.futures.values_mut() {
            if let FutureState::Materializing(reservation) = future {
                reservation.cancelled = true;
            }
        }
        drop(state);
        self.driver.cancel_all(CancellationReason::User);
        Ok(())
    }

    fn execute_spmd_gang<'a>(
        &'a self,
        call: runmat_runtime::execution::SpmdGangCall,
        runtime: &'a runmat_runtime::context::RuntimeContext,
    ) -> Result<
        runmat_runtime::execution::AdmittedExecutionSupervisor<
            'a,
            runmat_runtime::execution::SpmdGangResults,
            ExecutionServiceError,
        >,
        ExecutionServiceError,
    > {
        runmat_runtime::execution::admit_execution_supervisor_with(runtime, move || {
            spmd::execute(self, call, runtime)
        })
    }

    fn requires_program_capture(&self) -> bool {
        true
    }

    fn supports_callback_batches(&self) -> bool {
        true
    }
    fn submit_callback_batch(
        &self,
        calls: Vec<DeferredCall>,
        pool: &PoolHandle,
        cancellation: runmat_execution::cancellation::CancellationToken,
    ) -> Result<Vec<TaskHandle>, ExecutionServiceError> {
        materialization::submit_callback_batch(self, calls, pool, cancellation, None)
    }
    fn submit_callback_batch_in_context(
        &self,
        calls: Vec<DeferredCall>,
        pool: &PoolHandle,
        cancellation: runmat_execution::cancellation::CancellationToken,
        runtime: &runmat_runtime::context::RuntimeContext,
    ) -> Result<Vec<TaskHandle>, ExecutionServiceError> {
        self.driver
            .bind_console_parent(runtime, calls.iter().any(|call| call.console.is_some()))?;
        let owner =
            runmat_runtime::context::host_allocation::HostControlAllocationOwner::capture(runtime)
                .map_err(|error| {
                    runmat_runtime::context::host_allocation::host_allocation_runtime_error(
                        Some(runtime),
                        "RunMat:parallel:CpuControl",
                        "native callback control admission",
                        error,
                    )
                })?;
        materialization::submit_callback_batch(self, calls, pool, cancellation, owner)
    }
    fn release_callback_batch(&self, tasks: &[TaskHandle]) -> Result<(), ExecutionServiceError> {
        materialization::release_callback_batch(self, tasks)
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
        if let Value::Job(handle) = value {
            let record = self.durable_receipt(handle)?;
            return Ok(Some(job_output::wait(
                record,
                &self.durable,
                handle.clone(),
                runtime,
            )?));
        }
        if !matches!(value, Value::Future(_) | Value::Task(_)) {
            return Ok(None);
        }
        let completion = (|| {
            let future = self
                .future_for_value(value.clone())?
                .map_err(|_| ExecutionServiceError::UnknownHandle)?;
            self.validate_scope(future.scope_id)?;
            let state = self.state.lock().expect("native service poisoned");
            match state.futures.get(&future.id) {
                Some(FutureState::Running(running) | FutureState::CancellingGraph(running)) => {
                    let receipt = match running {
                        RunningCompletion::Graph { receipt, .. } => Some(Rc::clone(receipt)),
                        RunningCompletion::Direct { .. } => None,
                    };
                    Ok(Some((
                        Arc::clone(running.completion()),
                        receipt,
                        Rc::clone(running.output()),
                    )))
                }
                Some(_) => Ok(None),
                None => Err(ExecutionServiceError::UnknownHandle),
            }
        })();
        Ok(Some(
            runmat_runtime::execution::admit_execution_supervisor_with(
                runtime,
                move || async move {
                    if let Some((completion, receipt, output)) = completion? {
                        if let Some(receipt) = receipt.as_ref() {
                            receipt.check_issuing_context(runtime)?;
                        }
                        loop {
                            let wake = completion.raw_output_notification();
                            tokio::pin!(wake);
                            wake.as_mut().enable();
                            let pumped = match output.try_borrow_mut() {
                                Ok(mut output) => output.pump(&completion, runtime),
                                Err(_) => {
                                    return Err(ExecutionServiceError::Infrastructure(
                                        "application output delivery is already in progress".into(),
                                    ))
                                }
                            };
                            match pumped {
                                Ok(true) => {
                                    tokio::task::yield_now().await;
                                    continue;
                                }
                                Err(_) => {
                                    completion.stop_for_output_delivery();
                                    Arc::clone(&completion).wait_terminal().await;
                                    return Err(output.borrow_mut().settle_failure(
                                        &completion,
                                        runtime,
                                        receipt.as_deref(),
                                    ));
                                }
                                Ok(false) => {}
                            }
                            if completion.verified_terminal().is_some() {
                                break;
                            }
                            wake.await;
                        }
                    }
                    Ok(())
                },
            )?,
        ))
    }
    fn create_future(&self, call: DeferredCall) -> Result<FutureHandle, ExecutionServiceError> {
        self.sweep_completed_graph_cancellations();
        let requested_outputs = u16::try_from(call.invocation.requested_outputs())
            .map_err(|_| ExecutionServiceError::InvalidOutputContract)?;
        let mut state = self.state.lock().expect("native service poisoned");
        let sequence = state.next_future;
        state.next_future = sequence
            .checked_add(1)
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        let id = FutureId::derive(&[self.scope_id.bytes(), &sequence.to_be_bytes()]);
        let outputs = OutputContract {
            requested_outputs,
            owned_work: None,
        };
        state
            .futures
            .issue(id, outputs.clone(), FutureState::Deferred(Rc::new(call)));
        Ok(FutureHandle {
            id,
            scope_id: self.scope_id,
            outputs,
        })
    }

    fn spawn(&self, future: &FutureHandle) -> Result<TaskHandle, ExecutionServiceError> {
        self.sweep_completed_graph_cancellations();
        materialization::spawn(self, future, None)
    }

    fn spawn_in_context(
        &self,
        future: &FutureHandle,
        pool: Option<&PoolHandle>,
        runtime: &runmat_runtime::context::RuntimeContext,
    ) -> Result<TaskHandle, ExecutionServiceError> {
        let capture_requested = {
            let state = self.state.lock().expect("native service poisoned");
            match state.futures.get(&future.id) {
                Some(FutureState::Deferred(call)) => call.console.is_some(),
                Some(FutureState::Materializing(reservation)) => reservation.call.console.is_some(),
                _ => false,
            }
        };
        self.driver
            .bind_console_parent(runtime, capture_requested)?;
        if let Some(pool) = pool {
            self.inspect_pool(pool)?;
        }
        let owner =
            runmat_runtime::context::host_allocation::HostControlAllocationOwner::capture(runtime)
                .map_err(|error| {
                    runmat_runtime::context::host_allocation::host_allocation_runtime_error(
                        Some(runtime),
                        "RunMat:parallel:CpuControl",
                        "native task control admission",
                        error,
                    )
                })?;
        self.sweep_completed_graph_cancellations();
        materialization::spawn(self, future, owner)
    }

    fn inspect_future(
        &self,
        future: &FutureHandle,
    ) -> Result<ExecutionHandleSnapshot, ExecutionServiceError> {
        self.validate_scope(future.scope_id)?;
        self.sweep_completed_graph_cancellations();
        let state = self.state.lock().expect("native service poisoned");
        let stored_state = state.futures.validate(&future.id, &future.outputs)?;
        Ok(ExecutionHandleSnapshot {
            state: native_future_state(stored_state),
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
        self.sweep_completed_graph_cancellations();
        let state = self.state.lock().expect("native service poisoned");
        let record = state
            .tasks
            .get(&task.id)
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        if record.generation != task.generation {
            return Err(ExecutionServiceError::UnknownHandle);
        }
        let state = native_future_state(state.futures.validate(&record.future_id, &task.outputs)?);
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
        self.sweep_completed_graph_cancellations();
        let mut state = self.state.lock().expect("native service poisoned");
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
            let order =
                record
                    .completion_order
                    .or_else(|| match state.futures.get(&record.future_id) {
                        Some(
                            FutureState::Running(running) | FutureState::CancellingGraph(running),
                        ) => running.completion().completion_order(),
                        _ => None,
                    });
            if let Some(order) = order {
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
        let mut state = self.state.lock().expect("native service poisoned");
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

    fn submit_job(
        &self,
        call: DeferredCall,
        options: DurableJobOptions,
    ) -> Result<JobHandle, ExecutionServiceError> {
        // Best-effort reclamation cannot prevent a valid new submission when
        // the supervisor is temporarily unavailable.
        let _ = self.prune_expired_job_receipts();
        let requested_outputs = u16::try_from(call.invocation.requested_outputs())
            .map_err(|_| ExecutionServiceError::InvalidOutputContract)?;
        let outputs = OutputContract {
            requested_outputs,
            owned_work: None,
        };
        if matches!(
            &call.program,
            runmat_runtime::execution::DeferredProgram::SelectedGraph(_)
        ) {
            let target = runmat_execution_artifact::ProgramTarget::portable(format!(
                "{}-{}-interpreter-bytecode-v2",
                std::env::consts::ARCH,
                std::env::consts::OS
            ));
            let (_, request, receipt) = runmat_vm::materialize_deferred_graph_call(
                &call,
                outputs.clone(),
                target,
                runmat_execution::value::ValueLimits::default(),
            )?;
            let submitted = self.durable.submit_graph(request, options)?;
            if submitted.handle.outputs != outputs {
                return Err(ExecutionServiceError::InvalidOutputContract);
            }
            return self
                .retain_durable_receipt(submitted, JobResultAuthority::SelectedGraph(receipt))
                .map(|record| record.handle.clone());
        }
        let (callable, invocation_context, recipe, artifact, arguments) =
            materialize_call(&call, outputs)?;
        let submitted = self.durable.submit(
            runmat_execution_artifact::ProgramExecutionRequest {
                schema_version: runmat_execution_artifact::PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
                recipe,
                artifact,
                callable,
                context: invocation_context,
                assignment: None,
                job_id: None,
                arguments,
                requested_outputs,
                environment: call.environment.clone(),
                console: call.console,
            },
            options,
        )?;
        if submitted.handle.outputs.requested_outputs != requested_outputs {
            return Err(ExecutionServiceError::InvalidOutputContract);
        }
        self.retain_durable_receipt(submitted, JobResultAuthority::Direct)
            .map(|record| record.handle.clone())
    }

    fn await_job(
        &self,
        job: &runmat_value::SharedJobHandle,
    ) -> Result<Value, ExecutionServiceError> {
        if self
            .state
            .lock()
            .expect("native service poisoned")
            .durable_jobs
            .get(&job.id)
            .is_some_and(|record| matches!(&record.authority, JobResultAuthority::SelectedGraph(_)))
        {
            return Err(ExecutionServiceError::Failed(
                "graph job requires the issuing runtime context".into(),
            ));
        }
        self.durable.await_job(job.clone())
    }

    fn begin_await(&self, value: Value) -> Result<AwaitAction, ExecutionServiceError> {
        self.sweep_completed_graph_cancellations();
        if let Value::Job(handle) = &value {
            return self
                .durable
                .try_await_job(handle)
                .and_then(|value| {
                    value
                        .map(|value| {
                            ValueSequence::single(value)
                                .map_err(|error| ExecutionServiceError::Failed(error.to_string()))
                                .and_then(runmat_runtime::execution::RootedValueSequence::new)
                        })
                        .transpose()
                })
                .map(|result| match result {
                    Some(value) => AwaitAction::Completed(value),
                    None => AwaitAction::Pending(value),
                });
        }
        let original = value.clone();
        let future = match self.future_for_value(value)? {
            Ok(future) => future,
            Err(value) => return Ok(AwaitAction::Passthrough(value)),
        };
        self.validate_scope(future.scope_id)?;
        let completion = {
            let mut state = self.state.lock().expect("native service poisoned");
            match state
                .futures
                .get_mut(&future.id)
                .ok_or(ExecutionServiceError::UnknownHandle)?
            {
                FutureState::Deferred(_) => {
                    let retired = state
                        .futures
                        .insert(future.id, FutureState::ExecutingInCaller)
                        .expect("future was just resolved");
                    drop(state);
                    let FutureState::Deferred(call) = retired else {
                        unreachable!("validated deferred future changed under mutex")
                    };
                    let call = Rc::try_unwrap(call).map_err(|_| {
                        ExecutionServiceError::Infrastructure(
                            "deferred call retained an unexpected concurrent owner".into(),
                        )
                    })?;
                    return Ok(AwaitAction::ExecuteFuture {
                        handle: future,
                        call: Box::new(call),
                    });
                }
                FutureState::Materializing(reservation) if reservation.cancelled => {
                    return Err(ExecutionServiceError::Cancelled);
                }
                FutureState::Materializing(_) => return Ok(AwaitAction::Pending(original)),
                FutureState::ExecutingInCaller => {
                    return Err(ExecutionServiceError::Failed(
                        "future is executing in its caller".into(),
                    ));
                }
                FutureState::Running(RunningCompletion::Direct { completion, output }) => {
                    let output = output.try_borrow().map_err(|_| {
                        ExecutionServiceError::Infrastructure(
                            "application output delivery is already in progress".into(),
                        )
                    })?;
                    if output.pending(completion) {
                        return Err(ExecutionServiceError::Infrastructure(
                            "application output requires the issuing runtime context".into(),
                        ));
                    }
                    Arc::clone(completion)
                }
                FutureState::Running(RunningCompletion::Graph { .. }) => {
                    return Err(ExecutionServiceError::Infrastructure(
                        "graph result requires the issuing runtime context".into(),
                    ));
                }
                FutureState::CancellingGraph(_) => {
                    return Err(ExecutionServiceError::Infrastructure(
                        "graph result requires the issuing runtime context".into(),
                    ));
                }
                FutureState::GraphResolved { .. } => {
                    return Err(ExecutionServiceError::Infrastructure(
                        "graph result requires the issuing runtime context".into(),
                    ));
                }
                FutureState::Completed(result) => {
                    let result = result.clone();
                    return result.map(AwaitAction::Completed);
                }
                FutureState::Cancelled => return Err(ExecutionServiceError::Cancelled),
            }
        };
        let Some(outcome) = completion.try_value() else {
            return Ok(AwaitAction::Pending(original));
        };
        let result = outcome
            .and_then(|success| {
                let Some((outputs, result_objects)) = success.values() else {
                    return Err(crate::driver::TransferFailure::Infrastructure(
                        "SPMD task result requires the typed gang execution path".into(),
                    ));
                };
                if !result_objects.is_empty() {
                    return Err(crate::driver::TransferFailure::Infrastructure(
                        "native runtime call returned externalized objects without an artifact consumer"
                            .into(),
                    ));
                }
                outputs
                    .iter()
                    .map(runmat_runtime::execution::value_codec::decode_inline_value)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|error| {
                        crate::driver::TransferFailure::Infrastructure(error.to_string())
                    })
                    .and_then(|values| {
                        ValueSequence::comma_separated(values).map_err(|error| {
                            crate::driver::TransferFailure::Infrastructure(error.to_string())
                        })
                    })
            })
            .map_err(|failure| match failure {
                crate::driver::TransferFailure::Execution(message) => {
                    ExecutionServiceError::Failed(message)
                }
                crate::driver::TransferFailure::Infrastructure(message) => {
                    ExecutionServiceError::Infrastructure(message)
                }
                crate::driver::TransferFailure::WorkerLost(message)
                | crate::driver::TransferFailure::RetirementUnverified(message) => {
                    ExecutionServiceError::WorkerLost(message)
                }
                crate::driver::TransferFailure::Cancelled => ExecutionServiceError::Cancelled,
                crate::driver::TransferFailure::Runtime(failure) => {
                    ExecutionServiceError::RuntimeFailure(failure)
                }
            });
        let mut state = self.state.lock().expect("native service poisoned");
        let rooted = result.and_then(runmat_runtime::execution::RootedValueSequence::new);
        let retired = state
            .futures
            .insert(future.id, FutureState::Completed(rooted.clone()));
        drop(state);
        drop(retired);
        rooted.map(AwaitAction::Completed)
    }

    fn begin_await_in_context(
        &self,
        value: Value,
        runtime: &runmat_runtime::context::RuntimeContext,
    ) -> Result<AwaitAction, ExecutionServiceError> {
        self.sweep_completed_graph_cancellations();
        if let Value::Job(handle) = &value {
            self.prune_expired_job_receipt(handle.id)?;
            let record = self.durable_receipt(handle)?;
            record.authority.check_context(runtime)?;
            if let Some(terminal) = record.terminal.borrow().as_ref() {
                return terminal.clone().map(AwaitAction::Completed);
            }
            if !record
                .output
                .try_borrow()
                .is_ok_and(|output| output.complete())
            {
                return Ok(AwaitAction::Pending(value));
            }
            let Some(response) = self.durable.try_response(handle)? else {
                return Ok(AwaitAction::Pending(value));
            };
            let terminal = record.authority.decode(
                runtime,
                response,
                usize::from(record.handle.outputs.requested_outputs),
            );
            if matches!(
                &terminal,
                Err(ExecutionServiceError::ForeignScope | ExecutionServiceError::UnknownHandle)
            ) {
                // An expired issuer must not publish into a caller's retained cache.
                return terminal.map(AwaitAction::Completed);
            }
            let mut pending = Some(terminal);
            let published = {
                let mut cache = record.terminal.borrow_mut();
                if cache.is_none() {
                    *cache = pending.take();
                }
                cache
                    .as_ref()
                    .expect("durable terminal was published")
                    .clone()
            };
            // Drop any reentrant loser outside the RefCell borrow.
            drop(pending);
            return published.map(AwaitAction::Completed);
        }
        let original = value.clone();
        let future = match self.future_for_value(value)? {
            Ok(future) => future,
            Err(value) => return Ok(AwaitAction::Passthrough(value)),
        };
        self.validate_scope(future.scope_id)?;
        let pending_output = {
            let state = self.state.lock().expect("native service poisoned");
            // future_for_value validated the public handle's output contract
            // before reducing it to this internal execution identity.
            match state
                .futures
                .get(&future.id)
                .ok_or(ExecutionServiceError::UnknownHandle)?
            {
                FutureState::Running(running) | FutureState::CancellingGraph(running) => {
                    let receipt = match running {
                        RunningCompletion::Graph { receipt, .. } => Some(Rc::clone(receipt)),
                        RunningCompletion::Direct { .. } => None,
                    };
                    Some((
                        Arc::clone(running.completion()),
                        Rc::clone(running.output()),
                        receipt,
                    ))
                }
                _ => None,
            }
        };
        if let Some((completion, output, receipt)) = pending_output {
            if let Some(receipt) = receipt {
                receipt.check_issuing_context(runtime)?;
            }
            let pending = output
                .try_borrow()
                .map_or(true, |output| output.pending(&completion))
                || completion.raw_output_failure().is_some();
            if pending {
                return Ok(AwaitAction::Pending(original));
            }
        }
        let graph_running = {
            let state = self.state.lock().expect("native service poisoned");
            match state.futures.get(&future.id) {
                Some(FutureState::Running(RunningCompletion::Graph {
                    completion,
                    receipt,
                    ..
                })) => Some((Arc::clone(completion), Rc::clone(receipt))),
                Some(FutureState::CancellingGraph(RunningCompletion::Graph {
                    completion,
                    receipt,
                    ..
                })) => {
                    let completion = Arc::clone(completion);
                    let receipt = Rc::clone(receipt);
                    drop(state);
                    receipt.check_issuing_context(runtime)?;
                    return cancelling_graph_await(&completion, original);
                }
                Some(FutureState::GraphResolved { result, receipt }) => {
                    let result = result.clone();
                    let receipt = Rc::clone(receipt);
                    drop(state);
                    receipt.check_issuing_context(runtime)?;
                    return result.map(AwaitAction::Completed);
                }
                _ => None,
            }
        };
        let Some((completion, receipt)) = graph_running else {
            return self.begin_await(original);
        };
        receipt.check_issuing_context(runtime)?;
        let Some(transfer) = completion.try_value() else {
            return Ok(AwaitAction::Pending(original));
        };
        let rooted = match transfer {
            Ok(runmat_execution_runner::AttemptSuccess::Graph { graph, .. }) => {
                receipt.decode_in_context(runtime, &graph)
            }
            Ok(_) => Err(ExecutionServiceError::Infrastructure(
                "graph task returned a non-graph result".into(),
            )),
            Err(failure) => Err(match failure {
                crate::driver::TransferFailure::Execution(message) => {
                    ExecutionServiceError::Failed(message)
                }
                crate::driver::TransferFailure::Infrastructure(message) => {
                    ExecutionServiceError::Infrastructure(message)
                }
                crate::driver::TransferFailure::WorkerLost(message)
                | crate::driver::TransferFailure::RetirementUnverified(message) => {
                    ExecutionServiceError::WorkerLost(message)
                }
                crate::driver::TransferFailure::Cancelled => ExecutionServiceError::Cancelled,
                crate::driver::TransferFailure::Runtime(failure) => receipt
                    .decode_failure_in_context(runtime, &failure)
                    .map(|error| ExecutionServiceError::LocalizedRuntimeFailure(error.into()))
                    .unwrap_or_else(|error| error),
            }),
        };
        if matches!(&rooted, Err(ExecutionServiceError::ForeignScope)) {
            // A different session may inspect this handle, but it cannot
            // consume the issuing session's completed graph or receipt.
            return Err(ExecutionServiceError::ForeignScope);
        }
        let mut state = self.state.lock().expect("native service poisoned");
        match state.futures.get(&future.id) {
            Some(FutureState::Running(RunningCompletion::Graph { .. })) => {}
            Some(FutureState::Cancelled) => return Err(ExecutionServiceError::Cancelled),
            Some(FutureState::CancellingGraph(_)) => {
                return cancelling_graph_await(&completion, original);
            }
            Some(FutureState::GraphResolved { result, receipt }) => {
                let result = result.clone();
                let receipt = Rc::clone(receipt);
                drop(state);
                receipt.check_issuing_context(runtime)?;
                return result.map(AwaitAction::Completed);
            }
            Some(FutureState::Completed(_)) => {
                return Err(ExecutionServiceError::Infrastructure(
                    "graph task changed into a direct result".into(),
                ));
            }
            _ => {
                return Err(ExecutionServiceError::Infrastructure(
                    "graph task changed ownership during result publication".into(),
                ))
            }
        }
        let retired = state.futures.insert(
            future.id,
            FutureState::GraphResolved {
                result: rooted.clone(),
                receipt,
            },
        );
        drop(state);
        drop(retired);
        rooted.map(AwaitAction::Completed)
    }

    fn complete_future(
        &self,
        future: runmat_runtime::execution::FutureCompletionRef<'_>,
        result: Result<runmat_runtime::execution::RootedValueSequence, ExecutionServiceError>,
    ) -> Result<(), ExecutionServiceError> {
        self.validate_scope(future.scope_id)?;
        let mut state = self.state.lock().expect("native service poisoned");
        let record = state.futures.validate(&future.id, future.outputs)?;
        // A caller completing a future that was cancelled meanwhile learns of the cancellation.
        if matches!(record, FutureState::Cancelled) {
            return Err(ExecutionServiceError::Cancelled);
        }
        if !matches!(record, FutureState::ExecutingInCaller) {
            return Err(ExecutionServiceError::Failed(
                "future is not executing in its caller".into(),
            ));
        }
        if result.as_ref().is_ok_and(|sequence| {
            sequence.value().len() != usize::from(future.outputs.requested_outputs)
        }) {
            return Err(ExecutionServiceError::InvalidOutputContract);
        }
        let retired = state
            .futures
            .insert(future.id, FutureState::Completed(result));
        record_native_completion(&mut state, future.id);
        drop(state);
        drop(retired);
        Ok(())
    }

    fn cancel(
        &self,
        value: &Value,
        reason: CancellationReason,
    ) -> Result<(), ExecutionServiceError> {
        self.sweep_completed_graph_cancellations();
        let (future_id, outputs) = match value {
            Value::Future(handle) => {
                self.validate_scope(handle.scope_id)?;
                (handle.id, &handle.outputs)
            }
            Value::Task(handle) => {
                self.validate_scope(handle.scope_id)?;
                let state = self.state.lock().expect("native service poisoned");
                let task = state
                    .tasks
                    .get(&handle.id)
                    .ok_or(ExecutionServiceError::UnknownHandle)?;
                if task.generation != handle.generation {
                    return Err(ExecutionServiceError::UnknownHandle);
                }
                (task.future_id, &handle.outputs)
            }
            Value::Job(handle) => return self.durable.cancel(handle.clone(), reason),
            _ => return Err(ExecutionServiceError::UnknownHandle),
        };
        let mut state = self.state.lock().expect("native service poisoned");
        state.futures.validate(&future_id, outputs)?;
        let record = state
            .futures
            .get_mut(&future_id)
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        if let FutureState::Materializing(reservation) = record {
            reservation.cancelled = true;
            return Ok(());
        }
        // Preserve every terminal outcome through the canonical state query;
        // cancellation already in progress must not be restarted either.
        if matches!(record, FutureState::CancellingGraph(_)) || record.has_terminal_outcome() {
            return Ok(());
        }
        let previous = std::mem::replace(record, FutureState::Cancelled);
        let (running, retired) = match previous {
            FutureState::Running(RunningCompletion::Graph {
                completion,
                receipt,
                output,
            }) => {
                *state.futures.get_mut(&future_id).expect("future exists") =
                    FutureState::CancellingGraph(RunningCompletion::Graph {
                        completion: Arc::clone(&completion),
                        receipt,
                        output,
                    });
                state.cancelling_graphs.push(future_id);
                (Some(completion), None)
            }
            other => {
                let running = match &other {
                    FutureState::Running(running) => Some(Arc::clone(running.completion())),
                    _ => None,
                };
                record_native_completion(&mut state, future_id);
                (running, Some(other))
            }
        };
        let task_id = running.as_ref().and_then(|_| {
            state
                .tasks
                .iter()
                .find(|(_, task)| task.future_id == future_id)
                .map(|(task_id, _)| *task_id)
        });
        drop(state);
        if let Some(completion) = running {
            completion.cancel_with_reason(reason);
            let task_id = task_id.ok_or_else(|| {
                ExecutionServiceError::Infrastructure(
                    "running native future has no accepted scheduler task".into(),
                )
            })?;
            self.driver
                .cancel_task(task_id, reason)
                .map_err(|error| ExecutionServiceError::Infrastructure(error.to_string()))?;
        }
        drop(retired);
        Ok(())
    }

    fn drain_scope(&self, reason: CancellationReason) {
        self.sweep_completed_graph_cancellations();
        self.driver.cancel_all(reason);
        let mut state = self.state.lock().expect("native service poisoned");
        let mut retired = Vec::new();
        let mut running = Vec::new();
        let mut newly_cancelling = Vec::new();
        for (future_id, future) in state.futures.iter_mut() {
            if let FutureState::Materializing(reservation) = future {
                reservation.cancelled = true;
                continue;
            }
            if matches!(future, FutureState::CancellingGraph(_)) {
                continue;
            }
            if !future.has_terminal_outcome() {
                let previous = std::mem::replace(future, FutureState::Cancelled);
                if let FutureState::Running(RunningCompletion::Graph { .. }) = &previous {
                    *future = FutureState::CancellingGraph(match previous {
                        FutureState::Running(running) => running,
                        _ => unreachable!(),
                    });
                    newly_cancelling.push(*future_id);
                    continue;
                }
                if let FutureState::Running(completion) = &previous {
                    running.push(Arc::clone(completion.completion()));
                }
                retired.push(previous);
            }
        }
        state.cancelling_graphs.extend(newly_cancelling);
        let future_ids = state
            .tasks
            .values()
            .map(|task| task.future_id)
            .collect::<Vec<_>>();
        for future_id in future_ids {
            if !matches!(
                state.futures.get(&future_id),
                Some(FutureState::CancellingGraph(_))
            ) {
                record_native_completion(&mut state, future_id);
            }
        }
        drop(state);
        for completion in running {
            completion.cancel();
        }
        drop(retired);
    }
}

struct RetiredGraphValues {
    _values: Vec<FutureState>,
    callbacks: Vec<TaskId>,
}
fn retire_verified_graph_cancellations(state: &mut State) -> RetiredGraphValues {
    let mut still_pending = Vec::new();
    let mut completed = Vec::new();
    for id in std::mem::take(&mut state.cancelling_graphs) {
        let completion = match state.futures.get(&id) {
            Some(FutureState::CancellingGraph(running)) => Some(running.completion()),
            Some(_) => state
                .tasks
                .values()
                .find(|task| {
                    task.future_id == id
                        && task.retention
                            == runmat_runtime::execution::TaskRetention::ReleasingCallbackBatch
                })
                .and_then(|task| task.callback_completion.as_ref()),
            None => None,
        };
        if let Some(completion) = completion {
            match completion.verified_terminal() {
                Some(true) => completed.push((id, completion.completion_order())),
                None => still_pending.push(id),
                Some(false) => {}
            }
        }
    }
    state.cancelling_graphs = still_pending;
    let mut retired = Vec::with_capacity(completed.len());
    let mut callbacks = Vec::new();
    for (future_id, order) in completed {
        let releasing = state.tasks.iter().find_map(|(id, task)| {
            (task.future_id == future_id
                && task.retention
                    == runmat_runtime::execution::TaskRetention::ReleasingCallbackBatch)
                .then_some(*id)
        });
        if let Some(task) = releasing {
            if let Some(previous) = state.futures.remove(&future_id) {
                retired.push(previous);
            }
            state.tasks.remove(&task);
            callbacks.push(task);
            continue;
        }
        if let Some(previous) = state.futures.insert(future_id, FutureState::Cancelled) {
            retired.push(previous);
        }
        if let Some(task) = state
            .tasks
            .values_mut()
            .find(|task| task.future_id == future_id)
        {
            if task.completion_order.is_none() {
                task.completion_order = order;
            }
        }
    }
    RetiredGraphValues {
        _values: retired,
        callbacks,
    }
}

fn record_native_completion(state: &mut State, future_id: FutureId) {
    let Some(task) = state
        .tasks
        .values_mut()
        .find(|task| task.future_id == future_id)
    else {
        return;
    };
    if task.completion_order.is_none() {
        task.completion_order = Some(crate::driver::next_task_completion_order());
    }
}

fn native_future_state(state: &FutureState) -> ExecutionHandleState {
    match state {
        FutureState::Deferred(_) => ExecutionHandleState::Deferred,
        FutureState::ExecutingInCaller => ExecutionHandleState::Running,
        FutureState::Materializing(reservation) if reservation.cancelled => {
            ExecutionHandleState::Cancelled
        }
        FutureState::Materializing(_) => ExecutionHandleState::Running,
        FutureState::CancellingGraph(running)
            if running.completion().verified_terminal().is_none() =>
        {
            ExecutionHandleState::Running
        }
        FutureState::CancellingGraph(running)
            if running.completion().verified_terminal() == Some(false) =>
        {
            ExecutionHandleState::Failed
        }
        FutureState::CancellingGraph(_) => ExecutionHandleState::Cancelled,
        FutureState::Running(running) => match running.completion().terminal_succeeded() {
            Some(true) => ExecutionHandleState::Finished,
            Some(false) => ExecutionHandleState::Failed,
            None => ExecutionHandleState::Running,
        },
        FutureState::Completed(Ok(_)) => ExecutionHandleState::Finished,
        FutureState::Completed(Err(_)) => ExecutionHandleState::Failed,
        FutureState::GraphResolved { result: Ok(_), .. } => ExecutionHandleState::Finished,
        FutureState::GraphResolved { result: Err(_), .. } => ExecutionHandleState::Failed,
        FutureState::Cancelled => ExecutionHandleState::Cancelled,
    }
}

fn cancelling_graph_await(
    completion: &TaskCompletion,
    original: Value,
) -> Result<AwaitAction, ExecutionServiceError> {
    if completion.verified_terminal().is_none() {
        return Ok(AwaitAction::Pending(original));
    }
    match completion.original_failure() {
        Some(crate::driver::TransferFailure::RetirementUnverified(message)) => {
            Err(ExecutionServiceError::WorkerLost(message))
        }
        _ => Err(ExecutionServiceError::Cancelled),
    }
}

fn materialize_call(
    call: &DeferredCall,
    outputs: OutputContract,
) -> Result<
    (
        runmat_execution::ProgramCallable,
        runmat_execution::ProgramInvocationContext,
        runmat_execution_artifact::ProgramBuildRecipe,
        runmat_execution_artifact::ProgramArtifact,
        Vec<runmat_execution::value::ValuePayload>,
    ),
    ExecutionServiceError,
> {
    runmat_vm::materialize_deferred_call(
        call,
        outputs,
        runmat_execution_artifact::ProgramTarget::portable(format!(
            "{}-{}-interpreter-bytecode-v2",
            std::env::consts::ARCH,
            std::env::consts::OS,
        )),
    )
}

#[cfg(test)]
mod spmd_failure_tests {
    use super::{spmd::retain_first_spmd_failure, ExecutionServiceError};

    #[test]
    fn originating_completion_wins_over_rank_iteration_order() {
        let mut first = None;
        retain_first_spmd_failure(
            &mut first,
            Some(12),
            ExecutionServiceError::Failed("peer termination".into()),
        );
        retain_first_spmd_failure(
            &mut first,
            Some(11),
            ExecutionServiceError::RuntimeFailure(Box::new(
                runmat_execution::ProgramRuntimeFailure {
                    message: "source failure".into(),
                    identifier: Some("RunMat:IndexOutOfBounds".into()),
                    span: None,
                    builtin: None,
                    task_id: None,
                    call_frames: Vec::new(),
                    call_frames_elided: 0,
                    call_stack: Vec::new(),
                    phase: None,
                },
            )),
        );
        let Some((11, ExecutionServiceError::RuntimeFailure(failure))) = first else {
            panic!("earliest structured runtime failure was not retained");
        };
        assert_eq!(
            failure.identifier.as_deref(),
            Some("RunMat:IndexOutOfBounds")
        );
    }
}
