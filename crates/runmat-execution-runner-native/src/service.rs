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
use crate::supervisor::{GraphBatchSubmission, ProgramBatchSubmission};
use crate::{NativeExecutionConfig, NativeExecutionResult};

mod materialization;

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
    Direct(Arc<TaskCompletion>),
    Graph {
        completion: Arc<TaskCompletion>,
        receipt: Rc<dyn GraphResultReceipt>,
    },
}

impl RunningCompletion {
    fn completion(&self) -> &Arc<TaskCompletion> {
        match self {
            Self::Direct(completion) | Self::Graph { completion, .. } => completion,
        }
    }
}

struct TaskRecord {
    future_id: FutureId,
    generation: u64,
    read: bool,
    claimed: bool,
    completion_order: Option<u64>,
}

struct GraphJobReceipt {
    handle: JobHandle,
    receipt: Rc<dyn GraphResultReceipt>,
    // A successful RootedValueSequence retains its GC root lease until the
    // supervisor's terminal/retention prune drops this exact receipt.
    terminal: RefCell<
        Option<Result<runmat_runtime::execution::RootedValueSequence, ExecutionServiceError>>,
    >,
    retain_until_unix_millis: u64,
}

struct State {
    next_future: u64,
    next_task: u64,
    next_materialization: u64,
    futures: HashMap<FutureId, FutureState>,
    cancelling_graphs: Vec<FutureId>,
    tasks: HashMap<TaskId, TaskRecord>,
    graph_jobs: HashMap<runmat_execution::JobId, Rc<GraphJobReceipt>>,
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
        drop(retired);
    }

    pub fn new(mut config: NativeExecutionConfig) -> NativeExecutionResult<Self> {
        let nonce = NEXT_NATIVE_SCOPE.fetch_add(1, Ordering::Relaxed);
        let scope_id = fresh_scope_id(b"native-session", nonce);
        config.store_root.push(scope_id.to_string());
        Ok(Self {
            scope_id,
            driver: LocalDriver::new(config, scope_id)?,
            durable: DurableJobBridge::start()
                .map_err(|error| crate::NativeExecutionError::Configuration(error.to_string()))?,
            state: Mutex::new(State {
                next_future: 0,
                next_task: 0,
                next_materialization: 0,
                futures: HashMap::new(),
                cancelling_graphs: Vec::new(),
                tasks: HashMap::new(),
                graph_jobs: HashMap::new(),
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

    /// Release retained V3/source leases only when the durable supervisor's
    /// terminal/retention authority has expired. Never hold the service lock
    /// across the supervisor query, which may reenter a callback or block.
    fn prune_expired_graph_receipts(&self) -> Result<(), ExecutionServiceError> {
        let now = crate::supervisor::unix_millis();
        let candidates = {
            let state = self.state.lock().expect("native service poisoned");
            state
                .graph_jobs
                .iter()
                .filter(|(_, record)| record.retain_until_unix_millis <= now)
                .map(|(id, record)| (*id, Rc::clone(record)))
                .collect::<Vec<_>>()
        };
        if candidates.is_empty() {
            return Ok(());
        }
        let expired = self
            .durable
            .expired_graph_receipts(candidates.iter().map(|(id, _)| *id).collect())?;
        let mut state = self.state.lock().expect("native service poisoned");
        let mut retired = Vec::new();
        for id in expired {
            if let Some((_, original)) = candidates.iter().find(|(candidate, _)| *candidate == id) {
                if state
                    .graph_jobs
                    .get(&id)
                    .is_some_and(|record| Rc::ptr_eq(record, original))
                {
                    retired.extend(state.graph_jobs.remove(&id));
                }
            }
        }
        drop(state);
        drop(retired);
        Ok(())
    }

    fn future_for_value(
        &self,
        value: Value,
    ) -> Result<Result<FutureHandle, Value>, ExecutionServiceError> {
        match value {
            Value::Future(handle) => Ok(Ok(handle)),
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
                Ok(Ok(FutureHandle {
                    id: task.future_id,
                    scope_id: self.scope_id,
                    outputs: handle.outputs,
                }))
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

    fn execute_spmd_gang(
        &self,
        call: runmat_runtime::execution::SpmdGangCall,
    ) -> runmat_runtime::context::RuntimeServiceFuture<
        Result<Vec<runmat_runtime::execution::SpmdRankResult>, ExecutionServiceError>,
    > {
        if let Err(error) = call.validate() {
            return Box::pin(async move { Err(error) });
        }
        if let Err(error) = self.validate_scope(call.gang.scope_id) {
            return Box::pin(async move { Err(error) });
        }
        let pool = match self.current_pool() {
            Ok(Some(pool)) if pool.handle == call.gang.pool => pool,
            Ok(_) => {
                return Box::pin(async {
                    Err(ExecutionServiceError::Failed(
                        "SPMD gang does not belong to the active native process pool".into(),
                    ))
                })
            }
            Err(error) => return Box::pin(async move { Err(error) }),
        };
        if call.gang.labs.0 > pool.workers {
            return Box::pin(async {
                Err(ExecutionServiceError::Failed(
                    "SPMD gang exceeds the active native process pool".into(),
                ))
            });
        }
        let requested_outputs = match u16::try_from(call.requested_outputs) {
            Ok(outputs) => outputs,
            Err(_) => return Box::pin(async { Err(ExecutionServiceError::InvalidOutputContract) }),
        };
        let mut pending: Vec<(runmat_types::LabRank, Arc<TaskCompletion>)> =
            Vec::with_capacity(call.gang.labs.0 as usize);
        for rank_number in 1..=call.gang.labs.0 {
            let rank = runmat_types::LabRank(rank_number);
            let invocation = runmat_runtime::execution::DeferredCall {
                invocation: runmat_runtime::execution::DeferredInvocation::Program {
                    callable: runmat_execution::ProgramCallable::spmd_region(call.region),
                    context: runmat_execution::ProgramInvocationContext::SpmdTask {
                        task: runmat_execution::SpmdTaskContext {
                            gang: call.gang.clone(),
                            region: call.region,
                            rank,
                        },
                    },
                    arguments: call.captures.clone(),
                    requested_outputs: call.requested_outputs,
                },
                retry: runmat_execution::RetryPolicy::Never,
                program_revision: call.program_revision.clone(),
                capabilities: call.capabilities.clone(),
                program: runmat_runtime::execution::DeferredProgram::RegionSourcesV1(
                    call.program.clone().into(),
                ),
            };
            let (callable, invocation_context, recipe, artifact, inputs) =
                match materialize_call(&invocation, OutputContract { requested_outputs }) {
                    Ok(parts) => parts,
                    Err(error) => return Box::pin(async move { Err(error) }),
                };
            let task_id = {
                let mut state = self.state.lock().expect("native service poisoned");
                let sequence = state.next_task;
                state.next_task = sequence.wrapping_add(1);
                TaskId::derive(&[self.scope_id.bytes(), b"spmd", &sequence.to_be_bytes()])
            };
            let completion = match self.driver.submit(crate::driver::LocalProgramSubmission {
                task_id,
                callable,
                invocation_context,
                recipe,
                artifact,
                inputs,
                outputs: OutputContract { requested_outputs },
                retry: runmat_execution::RetryPolicy::Never,
            }) {
                Ok(completion) => completion,
                Err(error) => {
                    self.driver
                        .fail_collective_gang(&call.gang, error.to_string());
                    for (_, completion) in pending {
                        completion.cancel();
                    }
                    return Box::pin(async move {
                        Err(ExecutionServiceError::Failed(error.to_string()))
                    });
                }
            };
            pending.push((rank, completion));
        }
        let gang = call.gang;
        let driver = Arc::clone(&self.driver);
        Box::pin(async move {
            let mut results = vec![None; pending.len()];
            loop {
                let mut first_failure = None;
                for (index, (rank, completion)) in pending.iter().enumerate() {
                    if results[index].is_some() {
                        continue;
                    }
                    if completion.is_cancelled() {
                        retain_first_spmd_failure(
                            &mut first_failure,
                            completion.completion_order(),
                            ExecutionServiceError::Cancelled,
                        );
                        continue;
                    }
                    let Some(result) = completion.try_value() else {
                        continue;
                    };
                    match result {
                        Ok(runmat_execution_runner::AttemptSuccess::Spmd { outputs })
                            if outputs.len() == usize::from(requested_outputs) =>
                        {
                            results[index] = Some(runmat_runtime::execution::SpmdRankResult {
                                rank: *rank,
                                outputs,
                            });
                        }
                        Ok(runmat_execution_runner::AttemptSuccess::Spmd { .. }) => {
                            retain_first_spmd_failure(
                                &mut first_failure,
                                completion.completion_order(),
                                ExecutionServiceError::InvalidOutputContract,
                            );
                        }
                        Ok(
                            runmat_execution_runner::AttemptSuccess::Values { .. }
                            | runmat_execution_runner::AttemptSuccess::Graph { .. },
                        ) => {
                            retain_first_spmd_failure(
                                &mut first_failure,
                                completion.completion_order(),
                                ExecutionServiceError::Failed(
                                    "SPMD worker returned an ordinary task result".into(),
                                ),
                            );
                        }
                        Err(crate::driver::TransferFailure::Execution(message)) => {
                            retain_first_spmd_failure(
                                &mut first_failure,
                                completion.completion_order(),
                                ExecutionServiceError::Failed(message),
                            );
                        }
                        Err(crate::driver::TransferFailure::Infrastructure(message)) => {
                            retain_first_spmd_failure(
                                &mut first_failure,
                                completion.completion_order(),
                                ExecutionServiceError::Infrastructure(message),
                            );
                        }
                        Err(
                            crate::driver::TransferFailure::WorkerLost(message)
                            | crate::driver::TransferFailure::RetirementUnverified(message),
                        ) => {
                            retain_first_spmd_failure(
                                &mut first_failure,
                                completion.completion_order(),
                                ExecutionServiceError::WorkerLost(message),
                            );
                        }
                        Err(crate::driver::TransferFailure::Cancelled) => {
                            retain_first_spmd_failure(
                                &mut first_failure,
                                completion.completion_order(),
                                ExecutionServiceError::Cancelled,
                            );
                        }
                        Err(crate::driver::TransferFailure::Runtime(failure)) => {
                            retain_first_spmd_failure(
                                &mut first_failure,
                                completion.completion_order(),
                                ExecutionServiceError::RuntimeFailure(failure),
                            );
                        }
                    }
                }
                if let Some((_, failure)) = first_failure {
                    driver.fail_collective_gang(&gang, "SPMD execution failed");
                    for (_, completion) in &pending {
                        completion.cancel();
                    }
                    return Err(failure);
                }
                if results.iter().all(Option::is_some) {
                    return Ok(results.into_iter().flatten().collect());
                }
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            }
        })
    }

    fn requires_program_capture(&self) -> bool {
        true
    }

    fn create_future(&self, call: DeferredCall) -> Result<FutureHandle, ExecutionServiceError> {
        self.sweep_completed_graph_cancellations();
        let requested_outputs = u16::try_from(call.invocation.requested_outputs())
            .map_err(|_| ExecutionServiceError::InvalidOutputContract)?;
        let mut state = self.state.lock().expect("native service poisoned");
        let sequence = state.next_future;
        state.next_future = sequence.wrapping_add(1);
        let id = FutureId::derive(&[self.scope_id.bytes(), &sequence.to_be_bytes()]);
        state
            .futures
            .insert(id, FutureState::Deferred(Rc::new(call)));
        Ok(FutureHandle {
            id,
            scope_id: self.scope_id,
            outputs: OutputContract { requested_outputs },
        })
    }

    fn spawn(&self, future: &FutureHandle) -> Result<TaskHandle, ExecutionServiceError> {
        self.sweep_completed_graph_cancellations();
        materialization::spawn(self, future)
    }

    fn inspect_future(
        &self,
        future: &FutureHandle,
    ) -> Result<ExecutionHandleSnapshot, ExecutionServiceError> {
        self.validate_scope(future.scope_id)?;
        self.sweep_completed_graph_cancellations();
        let state = self.state.lock().expect("native service poisoned");
        let stored_state = state
            .futures
            .get(&future.id)
            .ok_or(ExecutionServiceError::UnknownHandle)?;
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
        let state = state
            .futures
            .get(&record.future_id)
            .map(native_future_state)
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        Ok(ExecutionHandleSnapshot {
            state,
            outputs: task.outputs.clone(),
            read: record.read,
        })
    }

    fn claim_next_task_result(
        &self,
        tasks: &[TaskHandle],
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
        let _ = self.prune_expired_graph_receipts();
        let requested_outputs = u16::try_from(call.invocation.requested_outputs())
            .map_err(|_| ExecutionServiceError::InvalidOutputContract)?;
        let outputs = OutputContract { requested_outputs };
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
            let submitted = self.durable.submit_graph(GraphBatchSubmission {
                request,
                idempotency_key: options.idempotency_key,
                retention_millis: options.retention_millis,
            })?;
            if submitted.handle.outputs != outputs {
                return Err(ExecutionServiceError::InvalidOutputContract);
            }
            loop {
                let existing = {
                    let state = self.state.lock().expect("native service poisoned");
                    state.graph_jobs.get(&submitted.handle.id).map(Rc::clone)
                };
                if let Some(existing) = existing {
                    if existing.handle != submitted.handle
                        || existing.retain_until_unix_millis != submitted.retain_until_unix_millis
                        || !existing.receipt.same_selected_admission(receipt.as_ref())
                    {
                        return Err(ExecutionServiceError::ForeignScope);
                    }
                    let state = self.state.lock().expect("native service poisoned");
                    if state
                        .graph_jobs
                        .get(&submitted.handle.id)
                        .is_some_and(|current| Rc::ptr_eq(current, &existing))
                    {
                        return Ok(existing.handle.clone());
                    }
                    continue;
                }
                if !submitted.created_new {
                    // A supervisor record from a previous service/session
                    // cannot mint this session's local source and definition
                    // leases, even when its canonical bytes are identical.
                    return Err(ExecutionServiceError::UnknownHandle);
                }
                let mut state = self.state.lock().expect("native service poisoned");
                if state.graph_jobs.contains_key(&submitted.handle.id) {
                    continue;
                }
                state.graph_jobs.insert(
                    submitted.handle.id,
                    Rc::new(GraphJobReceipt {
                        handle: submitted.handle.clone(),
                        receipt,
                        terminal: RefCell::new(None),
                        retain_until_unix_millis: submitted.retain_until_unix_millis,
                    }),
                );
                return Ok(submitted.handle);
            }
        }
        let (callable, invocation_context, recipe, artifact, arguments) =
            materialize_call(&call, outputs)?;
        self.durable.submit(ProgramBatchSubmission {
            recipe,
            artifact,
            callable,
            invocation_context,
            arguments,
            requested_outputs,
            idempotency_key: options.idempotency_key,
            retention_millis: options.retention_millis,
        })
    }

    fn await_job(&self, job: &JobHandle) -> Result<Value, ExecutionServiceError> {
        if self
            .state
            .lock()
            .expect("native service poisoned")
            .graph_jobs
            .contains_key(&job.id)
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
                .await_job(handle)
                .and_then(|value| {
                    ValueSequence::single(value)
                        .map_err(|error| ExecutionServiceError::Failed(error.to_string()))
                })
                .and_then(runmat_runtime::execution::RootedValueSequence::new)
                .map(AwaitAction::Completed);
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
                FutureState::Running(RunningCompletion::Direct(completion)) => {
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
        let Some(completion) = completion.try_value() else {
            return Ok(AwaitAction::Pending(original));
        };
        let result = completion
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
            self.prune_expired_graph_receipts()?;
            let graph = {
                let state = self.state.lock().expect("native service poisoned");
                state.graph_jobs.get(&handle.id).map(Rc::clone)
            };
            if let Some(record) = graph {
                if record.handle != *handle {
                    return Err(ExecutionServiceError::UnknownHandle);
                }
                let receipt = &record.receipt;
                receipt.check_issuing_context(runtime)?;
                if let Some(terminal) = record.terminal.borrow().as_ref() {
                    return terminal.clone().map(AwaitAction::Completed);
                }
                let response = self.durable.await_response(handle.clone())?;
                let terminal = match response {
                    runmat_execution_artifact::ProgramExecutionResponse::GraphSuccess { graph } => {
                        receipt
                            .decode_in_context(runtime, &graph)
                            .and_then(|result| {
                                (result.as_slice().len()
                                    == usize::from(record.handle.outputs.requested_outputs))
                                .then_some(result)
                                .ok_or(ExecutionServiceError::InvalidOutputContract)
                            })
                    }
                    runmat_execution_artifact::ProgramExecutionResponse::Failure { message } => {
                        Err(ExecutionServiceError::Failed(message))
                    }
                    runmat_execution_artifact::ProgramExecutionResponse::RuntimeFailure {
                        failure,
                    } => Err(ExecutionServiceError::LocalizedRuntimeFailure(
                        receipt.decode_failure_in_context(runtime, &failure)?.into(),
                    )),
                    _ => Err(ExecutionServiceError::Infrastructure(
                        "graph job returned an incompatible durable response".into(),
                    )),
                };
                if matches!(
                    &terminal,
                    Err(ExecutionServiceError::ForeignScope | ExecutionServiceError::UnknownHandle)
                ) {
                    // A session check can race issuer retirement. Never let a
                    // foreign or retired fetch publish this job's local cache.
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
                        .expect("graph terminal was published")
                        .clone()
                };
                // A reentrant fetch may have won publication while this call
                // decoded. Retire the unused rooted value outside the borrow.
                drop(pending);
                return published.map(AwaitAction::Completed);
            }
            return self.begin_await(value);
        }
        let original = value.clone();
        let future = match self.future_for_value(value)? {
            Ok(future) => future,
            Err(value) => return Ok(AwaitAction::Passthrough(value)),
        };
        self.validate_scope(future.scope_id)?;
        let graph_running = {
            let state = self.state.lock().expect("native service poisoned");
            match state.futures.get(&future.id) {
                Some(FutureState::Running(RunningCompletion::Graph {
                    completion,
                    receipt,
                })) => Some((Arc::clone(completion), Rc::clone(receipt))),
                Some(FutureState::CancellingGraph(RunningCompletion::Graph {
                    completion,
                    receipt,
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
            Ok(runmat_execution_runner::AttemptSuccess::Graph { graph }) => {
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
        future: &FutureHandle,
        result: Result<runmat_runtime::execution::RootedValueSequence, ExecutionServiceError>,
    ) -> Result<(), ExecutionServiceError> {
        self.validate_scope(future.scope_id)?;
        let mut state = self.state.lock().expect("native service poisoned");
        if !matches!(
            state.futures.get(&future.id),
            Some(FutureState::ExecutingInCaller)
        ) {
            return Err(ExecutionServiceError::Failed(
                "future is not executing in its caller".into(),
            ));
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
        let future_id = match value {
            Value::Future(handle) => {
                self.validate_scope(handle.scope_id)?;
                handle.id
            }
            Value::Task(handle) => {
                self.validate_scope(handle.scope_id)?;
                self.state
                    .lock()
                    .expect("native service poisoned")
                    .tasks
                    .get(&handle.id)
                    .ok_or(ExecutionServiceError::UnknownHandle)?
                    .future_id
            }
            Value::Job(handle) => return self.durable.cancel(handle.clone()),
            _ => return Err(ExecutionServiceError::UnknownHandle),
        };
        let mut state = self.state.lock().expect("native service poisoned");
        let record = state
            .futures
            .get_mut(&future_id)
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        if let FutureState::Materializing(reservation) = record {
            reservation.cancelled = true;
            return Ok(());
        }
        if matches!(record, FutureState::CancellingGraph(_)) {
            return Ok(());
        }
        let previous = std::mem::replace(record, FutureState::Cancelled);
        let (running, retired) = match previous {
            FutureState::Running(RunningCompletion::Graph {
                completion,
                receipt,
            }) => {
                *state.futures.get_mut(&future_id).expect("future exists") =
                    FutureState::CancellingGraph(RunningCompletion::Graph {
                        completion: Arc::clone(&completion),
                        receipt,
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
                .cancel_task(task_id)
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
        for (future_id, future) in &mut state.futures {
            if let FutureState::Materializing(reservation) = future {
                reservation.cancelled = true;
                continue;
            }
            if matches!(future, FutureState::CancellingGraph(_)) {
                continue;
            }
            if !matches!(
                future,
                FutureState::Completed(_) | FutureState::GraphResolved { .. }
            ) {
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

fn retire_verified_graph_cancellations(state: &mut State) -> Vec<FutureState> {
    let mut still_pending = Vec::new();
    let mut completed = Vec::new();
    for id in std::mem::take(&mut state.cancelling_graphs) {
        if let Some(FutureState::CancellingGraph(running)) = state.futures.get(&id) {
            match running.completion().verified_terminal() {
                Some(true) => completed.push((id, running.completion().completion_order())),
                None => still_pending.push(id),
                // Failed tree retirement is deliberately retained, but
                // it cannot later become a verified cancellation.
                Some(false) => {}
            }
        }
    }
    state.cancelling_graphs = still_pending;
    let mut retired = Vec::with_capacity(completed.len());
    for (future_id, order) in completed {
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
    retired
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
        FutureState::Running(running) if running.completion().try_value().is_none() => {
            ExecutionHandleState::Running
        }
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
        FutureState::Running(running) => match running.completion().try_value() {
            Some(Ok(_)) => ExecutionHandleState::Finished,
            Some(Err(_)) => ExecutionHandleState::Failed,
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
    match completion.try_value() {
        None => Ok(AwaitAction::Pending(original)),
        Some(Err(crate::driver::TransferFailure::RetirementUnverified(message))) => {
            Err(ExecutionServiceError::WorkerLost(message))
        }
        Some(_) => Err(ExecutionServiceError::Cancelled),
    }
}

fn retain_first_spmd_failure(
    first: &mut Option<(u64, ExecutionServiceError)>,
    completion_order: Option<u64>,
    failure: ExecutionServiceError,
) {
    let order = completion_order.unwrap_or(u64::MAX);
    if first
        .as_ref()
        .is_none_or(|(current_order, _)| order < *current_order)
    {
        *first = Some((order, failure));
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
            std::env::consts::OS
        )),
    )
}

#[cfg(test)]
mod spmd_failure_tests {
    use super::{retain_first_spmd_failure, ExecutionServiceError};

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
