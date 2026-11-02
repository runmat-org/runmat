use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use runmat_execution::resource::ResourceRequest;
use runmat_execution::{
    CancellationReason, ExecutionHandleSnapshot, ExecutionHandleState, ExecutionScopeId,
    FutureHandle, FutureId, JobHandle, OutputContract, PoolBackend, PoolHandle, PoolId,
    PoolRequest, PoolSnapshot, PoolState, TaskHandle, TaskId, TaskResultClaim,
};
use runmat_value::Value;

use super::{CpuCooperationPolicy, ExecutionServiceError};

#[path = "services/deferred_program.rs"]
mod deferred_program;
pub use deferred_program::{
    AdmittedCallableReentry, DeferredGraphCapture, DeferredGraphProgram, DeferredGraphSnapshot,
    DeferredProgram, DeferredSelectedCodeCapture, GraphResultReceipt, RestoredCallableGraph,
};

static NEXT_SERVICE_NONCE: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, PartialEq)]
pub enum DeferredInvocation {
    Callable(Box<crate::call::descriptor::CallableDescriptor>),
    Program {
        callable: runmat_execution::ProgramCallable,
        context: runmat_execution::ProgramInvocationContext,
        arguments: Vec<Value>,
        requested_outputs: usize,
    },
}

#[cfg(test)]
#[path = "services/deferred_roots_tests.rs"]
mod deferred_roots_tests;

impl DeferredInvocation {
    /// Only a deferred callable pays for this allocation. Ordinary callable
    /// descriptors remain inline at their non-deferred call sites.
    pub fn callable(descriptor: crate::call::descriptor::CallableDescriptor) -> Self {
        Self::Callable(Box::new(descriptor))
    }

    /// Values retained by this invocation, including its callable target graph.
    pub fn visit_retained_values(
        &self,
        visitor: &mut dyn FnMut(&Value) -> Option<()>,
    ) -> Option<()> {
        match self {
            Self::Callable(descriptor) => descriptor.visit_retained_values(visitor),
            Self::Program { arguments, .. } => {
                for value in arguments {
                    visitor(value)?;
                }
                Some(())
            }
        }
    }

    pub fn requested_outputs(&self) -> usize {
        match self {
            Self::Callable(descriptor) => descriptor.requested_outputs,
            Self::Program {
                requested_outputs, ..
            } => *requested_outputs,
        }
    }

    pub fn arguments(&self) -> &[Value] {
        match self {
            Self::Callable(descriptor) => &descriptor.args,
            Self::Program { arguments, .. } => arguments,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeferredCall {
    /// Retained at creation from the selected compiler/catalog owner, not
    /// reconstructed later from a Future's wire/type representation.
    pub spawn_safety: runmat_types::SpawnSafetyFact,
    pub invocation: DeferredInvocation,
    /// Retry semantics proven at the call's semantic origin. Execution
    /// adapters must carry this value unchanged into their task request.
    pub retry: runmat_execution::RetryPolicy,
    pub program_revision: Option<runmat_execution::ProgramRevision>,
    /// Invocation environment frozen at the call's semantic origin.
    pub environment: runmat_execution::EnvironmentSnapshot,
    /// Console capture controls requested by the invoking owner.
    pub console: Option<runmat_execution::console::ConsoleLimits>,
    /// Whole-call semantic capabilities emitted by analysis and retained in
    /// bytecode. Execution adapters translate these facts into leases; they do
    /// not rediscover requirements from source or builtin spelling.
    pub capabilities: runmat_types::CapabilitySet,
    /// Exact, runtime-opaque program description supplied by the VM.
    ///
    /// The serial service does not inspect this payload. Execution adapters may
    /// require it to reproduce the callable in an isolated worker.
    pub program: DeferredProgram,
}

impl DeferredCall {
    pub fn require_spawn_admission(
        &self,
        outputs: &OutputContract,
    ) -> Result<(), ExecutionServiceError> {
        if outputs.requested_outputs as usize != self.invocation.requested_outputs()
            || outputs.owned_work.is_some()
        {
            return Err(ExecutionServiceError::InvalidOutputContract);
        }
        if let runmat_types::SpawnSafetyFact::NotSpawnSafe { reason } = self.spawn_safety {
            return Err(ExecutionServiceError::SpawnUnsafe(reason));
        }
        Ok(())
    }
}

/// Exact compiler-bound SPMD program submitted as one gang operation.
/// Execution backends may place ranks in separate workers, but may not split
/// admission, retry, or result identity into unrelated ordinary futures.
#[derive(Clone, Debug, PartialEq)]
pub struct SpmdGangCall {
    pub gang: runmat_execution::GangHandle,
    pub region: runmat_types::ParallelRegionId,
    pub captures: Vec<Value>,
    pub requested_outputs: usize,
    pub program_revision: Option<runmat_execution::ProgramRevision>,
    pub capabilities: runmat_types::CapabilitySet,
    pub program: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpmdRankResult {
    pub rank: runmat_types::LabRank,
    pub outputs: Vec<Option<runmat_execution::SpmdOutputValue>>,
}

/// Rank results retain the exact parent-control backing until caller decoding ends.
pub type SpmdGangResults = crate::context::host_allocation::HostControlVec<SpmdRankResult>;

/// How an execution backend runs one compiler-owned SPMD gang.
///
/// Program capture and SPMD placement are independent capabilities: a browser
/// host may require captured programs for ordinary tasks while retaining SPMD
/// ranks in one cooperative runtime, whereas a native process pool executes
/// each rank through an isolated program worker.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpmdExecutionMode {
    Cooperative,
    IsolatedWorkers,
}

impl SpmdGangCall {
    pub fn validate(&self) -> Result<(), ExecutionServiceError> {
        self.gang
            .validate()
            .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
        if self.requested_outputs > u16::MAX as usize
            || self.captures.len() > 4096
            || self.program.is_empty()
        {
            return Err(ExecutionServiceError::Failed(
                "SPMD gang call is empty or exceeds its bounded execution contract".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DurableJobOptions {
    pub idempotency_key: Option<String>,
    pub retention_millis: u64,
    pub resources: ResourceRequest,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AwaitAction {
    Passthrough(Value),
    /// The host has started the operation but cannot complete it synchronously.
    ///
    /// The caller waits through `wait_pending` before polling `begin_await`
    /// again. Adapters can await an actual completion event without keeping
    /// an executor's microtask queue permanently runnable.
    Pending(Value),
    ExecuteFuture {
        handle: super::FutureExecutionHandle,
        call: Box<DeferredCall>,
    },
    Completed(super::RootedValueSequence),
}

/// Service-local result ownership. Handles alone cannot turn a public,
/// refetchable Future into an internal numerical temporary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TaskRetention {
    Refetchable,
    CallbackBatch,
    ReleasingCallbackBatch,
}

pub trait RuntimeExecutionServices {
    fn cpu_cooperation_policy(&self) -> CpuCooperationPolicy {
        CpuCooperationPolicy::NeverYield
    }

    fn cooperate_cpu(
        &self,
    ) -> crate::context::RuntimeServiceFuture<Result<(), ExecutionServiceError>> {
        Box::pin(async {
            Err(ExecutionServiceError::Infrastructure(
                "CPU task-turn scheduling is unavailable in this execution adapter".into(),
            ))
        })
    }

    /// Complete process-local values retained by this service. `None` means
    /// alias visibility is unavailable, not that the service retains nothing.
    fn visit_retained_values(&self, _visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        None
    }

    fn scope_id(&self) -> ExecutionScopeId;
    fn requires_program_capture(&self) -> bool {
        false
    }
    fn spmd_execution_mode(&self) -> SpmdExecutionMode {
        SpmdExecutionMode::Cooperative
    }
    fn current_pool(&self) -> Result<Option<PoolSnapshot>, ExecutionServiceError>;
    fn ensure_pool(&self, request: PoolRequest) -> Result<PoolSnapshot, ExecutionServiceError>;
    fn close_pool(&self, pool: &PoolHandle) -> Result<(), ExecutionServiceError>;
    fn execute_spmd_gang<'a>(
        &'a self,
        _call: SpmdGangCall,
        runtime: &'a crate::context::RuntimeContext,
    ) -> Result<
        super::AdmittedExecutionSupervisor<'a, SpmdGangResults, ExecutionServiceError>,
        ExecutionServiceError,
    > {
        super::admit_execution_supervisor_with(runtime, || async {
            Err(ExecutionServiceError::Failed(
                "this execution backend does not provide isolated SPMD gangs".into(),
            ))
        })
    }
    fn inspect_pool(&self, pool: &PoolHandle) -> Result<PoolSnapshot, ExecutionServiceError> {
        let snapshot = self
            .current_pool()?
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        if snapshot.handle == *pool {
            Ok(snapshot)
        } else if snapshot.handle.scope_id == pool.scope_id {
            Err(ExecutionServiceError::UnknownHandle)
        } else {
            Err(ExecutionServiceError::ForeignScope)
        }
    }
    /// Atomic worker publication and task-change wake, not a serial pool.
    fn supports_callback_batches(&self) -> bool {
        false
    }
    fn submit_callback_batch(
        &self,
        _calls: Vec<DeferredCall>,
        _pool: &PoolHandle,
        _cancellation: runmat_execution::cancellation::CancellationToken,
    ) -> Result<Vec<TaskHandle>, ExecutionServiceError> {
        Err(ExecutionServiceError::Infrastructure(
            "worker callback batches unavailable".into(),
        ))
    }
    fn submit_callback_batch_in_context(
        &self,
        calls: Vec<DeferredCall>,
        pool: &PoolHandle,
        cancellation: runmat_execution::cancellation::CancellationToken,
        _runtime: &crate::context::RuntimeContext,
    ) -> Result<Vec<TaskHandle>, ExecutionServiceError> {
        self.submit_callback_batch(calls, pool, cancellation)
    }
    /// Release only internal batch handles; pending cancellation keeps its
    /// exact receipt until verified physical retirement. Public Futures retain
    /// their ordinary repeated-fetch semantics.
    fn release_callback_batch(&self, _tasks: &[TaskHandle]) -> Result<(), ExecutionServiceError> {
        Err(ExecutionServiceError::Infrastructure(
            "worker callback retirement unavailable".into(),
        ))
    }
    /// Wait for progress on the exact value returned by `AwaitAction::Pending`.
    /// Event-backed adapters return a wake future. `None` preserves cooperative
    /// polling for handles whose progress is driven by their caller.
    /// The borrowed context owns any caller-visible output publication; it must
    /// not be retained by the service or transferred to a process-owner thread.
    /// A supervisor keeps cancellation cleanup in the adapter's state machine.
    fn wait_pending<'a>(
        &'a self,
        _value: &Value,
        _runtime: &'a crate::context::RuntimeContext,
    ) -> Result<
        Option<super::AdmittedExecutionSupervisor<'a, (), ExecutionServiceError>>,
        ExecutionServiceError,
    > {
        Ok(None)
    }
    fn create_future(&self, call: DeferredCall) -> Result<FutureHandle, ExecutionServiceError>;
    fn spawn(&self, future: &FutureHandle) -> Result<TaskHandle, ExecutionServiceError>;
    /// Launch using the caller's explicit allocation authority. A process
    /// adapter carries only the resulting control issuer, never the RuntimeContext.
    fn spawn_in_context(
        &self,
        future: &FutureHandle,
        pool: Option<&PoolHandle>,
        _runtime: &crate::context::RuntimeContext,
    ) -> Result<TaskHandle, ExecutionServiceError> {
        self.spawn_on(future, pool)
    }
    fn inspect_future(
        &self,
        future: &FutureHandle,
    ) -> Result<ExecutionHandleSnapshot, ExecutionServiceError>;
    fn inspect_task(
        &self,
        task: &TaskHandle,
    ) -> Result<ExecutionHandleSnapshot, ExecutionServiceError>;
    fn claim_next_task_result(
        &self,
        tasks: &[runmat_value::SharedTaskHandle],
    ) -> Result<TaskResultClaim, ExecutionServiceError>;
    fn mark_task_result_read(&self, task: &TaskHandle) -> Result<(), ExecutionServiceError>;
    fn spawn_on(
        &self,
        future: &FutureHandle,
        pool: Option<&PoolHandle>,
    ) -> Result<TaskHandle, ExecutionServiceError> {
        if let Some(pool) = pool {
            let current = self
                .current_pool()?
                .ok_or(ExecutionServiceError::UnknownHandle)?;
            if current.handle != *pool {
                return Err(if current.handle.scope_id == pool.scope_id {
                    ExecutionServiceError::UnknownHandle
                } else {
                    ExecutionServiceError::ForeignScope
                });
            }
        }
        self.spawn(future)
    }
    fn submit_job(
        &self,
        _call: DeferredCall,
        _options: DurableJobOptions,
    ) -> Result<JobHandle, ExecutionServiceError> {
        Err(ExecutionServiceError::Failed(
            "durable jobs are unavailable in this execution backend".into(),
        ))
    }
    fn await_job(
        &self,
        _job: &runmat_value::SharedJobHandle,
    ) -> Result<Value, ExecutionServiceError> {
        Err(ExecutionServiceError::Failed(
            "durable jobs are unavailable in this execution backend".into(),
        ))
    }
    fn begin_await(&self, value: Value) -> Result<AwaitAction, ExecutionServiceError>;

    /// Decode an admitted graph result under the caller's exact session and
    /// source authority. Direct callers retain their existing fast path.
    fn begin_await_in_context(
        &self,
        value: Value,
        _runtime: &crate::context::RuntimeContext,
    ) -> Result<AwaitAction, ExecutionServiceError> {
        self.begin_await(value)
    }
    fn complete_future(
        &self,
        future: super::FutureCompletionRef<'_>,
        result: Result<super::RootedValueSequence, ExecutionServiceError>,
    ) -> Result<(), ExecutionServiceError>;
    fn cancel(
        &self,
        value: &Value,
        reason: CancellationReason,
    ) -> Result<(), ExecutionServiceError>;
    fn drain_scope(&self, reason: CancellationReason);
}

#[derive(Clone, Debug)]
enum FutureState {
    Deferred(Box<DeferredCall>),
    Running,
    Completed(Result<super::RootedValueSequence, ExecutionServiceError>),
    Cancelled,
}

#[derive(Clone, Debug)]
struct TaskRecord {
    future_id: FutureId,
    generation: u64,
    read: bool,
    claimed: bool,
    completion_order: Option<u64>,
}

#[derive(Debug)]
struct ServiceState {
    next_future: u64,
    next_task: u64,
    next_completion: u64,
    futures: HashMap<FutureId, FutureState>,
    tasks: HashMap<TaskId, TaskRecord>,
    pool_generation: u64,
    pool_open: bool,
}

/// Root-scoped serial execution service.
///
/// It is the correctness backend used when no process/worker scheduler is
/// composed. Handles and state transitions are real; execution placement is
/// deliberately delegated to later scheduler adapters.
#[derive(Debug)]
pub struct RuntimeExecutionService {
    scope_id: ExecutionScopeId,
    state: Mutex<ServiceState>,
    cpu_turn: Option<CpuTaskTurn>,
}

#[derive(Clone, Copy, Debug)]
struct CpuTaskTurn {
    policy: CpuCooperationPolicy,
    schedule: fn() -> crate::context::RuntimeServiceFuture<Result<(), ExecutionServiceError>>,
}

impl RuntimeExecutionService {
    pub fn new() -> Self {
        let nonce = NEXT_SERVICE_NONCE.fetch_add(1, Ordering::Relaxed);
        Self {
            scope_id: ExecutionScopeId::derive(&[&nonce.to_be_bytes()]),
            cpu_turn: None,
            state: Mutex::new(ServiceState {
                next_future: 0,
                next_task: 0,
                next_completion: 0,
                futures: HashMap::new(),
                tasks: HashMap::new(),
                pool_generation: 1,
                pool_open: false,
            }),
        }
    }

    /// Select a real host task-turn ingress without composing another pool or
    /// scheduler. Ordinary serial execution retains its NeverYield default.
    pub fn with_cpu_task_turn(
        mut self,
        policy: CpuCooperationPolicy,
        turn: fn() -> crate::context::RuntimeServiceFuture<Result<(), ExecutionServiceError>>,
    ) -> Self {
        self.cpu_turn = Some(CpuTaskTurn {
            policy,
            schedule: turn,
        });
        self
    }

    fn validate_scope(&self, scope_id: ExecutionScopeId) -> Result<(), ExecutionServiceError> {
        if scope_id == self.scope_id {
            Ok(())
        } else {
            Err(ExecutionServiceError::ForeignScope)
        }
    }

    fn pool_snapshot(&self, generation: u64) -> PoolSnapshot {
        PoolSnapshot {
            handle: PoolHandle {
                id: PoolId::derive(&[self.scope_id.bytes(), b"serial"]),
                scope_id: self.scope_id,
                generation,
            },
            backend: PoolBackend::Serial,
            workers: 1,
            state: PoolState::Ready,
        }
    }
}

impl Default for RuntimeExecutionService {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeExecutionServices for RuntimeExecutionService {
    fn cpu_cooperation_policy(&self) -> CpuCooperationPolicy {
        self.cpu_turn
            .as_ref()
            .map_or(CpuCooperationPolicy::NeverYield, |turn| turn.policy)
    }

    fn cooperate_cpu(
        &self,
    ) -> crate::context::RuntimeServiceFuture<Result<(), ExecutionServiceError>> {
        match self.cpu_turn {
            Some(turn) => (turn.schedule)(),
            None => Box::pin(async {
                Err(ExecutionServiceError::Infrastructure(
                    "CPU task-turn scheduling is unavailable in this execution adapter".into(),
                ))
            }),
        }
    }

    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        let state = self.state.try_lock().ok()?;
        for future in state.futures.values() {
            match future {
                FutureState::Deferred(call) => call.invocation.visit_retained_values(visitor)?,
                FutureState::Completed(Ok(sequence)) => {
                    for value in sequence.value().iter() {
                        visitor(value)?;
                    }
                }
                // The caller owns the moved invocation while Running; its
                // execution frame must participate in the same root scan.
                FutureState::Running | FutureState::Completed(Err(_)) | FutureState::Cancelled => {}
            }
        }
        Some(())
    }

    fn scope_id(&self) -> ExecutionScopeId {
        self.scope_id
    }

    fn current_pool(&self) -> Result<Option<PoolSnapshot>, ExecutionServiceError> {
        let state = self.state.lock().expect("execution service state poisoned");
        Ok(state
            .pool_open
            .then(|| self.pool_snapshot(state.pool_generation)))
    }

    fn ensure_pool(&self, request: PoolRequest) -> Result<PoolSnapshot, ExecutionServiceError> {
        if request.workers.is_some_and(|workers| workers != 1)
            || request
                .backend
                .is_some_and(|backend| backend != PoolBackend::Serial)
        {
            return Err(ExecutionServiceError::Failed(
                "the serial execution backend provides exactly one worker".into(),
            ));
        }
        let mut state = self.state.lock().expect("execution service state poisoned");
        state.pool_open = true;
        Ok(self.pool_snapshot(state.pool_generation))
    }

    fn close_pool(&self, pool: &PoolHandle) -> Result<(), ExecutionServiceError> {
        self.validate_scope(pool.scope_id)?;
        let mut state = self.state.lock().expect("execution service state poisoned");
        if !state.pool_open || pool.generation != state.pool_generation {
            return Err(ExecutionServiceError::UnknownHandle);
        }
        state.pool_open = false;
        state.pool_generation = state.pool_generation.wrapping_add(1);
        drop(state);
        self.drain_scope(CancellationReason::User);
        Ok(())
    }

    fn create_future(&self, call: DeferredCall) -> Result<FutureHandle, ExecutionServiceError> {
        let requested_outputs = u16::try_from(call.invocation.requested_outputs())
            .map_err(|_| ExecutionServiceError::InvalidOutputContract)?;
        let mut state = self.state.lock().expect("execution service state poisoned");
        let sequence = state.next_future;
        state.next_future = state.next_future.wrapping_add(1);
        let id = FutureId::derive(&[self.scope_id.bytes(), &sequence.to_be_bytes()]);
        state
            .futures
            .insert(id, FutureState::Deferred(Box::new(call)));
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
        let mut state = self.state.lock().expect("execution service state poisoned");
        match state.futures.get(&future.id) {
            Some(FutureState::Deferred(call))
                if !state.tasks.values().any(|task| task.future_id == future.id) =>
            {
                call.require_spawn_admission(&future.outputs)?;
            }
            Some(_) => {
                return Err(ExecutionServiceError::Failed(
                    "future has already been scheduled".into(),
                ));
            }
            None => return Err(ExecutionServiceError::UnknownHandle),
        }
        let sequence = state.next_task;
        state.next_task = state.next_task.wrapping_add(1);
        let id = TaskId::derive(&[self.scope_id.bytes(), &sequence.to_be_bytes()]);
        let generation = 1;
        state.tasks.insert(
            id,
            TaskRecord {
                future_id: future.id,
                generation,
                read: false,
                claimed: false,
                completion_order: Some(sequence),
            },
        );
        Ok(TaskHandle {
            id,
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
        let state = self.state.lock().expect("execution service state poisoned");
        let stored_state = state
            .futures
            .get(&future.id)
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        Ok(ExecutionHandleSnapshot {
            state: future_state(stored_state),
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
        let state = self.state.lock().expect("execution service state poisoned");
        let record = state
            .tasks
            .get(&task.id)
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        if record.generation != task.generation {
            return Err(ExecutionServiceError::UnknownHandle);
        }
        let handle_state = state
            .futures
            .get(&record.future_id)
            .map(future_state)
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        Ok(ExecutionHandleSnapshot {
            state: handle_state,
            outputs: task.outputs.clone(),
            read: record.read,
        })
    }

    fn claim_next_task_result(
        &self,
        tasks: &[runmat_value::SharedTaskHandle],
    ) -> Result<TaskResultClaim, ExecutionServiceError> {
        let mut state = self.state.lock().expect("execution service state poisoned");
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
        let mut state = self.state.lock().expect("execution service state poisoned");
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
        if let Value::Job(handle) = &value {
            let value = self.await_job(handle)?;
            let sequence = runmat_value::ValueSequence::single(value)
                .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
            return Ok(AwaitAction::Completed(super::RootedValueSequence::new(
                sequence,
            )?));
        }
        let future = match value {
            Value::Future(handle) => super::FutureExecutionHandle::from(handle),
            Value::Task(task) => {
                self.validate_scope(task.scope_id)?;
                let state = self.state.lock().expect("execution service state poisoned");
                let record = state
                    .tasks
                    .get(&task.id)
                    .ok_or(ExecutionServiceError::UnknownHandle)?;
                if record.generation != task.generation {
                    return Err(ExecutionServiceError::UnknownHandle);
                }
                super::FutureExecutionHandle::from_task(task, record.future_id)
            }
            value => return Ok(AwaitAction::Passthrough(value)),
        };
        self.validate_scope(future.scope_id)?;
        let mut state = self.state.lock().expect("execution service state poisoned");
        let record = state
            .futures
            .get_mut(&future.id)
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        match record {
            FutureState::Deferred(_) => {
                let FutureState::Deferred(call) = std::mem::replace(record, FutureState::Running)
                else {
                    unreachable!("deferred state matched while holding its owner lock")
                };
                Ok(AwaitAction::ExecuteFuture {
                    handle: future,
                    call,
                })
            }
            FutureState::Running => Err(ExecutionServiceError::Failed(
                "execution is already being awaited".to_string(),
            )),
            FutureState::Completed(result) => {
                let result = result.clone();
                result.map(|value| AwaitAction::Completed(value.clone()))
            }
            FutureState::Cancelled => Err(ExecutionServiceError::Cancelled),
        }
    }

    fn complete_future(
        &self,
        future: super::FutureCompletionRef<'_>,
        result: Result<super::RootedValueSequence, ExecutionServiceError>,
    ) -> Result<(), ExecutionServiceError> {
        self.validate_scope(future.scope_id)?;
        let mut state = self.state.lock().expect("execution service state poisoned");
        let record = state
            .futures
            .get_mut(&future.id)
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        if !matches!(record, FutureState::Running) {
            return Err(ExecutionServiceError::UnknownHandle);
        }
        *record = FutureState::Completed(result);
        record_completion(&mut state, future.id);
        Ok(())
    }

    fn cancel(
        &self,
        value: &Value,
        _reason: CancellationReason,
    ) -> Result<(), ExecutionServiceError> {
        let mut state = self.state.lock().expect("execution service state poisoned");
        let future_id = match value {
            Value::Future(handle) => {
                self.validate_scope(handle.scope_id)?;
                handle.id
            }
            Value::Task(handle) => {
                self.validate_scope(handle.scope_id)?;
                let record = state
                    .tasks
                    .get(&handle.id)
                    .ok_or(ExecutionServiceError::UnknownHandle)?;
                if record.generation != handle.generation {
                    return Err(ExecutionServiceError::UnknownHandle);
                }
                record.future_id
            }
            _ => return Err(ExecutionServiceError::UnknownHandle),
        };
        let record = state
            .futures
            .get_mut(&future_id)
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        // The future owns the single terminal result for every task alias.
        if matches!(record, FutureState::Completed(_) | FutureState::Cancelled) {
            return Ok(());
        }
        let retired = std::mem::replace(record, FutureState::Cancelled);
        record_completion(&mut state, future_id);
        drop(state);
        drop(retired);
        Ok(())
    }

    fn drain_scope(&self, _reason: CancellationReason) {
        let mut state = self.state.lock().expect("execution service state poisoned");
        let mut completed = Vec::new();
        let mut retired = Vec::new();
        for future in state.futures.values_mut() {
            if !matches!(future, FutureState::Completed(_)) {
                let prior = std::mem::replace(future, FutureState::Cancelled);
                if let FutureState::Deferred(call) = prior {
                    retired.push(call);
                }
            }
        }
        for task in state.tasks.values() {
            completed.push(task.future_id);
        }
        for future_id in completed {
            record_completion(&mut state, future_id);
        }
        drop(state);
        // A deferred argument or callable generation may own a Foreign lease.
        // Its release callback is allowed to reenter the execution service, so
        // retire deferred calls only after releasing the service-state lock.
        drop(retired);
    }
}

fn record_completion(state: &mut ServiceState, future_id: FutureId) {
    let Some(task_id) = state
        .tasks
        .iter()
        .find_map(|(task_id, task)| (task.future_id == future_id).then_some(*task_id))
    else {
        return;
    };
    if state
        .tasks
        .get(&task_id)
        .is_some_and(|task| task.completion_order.is_some())
    {
        return;
    }
    let order = state.next_completion;
    state.next_completion = order.wrapping_add(1);
    state
        .tasks
        .get_mut(&task_id)
        .expect("task was resolved above")
        .completion_order = Some(order);
}

fn future_state(state: &FutureState) -> ExecutionHandleState {
    match state {
        FutureState::Deferred(_) => ExecutionHandleState::Deferred,
        FutureState::Running => ExecutionHandleState::Running,
        FutureState::Completed(Ok(_)) => ExecutionHandleState::Finished,
        FutureState::Completed(Err(_)) => ExecutionHandleState::Failed,
        FutureState::Cancelled => ExecutionHandleState::Cancelled,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::RootedValueSequence;
    use runmat_types::{ForeignAffinity, ForeignLifetime, ForeignOwnership, ForeignTypeIdentity};
    use runmat_value::{ForeignRef, ForeignResourceKey, ForeignResourceRelease};
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;
    use std::sync::Arc;

    thread_local! {
        static REENTRANT_DEFERRED_SERVICE: RefCell<Option<Rc<RuntimeExecutionService>>> =
            const { RefCell::new(None) };
        static REENTRANT_DEFERRED_RELEASES: Cell<usize> = const { Cell::new(0) };
        static REENTRANT_DEFERRED_OBSERVED_UNLOCKED: Cell<bool> = const { Cell::new(false) };
    }

    impl RuntimeExecutionService {
        fn retained_values(&self) -> Option<Vec<Value>> {
            let mut values = Vec::new();
            self.visit_retained_values(&mut |value| {
                values.push(value.clone());
                Some(())
            })?;
            Some(values)
        }
    }

    impl DeferredInvocation {
        fn retained_values(&self) -> Vec<Value> {
            let mut values = Vec::new();
            self.visit_retained_values(&mut |value| {
                values.push(value.clone());
                Some(())
            })
            .unwrap();
            values
        }
    }

    fn deferred_call(function: usize, arguments: Vec<Value>) -> DeferredCall {
        DeferredCall {
            spawn_safety: runmat_types::SpawnSafetyFact::RequiresIsolation,
            invocation: DeferredInvocation::callable(
                crate::call::descriptor::CallableDescriptor::resolved(
                    runmat_types::CallableIdentity::BoundFunction(runmat_types::FunctionId(
                        function,
                    )),
                    arguments,
                    1,
                    runmat_types::CallableFallbackPolicy::None,
                    crate::call::descriptor::CallableCallKind::Direct,
                ),
            ),
            retry: runmat_execution::RetryPolicy::Never,
            program_revision: None,
            environment: runmat_execution::EnvironmentSnapshot::default(),
            console: None,
            capabilities: Default::default(),
            program: DeferredProgram::Local,
        }
    }

    #[derive(Debug)]
    struct ReentrantDeferredRelease;

    impl ForeignResourceRelease for ReentrantDeferredRelease {
        fn release(&self, _key: &ForeignResourceKey) {
            REENTRANT_DEFERRED_RELEASES.set(REENTRANT_DEFERRED_RELEASES.get() + 1);
            let service = REENTRANT_DEFERRED_SERVICE.with(|current| current.borrow().clone());
            let unlocked = service.is_some_and(|service| service.state.try_lock().is_ok());
            REENTRANT_DEFERRED_OBSERVED_UNLOCKED.set(unlocked);
        }
    }

    fn reentrant_foreign(release: Arc<ReentrantDeferredRelease>) -> Value {
        Value::Foreign(ForeignRef::managed(
            ForeignResourceKey {
                host_identity: "deferred-release-probe".into(),
                handle: 7,
                generation: 1,
            },
            ForeignTypeIdentity {
                family: "native-ffi".into(),
                name: "test.deferred".into(),
                version: 1,
            },
            ForeignOwnership::Shared,
            ForeignAffinity::AnyThread,
            ForeignLifetime::Session,
            release,
        ))
    }

    #[test]
    fn draining_scope_retires_deferred_values_after_unlocking_service_state() {
        let service = Rc::new(RuntimeExecutionService::new());
        REENTRANT_DEFERRED_SERVICE.with(|current| {
            *current.borrow_mut() = Some(Rc::clone(&service));
        });
        REENTRANT_DEFERRED_RELEASES.set(0);
        REENTRANT_DEFERRED_OBSERVED_UNLOCKED.set(false);
        service
            .create_future(deferred_call(
                1,
                vec![reentrant_foreign(Arc::new(ReentrantDeferredRelease))],
            ))
            .unwrap();

        service.drain_scope(CancellationReason::Shutdown);

        assert_eq!(REENTRANT_DEFERRED_RELEASES.get(), 1);
        assert!(REENTRANT_DEFERRED_OBSERVED_UNLOCKED.get());
        REENTRANT_DEFERRED_SERVICE.with(|current| {
            current.borrow_mut().take();
        });
    }

    #[test]
    fn retained_values_cover_deferred_callable_and_completed_outputs() {
        let service = RuntimeExecutionService::new();
        let mut call = deferred_call(1, vec![Value::String("argument".into())]);
        let DeferredInvocation::Callable(descriptor) = &mut call.invocation else {
            unreachable!()
        };
        descriptor.target =
            crate::call::descriptor::CallableTarget::FevalForward(Value::String("callable".into()));
        let future = service.create_future(call).unwrap();
        assert_eq!(
            service.retained_values(),
            Some(vec![
                Value::String("argument".into()),
                Value::String("callable".into())
            ])
        );
        let action = service
            .begin_await(Value::Future(future.clone().into()))
            .unwrap();
        let AwaitAction::ExecuteFuture { call, .. } = action else {
            panic!("execute deferred call")
        };
        assert_eq!(call.invocation.retained_values().len(), 2);
        service
            .complete_future(
                (&future).into(),
                Ok(RootedValueSequence::new(
                    runmat_value::ValueSequence::single(Value::String("output".into())).unwrap(),
                )
                .unwrap()),
            )
            .unwrap();
        assert_eq!(
            service.retained_values(),
            Some(vec![Value::String("output".into())])
        );
        service
            .cancel(&Value::Future(future.into()), CancellationReason::User)
            .unwrap();
        // A late cancellation cannot discard a published, refetchable result.
        assert_eq!(
            service.retained_values(),
            Some(vec![Value::String("output".into())])
        );
    }

    #[test]
    fn retained_values_fail_closed_during_service_mutation() {
        let service = RuntimeExecutionService::new();
        let _state = service.state.lock().unwrap();
        assert!(service.retained_values().is_none());
    }

    #[test]
    fn services_reject_handles_from_another_scope() {
        let first = RuntimeExecutionService::new();
        let second = RuntimeExecutionService::new();
        let future = first.create_future(deferred_call(1, vec![])).unwrap();
        assert_eq!(
            second.spawn(&future),
            Err(ExecutionServiceError::ForeignScope)
        );
    }

    #[test]
    fn future_task_lifecycle_is_backed_by_service_state() {
        let service = RuntimeExecutionService::new();
        let mut rejected_call = deferred_call(7, vec![Value::Num(3.0)]);
        rejected_call.spawn_safety = runmat_types::SpawnSafetyFact::NotSpawnSafe {
            reason: runmat_types::SpawnSafetyReason::MutableLexicalCapture,
        };
        let rejected = service.create_future(rejected_call).unwrap();
        assert_eq!(
            service.spawn(&rejected),
            Err(ExecutionServiceError::SpawnUnsafe(
                runmat_types::SpawnSafetyReason::MutableLexicalCapture,
            ))
        );
        assert!(service.state.lock().unwrap().tasks.is_empty());
        assert_eq!(
            service.inspect_future(&rejected).unwrap().state,
            ExecutionHandleState::Deferred
        );
        // Local awaiting does not request isolated scheduling or discard the
        // original lexical owner. Refusing spawn must leave that path intact.
        assert!(matches!(
            service.begin_await(Value::Future(rejected.into())),
            Ok(AwaitAction::ExecuteFuture { .. })
        ));
        let future = service
            .create_future(deferred_call(7, vec![Value::Num(3.0)]))
            .unwrap();
        let mut altered = future.clone();
        altered.outputs.requested_outputs += 1;
        assert_eq!(
            service.spawn(&altered),
            Err(ExecutionServiceError::InvalidOutputContract)
        );
        assert!(service.state.lock().unwrap().tasks.is_empty());
        let task = service.spawn(&future).unwrap();
        let action = service
            .begin_await(Value::Task(task.clone().into()))
            .unwrap();
        let AwaitAction::ExecuteFuture { handle, call } = action else {
            panic!("expected deferred execution");
        };
        assert_eq!(handle.completion_ref(), (&future).into());
        let DeferredInvocation::Callable(descriptor) = call.invocation else {
            panic!("expected callable deferred invocation")
        };
        assert!(
            matches!(&descriptor.target, crate::call::descriptor::CallableTarget::Resolved { identity, .. }
            if matches!(identity.as_identity(), runmat_types::CallableIdentity::BoundFunction(runmat_types::FunctionId(7))))
        );
        service
            .complete_future(
                (&future).into(),
                Ok(RootedValueSequence::new(
                    runmat_value::ValueSequence::single(Value::Num(9.0)).unwrap(),
                )
                .unwrap()),
            )
            .unwrap();
        let completed = service
            .begin_await(Value::Future(future.clone().into()))
            .unwrap();
        for alias in [
            Value::Future(future.clone().into()),
            Value::Task(task.clone().into()),
        ] {
            service.cancel(&alias, CancellationReason::User).unwrap();
            assert_eq!(service.begin_await(alias), Ok(completed.clone()));
        }
        service.drain_scope(CancellationReason::Shutdown);
        assert_eq!(
            service.inspect_future(&future).unwrap().state,
            ExecutionHandleState::Finished
        );
        assert_eq!(
            service.inspect_task(&task).unwrap().state,
            ExecutionHandleState::Finished
        );
        assert_eq!(
            service.begin_await(Value::Task(task.into())),
            Ok(AwaitAction::Completed(
                RootedValueSequence::new(
                    runmat_value::ValueSequence::single(Value::Num(9.0)).unwrap()
                )
                .unwrap()
            ))
        );
        let failed = service.create_future(deferred_call(7, vec![])).unwrap();
        let failed_task = service.spawn(&failed).unwrap();
        assert!(matches!(
            service
                .begin_await(Value::Task(failed_task.clone().into()))
                .unwrap(),
            AwaitAction::ExecuteFuture { .. }
        ));
        let failure = ExecutionServiceError::Failed("original task failure".into());
        service
            .complete_future((&failed).into(), Err(failure.clone()))
            .unwrap();
        for alias in [
            Value::Future(failed.clone().into()),
            Value::Task(failed_task.clone().into()),
        ] {
            service.cancel(&alias, CancellationReason::User).unwrap();
            service.drain_scope(CancellationReason::Shutdown);
            assert_eq!(service.begin_await(alias), Err(failure.clone()));
        }
        assert_eq!(
            service.inspect_task(&failed_task).unwrap().state,
            ExecutionHandleState::Failed
        );

        // A genuinely active cancellation remains terminal through both aliases,
        // and late execution completion cannot resurrect it.
        let active = service.create_future(deferred_call(7, vec![])).unwrap();
        let active_task = service.spawn(&active).unwrap();
        assert!(matches!(
            service
                .begin_await(Value::Task(active_task.clone().into()))
                .unwrap(),
            AwaitAction::ExecuteFuture { .. }
        ));
        service
            .cancel(
                &Value::Task(active_task.clone().into()),
                CancellationReason::User,
            )
            .unwrap();
        assert!(service
            .complete_future((&active).into(), Err(failure))
            .is_err());
        for alias in [
            Value::Future(active.into()),
            Value::Task(active_task.clone().into()),
        ] {
            service
                .cancel(&alias, CancellationReason::Shutdown)
                .unwrap();
            assert_eq!(
                service.begin_await(alias),
                Err(ExecutionServiceError::Cancelled)
            );
        }
        assert_eq!(
            service.inspect_task(&active_task).unwrap().state,
            ExecutionHandleState::Cancelled
        );
    }

    #[test]
    fn await_handoff_moves_the_original_environment_without_cloning() {
        let environment =
            runmat_execution::EnvironmentSnapshot::new(vec![("NAME".into(), "original".into())])
                .unwrap();
        let environment_pointer = environment.entries()[0].1.as_ptr();
        let service = RuntimeExecutionService::new();
        let mut call = deferred_call(7, vec![Value::Num(3.0)]);
        call.environment = environment;
        let future = service.create_future(call).unwrap();
        let AwaitAction::ExecuteFuture { handle, call } = service
            .begin_await(Value::Future(future.clone().into()))
            .unwrap()
        else {
            panic!("original future did not enter execution")
        };
        assert_eq!(
            handle.completion_ref(),
            crate::execution::FutureCompletionRef::from(&future)
        );
        assert_eq!(
            call.environment.entries()[0].1.as_ptr(),
            environment_pointer
        );
        assert!(service.begin_await(Value::Future(future.into())).is_err());
    }

    #[test]
    fn completed_future_preserves_zero_one_and_many_output_cardinalities() {
        for values in [
            Vec::new(),
            vec![Value::Num(1.0)],
            vec![Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)],
        ] {
            let service = RuntimeExecutionService::new();
            let future = service
                .create_future(deferred_call(1, Vec::new()))
                .expect("create future");
            let action = service
                .begin_await(Value::Future(future.clone().into()))
                .expect("start future");
            assert!(matches!(action, AwaitAction::ExecuteFuture { .. }));
            service
                .complete_future(
                    (&future).into(),
                    Ok(RootedValueSequence::new(
                        runmat_value::ValueSequence::comma_separated(values.clone()).unwrap(),
                    )
                    .unwrap()),
                )
                .expect("complete future");
            let AwaitAction::Completed(sequence) = service
                .begin_await(Value::Future(future.into()))
                .expect("consume completion")
            else {
                panic!("expected completed future");
            };
            assert_eq!(sequence.value().as_slice(), values.as_slice());
        }
    }

    #[test]
    fn completed_future_state_roots_handle_outputs_until_service_discard() {
        runmat_gc::gc_test_context(|| {
            let service = RuntimeExecutionService::new();
            let future = service
                .create_future(deferred_call(1, Vec::new()))
                .expect("create future");
            assert!(matches!(
                service
                    .begin_await(Value::Future(future.clone().into()))
                    .expect("start future"),
                AwaitAction::ExecuteFuture { .. }
            ));

            let initial = runmat_gc::gc_allocate_rooted(Value::String("retained".into()))
                .expect("allocate rooted payload");
            let handle = initial.handle();
            let output = Value::HandleObject(runmat_value::HandleRef {
                class_name: "CompletionHandle".into(),
                target: handle,
                valid: true,
            });
            service
                .complete_future(
                    (&future).into(),
                    Ok(RootedValueSequence::new(
                        runmat_value::ValueSequence::single(output).expect("valid output"),
                    )
                    .expect("root output")),
                )
                .expect("complete future");
            initial.unroot().expect("transfer root to service state");

            runmat_gc::gc_collect_major().expect("collect with stored completion");
            assert_eq!(
                runmat_gc::gc_clone_value(&handle).expect("service retains completion root"),
                Value::String("retained".into())
            );

            drop(service);
            runmat_gc::gc_collect_major().expect("collect after service discard");
            assert!(runmat_gc::gc_clone_value(&handle).is_err());
        });
    }

    #[test]
    fn completed_await_owns_output_after_state_cancel_and_observer_collection() {
        thread_local! {
            static CLAIMED_RESULT_HANDLE: std::cell::Cell<Option<runmat_gc::GcHandle>> = const {
                std::cell::Cell::new(None)
            };
        }
        runmat_gc::gc_test_context(|| {
            let service = RuntimeExecutionService::new();
            let future = service
                .create_future(deferred_call(1, Vec::new()))
                .expect("create future");
            assert!(matches!(
                service.begin_await(Value::Future(future.clone().into())),
                Ok(AwaitAction::ExecuteFuture { .. })
            ));
            let initial = runmat_gc::gc_allocate_rooted(Value::String("awaited".into()))
                .expect("allocate managed result");
            let handle = initial.handle();
            let result = RootedValueSequence::new(
                runmat_value::ValueSequence::single(Value::HandleObject(runmat_value::HandleRef {
                    class_name: "AwaitedHandle".into(),
                    target: handle,
                    valid: true,
                }))
                .expect("single result"),
            )
            .expect("root result");
            service
                .complete_future((&future).into(), Ok(result))
                .expect("publish completion");
            initial.unroot().expect("transfer initial root");

            let AwaitAction::Completed(claimed) = service
                .begin_await(Value::Future(future.clone().into()))
                .expect("claim completed result")
            else {
                panic!("expected completed result")
            };
            let control =
                runmat_gc::gc_allocate_rooted(Value::Num(1.0)).expect("allocate observer trigger");
            let control_handle = control.handle();
            let control_identity = (control_handle.addr(), control_handle.epoch());
            CLAIMED_RESULT_HANDLE.with(|slot| slot.set(Some(handle)));
            let observed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let observed_callback = std::sync::Arc::clone(&observed);
            let _observer =
                runmat_gc::RootRetirementObserver::register(std::sync::Arc::new(move |event| {
                    let runmat_gc::RootRetirement::Explicit(trigger) = event else {
                        return;
                    };
                    let Some(handle) = CLAIMED_RESULT_HANDLE.with(std::cell::Cell::get) else {
                        return;
                    };
                    if (trigger.addr(), trigger.epoch()) != control_identity
                        || observed_callback.swap(true, std::sync::atomic::Ordering::SeqCst)
                    {
                        return;
                    }
                    runmat_gc::gc_collect_major().expect("collect during state cancellation");
                    assert_eq!(
                        runmat_gc::gc_clone_value(&handle)
                            .expect("claimed result is rooted during cancellation"),
                        Value::String("awaited".into())
                    );
                }))
                .expect("register collecting observer");
            service
                .cancel(&Value::Future(future.into()), CancellationReason::User)
                .expect("completed cancellation preserves the terminal result");
            // Completed aliases retain their original result. Retire the
            // service owner, not that immutable outcome, while the claimant
            // independently keeps its managed result rooted.
            drop(service);
            control.unroot().expect("trigger collecting observer");
            assert!(observed.load(std::sync::atomic::Ordering::SeqCst));
            runmat_gc::gc_collect_major().expect("collect while claimant owns result");
            assert_eq!(
                runmat_gc::gc_clone_value(&handle).expect("claimed result remains live"),
                Value::String("awaited".into())
            );
            drop(claimed);
            runmat_gc::gc_collect_major().expect("collect after claimant release");
            assert!(runmat_gc::gc_clone_value(&handle).is_err());
            CLAIMED_RESULT_HANDLE.with(|slot| slot.set(None));
        });
    }

    #[test]
    fn cloned_invocation_context_inherits_exact_service() {
        let service: std::rc::Rc<dyn RuntimeExecutionServices> =
            std::rc::Rc::new(RuntimeExecutionService::new());
        let parent = crate::context::RuntimeContext::new(service);
        let nested = parent.clone();
        assert_eq!(parent.execution().scope_id(), nested.execution().scope_id());
    }

    #[test]
    fn independently_created_thread_sessions_have_distinct_scopes() {
        let first = std::thread::spawn(|| RuntimeExecutionService::new().scope_id())
            .join()
            .unwrap();
        let second = std::thread::spawn(|| RuntimeExecutionService::new().scope_id())
            .join()
            .unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn scope_drain_cancels_unfinished_children() {
        let service = RuntimeExecutionService::new();
        let future = service.create_future(deferred_call(1, vec![])).unwrap();
        service.drain_scope(CancellationReason::Shutdown);
        assert_eq!(
            service.begin_await(Value::Future(future.into())),
            Err(ExecutionServiceError::Cancelled)
        );
    }

    #[test]
    fn serial_pool_lifecycle_fences_stale_handles() {
        let service = RuntimeExecutionService::new();
        assert!(service.current_pool().unwrap().is_none());
        let first = service.ensure_pool(PoolRequest::automatic()).unwrap();
        assert_eq!(first.backend, PoolBackend::Serial);
        assert_eq!(first.workers, 1);
        service.close_pool(&first.handle).unwrap();
        assert!(service.current_pool().unwrap().is_none());
        assert_eq!(
            service.close_pool(&first.handle),
            Err(ExecutionServiceError::UnknownHandle)
        );
        let reopened = service.ensure_pool(PoolRequest::automatic()).unwrap();
        assert_ne!(first.handle.generation, reopened.handle.generation);
    }
}
