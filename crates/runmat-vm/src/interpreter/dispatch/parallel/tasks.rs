use crate::bytecode::program::ExecutionContext;
use crate::program_execution::callable_admission::{
    bind_selected_semantic_target, callable_execution_contract, CallableExecutionContract,
};
use crate::FunctionRegistry;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

use super::{execution_error, pop};

mod outputs;
use outputs::{fetch_next_outputs, root_join_outputs, select_outputs, split_outputs};

pub(super) fn schedule_feval(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    context: &ExecutionContext,
    function_registry: &FunctionRegistry,
    call_arg_count: usize,
    on_all: bool,
) -> Result<(), RuntimeError> {
    let mut values = crate::call::builtins::collect_call_args(stack, call_arg_count)?;
    let explicit_pool = matches!(values.first(), Some(Value::Pool(_)));
    let pool = if explicit_pool {
        pool_handle(values.remove(0))?
    } else {
        context
            .runtime
            .execution()
            .ensure_pool(runmat_execution::PoolRequest::automatic())
            .map_err(execution_error)?
            .handle
    };
    if values.len() < 2 {
        return Err(crate::interpreter::errors::mex(
            "NotEnoughInputs",
            "parallel evaluation requires a function handle and output count",
        ));
    }
    let callable = values.remove(0);
    let requested_outputs = requested_outputs(values.remove(0))?;
    let descriptor =
        resolved_feval_descriptor(callable, values, requested_outputs, function_registry)?;
    if !on_all {
        let future = create_future(context, descriptor, function_registry)?;
        let task = context
            .runtime
            .execution()
            .spawn_in_context(&future, Some(&pool), &context.runtime)
            .map_err(execution_error)?;
        stack.try_push(runmat_runtime::parallel::future::wrap_task(task))?;
        return Ok(());
    }
    let worker_count = context
        .runtime
        .execution()
        .inspect_pool(&pool)
        .map_err(execution_error)?
        .workers;
    let mut tasks = Vec::with_capacity(worker_count as usize);
    for _ in 0..worker_count {
        let future = create_future(context, descriptor.clone(), function_registry)?;
        tasks.push(
            context
                .runtime
                .execution()
                .spawn_in_context(&future, Some(&pool), &context.runtime)
                .map_err(execution_error)?,
        );
    }
    stack.try_push(
        runmat_runtime::parallel::future::wrap_on_all_tasks(tasks)
            .map_err(|error| crate::interpreter::errors::mex("FutureArray", &error))?,
    )?;
    Ok(())
}

pub(super) fn spawn(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    context: &ExecutionContext,
    explicit_pool: bool,
) -> Result<(), RuntimeError> {
    let Value::Future(future) = pop(stack, "spawn instruction expected a future on the stack")?
    else {
        return Err(crate::interpreter::errors::mex(
            "SpawnOperandInvalid",
            "spawn expects a lazy future",
        ));
    };
    let task = if explicit_pool {
        let pool = pool_handle(pop(stack, "pool spawn expected a pool on the stack")?)?;
        context
            .runtime
            .execution()
            .spawn_in_context(&future, Some(&pool), &context.runtime)
            .map_err(execution_error)?
    } else {
        context
            .runtime
            .execution()
            .spawn_in_context(&future, None, &context.runtime)
            .map_err(execution_error)?
    };
    stack.try_push(if explicit_pool {
        runmat_runtime::parallel::future::wrap_task(task)
    } else {
        Value::Task(task.into())
    })?;
    Ok(())
}

pub(super) fn create_future(
    context: &ExecutionContext,
    descriptor: runmat_runtime::call::descriptor::CallableDescriptor,
    function_registry: &FunctionRegistry,
) -> Result<runmat_execution::FutureHandle, RuntimeError> {
    runmat_runtime::execution::validate_spawn_captures(&descriptor.args)?;
    let CallableExecutionContract {
        capabilities,
        spawn_safety,
    } = callable_execution_contract(&descriptor, function_registry)?;
    let mut invocation = runmat_runtime::execution::DeferredInvocation::callable(descriptor);
    let graph_required =
        runmat_runtime::execution::value_codec::deferred_invocation_requires_graph(&invocation);
    let builtin = matches!(
        &invocation,
        runmat_runtime::execution::DeferredInvocation::Callable(descriptor)
            if matches!(&descriptor.target, runmat_runtime::call::descriptor::CallableTarget::Resolved { identity, .. }
                if matches!(identity.as_identity(), runmat_hir::CallableIdentity::Builtin(_)))
    );
    let program = if !context.runtime.execution().requires_program_capture() {
        runmat_runtime::execution::DeferredProgram::Local
    } else if builtin && !graph_required && context.graph_program.is_none() {
        // A builtin has typed catalog authority but no V3 function body. Keep
        // its existing direct V2 worker route instead of inventing a graph
        // callable owner or rejecting previously supported async builtins.
        let bytes = crate::encode_interpreter_program_v2(function_registry).map_err(|error| {
            crate::interpreter::errors::mex(
                "ExecutionProgram",
                &format!("failed to capture the exact async builtin program: {error}"),
            )
        })?;
        runmat_runtime::execution::DeferredProgram::InterpreterV2(std::sync::Arc::from(bytes))
    } else {
        let selected = context.graph_program.as_ref().ok_or_else(|| {
            crate::interpreter::errors::mex(
                "ExecutionProgram",
                "async worker execution has no selected V3 executable authority",
            )
        })?;
        runmat_runtime::execution::DeferredProgram::SelectedGraph(
            selected.clone().with_validation_workspace(
                runmat_runtime::execution::value_codec::ValidationWorkspaceOwner::capture(
                    &context.runtime,
                ),
            ),
        )
    };
    if matches!(
        &program,
        runmat_runtime::execution::DeferredProgram::SelectedGraph(_)
    ) && builtin
    {
        let runmat_runtime::execution::DeferredInvocation::Callable(descriptor) = invocation else {
            unreachable!("future admission created a callable invocation")
        };
        let descriptor = *descriptor;
        let runmat_runtime::call::descriptor::CallableTarget::Resolved {
            identity,
            fallback_policy: runmat_hir::CallableFallbackPolicy::None,
        } = descriptor.target
        else {
            return Err(crate::interpreter::errors::mex(
                "ExecutionProgram",
                "graph builtin requires an exact selected catalog target",
            ));
        };
        let runmat_hir::CallableIdentity::Builtin(name) = identity.as_identity() else {
            return Err(crate::interpreter::errors::mex(
                "ExecutionProgram",
                "graph builtin requires an exact selected catalog target",
            ));
        };
        invocation = runmat_runtime::execution::DeferredInvocation::Program {
            callable: runmat_execution::ProgramCallable::Builtin { name: name.clone() },
            context: runmat_execution::ProgramInvocationContext::Direct,
            arguments: descriptor.args,
            requested_outputs: descriptor.requested_outputs,
        };
    }
    // A local service may retain this lazy call across commands too. Capture
    // the selected body now, before a later command can replace its numeric ID.
    // Bare VM embeddings have no selected source graph and remain local-only.
    if !builtin && context.graph_program.is_some() {
        let runmat_runtime::execution::DeferredInvocation::Callable(descriptor) = &mut invocation
        else {
            unreachable!("future admission created a callable invocation")
        };
        bind_selected_semantic_target(descriptor, function_registry)?;
    }
    let program_revision = program
        .program_revision(context.runtime.program_revision())
        .cloned();
    context
        .runtime
        .execution()
        .create_future(runmat_runtime::execution::DeferredCall {
            spawn_safety,
            invocation,
            retry: runmat_execution::RetryPolicy::Never,
            program_revision,
            environment: runmat_runtime::execution::capture_deferred_environment(&context.runtime)?,
            console: context
                .runtime
                .service_ports()
                .host()
                .and_then(|host| host.console_limits()),
            capabilities,
            program,
        })
        .map_err(execution_error)
}

pub(super) fn semantic_descriptor(
    function: runmat_hir::FunctionId,
    requested_outputs: usize,
    arguments: Vec<Value>,
    binding: Option<runmat_value::BoundFunctionBinding>,
) -> runmat_runtime::call::descriptor::CallableDescriptor {
    let mut descriptor = runmat_runtime::call::descriptor::CallableDescriptor::resolved(
        runmat_hir::CallableIdentity::BoundFunction(function),
        arguments,
        requested_outputs,
        runmat_hir::CallableFallbackPolicy::None,
        runmat_runtime::call::descriptor::CallableCallKind::Direct,
    );
    if let Some(binding) = binding {
        descriptor.target = runmat_runtime::call::descriptor::CallableTarget::Bound(binding);
    }
    descriptor
}

fn resolved_feval_descriptor(
    callable: Value,
    arguments: Vec<Value>,
    requested_outputs: usize,
    function_registry: &FunctionRegistry,
) -> Result<runmat_runtime::call::descriptor::CallableDescriptor, RuntimeError> {
    let descriptor = runmat_runtime::call::descriptor::CallableDescriptor::from_feval_value(
        callable,
        arguments,
        requested_outputs,
        function_registry,
    );
    if matches!(
        &descriptor.target,
        runmat_runtime::call::descriptor::CallableTarget::FevalForward(_)
    ) {
        return Err(crate::interpreter::errors::mex(
            "InvalidParallelCallable",
            "parfeval expects a named, anonymous, or captured function handle",
        ));
    }
    Ok(descriptor)
}

fn requested_outputs(value: Value) -> Result<usize, RuntimeError> {
    match value {
        Value::Int(value) => value.try_to_usize().ok_or_else(|| {
            crate::interpreter::errors::mex(
                "InvalidOutputCount",
                "parfeval output count must be a nonnegative integer scalar",
            )
        }),
        Value::Num(value)
            if value.is_finite()
                && value >= 0.0
                && value.fract() == 0.0
                && value <= usize::MAX as f64 =>
        {
            Ok(value as usize)
        }
        _ => Err(crate::interpreter::errors::mex(
            "InvalidOutputCount",
            "parfeval output count must be a nonnegative integer scalar",
        )),
    }
}

pub(super) async fn await_value(
    context: &ExecutionContext,
    value: Value,
) -> Result<runmat_runtime::execution::RootedValue, RuntimeError> {
    await_sequence(context, value).await?.try_into_single()
}

async fn await_sequence(
    context: &ExecutionContext,
    value: Value,
) -> Result<runmat_runtime::execution::RootedValueSequence, RuntimeError> {
    use runmat_runtime::execution::AwaitAction;

    let mut value = runmat_runtime::parallel::future::execution_value(&value)
        .cloned()
        .unwrap_or(value);
    loop {
        match context
            .runtime
            .execution()
            .begin_await_in_context(value, &context.runtime)
            .map_err(execution_error)?
        {
            AwaitAction::Passthrough(value) => {
                let sequence = runmat_value::ValueSequence::single(value)
                    .map_err(runmat_runtime::sequence::sequence_error_to_runtime)?;
                return runmat_runtime::execution::RootedValueSequence::new(sequence)
                    .map_err(execution_error);
            }
            AwaitAction::Completed(sequence) => return Ok(sequence),
            AwaitAction::Pending(pending) => {
                if let Some(waiting) = context
                    .runtime
                    .execution()
                    .wait_pending(&pending, &context.runtime)
                    .map_err(execution_error)?
                {
                    waiting.await.map_err(execution_error)?;
                } else {
                    yield_once(context).await?;
                }
                value = pending;
            }
            AwaitAction::ExecuteFuture { handle, call } => {
                let mut running = runmat_runtime::execution::RunningFutureLease::new(
                    context.runtime.clone(),
                    handle,
                );
                let result = match &call.invocation {
                    runmat_runtime::execution::DeferredInvocation::Callable(descriptor) => {
                        match context
                            .runtime
                            .clone()
                            .with_invocation_environment(&call.environment)
                        {
                            Ok(invocation) => {
                                invocation
                                    .scope(crate::call::execute_callable_descriptor(
                                        &invocation,
                                        descriptor.as_ref().clone(),
                                    ))
                                    .await
                            }
                            Err(error) => Err(execution_error(
                                runmat_runtime::execution::ExecutionServiceError::Failed(
                                    error.to_string(),
                                ),
                            )),
                        }
                    }
                    runmat_runtime::execution::DeferredInvocation::Program { .. } => {
                        if matches!(
                            &call.program,
                            runmat_runtime::execution::DeferredProgram::SelectedGraph(_)
                        ) {
                            execute_selected_graph_on_driver(context, *call).await
                        } else {
                            crate::execute_deferred_program_in_context(
                                *call,
                                context.runtime.clone(),
                            )
                            .await
                            .map_err(execution_error)
                            .and_then(|sequence| {
                                runmat_runtime::execution::RootedValueSequence::new(sequence)
                                    .map_err(execution_error)
                            })
                        }
                    }
                };
                let (stored, result) = match result {
                    Ok(sequence) => (Ok(sequence.clone()), Ok(sequence)),
                    Err(error) => {
                        let original = runmat_runtime::runtime_error::RuntimeErrorOwner::new(error);
                        (
                            Err(runmat_runtime::execution::ExecutionServiceError::LocalizedRuntimeFailure(original.clone())),
                            Err(RuntimeError::from_shared_original(original)),
                        )
                    }
                };
                running.complete(stored).map_err(execution_error)?;
                return result;
            }
        }
    }
}

// The service transfers a selected call to the awaiter. The queued owner
// keeps its argument roots alive even when the driver has not polled it yet;
// field order retires the call before the root lease on every exit.
struct QueuedSelectedGraphCall {
    call: runmat_runtime::execution::DeferredCall,
    _argument_roots: runmat_runtime::execution::RootedValueLease,
}

impl QueuedSelectedGraphCall {
    async fn execute(
        self,
        runtime: runmat_runtime::context::RuntimeContext,
    ) -> Result<runmat_runtime::execution::RootedValueSequence, RuntimeError> {
        crate::program_execution::execute_deferred_graph_in_context(&self.call, runtime)
            .await
            .map_err(execution_error)
    }
}

async fn execute_selected_graph_on_driver(
    context: &ExecutionContext,
    call: runmat_runtime::execution::DeferredCall,
) -> Result<runmat_runtime::execution::RootedValueSequence, RuntimeError> {
    let runmat_runtime::execution::DeferredInvocation::Program { arguments, .. } = &call.invocation
    else {
        return Err(crate::interpreter::errors::mex(
            "ParallelTaskInvocation",
            "selected graph task requires a program invocation",
        ));
    };
    let driver = context.process_driver.as_ref().ok_or_else(|| {
        crate::interpreter::errors::mex(
            "ParallelProcessDriver",
            "selected graph task requires its invocation continuation owner",
        )
    })?;
    let argument_roots = runmat_runtime::execution::RootedValueLease::for_value_slices_in_context(
        &[arguments.as_slice()],
        &context.runtime,
    )?;
    let queued = QueuedSelectedGraphCall {
        call,
        _argument_roots: argument_roots,
    };
    let child_runtime = context.runtime.clone();
    let policy = runmat_runtime::context::host_allocation::HostAllocationReceipt::policy(Some(
        &context.runtime,
    ))
    .map_err(|source| {
        runmat_runtime::context::host_allocation::host_allocation_runtime_error(
            Some(&context.runtime),
            "RunMat:Interpreter:ParallelTaskScope",
            "parallel graph task scope admission failed",
            source,
        )
    })?;
    match policy {
        runmat_runtime::context::host_allocation::HostAllocationPolicy::Unbounded => driver
            .enqueue_owned_in_context(&context.runtime, move || queued.execute(child_runtime))?,
        runmat_runtime::context::host_allocation::HostAllocationPolicy::Managed => {
            let scope = context.runtime.prepare_poll_scope()?;
            driver.enqueue_owned_scoped_in_context(&context.runtime, scope, move || {
                queued.execute(child_runtime)
            })?
        }
    }
    .await
}

pub(super) async fn fetch_next(
    context: &ExecutionContext,
    futures: &Value,
    timeout_seconds: Option<f64>,
    requested_outputs: usize,
) -> Result<runmat_runtime::execution::RootedValueSequence, RuntimeError> {
    use runmat_execution::TaskResultClaim;

    let tasks = runmat_runtime::parallel::future::tasks(futures).ok_or_else(|| {
        crate::interpreter::errors::mex(
            "InvalidFutureArray",
            "fetchNext expects a parallel.FevalFuture scalar or array",
        )
    })?;
    if tasks.is_empty() {
        return Err(crate::interpreter::errors::mex(
            "NoUnreadFutures",
            "fetchNext requires at least one unread future",
        ));
    }
    let started = runmat_time::Instant::now();
    loop {
        match context
            .runtime
            .execution()
            .claim_next_task_result(&tasks)
            .map_err(execution_error)?
        {
            TaskResultClaim::Claimed { index } => {
                let task = tasks
                    .get(index)
                    .expect("execution service returned a validated task index");
                let result_count = requested_outputs.saturating_sub(1);
                if requested_outputs > 0
                    && usize::from(task.outputs.requested_outputs) != result_count
                {
                    return Err(crate::interpreter::errors::mex(
                        "OutputCountMismatch",
                        "fetchNext output count must match the selected future",
                    ));
                }
                let result = await_sequence(context, Value::Task(task.clone())).await;
                context
                    .runtime
                    .execution()
                    .mark_task_result_read(task)
                    .map_err(execution_error)?;
                let result = result?;
                return fetch_next_outputs(index, result, requested_outputs, &context.runtime);
            }
            TaskResultClaim::Exhausted => {
                return Err(crate::interpreter::errors::mex(
                    "NoUnreadFutures",
                    "fetchNext found no unread futures",
                ));
            }
            TaskResultClaim::Pending => {
                if timeout_seconds.is_some_and(|timeout| started.elapsed().as_secs_f64() >= timeout)
                {
                    return empty_outputs(requested_outputs);
                }
                yield_once(context).await?;
            }
        }
    }
}

pub(super) async fn fetch_outputs(
    context: &ExecutionContext,
    futures: &Value,
    requested_outputs: usize,
    uniform_output: bool,
) -> Result<runmat_runtime::execution::RootedValueSequence, RuntimeError> {
    let tasks = runmat_runtime::parallel::future::output_tasks(futures).ok_or_else(|| {
        crate::interpreter::errors::mex(
            "InvalidFuture",
            "fetchOutputs expects a parallel future scalar or array",
        )
    })?;
    if tasks.is_empty() {
        return empty_outputs(requested_outputs);
    }
    let mut worker_outputs = Vec::with_capacity(tasks.len());
    for task in &tasks {
        if usize::from(task.outputs.requested_outputs) < requested_outputs {
            return Err(crate::interpreter::errors::mex(
                "OutputCountMismatch",
                "each future must provide at least the requested number of outputs",
            ));
        }
        let result = await_sequence(context, Value::Task(task.clone())).await;
        context
            .runtime
            .execution()
            .mark_task_result_read(task)
            .map_err(execution_error)?;
        let result = result?;
        if tasks.len() == 1 {
            return select_outputs(result, requested_outputs);
        }
        worker_outputs.push(split_outputs(result, requested_outputs, &context.runtime)?);
    }
    let mut combined =
        runmat_runtime::execution::RootedValuesBuilder::with_capacity(requested_outputs);
    for output_index in 0..requested_outputs {
        let values = worker_outputs
            .iter()
            .map(|outputs| outputs[output_index].clone())
            .collect::<Vec<_>>();
        if uniform_output {
            let descriptor = runmat_runtime::call::descriptor::CallableDescriptor::resolved(
                runmat_hir::CallableIdentity::Builtin(runmat_hir::BuiltinId("vertcat".into())),
                values,
                1,
                runmat_hir::CallableFallbackPolicy::None,
                runmat_runtime::call::descriptor::CallableCallKind::Direct,
            );
            let output = crate::call::execute_callable_descriptor(&context.runtime, descriptor)
                .await?
                .try_into_single()?;
            combined.push_owned(output);
        } else {
            combined
                .push(Value::Cell(
                    runmat_value::CellArray::new_with_shape(values, vec![tasks.len(), 1])
                        .map_err(|error| crate::interpreter::errors::mex("FetchOutputs", &error))?,
                ))
                .map_err(execution_error)?;
        }
    }
    combined
        .finish()
        .map_err(execution_error)?
        .with_owned_output_storage(|storage| root_join_outputs(storage, &context.runtime))
}

pub(super) fn fetch_outputs_options(arguments: &[Value]) -> Result<bool, RuntimeError> {
    let mut uniform_output = true;
    for pair in arguments.chunks_exact(2) {
        let name = String::try_from(&pair[0]).map_err(|_| {
            crate::interpreter::errors::mex(
                "InvalidOption",
                "fetchOutputs option names must be text scalars",
            )
        })?;
        if !name.eq_ignore_ascii_case("UniformOutput") {
            return Err(crate::interpreter::errors::mex(
                "InvalidOption",
                &format!("fetchOutputs does not recognize option '{name}'"),
            ));
        }
        uniform_output = match &pair[1] {
            Value::Bool(value) => *value,
            Value::Num(value) if *value == 0.0 || *value == 1.0 => *value != 0.0,
            Value::Int(value) if value.to_f64() == 0.0 || value.to_f64() == 1.0 => {
                value.to_f64() != 0.0
            }
            _ => {
                return Err(crate::interpreter::errors::mex(
                    "InvalidOption",
                    "fetchOutputs UniformOutput must be a logical scalar",
                ))
            }
        };
    }
    Ok(uniform_output)
}

pub(super) fn timeout_seconds(value: Value) -> Result<f64, RuntimeError> {
    let seconds = match value {
        Value::Num(value) => value,
        Value::Int(value) => value.to_f64(),
        _ => f64::NAN,
    };
    if !seconds.is_finite() || seconds < 0.0 {
        return Err(crate::interpreter::errors::mex(
            "InvalidTimeout",
            "fetchNext timeout must be a finite nonnegative real scalar",
        ));
    }
    Ok(seconds)
}

fn empty_outputs(
    requested_outputs: usize,
) -> Result<runmat_runtime::execution::RootedValueSequence, RuntimeError> {
    let empty = || {
        Value::Tensor(
            runmat_value::Tensor::new(Vec::new(), vec![0, 0])
                .expect("the canonical empty tensor shape is valid"),
        )
    };
    let sequence = match requested_outputs {
        0 => runmat_value::ValueSequence::empty(),
        1 => runmat_value::ValueSequence::single(empty())
            .map_err(runmat_runtime::sequence::sequence_error_to_runtime)?,
        count => {
            runmat_value::ValueSequence::comma_separated((0..count).map(|_| empty()).collect())
                .map_err(runmat_runtime::sequence::sequence_error_to_runtime)?
        }
    };
    runmat_runtime::execution::RootedValueSequence::new(sequence).map_err(execution_error)
}

pub(super) async fn yield_once(context: &ExecutionContext) -> Result<(), RuntimeError> {
    if matches!(
        context.runtime.execution().cpu_cooperation_policy(),
        runmat_runtime::execution::CpuCooperationPolicy::WorkQuantum(_)
    ) {
        return context
            .runtime
            .execution()
            .cooperate_cpu()
            .await
            .map_err(execution_error);
    }
    let mut yielded = false;
    futures::future::poll_fn(|context| {
        if yielded {
            std::task::Poll::Ready(())
        } else {
            yielded = true;
            context.waker().wake_by_ref();
            std::task::Poll::Pending
        }
    })
    .await;
    Ok(())
}

pub(super) fn pool_handle(value: Value) -> Result<runmat_execution::PoolHandle, RuntimeError> {
    match value {
        Value::Pool(pool) => Ok(pool),
        _ => Err(crate::interpreter::errors::mex(
            "PoolOperandInvalid",
            "parallel scheduling expects a valid pool",
        )),
    }
}
