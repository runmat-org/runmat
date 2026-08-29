use crate::bytecode::program::ExecutionContext;
use crate::FunctionRegistry;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

use super::{execution_error, pop};
use crate::interpreter::dispatch::calls;

pub(super) fn schedule_feval(
    stack: &mut Vec<Value>,
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
            .spawn_on(&future, Some(&pool))
            .map_err(execution_error)?;
        stack.push(runmat_runtime::parallel::future::wrap_task(task));
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
                .spawn_on(&future, Some(&pool))
                .map_err(execution_error)?,
        );
    }
    stack.push(
        runmat_runtime::parallel::future::wrap_on_all_tasks(tasks)
            .map_err(|error| crate::interpreter::errors::mex("FutureArray", &error))?,
    );
    Ok(())
}

pub(super) fn spawn(
    stack: &mut Vec<Value>,
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
            .spawn_on(&future, Some(&pool))
            .map_err(execution_error)?
    } else {
        context
            .runtime
            .execution()
            .spawn(&future)
            .map_err(execution_error)?
    };
    stack.push(if explicit_pool {
        runmat_runtime::parallel::future::wrap_task(task)
    } else {
        Value::Task(task)
    });
    Ok(())
}

pub(super) fn create_future(
    context: &ExecutionContext,
    descriptor: runmat_runtime::call::descriptor::CallableDescriptor,
    function_registry: &FunctionRegistry,
) -> Result<runmat_execution::FutureHandle, RuntimeError> {
    runmat_runtime::execution::validate_spawn_capture(&Value::OutputList(descriptor.args.clone()))?;
    let program = context
        .runtime
        .execution()
        .requires_program_capture()
        .then(|| {
            serde_json::to_vec(function_registry).map_err(|error| {
                crate::interpreter::errors::mex(
                    "ExecutionProgram",
                    &format!("failed to capture the exact async program: {error}"),
                )
            })
        })
        .transpose()?;
    context
        .runtime
        .execution()
        .create_future(runmat_runtime::execution::DeferredCall {
            invocation: runmat_runtime::execution::DeferredInvocation::Callable(descriptor),
            retry: runmat_execution::RetryPolicy::Never,
            program_revision: context.runtime.program_revision().cloned(),
            program,
        })
        .map_err(execution_error)
}

pub(super) fn semantic_descriptor(
    function: runmat_hir::FunctionId,
    requested_outputs: usize,
    arguments: Vec<Value>,
) -> runmat_runtime::call::descriptor::CallableDescriptor {
    runmat_runtime::call::descriptor::CallableDescriptor::resolved(
        runmat_hir::CallableIdentity::BoundFunction(function),
        arguments,
        requested_outputs,
        runmat_hir::CallableFallbackPolicy::None,
        runmat_runtime::call::descriptor::CallableCallKind::Direct,
    )
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
) -> Result<Value, RuntimeError> {
    use runmat_runtime::execution::AwaitAction;

    let mut value = runmat_runtime::parallel::future::execution_value(&value)
        .cloned()
        .unwrap_or(value);
    loop {
        match context
            .runtime
            .execution()
            .begin_await(value)
            .map_err(execution_error)?
        {
            AwaitAction::Passthrough(value) | AwaitAction::Completed(value) => return Ok(value),
            AwaitAction::Pending(pending) => {
                yield_once().await;
                value = pending;
            }
            AwaitAction::ExecuteFuture { handle, call } => {
                let requested_outputs = call.invocation.requested_outputs();
                let result = match &call.invocation {
                    runmat_runtime::execution::DeferredInvocation::Callable(descriptor) => {
                        runmat_runtime::call::descriptor::execute_callable_descriptor(
                            descriptor.clone(),
                        )
                        .await
                    }
                    runmat_runtime::execution::DeferredInvocation::Program { .. } => {
                        crate::execute_deferred_program_in_context(*call, context.runtime.clone())
                            .await
                            .map_err(execution_error)
                    }
                }
                .map(|value| calls::normalize_requested_outputs(value, requested_outputs));
                let stored = result.as_ref().map(Clone::clone).map_err(|error| {
                    runmat_runtime::execution::ExecutionServiceError::Failed(error.to_string())
                });
                context
                    .runtime
                    .execution()
                    .complete_future(&handle, stored)
                    .map_err(execution_error)?;
                return result;
            }
        }
    }
}

pub(super) async fn fetch_next(
    context: &ExecutionContext,
    futures: &Value,
    timeout_seconds: Option<f64>,
    requested_outputs: usize,
) -> Result<Value, RuntimeError> {
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
                let result = await_value(context, Value::Task(task.clone())).await;
                context
                    .runtime
                    .execution()
                    .mark_task_result_read(task)
                    .map_err(execution_error)?;
                let result = result?;
                return Ok(fetch_next_outputs(index, result, requested_outputs));
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
                    return Ok(empty_output_list(requested_outputs));
                }
                yield_once().await;
            }
        }
    }
}

pub(super) async fn fetch_outputs(
    context: &ExecutionContext,
    futures: &Value,
    requested_outputs: usize,
    uniform_output: bool,
) -> Result<Value, RuntimeError> {
    let tasks = runmat_runtime::parallel::future::output_tasks(futures).ok_or_else(|| {
        crate::interpreter::errors::mex(
            "InvalidFuture",
            "fetchOutputs expects a parallel future scalar or array",
        )
    })?;
    if tasks.is_empty() {
        return Ok(empty_output_list(requested_outputs));
    }
    let mut worker_outputs = Vec::with_capacity(tasks.len());
    for task in &tasks {
        if usize::from(task.outputs.requested_outputs) < requested_outputs {
            return Err(crate::interpreter::errors::mex(
                "OutputCountMismatch",
                "each future must provide at least the requested number of outputs",
            ));
        }
        let result = await_value(context, Value::Task(task.clone())).await;
        context
            .runtime
            .execution()
            .mark_task_result_read(task)
            .map_err(execution_error)?;
        let result = result?;
        worker_outputs.push(split_outputs(result, requested_outputs)?);
    }
    if tasks.len() == 1 {
        return Ok(join_outputs(
            worker_outputs
                .pop()
                .expect("one task produced one output row"),
        ));
    }
    let mut combined = Vec::with_capacity(requested_outputs);
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
            combined.push(
                runmat_runtime::call::descriptor::execute_callable_descriptor(descriptor).await?,
            );
        } else {
            combined.push(Value::Cell(
                runmat_value::CellArray::new_with_shape(values, vec![tasks.len(), 1])
                    .map_err(|error| crate::interpreter::errors::mex("FetchOutputs", &error))?,
            ));
        }
    }
    Ok(join_outputs(combined))
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

fn split_outputs(value: Value, count: usize) -> Result<Vec<Value>, RuntimeError> {
    match count {
        0 => Ok(Vec::new()),
        1 => Ok(vec![value]),
        _ => match value {
            Value::OutputList(values) if values.len() >= count => {
                Ok(values.into_iter().take(count).collect())
            }
            _ => Err(crate::interpreter::errors::mex(
                "OutputCountMismatch",
                "future result does not contain the requested number of outputs",
            )),
        },
    }
}

fn join_outputs(outputs: Vec<Value>) -> Value {
    match outputs.len() {
        0 => Value::OutputList(Vec::new()),
        1 => outputs.into_iter().next().expect("one output is present"),
        _ => Value::OutputList(outputs),
    }
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

fn fetch_next_outputs(index: usize, result: Value, requested_outputs: usize) -> Value {
    if requested_outputs == 0 {
        return Value::OutputList(Vec::new());
    }
    if requested_outputs == 1 {
        return Value::Num((index + 1) as f64);
    }
    let mut outputs = Vec::with_capacity(requested_outputs);
    outputs.push(Value::Num((index + 1) as f64));
    if requested_outputs > 1 {
        match result {
            Value::OutputList(values) => outputs.extend(values),
            value => outputs.push(value),
        }
    }
    Value::OutputList(outputs)
}

fn empty_output_list(requested_outputs: usize) -> Value {
    let empty = || {
        Value::Tensor(
            runmat_value::Tensor::new(Vec::new(), vec![0, 0])
                .expect("the canonical empty tensor shape is valid"),
        )
    };
    match requested_outputs {
        0 => Value::OutputList(Vec::new()),
        1 => empty(),
        count => Value::OutputList((0..count).map(|_| empty()).collect()),
    }
}

pub(super) async fn yield_once() {
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
