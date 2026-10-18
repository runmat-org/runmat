use crate::bytecode::program::ExecutionContext;
use crate::FunctionRegistry;
use runmat_runtime::sequence::ResolveValueSequence;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

use super::{execution_error, pop};

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
            .spawn_on(&future, Some(&pool))
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
                .spawn_on(&future, Some(&pool))
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
            .spawn_on(&future, Some(&pool))
            .map_err(execution_error)?
    } else {
        context
            .runtime
            .execution()
            .spawn(&future)
            .map_err(execution_error)?
    };
    stack.try_push(if explicit_pool {
        runmat_runtime::parallel::future::wrap_task(task)
    } else {
        Value::Task(task)
    })?;
    Ok(())
}

pub(super) fn create_future(
    context: &ExecutionContext,
    descriptor: runmat_runtime::call::descriptor::CallableDescriptor,
    function_registry: &FunctionRegistry,
) -> Result<runmat_execution::FutureHandle, RuntimeError> {
    runmat_runtime::execution::validate_spawn_captures(&descriptor.args)?;
    let capabilities = callable_capabilities(&descriptor, function_registry)?;
    let mut invocation = runmat_runtime::execution::DeferredInvocation::callable(descriptor);
    let graph_required =
        runmat_runtime::execution::value_codec::deferred_invocation_requires_graph(&invocation);
    let builtin = matches!(
        &invocation,
        runmat_runtime::execution::DeferredInvocation::Callable(descriptor)
            if matches!(descriptor.as_ref(), runmat_runtime::call::descriptor::CallableDescriptor {
                target: runmat_runtime::call::descriptor::CallableTarget::Resolved {
                    identity: runmat_hir::CallableIdentity::Builtin(_),
                    ..
                },
                ..
            })
    );
    let program = if !context.runtime.execution().requires_program_capture() {
        runmat_runtime::execution::DeferredProgram::Local
    } else if builtin && !graph_required {
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
        runmat_runtime::execution::DeferredProgram::SelectedGraph(selected.clone())
    };
    if matches!(
        &program,
        runmat_runtime::execution::DeferredProgram::SelectedGraph(_)
    ) {
        if builtin {
            let runmat_runtime::execution::DeferredInvocation::Callable(descriptor) = invocation
            else {
                unreachable!("future admission created a callable invocation")
            };
            let descriptor = *descriptor;
            let runmat_runtime::call::descriptor::CallableTarget::Resolved {
                identity: runmat_hir::CallableIdentity::Builtin(name),
                fallback_policy: runmat_hir::CallableFallbackPolicy::None,
            } = descriptor.target
            else {
                return Err(crate::interpreter::errors::mex(
                    "ExecutionProgram",
                    "graph builtin requires an exact selected catalog target",
                ));
            };
            invocation = runmat_runtime::execution::DeferredInvocation::Program {
                callable: runmat_execution::ProgramCallable::Builtin { name },
                context: runmat_execution::ProgramInvocationContext::Direct,
                arguments: descriptor.args,
                requested_outputs: descriptor.requested_outputs,
            };
        } else {
            let runmat_runtime::execution::DeferredInvocation::Callable(descriptor) =
                &mut invocation
            else {
                unreachable!("future admission created a callable invocation")
            };
            bind_selected_semantic_target(descriptor, function_registry)?;
        }
    }
    let program_revision = if matches!(
        &program,
        runmat_runtime::execution::DeferredProgram::SelectedGraph(_)
    ) {
        None
    } else {
        context.runtime.program_revision().cloned()
    };
    context
        .runtime
        .execution()
        .create_future(runmat_runtime::execution::DeferredCall {
            invocation,
            retry: runmat_execution::RetryPolicy::Never,
            program_revision,
            capabilities,
            program,
        })
        .map_err(execution_error)
}

fn callable_capabilities(
    descriptor: &runmat_runtime::call::descriptor::CallableDescriptor,
    registry: &FunctionRegistry,
) -> Result<runmat_types::CapabilitySet, RuntimeError> {
    use runmat_hir::CallableIdentity;

    use runmat_runtime::call::descriptor::CallableTarget;

    if let CallableTarget::Bound(binding) | CallableTarget::BoundLexical { binding, .. } =
        &descriptor.target
    {
        let function = runmat_hir::FunctionId(binding.function);
        let selected = registry
            .owned_view(function, &binding.authority)
            .and_then(|view| view.get(function).map(|body| body.capabilities.clone()))
            .ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "ExecutionProgram",
                    "async bound callable has no exact selected definition",
                )
            })?;
        return Ok(selected);
    }
    let identity = match &descriptor.target {
        CallableTarget::Resolved { identity, .. }
        | CallableTarget::CapturedResolved { identity, .. } => identity,
        _ => {
            return Err(crate::interpreter::errors::mex(
                "ExecutionProgram",
                "async execution requires an exact selected callable before capability binding",
            ));
        }
    };
    match identity {
        CallableIdentity::BoundFunction(function)
        | CallableIdentity::AnonymousFunction(function)
        | CallableIdentity::ExternalFunction { function, .. } => registry
            .get(*function)
            .map(|function| function.capabilities.clone())
            .ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "ExecutionProgram",
                    "async callable is absent from the exact function registry",
                )
            }),
        CallableIdentity::Builtin(name) if runmat_builtins::builtin_name_is_known(&name.0) => {
            Ok(runmat_builtins::builtin_required_capabilities(&name.0))
        }
        CallableIdentity::Builtin(_) => Err(crate::interpreter::errors::mex(
            "ExecutionProgram",
            "async builtin is absent from the exact builtin registry",
        )),
        _ => Err(crate::interpreter::errors::mex(
            "ExecutionProgram",
            "async callable kind has no portable execution identity",
        )),
    }
}

/// A semantic name is resolved only at Future admission, while its selected
/// registry and source owner are still present. The resulting local authority
/// retains the exact body/dependency leases; the worker snapshot never
/// resolves the name or a reused numeric function ID.
fn bind_selected_semantic_target(
    descriptor: &mut runmat_runtime::call::descriptor::CallableDescriptor,
    registry: &FunctionRegistry,
) -> Result<(), RuntimeError> {
    use runmat_hir::CallableIdentity;
    use runmat_runtime::call::descriptor::CallableTarget;

    let function = match &descriptor.target {
        CallableTarget::Resolved {
            identity:
                CallableIdentity::BoundFunction(function)
                | CallableIdentity::AnonymousFunction(function)
                | CallableIdentity::ExternalFunction { function, .. },
            ..
        }
        | CallableTarget::CapturedResolved {
            identity:
                CallableIdentity::BoundFunction(function)
                | CallableIdentity::AnonymousFunction(function)
                | CallableIdentity::ExternalFunction { function, .. },
            ..
        } => *function,
        _ => return Ok(()),
    };
    let binding = registry.exact_bound_binding(function).ok_or_else(|| {
        crate::interpreter::errors::mex(
            "ExecutionProgram",
            "async semantic callable has no exact selected definition and source graph",
        )
    })?;
    descriptor.target = match &descriptor.target {
        CallableTarget::CapturedResolved { closure, .. } => CallableTarget::BoundLexical {
            binding,
            closure: closure.clone(),
        },
        _ => CallableTarget::Bound(binding),
    };
    Ok(())
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
) -> Result<runmat_runtime::execution::RootedValue, RuntimeError> {
    use runmat_runtime::sequence::ResolveValueSequence;

    await_sequence(context, value).await?.try_map_owned(
        |sequence| {
            let values = sequence.resolve(
                runmat_types::SequenceUse::RequireSingle,
                runmat_runtime::sequence::SequenceResolutionContext::default(),
            )?;
            values
                .try_into_first()
                .map_err(|_| {
                    runmat_runtime::sequence::sequence_error_to_runtime(
                        runmat_value::ValueSequenceError::SharedOutputStorage,
                    )
                })?
                .ok_or_else(|| {
                    execution_error(
                        runmat_runtime::execution::ExecutionServiceError::InvalidOutputContract,
                    )
                })
        },
        runmat_runtime::execution::ExecutionServiceError::into_runtime_error,
    )
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
                yield_once().await;
                value = pending;
            }
            AwaitAction::ExecuteFuture { handle, call } => {
                let mut running = runmat_runtime::execution::RunningFutureLease::new(
                    context.runtime.clone(),
                    handle,
                );
                let result = match &call.invocation {
                    runmat_runtime::execution::DeferredInvocation::Callable(descriptor) => {
                        crate::call::execute_callable_descriptor(
                            &context.runtime,
                            descriptor.as_ref().clone(),
                        )
                        .await
                    }
                    runmat_runtime::execution::DeferredInvocation::Program { .. } => {
                        crate::execute_deferred_program_in_context(*call, context.runtime.clone())
                            .await
                            .map_err(execution_error)
                            .and_then(|sequence| {
                                runmat_runtime::execution::RootedValueSequence::new(sequence)
                                    .map_err(execution_error)
                            })
                    }
                };
                let stored = result.as_ref().map(Clone::clone).map_err(|error| {
                    runmat_runtime::execution::ExecutionServiceError::Failed(error.to_string())
                });
                running.complete(stored).map_err(execution_error)?;
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
                return fetch_next_outputs(index, result, requested_outputs);
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
        worker_outputs.push(split_outputs(result, requested_outputs)?);
    }
    if tasks.len() == 1 {
        return worker_outputs
            .pop()
            .expect("one task produced one output row")
            .with_owned_output_storage(root_join_outputs);
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
                .try_map_owned(
                    |sequence| {
                        let values = sequence.resolve(
                            runmat_types::SequenceUse::RequireSingle,
                            runmat_runtime::sequence::SequenceResolutionContext::default(),
                        )?;
                        values
                            .try_into_first()
                            .map_err(|_| {
                                runmat_runtime::sequence::sequence_error_to_runtime(
                                    runmat_value::ValueSequenceError::SharedOutputStorage,
                                )
                            })?
                            .ok_or_else(|| {
                                execution_error(
                            runmat_runtime::execution::ExecutionServiceError::InvalidOutputContract,
                        )
                            })
                    },
                    runmat_runtime::execution::ExecutionServiceError::into_runtime_error,
                )?;
            output.with_owned(|value| combined.push(value).map_err(execution_error))?;
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
        .with_owned_output_storage(root_join_outputs)
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

fn split_outputs(
    sequence: runmat_runtime::execution::RootedValueSequence,
    count: usize,
) -> Result<runmat_runtime::execution::RootedValues, RuntimeError> {
    sequence.with_owned_sequence(|sequence| {
        let values = match count {
            0 => Ok(runmat_value::OutputListStorage::empty()),
            1 => sequence.resolve(
                runmat_types::SequenceUse::SelectPrefix { count: 1 },
                runmat_runtime::sequence::SequenceResolutionContext::default(),
            ),
            _ => sequence
                .resolve(
                    runmat_types::SequenceUse::SelectPrefix { count },
                    runmat_runtime::sequence::SequenceResolutionContext::default(),
                )
                .map_err(|_| {
                    crate::interpreter::errors::mex(
                        "OutputCountMismatch",
                        "future result does not contain the requested number of outputs",
                    )
                }),
        }?;
        runmat_runtime::execution::RootedValues::from_output_storage(values)
            .map_err(execution_error)
    })
}

fn join_outputs(
    outputs: runmat_value::OutputListStorage,
) -> Result<runmat_value::ValueSequence, RuntimeError> {
    Ok(match outputs.len() {
        0 => runmat_value::ValueSequence::empty(),
        1 => runmat_value::ValueSequence::single(
            outputs
                .try_into_first()
                .map_err(|_| {
                    runmat_runtime::sequence::sequence_error_to_runtime(
                        runmat_value::ValueSequenceError::SharedOutputStorage,
                    )
                })?
                .expect("one output is present"),
        )
        .map_err(runmat_runtime::sequence::sequence_error_to_runtime)?,
        _ => runmat_value::ValueSequence::from_output_storage(outputs)
            .map_err(runmat_runtime::sequence::sequence_error_to_runtime)?,
    })
}

fn root_join_outputs(
    outputs: runmat_value::OutputListStorage,
) -> Result<runmat_runtime::execution::RootedValueSequence, RuntimeError> {
    runmat_runtime::execution::RootedValueSequence::new(join_outputs(outputs)?)
        .map_err(execution_error)
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

fn fetch_next_outputs(
    index: usize,
    result: runmat_runtime::execution::RootedValueSequence,
    requested_outputs: usize,
) -> Result<runmat_runtime::execution::RootedValueSequence, RuntimeError> {
    if requested_outputs == 0 {
        return runmat_runtime::execution::RootedValueSequence::new(
            runmat_value::ValueSequence::empty(),
        )
        .map_err(execution_error);
    }
    if requested_outputs == 1 {
        return runmat_runtime::execution::RootedValueSequence::new(
            runmat_value::ValueSequence::single(Value::Num((index + 1) as f64))
                .map_err(runmat_runtime::sequence::sequence_error_to_runtime)?,
        )
        .map_err(execution_error);
    }
    result.try_map_owned(
        |result| {
            let mut outputs = Vec::with_capacity(requested_outputs);
            outputs.push(Value::Num((index + 1) as f64));
            let values = result.into_output_storage().try_into_owned().map_err(|_| {
                runmat_runtime::sequence::sequence_error_to_runtime(
                    runmat_value::ValueSequenceError::SharedOutputStorage,
                )
            })?;
            outputs.extend(values);
            runmat_value::ValueSequence::comma_separated(outputs)
                .map_err(runmat_runtime::sequence::sequence_error_to_runtime)
        },
        runmat_runtime::execution::ExecutionServiceError::into_runtime_error,
    )
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
