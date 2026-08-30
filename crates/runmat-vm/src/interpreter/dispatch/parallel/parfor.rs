use crate::bytecode::program::ExecutionContext;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

use super::{execution_error, tasks};

#[derive(Clone, Copy)]
struct ParforExecution<'a> {
    bytecode: &'a crate::Bytecode,
    executable: &'a crate::BytecodeParforRegion,
    context: &'a ExecutionContext,
    current_function_name: &'a str,
}

struct ParforTaskPayload {
    inputs: Vec<Value>,
    iterations: Vec<Value>,
    randomness: runmat_execution::ParallelRandomnessContext,
}

pub(super) async fn execute(
    bytecode: &crate::bytecode::Bytecode,
    executable: &crate::BytecodeParforRegion,
    iterable: Value,
    maximum_workers: Option<u32>,
    vars: &mut [Value],
    context: &ExecutionContext,
    current_function_name: &str,
) -> Result<(), RuntimeError> {
    let mut iterations = runmat_runtime::iteration::ForColumnIterator::new(iterable).await?;
    let mut iteration_values = Vec::new();
    while let Some(iteration) = iterations.next().await? {
        iteration_values.push(iteration);
    }
    validate_parfor_iterations(&iteration_values).await?;
    let randomness = parallel_randomness_context(
        executable.contract.randomness,
        &context.runtime,
        iteration_values.len(),
    )?;
    let inputs = executable
        .input_variables()
        .map(|variable| {
            vars.get(variable.slot).cloned().ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "ParallelFrame",
                    "parallel task input is outside its compiled VM frame",
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let execution = ParforExecution {
        bytecode,
        executable,
        context,
        current_function_name,
    };
    if maximum_workers == Some(0) {
        return execute_parfor_serial(
            execution,
            ParforTaskPayload {
                inputs,
                iterations: iteration_values,
                randomness,
            },
            vars,
        )
        .await;
    }
    let pool = context
        .runtime
        .execution()
        .ensure_pool(runmat_execution::PoolRequest::automatic())
        .map_err(execution_error)?;
    let worker_budget = maximum_workers
        .map(|maximum| maximum.min(pool.workers))
        .unwrap_or(pool.workers)
        .max(1);
    let scheduled_inputs = crate::interpreter::parallel_placement::prepare(
        executable,
        &inputs,
        pool.backend,
        worker_budget,
    )
    .await;
    match scheduled_inputs {
        Ok(scheduled_inputs) => {
            for (slot, destination) in scheduled_inputs.sliced_destinations {
                let Some(frame_value) = vars.get_mut(slot) else {
                    return Err(crate::interpreter::errors::mex(
                        "ParallelFrame",
                        "parallel sliced destination is outside its compiled VM frame",
                    ));
                };
                *frame_value = destination;
            }
            return execute_parfor_tasks(
                execution,
                ParforTaskPayload {
                    inputs: scheduled_inputs.task_inputs,
                    iterations: iteration_values,
                    randomness,
                },
                vars,
                &pool.handle,
                worker_budget,
            )
            .await;
        }
        Err(reason) => {
            tracing::debug!(reason = %reason, effects = ?executable.contract.effects, "using serial parfor correctness backend");
        }
    }
    execute_parfor_serial(
        execution,
        ParforTaskPayload {
            inputs,
            iterations: iteration_values,
            randomness,
        },
        vars,
    )
    .await
}

async fn execute_parfor_serial(
    execution: ParforExecution<'_>,
    task: ParforTaskPayload,
    vars: &mut [Value],
) -> Result<(), RuntimeError> {
    let outputs = crate::interpreter::runner::interpret_parfor_task_in_context(
        crate::interpreter::runner::ParforTaskExecution {
            bytecode: execution.bytecode,
            region: execution.executable,
            inputs: task.inputs,
            iterations: task.iterations,
            randomness: task.randomness,
            mode: crate::interpreter::runner::ParforTaskMode::Sequential,
            current_function_name: execution.current_function_name,
            runtime: execution.context.runtime.clone(),
        },
    )
    .await?;
    for (variable, output) in execution.executable.output_variables().zip(outputs) {
        let Some(slot) = vars.get_mut(variable.slot) else {
            return Err(crate::interpreter::errors::mex(
                "ParallelFrame",
                "parallel task output is outside its compiled VM frame",
            ));
        };
        *slot = output;
    }
    Ok(())
}

async fn validate_parfor_iterations(iterations: &[Value]) -> Result<(), RuntimeError> {
    let mut direction = None;
    let mut previous = None;
    for iteration in iterations {
        let scalar = runmat_runtime::indexing::selectors::index_scalar_from_value(iteration)
            .await?
            .ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "ParallelIterationRange",
                    "parfor loop values must be integers",
                )
            })?;
        let current = match scalar {
            runmat_runtime::indexing::selectors::IndexScalar::Signed(value) => i128::from(value),
            runmat_runtime::indexing::selectors::IndexScalar::Unsigned(value) => i128::from(value),
        };
        if let Some(previous) = previous {
            let step = current.checked_sub(previous).ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "ParallelIterationRange",
                    "parfor loop range overflowed while validating its step",
                )
            })?;
            if !matches!(step, -1 | 1) || direction.is_some_and(|prior| prior != step) {
                return Err(crate::interpreter::errors::mex(
                    "ParallelIterationRange",
                    "parfor loop values must be consecutive integers with a step of 1 or -1",
                ));
            }
            direction = Some(step);
        }
        previous = Some(current);
    }
    Ok(())
}

async fn execute_parfor_tasks(
    execution: ParforExecution<'_>,
    task: ParforTaskPayload,
    vars: &mut [Value],
    pool: &runmat_execution::PoolHandle,
    worker_budget: u32,
) -> Result<(), RuntimeError> {
    let ParforTaskPayload {
        inputs,
        iterations,
        randomness,
    } = task;
    let executable = execution.executable;
    let context = execution.context;
    let graph = runmat_execution::ParallelTaskGraph::balanced(
        executable.contract.id,
        iterations.len(),
        worker_budget,
    )
    .map_err(|error| crate::interpreter::errors::mex("ParallelTaskGraph", &error.to_string()))?;
    if graph.chunks.is_empty() {
        return Ok(());
    }
    tracing::debug!(
        region_function = executable.contract.id.0.function.0,
        region_ordinal = executable.contract.id.0.ordinal,
        iterations = graph.iteration_count,
        chunks = graph.chunks.len(),
        workers = graph.worker_limit,
        "scheduled compiler-bound parfor task graph"
    );
    let program = serde_json::to_vec(execution.bytecode).map_err(|error| {
        crate::interpreter::errors::mex(
            "ExecutionProgram",
            &format!("failed to encode the exact parallel program: {error}"),
        )
    })?;
    let mut tasks = Vec::with_capacity(graph.chunks.len());
    for chunk in &graph.chunks {
        let values = iterations[chunk.start..chunk.start + chunk.len].to_vec();
        let iteration_argument = Value::Cell(
            runmat_value::CellArray::new(values, 1, chunk.len)
                .map_err(|error| crate::interpreter::errors::mex("ParallelTaskInput", &error))?,
        );
        let mut arguments = Vec::with_capacity(inputs.len() + 1);
        arguments.push(iteration_argument);
        arguments.extend(inputs.iter().cloned());
        let task_randomness = slice_randomness(&randomness, chunk)?;
        runmat_runtime::execution::validate_spawn_capture(&Value::OutputList(arguments.clone()))?;
        let future = context
            .runtime
            .execution()
            .create_future(runmat_runtime::execution::DeferredCall {
                invocation: runmat_runtime::execution::DeferredInvocation::Program {
                    callable: runmat_execution::ProgramCallable::parallel_region(
                        executable.contract.id,
                    ),
                    context: runmat_execution::ProgramInvocationContext::ParallelTask {
                        task: runmat_execution::ParallelTaskContext {
                            region: executable.contract.id,
                            chunk: *chunk,
                            randomness: task_randomness,
                        },
                    },
                    arguments,
                    requested_outputs: 1,
                },
                retry: runmat_execution::RetryPolicy::IdempotentInfrastructure,
                program_revision: context.runtime.program_revision().cloned(),
                capabilities: executable.contract.capabilities.clone(),
                program: Some(program.clone()),
            })
            .map_err(execution_error)?;
        tasks.push(
            context
                .runtime
                .execution()
                .spawn_on(&future, Some(pool))
                .map_err(execution_error)?,
        );
    }

    let results = await_parallel_tasks(context, &tasks).await?;
    assemble_sliced_results(executable, &graph, &iterations, results, vars).await
}

async fn await_parallel_tasks(
    context: &ExecutionContext,
    tasks: &[runmat_execution::TaskHandle],
) -> Result<Vec<Value>, RuntimeError> {
    use runmat_execution::TaskResultClaim;

    let mut results = vec![None; tasks.len()];
    let mut remaining = tasks.len();
    while remaining > 0 {
        if let Err(error) = crate::interpreter::engine::check_cancelled() {
            cancel_parallel_tasks(context, tasks, runmat_execution::CancellationReason::User);
            return Err(error);
        }
        let claim = match context.runtime.execution().claim_next_task_result(tasks) {
            Ok(claim) => claim,
            Err(error) => {
                cancel_parallel_tasks(
                    context,
                    tasks,
                    runmat_execution::CancellationReason::DependencyFailed,
                );
                return Err(execution_error(error));
            }
        };
        match claim {
            TaskResultClaim::Claimed { index } => {
                let Some(task) = tasks.get(index) else {
                    cancel_parallel_tasks(
                        context,
                        tasks,
                        runmat_execution::CancellationReason::DependencyFailed,
                    );
                    return Err(crate::interpreter::errors::mex(
                        "ParallelTaskState",
                        "parallel scheduler returned an invalid completed-task index",
                    ));
                };
                let result = tasks::await_value(context, Value::Task(task.clone())).await;
                if let Err(error) = context.runtime.execution().mark_task_result_read(task) {
                    cancel_parallel_tasks(
                        context,
                        tasks,
                        runmat_execution::CancellationReason::DependencyFailed,
                    );
                    return Err(execution_error(error));
                }
                match result {
                    Ok(value) => {
                        let slot = &mut results[index];
                        if slot.replace(value).is_some() {
                            cancel_parallel_tasks(
                                context,
                                tasks,
                                runmat_execution::CancellationReason::DependencyFailed,
                            );
                            return Err(crate::interpreter::errors::mex(
                                "ParallelTaskState",
                                "parallel scheduler completed one task more than once",
                            ));
                        }
                        remaining -= 1;
                    }
                    Err(error) => {
                        cancel_parallel_tasks(
                            context,
                            tasks,
                            runmat_execution::CancellationReason::DependencyFailed,
                        );
                        return Err(error);
                    }
                }
            }
            TaskResultClaim::Pending => tasks::yield_once().await,
            TaskResultClaim::Exhausted => {
                cancel_parallel_tasks(
                    context,
                    tasks,
                    runmat_execution::CancellationReason::DependencyFailed,
                );
                return Err(crate::interpreter::errors::mex(
                    "ParallelTaskState",
                    "parallel scheduler exhausted results before the task graph completed",
                ));
            }
        }
    }
    results
        .into_iter()
        .map(|result| {
            result.ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "ParallelTaskState",
                    "parallel scheduler omitted a task result",
                )
            })
        })
        .collect()
}

fn cancel_parallel_tasks(
    context: &ExecutionContext,
    tasks: &[runmat_execution::TaskHandle],
    reason: runmat_execution::CancellationReason,
) {
    for task in tasks {
        let _ = context
            .runtime
            .execution()
            .cancel(&Value::Task(task.clone()), reason);
    }
}

fn parallel_randomness_context(
    policy: runmat_types::ParallelRandomnessPolicy,
    runtime: &runmat_runtime::context::RuntimeContext,
    iteration_count: usize,
) -> Result<runmat_execution::ParallelRandomnessContext, RuntimeError> {
    match policy {
        runmat_types::ParallelRandomnessPolicy::Inherit => {
            Ok(runmat_execution::ParallelRandomnessContext::Inherit)
        }
        runmat_types::ParallelRandomnessPolicy::DeterministicSubstreams => {
            let streams =
                runmat_runtime::builtins::common::random::reserve_parallel_random_streams(
                    runtime,
                    iteration_count,
                )?;
            Ok(runmat_execution::ParallelRandomnessContext::Deterministic { streams })
        }
        runmat_types::ParallelRandomnessPolicy::Nondeterministic => {
            Ok(runmat_execution::ParallelRandomnessContext::Nondeterministic)
        }
    }
}

fn slice_randomness(
    randomness: &runmat_execution::ParallelRandomnessContext,
    chunk: &runmat_execution::ParallelChunk,
) -> Result<runmat_execution::ParallelRandomnessContext, RuntimeError> {
    match randomness {
        runmat_execution::ParallelRandomnessContext::Deterministic { streams } => {
            let end = chunk.start.checked_add(chunk.len).ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "ParallelRandomness",
                    "parallel random-stream extent overflowed",
                )
            })?;
            let streams = streams.get(chunk.start..end).ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "ParallelRandomness",
                    "parallel random-stream assignment does not cover the task chunk",
                )
            })?;
            Ok(runmat_execution::ParallelRandomnessContext::Deterministic {
                streams: streams.to_vec(),
            })
        }
        runmat_execution::ParallelRandomnessContext::Inherit => {
            Ok(runmat_execution::ParallelRandomnessContext::Inherit)
        }
        runmat_execution::ParallelRandomnessContext::Nondeterministic => {
            Ok(runmat_execution::ParallelRandomnessContext::Nondeterministic)
        }
    }
}

async fn assemble_sliced_results(
    executable: &crate::BytecodeParforRegion,
    graph: &runmat_execution::ParallelTaskGraph,
    iterations: &[Value],
    results: Vec<Value>,
    vars: &mut [Value],
) -> Result<(), RuntimeError> {
    let output_variables = executable.output_variables().collect::<Vec<_>>();
    for (chunk, result) in graph.chunks.iter().zip(results) {
        let Value::Cell(result) = result else {
            return Err(crate::interpreter::errors::mex(
                "ParallelTaskOutput",
                "parallel task did not return its typed result envelope",
            ));
        };
        if result.data.len() != output_variables.len() {
            return Err(crate::interpreter::errors::mex(
                "ParallelTaskOutput",
                "parallel task result does not match its compiler-owned output layout",
            ));
        }
        for (variable, worker_value) in output_variables.iter().zip(&result.data) {
            match &variable.contract.role {
                runmat_types::ParallelVariableRole::Sliced { access } => {
                    let Some(destination) = vars.get(variable.slot).cloned() else {
                        return Err(crate::interpreter::errors::mex(
                            "ParallelFrame",
                            "parallel sliced output is outside its compiled VM frame",
                        ));
                    };
                    let offset_value = slice_offset_value(executable, *access, vars)?;
                    let mut assembled = destination;
                    for iteration in &iterations[chunk.start..chunk.start + chunk.len] {
                        assembled = runmat_runtime::parallel::assembly::merge_slice(
                            assembled,
                            worker_value,
                            *access,
                            iteration,
                            offset_value.as_ref(),
                        )
                        .await?;
                    }
                    vars[variable.slot] = assembled;
                }
                runmat_types::ParallelVariableRole::Reduction { operator } => {
                    let Value::Cell(contributions) = worker_value else {
                        return Err(crate::interpreter::errors::mex(
                            "ParallelTaskOutput",
                            "parallel reduction task did not return per-iteration contributions",
                        ));
                    };
                    if contributions.data.len() != chunk.len {
                        return Err(crate::interpreter::errors::mex(
                            "ParallelTaskOutput",
                            "parallel reduction task returned the wrong contribution count",
                        ));
                    }
                    let Some(mut accumulator) = vars.get(variable.slot).cloned() else {
                        return Err(crate::interpreter::errors::mex(
                            "ParallelFrame",
                            "parallel reduction output is outside its compiled VM frame",
                        ));
                    };
                    for contribution in &contributions.data {
                        accumulator = runmat_runtime::parallel::reduction::combine(
                            *operator,
                            accumulator,
                            contribution.clone(),
                        )
                        .await?;
                    }
                    vars[variable.slot] = accumulator;
                }
                _ => {
                    return Err(crate::interpreter::errors::mex(
                        "ParallelTaskOutput",
                        "parallel scheduler received an unsupported output classification",
                    ))
                }
            }
        }
    }
    Ok(())
}

fn slice_offset_value(
    executable: &crate::BytecodeParforRegion,
    access: runmat_types::ParallelSliceAccess,
    vars: &[Value],
) -> Result<Option<Value>, RuntimeError> {
    let value = match access.offset {
        runmat_types::ParallelSliceOffset::Add(
            runmat_types::ParallelSliceOffsetOperand::Broadcast(value),
        )
        | runmat_types::ParallelSliceOffset::Subtract(
            runmat_types::ParallelSliceOffsetOperand::Broadcast(value),
        ) => value,
        _ => return Ok(None),
    };
    let variable = executable
        .variables
        .iter()
        .find(|variable| variable.contract.value == value)
        .ok_or_else(|| {
            crate::interpreter::errors::mex(
                "ParallelSliceOffset",
                "parallel slice offset has no compiler-bound frame value",
            )
        })?;
    vars.get(variable.slot).cloned().map(Some).ok_or_else(|| {
        crate::interpreter::errors::mex(
            "ParallelSliceOffset",
            "parallel slice offset is outside its compiled VM frame",
        )
    })
}

pub(super) fn maximum_worker_count(value: Value) -> Result<u32, RuntimeError> {
    let count = match value {
        Value::Int(value) => value.try_to_u64(),
        Value::Num(value)
            if value.is_finite()
                && value.fract() == 0.0
                && value >= 0.0
                && value <= u32::MAX as f64 =>
        {
            Some(value as u64)
        }
        _ => None,
    };
    count
        .and_then(|count| u32::try_from(count).ok())
        .ok_or_else(|| {
            crate::interpreter::errors::mex(
                "InvalidWorkerCount",
                "parfor maximum worker count must be a nonnegative integer scalar",
            )
        })
}
