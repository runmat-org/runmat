use crate::bytecode::program::ExecutionContext;
use crate::interpreter::vars::InterpreterVars;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

use super::{execution_error, task_group::SubmittedTaskGroup, tasks};

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
    // Last field: its registration must outlive both owned Value buffers if
    // an unpolled child is canceled and the queued payload is dropped.
    input_roots: Option<runmat_runtime::execution::RootedValueLease>,
}

pub(super) async fn execute(
    bytecode: &crate::bytecode::Bytecode,
    executable: &crate::BytecodeParforRegion,
    iterable: Value,
    maximum_workers: Option<u32>,
    vars: &mut InterpreterVars,
    context: &ExecutionContext,
    current_function_name: &str,
) -> Result<(), RuntimeError> {
    let mut iterations = runmat_runtime::iteration::ForColumnIterator::new(iterable).await?;
    let mut iteration_values = runmat_runtime::execution::RootedValuesBuilder::default();
    while let Some(iteration) = iterations.next().await? {
        iteration
            .with_owned(|iteration| iteration_values.push(iteration))
            .map_err(|error| error.into_runtime_error())?;
    }
    let iteration_values = iteration_values
        .finish()
        .map_err(|error| error.into_runtime_error())?;
    iteration_values.with_unbounded_values_async(|iteration_values| async move {
    validate_parfor_iterations(&iteration_values).await?;
    let randomness = parallel_randomness_context(
        executable.contract.randomness,
        &context.runtime,
        iteration_values.len(),
    ).await?;
    let inputs = executable
        .input_variables()
        .map(|variable| {
            vars.get(variable.slot).map(|value| value.clone()).ok_or_else(|| {
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
                input_roots: None,
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
        &context.runtime,
        executable,
        &inputs,
        pool.backend,
        worker_budget,
    )
    .await;
    match scheduled_inputs {
        Ok(scheduled_inputs) => {
            for (slot, destination) in scheduled_inputs.sliced_destinations {
                if vars.get(slot).is_none() {
                    return Err(crate::interpreter::errors::mex(
                        "ParallelFrame",
                        "parallel sliced destination is outside its compiled VM frame",
                    ));
                };
                let _retired = vars.replace_owned(slot, destination);
            }
            return execute_parfor_tasks(
                execution,
                ParforTaskPayload {
                    inputs: scheduled_inputs.task_inputs,
                    input_roots: Some(scheduled_inputs.task_roots),
                    iterations: iteration_values,
                    randomness,
                },
                vars,
                &pool.handle,
                worker_budget,
            )
            .await;
        }
        Err(crate::interpreter::parallel_placement::ParallelPlacementRejection::InputRootingFailed {
            error,
        }) => {
            return Err(error.into_runtime_error());
        }
        Err(crate::interpreter::parallel_placement::ParallelPlacementRejection::AutomaticGatherFailed {
            error,
            ..
        } | crate::interpreter::parallel_placement::ParallelPlacementRejection::ReductionIdentityFailed {
            error,
            ..
        } | crate::interpreter::parallel_placement::ParallelPlacementRejection::InputMaterializationFailed {
            error,
        }) => {
            // This is an operational failure, not an unavailable placement.
            // Preserve cancellation/provider provenance; serial execution
            // must not replay the region after a failed input transfer.
            return Err(runmat_runtime::RuntimeError::from_shared_original(error));
        }
        Err(reason) => {
            tracing::debug!(reason = %reason, effects = ?executable.contract.effects, "using serial parfor correctness backend");
        }
    }
    execute_parfor_serial(
        execution,
        ParforTaskPayload {
            inputs,
            input_roots: None,
            iterations: iteration_values,
            randomness,
        },
        vars,
    )
    .await
    }).await?
}

async fn execute_parfor_serial(
    execution: ParforExecution<'_>,
    task: ParforTaskPayload,
    vars: &mut InterpreterVars,
) -> Result<(), RuntimeError> {
    let ParforTaskPayload {
        inputs,
        input_roots,
        iterations,
        randomness,
    } = task;
    let runtime = execution.context.runtime.clone();
    // A queued child may be canceled before its first poll. Root the exact
    // input and iteration graphs before queue publication, not in its body.
    let queued_roots = runmat_runtime::execution::RootedValueLease::for_value_slices_in_context(
        &[inputs.as_slice(), iterations.as_slice()],
        &runtime,
    )?;
    drop(input_roots);
    let task = ParforTaskPayload {
        inputs,
        iterations,
        randomness,
        input_roots: Some(queued_roots),
    };
    let owner = execution.context.bytecode_owner.clone().ok_or_else(|| {
        crate::interpreter::errors::mex(
            "ParallelExecutableOwner",
            "serial parfor lost its retained compiler product",
        )
    })?;
    if !std::ptr::eq(execution.bytecode, &*owner) {
        return Err(crate::interpreter::errors::mex(
            "ParallelExecutableOwner",
            "serial parfor bytecode differs from its retained compiler product",
        ));
    }
    let region = execution.executable.contract.id;
    let driver = execution.context.process_driver.as_ref().ok_or_else(|| {
        crate::interpreter::errors::mex(
            "ParallelProcessDriver",
            "serial parfor requires its invocation continuation owner",
        )
    })?;
    if owner.display_name() != execution.current_function_name {
        return Err(crate::interpreter::errors::mex(
            "ParallelExecutableOwner",
            "serial parfor display context differs from its retained compiler product",
        ));
    }
    let child_runtime = runtime.clone();
    let ticket = driver.enqueue_owned_in_context(&runtime, move || async move {
        let executable = owner
            .parfor_regions
            .iter()
            .find(|candidate| candidate.contract.id == region)
            .ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "ParallelExecutableMissing",
                    "queued parfor has no matching region in its retained compiler product",
                )
            })?;
        // Keep the payload intact through fail-closed region lookup. Its root
        // lease is the last field to retire if this future is canceled or the
        // lookup fails before the child VM takes ownership.
        let ParforTaskPayload {
            inputs,
            iterations,
            randomness,
            input_roots,
        } = task;
        crate::interpreter::runner::interpret_parfor_task_in_context(
            crate::interpreter::runner::ParforTaskExecution {
                bytecode: &owner,
                bytecode_owner: Some(owner.clone()),
                region: executable,
                inputs: inputs.into(),
                iterations: iterations.into(),
                input_roots,
                randomness,
                mode: crate::interpreter::runner::ParforTaskMode::Sequential,
                current_function_name: owner.display_name(),
                runtime: child_runtime,
            },
        )
        .await
    })?;
    // Awaiting this ticket yields the outer VM poll. The invocation driver
    // polls the child independently, so two interpreter stacks never nest.
    let outputs = ticket.await?;
    outputs.with_owned_output_storage(|outputs| {
        let outputs = outputs.try_into_owned().map_err(|_| {
            runmat_runtime::sequence::sequence_error_to_runtime(
                runmat_value::ValueSequenceError::SharedOutputStorage,
            )
        })?;
        for (variable, output) in execution
            .executable
            .output_variables()
            .zip(outputs.into_iter())
        {
            if vars.get(variable.slot).is_none() {
                return Err(crate::interpreter::errors::mex(
                    "ParallelFrame",
                    "parallel task output is outside its compiled VM frame",
                ));
            }
            crate::runtime::workspace::commit_workspace_slot_value(
                vars,
                variable.slot,
                None,
                output,
            )?;
        }
        Ok::<(), RuntimeError>(())
    })?;
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
    vars: &mut InterpreterVars,
    pool: &runmat_execution::PoolHandle,
    worker_budget: u32,
) -> Result<(), RuntimeError> {
    let ParforTaskPayload {
        inputs,
        input_roots,
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
    // Every scheduled region, including one driven by an in-process pool,
    // needs its exact selected executable. `Local` is not an isolated region
    // program and cannot be reconstructed from a service's pool hint. Share
    // the already-issued graph owner across all chunks without copying the
    // compiler's Bytecode or its Value constants.
    let selected = context.graph_program.as_ref().ok_or_else(|| {
        crate::interpreter::errors::mex(
            "ExecutionProgram",
            "parallel worker execution has no selected V3 executable authority",
        )
    })?;
    let program = runmat_runtime::execution::DeferredProgram::SelectedGraph(
        selected.clone().with_validation_workspace(
            runmat_runtime::execution::value_codec::ValidationWorkspaceOwner::capture(
                &context.runtime,
            ),
        ),
    );
    let mut tasks = SubmittedTaskGroup::with_capacity(&context.runtime, graph.chunks.len());
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
        runmat_runtime::execution::validate_spawn_captures(&arguments)?;
        let future = context
            .runtime
            .execution()
            .create_future(runmat_runtime::execution::DeferredCall {
                spawn_safety: runmat_types::SpawnSafetyFact::RequiresIsolation,
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
                program_revision: program
                    .program_revision(context.runtime.program_revision())
                    .cloned(),
                environment: runmat_runtime::execution::capture_deferred_environment(
                    &context.runtime,
                )?,
                console: context
                    .runtime
                    .service_ports()
                    .host()
                    .and_then(|host| host.console_limits()),
                capabilities: executable.contract.capabilities.clone(),
                program: program.clone(),
            })
            .map_err(execution_error)?;
        match context
            .runtime
            .execution()
            .spawn_in_context(&future, Some(pool), &context.runtime)
        {
            Ok(task) => tasks.push(task),
            Err(error) => {
                // The service already owns the deferred call even though it
                // did not issue a task handle. Retire that future as well as
                // the earlier submissions owned by the group.
                let _ = context.runtime.execution().cancel(
                    &Value::Future(future.into()),
                    runmat_execution::CancellationReason::DependencyFailed,
                );
                return Err(execution_error(error));
            }
        }
    }

    let results = match await_parallel_tasks(context, &mut tasks).await {
        Ok(results) => results,
        Err(error) => {
            tasks.cancel(runmat_execution::CancellationReason::DependencyFailed);
            return Err(error);
        }
    };
    tasks.complete();
    assemble_sliced_results(
        &context.runtime,
        executable,
        &graph,
        &iterations,
        results,
        vars,
    )
    .await?;
    drop(input_roots);
    Ok(())
}

async fn await_parallel_tasks(
    context: &ExecutionContext,
    group: &mut SubmittedTaskGroup<'_>,
) -> Result<runmat_runtime::execution::RootedValues, RuntimeError> {
    use runmat_execution::TaskResultClaim;

    let mut results = vec![None; group.as_slice().len()];
    let mut remaining = group.as_slice().len();
    while remaining > 0 {
        if let Err(error) = crate::interpreter::engine::check_cancelled(&context.runtime) {
            group.cancel(runmat_execution::CancellationReason::User);
            return Err(error);
        }
        let claim = context
            .runtime
            .execution()
            .claim_next_task_result(group.as_slice())
            .map_err(execution_error)?;
        match claim {
            TaskResultClaim::Claimed { index } => {
                let Some(task) = group.as_slice().get(index) else {
                    return Err(crate::interpreter::errors::mex(
                        "ParallelTaskState",
                        "parallel scheduler returned an invalid completed-task index",
                    ));
                };
                let result = tasks::await_value(context, Value::Task(task.clone())).await;
                context
                    .runtime
                    .execution()
                    .mark_task_result_read(task)
                    .map_err(execution_error)?;
                match result {
                    Ok(value) => {
                        let slot = &mut results[index];
                        if slot.replace(value).is_some() {
                            return Err(crate::interpreter::errors::mex(
                                "ParallelTaskState",
                                "parallel scheduler completed one task more than once",
                            ));
                        }
                        remaining -= 1;
                    }
                    Err(error) => return Err(error),
                }
            }
            TaskResultClaim::Pending => tasks::yield_once(context).await?,
            TaskResultClaim::Exhausted => {
                return Err(crate::interpreter::errors::mex(
                    "ParallelTaskState",
                    "parallel scheduler exhausted results before the task graph completed",
                ));
            }
        }
    }
    let mut rooted = runmat_runtime::execution::RootedValuesBuilder::with_capacity(results.len());
    for result in results {
        let result = result.ok_or_else(|| {
            crate::interpreter::errors::mex(
                "ParallelTaskState",
                "parallel scheduler omitted a task result",
            )
        })?;
        result
            .with_owned(|value| rooted.push(value))
            .map_err(|error| error.into_runtime_error())?;
    }
    rooted.finish().map_err(|error| error.into_runtime_error())
}

async fn parallel_randomness_context(
    policy: runmat_types::ParallelRandomnessPolicy,
    runtime: &runmat_runtime::context::RuntimeContext,
    iteration_count: usize,
) -> Result<runmat_execution::ParallelRandomnessContext, RuntimeError> {
    match policy {
        runmat_types::ParallelRandomnessPolicy::Inherit => {
            Ok(runmat_execution::ParallelRandomnessContext::Inherit)
        }
        runmat_types::ParallelRandomnessPolicy::DeterministicSubstreams => {
            runmat_runtime::builtins::common::random::reserve_parallel_randomness_context_async(
                runtime,
                iteration_count,
            )
            .await
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
        runmat_execution::ParallelRandomnessContext::Mt19937InitialSeedRangeV1 {
            root,
            start,
            len,
        } => {
            let end = chunk.start.checked_add(chunk.len).ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "ParallelRandomness",
                    "parallel random-stream extent overflowed",
                )
            })?;
            if end > *len {
                return Err(crate::interpreter::errors::mex(
                    "ParallelRandomness",
                    "parallel random-stream assignment does not cover the task chunk",
                ));
            }
            let start = start.checked_add(chunk.start).ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "ParallelRandomness",
                    "parallel random-stream ordinal overflowed",
                )
            })?;
            Ok(
                runmat_execution::ParallelRandomnessContext::Mt19937InitialSeedRangeV1 {
                    root: *root,
                    start,
                    len: chunk.len,
                },
            )
        }
        runmat_execution::ParallelRandomnessContext::RunMatLcgStateRangeV1 { root, start, len } => {
            let end = chunk.start.checked_add(chunk.len).ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "ParallelRandomness",
                    "parallel random-stream extent overflowed",
                )
            })?;
            if end > *len {
                return Err(crate::interpreter::errors::mex(
                    "ParallelRandomness",
                    "parallel random-stream assignment does not cover the task chunk",
                ));
            }
            let start = start.checked_add(chunk.start).ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "ParallelRandomness",
                    "parallel random-stream ordinal overflowed",
                )
            })?;
            Ok(
                runmat_execution::ParallelRandomnessContext::RunMatLcgStateRangeV1 {
                    root: *root,
                    start,
                    len: chunk.len,
                },
            )
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
    runtime: &runmat_runtime::context::RuntimeContext,
    executable: &crate::BytecodeParforRegion,
    graph: &runmat_execution::ParallelTaskGraph,
    iterations: &[Value],
    results: runmat_runtime::execution::RootedValues,
    vars: &mut InterpreterVars,
) -> Result<(), RuntimeError> {
    let output_variables = executable.output_variables().collect::<Vec<_>>();
    for (chunk, result) in graph.chunks.iter().zip(results.iter()) {
        let Value::Cell(result) = result else {
            return Err(crate::interpreter::errors::mex(
                "ParallelTaskOutput",
                "parallel task did not return its typed result envelope",
            ));
        };
        if result.data().len() != output_variables.len() {
            return Err(crate::interpreter::errors::mex(
                "ParallelTaskOutput",
                "parallel task result does not match its compiler-owned output layout",
            ));
        }
        for (variable, worker_value) in output_variables.iter().zip(result.data()) {
            match &variable.contract.role {
                runmat_types::ParallelVariableRole::Sliced { access } => {
                    let Some(destination) = vars.get(variable.slot).map(|value| value.clone())
                    else {
                        return Err(crate::interpreter::errors::mex(
                            "ParallelFrame",
                            "parallel sliced output is outside its compiled VM frame",
                        ));
                    };
                    let offset_value = slice_offset_value(executable, *access, vars)?;
                    let mut assembled = destination;
                    for iteration in &iterations[chunk.start..chunk.start + chunk.len] {
                        assembled = runmat_runtime::parallel::assembly::merge_slice_in_context(
                            runtime,
                            assembled,
                            worker_value,
                            *access,
                            iteration,
                            offset_value.as_ref(),
                        )
                        .await?;
                    }
                    crate::runtime::workspace::commit_workspace_slot_value(
                        vars,
                        variable.slot,
                        None,
                        assembled,
                    )?;
                }
                runmat_types::ParallelVariableRole::Reduction { operator } => {
                    let Value::Cell(contributions) = worker_value else {
                        return Err(crate::interpreter::errors::mex(
                            "ParallelTaskOutput",
                            "parallel reduction task did not return per-iteration contributions",
                        ));
                    };
                    if contributions.data().len() != chunk.len {
                        return Err(crate::interpreter::errors::mex(
                            "ParallelTaskOutput",
                            "parallel reduction task returned the wrong contribution count",
                        ));
                    }
                    let Some(accumulator) = vars.get(variable.slot).map(|value| value.clone())
                    else {
                        return Err(crate::interpreter::errors::mex(
                            "ParallelFrame",
                            "parallel reduction output is outside its compiled VM frame",
                        ));
                    };
                    let mut accumulator = runmat_runtime::execution::RootedValue::new(accumulator)
                        .map_err(|error| error.into_runtime_error())?;
                    for contribution in contributions.data() {
                        accumulator = runmat_runtime::parallel::reduction::combine_in_context(
                            Some(runtime),
                            *operator,
                            accumulator,
                            contribution.clone(),
                        )
                        .await?;
                    }
                    accumulator.with_owned(|value| {
                        crate::runtime::workspace::commit_workspace_slot_value(
                            vars,
                            variable.slot,
                            None,
                            value,
                        )
                    })?;
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
    vars: &InterpreterVars,
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
    vars.get(variable.slot)
        .map(|value| value.clone())
        .map(Some)
        .ok_or_else(|| {
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
