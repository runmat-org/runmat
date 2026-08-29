use crate::bytecode::{program::ExecutionContext, FunctionRegistry, Instr};
use crate::{InterpreterOutcome, InterpreterResumeState};
use runmat_runtime::execution::value_codec::{encode_inline_value, ValueCodecError};
use runmat_runtime::RuntimeError;
use runmat_value::Value;
use std::collections::{HashMap, HashSet};

use super::{build_user_function_expand_multi_args, calls, DispatchDecision, DispatchHandled};

pub(super) struct ParallelDispatchContext<'a> {
    pub bytecode: &'a crate::bytecode::Bytecode,
    pub execution: &'a ExecutionContext,
    pub function_registry: &'a FunctionRegistry,
    pub current_function_name: &'a str,
}

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

pub(super) async fn dispatch(
    instr: &Instr,
    stack: &mut Vec<Value>,
    vars: &mut [Value],
    pc: &mut usize,
    context: ParallelDispatchContext<'_>,
) -> Result<Option<DispatchHandled>, RuntimeError> {
    let ParallelDispatchContext {
        bytecode,
        execution,
        function_registry,
        current_function_name,
    } = context;
    match instr {
        Instr::CreateSemanticFuture(function, arg_count, out_count) => {
            let arguments = crate::call::builtins::collect_call_args(stack, *arg_count)?;
            stack.push(Value::Future(create_future(
                execution,
                semantic_descriptor(*function, *out_count, arguments),
                function_registry,
            )?));
        }
        Instr::CreateSemanticFutureExpandMultiOutput(function, specs, out_count) => {
            let arguments =
                build_user_function_expand_multi_args(stack, specs, &execution.runtime).await?;
            stack.push(Value::Future(create_future(
                execution,
                semantic_descriptor(*function, *out_count, arguments),
                function_registry,
            )?));
        }
        Instr::ScheduleFeval { arg_count, on_all } => {
            schedule_feval(stack, execution, function_registry, *arg_count, *on_all)?;
        }
        Instr::Spawn => spawn(stack, execution, false)?,
        Instr::SpawnOn => spawn(stack, execution, true)?,
        Instr::Await => {
            let value = pop(stack, "await instruction expected a value on the stack")?;
            stack.push(await_value(execution, value).await?);
        }
        Instr::FetchOutputs {
            arg_count,
            requested_outputs,
        } => {
            let mut arguments = crate::call::builtins::collect_call_args(stack, *arg_count)?;
            let futures = arguments.remove(0);
            let uniform_output = fetch_outputs_options(&arguments)?;
            stack.push(
                fetch_outputs(execution, &futures, *requested_outputs, uniform_output).await?,
            );
        }
        Instr::FetchNext {
            has_timeout,
            requested_outputs,
        } => {
            let timeout = has_timeout
                .then(|| pop(stack, "fetchNext expected a timeout on the stack"))
                .transpose()?
                .map(timeout_seconds)
                .transpose()?;
            let futures = pop(stack, "fetchNext expected a future array on the stack")?;
            stack.push(fetch_next(execution, &futures, timeout, *requested_outputs).await?);
        }
        Instr::EnsurePool(arg_count) => {
            let arguments = crate::call::builtins::collect_call_args(stack, *arg_count)?;
            stack.push(runmat_runtime::parallel::pool::ensure(
                &execution.runtime,
                &arguments,
            )?);
        }
        Instr::CurrentPool(arg_count) => {
            let arguments = crate::call::builtins::collect_call_args(stack, *arg_count)?;
            stack.push(runmat_runtime::parallel::pool::current(
                &execution.runtime,
                &arguments,
            )?);
        }
        Instr::ExecuteParfor {
            region,
            has_maximum_workers,
        } => {
            let maximum_workers = has_maximum_workers
                .then(|| pop(stack, "parfor expected a maximum worker count on the stack"))
                .transpose()?
                .map(maximum_worker_count)
                .transpose()?;
            let iterable = pop(stack, "parfor expected an iterable on the stack")?;
            let executable = bytecode
                .parfor_regions
                .iter()
                .find(|candidate| candidate.contract.id == *region)
                .cloned()
                .ok_or_else(|| {
                    crate::interpreter::errors::mex(
                        "ParallelExecutableMissing",
                        "parfor has no compiler-bound executable region",
                    )
                })?;
            execute_parfor(
                bytecode,
                &executable,
                iterable,
                maximum_workers,
                vars,
                execution,
                current_function_name,
            )
            .await?;
            *pc = executable.exit.pc;
            return Ok(Some(DispatchHandled::Generic(
                DispatchDecision::ContinueLoop,
            )));
        }
        Instr::ExecuteSpmd { region, header } => {
            let executable = bytecode
                .spmd_regions
                .iter()
                .find(|candidate| candidate.contract.id == *region)
                .cloned()
                .ok_or_else(|| {
                    crate::interpreter::errors::mex(
                        "SpmdExecutableMissing",
                        "SPMD has no compiler-bound executable region",
                    )
                })?;
            let operands = crate::call::builtins::collect_call_args(stack, header.operand_count())?;
            execute_spmd(
                bytecode,
                &executable,
                *header,
                operands,
                vars,
                execution,
                current_function_name,
            )
            .await?;
            *pc = executable.exit.pc;
            return Ok(Some(DispatchHandled::Generic(
                DispatchDecision::ContinueLoop,
            )));
        }
        Instr::Collective { id, operation } => {
            let arguments =
                crate::call::builtins::collect_call_args(stack, operation.operand_count())?;
            stack.push(execute_collective(bytecode, *id, *operation, arguments, execution).await?);
        }
        Instr::Distributed(operation) => {
            let arguments =
                crate::call::builtins::collect_call_args(stack, operation.operand_count())?;
            stack.push(execute_distributed(bytecode, operation, arguments, execution).await?);
        }
        _ => return Ok(None),
    }
    Ok(Some(DispatchHandled::Generic(
        DispatchDecision::FallThrough,
    )))
}

async fn execute_distributed(
    bytecode: &crate::Bytecode,
    operation: &crate::BytecodeDistributedOp,
    arguments: Vec<Value>,
    execution: &ExecutionContext,
) -> Result<Value, RuntimeError> {
    use crate::BytecodeDistributedOp as Op;

    let service = execution
        .runtime
        .service_ports()
        .require_distributed("distributed value operation")
        .map_err(capability_error)?
        .clone();
    match operation {
        Op::Create { id, owner, scheme } => {
            let [input] = distributed_arguments(arguments)?;
            let contract = bytecode
                .distributed_values
                .iter()
                .find(|contract| contract.id == *id)
                .filter(|contract| contract.owner == *owner && contract.scheme == *scheme)
                .cloned()
                .ok_or_else(|| {
                    crate::interpreter::errors::mex(
                        "DistributedContractMissing",
                        "distributed instruction has no matching compiler-owned semantic contract",
                    )
                })?;
            let pool = execution
                .runtime
                .execution()
                .ensure_pool(runmat_execution::PoolRequest::automatic())
                .map_err(execution_error)?;
            service
                .create(contract, input, pool)
                .await
                .map(|handle| Value::Distributed(Box::new(handle)))
        }
        Op::LocalPart => {
            let [input] = distributed_arguments(arguments)?;
            let Value::Distributed(handle) = input else {
                return Err(crate::interpreter::errors::mex(
                    "DistributedValueRequired",
                    "getLocalPart requires a distributed value",
                ));
            };
            runmat_runtime::parallel::lease::validate_distributed(&execution.runtime, &handle)?;
            let rank = if let Some(collective) = execution.runtime.service_ports().collective() {
                collective.context().rank
            } else if handle.partition_count.0 == 1 {
                runmat_types::LabRank(1)
            } else {
                return Err(crate::interpreter::errors::mex(
                    "DistributedRankContextRequired",
                    "getLocalPart requires an SPMD rank context for a multi-partition value",
                ));
            };
            service.local_part(*handle, rank).await
        }
        Op::Materialize => {
            let [input] = distributed_arguments(arguments)?;
            let Value::Distributed(handle) = input else {
                return Err(crate::interpreter::errors::mex(
                    "DistributedValueRequired",
                    "gather requires a distributed value on this execution path",
                ));
            };
            runmat_runtime::parallel::lease::validate_distributed(&execution.runtime, &handle)?;
            service.materialize(*handle).await
        }
        Op::Codistributor => {
            let [input] = distributed_arguments(arguments)?;
            let Value::Distributed(handle) = input else {
                return Err(crate::interpreter::errors::mex(
                    "DistributedValueRequired",
                    "getCodistributor requires a distributed value",
                ));
            };
            runmat_runtime::parallel::lease::validate_distributed(&execution.runtime, &handle)?;
            runmat_runtime::parallel::codistributor::from_scheme(
                &handle.scheme,
                &handle.global_shape,
            )
        }
        Op::Redistribute => {
            let [input, codistributor] = distributed_arguments(arguments)?;
            let Value::Distributed(handle) = input else {
                return Err(crate::interpreter::errors::mex(
                    "DistributedValueRequired",
                    "redistribute requires a distributed value",
                ));
            };
            runmat_runtime::parallel::lease::validate_distributed(&execution.runtime, &handle)?;
            let scheme = runmat_runtime::parallel::codistributor::resolve(
                &codistributor,
                &handle.global_shape,
                handle.partition_count,
            )?;
            service
                .redistribute(*handle, scheme)
                .await
                .map(|handle| Value::Distributed(Box::new(handle)))
        }
    }
}

fn distributed_arguments<const N: usize>(
    arguments: Vec<Value>,
) -> Result<[Value; N], RuntimeError> {
    arguments.try_into().map_err(|_| {
        crate::interpreter::errors::mex(
            "InvalidDistributedInstruction",
            "distributed instruction operand count does not match its bytecode contract",
        )
    })
}

async fn execute_collective(
    bytecode: &crate::Bytecode,
    id: runmat_types::CollectiveId,
    operation: crate::BytecodeCollectiveOp,
    arguments: Vec<Value>,
    execution: &ExecutionContext,
) -> Result<Value, RuntimeError> {
    let contract = bytecode
        .collective_contracts
        .iter()
        .find(|contract| contract.id == id)
        .ok_or_else(|| {
            crate::interpreter::errors::mex(
                "CollectiveContractMissing",
                "collective instruction has no compiler-owned semantic contract",
            )
        })?;
    if !collective_operation_matches(&contract.operation, operation) {
        return Err(crate::interpreter::errors::mex(
            "CollectiveContractMismatch",
            "collective instruction disagrees with its compiler-owned semantic contract",
        ));
    }
    let collective = execution
        .runtime
        .service_ports()
        .require_collective("SPMD collective")
        .map_err(capability_error)?
        .clone();
    let sequence = collective.next_sequence(id)?;
    let invocation = collective_invocation(operation, arguments, collective.context())?;
    let response = collective
        .execute(runmat_execution::CollectiveRequest {
            context: collective.context().clone(),
            id,
            sequence,
            invocation,
        })
        .await?;
    collective_response(response, operation, &bytecode.function_registry).await
}

fn collective_operation_matches(
    contract: &runmat_types::CollectiveOperation,
    instruction: crate::BytecodeCollectiveOp,
) -> bool {
    use crate::BytecodeCollectiveOp as Bytecode;
    use runmat_types::CollectiveOperation as Contract;

    match (contract, instruction) {
        (Contract::Barrier, Bytecode::Barrier)
        | (Contract::Broadcast, Bytecode::Broadcast { .. })
        | (Contract::Gather, Bytecode::Gather)
        | (Contract::Scatter, Bytecode::Scatter)
        | (Contract::AllGather, Bytecode::AllGather)
        | (Contract::Cat, Bytecode::Cat { .. })
        | (Contract::FunctionalReduce, Bytecode::FunctionalReduce { .. })
        | (Contract::Send, Bytecode::Send { .. })
        | (Contract::SendReceive, Bytecode::SendReceive { .. })
        | (Contract::Probe, Bytecode::Probe { .. }) => true,
        (
            Contract::Receive {
                requested_outputs: expected,
            },
            Bytecode::Receive {
                requested_outputs, ..
            },
        ) => *expected == requested_outputs,
        (Contract::Reduce { operator: expected }, Bytecode::Reduce { operator })
        | (Contract::AllReduce { operator: expected }, Bytecode::AllReduce { operator }) => {
            *expected == operator
        }
        _ => false,
    }
}

fn collective_invocation(
    operation: crate::BytecodeCollectiveOp,
    mut arguments: Vec<Value>,
    context: &runmat_execution::SpmdTaskContext,
) -> Result<runmat_execution::CollectiveInvocation, RuntimeError> {
    use crate::BytecodeCollectiveOp as Op;
    use runmat_execution::{CollectiveInvocation, CollectiveMessageTag, ReceiveSelection};

    let encode = |value: &Value| encode_inline_value(value).map_err(value_codec_error);
    Ok(match operation {
        Op::Barrier => CollectiveInvocation::Barrier,
        Op::Broadcast { has_input } => {
            let input = has_input.then(|| arguments.remove(0));
            let root = lab_rank(arguments.remove(0), context)?;
            CollectiveInvocation::Broadcast {
                root,
                value: if context.rank == root {
                    input.as_ref().map(encode).transpose()?
                } else {
                    None
                },
            }
        }
        Op::Gather => {
            let value = encode(&arguments.remove(0))?;
            let root = lab_rank(arguments.remove(0), context)?;
            CollectiveInvocation::Gather { root, value }
        }
        Op::Scatter => {
            let value = arguments.remove(0);
            let root = lab_rank(arguments.remove(0), context)?;
            let values = if context.rank == root {
                let Value::Cell(values) = value else {
                    return Err(crate::interpreter::errors::mex(
                        "CollectiveScatterInput",
                        "scatter root must provide one cell entry per lab",
                    ));
                };
                Some(
                    values
                        .data
                        .iter()
                        .map(encode)
                        .collect::<Result<Vec<_>, _>>()?,
                )
            } else {
                None
            };
            CollectiveInvocation::Scatter { root, values }
        }
        Op::AllGather => CollectiveInvocation::AllGather {
            value: encode(&arguments.remove(0))?,
        },
        Op::Reduce { operator } => {
            let value = encode(&arguments.remove(0))?;
            let root = lab_rank(arguments.remove(0), context)?;
            CollectiveInvocation::Reduce {
                root: Some(root),
                operator,
                value,
            }
        }
        Op::AllReduce { operator } => CollectiveInvocation::Reduce {
            root: None,
            operator,
            value: encode(&arguments.remove(0))?,
        },
        Op::Cat { has_root } => {
            let value = encode(&arguments.remove(0))?;
            let dimension = positive_u32(arguments.remove(0), "concatenation dimension")?;
            let root = has_root
                .then(|| lab_rank(arguments.remove(0), context))
                .transpose()?;
            CollectiveInvocation::Cat {
                root,
                dimension,
                value,
            }
        }
        Op::FunctionalReduce { has_root } => {
            let reducer = encode(&arguments.remove(0))?;
            let value = encode(&arguments.remove(0))?;
            let root = has_root
                .then(|| optional_lab_rank(arguments.remove(0), context))
                .transpose()?
                .flatten();
            CollectiveInvocation::FunctionalReduce {
                root,
                reducer,
                value,
            }
        }
        Op::Send { has_tag } => {
            let value = encode(&arguments.remove(0))?;
            let destination = lab_rank(arguments.remove(0), context)?;
            let tag = if has_tag {
                message_tag(arguments.remove(0))?
            } else {
                CollectiveMessageTag(0)
            };
            CollectiveInvocation::Send {
                destination,
                tag,
                value,
            }
        }
        Op::Receive {
            has_source,
            has_tag,
            ..
        }
        | Op::Probe {
            has_source,
            has_tag,
        } => {
            let source = if has_source {
                receive_source(arguments.remove(0), context)?
            } else {
                None
            };
            let tag = has_tag
                .then(|| message_tag(arguments.remove(0)))
                .transpose()?;
            let selection = ReceiveSelection { source, tag };
            if matches!(operation, Op::Receive { .. }) {
                CollectiveInvocation::Receive { selection }
            } else {
                CollectiveInvocation::Probe { selection }
            }
        }
        Op::SendReceive { has_tag } => {
            let destination = optional_lab_rank(arguments.remove(0), context)?;
            let source = optional_lab_rank(arguments.remove(0), context)?;
            let value = encode(&arguments.remove(0))?;
            let tag = if has_tag {
                message_tag(arguments.remove(0))?
            } else {
                CollectiveMessageTag(0)
            };
            CollectiveInvocation::SendReceive {
                destination,
                source,
                tag,
                value,
            }
        }
    })
}

async fn collective_response(
    response: runmat_execution::CollectiveResponse,
    operation: crate::BytecodeCollectiveOp,
    function_registry: &crate::FunctionRegistry,
) -> Result<Value, RuntimeError> {
    use runmat_execution::CollectiveResponse;

    match response {
        CollectiveResponse::Complete => Ok(empty_value()),
        CollectiveResponse::Value { value } => {
            runmat_runtime::execution::value_codec::decode_inline_value(&value)
                .map_err(value_codec_error)
        }
        CollectiveResponse::Received { value, source, tag } => {
            let value = runmat_runtime::execution::value_codec::decode_inline_value(&value)
                .map_err(value_codec_error)?;
            let requested_outputs = match operation {
                crate::BytecodeCollectiveOp::Receive {
                    requested_outputs, ..
                } => requested_outputs,
                crate::BytecodeCollectiveOp::SendReceive { .. } => 1,
                _ => 1,
            };
            if requested_outputs == 1 {
                Ok(value)
            } else {
                let mut outputs = vec![value, Value::Num(f64::from(source.0))];
                if requested_outputs == 3 {
                    outputs.push(Value::Int(runmat_value::IntValue::U64(tag.0)));
                }
                Ok(Value::OutputList(outputs))
            }
        }
        CollectiveResponse::Values { values } => {
            let count = values.len();
            let values = values
                .iter()
                .map(runmat_runtime::execution::value_codec::decode_inline_value)
                .collect::<Result<Vec<_>, _>>()
                .map_err(value_codec_error)?;
            runmat_value::CellArray::new_with_shape(values, vec![count, 1])
                .map(Value::Cell)
                .map_err(|error| {
                    runmat_runtime::runtime_error::semantic_error("RunMat:CollectiveResult", error)
                })
        }
        CollectiveResponse::ReductionInputs { operator, values } => {
            let mut values = values
                .iter()
                .map(runmat_runtime::execution::value_codec::decode_inline_value)
                .collect::<Result<Vec<_>, _>>()
                .map_err(value_codec_error)?
                .into_iter();
            let mut accumulator = values.next().ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "CollectiveReductionEmpty",
                    "collective reduction received no lab contributions",
                )
            })?;
            for contribution in values {
                accumulator = runmat_runtime::parallel::reduction::combine(
                    operator,
                    accumulator,
                    contribution,
                )
                .await?;
            }
            Ok(accumulator)
        }
        CollectiveResponse::ConcatenationInputs { dimension, values } => {
            if values.len() == 1 {
                return runmat_runtime::execution::value_codec::decode_inline_value(&values[0])
                    .map_err(value_codec_error);
            }
            let mut arguments = Vec::with_capacity(values.len() + 1);
            arguments.push(Value::Int(runmat_value::IntValue::U32(dimension)));
            arguments.extend(
                values
                    .iter()
                    .map(runmat_runtime::execution::value_codec::decode_inline_value)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(value_codec_error)?,
            );
            runmat_runtime::call::descriptor::execute_callable_descriptor(
                runmat_runtime::call::descriptor::CallableDescriptor::resolved(
                    runmat_hir::CallableIdentity::Builtin(runmat_types::BuiltinId("cat".into())),
                    arguments,
                    1,
                    runmat_hir::CallableFallbackPolicy::None,
                    runmat_runtime::call::descriptor::CallableCallKind::Direct,
                ),
            )
            .await
        }
        CollectiveResponse::FunctionalReductionInputs { reducer, values } => {
            let reducer = runmat_runtime::execution::value_codec::decode_inline_value(&reducer)
                .map_err(value_codec_error)?;
            let mut values = values
                .iter()
                .map(runmat_runtime::execution::value_codec::decode_inline_value)
                .collect::<Result<Vec<_>, _>>()
                .map_err(value_codec_error)?
                .into_iter();
            let mut accumulator = values.next().ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "CollectiveReductionEmpty",
                    "functional reduction received no lab contributions",
                )
            })?;
            for contribution in values {
                let descriptor =
                    runmat_runtime::call::descriptor::CallableDescriptor::from_feval_value(
                        reducer.clone(),
                        vec![accumulator, contribution],
                        1,
                        function_registry,
                    );
                accumulator =
                    runmat_runtime::call::descriptor::execute_callable_descriptor(descriptor)
                        .await?;
            }
            Ok(accumulator)
        }
        CollectiveResponse::Probe { available } => Ok(Value::Bool(available)),
    }
}

fn receive_source(
    value: Value,
    context: &runmat_execution::SpmdTaskContext,
) -> Result<Option<runmat_types::LabRank>, RuntimeError> {
    match value {
        Value::String(value) if value.eq_ignore_ascii_case("any") => Ok(None),
        Value::CharArray(value)
            if value
                .row_string()
                .is_some_and(|value| value.eq_ignore_ascii_case("any")) =>
        {
            Ok(None)
        }
        value => lab_rank(value, context).map(Some),
    }
}

fn optional_lab_rank(
    value: Value,
    context: &runmat_execution::SpmdTaskContext,
) -> Result<Option<runmat_types::LabRank>, RuntimeError> {
    if is_empty_value(&value) {
        Ok(None)
    } else {
        lab_rank(value, context).map(Some)
    }
}

fn is_empty_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Tensor(value) if value.is_empty()
    ) || matches!(value, Value::Cell(value) if value.data.is_empty())
        || matches!(value, Value::LogicalArray(value) if value.is_empty())
}

fn empty_value() -> Value {
    Value::Tensor(runmat_value::Tensor::zeros(vec![0, 0]))
}

fn lab_rank(
    value: Value,
    context: &runmat_execution::SpmdTaskContext,
) -> Result<runmat_types::LabRank, RuntimeError> {
    let rank = positive_u32(value, "lab rank")?;
    if rank > context.gang.labs.0 {
        return Err(crate::interpreter::errors::mex(
            "CollectiveRankOutsideGang",
            "collective lab rank lies outside the admitted gang",
        ));
    }
    Ok(runmat_types::LabRank(rank))
}

fn message_tag(value: Value) -> Result<runmat_execution::CollectiveMessageTag, RuntimeError> {
    let tag = match value {
        Value::Int(value) => value.try_to_u64(),
        Value::Num(value)
            if value.is_finite()
                && value.fract() == 0.0
                && value >= 0.0
                && value <= u64::MAX as f64 =>
        {
            Some(value as u64)
        }
        _ => None,
    }
    .ok_or_else(|| {
        crate::interpreter::errors::mex(
            "CollectiveTagInvalid",
            "collective message tag must be a nonnegative integer scalar",
        )
    })?;
    Ok(runmat_execution::CollectiveMessageTag(tag))
}

fn positive_u32(value: Value, label: &str) -> Result<u32, RuntimeError> {
    let value = match value {
        Value::Int(value) => value.try_to_u64(),
        Value::Num(value)
            if value.is_finite()
                && value.fract() == 0.0
                && value >= 1.0
                && value <= u32::MAX as f64 =>
        {
            Some(value as u64)
        }
        _ => None,
    };
    value
        .filter(|value| *value > 0)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| {
            runmat_runtime::runtime_error::semantic_error(
                "RunMat:CollectiveIntegerInvalid",
                format!("{label} must be a positive integer scalar"),
            )
        })
}

async fn execute_spmd(
    bytecode: &crate::Bytecode,
    executable: &crate::BytecodeSpmdRegion,
    header: crate::BytecodeSpmdHeader,
    operands: Vec<Value>,
    vars: &mut [Value],
    execution: &ExecutionContext,
    current_function_name: &str,
) -> Result<(), RuntimeError> {
    let (pool, labs) = spmd_request(header, operands, execution)?;
    let available_labs = runmat_types::LabCount(
        execution
            .runtime
            .execution()
            .current_pool()
            .map_err(execution_error)?
            .filter(|snapshot| snapshot.handle == pool)
            .map(|snapshot| snapshot.workers)
            .ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "SpmdPoolUnavailable",
                    "SPMD requires a live pool in the current execution scope",
                )
            })?,
    );
    let captures = executable
        .captures
        .iter()
        .map(|capture| {
            vars.get(capture.slot).cloned().ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "SpmdFrame",
                    "SPMD capture is outside its compiler-bound VM frame",
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let spmd = execution
        .runtime
        .service_ports()
        .require_spmd("SPMD execution")
        .map_err(capability_error)?
        .clone();
    let admission = spmd
        .admit(
            runmat_execution::GangRequest { pool, labs },
            available_labs,
            executable.contract.id,
        )
        .await?;
    admission.validate()?;
    let gang = admission.gang.handle.clone();
    let execution_result = async {
        let rank_results = match execution.runtime.execution().spmd_execution_mode() {
            runmat_runtime::execution::SpmdExecutionMode::IsolatedWorkers => {
                runmat_runtime::execution::validate_spawn_capture(&Value::OutputList(
                    captures.clone(),
                ))?;
                let program = serde_json::to_vec(bytecode).map_err(|error| {
                    crate::interpreter::errors::mex(
                        "ExecutionProgram",
                        &format!("failed to encode the exact SPMD program: {error}"),
                    )
                })?;
                execution
                    .runtime
                    .execution()
                    .execute_spmd_gang(runmat_runtime::execution::SpmdGangCall {
                        gang: gang.clone(),
                        region: executable.contract.id,
                        captures: captures.clone(),
                        requested_outputs: executable.outputs.len(),
                        program_revision: execution.runtime.program_revision().cloned(),
                        program,
                    })
                    .await
                    .map_err(execution_error)?
            }
            runmat_runtime::execution::SpmdExecutionMode::Cooperative => {
                execute_spmd_in_process(
                    bytecode,
                    executable,
                    captures,
                    admission.labs,
                    execution,
                    current_function_name,
                )
                .await?
            }
        };
        validate_spmd_rank_results(&gang, executable.outputs.len(), &rank_results)?;
        let outputs = executable
            .outputs
            .iter()
            .enumerate()
            .map(|(output_index, output)| {
                let entries = rank_results
                    .iter()
                    .map(|rank| rank.outputs[output_index].clone())
                    .collect();
                Ok(runmat_runtime::context::RuntimeSpmdOutput {
                    value: output.contract.value,
                    fact: output.contract.fact.clone(),
                    entries,
                })
            })
            .collect::<Result<Vec<_>, RuntimeError>>()?;
        let handles = spmd
            .retain_outputs(gang.clone(), executable.contract.id, outputs)
            .await?;
        if handles.len() != executable.outputs.len() {
            return Err(crate::interpreter::errors::mex(
                "SpmdOutputContract",
                "SPMD runtime returned a different number of outputs than the compiler contract",
            ));
        }
        for (output, handle) in executable.outputs.iter().zip(handles) {
            let destination = vars.get_mut(output.slot).ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "SpmdFrame",
                    "SPMD output is outside its compiler-bound VM frame",
                )
            })?;
            *destination = Value::Composite(Box::new(handle));
            crate::runtime::workspace::mark_workspace_assigned(output.slot);
        }
        Ok(())
    }
    .await;
    let retirement = spmd.retire(gang).await;
    match execution_result {
        Err(error) => Err(error),
        Ok(()) => retirement,
    }
}

async fn execute_spmd_in_process(
    bytecode: &crate::Bytecode,
    executable: &crate::BytecodeSpmdRegion,
    captures: Vec<Value>,
    labs: Vec<std::rc::Rc<dyn runmat_runtime::context::RuntimeCollectiveService>>,
    execution: &ExecutionContext,
    current_function_name: &str,
) -> Result<Vec<runmat_runtime::execution::SpmdRankResult>, RuntimeError> {
    let mut tasks = Vec::with_capacity(labs.len());
    for collective in labs {
        let rank = collective.context().rank;
        let gang = collective.context().gang.clone();
        let mut frame = vec![None; bytecode.var_count];
        for (capture, value) in executable.captures.iter().zip(&captures) {
            frame[capture.slot] = Some(value.clone());
        }
        let services = execution
            .runtime
            .service_ports()
            .clone()
            .with_collective(collective);
        let runtime = execution.runtime.fork_parallel_lab(services);
        let spmd = execution
            .runtime
            .service_ports()
            .require_spmd("SPMD execution")
            .map_err(capability_error)?
            .clone();
        let bytecode = bytecode.clone();
        let function_name = current_function_name.to_string();
        let body_pc = executable.body.pc;
        tasks.push(async move {
            let resume = InterpreterResumeState {
                pc: body_pc,
                completion_boundary: Some(
                    crate::interpreter::state::InterpreterCompletionBoundary::before(
                        executable.exit.pc,
                    ),
                ),
                vars: frame,
                supplied_inputs: 0,
                requested_outputs: 0,
                missing_input_slots: HashSet::new(),
                global_aliases: HashMap::new(),
                persistent_aliases: HashMap::new(),
                side_effect_epoch: 0,
            };
            let result = crate::interpreter::runner::interpret_resume_in_context(
                &bytecode,
                resume,
                Some(&function_name),
                runtime,
            )
            .await;
            match result {
                Ok(InterpreterOutcome::Completed(completion)) => {
                    spmd.rank_finished(&gang, rank)?;
                    Ok((rank, completion))
                }
                Err(error) => {
                    // Wake peers that may be waiting in a collective, but keep
                    // the rank's source-mapped failure as the public result.
                    let _ = spmd.rank_failed(&gang, rank);
                    Err(error)
                }
            }
        });
    }
    futures::future::try_join_all(tasks)
        .await?
        .into_iter()
        .map(|(rank, completion)| {
            let outputs = executable
                .outputs
                .iter()
                .map(|output| {
                    if !completion.assigned_slots.contains(&output.slot) {
                        return Ok(None);
                    }
                    let value = completion.values.get(output.slot).ok_or_else(|| {
                        crate::interpreter::errors::mex(
                            "SpmdFrame",
                            "an assigned SPMD output is outside its completed VM frame",
                        )
                    })?;
                    encode_inline_value(value)
                        .map(Some)
                        .map_err(value_codec_error)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(runmat_runtime::execution::SpmdRankResult { rank, outputs })
        })
        .collect()
}

fn validate_spmd_rank_results(
    gang: &runmat_execution::GangHandle,
    output_count: usize,
    ranks: &[runmat_runtime::execution::SpmdRankResult],
) -> Result<(), RuntimeError> {
    if ranks.len() != gang.labs.0 as usize {
        return Err(crate::interpreter::errors::mex(
            "SpmdOutputContract",
            "SPMD execution returned a different number of ranks than the admitted gang",
        ));
    }
    for (index, result) in ranks.iter().enumerate() {
        let expected_rank = runmat_types::LabRank(u32::try_from(index + 1).map_err(|_| {
            crate::interpreter::errors::mex(
                "SpmdOutputContract",
                "SPMD rank index exceeds the supported range",
            )
        })?);
        if result.rank != expected_rank || result.outputs.len() != output_count {
            return Err(crate::interpreter::errors::mex(
                "SpmdOutputContract",
                "SPMD rank results differ from the admitted rank or compiler output contract",
            ));
        }
    }
    Ok(())
}

fn spmd_request(
    header: crate::BytecodeSpmdHeader,
    mut operands: Vec<Value>,
    execution: &ExecutionContext,
) -> Result<
    (
        runmat_execution::PoolHandle,
        runmat_types::SpmdLabRequirement,
    ),
    RuntimeError,
> {
    use runmat_types::{LabCount, SpmdLabRequirement};

    let explicit_pool = matches!(header, crate::BytecodeSpmdHeader::PoolRange);
    let pool = if explicit_pool {
        pool_handle(operands.remove(0))?
    } else {
        execution
            .runtime
            .execution()
            .ensure_pool(runmat_execution::PoolRequest::automatic())
            .map_err(execution_error)?
            .handle
    };
    let labs = match header {
        crate::BytecodeSpmdHeader::Default => SpmdLabRequirement::Default,
        crate::BytecodeSpmdHeader::Exact => SpmdLabRequirement::Exact {
            labs: LabCount(spmd_lab_count(operands.remove(0))?),
        },
        crate::BytecodeSpmdHeader::Range | crate::BytecodeSpmdHeader::PoolRange => {
            let minimum = spmd_lab_count(operands.remove(0))?;
            let maximum = spmd_lab_count(operands.remove(0))?;
            if minimum > maximum {
                return Err(crate::interpreter::errors::mex(
                    "InvalidSpmdRange",
                    "SPMD minimum lab count cannot exceed its maximum",
                ));
            }
            SpmdLabRequirement::Range {
                minimum: LabCount(minimum),
                maximum: LabCount(maximum),
            }
        }
    };
    Ok((pool, labs))
}

fn spmd_lab_count(value: Value) -> Result<u32, RuntimeError> {
    let count = match value {
        Value::Int(value) => value.try_to_u64(),
        Value::Num(value)
            if value.is_finite()
                && value.fract() == 0.0
                && value >= 1.0
                && value <= u32::MAX as f64 =>
        {
            Some(value as u64)
        }
        _ => None,
    };
    count
        .filter(|count| *count > 0)
        .and_then(|count| u32::try_from(count).ok())
        .ok_or_else(|| {
            crate::interpreter::errors::mex(
                "InvalidSpmdLabCount",
                "SPMD lab counts must be positive integer scalars",
            )
        })
}

fn capability_error(error: runmat_runtime::context::RuntimeCapabilityError) -> RuntimeError {
    runmat_runtime::runtime_error::semantic_error(
        "RunMat:RuntimeCapabilityUnavailable",
        error.to_string(),
    )
}

fn value_codec_error(error: ValueCodecError) -> RuntimeError {
    runmat_runtime::runtime_error::semantic_error("RunMat:ParallelValueEncoding", error.to_string())
}

async fn execute_parfor(
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
                let result = await_value(context, Value::Task(task.clone())).await;
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
            TaskResultClaim::Pending => yield_once().await,
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

fn maximum_worker_count(value: Value) -> Result<u32, RuntimeError> {
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

fn schedule_feval(
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

fn spawn(
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

fn create_future(
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

fn semantic_descriptor(
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

async fn await_value(context: &ExecutionContext, value: Value) -> Result<Value, RuntimeError> {
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

async fn fetch_next(
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

async fn fetch_outputs(
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

fn fetch_outputs_options(arguments: &[Value]) -> Result<bool, RuntimeError> {
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

fn timeout_seconds(value: Value) -> Result<f64, RuntimeError> {
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

async fn yield_once() {
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

fn pool_handle(value: Value) -> Result<runmat_execution::PoolHandle, RuntimeError> {
    match value {
        Value::Pool(pool) => Ok(pool),
        _ => Err(crate::interpreter::errors::mex(
            "PoolOperandInvalid",
            "parallel scheduling expects a valid pool",
        )),
    }
}

fn pop(stack: &mut Vec<Value>, message: &str) -> Result<Value, RuntimeError> {
    stack
        .pop()
        .ok_or_else(|| crate::interpreter::errors::mex("StackUnderflow", message))
}

fn execution_error(error: runmat_runtime::execution::ExecutionServiceError) -> RuntimeError {
    match error {
        runmat_runtime::execution::ExecutionServiceError::RuntimeFailure(failure) => {
            runmat_runtime::execution::decode_runtime_failure(*failure).unwrap_or_else(|error| {
                crate::interpreter::errors::mex("ExecutionProtocol", &error)
            })
        }
        error => crate::interpreter::errors::mex("ExecutionService", &error.to_string()),
    }
}
