use crate::bytecode::{program::ExecutionContext, FunctionRegistry, Instr};
use runmat_runtime::execution::value_codec::ValueCodecError;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

use super::{build_user_function_expand_multi_args, DispatchDecision, DispatchHandled};

mod collective;
mod distributed;
mod parfor;
mod spmd;
mod tasks;
pub(crate) use distributed::encode_spmd_output;

pub(super) struct ParallelDispatchContext<'a> {
    pub bytecode: &'a crate::bytecode::Bytecode,
    pub execution: &'a ExecutionContext,
    pub function_registry: &'a FunctionRegistry,
    pub current_function_name: &'a str,
}

pub(super) async fn dispatch(
    instr: &Instr,
    stack: &mut Vec<Value>,
    vars: &mut [Value],
    pc: &mut usize,
    sequence_state: &mut super::SequenceState,
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
            stack.push(Value::Future(tasks::create_future(
                execution,
                tasks::semantic_descriptor(*function, *out_count, arguments),
                function_registry,
            )?));
        }
        Instr::CreateSemanticFutureExpandMultiOutput(function, specs, out_count) => {
            let arguments = build_user_function_expand_multi_args(
                stack,
                specs,
                sequence_state,
                &execution.runtime,
            )
            .await?;
            stack.push(Value::Future(tasks::create_future(
                execution,
                tasks::semantic_descriptor(*function, *out_count, arguments),
                function_registry,
            )?));
        }
        Instr::ScheduleFeval { arg_count, on_all } => {
            tasks::schedule_feval(stack, execution, function_registry, *arg_count, *on_all)?;
        }
        Instr::Spawn => tasks::spawn(stack, execution, false)?,
        Instr::SpawnOn => tasks::spawn(stack, execution, true)?,
        Instr::Await => {
            let value = pop(stack, "await instruction expected a value on the stack")?;
            stack.push(tasks::await_value(execution, value).await?);
        }
        Instr::FetchOutputs {
            arg_count,
            requested_outputs,
        } => {
            let mut arguments = crate::call::builtins::collect_call_args(stack, *arg_count)?;
            let futures = arguments.remove(0);
            let uniform_output = tasks::fetch_outputs_options(&arguments)?;
            stack.push(
                tasks::fetch_outputs(execution, &futures, *requested_outputs, uniform_output)
                    .await?,
            );
        }
        Instr::FetchNext {
            has_timeout,
            requested_outputs,
        } => {
            let timeout = has_timeout
                .then(|| pop(stack, "fetchNext expected a timeout on the stack"))
                .transpose()?
                .map(tasks::timeout_seconds)
                .transpose()?;
            let futures = pop(stack, "fetchNext expected a future array on the stack")?;
            stack.push(tasks::fetch_next(execution, &futures, timeout, *requested_outputs).await?);
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
                .map(parfor::maximum_worker_count)
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
            parfor::execute(
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
            spmd::execute(
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
            stack.push(collective::execute(bytecode, *id, *operation, arguments, execution).await?);
        }
        Instr::Distributed(operation) => {
            let arguments =
                crate::call::builtins::collect_call_args(stack, operation.operand_count())?;
            stack.push(distributed::execute(bytecode, operation, arguments, execution).await?);
        }
        _ => return Ok(None),
    }
    Ok(Some(DispatchHandled::Generic(
        DispatchDecision::FallThrough,
    )))
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

fn pop(stack: &mut Vec<Value>, message: &str) -> Result<Value, RuntimeError> {
    stack
        .pop()
        .ok_or_else(|| crate::interpreter::errors::mex("StackUnderflow", message))
}

fn execution_error(error: runmat_runtime::execution::ExecutionServiceError) -> RuntimeError {
    error.into_runtime_error()
}
