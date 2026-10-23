use crate::bytecode::{program::ExecutionContext, FunctionRegistry, Instr};
use crate::interpreter::vars::InterpreterVars;
use runmat_runtime::execution::value_codec::ValueCodecError;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

use super::future::control;
use super::{build_user_function_expand_multi_args, calls, DispatchDecision, DispatchHandled};

mod data_dispatch;
mod future_creation;
mod future_wait;
mod pool_dispatch;
mod region_dispatch;

mod collective;
mod distributed;
mod parfor;
mod spmd;
mod task_group;
mod tasks;
pub(crate) use distributed::encode_spmd_interpreter_output;

pub(super) struct ParallelDispatchContext<'a> {
    pub bytecode: &'a crate::bytecode::Bytecode,
    pub execution: &'a ExecutionContext,
    pub function_registry: &'a FunctionRegistry,
    pub current_function_name: &'a str,
}

type ParallelDispatchResult = Result<Option<DispatchHandled>, RuntimeError>;
type ParallelDispatchFuture<'a> = control::ControlDispatchFuture<'a>;

struct ParallelDispatchState<'a> {
    instr: &'a Instr,
    stack: &'a mut crate::interpreter::stack::InterpreterStack,
    vars: &'a mut InterpreterVars,
    pc: &'a mut usize,
    sequence_state: &'a mut super::SequenceState,
    context: ParallelDispatchContext<'a>,
}

pub(super) fn dispatch<'a>(
    instr: &'a Instr,
    stack: &'a mut crate::interpreter::stack::InterpreterStack,
    vars: &'a mut InterpreterVars,
    pc: &'a mut usize,
    sequence_state: &'a mut super::SequenceState,
    context: ParallelDispatchContext<'a>,
) -> ParallelDispatchFuture<'a> {
    let state = ParallelDispatchState {
        instr,
        stack,
        vars,
        pc,
        sequence_state,
        context,
    };
    // Each leaf constructs only its selected concrete future. Reuse the
    // instruction's existing boxed owner instead of stacking a second wrapper
    // whose debug poll frame reserves every parallel operation's temporaries.
    match instr {
        Instr::CreateSemanticFuture(..)
        | Instr::CreateSemanticFutureExpandMultiOutput(..)
        | Instr::CreateExternalFuture(..)
        | Instr::ScheduleFeval { .. }
        | Instr::Spawn
        | Instr::SpawnOn => future_creation::suspend(state),
        Instr::Await | Instr::FetchOutputs { .. } | Instr::FetchNext { .. } => {
            future_wait::suspend(state)
        }
        Instr::EnsurePool(..) | Instr::CurrentPool(..) => pool_dispatch::suspend(state),
        Instr::ExecuteParfor { .. } | Instr::ExecuteSpmd { .. } => region_dispatch::suspend(state),
        Instr::Collective { .. } | Instr::Distributed(..) => data_dispatch::suspend(state),
        _ => control::admit(&state.context.execution.runtime, || async { Ok(None) }),
    }
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

fn pop(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    message: &str,
) -> Result<Value, RuntimeError> {
    stack
        .pop()
        .ok_or_else(|| crate::interpreter::errors::mex("StackUnderflow", message))
}

fn execution_error(error: runmat_runtime::execution::ExecutionServiceError) -> RuntimeError {
    error.into_runtime_error()
}
