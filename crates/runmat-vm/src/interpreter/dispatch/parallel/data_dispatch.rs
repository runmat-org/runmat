//! Collective and distributed operations.
use super::*;

#[inline(never)]
pub(super) fn suspend<'a>(state: ParallelDispatchState<'a>) -> ParallelDispatchFuture<'a> {
    let execution = state.context.execution;
    control::admit(&execution.runtime, move || dispatch(state))
}

async fn dispatch(state: ParallelDispatchState<'_>) -> ParallelDispatchResult {
    let ParallelDispatchState {
        instr,
        stack,
        context,
        ..
    } = state;
    let ParallelDispatchContext {
        bytecode,
        execution,
        ..
    } = context;
    match instr {
        Instr::Collective { id, operation } => {
            let arguments =
                crate::call::builtins::collect_call_args(stack, operation.operand_count())?;
            let requested_outputs = match operation {
                crate::BytecodeCollectiveOp::Receive {
                    requested_outputs, ..
                } => usize::from(*requested_outputs),
                _ => 1,
            };
            super::calls::publish_rooted_fixed_outputs(
                stack,
                collective::execute(bytecode, *id, *operation, arguments, execution).await?,
                requested_outputs,
            )?;
        }
        Instr::Distributed(operation) => {
            let arguments =
                crate::call::builtins::collect_call_args(stack, operation.operand_count())?;
            let requested_outputs = match operation {
                crate::BytecodeDistributedOp::GlobalIndices {
                    requested_outputs, ..
                } => usize::from(*requested_outputs),
                _ => 1,
            };
            super::calls::publish_rooted_fixed_outputs(
                stack,
                runmat_runtime::execution::admit_executable_sequence_with(
                    &execution.runtime,
                    || distributed::execute(bytecode, operation, arguments, execution),
                )?
                .await?,
                requested_outputs,
            )?;
        }
        _ => return Ok(None),
    }
    Ok(Some(DispatchHandled::Generic(
        DispatchDecision::FallThrough,
    )))
}
