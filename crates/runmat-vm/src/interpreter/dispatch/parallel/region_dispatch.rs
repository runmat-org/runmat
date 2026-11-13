//! Parallel region execution.
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
        vars,
        pc,
        context,
        ..
    } = state;
    let ParallelDispatchContext {
        bytecode,
        execution,
        current_function_name,
        ..
    } = context;
    match instr {
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
                .ok_or_else(|| {
                    crate::interpreter::errors::mex(
                        "ParallelExecutableMissing",
                        "parfor has no compiler-bound executable region",
                    )
                })?;
            runmat_runtime::execution::admit_domain_continuation_with(&execution.runtime, || {
                parfor::execute(
                    bytecode,
                    executable,
                    iterable,
                    maximum_workers,
                    vars,
                    execution,
                    current_function_name,
                )
            })?
            .await?;
            *pc = executable.exit.pc;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::ContinueLoop,
            )))
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
            runmat_runtime::execution::admit_domain_continuation_with(&execution.runtime, || {
                spmd::execute(
                    bytecode,
                    &executable,
                    *header,
                    operands,
                    vars,
                    execution,
                    current_function_name,
                )
            })?
            .await?;
            *pc = executable.exit.pc;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::ContinueLoop,
            )))
        }
        _ => Ok(None),
    }
}
