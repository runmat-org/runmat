//! Pool creation and inspection.
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
    let ParallelDispatchContext { execution, .. } = context;
    match instr {
        Instr::EnsurePool(arg_count) => {
            let arguments = crate::call::builtins::collect_call_args(stack, *arg_count)?;
            stack.try_push(runmat_runtime::parallel::pool::ensure(
                &execution.runtime,
                runmat_runtime::call::FunctionArgumentView::from_slice(&arguments),
            )?)?;
        }
        Instr::CurrentPool(arg_count) => {
            let arguments = crate::call::builtins::collect_call_args(stack, *arg_count)?;
            stack.try_push(runmat_runtime::parallel::pool::current(
                &execution.runtime,
                runmat_runtime::call::FunctionArgumentView::from_slice(&arguments),
            )?)?;
        }
        _ => return Ok(None),
    }
    Ok(Some(DispatchHandled::Generic(
        DispatchDecision::FallThrough,
    )))
}
