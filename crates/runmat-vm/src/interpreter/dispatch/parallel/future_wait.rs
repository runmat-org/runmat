//! Future waiting and result publication.
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
        Instr::Await => {
            let value = pop(stack, "await instruction expected a value on the stack")?;
            tasks::await_value(execution, value)
                .await?
                .with_owned(|value| stack.try_push(value))?;
        }
        Instr::FetchOutputs {
            arg_count,
            requested_outputs,
        } => {
            let mut arguments = crate::call::builtins::collect_call_args(stack, *arg_count)?;
            let futures = arguments.remove(0);
            let uniform_output = tasks::fetch_outputs_options(&arguments)?;
            super::calls::publish_rooted_fixed_outputs(
                stack,
                tasks::fetch_outputs(execution, &futures, *requested_outputs, uniform_output)
                    .await?,
                *requested_outputs,
            )?;
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
            super::calls::publish_rooted_fixed_outputs(
                stack,
                tasks::fetch_next(execution, &futures, timeout, *requested_outputs).await?,
                *requested_outputs,
            )?;
        }
        _ => return Ok(None),
    }
    Ok(Some(DispatchHandled::Generic(
        DispatchDecision::FallThrough,
    )))
}
