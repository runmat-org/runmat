//! Future creation and task scheduling.
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
        pc,
        sequence_state,
        context,
        ..
    } = state;
    let ParallelDispatchContext {
        execution,
        function_registry,
        ..
    } = context;
    match instr {
        Instr::CreateSemanticFuture(function, arg_count, out_count) => {
            let arguments = crate::call::builtins::collect_call_args(stack, *arg_count)?;
            stack.try_push(Value::Future(
                tasks::create_future(
                    execution,
                    tasks::semantic_descriptor(
                        *function,
                        *out_count,
                        arguments,
                        super::calls::selected_static_call_binding(
                            &execution.runtime,
                            instr,
                            *pc,
                            *function,
                        )?
                        .or_else(|| function_registry.static_binding(*function)),
                    ),
                    function_registry,
                )?
                .into(),
            ))?;
        }
        // External futures keep the retained target's original function ID.
        Instr::CreateSemanticFutureExpandMultiOutput(function, specs, out_count)
        | Instr::CreateExternalFuture(function, specs, out_count) => {
            let arguments = build_user_function_expand_multi_args(
                stack,
                specs,
                sequence_state,
                &execution.runtime,
            )
            .await?;
            arguments.with_unbounded_values(|arguments| {
                stack.try_push(Value::Future(
                    tasks::create_future(
                        execution,
                        tasks::semantic_descriptor(
                            *function,
                            *out_count,
                            arguments,
                            super::calls::selected_static_call_binding(
                                &execution.runtime,
                                instr,
                                *pc,
                                *function,
                            )?
                            .or_else(|| function_registry.static_binding(*function)),
                        ),
                        function_registry,
                    )?
                    .into(),
                ))?;
                Ok::<_, RuntimeError>(())
            })??;
        }
        Instr::ScheduleFeval { arg_count, on_all } => {
            tasks::schedule_feval(stack, execution, function_registry, *arg_count, *on_all)?;
        }
        Instr::Spawn => tasks::spawn(stack, execution, false)?,
        Instr::SpawnOn => tasks::spawn(stack, execution, true)?,
        _ => return Ok(None),
    }
    Ok(Some(DispatchHandled::Generic(
        DispatchDecision::FallThrough,
    )))
}
