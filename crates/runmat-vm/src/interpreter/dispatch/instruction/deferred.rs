use super::super::*;

pub(super) async fn execute(
    meta: DispatchMeta<'_>,
    state: DispatchState<'_>,
    hooks: DispatchHooks<'_>,
) -> Result<Option<DispatchHandled>, RuntimeError> {
    let instr = meta.instr;
    let function_registry = meta.function_registry;
    let DispatchState { stack, context, .. } = state;
    let _ = hooks;
    match instr {
        Instr::CreateSemanticFuture(function, arg_count, out_count) => {
            let args = crate::call::builtins::collect_call_args(stack, *arg_count)?;
            stack.push(create_async_future_value(
                context, *function, *out_count, args,
            )?);
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateSemanticFutureExpandMultiOutput(function, specs, out_count) => {
            let args = build_user_function_expand_multi_args(stack, specs).await?;
            stack.push(create_async_future_value(
                context, *function, *out_count, args,
            )?);
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::Spawn => {
            let value = stack.pop().ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "StackUnderflow",
                    "spawn instruction expected a value on the stack",
                )
            })?;
            let Value::Future(future) = value else {
                return Err(crate::interpreter::errors::mex(
                    "SpawnOperandInvalid",
                    "spawn expects a lazy future produced by an async call",
                ));
            };
            let task = context
                .runtime
                .execution()
                .spawn(&future)
                .map_err(execution_error)?;
            stack.push(Value::Task(task));
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::Await => {
            let value = stack.pop().ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "StackUnderflow",
                    "await instruction expected a value on the stack",
                )
            })?;
            let value = await_execution_value(context, value, function_registry).await?;
            stack.push(value);
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        _ => Ok(None),
    }
}
