use super::super::*;

pub(super) async fn execute(
    meta: DispatchMeta<'_>,
    state: DispatchState<'_>,
    hooks: DispatchHooks<'_>,
) -> Result<Option<DispatchHandled>, RuntimeError> {
    let instr = meta.instr;
    let DispatchMeta {
        function_registry,
        source_id,
        call_arg_spans,
        call_counts,
        current_function_name,
        ..
    } = meta;
    let DispatchState {
        stack,
        vars,
        try_stack,
        last_exception,
        imports,
        pc,
        ..
    } = state;
    let _ = hooks;
    match instr {
        Instr::CallBuiltinMulti(name, arg_count, out_count) => {
            match handle_builtin_call_multi(
                calls::BuiltinCallContext {
                    stack,
                    name,
                    arg_count: *arg_count,
                    source_id,
                    call_arg_spans: call_arg_spans.clone(),
                    imports: imports.as_slice(),
                    call_counts,
                    function_registry,
                    current_function_name,
                    exception: calls::ExceptionRouteContext {
                        try_stack,
                        vars,
                        last_exception,
                        pc,
                    },
                },
                *out_count,
                refresh_workspace_state,
            )
            .await?
            {
                BuiltinHandling::Completed => {}
                BuiltinHandling::Caught => {
                    return Ok(Some(DispatchHandled::Generic(
                        DispatchDecision::ContinueLoop,
                    )))
                }
                BuiltinHandling::Uncaught(err) => return Err(*err),
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallBuiltinMultiUsingOutputSlot(name, arg_count, out_count_slot) => {
            let out_count = requested_outputs_from_slot(vars.as_slice(), *out_count_slot)?;
            match handle_builtin_call_multi(
                calls::BuiltinCallContext {
                    stack,
                    name,
                    arg_count: *arg_count,
                    source_id,
                    call_arg_spans: call_arg_spans.clone(),
                    imports: imports.as_slice(),
                    call_counts,
                    function_registry,
                    current_function_name,
                    exception: calls::ExceptionRouteContext {
                        try_stack,
                        vars,
                        last_exception,
                        pc,
                    },
                },
                out_count,
                refresh_workspace_state,
            )
            .await?
            {
                BuiltinHandling::Completed => {}
                BuiltinHandling::Caught => {
                    return Ok(Some(DispatchHandled::Generic(
                        DispatchDecision::ContinueLoop,
                    )))
                }
                BuiltinHandling::Uncaught(err) => return Err(*err),
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallSuperConstructorMulti {
            current_class,
            super_class,
            arg_count,
            out_count,
        } => {
            let args = crate::call::builtins::collect_call_args(stack, *arg_count)?;
            let _output_guard = runmat_runtime::output_context::push_output_count(*out_count);
            let result = runmat_runtime::call_super_constructor(
                current_class.clone(),
                super_class.clone(),
                args,
            )
            .await?;
            stack.push(calls::normalize_requested_outputs(result, *out_count));
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallSuperMethodMulti {
            current_class,
            super_class,
            method,
            arg_count,
            out_count,
        } => {
            let args = crate::call::builtins::collect_call_args(stack, *arg_count)?;
            let _output_guard = runmat_runtime::output_context::push_output_count(*out_count);
            let result = runmat_runtime::call_super_method(
                current_class.clone(),
                super_class.clone(),
                method.clone(),
                args,
            )
            .await?;
            stack.push(calls::normalize_requested_outputs(result, *out_count));
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallFevalMulti(argc, out_count) => {
            let args = crate::call::builtins::collect_call_args(stack, *argc)?;
            let func_val = crate::interpreter::stack::pop_value(stack)?;
            let _function_input_callsite_guard =
                runmat_runtime::callsite::push_function_input_callsite(
                    source_id,
                    call_arg_spans.clone(),
                );
            match crate::call::feval::execute_feval(func_val, args, *out_count, function_registry)
                .await?
            {
                crate::call::feval::FevalDispatch::Completed(result) => {
                    stack.push(calls::normalize_requested_outputs(result, *out_count));
                }
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallFevalMultiUsingOutputSlot(argc, out_count_slot) => {
            let out_count = requested_outputs_from_slot(vars.as_slice(), *out_count_slot)?;
            let args = crate::call::builtins::collect_call_args(stack, *argc)?;
            let func_val = crate::interpreter::stack::pop_value(stack)?;
            let _function_input_callsite_guard =
                runmat_runtime::callsite::push_function_input_callsite(
                    source_id,
                    call_arg_spans.clone(),
                );
            match crate::call::feval::execute_feval(func_val, args, out_count, function_registry)
                .await?
            {
                crate::call::feval::FevalDispatch::Completed(result) => {
                    stack.push(calls::normalize_requested_outputs(result, out_count));
                }
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallFevalExpandMultiOutput(specs, out_count) => {
            let args = build_feval_expand_multi_args(stack, specs).await?;
            let func_val = crate::interpreter::stack::pop_value(stack)?;
            let _function_input_callsite_guard =
                runmat_runtime::callsite::push_function_input_callsite(
                    source_id,
                    call_arg_spans.clone(),
                );
            match crate::call::feval::execute_feval(func_val, args, *out_count, function_registry)
                .await?
            {
                crate::call::feval::FevalDispatch::Completed(result) => {
                    stack.push(calls::normalize_requested_outputs(result, *out_count));
                }
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallFevalExpandMultiOutputUsingOutputSlot(specs, out_count_slot) => {
            let out_count = requested_outputs_from_slot(vars.as_slice(), *out_count_slot)?;
            let args = build_feval_expand_multi_args(stack, specs).await?;
            let func_val = crate::interpreter::stack::pop_value(stack)?;
            let _function_input_callsite_guard =
                runmat_runtime::callsite::push_function_input_callsite(
                    source_id,
                    call_arg_spans.clone(),
                );
            match crate::call::feval::execute_feval(func_val, args, out_count, function_registry)
                .await?
            {
                crate::call::feval::FevalDispatch::Completed(result) => {
                    stack.push(calls::normalize_requested_outputs(result, out_count));
                }
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallBuiltinExpandMultiOutput(name, specs, out_count) => {
            let args = build_builtin_expand_multi_args(stack, specs).await?;
            let _output_guard = runmat_runtime::output_context::push_output_count(*out_count);
            let result =
                runmat_runtime::call_builtin_async_with_outputs(name, &args, *out_count).await?;
            stack.push(calls::normalize_requested_outputs(result, *out_count));
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallSuperConstructorExpandMultiOutput {
            current_class,
            super_class,
            specs,
            out_count,
        } => {
            let args = build_user_function_expand_multi_args(stack, specs).await?;
            let _output_guard = runmat_runtime::output_context::push_output_count(*out_count);
            let result = runmat_runtime::call_super_constructor(
                current_class.clone(),
                super_class.clone(),
                args,
            )
            .await?;
            stack.push(calls::normalize_requested_outputs(result, *out_count));
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallSuperMethodExpandMultiOutput {
            current_class,
            super_class,
            method,
            specs,
            out_count,
        } => {
            let args = build_user_function_expand_multi_args(stack, specs).await?;
            let _output_guard = runmat_runtime::output_context::push_output_count(*out_count);
            let result = runmat_runtime::call_super_method(
                current_class.clone(),
                super_class.clone(),
                method.clone(),
                args,
            )
            .await?;
            stack.push(calls::normalize_requested_outputs(result, *out_count));
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        _ => Ok(None),
    }
}
