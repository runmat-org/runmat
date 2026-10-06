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
        current_function_name,
        ..
    } = meta;
    let DispatchState {
        stack,
        vars,
        context,
        try_stack,
        last_exception,
        imports,
        pc,
        ..
    } = state;
    let _ = hooks;
    match instr {
        Instr::CallSemanticFunctionMulti(function, arg_count, out_count) => {
            match handle_user_function_call(
                calls::UserCallContext {
                    stack,
                    identity: runmat_hir::CallableIdentity::BoundFunction(*function),
                    fallback_policy: runmat_hir::CallableFallbackPolicy::None,
                    out_count: *out_count,
                    source_id,
                    call_arg_spans: call_arg_spans.clone(),
                    current_function_name,
                    imports: imports.as_slice(),
                    exception: calls::ExceptionRouteContext {
                        try_stack,
                        vars,
                        last_exception,
                        pc,
                    },
                },
                *arg_count,
                refresh_workspace_state,
            )
            .await?
            {
                UserCallHandling::Completed => {}
                UserCallHandling::Caught => {
                    return Ok(Some(DispatchHandled::Generic(
                        DispatchDecision::ContinueLoop,
                    )))
                }
                UserCallHandling::Uncaught(err) => return Err(*err),
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallSemanticFunctionMultiUsingOutputSlot(function, arg_count, out_count_slot) => {
            let out_count = requested_outputs_from_slot(vars.as_slice(), *out_count_slot)?;
            match handle_user_function_call(
                calls::UserCallContext {
                    stack,
                    identity: runmat_hir::CallableIdentity::BoundFunction(*function),
                    fallback_policy: runmat_hir::CallableFallbackPolicy::None,
                    out_count,
                    source_id,
                    call_arg_spans: call_arg_spans.clone(),
                    current_function_name,
                    imports: imports.as_slice(),
                    exception: calls::ExceptionRouteContext {
                        try_stack,
                        vars,
                        last_exception,
                        pc,
                    },
                },
                *arg_count,
                refresh_workspace_state,
            )
            .await?
            {
                UserCallHandling::Completed => {}
                UserCallHandling::Caught => {
                    return Ok(Some(DispatchHandled::Generic(
                        DispatchDecision::ContinueLoop,
                    )))
                }
                UserCallHandling::Uncaught(err) => return Err(*err),
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallSemanticNestedFunctionMulti {
            function,
            capture_slots,
            arg_count,
            out_count,
        } => {
            let args = crate::call::builtins::collect_call_args(stack, *arg_count)?;
            let mut call_args = Vec::with_capacity(capture_slots.len() + args.len());
            for slot in capture_slots {
                call_args.push(vars.get(*slot).cloned().unwrap_or(Value::Num(0.0)));
            }
            call_args.extend(args);
            let _output_guard = runmat_runtime::output_context::push_output_count(*out_count);
            let _function_input_callsite_guard =
                runmat_runtime::callsite::push_function_input_callsite(
                    source_id,
                    call_arg_spans.clone(),
                );
            let (result, updated_captures) =
                crate::interpreter::runner::invoke_semantic_function_value_with_capture_updates(
                    function.0,
                    &call_args,
                    *out_count,
                    function_registry,
                    context.runtime.clone(),
                )
                .await?;
            let mut captures_updated = false;
            for (slot, value) in capture_slots.iter().zip(updated_captures.into_iter()) {
                if *slot < vars.len() {
                    vars[*slot] = value;
                    captures_updated = true;
                }
            }
            if captures_updated {
                refresh_workspace_state(vars);
            }
            stack.push(calls::normalize_requested_outputs(result, *out_count));
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallSemanticNestedFunctionMultiUsingOutputSlot {
            function,
            capture_slots,
            arg_count,
            out_count_slot,
        } => {
            let out_count = requested_outputs_from_slot(vars.as_slice(), *out_count_slot)?;
            let args = crate::call::builtins::collect_call_args(stack, *arg_count)?;
            let mut call_args = Vec::with_capacity(capture_slots.len() + args.len());
            for slot in capture_slots {
                call_args.push(vars.get(*slot).cloned().unwrap_or(Value::Num(0.0)));
            }
            call_args.extend(args);
            let _output_guard = runmat_runtime::output_context::push_output_count(out_count);
            let _function_input_callsite_guard =
                runmat_runtime::callsite::push_function_input_callsite(
                    source_id,
                    call_arg_spans.clone(),
                );
            let (result, updated_captures) =
                crate::interpreter::runner::invoke_semantic_function_value_with_capture_updates(
                    function.0,
                    &call_args,
                    out_count,
                    function_registry,
                    context.runtime.clone(),
                )
                .await?;
            let mut captures_updated = false;
            for (slot, value) in capture_slots.iter().zip(updated_captures.into_iter()) {
                if *slot < vars.len() {
                    vars[*slot] = value;
                    captures_updated = true;
                }
            }
            if captures_updated {
                refresh_workspace_state(vars);
            }
            stack.push(calls::normalize_requested_outputs(result, out_count));
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallSemanticFunctionExpandMultiOutput(function, specs, out_count) => {
            let args = build_user_function_expand_multi_args(stack, specs).await?;
            match handle_prepared_user_function_call(
                calls::UserCallContext {
                    stack,
                    identity: runmat_hir::CallableIdentity::BoundFunction(*function),
                    fallback_policy: runmat_hir::CallableFallbackPolicy::None,
                    out_count: *out_count,
                    source_id,
                    call_arg_spans: call_arg_spans.clone(),
                    current_function_name,
                    imports: imports.as_slice(),
                    exception: calls::ExceptionRouteContext {
                        try_stack,
                        vars,
                        last_exception,
                        pc,
                    },
                },
                args,
                refresh_workspace_state,
            )
            .await?
            {
                UserCallHandling::Completed => {}
                UserCallHandling::Caught => {
                    return Ok(Some(DispatchHandled::Generic(
                        DispatchDecision::ContinueLoop,
                    )))
                }
                UserCallHandling::Uncaught(err) => return Err(*err),
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallSemanticNestedFunctionExpandMultiOutput {
            function,
            capture_slots,
            specs,
            out_count,
        } => {
            let args = build_user_function_expand_multi_args(stack, specs).await?;
            let mut call_args = Vec::with_capacity(capture_slots.len() + args.len());
            for slot in capture_slots {
                call_args.push(vars.get(*slot).cloned().unwrap_or(Value::Num(0.0)));
            }
            call_args.extend(args);
            let _output_guard = runmat_runtime::output_context::push_output_count(*out_count);
            let (result, updated_captures) =
                crate::interpreter::runner::invoke_semantic_function_value_with_capture_updates(
                    function.0,
                    &call_args,
                    *out_count,
                    function_registry,
                    context.runtime.clone(),
                )
                .await?;
            let mut captures_updated = false;
            for (slot, value) in capture_slots.iter().zip(updated_captures.into_iter()) {
                if *slot < vars.len() {
                    vars[*slot] = value;
                    captures_updated = true;
                }
            }
            if captures_updated {
                refresh_workspace_state(vars);
            }
            stack.push(calls::normalize_requested_outputs(result, *out_count));
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        _ => Ok(None),
    }
}
