use super::*;

pub(super) async fn dispatch(
    ctx: CallDispatchContext<'_>,
) -> Result<Option<DispatchHandled>, RuntimeError> {
    let CallDispatchContext {
        instr,
        stack,
        vars,
        sequence_register,
        context,
        try_stack,
        last_exception,
        imports,
        pc,
        source_id,
        call_arg_spans,
        function_registry,
        current_function_name,
        ..
    } = ctx;
    match instr {
        Instr::CallSemanticFunctionMulti(function, arg_count, out_count) => {
            match handle_user_function_call(
                calls::UserCallContext {
                    stack,
                    runtime: &context.runtime,
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
                UserCallHandling::Completed(sequence) => {
                    calls::publish_fixed_outputs(stack, sequence, out_count)?;
                }
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
                    runtime: &context.runtime,
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
                UserCallHandling::Completed(sequence) => {
                    sequence_register.stage_call_output_sequence(stack, sequence)?;
                }
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
            calls::publish_fixed_outputs(stack, result, *out_count)?;
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
            sequence_register.stage_call_output_sequence(stack, result)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallFunctionMulti {
            identity,
            fallback_policy,
            arg_count,
            out_count,
        } => {
            match handle_user_function_call(
                calls::UserCallContext {
                    stack,
                    runtime: &context.runtime,
                    identity: identity.clone(),
                    fallback_policy: *fallback_policy,
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
                UserCallHandling::Completed(sequence) => {
                    calls::publish_fixed_outputs(stack, sequence, out_count)?;
                }
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
        Instr::CallFunctionMultiUsingOutputSlot {
            identity,
            fallback_policy,
            arg_count,
            out_count_slot,
        } => {
            let out_count = requested_outputs_from_slot(vars.as_slice(), *out_count_slot)?;
            match handle_user_function_call(
                calls::UserCallContext {
                    stack,
                    runtime: &context.runtime,
                    identity: identity.clone(),
                    fallback_policy: *fallback_policy,
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
                UserCallHandling::Completed(sequence) => {
                    sequence_register.stage_call_output_sequence(stack, sequence)?;
                }
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
        _ => Ok(None),
    }
}
