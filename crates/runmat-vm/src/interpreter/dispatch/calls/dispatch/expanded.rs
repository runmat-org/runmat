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
        Instr::CallBuiltinExpandMultiOutput(name, specs, out_count) => {
            let args =
                build_builtin_expand_multi_args(stack, specs, sequence_register, &context.runtime)
                    .await?;
            let _output_guard = runmat_runtime::output_context::push_output_count(*out_count);
            let result = crate::call::call_builtin_async_with_outputs(
                &context.runtime,
                name,
                &args,
                *out_count,
            )
            .await?;
            calls::publish_fixed_outputs(
                stack,
                runmat_runtime::call::arguments::adapt_legacy_builtin_result(result)?,
                *out_count,
            )?;
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
            let args = build_user_function_expand_multi_args(
                stack,
                specs,
                sequence_register,
                &context.runtime,
            )
            .await?;
            let _output_guard = runmat_runtime::output_context::push_output_count(*out_count);
            let _roots = crate::call::root_values(&args)?;
            let result = runmat_runtime::call_super_constructor(
                current_class.clone(),
                super_class.clone(),
                args,
            )
            .await?;
            calls::publish_fixed_outputs(
                stack,
                runmat_runtime::call::arguments::adapt_legacy_builtin_result(result)?,
                *out_count,
            )?;
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
            let args = build_user_function_expand_multi_args(
                stack,
                specs,
                sequence_register,
                &context.runtime,
            )
            .await?;
            let _output_guard = runmat_runtime::output_context::push_output_count(*out_count);
            let _roots = crate::call::root_values(&args)?;
            let result = runmat_runtime::call_super_method(
                current_class.clone(),
                super_class.clone(),
                method.clone(),
                args,
            )
            .await?;
            calls::publish_fixed_outputs(
                stack,
                runmat_runtime::call::arguments::adapt_legacy_builtin_result(result)?,
                *out_count,
            )?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallFunctionExpandMultiOutput {
            identity,
            fallback_policy,
            specs,
            out_count,
        } => {
            let args = build_user_function_expand_multi_args(
                stack,
                specs,
                sequence_register,
                &context.runtime,
            )
            .await?;
            match handle_prepared_user_function_call(
                calls::UserCallContext {
                    stack,
                    runtime: &context.runtime,
                    identity: identity.clone(),
                    fallback_policy: *fallback_policy,
                    out_count: *out_count,
                    source_id,
                    call_arg_spans: call_arg_spans.as_ref().cloned(),
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
        Instr::CallWorkspaceFirstExpandMultiOutput {
            name,
            identity,
            fallback_policy,
            bare_identifier,
            specs,
            out_count,
        } => {
            let args = build_user_function_expand_multi_args(
                stack,
                specs,
                sequence_register,
                &context.runtime,
            )
            .await?;
            match handle_workspace_first_prepared_call(
                calls::WorkspaceFirstCallContext {
                    stack,
                    runtime: &context.runtime,
                    workspace_name: name,
                    identity: identity.clone(),
                    fallback_policy: *fallback_policy,
                    bare_identifier: *bare_identifier,
                    out_count: *out_count,
                    source_id,
                    call_arg_spans: call_arg_spans.as_ref().cloned(),
                    current_function_name,
                    imports: imports.as_slice(),
                    function_registry,
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
        Instr::CallWorkspaceFirstExpandMultiOutputUsingOutputSlot {
            name,
            identity,
            fallback_policy,
            bare_identifier,
            specs,
            out_count_slot,
        } => {
            let out_count = requested_outputs_from_slot(vars.as_slice(), *out_count_slot)?;
            let args = build_user_function_expand_multi_args(
                stack,
                specs,
                sequence_register,
                &context.runtime,
            )
            .await?;
            match handle_workspace_first_prepared_call(
                calls::WorkspaceFirstCallContext {
                    stack,
                    runtime: &context.runtime,
                    workspace_name: name,
                    identity: identity.clone(),
                    fallback_policy: *fallback_policy,
                    bare_identifier: *bare_identifier,
                    out_count,
                    source_id,
                    call_arg_spans: call_arg_spans.as_ref().cloned(),
                    current_function_name,
                    imports: imports.as_slice(),
                    function_registry,
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
        Instr::CallSemanticFunctionExpandMultiOutput(function, specs, out_count) => {
            let args = build_user_function_expand_multi_args(
                stack,
                specs,
                sequence_register,
                &context.runtime,
            )
            .await?;
            match handle_prepared_user_function_call(
                calls::UserCallContext {
                    stack,
                    runtime: &context.runtime,
                    identity: runmat_hir::CallableIdentity::BoundFunction(*function),
                    fallback_policy: runmat_hir::CallableFallbackPolicy::None,
                    out_count: *out_count,
                    source_id,
                    call_arg_spans: call_arg_spans.as_ref().cloned(),
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
        Instr::CallSemanticNestedFunctionExpandMultiOutput {
            function,
            capture_slots,
            specs,
            out_count,
        } => {
            let args = build_user_function_expand_multi_args(
                stack,
                specs,
                sequence_register,
                &context.runtime,
            )
            .await?;
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
            calls::publish_fixed_outputs(stack, result, *out_count)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        _ => Ok(None),
    }
}
