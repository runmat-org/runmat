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
                    bound: function_registry.static_binding(*function),
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
                    bound: function_registry.static_binding(*function),
                    fallback_policy: runmat_hir::CallableFallbackPolicy::None,
                    out_count,
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
                    bound: None,
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
                    bound: None,
                    fallback_policy: *fallback_policy,
                    out_count,
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
        _ => unreachable!("ordinary semantic-call route received another instruction"),
    }
}
