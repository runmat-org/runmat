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
        try_stack,
        last_exception,
        imports,
        pc,
        ..
    } = state;
    let _ = hooks;
    match instr {
        Instr::CallFunctionMulti {
            identity,
            fallback_policy,
            arg_count,
            out_count,
        } => {
            match handle_user_function_call(
                calls::UserCallContext {
                    stack,
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
        Instr::CallWorkspaceFirstMulti {
            name,
            identity,
            fallback_policy,
            bare_identifier,
            arg_count,
            out_count,
        } => {
            let args = crate::call::builtins::collect_call_args(stack, *arg_count)?;
            match handle_workspace_first_prepared_call(
                calls::WorkspaceFirstCallContext {
                    stack,
                    workspace_name: name,
                    identity: identity.clone(),
                    fallback_policy: *fallback_policy,
                    bare_identifier: *bare_identifier,
                    out_count: *out_count,
                    source_id,
                    call_arg_spans: call_arg_spans.clone(),
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
        Instr::CallWorkspaceFirstMultiUsingOutputSlot {
            name,
            identity,
            fallback_policy,
            bare_identifier,
            arg_count,
            out_count_slot,
        } => {
            let out_count = requested_outputs_from_slot(vars.as_slice(), *out_count_slot)?;
            let args = crate::call::builtins::collect_call_args(stack, *arg_count)?;
            match handle_workspace_first_prepared_call(
                calls::WorkspaceFirstCallContext {
                    stack,
                    workspace_name: name,
                    identity: identity.clone(),
                    fallback_policy: *fallback_policy,
                    bare_identifier: *bare_identifier,
                    out_count,
                    source_id,
                    call_arg_spans: call_arg_spans.clone(),
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
        Instr::CallFunctionExpandMultiOutput {
            identity,
            fallback_policy,
            specs,
            out_count,
        } => {
            let args = build_user_function_expand_multi_args(stack, specs).await?;
            match handle_prepared_user_function_call(
                calls::UserCallContext {
                    stack,
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
        Instr::CallWorkspaceFirstExpandMultiOutput {
            name,
            identity,
            fallback_policy,
            bare_identifier,
            specs,
            out_count,
        } => {
            let args = build_user_function_expand_multi_args(stack, specs).await?;
            match handle_workspace_first_prepared_call(
                calls::WorkspaceFirstCallContext {
                    stack,
                    workspace_name: name,
                    identity: identity.clone(),
                    fallback_policy: *fallback_policy,
                    bare_identifier: *bare_identifier,
                    out_count: *out_count,
                    source_id,
                    call_arg_spans: call_arg_spans.clone(),
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
        Instr::CallWorkspaceFirstExpandMultiOutputUsingOutputSlot {
            name,
            identity,
            fallback_policy,
            bare_identifier,
            specs,
            out_count_slot,
        } => {
            let out_count = requested_outputs_from_slot(vars.as_slice(), *out_count_slot)?;
            let args = build_user_function_expand_multi_args(stack, specs).await?;
            match handle_workspace_first_prepared_call(
                calls::WorkspaceFirstCallContext {
                    stack,
                    workspace_name: name,
                    identity: identity.clone(),
                    fallback_policy: *fallback_policy,
                    bare_identifier: *bare_identifier,
                    out_count,
                    source_id,
                    call_arg_spans: call_arg_spans.clone(),
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
        _ => Ok(None),
    }
}
