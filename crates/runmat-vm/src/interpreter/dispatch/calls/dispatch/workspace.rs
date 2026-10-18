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
        source_ref,
        call_arg_spans,
        function_registry,
        current_function_name,
        ..
    } = ctx;
    match instr {
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
                    runtime: &context.runtime,
                    process_driver: context.process_driver.clone(),
                    workspace_name: name,
                    identity: identity.clone(),
                    fallback_policy: *fallback_policy,
                    bare_identifier: *bare_identifier,
                    out_count: *out_count,
                    source_id,
                    source_ref,
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
            )
            .await?
            {
                UserCallHandling::Completed(sequence) => {
                    calls::publish_rooted_fixed_outputs(stack, sequence, out_count)?;
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
        Instr::CallWorkspaceFirstMultiUsingOutputSlot {
            name,
            identity,
            fallback_policy,
            bare_identifier,
            arg_count,
            out_count_slot,
        } => {
            let out_count = requested_outputs_from_slot(vars, *out_count_slot)?;
            let args = crate::call::builtins::collect_call_args(stack, *arg_count)?;
            match handle_workspace_first_prepared_call(
                calls::WorkspaceFirstCallContext {
                    stack,
                    runtime: &context.runtime,
                    process_driver: context.process_driver.clone(),
                    workspace_name: name,
                    identity: identity.clone(),
                    fallback_policy: *fallback_policy,
                    bare_identifier: *bare_identifier,
                    out_count,
                    source_id,
                    source_ref,
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
            )
            .await?
            {
                UserCallHandling::Completed(sequence) => {
                    sequence_register.stage_rooted_call_output_sequence(stack, sequence)?;
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
