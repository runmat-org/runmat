use super::*;

pub(super) async fn dispatch(
    ctx: CallDispatchContext<'_>,
) -> Result<Option<DispatchHandled>, RuntimeError> {
    let CallDispatchContext {
        instr,
        stack,
        sequence_register,
        context,
        current_function_name,
        ..
    } = ctx;
    match instr {
        Instr::CallMethodOrMemberIndexMulti {
            identity,
            fallback_policy,
            arg_count,
            out_count,
        } => {
            let handling = handle_method_or_member_index_multi_call(
                &context.runtime,
                stack,
                identity.clone(),
                *fallback_policy,
                *arg_count,
                *out_count,
                current_function_name,
            )
            .await?;
            if let Some(sequence) = handling {
                calls::publish_rooted_fixed_outputs(stack, sequence, *out_count)?;
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallMethodOrMemberIndexExpandMultiOutput {
            identity,
            fallback_policy,
            specs,
            out_count,
        } => {
            let handling = handle_method_or_member_index_expand_multi_call(
                stack,
                sequence_register,
                calls::MethodExpandCallContext {
                    identity: identity.clone(),
                    fallback_policy: *fallback_policy,
                    specs,
                    requested_outputs: *out_count,
                    current_function_name,
                    runtime: &context.runtime,
                },
            )
            .await?;
            if let Some(sequence) = handling {
                calls::publish_rooted_fixed_outputs(stack, sequence, *out_count)?;
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        _ => Ok(None),
    }
}
