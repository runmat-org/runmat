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
        invocation_arity,
        function_registry,
        current_function_name,
        ..
    } = ctx;
    // A dynamic count is validated before argument collection or builtin work.
    let (name, arg_count, out_count, publication) = match instr {
        Instr::CallBuiltinMulti(name, arg_count, out_count) => (
            name,
            *arg_count,
            *out_count,
            OutputPublication::Fixed(*out_count),
        ),
        Instr::CallBuiltinMultiUsingOutputSlot(name, arg_count, out_count_slot) => {
            let out_count = requested_outputs_from_slot(vars, *out_count_slot)?;
            (name, *arg_count, out_count, OutputPublication::Sequence)
        }
        _ => unreachable!("ordinary builtin route received another instruction"),
    };
    let result = handle_builtin_call_multi(
        calls::BuiltinCallContext {
            runtime: &context.runtime,
            process_driver: context.process_driver.clone(),
            stack,
            name,
            arg_count,
            source_id,
            call_arg_spans: call_arg_spans.as_ref().cloned(),
            imports: imports.as_slice(),
            invocation_arity,
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
    )
    .await?;
    match result {
        BuiltinHandling::Completed(sequence) => match publication {
            OutputPublication::Fixed(count) => {
                calls::publish_rooted_fixed_outputs(stack, sequence, count)?
            }
            OutputPublication::Sequence => {
                sequence_register.stage_rooted_call_output_sequence(stack, sequence)?;
            }
        },
        BuiltinHandling::Caught => {
            return Ok(Some(DispatchHandled::Generic(
                DispatchDecision::ContinueLoop,
            )));
        }
        BuiltinHandling::Uncaught(error) => return Err(*error),
    }
    Ok(Some(DispatchHandled::Generic(
        DispatchDecision::FallThrough,
    )))
}
