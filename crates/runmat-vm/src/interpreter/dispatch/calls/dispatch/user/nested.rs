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
        source_id,
        call_arg_spans,
        function_registry,
        ..
    } = ctx;
    match instr {
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
                    call_arg_spans.as_ref().cloned(),
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
            for (slot, value) in capture_slots.iter().zip(updated_captures.into_iter()) {
                if *slot < vars.len() {
                    crate::runtime::workspace::admit_workspace_slot_write(*slot, &value)?;
                    vars[*slot] = value;
                }
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
                    call_arg_spans.as_ref().cloned(),
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
            for (slot, value) in capture_slots.iter().zip(updated_captures.into_iter()) {
                if *slot < vars.len() {
                    crate::runtime::workspace::admit_workspace_slot_write(*slot, &value)?;
                    vars[*slot] = value;
                }
            }
            sequence_register.stage_call_output_sequence(stack, result)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        _ => unreachable!("nested semantic-call route received another instruction"),
    }
}
