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
    // Decode the instruction and its output count before collecting arguments or
    // entering a callable. One prepared call then owns the only suspending path.
    let (identity, bound, fallback_policy, arg_count, out_count, publication) = match instr {
        Instr::CallSemanticFunctionMulti(function, arg_count, out_count) => (
            runmat_hir::CallableIdentity::BoundFunction(*function),
            calls::selected_static_call_binding(&context.runtime, instr, *pc, *function)?
                .or_else(|| function_registry.static_binding(*function)),
            runmat_hir::CallableFallbackPolicy::None,
            *arg_count,
            *out_count,
            OutputPublication::Fixed(*out_count),
        ),
        Instr::CallSemanticFunctionMultiUsingOutputSlot(function, arg_count, out_count_slot) => {
            let out_count = requested_outputs_from_slot(vars, *out_count_slot)?;
            (
                runmat_hir::CallableIdentity::BoundFunction(*function),
                calls::selected_static_call_binding(&context.runtime, instr, *pc, *function)?
                    .or_else(|| function_registry.static_binding(*function)),
                runmat_hir::CallableFallbackPolicy::None,
                *arg_count,
                out_count,
                OutputPublication::Sequence,
            )
        }
        Instr::CallFunctionMulti {
            identity,
            fallback_policy,
            arg_count,
            out_count,
        } => {
            let bound = match identity {
                runmat_hir::CallableIdentity::BoundFunction(function)
                | runmat_hir::CallableIdentity::AnonymousFunction(function)
                | runmat_hir::CallableIdentity::ExternalFunction { function, .. } => {
                    calls::selected_static_call_binding(&context.runtime, instr, *pc, *function)?
                }
                _ => None,
            };
            (
                identity.clone(),
                bound,
                *fallback_policy,
                *arg_count,
                *out_count,
                OutputPublication::Fixed(*out_count),
            )
        }
        Instr::CallFunctionMultiUsingOutputSlot {
            identity,
            fallback_policy,
            arg_count,
            out_count_slot,
        } => {
            let out_count = requested_outputs_from_slot(vars, *out_count_slot)?;
            (
                identity.clone(),
                match identity {
                    runmat_hir::CallableIdentity::BoundFunction(function)
                    | runmat_hir::CallableIdentity::AnonymousFunction(function)
                    | runmat_hir::CallableIdentity::ExternalFunction { function, .. } => {
                        calls::selected_static_call_binding(
                            &context.runtime,
                            instr,
                            *pc,
                            *function,
                        )?
                    }
                    _ => None,
                },
                *fallback_policy,
                *arg_count,
                out_count,
                OutputPublication::Sequence,
            )
        }
        _ => unreachable!("ordinary semantic-call route received another instruction"),
    };
    let result = handle_user_function_call(
        calls::UserCallContext {
            stack,
            runtime: &context.runtime,
            process_driver: context.process_driver.clone(),
            identity,
            bound,
            fallback_policy,
            out_count,
            source_id,
            source_ref,
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
        arg_count,
    )
    .await?;
    match result {
        UserCallHandling::Completed(sequence) => match publication {
            OutputPublication::Fixed(count) => {
                calls::publish_rooted_fixed_outputs(stack, sequence, count)?
            }
            OutputPublication::Sequence => {
                sequence_register.stage_rooted_call_output_sequence(stack, sequence)?;
            }
        },
        UserCallHandling::Caught => {
            return Ok(Some(DispatchHandled::Generic(
                DispatchDecision::ContinueLoop,
            )));
        }
        UserCallHandling::Uncaught(error) => return Err(*error),
    }
    Ok(Some(DispatchHandled::Generic(
        DispatchDecision::FallThrough,
    )))
}
