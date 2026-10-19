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
        Instr::CallBuiltinExpandMultiOutput(..)
        | Instr::CallBuiltinExpandMultiOutputUsingOutputSlot(..) => {
            // Resolve a dynamic output count before consuming captured inputs
            // or performing the builtin call.
            let (name, specs, out_count, publication) = match instr {
                Instr::CallBuiltinExpandMultiOutput(name, specs, out_count) => (
                    name,
                    specs,
                    *out_count,
                    OutputPublication::Fixed(*out_count),
                ),
                Instr::CallBuiltinExpandMultiOutputUsingOutputSlot(name, specs, slot) => (
                    name,
                    specs,
                    requested_outputs_from_slot(vars, *slot)?,
                    OutputPublication::Sequence,
                ),
                _ => unreachable!(),
            };
            let args =
                build_builtin_expand_multi_args(stack, specs, sequence_register, &context.runtime)
                    .await?;
            let _output_guard = runmat_runtime::output_context::push_output_count(out_count);
            let result = crate::call::call_builtin_async_with_outputs(
                &context.runtime,
                name,
                args.values(),
                out_count,
            )
            .await?;
            match publication {
                OutputPublication::Fixed(count) => {
                    calls::publish_rooted_fixed_outputs(stack, result, count)?
                }
                OutputPublication::Sequence => {
                    sequence_register.stage_rooted_call_output_sequence(stack, result)?;
                }
            }
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
            let result = args
                .with_unbounded_values_async(|args| async move {
                    runmat_runtime::call_super_constructor(
                        &context.runtime,
                        current_class.clone(),
                        super_class.clone(),
                        args,
                    )
                    .await
                })
                .await??;
            let result = result.try_map_owned(
                runmat_runtime::call::arguments::adapt_legacy_builtin_result,
                |error| error.into_runtime_error(),
            )?;
            calls::publish_rooted_fixed_outputs(stack, result, *out_count)?;
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
            let result = args
                .with_unbounded_values_async(|args| async move {
                    runmat_runtime::call_super_method_in_context(
                        &context.runtime,
                        current_class.clone(),
                        super_class.clone(),
                        method.clone(),
                        args,
                    )
                    .await
                })
                .await??;
            let result = result.try_map_owned(
                runmat_runtime::call::arguments::adapt_legacy_builtin_result,
                |error| error.into_runtime_error(),
            )?;
            calls::publish_rooted_fixed_outputs(stack, result, *out_count)?;
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
            let selected_bound = match identity {
                runmat_hir::CallableIdentity::BoundFunction(function)
                | runmat_hir::CallableIdentity::AnonymousFunction(function)
                | runmat_hir::CallableIdentity::ExternalFunction { function, .. } => {
                    calls::selected_static_call_binding(&context.runtime, instr, *pc, *function)?
                }
                _ => None,
            };
            let args = build_user_function_expand_multi_args(
                stack,
                specs,
                sequence_register,
                &context.runtime,
            )
            .await?;
            match args
                .with_unbounded_values_async(|args| {
                    handle_prepared_user_function_call(
                        calls::UserCallContext {
                            stack,
                            runtime: &context.runtime,
                            process_driver: context.process_driver.clone(),
                            identity: identity.clone(),
                            bound: selected_bound,
                            fallback_policy: *fallback_policy,
                            out_count: *out_count,
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
                        args,
                    )
                })
                .await??
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
            match args
                .with_unbounded_values_async(|args| {
                    handle_workspace_first_prepared_call(
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
                })
                .await??
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
        Instr::CallWorkspaceFirstExpandMultiOutputUsingOutputSlot {
            name,
            identity,
            fallback_policy,
            bare_identifier,
            specs,
            out_count_slot,
        } => {
            let out_count = requested_outputs_from_slot(vars, *out_count_slot)?;
            let args = build_user_function_expand_multi_args(
                stack,
                specs,
                sequence_register,
                &context.runtime,
            )
            .await?;
            match args
                .with_unbounded_values_async(|args| {
                    handle_workspace_first_prepared_call(
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
                })
                .await??
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
        Instr::CallSemanticFunctionExpandMultiOutput(function, specs, out_count) => {
            let selected_bound =
                calls::selected_static_call_binding(&context.runtime, instr, *pc, *function)?
                    .or_else(|| function_registry.static_binding(*function));
            let args = build_user_function_expand_multi_args(
                stack,
                specs,
                sequence_register,
                &context.runtime,
            )
            .await?;
            match args
                .with_unbounded_values_async(|args| {
                    handle_prepared_user_function_call(
                        calls::UserCallContext {
                            stack,
                            runtime: &context.runtime,
                            process_driver: context.process_driver.clone(),
                            identity: runmat_hir::CallableIdentity::BoundFunction(*function),
                            bound: selected_bound,
                            fallback_policy: runmat_hir::CallableFallbackPolicy::None,
                            out_count: *out_count,
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
                        args,
                    )
                })
                .await??
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
            let call_args = args.with_unbounded_values(|args| {
                let mut call_args = Vec::with_capacity(capture_slots.len() + args.len());
                for slot in capture_slots {
                    crate::interpreter::dispatch::lexical::refresh(context, vars, *slot)?;
                    call_args.push(
                        vars.get(*slot)
                            .map(|value| value.clone())
                            .unwrap_or(Value::Num(0.0)),
                    );
                }
                call_args.extend(args);
                runmat_runtime::execution::RootedValues::new(call_args)
                    .map_err(|error| error.into_runtime_error())
            })??;
            let _output_guard = runmat_runtime::output_context::push_output_count(*out_count);
            let (result, updated_captures) =
                crate::interpreter::runner::invoke_semantic_function_value_with_capture_updates(
                    function.0,
                    call_args.values(),
                    *out_count,
                    function_registry,
                    context.runtime.clone(),
                    context.process_driver.clone(),
                    context.lexical_environment.clone(),
                )
                .await?;
            updated_captures.with_unbounded_values(|captures| {
                if context.lexical_environment.is_none() {
                    for (slot, value) in capture_slots.iter().zip(captures.into_iter()) {
                        if *slot < vars.len() {
                            crate::runtime::workspace::commit_workspace_slot_value(
                                vars, *slot, None, value,
                            )?;
                        }
                    }
                }
                Ok::<_, RuntimeError>(())
            })??;
            calls::publish_rooted_fixed_outputs(stack, result, *out_count)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        _ => Ok(None),
    }
}
