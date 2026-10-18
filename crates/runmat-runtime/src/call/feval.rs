//! Callable forwarding retains typed output sequences until a language consumer
//! selects or discards them.

use crate::execution::RootedValueSequence;
use crate::{
    build_runtime_error, callable_identity_for_handle_name,
    dispatch_callable_with_policy_in_runtime, runtime_descriptor_error,
    runtime_descriptor_error_with_detail, CALL_METHOD_BUILTIN_NAME,
};
use runmat_builtins::{
    FEVAL_ERROR_FUNCTION_VALUE_UNSUPPORTED, FEVAL_ERROR_HANDLE_NAME_INVALID,
    FEVAL_ERROR_HANDLE_SHAPE_INVALID, FEVAL_ERROR_SEMANTIC_UNAVAILABLE,
};
use runmat_value::{Value, ValueSequence};

#[cfg(test)]
#[path = "feval/roots_tests.rs"]
mod roots_tests;

/// Acquire the invocation's input roots when the public future is created.
/// In particular, a bound closure's lexical activation must survive an
/// unpolled future after its caller releases the original handle.
pub(crate) fn admit_inputs(
    target: &Value,
    arguments: &[Value],
) -> Result<
    (
        crate::execution::RootedValueLease,
        crate::execution::RootedValueLease,
    ),
    crate::RuntimeError,
> {
    let target_root = crate::execution::RootedValueLease::for_traced(target)
        .map_err(crate::execution::ExecutionServiceError::into_runtime_error)?;
    let argument_roots = crate::execution::RootedValueLease::for_values(arguments.iter())
        .map_err(crate::execution::ExecutionServiceError::into_runtime_error)?;
    Ok((target_root, argument_roots))
}

pub(crate) async fn invoke(
    runtime: Option<&crate::context::RuntimeContext>,
    invocation_outputs: Option<usize>,
    f: Value,
    rest: Vec<Value>,
    input_callsite: Option<crate::callsite::CallsiteInfo>,
) -> crate::BuiltinResult<RootedValueSequence> {
    fn normalize_feval_handle_name(name: &str) -> Option<String> {
        let trimmed = name.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    }

    async fn call_by_identity(
        runtime: Option<&crate::context::RuntimeContext>,
        identity: runmat_types::CallableIdentity,
        fallback_policy: runmat_types::CallableFallbackPolicy,
        args: &[Value],
        requested_outputs: usize,
        input_callsite: Option<crate::callsite::CallsiteInfo>,
    ) -> crate::BuiltinResult<RootedValueSequence> {
        dispatch_callable_with_policy_in_runtime(
            runtime,
            identity,
            fallback_policy,
            args.to_vec(),
            requested_outputs,
            input_callsite,
        )
        .await
    }

    async fn call_by_name(
        runtime: Option<&crate::context::RuntimeContext>,
        name: &str,
        args: &[Value],
        requested_outputs: usize,
        input_callsite: Option<crate::callsite::CallsiteInfo>,
    ) -> crate::BuiltinResult<RootedValueSequence> {
        let normalized = normalize_feval_handle_name(name)
            .ok_or_else(|| runtime_descriptor_error("feval", &FEVAL_ERROR_HANDLE_NAME_INVALID))?;
        let (identity, fallback_policy) = callable_identity_for_handle_name(&normalized);
        call_by_identity(
            runtime,
            identity,
            fallback_policy,
            args,
            requested_outputs,
            input_callsite,
        )
        .await
    }

    crate::call::admission::admit_feval_target(&f)?;

    let requested_outputs = invocation_outputs
        .or_else(crate::output_count::current_output_count)
        .unwrap_or(1);

    match f {
        Value::String(s) => {
            let name = crate::call::admission::text_target_name(&s).ok_or_else(|| {
                runtime_descriptor_error("feval", &FEVAL_ERROR_HANDLE_NAME_INVALID)
            })?;
            call_by_name(runtime, name, &rest, requested_outputs, input_callsite).await
        }
        Value::CharArray(ca) => {
            if ca.rows == 1 {
                let s: String = ca.data.iter().collect();
                let name = crate::call::admission::text_target_name(&s).ok_or_else(|| {
                    runtime_descriptor_error("feval", &FEVAL_ERROR_HANDLE_NAME_INVALID)
                })?;
                call_by_name(runtime, name, &rest, requested_outputs, input_callsite).await
            } else {
                Err(runtime_descriptor_error_with_detail(
                    "feval",
                    &FEVAL_ERROR_HANDLE_SHAPE_INVALID,
                    "char array must be a row vector",
                ))
            }
        }
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                let s = &sa.data[0];
                let name = crate::call::admission::text_target_name(s).ok_or_else(|| {
                    runtime_descriptor_error("feval", &FEVAL_ERROR_HANDLE_NAME_INVALID)
                })?;
                call_by_name(runtime, name, &rest, requested_outputs, input_callsite).await
            } else {
                Err(runtime_descriptor_error_with_detail(
                    "feval",
                    &FEVAL_ERROR_HANDLE_SHAPE_INVALID,
                    "string array must be scalar",
                ))
            }
        }
        Value::FunctionHandle(name) => {
            call_by_name(runtime, &name, &rest, requested_outputs, input_callsite).await
        }
        Value::ExternalFunctionHandle(name) => {
            call_by_name(runtime, &name, &rest, requested_outputs, input_callsite).await
        }
        Value::MethodFunctionHandle(name) => {
            let method_name = name.trim().to_string();
            if method_name.is_empty() {
                return Err(runtime_descriptor_error(
                    "feval",
                    &FEVAL_ERROR_HANDLE_NAME_INVALID,
                ));
            }
            dispatch_callable_with_policy_in_runtime(
                runtime,
                runmat_types::CallableIdentity::Method(runmat_types::MethodId(method_name)),
                runmat_types::CallableFallbackPolicy::RuntimeNameResolution,
                rest,
                requested_outputs,
                input_callsite,
            )
            .await
        }
        Value::BoundFunctionHandle(handle) => {
            let data = handle.into_data();
            let function = data.function;
            let name = data.name;
            let request = crate::user_functions::CallableRequest::bound(
                runmat_value::BoundFunctionBinding {
                    function,
                    authority: data.authority,
                },
                rest.clone(),
                requested_outputs,
            )
            .with_selected_runtime(runtime)
            .with_function_input_callsite(input_callsite);
            if let Some(result) = crate::user_functions::try_call_semantic_descriptor(request).await
            {
                return result;
            }
            Err(runtime_descriptor_error_with_detail(
                "feval",
                &FEVAL_ERROR_SEMANTIC_UNAVAILABLE,
                format!("semantic function handle '{name}' ({function}) is unavailable"),
            ))
        }
        Value::Closure(c) => {
            if let Some(binding) = c.binding.clone() {
                let function = binding.function;
                let function_name = c.function_name.clone();
                let retained = crate::call::descriptor::RetainedClosureValue::new(c);
                if let Some(result) =
                    crate::call::descriptor::try_execute_retained_bound_closure_in_context(
                        runtime,
                        retained,
                        rest,
                        requested_outputs,
                        input_callsite.clone(),
                    )
                    .await?
                {
                    return Ok(result);
                }
                return Err(runtime_descriptor_error_with_detail(
                    "feval",
                    &FEVAL_ERROR_SEMANTIC_UNAVAILABLE,
                    format!(
                        "semantic closure '{}' ({function}) is unavailable",
                        function_name
                    ),
                ));
            }

            let c = c.into_data();

            if c.function_name == CALL_METHOD_BUILTIN_NAME && c.captures.len() >= 2 {
                let base = c.captures[0].clone();
                let method = match &c.captures[1] {
                    Value::String(name) => name.clone(),
                    Value::CharArray(chars) if chars.rows == 1 => chars.data.iter().collect(),
                    _ => {
                        return Err(build_runtime_error(
                            "call_method: closure captures must include method name text",
                        )
                        .with_builtin("call_method")
                        .with_identifier("RunMat:CallMethodNameInvalid")
                        .build())
                    }
                };
                let mut method_args = c.captures.iter().skip(2).cloned().collect::<Vec<_>>();
                method_args.extend(rest);
                return crate::builtins::introspection::call_method::dispatch_call_method(
                    base,
                    method,
                    method_args,
                )
                .await
                .and_then(|value| {
                    value.try_map_owned(
                        crate::call::arguments::adapt_legacy_builtin_result,
                        crate::execution::ExecutionServiceError::into_runtime_error,
                    )
                });
            }

            let mut args = c.captures.clone();
            args.extend(rest);
            if let Some(result) = crate::callsite::with_function_input_callsite(
                input_callsite.clone(),
                crate::user_functions::try_call_semantic_function_by_name_in_context(
                    runtime,
                    &c.function_name,
                    &args,
                    requested_outputs,
                ),
            )
            .await?
            {
                return result;
            }
            call_by_name(
                runtime,
                &c.function_name,
                &args,
                requested_outputs,
                input_callsite,
            )
            .await
        }
        receiver @ Value::Object(_) | receiver @ Value::HandleObject(_) => {
            let path = crate::object::indexing::ObjectSubscriptPath::single(
                crate::object::indexing::ObjectSubscript::parentheses(
                    crate::object::indexing::ObjectIndexSelector::IndexValues {
                        // Preserve standard-substruct selector semantics from
                        // the former encode/decode boundary without rebuilding
                        // the whole descriptor. Overloads still receive the
                        // original protocol payload through protocol_value().
                        components: rest
                            .into_iter()
                            .map(crate::object::indexing::ObjectIndexComponent::from_protocol_value)
                            .collect(),
                    },
                ),
            );
            let request = crate::object::protocol::SubscriptReadRequest::select_outputs(
                requested_outputs,
                crate::object::protocol::ObjectAccessContext::default(),
            );
            let usage = request.sequence_use;
            let sequence = match runtime {
                Some(runtime) => {
                    crate::object::protocol::read_subscript_path_sequence_with_access_in_context(
                        runtime, receiver, path, request,
                    )
                    .await?
                }
                None => {
                    crate::object::protocol::read_subscript_path_sequence_with_access(
                        receiver, path, request,
                    )
                    .await?
                }
            };
            use crate::sequence::ResolveValueSequence;
            sequence.try_map_owned(
                |sequence| {
                    let values = sequence
                        .resolve(usage, crate::sequence::SequenceResolutionContext::default())?;
                    if requested_outputs == 1 {
                        ValueSequence::single(
                            crate::sequence::into_first_output(values)?
                                .expect("one requested output was resolved"),
                        )
                        .map_err(crate::sequence::sequence_error_to_runtime)
                    } else {
                        ValueSequence::from_output_storage(values)
                            .map_err(crate::sequence::sequence_error_to_runtime)
                    }
                },
                |error| error.into_runtime_error(),
            )
        }
        other => Err(runtime_descriptor_error_with_detail(
            "feval",
            &FEVAL_ERROR_FUNCTION_VALUE_UNSUPPORTED,
            format!("{other:?}"),
        )),
    }
}
