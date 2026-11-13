use crate::execution::RootedValue;
use runmat_builtins::{CALL_METHOD_ERROR_NAME_INVALID, CALL_METHOD_ERROR_RECEIVER_INVALID};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

pub(crate) async fn dispatch_call_method(
    runtime: Option<&crate::context::RuntimeContext>,
    base: Value,
    method: impl AsRef<str>,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<RootedValue> {
    let method = method.as_ref().trim();
    if method.is_empty() {
        return Err(crate::runtime_descriptor_error(
            "call_method",
            &CALL_METHOD_ERROR_NAME_INVALID,
        ));
    }
    match base {
        receiver @ Value::Object(_) | receiver @ Value::HandleObject(_) => {
            let class_name = crate::object_receiver_class_name(&receiver).ok_or_else(|| {
                crate::runtime_descriptor_error("call_method", &CALL_METHOD_ERROR_RECEIVER_INVALID)
            })?;
            let args = rest
                .into_function_arguments()
                .prepend_routing(receiver, runtime)?;
            let args = crate::call::BuiltinArguments::new(args)
                .into_supplied_function_arguments(runtime)?;
            let requested_outputs = crate::current_requested_outputs();
            let request = crate::user_functions::CallableRequest::rooted_resolved(
                crate::class_member_identity(&class_name, method),
                runmat_types::CallableFallbackPolicy::ExternalBoundary,
                args,
                requested_outputs,
            )
            .with_selected_runtime(runtime);
            let args = match crate::user_functions::try_call_semantic_descriptor_owned(request)
                .await
            {
                std::ops::ControlFlow::Break(result) => {
                    return result
                        .and_then(crate::call::arguments::project_rooted_legacy_builtin_value_abi);
                }
                std::ops::ControlFlow::Continue(args) => crate::call::BuiltinArguments::new(args),
            };
            let (identity, fallback_policy) = crate::callable_identity_for_handle_name(method);
            crate::dispatch_retained_callable_with_policy_in_context(
                runtime,
                identity,
                fallback_policy,
                args,
                requested_outputs,
            )
            .await
        }
        other => Err(crate::runtime_descriptor_error_with_detail(
            "call_method",
            &CALL_METHOD_ERROR_RECEIVER_INVALID,
            format!("unsupported receiver {other:?} for method '{method}'"),
        )),
    }
}

#[runtime_builtin(
    name = "call_method",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::call_method"
)]
pub async fn call_method_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    base: Value,
    method: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<RootedValue> {
    // A semantic method name must be scalar text. The shared formatting
    // converter deliberately also accepts numbers and booleans elsewhere.
    if !crate::builtins::common::validation::value_is_text_scalar(&method) {
        return Err(crate::runtime_descriptor_error(
            "call_method",
            &CALL_METHOD_ERROR_NAME_INVALID,
        ));
    }
    let method = crate::call::argument_text::convert(&method, runtime.as_ref())?;
    dispatch_call_method(runtime.as_ref(), base, method, rest).await
}
