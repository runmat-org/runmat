use crate::execution::RootedValue;
use runmat_builtins::{CALL_METHOD_ERROR_NAME_INVALID, CALL_METHOD_ERROR_RECEIVER_INVALID};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

pub(crate) async fn dispatch_call_method(
    base: Value,
    method: String,
    rest: Vec<Value>,
) -> crate::BuiltinResult<RootedValue> {
    let method = method.trim().to_string();
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
            let mut args = Vec::with_capacity(1 + rest.len());
            args.push(receiver.clone());
            args.extend(rest);
            let requested_outputs = crate::current_requested_outputs();
            match crate::dispatch_object_external_member(
                class_name,
                &method,
                args.clone(),
                requested_outputs,
            )
            .await
            {
                Ok(v) => return Ok(v),
                Err(err) if crate::is_undefined_function_error(&err) => {}
                Err(err) => return Err(err),
            }
            let (identity, fallback_policy) = crate::callable_identity_for_handle_name(&method);
            crate::dispatch_callable_with_policy(identity, fallback_policy, args, requested_outputs)
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
    builtin_path = "crate::builtins::introspection::call_method"
)]
pub async fn call_method_builtin(
    base: Value,
    method: String,
    rest: Vec<Value>,
) -> crate::BuiltinResult<RootedValue> {
    dispatch_call_method(base, method, rest).await
}
