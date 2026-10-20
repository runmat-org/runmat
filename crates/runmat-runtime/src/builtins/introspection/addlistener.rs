use runmat_macros::runtime_builtin;
use runmat_value::Value;

/// Registration is effectful even when its result is discarded.
#[runtime_builtin(
    name = "addlistener",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::addlistener"
)]
pub async fn addlistener_builtin(
    target: Value,
    event_name: runmat_value::Utf8Text,
    callback: Value,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::ADDLISTENER_DESCRIPTOR,
        &runmat_builtins::ADDLISTENER_ERROR_OUTPUTS,
        "addlistener",
    )?;
    // The persistent event registry still owns String keys. This is that
    // explicit metadata boundary, not a general argument detachment.
    crate::addlistener_builtin(target, event_name.as_str().to_owned(), callback).await
}
