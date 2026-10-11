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
    event_name: String,
    callback: Value,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::ADDLISTENER_DESCRIPTOR,
        &runmat_builtins::ADDLISTENER_ERROR_OUTPUTS,
        "addlistener",
    )?;
    crate::addlistener_builtin(target, event_name, callback).await
}
