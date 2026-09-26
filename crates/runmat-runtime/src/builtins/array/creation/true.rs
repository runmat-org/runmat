//! Logical true constructor binding.
use runmat_macros::runtime_builtin;
use runmat_value::Value;
#[runtime_builtin(
    name = "true",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::creation::r#true"
)]
pub(crate) async fn true_builtin(rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    super::logical_constructor::logical_fill(
        rest,
        true,
        &super::logical_constructor::TRUE_FILL_CONFIG,
    )
    .await
}

runmat_macros::register_constant!(
    "true",
    Value::Bool(true),
    "crate::builtins::array::creation::r#true"
);
