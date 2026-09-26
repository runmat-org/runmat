//! Logical false constructor binding.
use runmat_macros::runtime_builtin;
use runmat_value::Value;
#[runtime_builtin(
    name = "false",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::creation::r#false"
)]
pub(crate) async fn false_builtin(rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    super::logical_constructor::logical_fill(
        rest,
        false,
        &super::logical_constructor::FALSE_FILL_CONFIG,
    )
    .await
}

runmat_macros::register_constant!(
    "false",
    Value::Bool(false),
    "crate::builtins::array::creation::r#false"
);
