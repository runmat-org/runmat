//! Logical false constructor binding.
use runmat_macros::runtime_builtin;
use runmat_value::Value;
#[runtime_builtin(
    name = "false",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::creation::r#false"
)]
pub(crate) async fn false_builtin(
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
        let error = &runmat_builtins::FALSE_ERROR_TOO_MANY_OUTPUTS;
        return Err(crate::build_runtime_error(error.message)
            .with_builtin("false")
            .with_identifier(error.identifier.expect("output count error identifier"))
            .build());
    }
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
