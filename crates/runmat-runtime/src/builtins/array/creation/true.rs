//! Logical true constructor binding.
use runmat_macros::runtime_builtin;
use runmat_value::Value;
#[runtime_builtin(
    name = "true",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::creation::r#true"
)]
pub(crate) async fn true_builtin(
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
        let error = &runmat_builtins::TRUE_ERROR_TOO_MANY_OUTPUTS;
        return Err(crate::build_runtime_error(error.message)
            .with_builtin("true")
            .with_identifier(error.identifier.expect("output count error identifier"))
            .build());
    }
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
