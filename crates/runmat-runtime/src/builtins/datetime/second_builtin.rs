//! Second extraction consumes stored datetime serials, never the Clock port.
use super::*;

#[runmat_macros::runtime_builtin(
    name = "second",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn second_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::DATETIME_SECOND_DESCRIPTOR,
        &runmat_builtins::SECOND_ERROR_TOO_MANY_OUTPUTS,
        "second",
    )?;
    map_stored_datetime_component(runtime.as_ref(), &value, "second", |naive| {
        naive.second() as f64 + f64::from(naive.nanosecond()) / 1_000_000_000.0
    })
    .await
}
