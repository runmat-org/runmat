//! Letter classification for host text.

use runmat_builtins::{ISLETTER_DESCRIPTOR, ISLETTER_ERROR_TOO_MANY_OUTPUTS};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::strings::core::classification::classify_text_value;
use crate::{gather_if_needed_async, BuiltinResult};

#[runtime_builtin(
    name = "isletter",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::core::isletter"
)]
pub(crate) async fn isletter_builtin(text: Value) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &ISLETTER_DESCRIPTOR,
        &ISLETTER_ERROR_TOO_MANY_OUTPUTS,
        "isletter",
    )?;
    if !matches!(text, Value::String(_) | Value::CharArray(_)) {
        return Ok(Value::Bool(false));
    }
    let text = gather_if_needed_async(&text)
        .await
        .map_err(|err| map_control_flow_with_builtin(err, "isletter"))?;
    classify_text_value(text, "isletter", |ch| ch.is_alphabetic())
}
