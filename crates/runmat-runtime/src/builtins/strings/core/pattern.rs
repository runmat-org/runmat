//! Literal pattern construction.

use runmat_builtins::{PATTERN_DESCRIPTOR, PATTERN_ERROR_TOO_MANY_OUTPUTS};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::strings::core::compat::{compat_error, pattern_object, scalar_text};
use crate::BuiltinResult;

#[runtime_builtin(
    name = "pattern",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::core::pattern"
)]
pub(crate) async fn pattern_builtin(text: Value) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &PATTERN_DESCRIPTOR,
        &PATTERN_ERROR_TOO_MANY_OUTPUTS,
        "pattern",
    )?;
    if crate::dispatcher::value_contains_gpu(&text) {
        return Err(compat_error(
            "pattern",
            "pattern: expected a host text scalar",
        ));
    }
    Ok(pattern_object(&regex::escape(&scalar_text(
        &text, "pattern",
    )?)))
}
