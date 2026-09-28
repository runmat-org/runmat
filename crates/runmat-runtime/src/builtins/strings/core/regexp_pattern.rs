//! Regex-backed pattern construction.

use runmat_builtins::{REGEXP_PATTERN_DESCRIPTOR, REGEXP_PATTERN_ERROR_TOO_MANY_OUTPUTS};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::strings::core::compat::{compat_error, pattern_object, scalar_text};
use crate::BuiltinResult;

#[runtime_builtin(
    name = "regexpPattern",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::core::regexp_pattern"
)]
pub(crate) async fn regexp_pattern_builtin(text: Value) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &REGEXP_PATTERN_DESCRIPTOR,
        &REGEXP_PATTERN_ERROR_TOO_MANY_OUTPUTS,
        "regexpPattern",
    )?;
    if crate::dispatcher::value_contains_gpu(&text) {
        return Err(compat_error(
            "regexpPattern",
            "regexpPattern: expected a host text scalar",
        ));
    }
    Ok(pattern_object(&scalar_text(&text, "regexpPattern")?))
}
