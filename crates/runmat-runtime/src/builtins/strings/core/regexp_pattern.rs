//! Regex-backed pattern construction.

use runmat_builtins::{REGEXP_PATTERN_DESCRIPTOR, REGEXP_PATTERN_ERROR_TOO_MANY_OUTPUTS};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::strings::common::{scalar_text_input, text_input_admission};
use crate::builtins::strings::core::compat::{compat_error, pattern_object_in_context};
use crate::BuiltinResult;

#[runtime_builtin(
    name = "regexpPattern",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::strings::core::regexp_pattern"
)]
pub(crate) async fn regexp_pattern_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    text: Value,
) -> BuiltinResult<Value> {
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
    let text = scalar_text_input(&text, runtime.as_ref(), |runtime, cause| {
        text_input_admission("regexpPattern", runtime, cause)
    })?
    .ok_or_else(|| compat_error("regexpPattern", "regexpPattern: expected a text scalar"))?;
    pattern_object_in_context(runtime.as_ref(), text.as_str())
}
