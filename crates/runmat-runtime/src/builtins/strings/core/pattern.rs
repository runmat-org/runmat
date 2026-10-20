//! Literal pattern construction.

use runmat_builtins::{PATTERN_DESCRIPTOR, PATTERN_ERROR_TOO_MANY_OUTPUTS};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::strings::common::{scalar_text_input, text_input_admission};
use crate::builtins::strings::core::compat::{compat_error, pattern_object_in_context};
use crate::BuiltinResult;

#[runtime_builtin(
    name = "pattern",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::strings::core::pattern"
)]
pub(crate) async fn pattern_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    text: Value,
) -> BuiltinResult<Value> {
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
    let text = scalar_text_input(&text, runtime.as_ref(), |runtime, cause| {
        text_input_admission("pattern", runtime, cause)
    })?
    .ok_or_else(|| compat_error("pattern", "pattern: expected a text scalar"))?;
    pattern_object_in_context(runtime.as_ref(), &regex::escape(text.as_str()))
}
