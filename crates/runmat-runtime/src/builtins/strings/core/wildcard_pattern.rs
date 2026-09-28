//! Lazy wildcard pattern construction.

use runmat_builtins::{WILDCARD_PATTERN_DESCRIPTOR, WILDCARD_PATTERN_ERROR_TOO_MANY_OUTPUTS};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::strings::core::compat::pattern_object;
use crate::builtins::strings::core::patterns::{bounded_regex, BoundedPatternKind};
use crate::BuiltinResult;

#[runtime_builtin(
    name = "wildcardPattern",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::core::wildcard_pattern"
)]
pub(crate) async fn wildcard_pattern_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &WILDCARD_PATTERN_DESCRIPTOR,
        &WILDCARD_PATTERN_ERROR_TOO_MANY_OUTPUTS,
        "wildcardPattern",
    )?;
    let regex = bounded_regex(rest, BoundedPatternKind::Wildcard).await?;
    Ok(pattern_object(&regex))
}
