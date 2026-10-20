//! Lazy wildcard pattern construction.

use runmat_builtins::{WILDCARD_PATTERN_DESCRIPTOR, WILDCARD_PATTERN_ERROR_TOO_MANY_OUTPUTS};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::strings::core::compat::pattern_object_in_context;
use crate::builtins::strings::core::patterns::{bounded_regex, BoundedPatternKind};
use crate::BuiltinResult;

#[runtime_builtin(
    name = "wildcardPattern",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::core::wildcard_pattern"
)]
pub(crate) async fn wildcard_pattern_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &WILDCARD_PATTERN_DESCRIPTOR,
        &WILDCARD_PATTERN_ERROR_TOO_MANY_OUTPUTS,
        "wildcardPattern",
    )?;
    let regex = bounded_regex(rest, BoundedPatternKind::Wildcard).await?;
    pattern_object_in_context(runtime.as_ref(), &regex)
}
