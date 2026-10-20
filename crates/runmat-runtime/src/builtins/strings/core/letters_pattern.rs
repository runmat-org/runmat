//! Greedy Unicode-letter pattern construction.

use runmat_builtins::{LETTERS_PATTERN_DESCRIPTOR, LETTERS_PATTERN_ERROR_TOO_MANY_OUTPUTS};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::strings::core::compat::pattern_object_in_context;
use crate::builtins::strings::core::patterns::{bounded_regex, BoundedPatternKind};
use crate::BuiltinResult;

#[runtime_builtin(
    name = "lettersPattern",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::core::letters_pattern"
)]
pub(crate) async fn letters_pattern_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &LETTERS_PATTERN_DESCRIPTOR,
        &LETTERS_PATTERN_ERROR_TOO_MANY_OUTPUTS,
        "lettersPattern",
    )?;
    let regex = bounded_regex(rest, BoundedPatternKind::Letters).await?;
    pattern_object_in_context(runtime.as_ref(), &regex)
}
