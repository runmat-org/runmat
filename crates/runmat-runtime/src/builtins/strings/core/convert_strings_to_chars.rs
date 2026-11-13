//! Top-level string-to-character conversion with non-string passthrough.

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::strings::core::conversion::{convert_variadic, string_to_chars};
use crate::BuiltinResult;

#[runtime_builtin(
    name = "convertStringsToChars",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::core::convert_strings_to_chars"
)]
pub(crate) async fn convert_strings_to_chars_builtin(
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    convert_variadic(value, rest, |value| {
        string_to_chars(value, "convertStringsToChars")
    })
}
