use runmat_value::{decode_utf16_code_units, CharArray, Utf16CodeUnit};

use crate::{BuiltinResult, RuntimeError};

/// Decode an exact UTF-16 payload while preserving the caller's canonical
/// builtin descriptor and retaining the concrete malformed-code-unit cause.
pub fn decode_utf16_with_error<F>(units: &[Utf16CodeUnit], error: F) -> BuiltinResult<String>
where
    F: FnOnce() -> RuntimeError,
{
    decode_utf16_code_units(units).map_err(|source| {
        let mut error = error();
        error.source = Some(Box::new(source));
        error
    })
}

/// Decode a character array whose geometry has already been validated by the
/// owning builtin. This does not reinterpret pages or relax scalar/row policy.
pub fn decode_char_array_with_error<F>(array: &CharArray, error: F) -> BuiltinResult<String>
where
    F: FnOnce() -> RuntimeError,
{
    decode_utf16_with_error(array.data(), error)
}
