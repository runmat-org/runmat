//! Conjugation-specific compatibility admission.

use runmat_builtins::CONJ_CHARACTER_INPUT_EXTENSION;
use runmat_value::Value;

use super::BUILTIN_NAME;
use crate::BuiltinResult;

pub(super) fn ensure_character_enabled(value: &Value) -> BuiltinResult<()> {
    if matches!(value, Value::CharArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CONJ_CHARACTER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}
