//! Retain the original invocation selection without activating a GUI role.

use super::*;

pub(crate) fn retain_behavior(
    value: Value,
    builtin: &'static str,
) -> BuiltinResult<PlotCallableToken> {
    let empty = match &value {
        Value::String(text) => text.trim().is_empty(),
        Value::CharArray(chars) => chars.data.is_empty(),
        Value::Tensor(tensor) => tensor.is_empty(),
        Value::Cell(cell) => cell.data.is_empty(),
        _ => false,
    };
    let authority = if empty {
        None
    } else {
        crate::context::legacy::active()
            .map(|runtime| runtime.capture_background_callback(false))
            .transpose()?
    };
    retain_root(value, authority, builtin)
}
