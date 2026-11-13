//! Stable timer error and method-output compatibility admission.

use super::*;

pub(super) const BUILTIN_TIMER: &str = "timer";

pub(super) fn timer_error(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.is_empty() {
        error.message.to_string()
    } else {
        detail.to_string()
    };
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_TIMER);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn timer_runtime(runtime: Option<RuntimeContext>) -> BuiltinResult<RuntimeContext> {
    runtime.ok_or_else(|| {
        timer_error(
            &TIMER_ERROR_INVALID_INPUT,
            "timer: runtime session is required for timer ownership",
        )
    })
}

pub(super) fn admit_timer_method_outputs(
    runtime: &RuntimeContext,
    requested_outputs: Option<usize>,
    method: &str,
) -> BuiltinResult<()> {
    match requested_outputs {
        None | Some(0) => Ok(()),
        Some(1) => crate::compatibility::ensure_builtin_extension_enabled_in_context(
            runtime,
            &TIMER_METHOD_STATUS_EXTENSION,
            method,
        ),
        Some(_) => Err(timer_error(
            &TIMER_ERROR_INVALID_INPUT,
            format!("{method}: at most one status output is supported"),
        )),
    }
}
