use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::{
    BuiltinErrorDescriptor, ARRAYFUN_ERROR_CALLBACK_FAILED, ARRAYFUN_ERROR_INTERNAL,
    ARRAYFUN_ERROR_INVALID_INPUT,
};

use super::BUILTIN_NAME;

pub(super) fn arrayfun_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    arrayfun_error_with_message(error.message, error)
}

pub(super) fn arrayfun_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn arrayfun_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    arrayfun_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

pub(super) fn arrayfun_error_with_source(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
    source: RuntimeError,
) -> RuntimeError {
    let identifier = source.identifier().map(str::to_string);
    let mut builder = build_runtime_error(message.into())
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = identifier.as_deref().or(error.identifier) {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn arrayfun_flow(message: impl Into<String>) -> RuntimeError {
    arrayfun_error_with_message(message, &ARRAYFUN_ERROR_INVALID_INPUT)
}

pub(super) fn arrayfun_internal(message: impl Into<String>) -> RuntimeError {
    arrayfun_error_with_message(message, &ARRAYFUN_ERROR_INTERNAL)
}

pub(super) fn arrayfun_flow_with_source(
    message: impl Into<String>,
    source: RuntimeError,
) -> RuntimeError {
    arrayfun_error_with_source(message, &ARRAYFUN_ERROR_CALLBACK_FAILED, source)
}

pub(super) fn format_handler_error(err: &RuntimeError) -> String {
    if let Some(identifier) = err.identifier() {
        if err.message().is_empty() {
            return identifier.to_string();
        }
        if err.message().starts_with(identifier) {
            return err.message().to_string();
        }
        return format!("{identifier}: {}", err.message());
    }
    err.message().to_string()
}
