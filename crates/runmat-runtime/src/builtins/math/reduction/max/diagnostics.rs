use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::{
    BuiltinErrorDescriptor, MAX_ERROR_INTERNAL, MAX_ERROR_INVALID_ARGUMENT,
    MAX_ERROR_INVALID_INPUT, MAX_ERROR_SIZE_MISMATCH,
};
const NAME: &str = "max";

fn max_descriptor_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn max_descriptor_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    max_descriptor_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

pub(super) fn max_invalid_argument(detail: impl AsRef<str>) -> RuntimeError {
    max_descriptor_error_with_detail(&MAX_ERROR_INVALID_ARGUMENT, detail)
}

pub(super) fn max_invalid_input(detail: impl AsRef<str>) -> RuntimeError {
    max_descriptor_error_with_detail(&MAX_ERROR_INVALID_INPUT, detail)
}

pub(super) fn max_size_mismatch(detail: impl AsRef<str>) -> RuntimeError {
    max_descriptor_error_with_detail(&MAX_ERROR_SIZE_MISMATCH, detail)
}

pub(super) fn max_internal_error(detail: impl AsRef<str>) -> RuntimeError {
    max_descriptor_error_with_detail(&MAX_ERROR_INTERNAL, detail)
}
