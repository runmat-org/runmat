use super::*;
pub(super) fn invalid_argument(message: impl Into<String>) -> RuntimeError {
    descriptor_error(&ERROR_INVALID_ARGUMENT, message)
}

pub(super) fn internal_error(message: impl Into<String>) -> RuntimeError {
    descriptor_error(&ERROR_INTERNAL, message)
}

pub(super) fn descriptor_error(
    descriptor: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
