use super::*;
pub(super) fn corr_error(message: impl Into<String>) -> RuntimeError {
    corr_descriptor_error(message, &ERROR_INVALID_ARGUMENT)
}

pub(super) fn corr_internal(message: impl Into<String>) -> RuntimeError {
    corr_descriptor_error(message, &ERROR_INTERNAL)
}

pub(super) fn corr_descriptor_error(
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
