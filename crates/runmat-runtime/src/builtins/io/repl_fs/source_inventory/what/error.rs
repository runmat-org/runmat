use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::BuiltinErrorDescriptor;

pub(super) fn contract(descriptor: &'static BuiltinErrorDescriptor) -> RuntimeError {
    message(descriptor, descriptor.message)
}

pub(super) fn message(
    descriptor: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("what");
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
