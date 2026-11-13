use runmat_builtins::BuiltinErrorDescriptor;

pub(super) fn builtin(
    identity: &'static str,
    descriptor: &'static BuiltinErrorDescriptor,
) -> crate::RuntimeError {
    message(identity, descriptor, descriptor.message)
}

pub(super) fn message(
    identity: &'static str,
    descriptor: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> crate::RuntimeError {
    let mut builder = crate::build_runtime_error(message).with_builtin(identity);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
