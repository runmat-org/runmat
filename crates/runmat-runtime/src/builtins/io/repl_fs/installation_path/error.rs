use runmat_builtins::BuiltinErrorDescriptor;

pub(super) fn from_descriptor(
    identity: &'static str,
    descriptor: &'static BuiltinErrorDescriptor,
) -> crate::RuntimeError {
    let mut error = crate::build_runtime_error(descriptor.message).with_builtin(identity);
    if let Some(identifier) = descriptor.identifier {
        error = error.with_identifier(identifier);
    }
    error.build()
}
