use runmat_builtins::BuiltinErrorDescriptor;

pub(super) fn catalog(
    error: &'static BuiltinErrorDescriptor,
    identity: &'static str,
) -> crate::RuntimeError {
    let mut builder = crate::build_runtime_error(error.message).with_builtin(identity);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn detail(
    error: &'static BuiltinErrorDescriptor,
    identity: &'static str,
    detail: impl Into<String>,
) -> crate::RuntimeError {
    let mut builder = crate::build_runtime_error(detail).with_builtin(identity);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn source(
    error: &'static BuiltinErrorDescriptor,
    identity: &'static str,
    source: crate::RuntimeError,
) -> crate::RuntimeError {
    let mut builder = crate::build_runtime_error(error.message)
        .with_builtin(identity)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
