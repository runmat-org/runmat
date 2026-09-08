use runmat_builtins::BuiltinErrorDescriptor;

pub(super) fn invalid_target(message: impl Into<String>) -> crate::RuntimeError {
    build(message, &runmat_builtins::FIELDNAMES_ERROR_INVALID_TARGET)
}

pub(super) fn invalid_struct_array() -> crate::RuntimeError {
    let error = &runmat_builtins::FIELDNAMES_ERROR_STRUCT_ARRAY_CONTENTS;
    build(error.message, error)
}

pub(super) fn internal(message: impl Into<String>) -> crate::RuntimeError {
    build(message, &runmat_builtins::FIELDNAMES_ERROR_INTERNAL)
}

fn build(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> crate::RuntimeError {
    let mut builder = crate::build_runtime_error(message).with_builtin(super::BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
