use runmat_builtins::BuiltinErrorDescriptor;

pub(super) fn field_name_type() -> crate::RuntimeError {
    build(
        runmat_builtins::ISFIELD_ERROR_FIELD_NAME_TYPE.message,
        &runmat_builtins::ISFIELD_ERROR_FIELD_NAME_TYPE,
    )
}

pub(super) fn cell_element(value: &runmat_value::Value) -> crate::RuntimeError {
    let error = &runmat_builtins::ISFIELD_ERROR_CELL_ELEMENT_TYPE;
    build(format!("{} (got {value:?})", error.message), error)
}

pub(super) fn internal(message: impl Into<String>) -> crate::RuntimeError {
    build(message, &runmat_builtins::ISFIELD_ERROR_INTERNAL)
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
