use runmat_builtins::BuiltinErrorDescriptor;

pub(super) fn too_many_inputs() -> crate::RuntimeError {
    build_from(&runmat_builtins::ORDERFIELDS_ERROR_TOO_MANY_INPUTS)
}

pub(super) fn invalid_target(value: &runmat_value::Value) -> crate::RuntimeError {
    let error = &runmat_builtins::ORDERFIELDS_ERROR_INVALID_TARGET;
    build(format!("{} (got {value:?})", error.message), error)
}

pub(super) fn empty_struct_array() -> crate::RuntimeError {
    build_from(&runmat_builtins::ORDERFIELDS_ERROR_EMPTY_STRUCT_ARRAY)
}

pub(super) fn no_fields() -> crate::RuntimeError {
    build_from(&runmat_builtins::ORDERFIELDS_ERROR_NO_FIELDS)
}

pub(super) fn invalid_name(context: impl std::fmt::Display) -> crate::RuntimeError {
    contextual(
        &runmat_builtins::ORDERFIELDS_ERROR_INVALID_NAME_LIST,
        context,
    )
}

pub(super) fn empty_name(context: impl std::fmt::Display) -> crate::RuntimeError {
    contextual(
        &runmat_builtins::ORDERFIELDS_ERROR_EMPTY_FIELD_NAME,
        context,
    )
}

pub(super) fn invalid_permutation() -> crate::RuntimeError {
    build_from(&runmat_builtins::ORDERFIELDS_ERROR_INVALID_PERMUTATION)
}

pub(super) fn index_not_integer() -> crate::RuntimeError {
    build_from(&runmat_builtins::ORDERFIELDS_ERROR_INDEX_NOT_INTEGER)
}

pub(super) fn index_out_of_range() -> crate::RuntimeError {
    build_from(&runmat_builtins::ORDERFIELDS_ERROR_INDEX_OUT_OF_RANGE)
}

pub(super) fn duplicate_index() -> crate::RuntimeError {
    build_from(&runmat_builtins::ORDERFIELDS_ERROR_DUPLICATE_INDEX)
}

pub(super) fn field_mismatch() -> crate::RuntimeError {
    build_from(&runmat_builtins::ORDERFIELDS_ERROR_FIELD_MISMATCH)
}

pub(super) fn unknown_field(name: &str) -> crate::RuntimeError {
    contextual(
        &runmat_builtins::ORDERFIELDS_ERROR_UNKNOWN_FIELD,
        format!("'{name}'"),
    )
}

pub(super) fn duplicate_field(name: &str) -> crate::RuntimeError {
    contextual(
        &runmat_builtins::ORDERFIELDS_ERROR_DUPLICATE_FIELD,
        format!("'{name}'"),
    )
}

pub(super) fn invalid_order() -> crate::RuntimeError {
    build_from(&runmat_builtins::ORDERFIELDS_ERROR_INVALID_ORDER)
}

pub(super) fn rebuild(message: impl std::fmt::Display) -> crate::RuntimeError {
    contextual(&runmat_builtins::ORDERFIELDS_ERROR_REBUILD_FAILED, message)
}

fn contextual(
    error: &'static BuiltinErrorDescriptor,
    context: impl std::fmt::Display,
) -> crate::RuntimeError {
    build(format!("{} ({context})", error.message), error)
}

fn build_from(error: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    build(error.message, error)
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
