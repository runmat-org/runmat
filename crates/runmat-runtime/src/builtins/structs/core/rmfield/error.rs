use runmat_builtins::BuiltinErrorDescriptor;

pub(super) fn not_enough_inputs() -> crate::RuntimeError {
    build_from(&runmat_builtins::RMFIELD_ERROR_NOT_ENOUGH_INPUTS)
}

pub(super) fn invalid_target(value: &runmat_value::Value) -> crate::RuntimeError {
    let error = &runmat_builtins::RMFIELD_ERROR_INVALID_TARGET;
    build(format!("{} (got {value:?})", error.message), error)
}

pub(super) fn field_name_type(context: Option<&str>) -> crate::RuntimeError {
    contextual(&runmat_builtins::RMFIELD_ERROR_FIELD_NAME_TYPE, context)
}

pub(super) fn empty_field_name(context: Option<&str>) -> crate::RuntimeError {
    contextual(&runmat_builtins::RMFIELD_ERROR_FIELD_NAME_EMPTY, context)
}

pub(super) fn missing_field(name: &str) -> crate::RuntimeError {
    let error = &runmat_builtins::RMFIELD_ERROR_MISSING_FIELD;
    build(format!("{} '{name}'.", error.message), error)
}

fn contextual(
    error: &'static BuiltinErrorDescriptor,
    context: Option<&str>,
) -> crate::RuntimeError {
    let message = context.map_or_else(
        || error.message.to_string(),
        |context| format!("{} ({context})", error.message),
    );
    build(message, error)
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
