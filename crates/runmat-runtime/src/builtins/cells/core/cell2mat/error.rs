use runmat_builtins::{BuiltinErrorDescriptor, CELL2MAT_ERROR_INVALID_INPUT};
pub(super) use runmat_builtins::{
    CELL2MAT_ERROR_INTERNAL, CELL2MAT_ERROR_INVALID_CONTENTS, CELL2MAT_ERROR_SIZE_EXCEEDED,
};

pub(super) fn invalid_input(message: impl Into<String>) -> crate::RuntimeError {
    cell2mat_error_with_message(
        format!("cell2mat: {}", message.into()),
        &CELL2MAT_ERROR_INVALID_INPUT,
    )
}

pub(super) fn cell2mat_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> crate::RuntimeError {
    let mut builder = crate::build_runtime_error(message).with_builtin("cell2mat");
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
