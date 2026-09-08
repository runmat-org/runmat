use runmat_builtins::{
    BuiltinErrorDescriptor, CELL_ERROR_INTERNAL, CELL_ERROR_INVALID_INPUT, CELL_ERROR_INVALID_SIZE,
};

fn build(
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl Into<String>,
) -> crate::RuntimeError {
    let mut error =
        crate::build_runtime_error(format!("cell: {}", detail.into())).with_builtin("cell");
    if let Some(identifier) = descriptor.identifier {
        error = error.with_identifier(identifier);
    }
    error.build()
}

pub(super) fn invalid_input(detail: impl Into<String>) -> crate::RuntimeError {
    build(&CELL_ERROR_INVALID_INPUT, detail)
}

pub(super) fn invalid_size(detail: impl Into<String>) -> crate::RuntimeError {
    build(&CELL_ERROR_INVALID_SIZE, detail)
}

pub(super) fn internal(detail: impl Into<String>) -> crate::RuntimeError {
    build(&CELL_ERROR_INTERNAL, detail)
}
