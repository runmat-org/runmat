use runmat_builtins::{CELL2STRUCT_ERROR_INVALID_INPUT, CELL2STRUCT_ERROR_SHAPE};

fn build(
    descriptor: &'static runmat_builtins::BuiltinErrorDescriptor,
    detail: impl Into<String>,
) -> crate::RuntimeError {
    let mut error = crate::build_runtime_error(format!("cell2struct: {}", detail.into()))
        .with_builtin("cell2struct");
    if let Some(identifier) = descriptor.identifier {
        error = error.with_identifier(identifier);
    }
    error.build()
}

pub(super) fn invalid(detail: impl Into<String>) -> crate::RuntimeError {
    build(&CELL2STRUCT_ERROR_INVALID_INPUT, detail)
}

pub(super) fn shape(detail: impl Into<String>) -> crate::RuntimeError {
    build(&CELL2STRUCT_ERROR_SHAPE, detail)
}
