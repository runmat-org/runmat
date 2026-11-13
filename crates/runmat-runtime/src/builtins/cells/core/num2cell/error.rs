use runmat_builtins::{
    BuiltinErrorDescriptor, NUM2CELL_ERROR_INTERNAL, NUM2CELL_ERROR_INVALID_INPUT,
};

fn build(
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl Into<String>,
) -> crate::RuntimeError {
    let mut error =
        crate::build_runtime_error(format!("num2cell: {}", detail.into())).with_builtin("num2cell");
    if let Some(identifier) = descriptor.identifier {
        error = error.with_identifier(identifier);
    }
    error.build()
}

pub(super) fn invalid_input(detail: impl Into<String>) -> crate::RuntimeError {
    build(&NUM2CELL_ERROR_INVALID_INPUT, detail)
}

pub(super) fn internal(detail: impl Into<String>) -> crate::RuntimeError {
    build(&NUM2CELL_ERROR_INTERNAL, detail)
}
