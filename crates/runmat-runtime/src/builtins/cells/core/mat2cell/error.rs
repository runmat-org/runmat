use runmat_builtins::BuiltinErrorDescriptor;

pub(super) use runmat_builtins::{
    MAT2CELL_ERROR_INTERNAL, MAT2CELL_ERROR_INVALID_INPUT, MAT2CELL_ERROR_INVALID_PARTITION,
    MAT2CELL_ERROR_SIZE_EXCEEDED,
};

pub(super) fn mat2cell_error_with_message(
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> crate::RuntimeError {
    let mut error = crate::build_runtime_error(message).with_builtin("mat2cell");
    if let Some(identifier) = descriptor.identifier {
        error = error.with_identifier(identifier);
    }
    error.build()
}
