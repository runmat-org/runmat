use runmat_builtins::{
    BuiltinErrorDescriptor, CELLSTR_ERROR_INTERNAL, CELLSTR_ERROR_INVALID_CONTENTS,
    CELLSTR_ERROR_INVALID_INPUT,
};

fn build(
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl Into<String>,
) -> crate::RuntimeError {
    let mut error =
        crate::build_runtime_error(format!("cellstr: {}", detail.into())).with_builtin("cellstr");
    if let Some(identifier) = descriptor.identifier {
        error = error.with_identifier(identifier);
    }
    error.build()
}

pub(super) fn invalid_input() -> crate::RuntimeError {
    build(
        &CELLSTR_ERROR_INVALID_INPUT,
        "input must be a character array, string array, or supported RunMat text value",
    )
}

pub(super) fn invalid_contents() -> crate::RuntimeError {
    build(
        &CELLSTR_ERROR_INVALID_CONTENTS,
        "cell array elements must be character vectors or string scalars",
    )
}

pub(super) fn internal(detail: impl Into<String>) -> crate::RuntimeError {
    build(&CELLSTR_ERROR_INTERNAL, detail)
}
