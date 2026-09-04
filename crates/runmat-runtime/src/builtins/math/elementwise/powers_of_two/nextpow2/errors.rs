use runmat_builtins::{
    BuiltinErrorDescriptor, NEXTPOW2_ERROR_INTERNAL, NEXTPOW2_ERROR_INVALID_INPUT,
    NEXTPOW2_ERROR_TOO_MANY_OUTPUTS,
};

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use super::BUILTIN_NAME;

pub(super) fn invalid(detail: impl std::fmt::Display) -> RuntimeError {
    with_detail(&NEXTPOW2_ERROR_INVALID_INPUT, detail)
}

pub(super) fn internal(detail: impl std::fmt::Display) -> RuntimeError {
    with_detail(&NEXTPOW2_ERROR_INTERNAL, detail)
}

pub(super) fn reject_excess_outputs() -> BuiltinResult<()> {
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
        Err(with_detail(
            &NEXTPOW2_ERROR_TOO_MANY_OUTPUTS,
            "only one output is defined",
        ))
    } else {
        Ok(())
    }
}

fn with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {detail}", error.message)).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
