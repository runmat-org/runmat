use runmat_builtins::BuiltinErrorDescriptor;

use crate::{build_runtime_error, RuntimeError};

use super::operation::ExponentialOperation;

pub(super) fn invalid(
    operation: ExponentialOperation,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    described(operation, operation.invalid_input(), detail)
}

pub(super) fn internal(
    operation: ExponentialOperation,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    described(operation, operation.internal_error(), detail)
}

fn described(
    operation: ExponentialOperation,
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {detail}", descriptor.message))
        .with_builtin(operation.name());
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
