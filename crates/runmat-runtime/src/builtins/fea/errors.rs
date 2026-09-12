use crate::build_runtime_error;
use crate::builtins::fea::contracts::descriptors::{ERROR_INTERNAL, ERROR_OPERATION};
use crate::operations::OperationErrorEnvelope;
use crate::RuntimeError;
use runmat_builtins::BuiltinErrorDescriptor;

pub(in crate::builtins::fea) fn operation_error(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    source: OperationErrorEnvelope,
) -> RuntimeError {
    let message = format!(
        "{}: {}: {}",
        error.message, source.error_code, source.message
    );
    build_runtime_error(message)
        .with_builtin(builtin)
        .with_identifier(
            error
                .identifier
                .unwrap_or(ERROR_OPERATION.identifier.expect("descriptor identifier")),
        )
        .build()
}

pub(in crate::builtins::fea) fn builtin_error(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    build_runtime_error(format!("{}: {}", error.message, message.into()))
        .with_builtin(builtin)
        .with_identifier(
            error
                .identifier
                .unwrap_or(ERROR_INTERNAL.identifier.expect("descriptor identifier")),
        )
        .build()
}

pub(in crate::builtins::fea) fn builtin_error_with_source<E>(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
    source: E,
) -> RuntimeError
where
    E: std::error::Error + Send + Sync + 'static,
{
    build_runtime_error(format!("{}: {}", error.message, message.into()))
        .with_builtin(builtin)
        .with_identifier(
            error
                .identifier
                .unwrap_or(ERROR_INTERNAL.identifier.expect("descriptor identifier")),
        )
        .with_source(source)
        .build()
}

pub(in crate::builtins::fea) fn sanitize_id(id: &str) -> String {
    id.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}
