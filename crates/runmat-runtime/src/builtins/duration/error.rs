use super::*;
use runmat_builtins::DurationErrorKind;

pub(super) fn duration_error(kind: DurationErrorKind, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_identifier(kind.identifier())
        .build()
}

/// Keep the delegated diagnostic as the cause, and retain its identifier when
/// it has one. The duration identifier is only a fallback for an unclassified
/// delegated failure; the caller supplies the same public message prefix as
/// before the cause was retained.
pub(super) fn duration_error_from(
    kind: DurationErrorKind,
    prefix: &str,
    source: RuntimeError,
) -> RuntimeError {
    let identifier = source.identifier().unwrap_or(kind.identifier()).to_owned();
    let retry_policy = source.gpu_gather_retry();
    let message = format!("{prefix}{}", source.message());
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_identifier(identifier)
        .with_gpu_gather_retry(retry_policy)
        .with_source(source)
        .build()
}
