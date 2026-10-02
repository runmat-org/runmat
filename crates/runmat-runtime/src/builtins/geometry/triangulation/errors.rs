use super::*;

pub(super) fn invalid(detail: impl AsRef<str>) -> crate::RuntimeError {
    error(&ERROR_INVALID_ARGUMENT, detail)
}

pub(super) fn unsupported(detail: impl AsRef<str>) -> crate::RuntimeError {
    error(&ERROR_UNSUPPORTED, detail)
}

pub(super) fn error(
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> crate::RuntimeError {
    build_runtime_error(format!("{}: {}", descriptor.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME)
        .with_identifier(descriptor.identifier.unwrap_or("RunMat:DelaunayTri:Error"))
        .build()
}
