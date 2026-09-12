pub(crate) fn plotting_error(builtin: &str, message: impl Into<String>) -> crate::RuntimeError {
    crate::build_runtime_error(message)
        .with_builtin(builtin)
        .build()
}

pub(crate) fn plotting_error_with_source(
    builtin: &str,
    message: impl Into<String>,
    source: impl std::error::Error + Send + Sync + 'static,
) -> crate::RuntimeError {
    crate::build_runtime_error(message)
        .with_builtin(builtin)
        .with_source(source)
        .build()
}
