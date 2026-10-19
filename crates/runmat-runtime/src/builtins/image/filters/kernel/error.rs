//! Pure filter-plan diagnostic retained across host and provider adapters.

#[derive(Debug)]
pub struct ImfilterKernelError {
    builtin: String,
    pub(in crate::builtins::image::filters) message: String,
}

impl ImfilterKernelError {
    pub(in crate::builtins::image::filters) fn new(
        builtin: &str,
        message: impl Into<String>,
    ) -> Self {
        Self {
            builtin: builtin.to_owned(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ImfilterKernelError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ImfilterKernelError {}

impl From<ImfilterKernelError> for crate::RuntimeError {
    fn from(error: ImfilterKernelError) -> Self {
        let mut result = crate::build_runtime_error(error.message.clone())
            .with_builtin(&error.builtin)
            .build();
        result.source = Some(Box::new(error));
        result
    }
}

pub type ImfilterKernelResult<T> = Result<T, ImfilterKernelError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_error_keeps_calling_builtin_and_downcastable_source() {
        let error: crate::RuntimeError =
            ImfilterKernelError::new("filter2", "filter2: invalid kernel").into();
        assert_eq!(error.context.builtin.as_deref(), Some("filter2"));
        assert_eq!(error.message(), "filter2: invalid kernel");
        assert!(error
            .source
            .as_ref()
            .is_some_and(|source| source.downcast_ref::<ImfilterKernelError>().is_some()));
    }
}
