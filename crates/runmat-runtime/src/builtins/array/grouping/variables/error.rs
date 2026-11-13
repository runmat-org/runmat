//! Grouping-variable failures retain their original typed source.
#[derive(Debug)]
pub(crate) struct VariableError {
    message: String,
    source: Option<Box<runmat_async::runtime_error::RuntimeErrorSource>>,
}

impl VariableError {
    pub(crate) fn with_source(
        message: impl Into<String>,
        source: impl runmat_async::RuntimeErrorSourceBound,
    ) -> Self {
        Self {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    pub(crate) fn attach(self, mut error: crate::RuntimeError) -> crate::RuntimeError {
        error.source = Some(Box::new(self));
        error
    }
}

impl From<String> for VariableError {
    fn from(message: String) -> Self {
        Self {
            message,
            source: None,
        }
    }
}

impl From<&str> for VariableError {
    fn from(message: &str) -> Self {
        message.to_owned().into()
    }
}

impl std::fmt::Display for VariableError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for VariableError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_deref().map(|source| source as _)
    }
}

pub(crate) type VariableResult<T> = Result<T, VariableError>;

#[cfg(test)]
#[path = "error/tests.rs"]
mod tests;
