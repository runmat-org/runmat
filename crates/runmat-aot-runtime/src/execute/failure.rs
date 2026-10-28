//! Preserve admission versus execution failure through the process boundary.

#[derive(Debug)]
pub(crate) enum ProcessFailure {
    InvalidProgram(String),
    Runtime(String),
}

impl std::fmt::Display for ProcessFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidProgram(message) | Self::Runtime(message) => message.fmt(formatter),
        }
    }
}

impl std::error::Error for ProcessFailure {}

impl ProcessFailure {
    pub(crate) fn exit_code(&self) -> i32 {
        match self {
            Self::InvalidProgram(_) => crate::EXIT_INVALID_PROGRAM,
            Self::Runtime(_) => crate::EXIT_RUNTIME_FAILURE,
        }
    }
}

impl From<String> for ProcessFailure {
    fn from(message: String) -> Self {
        Self::InvalidProgram(message)
    }
}

impl From<&str> for ProcessFailure {
    fn from(message: &str) -> Self {
        Self::InvalidProgram(message.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_status_follows_failure_kind_not_diagnostic_text() {
        let diagnostic = "the same diagnostic";
        let invalid = ProcessFailure::from(diagnostic);
        let runtime = ProcessFailure::Runtime(diagnostic.into());
        assert_eq!(invalid.to_string(), runtime.to_string());
        assert_eq!(invalid.exit_code(), crate::EXIT_INVALID_PROGRAM);
        assert_eq!(runtime.exit_code(), crate::EXIT_RUNTIME_FAILURE);
    }
}
