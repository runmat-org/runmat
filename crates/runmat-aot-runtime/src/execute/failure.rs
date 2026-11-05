//! Preserve admission versus execution failure through the process boundary.

#[derive(Debug)]
pub(crate) enum ProcessFailure {
    InvalidProgram(String),
    Runtime(String),
    RuntimeCause(runmat_runtime::RuntimeError),
}

impl std::fmt::Display for ProcessFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidProgram(message) | Self::Runtime(message) => message.fmt(formatter),
            Self::RuntimeCause(cause) => cause.fmt(formatter),
        }
    }
}

impl std::error::Error for ProcessFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::RuntimeCause(cause) => Some(cause),
            Self::InvalidProgram(_) | Self::Runtime(_) => None,
        }
    }
}

impl ProcessFailure {
    pub(crate) fn exit_code(&self) -> i32 {
        match self {
            Self::InvalidProgram(_) => crate::EXIT_INVALID_PROGRAM,
            Self::Runtime(_) | Self::RuntimeCause(_) => crate::EXIT_RUNTIME_FAILURE,
        }
    }
}

impl From<runmat_runtime::RuntimeError> for ProcessFailure {
    fn from(cause: runmat_runtime::RuntimeError) -> Self {
        Self::RuntimeCause(cause)
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

        let cause = runmat_runtime::context::host_allocation::host_allocation_runtime_error(
            None,
            "RunMat:NativeLivenessAdmission",
            "native liveness metadata admission failed",
            runmat_runtime::context::host_allocation::HostAllocationError::Overflow,
        );
        let typed = ProcessFailure::from(cause);
        assert_eq!(typed.exit_code(), crate::EXIT_RUNTIME_FAILURE);
        let ProcessFailure::RuntimeCause(original) = &typed else {
            panic!("runtime admission must retain its original typed cause");
        };
        assert_eq!(
            original.identifier(),
            Some("RunMat:NativeLivenessAdmission")
        );
        assert!(std::error::Error::source(&typed).is_some());
        assert!(matches!(
            std::error::Error::source(original).and_then(|source| {
                source
                    .downcast_ref::<runmat_runtime::context::host_allocation::HostAllocationError>()
            }),
            Some(runmat_runtime::context::host_allocation::HostAllocationError::Overflow)
        ));
    }
}
