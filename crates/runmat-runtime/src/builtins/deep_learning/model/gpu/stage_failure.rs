//! Failure provenance at an awaited resident layer boundary.
//!
//! Cancellation is identified where the caller's cancellation check runs;
//! provider and validation failures remain operational even if cancellation
//! arrives later. No error text or ambient state is used to reclassify them.

pub(in crate::builtins::deep_learning) enum StageExecutionFailure {
    Abandoned(crate::RuntimeError),
    Operational(crate::RuntimeError),
}

impl StageExecutionFailure {
    pub(in crate::builtins::deep_learning) fn into_error(self) -> crate::RuntimeError {
        match self {
            Self::Abandoned(error) | Self::Operational(error) => error,
        }
    }
}

impl From<crate::RuntimeError> for StageExecutionFailure {
    fn from(error: crate::RuntimeError) -> Self {
        Self::Operational(error)
    }
}
