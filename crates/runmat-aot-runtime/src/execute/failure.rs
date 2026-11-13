//! Preserve admission versus execution failure through the process boundary.
use runmat_runtime::context::host_allocation::{HostAllocationError, HostControlAllocationOwner};
use runmat_runtime::context::RuntimeContext;
use runmat_value::ExactHostAllocationClaim;

#[derive(Debug)]
pub(crate) struct RuntimeFailure {
    // Fields retire in declaration order: free the diagnostic backing before
    // returning its physical allocation claim.
    cause: Box<runmat_runtime::RuntimeError>,
    _allocation: Option<ExactHostAllocationClaim>,
}

impl std::ops::Deref for RuntimeFailure {
    type Target = runmat_runtime::RuntimeError;
    fn deref(&self) -> &Self::Target {
        &self.cause
    }
}

impl std::fmt::Display for RuntimeFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.cause.fmt(formatter)
    }
}

impl std::error::Error for RuntimeFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        std::error::Error::source(self.cause.as_ref())
    }
}

pub(super) struct RuntimeFailureReservation {
    allocation: Option<ExactHostAllocationClaim>,
}

impl RuntimeFailureReservation {
    pub(super) fn prepare(runtime: &RuntimeContext) -> Result<Self, ProcessFailure> {
        // Reserve before binding can produce the error. Failure to admit a
        // diagnostic must never discard an already-produced execution cause.
        let allocation = HostControlAllocationOwner::capture(runtime)
            .and_then(|owner| {
                owner
                    .map(|owner| owner.claim(std::mem::size_of::<runmat_runtime::RuntimeError>()))
                    .transpose()
            })
            .map_err(ProcessFailure::RuntimeCauseAdmission)?;
        Ok(Self { allocation })
    }

    pub(super) fn into_failure(self, cause: runmat_runtime::RuntimeError) -> ProcessFailure {
        ProcessFailure::RuntimeCause(RuntimeFailure {
            cause: Box::new(cause),
            _allocation: self.allocation,
        })
    }
}

#[derive(Debug)]
pub(crate) enum ProcessFailure {
    InvalidProgram(String),
    Runtime(String),
    RuntimeCause(RuntimeFailure),
    RuntimeCauseAdmission(HostAllocationError),
}

impl std::fmt::Display for ProcessFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidProgram(message) | Self::Runtime(message) => message.fmt(formatter),
            Self::RuntimeCause(cause) => cause.fmt(formatter),
            Self::RuntimeCauseAdmission(cause) => {
                write!(
                    formatter,
                    "standalone diagnostic allocation admission failed: {cause}"
                )
            }
        }
    }
}

impl std::error::Error for ProcessFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::RuntimeCause(cause) => Some(cause.cause.as_ref()),
            Self::RuntimeCauseAdmission(cause) => Some(cause),
            Self::InvalidProgram(_) | Self::Runtime(_) => None,
        }
    }
}

impl ProcessFailure {
    pub(crate) fn exit_code(&self) -> i32 {
        match self {
            Self::InvalidProgram(_) => crate::EXIT_INVALID_PROGRAM,
            Self::Runtime(_) | Self::RuntimeCause(_) | Self::RuntimeCauseAdmission(_) => {
                crate::EXIT_RUNTIME_FAILURE
            }
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
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };

    #[derive(Debug)]
    struct ClaimProbe {
        source_dropped: Arc<AtomicBool>,
        released: Arc<AtomicBool>,
    }
    impl runmat_value::ExactHostAllocationReservation for ClaimProbe {}
    impl Drop for ClaimProbe {
        fn drop(&mut self) {
            assert!(self.source_dropped.load(Ordering::SeqCst));
            self.released.store(true, Ordering::SeqCst);
        }
    }

    #[derive(Debug)]
    struct SourceProbe {
        source_dropped: Arc<AtomicBool>,
        released: Arc<AtomicBool>,
    }
    impl std::fmt::Display for SourceProbe {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("retained process cause")
        }
    }
    impl std::error::Error for SourceProbe {}
    impl Drop for SourceProbe {
        fn drop(&mut self) {
            assert!(!self.released.load(Ordering::SeqCst));
            self.source_dropped.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn terminal_cause_retires_before_its_allocation_claim() {
        let source_dropped = Arc::new(AtomicBool::new(false));
        let released = Arc::new(AtomicBool::new(false));
        let reservation = RuntimeFailureReservation {
            allocation: Some(ExactHostAllocationClaim::owned(Box::new(ClaimProbe {
                source_dropped: source_dropped.clone(),
                released: released.clone(),
            }))),
        };
        let cause = runmat_runtime::build_runtime_error("process failure")
            .with_source(SourceProbe {
                source_dropped: source_dropped.clone(),
                released: released.clone(),
            })
            .build();
        let failure = reservation.into_failure(cause);
        assert!(!released.load(Ordering::SeqCst));
        drop(failure);
        assert!(source_dropped.load(Ordering::SeqCst));
        assert!(released.load(Ordering::SeqCst));
    }

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
        let runtime = RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let typed = RuntimeFailureReservation::prepare(&runtime)
            .expect("process diagnostic reservation")
            .into_failure(cause);
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
