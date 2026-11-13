#[derive(Debug, thiserror::Error)]
pub enum NativeExecutorError {
    #[error("native IR product is invalid: {0}")]
    InvalidProgram(#[from] runmat_native_codegen::NativeCodegenError),
    #[error("native executable is invalid: {0}")]
    Executable(String),
    #[error("native value reference is stale or invalid")]
    StaleValue,
    #[error("native execution failed: {0}")]
    Runtime(#[source] Box<runmat_runtime::RuntimeError>),
    #[error("native source admission failed: {0}")]
    SourceAdmission(#[from] runmat_runtime::context::ProgramSourceSelectionError),
    #[error("native execution host failed: {0}")]
    Host(String),
    #[error("native execution was cancelled")]
    Cancelled,
    #[error("native execution deadline elapsed")]
    Deadline,
    #[error("native execution returned unsupported exit kind {0}")]
    UnsupportedExit(u32),
    #[error("native IR site is not executable by the generic host: {0}")]
    UnsupportedSite(String),
    #[error("native call suspended for asynchronous completion")]
    CallSuspended,
    #[error("native execution is unavailable on this platform")]
    Unavailable,
}

pub type NativeExecutorResult<T> = Result<T, NativeExecutorError>;

impl From<runmat_runtime::RuntimeError> for NativeExecutorError {
    fn from(error: runmat_runtime::RuntimeError) -> Self {
        Self::Runtime(Box::new(error))
    }
}

impl From<Box<runmat_runtime::RuntimeError>> for NativeExecutorError {
    fn from(error: Box<runmat_runtime::RuntimeError>) -> Self {
        Self::Runtime(error)
    }
}

impl NativeExecutorError {
    pub(crate) fn with_secondary(self, secondary: Self) -> Self {
        match runmat_runtime::runtime_error::complete_with_cleanup::<()>(
            Err(self.into_runtime_error()),
            Err(secondary.into_runtime_error()),
        ) {
            Err(error) => error.into(),
            Ok(()) => unreachable!("two failures cannot complete successfully"),
        }
    }

    /// Canonical host mapping also used when an original native failure must
    /// retain a secondary cleanup cause in RuntimeError's existing container.
    pub fn into_runtime_error(self) -> runmat_runtime::RuntimeError {
        use runmat_runtime::runtime_error::RuntimeFailureOrigin;
        match self {
            Self::Runtime(error) => *error,
            Self::InvalidProgram(error) if error.allocation.is_some() => {
                runmat_runtime::context::host_allocation::host_allocation_runtime_error(
                    None,
                    "RunMat:NativeLivenessAdmission",
                    error.message,
                    error.allocation.expect("checked allocation cause"),
                )
            }
            error @ Self::Cancelled => {
                runmat_runtime::build_runtime_error("native execution was cancelled")
                    .with_identifier("RunMat:ExecutionCancelled")
                    .with_phase("native")
                    .with_failure_origin(RuntimeFailureOrigin::ExecutionControlCancellation)
                    .with_source(error)
                    .build()
            }
            error @ Self::Deadline => {
                runmat_runtime::build_runtime_error("native execution deadline elapsed")
                    .with_identifier("RunMat:ExecutionDeadline")
                    .with_phase("native")
                    .with_source(error)
                    .build()
            }
            Self::UnsupportedSite(message) => runmat_runtime::build_runtime_error(message)
                .with_identifier("RunMat:NativeUnsupportedSite")
                .with_phase("native")
                .build(),
            Self::SourceAdmission(error) => runmat_runtime::build_runtime_error(error.to_string())
                .with_identifier("RunMat:NativeSourceAdmission")
                .with_phase("native")
                .with_source(error)
                .build(),
            Self::Unavailable => runmat_runtime::build_runtime_error(
                "native execution is unavailable on this platform",
            )
            .with_identifier("RunMat:NativeUnavailable")
            .with_phase("native")
            .build(),
            other => runmat_runtime::build_runtime_error(other.to_string())
                .with_identifier("RunMat:NativeExecution")
                .with_phase("native")
                .with_source(other)
                .build(),
        }
    }
}

/// Native metadata uses its original runtime for both control provenance and
/// ordinary admission diagnostics, even if another session is currently entered.
pub(crate) fn allocation_error(
    runtime: Option<&runmat_runtime::context::RuntimeContext>,
    message: impl Into<String>,
    cause: runmat_runtime::context::host_allocation::HostAllocationError,
) -> runmat_runtime::RuntimeError {
    let _scope = runtime.map(runmat_runtime::context::RuntimeContext::enter);
    let namespace = runtime.map_or_else(
        || "RunMat".to_owned(),
        runmat_runtime::context::RuntimeContext::error_namespace,
    );
    runmat_runtime::context::host_allocation::host_allocation_runtime_error(
        runtime,
        &format!("{namespace}:{}", cause.runtime_identifier_suffix()),
        message,
        cause,
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn allocation_errors_keep_original_control_and_namespace() {
        use runmat_runtime::context::{host_allocation::HostAllocationError, RuntimeContext};
        use std::rc::Rc;
        let runtime = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        runtime.set_error_namespace("OriginalNative");
        let other = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        other.set_error_namespace("OtherNative");
        let _foreign = other.enter();
        let cancelled = super::allocation_error(
            Some(&runtime),
            "completion admission",
            HostAllocationError::Cancelled,
        );
        assert_eq!(cancelled.identifier(), Some("OriginalNative:Cancelled"));
        assert!(runmat_runtime::runtime_error::error_has_control_cancellation_origin(&cancelled));
        assert_eq!(
            cancelled.gpu_gather_retry(),
            runmat_runtime::GpuGatherRetry::Never
        );
        let ordinary = super::allocation_error(
            Some(&runtime),
            "completion admission",
            HostAllocationError::Overflow,
        );
        assert_eq!(
            ordinary.identifier(),
            Some(
                format!(
                    "OriginalNative:{}",
                    HostAllocationError::Overflow.runtime_identifier_suffix()
                )
                .as_str()
            )
        );
        assert!(!runmat_runtime::runtime_error::error_has_control_cancellation_origin(&ordinary));
        assert_eq!(
            runmat_runtime::context::legacy::active()
                .unwrap()
                .local_identity(),
            other.local_identity()
        );
    }
}
