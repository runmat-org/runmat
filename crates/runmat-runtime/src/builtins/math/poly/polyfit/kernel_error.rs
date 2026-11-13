//! Pure least-squares diagnostics, without a runtime execution error carrier.
#[derive(Debug)]
pub enum PolyfitKernelError {
    InvalidInput(String),
    InvalidArgument(String),
}
impl PolyfitKernelError {
    pub(super) fn input(message: impl Into<String>) -> Self {
        Self::InvalidInput(message.into())
    }
    pub(super) fn argument(message: impl Into<String>) -> Self {
        Self::InvalidArgument(message.into())
    }
    fn language_error(&self) -> crate::RuntimeError {
        let descriptor = match self {
            Self::InvalidInput(_) => &super::POLYFIT_ERROR_INVALID_INPUT,
            Self::InvalidArgument(_) => &super::POLYFIT_ERROR_INVALID_ARGUMENT,
        };
        super::polyfit_error_with(self.to_string(), descriptor)
    }
}
impl std::fmt::Display for PolyfitKernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(message) | Self::InvalidArgument(message) => f.write_str(message),
        }
    }
}
impl std::error::Error for PolyfitKernelError {}
impl From<PolyfitKernelError> for crate::RuntimeError {
    fn from(error: PolyfitKernelError) -> Self {
        let mut result = error.language_error();
        result.source = Some(Box::new(error));
        result
    }
}
pub type PolyfitKernelResult<T> = Result<T, PolyfitKernelError>;

/// Restore the operation-owned language diagnostic without discarding the
/// provider carrier or any intermediate source in its chain.
pub(super) fn with_provider_source(
    mut fallback: crate::RuntimeError,
    source: runmat_accelerate_api::ProviderExecutionError,
) -> crate::RuntimeError {
    let mut cursor: &(dyn std::error::Error + 'static) = &source;
    loop {
        if let Some(kernel) = cursor.downcast_ref::<PolyfitKernelError>() {
            fallback = kernel.language_error();
            break;
        }
        let Some(next) = cursor.source() else { break };
        cursor = next;
    }
    crate::runtime_error::with_provider_execution_source(fallback, source)
}
