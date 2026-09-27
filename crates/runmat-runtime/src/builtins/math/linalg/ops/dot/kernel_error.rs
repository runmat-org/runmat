//! Pure dot-product failures, independent of runtime-local execution sources.
#[derive(Debug)]
pub enum DotKernelError {
    InvalidInput,
    Internal(String),
}
impl DotKernelError {
    pub(super) fn invalid_input() -> Self {
        Self::InvalidInput
    }
    pub(super) fn internal(message: impl Into<String>) -> Self {
        Self::Internal(message.into())
    }
    fn language_error(&self) -> crate::RuntimeError {
        match self {
            Self::InvalidInput => super::dot_error(&super::DOT_ERROR_INVALID_INPUT),
            Self::Internal(message) => super::dot_internal_error(message),
        }
    }
}
impl std::fmt::Display for DotKernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput => f.write_str(super::DOT_ERROR_INVALID_INPUT.message),
            Self::Internal(message) => f.write_str(message),
        }
    }
}
impl std::error::Error for DotKernelError {}
impl From<DotKernelError> for crate::RuntimeError {
    fn from(error: DotKernelError) -> Self {
        let mut result = error.language_error();
        result.source = Some(Box::new(error));
        result
    }
}
pub type DotKernelResult<T> = Result<T, DotKernelError>;

pub(super) fn with_provider_source(
    mut fallback: crate::RuntimeError,
    source: runmat_accelerate_api::ProviderExecutionError,
) -> crate::RuntimeError {
    let mut cursor: &(dyn std::error::Error + 'static) = &source;
    loop {
        if let Some(kernel) = cursor.downcast_ref::<DotKernelError>() {
            fallback = kernel.language_error();
            break;
        }
        let Some(next) = cursor.source() else { break };
        cursor = next;
    }
    crate::runtime_error::with_provider_execution_source(fallback, source)
}
