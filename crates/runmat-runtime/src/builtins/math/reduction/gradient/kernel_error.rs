//! Errors produced by the pure finite-difference evaluator.
#[derive(Debug)]
pub struct GradientKernelError {
    detail: String,
}
impl GradientKernelError {
    pub(super) fn new(detail: impl Into<String>) -> Self {
        Self {
            detail: detail.into(),
        }
    }
    fn language_error(&self) -> crate::RuntimeError {
        super::gradient_internal_error(&self.detail)
    }
}
impl std::fmt::Display for GradientKernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: {}",
            super::GRADIENT_ERROR_INTERNAL.message,
            self.detail
        )
    }
}
impl std::error::Error for GradientKernelError {}
impl From<GradientKernelError> for crate::RuntimeError {
    fn from(error: GradientKernelError) -> Self {
        let mut result = error.language_error();
        result.source = Some(Box::new(error));
        result
    }
}
pub type GradientKernelResult<T> = Result<T, GradientKernelError>;
pub(super) fn with_provider_source(
    mut fallback: crate::RuntimeError,
    source: runmat_accelerate_api::ProviderExecutionError,
) -> crate::RuntimeError {
    let mut cursor: &(dyn std::error::Error + 'static) = &source;
    loop {
        if let Some(kernel) = cursor.downcast_ref::<GradientKernelError>() {
            fallback = kernel.language_error();
            break;
        }
        let Some(next) = cursor.source() else { break };
        cursor = next;
    }
    crate::runtime_error::with_provider_execution_source(fallback, source)
}
