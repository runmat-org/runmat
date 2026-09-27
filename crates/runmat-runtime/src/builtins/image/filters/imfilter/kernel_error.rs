//! Diagnostics from the shared pure filtering plan/evaluator.
#[derive(Debug)]
pub struct ImfilterKernelError {
    builtin: String,
    message: String,
}
impl ImfilterKernelError {
    pub(super) fn new(builtin: &str, message: impl Into<String>) -> Self {
        Self {
            builtin: builtin.to_owned(),
            message: message.into(),
        }
    }
}
impl std::fmt::Display for ImfilterKernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for ImfilterKernelError {}
impl From<ImfilterKernelError> for crate::RuntimeError {
    fn from(error: ImfilterKernelError) -> Self {
        let mut result = super::filter_error(&error.builtin, error.message.clone());
        result.source = Some(Box::new(error));
        result
    }
}
pub type ImfilterKernelResult<T> = Result<T, ImfilterKernelError>;

pub(super) fn with_provider_source(
    mut fallback: crate::RuntimeError,
    source: runmat_accelerate_api::ProviderExecutionError,
) -> crate::RuntimeError {
    let mut cursor: &(dyn std::error::Error + 'static) = &source;
    loop {
        if let Some(kernel) = cursor.downcast_ref::<ImfilterKernelError>() {
            // Match the host adapter's descriptor while retaining the entire
            // provider cause below, including the original pure diagnostic.
            fallback = super::imfilter_error_with_detail(
                &super::IMFILTER_ERROR_INVALID_INPUT,
                &kernel.message,
            );
            break;
        }
        let Some(next) = cursor.source() else { break };
        cursor = next;
    }
    crate::runtime_error::with_provider_execution_source(fallback, source)
}
