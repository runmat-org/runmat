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
