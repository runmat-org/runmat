//! Pure filter generation errors; no runtime/provider ownership crosses here.
use runmat_builtins::BuiltinErrorDescriptor;

#[derive(Debug)]
pub enum FspecialKernelError {
    InvalidInput(String),
    Internal(String),
}
impl FspecialKernelError {
    pub(super) fn invalid(detail: impl AsRef<str>) -> Self {
        Self::InvalidInput(message(
            &super::FSPECIAL_ERROR_INVALID_INPUT,
            detail.as_ref(),
        ))
    }
    pub(super) fn internal(detail: impl AsRef<str>) -> Self {
        Self::Internal(message(&super::FSPECIAL_ERROR_INTERNAL, detail.as_ref()))
    }
    fn descriptor(&self) -> &'static BuiltinErrorDescriptor {
        match self {
            Self::InvalidInput(_) => &super::FSPECIAL_ERROR_INVALID_INPUT,
            Self::Internal(_) => &super::FSPECIAL_ERROR_INTERNAL,
        }
    }
}
pub(super) fn message(error: &BuiltinErrorDescriptor, detail: &str) -> String {
    if detail.starts_with("fspecial:") {
        detail.to_owned()
    } else {
        format!("{}: {}", error.message, detail)
    }
}
impl std::fmt::Display for FspecialKernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(message) | Self::Internal(message) => f.write_str(message),
        }
    }
}
impl std::error::Error for FspecialKernelError {}
impl From<FspecialKernelError> for crate::RuntimeError {
    fn from(error: FspecialKernelError) -> Self {
        let mut result = super::fspecial_descriptor_error(error.descriptor(), error.to_string());
        result.source = Some(Box::new(error));
        result
    }
}
pub type FspecialKernelResult<T> = Result<T, FspecialKernelError>;
