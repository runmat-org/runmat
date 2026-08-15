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
    #[error("native execution host failed: {0}")]
    Host(String),
    #[error("native execution was cancelled")]
    Cancelled,
    #[error("native execution returned unsupported exit kind {0}")]
    UnsupportedExit(u32),
    #[error("native IR site is not executable by the generic host: {0}")]
    UnsupportedSite(String),
    #[error("native execution is unavailable on this platform")]
    Unavailable,
}

pub type NativeExecutorResult<T> = Result<T, NativeExecutorError>;

impl From<runmat_runtime::RuntimeError> for NativeExecutorError {
    fn from(error: runmat_runtime::RuntimeError) -> Self {
        Self::Runtime(Box::new(error))
    }
}
