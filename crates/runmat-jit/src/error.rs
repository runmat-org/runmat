#[derive(Debug, thiserror::Error)]
pub enum JitError {
    #[error("native code generation failed: {0}")]
    Codegen(#[from] runmat_native_codegen::NativeCodegenError),
    #[error("Cranelift module operation failed: {0}")]
    Module(String),
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

pub type JitResult<T> = Result<T, JitError>;

impl From<runmat_runtime::RuntimeError> for JitError {
    fn from(error: runmat_runtime::RuntimeError) -> Self {
        Self::Runtime(Box::new(error))
    }
}
