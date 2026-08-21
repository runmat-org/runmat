#[derive(Debug, thiserror::Error)]
pub enum JitError {
    #[error("native code generation failed: {0}")]
    Codegen(#[from] runmat_native_codegen::NativeCodegenError),
    #[error("Cranelift module operation failed: {0}")]
    Module(String),
    #[error(transparent)]
    Executor(#[from] runmat_native_executor::NativeExecutorError),
}

pub type JitResult<T> = Result<T, JitError>;
