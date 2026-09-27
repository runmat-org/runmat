//! Owned errors from pure structure kernels, independent of execution resources.
use runmat_builtins::BuiltinErrorDescriptor;

#[derive(Debug)]
pub struct StructureKernelError {
    builtin: &'static str,
    descriptor: &'static BuiltinErrorDescriptor,
    message: String,
}

impl StructureKernelError {
    pub(super) fn detail(
        builtin: &'static str,
        descriptor: &'static BuiltinErrorDescriptor,
        detail: impl AsRef<str>,
    ) -> Self {
        Self {
            builtin,
            descriptor,
            message: format!("{}: {}", descriptor.message, detail.as_ref()),
        }
    }

    pub fn identifier(&self) -> Option<&'static str> {
        self.descriptor.identifier
    }
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for StructureKernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for StructureKernelError {}

impl From<StructureKernelError> for crate::RuntimeError {
    fn from(error: StructureKernelError) -> Self {
        let mut builder =
            crate::build_runtime_error(error.message.clone()).with_builtin(error.builtin);
        if let Some(identifier) = error.descriptor.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.with_source(error).build()
    }
}

pub type StructureKernelResult<T> = Result<T, StructureKernelError>;
