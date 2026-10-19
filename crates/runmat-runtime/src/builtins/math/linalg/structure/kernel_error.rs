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

    fn language_error(&self) -> crate::RuntimeError {
        let mut builder =
            crate::build_runtime_error(self.message.clone()).with_builtin(self.builtin);
        if let Some(identifier) = self.descriptor.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
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
        let mut result = error.language_error();
        result.source = Some(Box::new(error));
        result
    }
}

pub type StructureKernelResult<T> = Result<T, StructureKernelError>;

/// Recover this domain's language diagnostic after a provider round trip,
/// retaining the entire owning provider cause, not only the discovered leaf.
pub(super) fn with_provider_source(
    mut fallback: crate::RuntimeError,
    source: runmat_accelerate_api::ProviderExecutionError,
) -> crate::RuntimeError {
    let mut cursor: &(dyn std::error::Error + 'static) = &source;
    loop {
        if let Some(kernel) = cursor.downcast_ref::<StructureKernelError>() {
            fallback = kernel.language_error();
            break;
        }
        let Some(next) = cursor.source() else { break };
        cursor = next;
    }
    crate::runtime_error::with_provider_execution_source(fallback, source)
}
