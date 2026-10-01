//! Owner-specific structured errors for the shared reference-line service.
use super::*;

pub(super) trait ReferenceLineRuntimeErrors {
    fn invalid(self, message: impl Into<String>) -> RuntimeError;
    fn internal(self, message: impl Into<String>) -> RuntimeError;
    fn invalid_with_source(self, message: impl Into<String>, cause: RuntimeError) -> RuntimeError;
    fn internal_with_source(self, message: impl Into<String>, cause: RuntimeError) -> RuntimeError;
    fn internal_with_plot_source(
        self,
        message: impl Into<String>,
        cause: runmat_plot::plots::PlotDataExportError,
    ) -> RuntimeError;
}

impl ReferenceLineRuntimeErrors for ReferenceLineOwner {
    fn invalid(self, message: impl Into<String>) -> RuntimeError {
        self.error(self.invalid_descriptor(), message)
    }

    fn internal(self, message: impl Into<String>) -> RuntimeError {
        self.error(self.internal_descriptor(), message)
    }

    fn invalid_with_source(self, message: impl Into<String>, cause: RuntimeError) -> RuntimeError {
        self.sourced_error(self.invalid_descriptor(), message, cause)
    }

    fn internal_with_source(self, message: impl Into<String>, cause: RuntimeError) -> RuntimeError {
        self.sourced_error(self.internal_descriptor(), message, cause)
    }

    fn internal_with_plot_source(
        self,
        message: impl Into<String>,
        cause: runmat_plot::plots::PlotDataExportError,
    ) -> RuntimeError {
        let mut builder = build_runtime_error(message)
            .with_builtin(self.name())
            .with_source(cause);
        if let Some(identifier) = self.internal_descriptor().identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    }
}

trait ReferenceLineErrorBuilder {
    fn error(
        self,
        descriptor: &'static BuiltinErrorDescriptor,
        message: impl Into<String>,
    ) -> RuntimeError;
    fn sourced_error(
        self,
        descriptor: &'static BuiltinErrorDescriptor,
        message: impl Into<String>,
        cause: RuntimeError,
    ) -> RuntimeError;
}

impl ReferenceLineErrorBuilder for ReferenceLineOwner {
    fn error(
        self,
        descriptor: &'static BuiltinErrorDescriptor,
        message: impl Into<String>,
    ) -> RuntimeError {
        let mut builder = build_runtime_error(message).with_builtin(self.name());
        if let Some(identifier) = descriptor.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    }

    fn sourced_error(
        self,
        descriptor: &'static BuiltinErrorDescriptor,
        message: impl Into<String>,
        cause: RuntimeError,
    ) -> RuntimeError {
        let span = cause.span;
        let mut builder = build_runtime_error(message)
            .with_builtin(self.name())
            .with_gpu_gather_retry(cause.gpu_gather_retry())
            .with_source(cause);
        if let Some(span) = span {
            builder = builder.with_span(span);
        }
        if let Some(identifier) = descriptor.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    }
}
