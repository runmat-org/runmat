pub use runmat_async::{
    runtime_error as build_runtime_error, CallFrame, ErrorContext, GpuGatherRetry, RuntimeError,
    RuntimeErrorBuilder,
};

/// Construct a language-semantic runtime error in the active session's
/// namespace. Executors may attach frame and source-span information, but the
/// identifier and message are executor-neutral runtime semantics.
pub fn semantic_error(identifier: &str, message: impl Into<String>) -> RuntimeError {
    let suffix = identifier
        .split_once(':')
        .map_or(identifier, |(_, suffix)| suffix);
    let namespace =
        crate::context::legacy::error_namespace().unwrap_or_else(|| "RunMat".to_string());
    build_runtime_error(message)
        .with_identifier(format!("{namespace}:{suffix}"))
        .build()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayErrorKind {
    UnsupportedSchema,
    PayloadTooLarge,
    DecodeFailed,
    ExportRejected,
    ImportRejected,
}

impl ReplayErrorKind {
    pub fn identifier(self) -> &'static str {
        match self {
            Self::UnsupportedSchema => "RunMat:ReplayUnsupportedSchema",
            Self::PayloadTooLarge => "RunMat:ReplayPayloadTooLarge",
            Self::DecodeFailed => "RunMat:ReplayDecodeFailed",
            Self::ExportRejected => "RunMat:ReplayExportRejected",
            Self::ImportRejected => "RunMat:ReplayImportRejected",
        }
    }
}

pub fn replay_error(kind: ReplayErrorKind, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("replay")
        .with_identifier(kind.identifier())
        .build()
}

pub fn replay_error_with_source(
    kind: ReplayErrorKind,
    message: impl Into<String>,
    source: impl std::error::Error + Send + Sync + 'static,
) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("replay")
        .with_identifier(kind.identifier())
        .with_source(source)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{RuntimeContext, RuntimeContextGuard};
    use crate::execution::RuntimeExecutionService;
    use std::rc::Rc;

    #[test]
    fn semantic_errors_use_the_active_session_namespace() {
        let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        context.set_error_namespace("Acme");
        let _scope = RuntimeContextGuard::enter(context);

        assert_eq!(
            semantic_error("MATLAB:badsubscript", "bad index").identifier(),
            Some("Acme:badsubscript")
        );
        assert_eq!(
            semantic_error("IndexOutOfBounds", "bad index").identifier(),
            Some("Acme:IndexOutOfBounds")
        );
    }
}
