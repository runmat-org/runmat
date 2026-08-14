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

/// Materialize the language-level exception value caught by `try`/`catch`.
///
/// Executors own control transfer and frame/source decoration; Runtime owns
/// the stable conversion from its semantic error contract to `MException`.
pub fn exception_from_error(error: &RuntimeError) -> runmat_value::MException {
    if let Some(identifier) = error.identifier() {
        return runmat_value::MException::new(identifier.to_string(), error.message().to_string());
    }
    let message = error.message();
    if let Some(index) = message.rfind(": ") {
        let (identifier, detail) = message.split_at(index);
        return runmat_value::MException::new(
            exception_identifier(identifier),
            detail.trim_start_matches(':').trim().to_string(),
        );
    }
    if let Some(index) = message.rfind(':') {
        let (identifier, detail) = message.split_at(index);
        return runmat_value::MException::new(
            exception_identifier(identifier),
            detail.trim_start_matches(':').trim().to_string(),
        );
    }
    runmat_value::MException::new(exception_identifier(""), message.to_string())
}

fn exception_identifier(identifier: &str) -> String {
    if identifier.trim().is_empty() {
        let namespace =
            crate::context::legacy::error_namespace().unwrap_or_else(|| "RunMat".to_string());
        format!("{namespace}:error")
    } else {
        identifier.trim().to_string()
    }
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

    #[test]
    fn caught_exception_materialization_preserves_structured_and_legacy_errors() {
        let structured = semantic_error("IndexOutOfBounds", "bad index");
        let exception = exception_from_error(&structured);
        assert_eq!(exception.identifier, "RunMat:IndexOutOfBounds");
        assert_eq!(exception.message, "bad index");

        let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        context.set_error_namespace("Acme");
        let _scope = RuntimeContextGuard::enter(context);
        let legacy = crate::build_runtime_error("legacy detail").build();
        let exception = exception_from_error(&legacy);
        assert_eq!(exception.identifier, "Acme:error");
        assert_eq!(exception.message, "legacy detail");
    }
}
