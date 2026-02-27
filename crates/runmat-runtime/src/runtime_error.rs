pub use runmat_async::{
    runtime_error as build_runtime_error, CallFrame, ErrorContext, RuntimeError,
    RuntimeErrorBuilder,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayErrorKind {
    UnsupportedSchema,
    PayloadTooLarge,
    DecodeFailed,
    ImportRejected,
}

impl ReplayErrorKind {
    pub fn identifier(self) -> &'static str {
        match self {
            Self::UnsupportedSchema => "RunMat:ReplayUnsupportedSchema",
            Self::PayloadTooLarge => "RunMat:ReplayPayloadTooLarge",
            Self::DecodeFailed => "RunMat:ReplayDecodeFailed",
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
