use miette::{SourceOffset, SourceSpan};
use runmat_hir::{SemanticError, Span};
use runmat_runtime::{build_runtime_error, RuntimeError};

#[derive(Debug, Clone)]
pub struct CompileError {
    pub message: String,
    pub span: Option<Span>,
    pub identifier: Option<String>,
}

impl CompileError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            span: None,
            identifier: None,
        }
    }

    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    pub fn with_identifier(mut self, identifier: impl Into<String>) -> Self {
        self.identifier = Some(identifier.into());
        self
    }
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for CompileError {}

impl From<String> for CompileError {
    fn from(value: String) -> Self {
        CompileError::new(value)
    }
}

impl From<&str> for CompileError {
    fn from(value: &str) -> Self {
        CompileError::new(value)
    }
}

impl From<SemanticError> for CompileError {
    fn from(value: SemanticError) -> Self {
        let mut err = CompileError::new(value.message);
        if let Some(span) = value.span {
            err = err.with_span(span);
        }
        if let Some(identifier) = value.identifier {
            err = err.with_identifier(identifier);
        }
        err
    }
}

impl From<CompileError> for RuntimeError {
    fn from(value: CompileError) -> Self {
        let mut builder = build_runtime_error(value.message);
        if let Some(identifier) = value.identifier {
            builder = builder.with_identifier(identifier);
        }
        if let Some(span) = value.span {
            let len = span.end.saturating_sub(span.start).max(1);
            builder = builder.with_span(SourceSpan::new(SourceOffset::from(span.start), len));
        }
        builder.build()
    }
}
