use crate::Span;
use std::sync::{OnceLock, RwLock};

const DEFAULT_ERROR_NAMESPACE: &str = "RunMat";
static ERROR_NAMESPACE: OnceLock<RwLock<String>> = OnceLock::new();

fn error_namespace_store() -> &'static RwLock<String> {
    ERROR_NAMESPACE.get_or_init(|| RwLock::new(DEFAULT_ERROR_NAMESPACE.to_string()))
}

pub fn set_error_namespace(namespace: &str) {
    let namespace = if namespace.trim().is_empty() {
        DEFAULT_ERROR_NAMESPACE.to_string()
    } else {
        namespace.to_string()
    };
    if let Ok(mut guard) = error_namespace_store().write() {
        *guard = namespace;
    }
}

pub(crate) fn error_namespace() -> String {
    error_namespace_store()
        .read()
        .map(|guard| guard.clone())
        .unwrap_or_else(|_| DEFAULT_ERROR_NAMESPACE.to_string())
}

#[derive(Debug, Clone)]
pub struct SemanticError {
    pub message: String,
    pub span: Option<Span>,
    pub identifier: Option<String>,
}

impl SemanticError {
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

impl std::fmt::Display for SemanticError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for SemanticError {}

impl From<String> for SemanticError {
    fn from(value: String) -> Self {
        SemanticError::new(value)
    }
}
