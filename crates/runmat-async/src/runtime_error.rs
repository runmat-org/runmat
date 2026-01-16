use std::error::Error as StdError;

use miette::SourceSpan;
use thiserror::Error;

use crate::RuntimeControlFlow;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ErrorContext {
    pub builtin: Option<String>,
    pub task_id: Option<String>,
    pub call_stack: Vec<String>,
    pub phase: Option<String>,
}

impl ErrorContext {
    pub fn with_builtin(mut self, builtin: impl Into<String>) -> Self {
        self.builtin = Some(builtin.into());
        self
    }

    pub fn with_task_id(mut self, task_id: impl Into<String>) -> Self {
        self.task_id = Some(task_id.into());
        self
    }

    pub fn with_call_stack(mut self, call_stack: Vec<String>) -> Self {
        self.call_stack = call_stack;
        self
    }

    pub fn with_phase(mut self, phase: impl Into<String>) -> Self {
        self.phase = Some(phase.into());
        self
    }
}

#[derive(Debug, Error, miette::Diagnostic)]
#[error("{message}")]
#[diagnostic(code(runmat::runtime::error))]
pub struct RuntimeError {
    pub message: String,
    #[label]
    pub span: Option<SourceSpan>,
    #[source]
    pub source: Option<Box<dyn StdError + Send + Sync>>,
    pub identifier: Option<String>,
    pub context: ErrorContext,
}

impl RuntimeError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            span: None,
            source: None,
            identifier: None,
            context: ErrorContext::default(),
        }
    }

    pub fn identifier(&self) -> Option<&str> {
        self.identifier.as_deref()
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl From<String> for RuntimeError {
    fn from(value: String) -> Self {
        RuntimeError::new(value)
    }
}

impl From<&str> for RuntimeError {
    fn from(value: &str) -> Self {
        RuntimeError::new(value)
    }
}

impl From<RuntimeError> for RuntimeControlFlow<RuntimeError> {
    fn from(value: RuntimeError) -> Self {
        RuntimeControlFlow::Error(value)
    }
}

impl From<String> for RuntimeControlFlow<RuntimeError> {
    fn from(value: String) -> Self {
        RuntimeControlFlow::Error(RuntimeError::from(value))
    }
}

impl From<&str> for RuntimeControlFlow<RuntimeError> {
    fn from(value: &str) -> Self {
        RuntimeControlFlow::Error(RuntimeError::from(value))
    }
}

pub struct RuntimeErrorBuilder {
    error: RuntimeError,
}

impl RuntimeErrorBuilder {
    pub fn with_identifier(mut self, identifier: impl Into<String>) -> Self {
        self.error.identifier = Some(identifier.into());
        self
    }

    pub fn with_builtin(mut self, builtin: impl Into<String>) -> Self {
        self.error.context = self.error.context.with_builtin(builtin);
        self
    }

    pub fn with_task_id(mut self, task_id: impl Into<String>) -> Self {
        self.error.context = self.error.context.with_task_id(task_id);
        self
    }

    pub fn with_call_stack(mut self, call_stack: Vec<String>) -> Self {
        self.error.context = self.error.context.with_call_stack(call_stack);
        self
    }

    pub fn with_phase(mut self, phase: impl Into<String>) -> Self {
        self.error.context = self.error.context.with_phase(phase);
        self
    }

    pub fn with_span(mut self, span: SourceSpan) -> Self {
        self.error.span = Some(span);
        self
    }

    pub fn with_source(mut self, source: impl StdError + Send + Sync + 'static) -> Self {
        self.error.source = Some(Box::new(source));
        self
    }

    pub fn build(self) -> RuntimeError {
        self.error
    }
}

pub fn runtime_error(message: impl Into<String>) -> RuntimeErrorBuilder {
    RuntimeErrorBuilder {
        error: RuntimeError::new(message),
    }
}
