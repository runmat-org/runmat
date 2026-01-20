use std::error::Error as StdError;

use miette::SourceSpan;
use thiserror::Error;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ErrorContext {
    pub builtin: Option<String>,
    pub task_id: Option<String>,
    pub call_frames: Vec<CallFrame>,
    pub call_stack: Vec<String>,
    pub phase: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallFrame {
    pub function: String,
    pub source_id: Option<usize>,
    pub span: Option<(usize, usize)>,
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

    pub fn with_call_frames(mut self, call_frames: Vec<CallFrame>) -> Self {
        self.call_frames = call_frames;
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

    pub fn contains(&self, needle: &str) -> bool {
        self.message.contains(needle)
    }

    pub fn starts_with(&self, prefix: &str) -> bool {
        self.message.starts_with(prefix)
    }

    pub fn format_diagnostic(&self) -> String {
        self.format_diagnostic_with_source(None, None)
    }

    pub fn format_diagnostic_with_source(
        &self,
        source_name: Option<&str>,
        source: Option<&str>,
    ) -> String {
        let mut lines = Vec::new();
        lines.push(format!("error: {}", self.message));
        let identifier = self
            .identifier
            .as_deref()
            .or_else(|| infer_identifier(&self.message));
        if let Some(identifier) = identifier {
            lines.push(format!("id: {identifier}"));
        }
        if let Some(((source_name, source), span)) = source_name.zip(source).zip(self.span.as_ref())
        {
            let (line, col, line_text, caret) = render_span(source, span);
            lines.push(format!("--> {source_name}:{line}:{col}"));
            lines.push(format!("{line} | {line_text}"));
            lines.push(format!("  | {caret}"));
        }
        if let Some(builtin) = self.context.builtin.as_deref() {
            lines.push(format!("builtin: {builtin}"));
        }
        if let Some(task_id) = self.context.task_id.as_deref() {
            lines.push(format!("task: {task_id}"));
        }
        if let Some(phase) = self.context.phase.as_deref() {
            lines.push(format!("phase: {phase}"));
        }
        if !self.context.call_stack.is_empty() {
            lines.push("callstack:".to_string());
            for frame in &self.context.call_stack {
                lines.push(format!("  {frame}"));
            }
        } else if !self.context.call_frames.is_empty() {
            lines.push("callstack:".to_string());
            for frame in &self.context.call_frames {
                lines.push(format!("  {}", frame.function));
            }
        }
        lines.join("\n")
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

    pub fn with_call_frames(mut self, call_frames: Vec<CallFrame>) -> Self {
        self.error.context = self.error.context.with_call_frames(call_frames);
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

fn infer_identifier(message: &str) -> Option<&'static str> {
    if message.starts_with("Undefined function:") {
        Some("MATLAB:UndefinedFunction")
    } else {
        None
    }
}

fn render_span(source: &str, span: &SourceSpan) -> (usize, usize, String, String) {
    let offset = span.offset();
    let len = span.len();
    let mut line = 1;
    let mut line_start = 0;
    for (idx, ch) in source.char_indices() {
        if idx >= offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            line_start = idx + 1;
        }
    }
    let line_end = source[line_start..]
        .find('\n')
        .map(|rel| line_start + rel)
        .unwrap_or(source.len());
    let line_text = source[line_start..line_end].to_string();
    let col = offset.saturating_sub(line_start) + 1;
    let available = line_end.saturating_sub(offset).max(1);
    let caret_len = len.max(1).min(available);
    let caret = format!(
        "{}{}",
        " ".repeat(col.saturating_sub(1)),
        "^".repeat(caret_len)
    );
    (line, col, line_text, caret)
}
