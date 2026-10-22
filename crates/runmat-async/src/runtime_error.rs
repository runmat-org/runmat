use std::error::Error as StdError;
use std::sync::Arc;

use miette::SourceSpan;
use thiserror::Error;

#[path = "runtime_error/source_bound.rs"]
mod source_bound;
pub use source_bound::RuntimeErrorSourceBound;

/// Native errors retain the thread-safe source required by worker transport.
/// Browser sources may own thread-local resources and stay on that execution
/// lane; formatting a diagnostic does not replace or release the source.
#[cfg(not(target_arch = "wasm32"))]
pub type RuntimeErrorSource = dyn StdError + Send + Sync + 'static;
#[cfg(target_arch = "wasm32")]
pub type RuntimeErrorSource = dyn StdError + 'static;

#[path = "runtime_error/owner.rs"]
mod owner;
pub use owner::RuntimeErrorOwner;

#[path = "runtime_error/diagnostic.rs"]
mod diagnostic;
pub use diagnostic::{
    DiagnosticConstructionError, DiagnosticCopyProgress, RuntimeDiagnosticConstruction,
    RuntimeDiagnosticField, RuntimeDiagnosticLayout,
};

#[path = "runtime_error/diagnostic_text.rs"]
mod diagnostic_text;
pub use diagnostic_text::{DiagnosticText, DiagnosticTextCopy};

#[path = "runtime_error/allocation.rs"]
mod allocation;
pub use allocation::{
    DiagnosticAllocationLease, ManagedDiagnosticAllocation, ManagedDiagnosticAllocationLease,
    RetainedDiagnosticAllocation,
};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ErrorContext {
    pub builtin: Option<String>,
    pub task_id: Option<String>,
    pub call_frames: Vec<CallFrame>,
    pub call_frames_elided: usize,
    pub call_stack: Vec<String>,
    pub phase: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallFrame {
    pub function: String,
    pub source_id: Option<usize>,
    pub span: Option<(usize, usize)>,
    /// Local issuance proof only. Portable failures carry an admitted source
    /// ordinal instead; no process token enters the worker wire format.
    pub source_owner: Option<CallFrameSourceOwner>,
    /// Shared presentation text captured at the exact frame producer. It
    /// remains useful after the executable definition retires, but cannot
    /// authenticate a callable or portable source ordinal.
    pub diagnostic_source: Option<CallFrameDiagnosticSource>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CallFrameDiagnosticSource {
    Local {
        display_name: DiagnosticText,
        text: DiagnosticText,
    },
    Detached {
        display_name: DiagnosticText,
        line: u64,
        column: u64,
    },
    DetachedUnknown {
        display_name: DiagnosticText,
    },
}

impl CallFrameDiagnosticSource {
    pub fn new(display_name: Arc<str>, text: Arc<str>) -> Self {
        Self::Local {
            display_name: display_name.into(),
            text: text.into(),
        }
    }

    pub fn detached(display_name: Arc<str>, line: u64, column: u64) -> Self {
        Self::detached_from_text_owner(display_name.into(), line, column)
    }

    pub fn detached_from_text_owner(display_name: DiagnosticText, line: u64, column: u64) -> Self {
        Self::Detached {
            display_name,
            line,
            column,
        }
    }

    pub fn detached_unknown(display_name: Arc<str>) -> Self {
        Self::detached_unknown_from_text_owner(display_name.into())
    }

    pub fn detached_unknown_from_text_owner(display_name: DiagnosticText) -> Self {
        Self::DetachedUnknown { display_name }
    }

    pub fn display_name(&self) -> &str {
        match self {
            Self::Local { display_name, .. }
            | Self::Detached { display_name, .. }
            | Self::DetachedUnknown { display_name } => display_name,
        }
    }

    pub fn is_detached(&self) -> bool {
        matches!(self, Self::Detached { .. } | Self::DetachedUnknown { .. })
    }

    /// Use the same byte-offset and one-based coordinate convention as the
    /// existing source call-stack renderer, including its end clamp.
    pub fn line_column(&self, span: Option<(usize, usize)>) -> Option<(u64, u64)> {
        match self {
            Self::Local { text, .. } => Self::line_column_from_text(text, span),
            Self::Detached { line, column, .. } => Some((*line, *column)),
            Self::DetachedUnknown { .. } => None,
        }
    }

    /// Shared byte-offset coordinate rule for retained and still-live text.
    pub fn line_column_from_text(text: &str, span: Option<(usize, usize)>) -> Option<(u64, u64)> {
        let (offset, _) = span?;
        let prefix = &text.as_bytes()[..offset.min(text.len())];
        let line = prefix.iter().filter(|byte| **byte == b'\n').count() + 1;
        let column = prefix
            .iter()
            .rev()
            .take_while(|byte| **byte != b'\n')
            .count()
            + 1;
        Some((u64::try_from(line).ok()?, u64::try_from(column).ok()?))
    }
}

/// One source-table issuance shared by the binding, selected body, and error
/// frame. Equality is identity of the allocation, not equality of its empty
/// payload; possession alone never authenticates a portable source ordinal.
#[derive(Clone, Debug)]
pub struct CallFrameSourceOwner(Arc<()>);

impl CallFrameSourceOwner {
    pub fn issue() -> Self {
        Self(Arc::new(()))
    }

    pub fn same_issuance(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl PartialEq for CallFrameSourceOwner {
    fn eq(&self, other: &Self) -> bool {
        self.same_issuance(other)
    }
}

impl Eq for CallFrameSourceOwner {}

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

    pub fn with_call_frames_elided(mut self, count: usize) -> Self {
        self.call_frames_elided = count;
        self
    }

    pub fn with_phase(mut self, phase: impl Into<String>) -> Self {
        self.phase = Some(phase.into());
        self
    }
}

#[derive(Debug, Error)]
#[error("{message}")]
pub struct RuntimeError {
    pub message: String,
    pub span: Option<SourceSpan>,
    #[source]
    pub source: Option<Box<RuntimeErrorSource>>,
    pub identifier: Option<String>,
    pub context: ErrorContext,
    pub gpu_gather_retry: GpuGatherRetry,
    pub failure_origin: RuntimeFailureOrigin,
    /// An owning charge for original diagnostic allocations only. Its final
    /// drop follows message, source, identifier and context physical drops.
    /// Mutating public text/context fields requires a separate authority.
    pub retained_diagnostic_allocation: Option<RetainedDiagnosticAllocation>,
}

/// An owning reference to the exact error caught by an exception value.
///
/// A rethrow may have several live references to that original error (for
/// example, a catch variable and the VM's active exception). This source is
/// not a formatted copy: traversing `source()` reaches the original
/// `RuntimeError` and its downcastable provider or host cause.
#[derive(Debug)]
pub struct SharedRuntimeErrorSource {
    original: RuntimeErrorOwner,
}

impl SharedRuntimeErrorSource {
    pub fn original(&self) -> &RuntimeError {
        &self.original
    }
}

impl std::fmt::Display for SharedRuntimeErrorSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.original)
    }
}

impl StdError for SharedRuntimeErrorSource {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(self.original.as_ref())
    }
}

/// Producer-owned failure provenance, independent of diagnostic spelling or
/// host presentation policy. Unclassified failures must not be retried based
/// on their message or identifier.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RuntimeFailureOrigin {
    #[default]
    Unclassified,
    LanguageEvaluation,
    ExecutionControlCancellation,
}

/// Whether dispatcher-level legacy GPU gather-and-retry may transform this error.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GpuGatherRetry {
    /// Preserve the historical message-based fallback while callers migrate to an explicit policy.
    #[default]
    Legacy,
    /// This is a final policy, semantic, or provider-integrity error and must be returned unchanged.
    Never,
    /// The implementation explicitly requests dispatcher-owned gather and one host retry.
    Requested,
}

impl RuntimeError {
    /// Recover the original caught error when it has one owner; otherwise
    /// retain that exact original as an owning source while copying only its
    /// language-visible metadata into the new outward error.
    pub fn from_shared_original(original: RuntimeErrorOwner) -> Self {
        match RuntimeErrorOwner::try_unwrap(original) {
            Ok(error) => error,
            Err(original) => Self {
                message: original.message.clone(),
                span: original.span,
                source: Some(Box::new(SharedRuntimeErrorSource {
                    original: original.clone(),
                })),
                identifier: original.identifier.clone(),
                context: original.context.clone(),
                gpu_gather_retry: original.gpu_gather_retry,
                failure_origin: original.failure_origin,
                // The source retains the original charge. These copied
                // presentation fields require independent allocation authority.
                retained_diagnostic_allocation: None,
            },
        }
    }

    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            span: None,
            source: None,
            identifier: None,
            context: ErrorContext::default(),
            gpu_gather_retry: GpuGatherRetry::Legacy,
            failure_origin: RuntimeFailureOrigin::Unclassified,
            retained_diagnostic_allocation: None,
        }
    }

    pub fn identifier(&self) -> Option<&str> {
        self.identifier.as_deref()
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn gpu_gather_retry(&self) -> GpuGatherRetry {
        self.gpu_gather_retry
    }

    pub fn failure_origin(&self) -> RuntimeFailureOrigin {
        self.failure_origin
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
            if self.context.call_frames_elided > 0 {
                lines.push(format!(
                    "  ... {} frames elided ...",
                    self.context.call_frames_elided
                ));
            }
            for frame in &self.context.call_frames {
                lines.push(format!("  {}", frame.function));
            }
        }
        lines.join("\n")
    }
}

impl miette::Diagnostic for RuntimeError {
    fn code<'a>(&'a self) -> Option<Box<dyn std::fmt::Display + 'a>> {
        Some(Box::new("runmat::runtime::error"))
    }

    fn labels(&self) -> Option<Box<dyn Iterator<Item = miette::LabeledSpan> + '_>> {
        self.span.map(|span| {
            Box::new(std::iter::once(miette::LabeledSpan::underline(span)))
                as Box<dyn Iterator<Item = miette::LabeledSpan>>
        })
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
    pub fn with_retained_diagnostic_allocation(
        mut self,
        allocation: RetainedDiagnosticAllocation,
    ) -> Self {
        self.error.retained_diagnostic_allocation = Some(allocation);
        self
    }

    pub fn with_failure_origin(mut self, origin: RuntimeFailureOrigin) -> Self {
        self.error.failure_origin = origin;
        self
    }

    pub fn with_identifier(mut self, identifier: impl Into<String>) -> Self {
        self.error.identifier = Some(identifier.into());
        self
    }

    pub fn with_gpu_gather_retry(mut self, policy: GpuGatherRetry) -> Self {
        self.error.gpu_gather_retry = policy;
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

    pub fn with_call_frames_elided(mut self, count: usize) -> Self {
        self.error.context = self.error.context.with_call_frames_elided(count);
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

    pub fn with_source(mut self, source: impl RuntimeErrorSourceBound) -> Self {
        self.error.source = Some(Box::new(source));
        self
    }

    pub fn build(self) -> RuntimeError {
        self.error
    }
}

#[cfg(test)]
#[path = "runtime_error/source_tests.rs"]
mod source_tests;

#[cfg(test)]
#[path = "runtime_error/origin_tests.rs"]
mod origin_tests;

pub fn runtime_error(message: impl Into<String>) -> RuntimeErrorBuilder {
    RuntimeErrorBuilder {
        error: RuntimeError::new(message),
    }
}

fn infer_identifier(message: &str) -> Option<&'static str> {
    if message.starts_with("Undefined function:") {
        Some("RunMat:UndefinedFunction")
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
