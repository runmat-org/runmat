use js_sys::Error as JsError;
use js_sys::Reflect;
use miette::{SourceOffset, SourceSpan};
use runmat_core::RunError;
use runmat_runtime::{build_runtime_error, RuntimeError};
use serde::Serialize;
use wasm_bindgen::prelude::JsValue;

#[derive(Clone, Copy)]
pub(crate) enum InitErrorCode {
    InvalidOptions,
    SnapshotResolution,
    FilesystemProvider,
    SessionCreation,
    PlotCanvas,
}

impl InitErrorCode {
    fn as_str(self) -> &'static str {
        match self {
            InitErrorCode::InvalidOptions => "InvalidOptions",
            InitErrorCode::SnapshotResolution => "SnapshotResolution",
            InitErrorCode::FilesystemProvider => "FilesystemProvider",
            InitErrorCode::SessionCreation => "SessionCreation",
            InitErrorCode::PlotCanvas => "PlotCanvas",
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) enum RunMatErrorKind {
    Syntax,
    Semantic,
    Compile,
    Runtime,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RunMatErrorSpanPayload {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) line: usize,
    pub(crate) column: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RunMatErrorPayload {
    pub(crate) kind: RunMatErrorKind,
    pub(crate) message: String,
    pub(crate) identifier: Option<String>,
    pub(crate) diagnostic: String,
    pub(crate) span: Option<RunMatErrorSpanPayload>,
    pub(crate) callstack: Vec<String>,
    pub(crate) callstack_elided: usize,
}

pub(crate) fn js_error(message: &str) -> JsValue {
    JsValue::from_str(message)
}

pub(crate) fn js_value_to_string(value: JsValue) -> String {
    value.as_string().unwrap_or_else(|| format!("{value:?}"))
}

pub(crate) fn runtime_error_to_js(err: &RuntimeError) -> JsValue {
    if let Some(identifier) = err.identifier() {
        js_error(&format!("{identifier}: {}", err.message()))
    } else {
        js_error(err.message())
    }
}

pub(crate) fn run_error_to_js(err: &RunError, source: &str) -> JsValue {
    serde_wasm_bindgen::to_value(&run_error_payload(err, source))
        .unwrap_or_else(|_| JsValue::from_str("RunMat error"))
}

pub(crate) fn init_error(code: InitErrorCode, message: impl Into<String>) -> JsValue {
    init_error_with_details(code, message, None)
}

pub(crate) fn init_error_with_details(
    code: InitErrorCode,
    message: impl Into<String>,
    details: Option<JsValue>,
) -> JsValue {
    #[cfg(target_arch = "wasm32")]
    {
        let msg = message.into();
        let error = JsError::new(&msg);
        let _ = Reflect::set(
            error.as_ref(),
            &JsValue::from_str("code"),
            &JsValue::from_str(code.as_str()),
        );
        if let Some(detail) = details {
            let _ = Reflect::set(error.as_ref(), &JsValue::from_str("details"), &detail);
        }
        error.into()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let msg = message.into();
        JsValue::from_str(&format!("{}: {msg}", code.as_str()))
    }
}

fn line_col_from_offset(source: &str, offset: usize) -> (usize, usize) {
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
    let col = offset.saturating_sub(line_start) + 1;
    (line, col)
}

fn span_payload_from_source(source: &str, start: usize, end: usize) -> RunMatErrorSpanPayload {
    let (line, column) = line_col_from_offset(source, start);
    RunMatErrorSpanPayload {
        start,
        end,
        line,
        column,
    }
}

fn format_run_error(err: &RunError, source: &str) -> String {
    match err {
        RunError::Syntax(err) => {
            let mut message = err.message.clone();
            if let Some(expected) = &err.expected {
                message = format!("{message} (expected {expected})");
            }
            if let Some(found) = &err.found_token {
                message = format!("{message} (found '{found}')");
            }
            let span = SourceSpan::new(SourceOffset::from(err.position), 1);
            build_runtime_error(message)
                .with_identifier("RunMat:SyntaxError")
                .with_span(span)
                .build()
                .format_diagnostic_with_source(Some("<wasm>"), Some(source))
        }
        RunError::Semantic(err) => {
            let span = err.span.map(|span| {
                SourceSpan::new(
                    SourceOffset::from(span.start),
                    span.end.saturating_sub(span.start).max(1),
                )
            });
            let mut builder = build_runtime_error(err.message.clone());
            if let Some(identifier) = err.identifier.as_deref() {
                builder = builder.with_identifier(identifier);
            }
            if let Some(span) = span {
                builder = builder.with_span(span);
            }
            builder
                .build()
                .format_diagnostic_with_source(Some("<wasm>"), Some(source))
        }
        RunError::Compile(err) => {
            let span = err.span.map(|span| {
                SourceSpan::new(
                    SourceOffset::from(span.start),
                    span.end.saturating_sub(span.start).max(1),
                )
            });
            let mut builder = build_runtime_error(err.message.clone());
            if let Some(identifier) = err.identifier.as_deref() {
                builder = builder.with_identifier(identifier);
            }
            if let Some(span) = span {
                builder = builder.with_span(span);
            }
            builder
                .build()
                .format_diagnostic_with_source(Some("<wasm>"), Some(source))
        }
        RunError::Runtime(err) => err.format_diagnostic_with_source(Some("<wasm>"), Some(source)),
    }
}

pub(crate) fn runtime_error_payload(
    err: &RuntimeError,
    source: Option<&str>,
) -> RunMatErrorPayload {
    let identifier = err.identifier().map(|id| id.to_string());
    let span = match (source, err.span.as_ref()) {
        (Some(source), Some(span)) => {
            let start = span.offset();
            let end = start + span.len();
            Some(span_payload_from_source(source, start, end))
        }
        _ => None,
    };
    let diagnostic = match source {
        Some(source) => err.format_diagnostic_with_source(Some("<wasm>"), Some(source)),
        None => err.format_diagnostic(),
    };
    let callstack = if !err.context.call_stack.is_empty() {
        err.context.call_stack.clone()
    } else {
        err.context
            .call_frames
            .iter()
            .map(|frame| frame.function.clone())
            .collect()
    };
    RunMatErrorPayload {
        kind: RunMatErrorKind::Runtime,
        message: err.message().to_string(),
        identifier,
        diagnostic,
        span,
        callstack,
        callstack_elided: err.context.call_frames_elided,
    }
}

pub(crate) fn run_error_payload(err: &RunError, source: &str) -> RunMatErrorPayload {
    let diagnostic = format_run_error(err, source);
    match err {
        RunError::Syntax(err) => {
            let mut message = err.message.clone();
            if let Some(expected) = &err.expected {
                message = format!("{message} (expected {expected})");
            }
            if let Some(found) = &err.found_token {
                message = format!("{message} (found '{found}')");
            }
            let span = span_payload_from_source(source, err.position, err.position + 1);
            RunMatErrorPayload {
                kind: RunMatErrorKind::Syntax,
                message,
                identifier: Some("RunMat:SyntaxError".to_string()),
                diagnostic,
                span: Some(span),
                callstack: Vec::new(),
                callstack_elided: 0,
            }
        }
        RunError::Semantic(err) => RunMatErrorPayload {
            kind: RunMatErrorKind::Semantic,
            message: err.message.clone(),
            identifier: err.identifier.clone(),
            diagnostic,
            span: err.span.map(|span| {
                let end = span.end.max(span.start + 1);
                span_payload_from_source(source, span.start, end)
            }),
            callstack: Vec::new(),
            callstack_elided: 0,
        },
        RunError::Compile(err) => RunMatErrorPayload {
            kind: RunMatErrorKind::Compile,
            message: err.message.clone(),
            identifier: err.identifier.clone(),
            diagnostic,
            span: err.span.map(|span| {
                let end = span.end.max(span.start + 1);
                span_payload_from_source(source, span.start, end)
            }),
            callstack: Vec::new(),
            callstack_elided: 0,
        },
        RunError::Runtime(err) => runtime_error_payload(err, Some(source)),
    }
}
