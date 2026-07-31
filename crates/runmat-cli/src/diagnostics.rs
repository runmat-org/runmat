use miette::{SourceOffset, SourceSpan};
use runmat_config::runtime::{self as config, RunMatRuntimeConfig};
use runmat_core::{
    abi::{DiagnosticSeverity, RuntimeDiagnostic},
    RunError,
};
use runmat_runtime::build_runtime_error;

use crate::presentation::{self, StreamStyles, Tone};

pub fn parser_compat(mode: config::LanguageCompatMode) -> runmat_parser::CompatMode {
    match mode {
        config::LanguageCompatMode::RunMat => runmat_parser::CompatMode::RunMat,
        config::LanguageCompatMode::Matlab => runmat_parser::CompatMode::Matlab,
        config::LanguageCompatMode::Strict => runmat_parser::CompatMode::Strict,
    }
}

pub fn resolved_error_namespace(cfg: &RunMatRuntimeConfig) -> String {
    let configured = cfg.runtime.error_namespace.trim();
    if configured.is_empty() {
        config::error_namespace_for_language_compat(cfg.language.compat).to_string()
    } else {
        configured.to_string()
    }
}

pub fn format_frontend_error(err: &RunError, source_name: &str, source: &str) -> Option<String> {
    let styles = presentation::stderr();
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
            Some(format_diagnostic_with_styles(
                &message,
                Some("RunMat:SyntaxError"),
                Some(span),
                source_name,
                source,
                &styles,
            ))
        }
        RunError::Semantic(err) => {
            let span = err.span.map(|span| {
                SourceSpan::new(
                    SourceOffset::from(span.start),
                    span.end.saturating_sub(span.start).max(1),
                )
            });
            let identifier = err.identifier.as_deref().or(Some("RunMat:HirError"));
            Some(format_diagnostic_with_styles(
                &err.message,
                identifier,
                span,
                source_name,
                source,
                &styles,
            ))
        }
        RunError::Compile(err) => {
            let span = err.span.map(|span| {
                SourceSpan::new(
                    SourceOffset::from(span.start),
                    span.end.saturating_sub(span.start).max(1),
                )
            });
            let identifier = err.identifier.as_deref().or(Some("RunMat:CompileError"));
            Some(format_diagnostic_with_styles(
                &err.message,
                identifier,
                span,
                source_name,
                source,
                &styles,
            ))
        }
        RunError::Runtime(err) => Some(render_runtime_error(
            err,
            DiagnosticSeverity::Error,
            Some(source_name),
            Some(source),
            &styles,
        )),
    }
}

pub fn format_runtime_diagnostic(
    diagnostic: &RuntimeDiagnostic,
    source_name: Option<&str>,
    source: Option<&str>,
) -> String {
    let styles = presentation::stderr();
    let span = diagnostic.span.as_ref().map(|span| {
        SourceSpan::new(
            SourceOffset::from(span.start),
            span.end.saturating_sub(span.start).max(1),
        )
    });
    let mut builder = build_runtime_error(diagnostic.message.clone());
    builder = builder.with_identifier(diagnostic.code.clone());
    if let Some(span) = span {
        builder = builder.with_span(span);
    }

    let error = builder.build();
    let mut rendered =
        render_runtime_error(&error, diagnostic.severity, source_name, source, &styles);
    if !diagnostic.callstack.is_empty() {
        rendered.push_str(&format!("\n{}:", styles.label("callstack")));
        if diagnostic.callstack_elided > 0 {
            rendered.push_str(&format!(
                "\n  {}",
                styles.muted(format!(
                    "... {} frames elided ...",
                    diagnostic.callstack_elided
                ))
            ));
        }
        for frame in &diagnostic.callstack {
            rendered.push_str(&format!("\n  {}", styles.identifier(frame)));
        }
    }
    rendered
}

pub fn format_compact_runtime_diagnostic(diagnostic: &RuntimeDiagnostic) -> String {
    let styles = presentation::stderr();
    let tone = match diagnostic.severity {
        DiagnosticSeverity::Error => Tone::Error,
        DiagnosticSeverity::Warning => Tone::Warning,
        DiagnosticSeverity::Info => Tone::Info,
        DiagnosticSeverity::Hint => Tone::Help,
    };
    format!(
        "{}: {}",
        styles.paint(tone, &diagnostic.code),
        diagnostic.message
    )
}

fn diagnostic_severity_label(severity: DiagnosticSeverity) -> &'static str {
    match severity {
        DiagnosticSeverity::Error => "error",
        DiagnosticSeverity::Warning => "warning",
        DiagnosticSeverity::Info => "info",
        DiagnosticSeverity::Hint => "hint",
    }
}

pub fn format_diagnostic(
    message: &str,
    identifier: Option<&str>,
    span: Option<SourceSpan>,
    source_name: &str,
    source: &str,
) -> String {
    format_diagnostic_with_styles(
        message,
        identifier,
        span,
        source_name,
        source,
        &presentation::stderr(),
    )
}

fn format_diagnostic_with_styles(
    message: &str,
    identifier: Option<&str>,
    span: Option<SourceSpan>,
    source_name: &str,
    source: &str,
    styles: &StreamStyles,
) -> String {
    let mut builder = build_runtime_error(message);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    if let Some(span) = span {
        builder = builder.with_span(span);
    }
    render_runtime_error(
        &builder.build(),
        DiagnosticSeverity::Error,
        Some(source_name),
        Some(source),
        styles,
    )
}

fn render_runtime_error(
    error: &runmat_runtime::RuntimeError,
    severity: DiagnosticSeverity,
    source_name: Option<&str>,
    source: Option<&str>,
    styles: &StreamStyles,
) -> String {
    let severity_label = diagnostic_severity_label(severity);
    let severity_tone = match severity {
        DiagnosticSeverity::Error => Tone::Error,
        DiagnosticSeverity::Warning => Tone::Warning,
        DiagnosticSeverity::Info => Tone::Info,
        DiagnosticSeverity::Hint => Tone::Help,
    };
    let mut lines = vec![format!(
        "{}: {}",
        styles.paint(severity_tone, severity_label),
        error.message
    )];
    let identifier = error.identifier().or_else(|| {
        error
            .message
            .starts_with("Undefined function:")
            .then_some("RunMat:UndefinedFunction")
    });
    if let Some(identifier) = identifier {
        lines.push(format!(
            "{} {}",
            styles.muted("id:"),
            styles.identifier(identifier)
        ));
    }
    if let Some(((source_name, source), span)) = source_name.zip(source).zip(error.span.as_ref()) {
        let (line, column, line_text, caret) = render_span(source, span);
        lines.push(format!(
            "{} {}",
            styles.muted("-->"),
            styles.path(format!("{source_name}:{line}:{column}"))
        ));
        lines.push(format!("{line} {} {line_text}", styles.muted("|")));
        lines.push(format!(
            "  {} {}",
            styles.muted("|"),
            styles.paint(severity_tone, caret)
        ));
    }
    if let Some(builtin) = error.context.builtin.as_deref() {
        lines.push(format!(
            "{} {}",
            styles.label("builtin:"),
            styles.identifier(builtin)
        ));
    }
    if let Some(task_id) = error.context.task_id.as_deref() {
        lines.push(format!(
            "{} {}",
            styles.label("task:"),
            styles.identifier(task_id)
        ));
    }
    if let Some(phase) = error.context.phase.as_deref() {
        lines.push(format!("{} {phase}", styles.label("phase:")));
    }
    if !error.context.call_stack.is_empty() {
        lines.push(format!("{}:", styles.label("callstack")));
        for frame in &error.context.call_stack {
            lines.push(format!("  {}", styles.identifier(frame)));
        }
    } else if !error.context.call_frames.is_empty() {
        lines.push(format!("{}:", styles.label("callstack")));
        if error.context.call_frames_elided > 0 {
            lines.push(format!(
                "  {}",
                styles.muted(format!(
                    "... {} frames elided ...",
                    error.context.call_frames_elided
                ))
            ));
        }
        for frame in &error.context.call_frames {
            lines.push(format!("  {}", styles.identifier(&frame.function)));
        }
    }
    lines.join("\n")
}

fn render_span(source: &str, span: &SourceSpan) -> (usize, usize, String, String) {
    let offset = span.offset();
    let len = span.len();
    let mut line = 1;
    let mut line_start = 0;
    for (index, character) in source.char_indices() {
        if index >= offset {
            break;
        }
        if character == '\n' {
            line += 1;
            line_start = index + 1;
        }
    }
    let line_end = source[line_start..]
        .find('\n')
        .map(|relative| line_start + relative)
        .unwrap_or(source.len());
    let line_text = source[line_start..line_end].to_string();
    let column = offset.saturating_sub(line_start) + 1;
    let available = line_end.saturating_sub(offset).max(1);
    let caret_len = len.max(1).min(available);
    let caret = format!(
        "{}{}",
        " ".repeat(column.saturating_sub(1)),
        "^".repeat(caret_len)
    );
    (line, column, line_text, caret)
}

#[cfg(test)]
mod compat_tests {
    use super::*;
    use runmat_core::abi::{DiagnosticSeverity, RuntimeDiagnostic};

    #[test]
    fn resolved_error_namespace_defaults_from_language_compat() {
        let mut cfg = RunMatRuntimeConfig::default();
        cfg.runtime.error_namespace.clear();

        cfg.language.compat = config::LanguageCompatMode::RunMat;
        assert_eq!(resolved_error_namespace(&cfg), "RunMat");

        cfg.language.compat = config::LanguageCompatMode::Matlab;
        assert_eq!(resolved_error_namespace(&cfg), "MATLAB");

        cfg.language.compat = config::LanguageCompatMode::Strict;
        assert_eq!(resolved_error_namespace(&cfg), "RunMat");
    }

    #[test]
    fn resolved_error_namespace_honors_explicit_override() {
        let mut cfg = RunMatRuntimeConfig::default();
        cfg.language.compat = config::LanguageCompatMode::Matlab;
        cfg.runtime.error_namespace = "CustomNS".to_string();
        assert_eq!(resolved_error_namespace(&cfg), "CustomNS");
    }

    #[test]
    fn runtime_diagnostic_render_includes_source_and_callstack() {
        let diagnostic = RuntimeDiagnostic {
            code: "RunMat:UndefinedFunction".to_string(),
            severity: DiagnosticSeverity::Error,
            message: "Undefined function: butter".to_string(),
            span: Some(runmat_hir::Span { start: 4, end: 10 }),
            callstack: vec!["main".to_string()],
            callstack_elided: 0,
        };

        let rendered =
            format_runtime_diagnostic(&diagnostic, Some("main.m"), Some("y = butter(4);"));

        assert!(rendered.contains("error: Undefined function: butter"));
        assert!(rendered.contains("id: RunMat:UndefinedFunction"));
        assert!(rendered.contains("--> main.m:1:5"));
        assert!(rendered.contains("callstack:\n  main"));
    }

    #[test]
    fn runtime_diagnostic_render_preserves_warning_severity() {
        let diagnostic = RuntimeDiagnostic {
            code: "RunMat:Warning".to_string(),
            severity: DiagnosticSeverity::Warning,
            message: "careful".to_string(),
            span: None,
            callstack: Vec::new(),
            callstack_elided: 0,
        };

        let rendered = format_runtime_diagnostic(&diagnostic, None, None);

        assert!(rendered.starts_with("warning: careful"));
        assert!(!rendered.starts_with("error: careful"));
    }

    #[test]
    fn styled_runtime_diagnostic_preserves_plain_text() {
        let error = build_runtime_error("Undefined function: butter")
            .with_identifier("RunMat:UndefinedFunction")
            .with_span(SourceSpan::new(SourceOffset::from(4), 6))
            .build();
        let plain = render_runtime_error(
            &error,
            DiagnosticSeverity::Error,
            Some("main.m"),
            Some("x = butter(2);"),
            &StreamStyles::plain(),
        );
        let styled = render_runtime_error(
            &error,
            DiagnosticSeverity::Error,
            Some("main.m"),
            Some("x = butter(2);"),
            &StreamStyles::new(crate::presentation::ColorLevel::Basic),
        );
        assert!(styled.contains("\u{1b}["));
        assert_eq!(strip_ansi(&styled), plain);
    }

    fn strip_ansi(value: &str) -> String {
        let mut output = String::new();
        let mut chars = value.chars().peekable();
        while let Some(character) = chars.next() {
            if character == '\u{1b}' && chars.peek() == Some(&'[') {
                chars.next();
                for next in chars.by_ref() {
                    if next.is_ascii_alphabetic() {
                        break;
                    }
                }
            } else {
                output.push(character);
            }
        }
        output
    }
}
