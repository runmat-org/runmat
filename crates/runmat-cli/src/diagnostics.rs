use miette::{SourceOffset, SourceSpan};
use runmat_config::runtime::{self as config, RunMatRuntimeConfig};
use runmat_core::{abi::RuntimeDiagnostic, RunError};
use runmat_runtime::build_runtime_error;

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
            Some(format_diagnostic(
                &message,
                Some("RunMat:SyntaxError"),
                Some(span),
                source_name,
                source,
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
            Some(format_diagnostic(
                &err.message,
                identifier,
                span,
                source_name,
                source,
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
            Some(format_diagnostic(
                &err.message,
                identifier,
                span,
                source_name,
                source,
            ))
        }
        RunError::Runtime(err) => {
            Some(err.format_diagnostic_with_source(Some(source_name), Some(source)))
        }
    }
}

pub fn format_runtime_diagnostic(
    diagnostic: &RuntimeDiagnostic,
    source_name: Option<&str>,
    source: Option<&str>,
) -> String {
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

    let mut rendered = builder
        .build()
        .format_diagnostic_with_source(source_name, source);
    if !diagnostic.callstack.is_empty() {
        rendered.push_str("\ncallstack:");
        if diagnostic.callstack_elided > 0 {
            rendered.push_str(&format!(
                "\n  ... {} frames elided ...",
                diagnostic.callstack_elided
            ));
        }
        for frame in &diagnostic.callstack {
            rendered.push_str(&format!("\n  {frame}"));
        }
    }
    rendered
}

pub fn format_diagnostic(
    message: &str,
    identifier: Option<&str>,
    span: Option<SourceSpan>,
    source_name: &str,
    source: &str,
) -> String {
    let mut builder = build_runtime_error(message);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    if let Some(span) = span {
        builder = builder.with_span(span);
    }
    builder
        .build()
        .format_diagnostic_with_source(Some(source_name), Some(source))
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
}
