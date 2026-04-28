use miette::{SourceOffset, SourceSpan};
use runmat_config::{self as config, RunMatConfig};
use runmat_core::RunError;
use runmat_runtime::build_runtime_error;

pub fn parser_compat(mode: config::LanguageCompatMode) -> runmat_parser::CompatMode {
    match mode {
        config::LanguageCompatMode::RunMat | config::LanguageCompatMode::Matlab => {
            runmat_parser::CompatMode::Matlab
        }
        config::LanguageCompatMode::Strict => runmat_parser::CompatMode::Strict,
    }
}

pub fn resolved_error_namespace(cfg: &RunMatConfig) -> String {
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
            let identifier = err.identifier.as_deref().or(Some("RunMat:SemanticError"));
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

    #[test]
    fn resolved_error_namespace_defaults_from_language_compat() {
        let mut cfg = RunMatConfig::default();
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
        let mut cfg = RunMatConfig::default();
        cfg.language.compat = config::LanguageCompatMode::Matlab;
        cfg.runtime.error_namespace = "CustomNS".to_string();
        assert_eq!(resolved_error_namespace(&cfg), "CustomNS");
    }
}
