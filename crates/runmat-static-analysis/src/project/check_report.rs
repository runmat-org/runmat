//! Lossless host-neutral diagnostic projection shared by CLI and SDK checks.
pub use crate::frontend::DiagnosticStage as CheckStage;
use runmat_hir::{HirDiagnostic, HirDiagnosticSeverity, Span};
use runmat_types::ProgramSourceId;
use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize)]
pub struct CheckLocation {
    pub source_id: ProgramSourceId,
    pub path: Option<PathBuf>,
    pub start: usize,
    pub end: usize,
    /// One-based Unicode-scalar coordinates. Byte offsets remain authoritative.
    pub line: Option<usize>,
    pub column: Option<usize>,
    pub end_line: Option<usize>,
    pub end_column: Option<usize>,
    pub label: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CheckSuggestion {
    pub span: CheckLocation,
    pub replacement: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CheckDiagnostic {
    pub code: String,
    pub severity: &'static str,
    pub stage: CheckStage,
    pub message: String,
    pub category: Option<String>,
    pub primary: CheckLocation,
    /// Alias for hosts consuming the same location/related shape as FEA checks.
    pub location: CheckLocation,
    pub secondary: Vec<CheckLocation>,
    pub related: Vec<CheckLocation>,
    pub notes: Vec<String>,
    pub help: Option<String>,
    pub suggestions: Vec<CheckSuggestion>,
}

/// Resolve every span independently; an unknown explicit source ID never falls
/// back to primary-source coordinates. No filesystem or ambient lookup occurs.
pub fn project_diagnostic<'a>(
    diagnostic: &HirDiagnostic,
    stage: CheckStage,
    source: impl Fn(Option<ProgramSourceId>) -> Option<(&'a Path, &'a str)>,
) -> CheckDiagnostic {
    let locate = |span: Span, id, label: Option<&str>| {
        let selected = source(id);
        let start = selected.map(|(_, text)| source_position(text, span.start));
        let end = selected.map(|(_, text)| source_position(text, span.end));
        CheckLocation {
            source_id: id.unwrap_or(ProgramSourceId(0)),
            path: selected.map(|(path, _)| path.to_path_buf()),
            start: span.start,
            end: span.end,
            line: start.map(|p| p.0),
            column: start.map(|p| p.1),
            end_line: end.map(|p| p.0),
            end_column: end.map(|p| p.1),
            label: label.map(str::to_owned),
        }
    };
    let primary = locate(
        diagnostic.primary.span,
        diagnostic.primary.source,
        diagnostic.primary.label.as_deref(),
    );
    let secondary: Vec<_> = diagnostic
        .secondary
        .iter()
        .map(|s| locate(s.span, s.source, s.label.as_deref()))
        .collect();
    CheckDiagnostic {
        code: diagnostic.code.clone(),
        severity: match diagnostic.severity {
            HirDiagnosticSeverity::Error => "error",
            HirDiagnosticSeverity::Warning => "warning",
            HirDiagnosticSeverity::Information => "information",
            HirDiagnosticSeverity::Help => "help",
        },
        stage,
        message: diagnostic.message.clone(),
        category: diagnostic.category.clone(),
        location: primary.clone(),
        primary,
        related: secondary.clone(),
        secondary,
        notes: diagnostic.notes.iter().map(|n| n.message.clone()).collect(),
        help: diagnostic.help.clone(),
        suggestions: diagnostic
            .suggestions
            .iter()
            .map(|s| CheckSuggestion {
                span: locate(s.span, s.source, None),
                replacement: s.replacement.clone(),
                message: s.message.clone(),
            })
            .collect(),
    }
}

/// One-based line/scalar column for a UTF-8 byte offset, clamped safely.
pub fn source_position(source: &str, offset: usize) -> (usize, usize) {
    let mut offset = offset.min(source.len());
    while !source.is_char_boundary(offset) {
        offset -= 1;
    }
    let start = source[..offset].rfind('\n').map_or(0, |i| i + 1);
    (
        source[..start].bytes().filter(|b| *b == b'\n').count() + 1,
        source[start..offset].chars().count() + 1,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn locations_preserve_unicode_secondary_suggestions_and_unknown_sources() {
        let text = "% π\nx=1;\n";
        let mut diagnostic = HirDiagnostic::new(
            "test",
            HirDiagnosticSeverity::Error,
            "message",
            Span { start: 5, end: 6 },
        )
        .with_category("display-only-category")
        .with_secondary(Span { start: 5, end: 6 }, "related")
        .with_suggestion(Span { start: 5, end: 6 }, "y", "rename");
        diagnostic.secondary[0].source = Some(ProgramSourceId(99));
        let projected = project_diagnostic(&diagnostic, CheckStage::Semantic, |id| {
            id.is_none_or(|id| id == ProgramSourceId(0))
                .then_some((Path::new("/main.m"), text))
        });
        assert_eq!(projected.primary.line, Some(2));
        assert_eq!(projected.primary.column, Some(1));
        assert!(projected.related[0].path.is_none());
        assert_eq!(projected.related[0].source_id, ProgramSourceId(99));
        assert_eq!(projected.suggestions[0].replacement, "y");
        assert!(matches!(projected.stage, CheckStage::Semantic));
    }
}
