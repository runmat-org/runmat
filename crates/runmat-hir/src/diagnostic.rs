use crate::Span;
use runmat_types::ProgramSourceId;
use serde::{Deserialize, Serialize};

/// Static presentation and executable admission are separate decisions.
/// A known failing value operation remains a diagnostic error, but it is
/// valid executable code whose failure can be handled by its caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiagnosticExecution {
    RejectOnError,
    RuntimeCheck,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HirDiagnosticSeverity {
    Error,
    Warning,
    Information,
    Help,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HirDiagnosticSpan {
    pub span: Span,
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<ProgramSourceId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HirDiagnosticNote {
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HirDiagnosticSuggestion {
    pub span: Span,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<ProgramSourceId>,
    pub replacement: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HirDiagnostic {
    pub code: String,
    pub severity: HirDiagnosticSeverity,
    pub execution: DiagnosticExecution,
    pub message: String,
    pub primary: HirDiagnosticSpan,
    pub secondary: Vec<HirDiagnosticSpan>,
    pub notes: Vec<HirDiagnosticNote>,
    pub help: Option<String>,
    pub suggestions: Vec<HirDiagnosticSuggestion>,
    pub category: Option<String>,
}

impl HirDiagnostic {
    pub fn new(
        code: impl Into<String>,
        severity: HirDiagnosticSeverity,
        message: impl Into<String>,
        span: Span,
    ) -> Self {
        Self {
            code: code.into(),
            severity,
            execution: DiagnosticExecution::RejectOnError,
            message: message.into(),
            primary: HirDiagnosticSpan {
                span,
                label: None,
                source: None,
            },
            secondary: Vec::new(),
            notes: Vec::new(),
            help: None,
            suggestions: Vec::new(),
            category: None,
        }
    }

    pub fn with_runtime_check(mut self) -> Self {
        self.execution = DiagnosticExecution::RuntimeCheck;
        self
    }

    pub fn blocks_execution(&self) -> bool {
        self.severity == HirDiagnosticSeverity::Error
            && self.execution == DiagnosticExecution::RejectOnError
    }

    pub fn with_primary_label(mut self, label: impl Into<String>) -> Self {
        self.primary.label = Some(label.into());
        self
    }

    pub fn with_source(mut self, source: ProgramSourceId) -> Self {
        self.set_source(source);
        self
    }

    pub fn set_source(&mut self, source: ProgramSourceId) {
        self.primary.source = Some(source);
        for secondary in &mut self.secondary {
            secondary.source.get_or_insert(source);
        }
        for suggestion in &mut self.suggestions {
            suggestion.source.get_or_insert(source);
        }
    }

    pub fn with_secondary(mut self, span: Span, label: impl Into<String>) -> Self {
        self.secondary.push(HirDiagnosticSpan {
            span,
            label: Some(label.into()),
            source: self.primary.source,
        });
        self
    }

    pub fn with_note(mut self, message: impl Into<String>) -> Self {
        self.notes.push(HirDiagnosticNote {
            message: message.into(),
        });
        self
    }

    pub fn with_help(mut self, message: impl Into<String>) -> Self {
        self.help = Some(message.into());
        self
    }

    pub fn with_suggestion(
        mut self,
        span: Span,
        replacement: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        self.suggestions.push(HirDiagnosticSuggestion {
            span,
            source: self.primary.source,
            replacement: replacement.into(),
            message: message.into(),
        });
        self
    }

    pub fn with_category(mut self, category: &'static str) -> Self {
        self.category = Some(category.to_string());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_builder_records_structured_context() {
        let span = Span { start: 1, end: 4 };
        let secondary = Span { start: 8, end: 12 };

        let diagnostic = HirDiagnostic::new(
            "RM0001",
            HirDiagnosticSeverity::Error,
            "undefined binding",
            span,
        )
        .with_primary_label("binding used here")
        .with_secondary(secondary, "candidate declared here")
        .with_note("name resolution is lexical")
        .with_help("declare the binding before use")
        .with_suggestion(span, "x", "use an existing binding")
        .with_category("resolution");

        assert_eq!(diagnostic.code, "RM0001");
        assert_eq!(diagnostic.primary.span, span);
        assert_eq!(
            diagnostic.primary.label.as_deref(),
            Some("binding used here")
        );
        assert_eq!(diagnostic.secondary[0].span, secondary);
        assert_eq!(diagnostic.notes[0].message, "name resolution is lexical");
        assert_eq!(
            diagnostic.help.as_deref(),
            Some("declare the binding before use")
        );
        assert_eq!(diagnostic.suggestions[0].replacement, "x");
        assert_eq!(diagnostic.category.as_deref(), Some("resolution"));
    }
}
