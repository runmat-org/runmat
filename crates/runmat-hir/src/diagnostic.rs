use crate::Span;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HirDiagnosticSeverity {
    Warning,
    Information,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HirDiagnostic {
    pub message: String,
    pub span: Span,
    pub code: &'static str,
    pub severity: HirDiagnosticSeverity,
}
