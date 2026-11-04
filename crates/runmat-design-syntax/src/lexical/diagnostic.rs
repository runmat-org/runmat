use runmat_types::Span;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeaLexDiagnosticKind {
    InvalidCharacter,
    UnterminatedString,
    InvalidEscape,
    InvalidNumber,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaLexDiagnostic {
    pub kind: FeaLexDiagnosticKind,
    pub span: Span,
    pub message: String,
}
