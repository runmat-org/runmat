use runmat_types::Span;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeaSyntaxDiagnosticKind {
    Lexical,
    ExpectedImportKind,
    ExpectedIdentifier,
    ExpectedString,
    ExpectedToken,
    ExpectedDeclaration,
    ImportAfterDeclaration,
    UnclosedBlock,
    InvalidExpression,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaSyntaxDiagnostic {
    pub kind: FeaSyntaxDiagnosticKind,
    pub span: Span,
    pub message: String,
}
