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
    /// A value's unit measures something other than what its field needs (the shared
    /// `runmat_types::physical::unit_mismatch_message`).
    UnitMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaSyntaxDiagnostic {
    pub kind: FeaSyntaxDiagnosticKind,
    pub span: Span,
    pub message: String,
}
