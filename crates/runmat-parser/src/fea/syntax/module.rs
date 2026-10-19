use super::{FeaDeclaration, FeaImport, FeaTokenRange};
use crate::fea::FeaSyntaxDiagnostic;
use runmat_lexer::fea::FeaTokenStream;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeaSyntaxModule {
    pub tokens: FeaTokenStream,
    pub items: Vec<FeaModuleItem>,
    pub diagnostics: Vec<FeaSyntaxDiagnostic>,
}

impl FeaSyntaxModule {
    pub fn source(&self) -> &str {
        &self.tokens.source
    }

    pub fn is_valid(&self) -> bool {
        self.diagnostics.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeaModuleItem {
    Import(FeaImport),
    Declaration(FeaDeclaration),
    Error(FeaErrorNode),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeaErrorNode {
    pub syntax: FeaTokenRange,
}
