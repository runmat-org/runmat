use super::{FeaDeclaration, FeaImport, FeaTokenRange};
use crate::lexical::FeaTokenStream;
use crate::tree::{GreenNode, SyntaxNode};
use crate::FeaSyntaxDiagnostic;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeaSyntaxModule {
    pub tokens: FeaTokenStream,
    pub items: Vec<FeaModuleItem>,
    pub diagnostics: Vec<FeaSyntaxDiagnostic>,
    /// The lossless tree the items were read from. Its text is the source, byte for byte.
    pub tree: GreenNode,
}

impl FeaSyntaxModule {
    pub fn source(&self) -> &str {
        &self.tokens.source
    }

    /// A positioned cursor over [`Self::tree`].
    pub fn syntax(&self) -> SyntaxNode {
        SyntaxNode::new_root(self.tree.clone())
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
