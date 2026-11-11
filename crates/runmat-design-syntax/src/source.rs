use crate::dialect::Dialect;
use crate::lexical::FeaTokenStream;
use crate::tree::{GreenNode, SyntaxNode};
use crate::FeaSyntaxDiagnostic;

/// A parsed source of any dialect: its lossless tree and syntax diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceSyntax {
    pub dialect: Dialect,
    pub tokens: FeaTokenStream,
    pub tree: GreenNode,
    pub diagnostics: Vec<FeaSyntaxDiagnostic>,
}

impl SourceSyntax {
    pub fn source(&self) -> &str {
        &self.tokens.source
    }

    pub fn syntax(&self) -> SyntaxNode {
        SyntaxNode::new_root(self.tree.clone())
    }

    /// The typed view of the file.
    pub fn file(&self) -> crate::ast::SourceFile {
        crate::ast::SourceFile::cast(self.syntax()).expect("the root is a source file")
    }

    pub fn is_valid(&self) -> bool {
        self.diagnostics.is_empty()
    }
}
