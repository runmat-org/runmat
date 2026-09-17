use super::{FeaBlock, FeaName, FeaTokenRange};
use runmat_types::Span;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeaImportKind {
    Geometry,
    Mesh,
    Material,
    Model,
    Study,
    Sweep,
    Execution,
    Function,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeaImport {
    pub kind: FeaImportKind,
    pub name: FeaName,
    pub source: String,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeaDeclarationKind {
    Geometry,
    Mesh,
    Material,
    Model,
    Study,
    Sweep,
    Execution,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeaDeclaration {
    pub kind: FeaDeclarationKind,
    pub name: FeaName,
    pub header: FeaTokenRange,
    pub body: FeaBlock,
    pub span: Span,
}
