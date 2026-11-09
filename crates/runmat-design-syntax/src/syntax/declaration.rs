use super::{FeaBlock, FeaExpression, FeaName, FeaReference, FeaTokenRange};
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
    pub source_span: Span,
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
    /// A named quantity over a domain, e.g. `table z : impedance over frequency { ... }`.
    Table,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeaDeclaration {
    pub kind: FeaDeclarationKind,
    pub name: FeaName,
    pub source: Option<FeaReference>,
    pub material_source: Option<FeaExpression>,
    pub target: Option<FeaReference>,
    pub header: FeaTokenRange,
    pub body: FeaBlock,
    pub span: Span,
}
