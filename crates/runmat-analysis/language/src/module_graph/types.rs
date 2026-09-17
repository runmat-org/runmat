use crate::{FeaDiagnostic, LogicalSourcePath, SourceDigest, SourceModule};
use runmat_parser::fea::{FeaImportKind, FeaSyntaxModule};
use runmat_types::Span;
use std::collections::BTreeMap;

#[derive(Debug)]
pub struct ParsedFeaModule {
    pub source: SourceModule,
    pub syntax: Option<FeaSyntaxModule>,
    pub diagnostics: Vec<FeaDiagnostic>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeaModuleEdge {
    pub source: LogicalSourcePath,
    pub target: LogicalSourcePath,
    pub kind: FeaImportKind,
    pub symbol: String,
    pub span: Span,
}

#[derive(Debug)]
pub struct FeaModuleGraph {
    pub root: LogicalSourcePath,
    pub source_root: LogicalSourcePath,
    pub catalog_fingerprint: [u8; 32],
    pub modules: BTreeMap<LogicalSourcePath, ParsedFeaModule>,
    pub edges: Vec<FeaModuleEdge>,
    pub diagnostics: Vec<FeaDiagnostic>,
}

impl FeaModuleGraph {
    pub fn is_valid(&self) -> bool {
        !self.diagnostics.iter().any(|diagnostic| {
            diagnostic.severity == crate::FeaDiagnosticSeverity::Error
                && diagnostic.availability == crate::SourceAvailability::Available
        })
    }

    pub fn source_digests(&self) -> BTreeMap<LogicalSourcePath, SourceDigest> {
        self.modules
            .iter()
            .map(|(path, module)| (path.clone(), module.source.source_digest))
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleGraphRequest {
    pub source_root: LogicalSourcePath,
    pub root: SourceModule,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckRequest {
    pub modules: ModuleGraphRequest,
}
