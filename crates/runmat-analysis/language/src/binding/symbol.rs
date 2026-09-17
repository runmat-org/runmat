use crate::{FeaDiagnostic, FeaModuleGraph, FunctionFacts, LogicalSourcePath};
use runmat_types::Span;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FeaSymbolKind {
    Geometry,
    Mesh,
    Material,
    Model,
    Study,
    Sweep,
    Execution,
    Function,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FeaSymbolKey {
    pub kind: FeaSymbolKind,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeaSymbolOrigin {
    Local,
    Import {
        target_module: LogicalSourcePath,
        target: FeaSymbolKey,
    },
    Function,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundFeaSymbol {
    pub key: FeaSymbolKey,
    pub module: LogicalSourcePath,
    pub span: Span,
    pub origin: FeaSymbolOrigin,
}

#[derive(Debug)]
pub struct BoundFeaProgram {
    pub graph: FeaModuleGraph,
    pub symbols: BTreeMap<LogicalSourcePath, BTreeMap<FeaSymbolKey, BoundFeaSymbol>>,
    pub functions: BTreeMap<(LogicalSourcePath, String), FunctionFacts>,
    pub diagnostics: Vec<FeaDiagnostic>,
}

impl BoundFeaProgram {
    pub fn is_valid(&self) -> bool {
        self.graph.is_valid()
            && !self.diagnostics.iter().any(|diagnostic| {
                diagnostic.severity == crate::FeaDiagnosticSeverity::Error
                    && diagnostic.availability == crate::SourceAvailability::Available
            })
    }
}
