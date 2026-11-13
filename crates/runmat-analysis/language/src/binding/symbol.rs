use crate::{FeaDiagnostic, FeaModuleGraph, FunctionFacts, LogicalSourcePath};
use runmat_types::Span;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "tool-schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum FeaSymbolKind {
    Geometry,
    Mesh,
    Material,
    Model,
    Study,
    Sweep,
    Execution,
    Table,
    Function,
}

impl FeaSymbolKind {
    pub const fn namespace(self) -> &'static str {
        match self {
            Self::Geometry => "geometry",
            Self::Mesh => "mesh",
            Self::Material => "material",
            Self::Model => "model",
            Self::Study => "study",
            Self::Sweep => "sweep",
            Self::Execution => "execution",
            Self::Table => "table",
            Self::Function => "function",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "tool-schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundFeaSymbolUse {
    pub module: LogicalSourcePath,
    pub span: Span,
    pub target_module: LogicalSourcePath,
    pub target: FeaSymbolKey,
}

#[derive(Debug)]
pub struct BoundFeaProgram {
    pub graph: FeaModuleGraph,
    pub symbols: BTreeMap<LogicalSourcePath, BTreeMap<FeaSymbolKey, BoundFeaSymbol>>,
    pub functions: BTreeMap<(LogicalSourcePath, String), FunctionFacts>,
    pub symbol_uses: Vec<BoundFeaSymbolUse>,
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
