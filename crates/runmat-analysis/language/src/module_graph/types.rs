use crate::{
    resolve_logical_path, FeaDiagnostic, FunctionRequest, LogicalPathError, LogicalSourcePath,
    SourceDigest, SourceIdentityError, SourceModule, SourceSymbol,
};
use runmat_parser::fea::{FeaImportKind, FeaModuleItem, FeaSyntaxModule};
use runmat_types::Span;
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
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

    pub fn function_requests(&self) -> Result<Vec<FunctionRequest>, FunctionRequestError> {
        self.modules
            .iter()
            .flat_map(|(module_path, module)| {
                module.syntax.iter().flat_map(move |syntax| {
                    syntax.items.iter().filter_map(move |item| {
                        let FeaModuleItem::Import(import) = item else {
                            return None;
                        };
                        (import.kind == FeaImportKind::Function).then_some((module_path, import))
                    })
                })
            })
            .map(|(module_path, import)| {
                Ok((
                    resolve_logical_path(&self.source_root, module_path, &import.source)?,
                    SourceSymbol::new(import.name.text.clone())?,
                ))
            })
            .collect::<Result<std::collections::BTreeSet<_>, FunctionRequestError>>()
            .map(|requests| {
                requests
                    .into_iter()
                    .map(|(module_path, symbol)| FunctionRequest {
                        module_path,
                        symbol,
                    })
                    .collect()
            })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FunctionRequestError {
    #[error("invalid imported function path: {0}")]
    Path(#[from] LogicalPathError),
    #[error("invalid imported function symbol: {0}")]
    Symbol(#[from] SourceIdentityError),
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
