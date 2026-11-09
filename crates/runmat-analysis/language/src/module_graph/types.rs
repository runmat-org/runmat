use crate::{
    resolve_logical_path, FeaDiagnostic, FunctionRequest, LogicalPathError, LogicalSourcePath,
    SourceDigest, SourceIdentityError, SourceModule, SourceSymbol,
};
use runmat_fea_syntax::{FeaImportKind, FeaModuleItem, FeaSyntaxModule};
use runmat_types::Span;
use std::collections::{BTreeMap, BTreeSet};

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
    /// Non-FEA files a module reads at check time, by logical path: the board layout a
    /// `cross_section { stackup = "<file>" }` takes its layers from.
    pub resources: BTreeMap<LogicalSourcePath, SourceModule>,
    pub diagnostics: Vec<FeaDiagnostic>,
}

impl FeaModuleGraph {
    /// Source invalidation includes unresolved import targets as well as loaded modules.
    pub fn source_dependencies(&self) -> Result<BTreeSet<LogicalSourcePath>, FunctionRequestError> {
        let mut paths = self.modules.keys().cloned().collect::<BTreeSet<_>>();
        for (module_path, module) in &self.modules {
            for (relative, _) in module.syntax.iter().flat_map(stackup_references) {
                paths.insert(resolve_logical_path(&self.source_root, module_path, &relative)?);
            }
        }
        paths.extend(self.edges.iter().map(|edge| edge.target.clone()));
        paths.extend(
            self.function_requests()?
                .into_iter()
                .map(|request| request.module_path),
        );
        for (module_path, module) in &self.modules {
            for item in module.syntax.iter().flat_map(|s| &s.items) {
                let FeaModuleItem::Declaration(decl) = item else {
                    continue;
                };
                if decl.kind != runmat_fea_syntax::FeaDeclarationKind::Geometry {
                    continue;
                }
                for statement in &decl.body.statements {
                    let runmat_fea_syntax::FeaStatementKind::Assignment { target, value, .. } =
                        &statement.kind
                    else {
                        continue;
                    };
                    if target.segments.len() != 1 || target.segments[0].text != "file" {
                        continue;
                    }
                    let runmat_fea_syntax::FeaExpressionKind::String(relative) = &value.kind else {
                        continue;
                    };
                    let cad = resolve_logical_path(&self.source_root, module_path, relative)?;
                    paths.insert(cad);
                }
            }
        }
        Ok(paths)
    }
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
            .chain(
                self.resources
                    .iter()
                    .map(|(path, resource)| (path.clone(), resource.source_digest)),
            )
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

/// The `stackup = "<file>"` paths of a module's `cross_section` geometry blocks, as authored,
/// with the span of each path.
pub fn stackup_references(syntax: &FeaSyntaxModule) -> Vec<(String, Span)> {
    use runmat_fea_syntax::{FeaDeclarationKind, FeaExpressionKind, FeaStatementKind};
    let mut found = Vec::new();
    for item in &syntax.items {
        let FeaModuleItem::Declaration(declaration) = item else {
            continue;
        };
        if declaration.kind != FeaDeclarationKind::Geometry {
            continue;
        }
        for statement in &declaration.body.statements {
            let FeaStatementKind::Block { header, body } = &statement.kind else {
                continue;
            };
            let text = syntax
                .source()
                .get(header.span.start..header.span.end)
                .unwrap_or_default();
            if text.trim() != "cross_section" {
                continue;
            }
            for member in &body.statements {
                let FeaStatementKind::Assignment { target, value, .. } = &member.kind else {
                    continue;
                };
                if target.segments.len() == 1 && target.segments[0].text == "stackup" {
                    if let FeaExpressionKind::String(path) = &value.kind {
                        found.push((path.clone(), value.span));
                    }
                }
            }
        }
    }
    found
}
