use super::{
    logical_path_is_within, resolve_logical_path, FeaModuleEdge, FeaModuleGraph,
    ModuleGraphRequest, ParsedFeaModule,
};
use crate::{
    AnalysisOperationCatalog, FeaDiagnostic, FeaDiagnosticLocation, FeaDiagnosticPhase,
    FeaModuleSource, FeaRelatedDiagnostic, LogicalSourcePath, ModuleRequest, SourceAvailability,
};
use runmat_design_syntax::{FeaImportKind, FeaModuleItem, FeaSyntaxDiagnostic};
use std::collections::{BTreeMap, BTreeSet};

pub fn build_module_graph<H>(host: &H, request: ModuleGraphRequest) -> FeaModuleGraph
where
    H: FeaModuleSource + AnalysisOperationCatalog + ?Sized,
{
    let root = request.root.logical_path.clone();
    let mut builder = ModuleGraphBuilder {
        host,
        source_root: request.source_root.clone(),
        modules: BTreeMap::new(),
        edges: Vec::new(),
        diagnostics: Vec::new(),
        visiting: Vec::new(),
        complete: BTreeSet::new(),
    };
    if !logical_path_is_within(&request.source_root, &root) {
        builder.diagnostics.push(FeaDiagnostic::error(
            "fea.module.root_outside_source_root",
            format!(
                "root module `{root}` is outside source root `{}`",
                request.source_root
            ),
            FeaDiagnosticPhase::ModuleGraph,
            None,
        ));
    }
    builder.insert_and_visit(ParsedFeaModule::from_source(request.root));
    FeaModuleGraph {
        root,
        source_root: request.source_root,
        catalog_fingerprint: host.fingerprint(),
        modules: builder.modules,
        edges: builder.edges,
        diagnostics: builder.diagnostics,
    }
}

struct ModuleGraphBuilder<'a, H: ?Sized> {
    host: &'a H,
    source_root: LogicalSourcePath,
    modules: BTreeMap<LogicalSourcePath, ParsedFeaModule>,
    edges: Vec<FeaModuleEdge>,
    diagnostics: Vec<FeaDiagnostic>,
    visiting: Vec<LogicalSourcePath>,
    complete: BTreeSet<LogicalSourcePath>,
}

impl<H> ModuleGraphBuilder<'_, H>
where
    H: FeaModuleSource + AnalysisOperationCatalog + ?Sized,
{
    fn insert_and_visit(&mut self, module: ParsedFeaModule) {
        let path = module.source.logical_path.clone();
        self.diagnostics.extend(module.diagnostics.iter().cloned());
        self.modules.insert(path.clone(), module);
        self.visit(path);
    }

    fn visit(&mut self, path: LogicalSourcePath) {
        if self.complete.contains(&path) {
            return;
        }
        self.visiting.push(path.clone());
        let imports = self
            .modules
            .get(&path)
            .and_then(|module| module.syntax.as_ref())
            .map(|syntax| {
                syntax
                    .items
                    .iter()
                    .filter_map(|item| match item {
                        FeaModuleItem::Import(import) if import.kind != FeaImportKind::Function => {
                            Some(import.clone())
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        for import in imports {
            let target = match resolve_logical_path(&self.source_root, &path, &import.source) {
                Ok(target) => target,
                Err(error) => {
                    self.diagnostics.push(FeaDiagnostic::error(
                        "fea.module.invalid_path",
                        format!("cannot resolve `{}`: {error}", import.source),
                        FeaDiagnosticPhase::ModuleGraph,
                        Some(FeaDiagnosticLocation {
                            path: path.clone(),
                            span: import.span,
                        }),
                    ));
                    continue;
                }
            };
            self.edges.push(FeaModuleEdge {
                source: path.clone(),
                target: target.clone(),
                kind: import.kind,
                symbol: import.name.text.clone(),
                span: import.span,
            });
            if let Some(cycle_start) = self
                .visiting
                .iter()
                .position(|candidate| candidate == &target)
            {
                let cycle = self.visiting[cycle_start..]
                    .iter()
                    .chain(std::iter::once(&target))
                    .map(LogicalSourcePath::as_str)
                    .collect::<Vec<_>>()
                    .join(" -> ");
                self.diagnostics.push(
                    FeaDiagnostic::error(
                        "fea.module.import_cycle",
                        format!("FEA import cycle: {cycle}"),
                        FeaDiagnosticPhase::ModuleGraph,
                        Some(FeaDiagnosticLocation {
                            path: path.clone(),
                            span: import.span,
                        }),
                    )
                    .with_related(FeaRelatedDiagnostic {
                        message: "cycle returns to this module".to_string(),
                        location: FeaDiagnosticLocation {
                            path: target,
                            span: runmat_types::Span { start: 0, end: 0 },
                        },
                    }),
                );
                continue;
            }
            if !self.modules.contains_key(&target) {
                let loaded = self.host.load(&ModuleRequest {
                    referrer: Some(path.clone()),
                    logical_path: target.clone(),
                });
                match loaded {
                    Ok(source) if source.logical_path == target => {
                        let parsed = ParsedFeaModule::from_source(source);
                        self.diagnostics.extend(parsed.diagnostics.iter().cloned());
                        self.modules.insert(target.clone(), parsed);
                    }
                    Ok(source) => {
                        self.diagnostics.push(FeaDiagnostic::error(
                            "fea.module.identity_mismatch",
                            format!(
                                "module source returned `{}` for requested `{target}`",
                                source.logical_path
                            ),
                            FeaDiagnosticPhase::ModuleGraph,
                            Some(FeaDiagnosticLocation {
                                path: path.clone(),
                                span: import.span,
                            }),
                        ));
                        continue;
                    }
                    Err(error) => {
                        let mut diagnostic = FeaDiagnostic::error(
                            error.code,
                            error.message,
                            FeaDiagnosticPhase::ModuleGraph,
                            Some(FeaDiagnosticLocation {
                                path: path.clone(),
                                span: import.span,
                            }),
                        );
                        diagnostic.availability = error.availability;
                        self.diagnostics.push(diagnostic);
                        continue;
                    }
                }
            }
            self.visit(target);
        }
        self.visiting.pop();
        self.complete.insert(path);
    }
}

pub fn syntax_diagnostics(
    path: &LogicalSourcePath,
    diagnostics: &[FeaSyntaxDiagnostic],
) -> Vec<FeaDiagnostic> {
    diagnostics
        .iter()
        .map(|diagnostic| FeaDiagnostic {
            code: format!("fea.syntax.{:?}", diagnostic.kind).to_ascii_lowercase(),
            message: diagnostic.message.clone(),
            severity: crate::FeaDiagnosticSeverity::Error,
            phase: FeaDiagnosticPhase::Syntax,
            location: Some(FeaDiagnosticLocation {
                path: path.clone(),
                span: diagnostic.span,
            }),
            label: None,
            related: Vec::new(),
            availability: SourceAvailability::Available,
        })
        .collect()
}
