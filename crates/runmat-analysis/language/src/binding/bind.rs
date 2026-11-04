use super::{
    uses::collect_symbol_uses, BoundFeaProgram, BoundFeaSymbol, FeaSymbolKey, FeaSymbolKind,
    FeaSymbolOrigin,
};
use crate::{
    resolve_logical_path, FeaDiagnostic, FeaDiagnosticLocation, FeaDiagnosticPhase, FeaModuleGraph,
    FeaRelatedDiagnostic, FunctionRequest, LogicalSourcePath, RunMatFunctionSource, SourceSymbol,
};
use runmat_design_syntax::{FeaDeclarationKind, FeaImportKind, FeaModuleItem};
use runmat_types::Span;
use std::collections::BTreeMap;

pub fn bind_program<H>(host: &H, graph: FeaModuleGraph) -> BoundFeaProgram
where
    H: RunMatFunctionSource + ?Sized,
{
    let mut symbols = BTreeMap::<LogicalSourcePath, BTreeMap<FeaSymbolKey, BoundFeaSymbol>>::new();
    let mut functions = BTreeMap::new();
    let mut diagnostics = Vec::new();

    for (path, module) in &graph.modules {
        let Some(syntax) = &module.syntax else {
            continue;
        };
        for item in &syntax.items {
            let FeaModuleItem::Declaration(declaration) = item else {
                continue;
            };
            let key = FeaSymbolKey {
                kind: declaration_kind(declaration.kind),
                name: declaration.name.text.clone(),
            };
            insert_symbol(
                &mut symbols,
                &mut diagnostics,
                BoundFeaSymbol {
                    key,
                    module: path.clone(),
                    span: declaration.name.span,
                    origin: FeaSymbolOrigin::Local,
                },
            );
        }
    }

    for (path, module) in &graph.modules {
        let Some(syntax) = &module.syntax else {
            continue;
        };
        for item in &syntax.items {
            let FeaModuleItem::Import(import) = item else {
                continue;
            };
            let key = FeaSymbolKey {
                kind: import_kind(import.kind),
                name: import.name.text.clone(),
            };
            let target_path = match resolve_logical_path(&graph.source_root, path, &import.source) {
                Ok(target) => target,
                Err(_) => continue,
            };
            if import.kind == FeaImportKind::Function {
                let symbol = match SourceSymbol::new(import.name.text.clone()) {
                    Ok(symbol) => symbol,
                    Err(_) => continue,
                };
                match host.resolve(&FunctionRequest {
                    module_path: target_path.clone(),
                    symbol: symbol.clone(),
                }) {
                    Ok(facts)
                        if facts.module_path == target_path && facts.symbol == symbol =>
                    {
                        functions.insert(
                            (path.clone(), import.name.text.clone()),
                            facts,
                        );
                        insert_symbol(
                            &mut symbols,
                            &mut diagnostics,
                            BoundFeaSymbol {
                                key,
                                module: path.clone(),
                                span: import.name.span,
                                origin: FeaSymbolOrigin::Function,
                            },
                        );
                    }
                    Ok(facts) => diagnostics.push(FeaDiagnostic::error(
                        "fea.binding.function_identity_mismatch",
                        format!(
                            "function resolver returned `{}::{}` for requested `{target_path}::{symbol}`",
                            facts.module_path, facts.symbol
                        ),
                        FeaDiagnosticPhase::Binding,
                        Some(location(path, import.span)),
                    )),
                    Err(error) => {
                        let mut diagnostic = FeaDiagnostic::error(
                            error.code,
                            error.message,
                            FeaDiagnosticPhase::Binding,
                            Some(location(path, import.span)),
                        );
                        diagnostic.availability = error.availability;
                        diagnostics.push(diagnostic);
                    }
                }
                continue;
            }

            let target_key = key.clone();
            if !graph.modules.contains_key(&target_path) {
                continue;
            }
            let target = symbols
                .get(&target_path)
                .and_then(|module_symbols| module_symbols.get(&target_key));
            if target.is_none() {
                diagnostics.push(FeaDiagnostic::error(
                    "fea.binding.missing_export",
                    format!(
                        "module `{target_path}` does not declare {} `{}`",
                        symbol_kind_name(key.kind),
                        key.name
                    ),
                    FeaDiagnosticPhase::Binding,
                    Some(location(path, import.span)),
                ));
                continue;
            }
            insert_symbol(
                &mut symbols,
                &mut diagnostics,
                BoundFeaSymbol {
                    key,
                    module: path.clone(),
                    span: import.name.span,
                    origin: FeaSymbolOrigin::Import {
                        target_module: target_path,
                        target: target_key,
                    },
                },
            );
        }
    }

    let symbol_uses = collect_symbol_uses(&graph, &symbols, &functions);
    BoundFeaProgram {
        graph,
        symbols,
        functions,
        symbol_uses,
        diagnostics,
    }
}

fn insert_symbol(
    symbols: &mut BTreeMap<LogicalSourcePath, BTreeMap<FeaSymbolKey, BoundFeaSymbol>>,
    diagnostics: &mut Vec<FeaDiagnostic>,
    symbol: BoundFeaSymbol,
) {
    let module_symbols = symbols.entry(symbol.module.clone()).or_default();
    if let Some(existing) = module_symbols.get(&symbol.key) {
        diagnostics.push(
            FeaDiagnostic::error(
                "fea.binding.duplicate_symbol",
                format!(
                    "duplicate {} `{}`",
                    symbol_kind_name(symbol.key.kind),
                    symbol.key.name
                ),
                FeaDiagnosticPhase::Binding,
                Some(location(&symbol.module, symbol.span)),
            )
            .with_related(FeaRelatedDiagnostic {
                message: "first declaration is here".to_string(),
                location: location(&existing.module, existing.span),
            }),
        );
        return;
    }
    module_symbols.insert(symbol.key.clone(), symbol);
}

fn declaration_kind(kind: FeaDeclarationKind) -> FeaSymbolKind {
    match kind {
        FeaDeclarationKind::Geometry => FeaSymbolKind::Geometry,
        FeaDeclarationKind::Mesh => FeaSymbolKind::Mesh,
        FeaDeclarationKind::Material => FeaSymbolKind::Material,
        FeaDeclarationKind::Model => FeaSymbolKind::Model,
        FeaDeclarationKind::Study => FeaSymbolKind::Study,
        FeaDeclarationKind::Sweep => FeaSymbolKind::Sweep,
        FeaDeclarationKind::Execution => FeaSymbolKind::Execution,
        FeaDeclarationKind::Table => FeaSymbolKind::Table,
    }
}

fn import_kind(kind: FeaImportKind) -> FeaSymbolKind {
    match kind {
        FeaImportKind::Geometry => FeaSymbolKind::Geometry,
        FeaImportKind::Mesh => FeaSymbolKind::Mesh,
        FeaImportKind::Material => FeaSymbolKind::Material,
        FeaImportKind::Model => FeaSymbolKind::Model,
        FeaImportKind::Study => FeaSymbolKind::Study,
        FeaImportKind::Sweep => FeaSymbolKind::Sweep,
        FeaImportKind::Execution => FeaSymbolKind::Execution,
        FeaImportKind::Function => FeaSymbolKind::Function,
    }
}

fn symbol_kind_name(kind: FeaSymbolKind) -> &'static str {
    kind.namespace()
}

fn location(path: &LogicalSourcePath, span: Span) -> FeaDiagnosticLocation {
    FeaDiagnosticLocation {
        path: path.clone(),
        span,
    }
}
