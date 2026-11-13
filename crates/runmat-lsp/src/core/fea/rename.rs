use std::collections::HashSet;

use lsp_types::{Location, Position, Range, Url};
use runmat_analysis_language::{
    BoundFeaSymbol, FeaSymbolKey, FeaSymbolKind, FeaSymbolOrigin, LogicalSourcePath,
};
use runmat_parser::fea::{FeaBlock, FeaImportKind, FeaModuleItem, FeaStatementKind};
use runmat_types::Span;

use super::{span_range, token_at, FeaDocumentAnalysis};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FeaRenameTarget {
    pub definition_uri: Url,
    pub key: FeaSymbolKey,
    pub range: Range,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FeaFunctionImportPath {
    pub location: Location,
    pub source: String,
}

impl FeaDocumentAnalysis {
    pub(crate) fn rename_target(
        &self,
        source: &str,
        position: &Position,
    ) -> Option<FeaRenameTarget> {
        let syntax = self.syntax()?;
        let token = token_at(syntax, source, position)?;
        let (target_module, key) = self.target_at(source, position)?;
        Some(FeaRenameTarget {
            definition_uri: self.logical_uri(&target_module)?,
            key,
            range: span_range(source, token.span),
        })
    }

    pub(crate) fn rename_locations(&self, target: &FeaRenameTarget) -> Vec<Location> {
        let mut locations = Vec::new();
        for symbols in self.program.bound.symbols.values() {
            for symbol in symbols.values() {
                if self.symbol_resolves_to(symbol, target) {
                    if let Some(location) = self.location(&symbol.module, symbol.span) {
                        locations.push(location);
                    }
                }
            }
        }
        locations.extend(
            self.program
                .bound
                .symbol_uses
                .iter()
                .filter(|usage| self.identity_matches(&usage.target_module, &usage.target, target))
                .filter_map(|usage| self.location(&usage.module, usage.span)),
        );
        if target.key.kind == FeaSymbolKind::Function {
            if let Some((module, facts)) = self.program.bound.functions.values().find_map(|facts| {
                (facts.symbol.as_str() == target.key.name
                    && self.logical_uri(&facts.module_path).as_ref()
                        == Some(&target.definition_uri))
                .then_some((&facts.module_path, facts))
            }) {
                if let Some(source) = self.function_sources.get(module) {
                    locations.push(Location {
                        uri: target.definition_uri.clone(),
                        range: span_range(source, facts.declaration_span),
                    });
                }
            }
        }
        dedupe_locations(&mut locations);
        locations
    }

    pub(crate) fn function_import_paths(
        &self,
        target: &FeaRenameTarget,
    ) -> Vec<FeaFunctionImportPath> {
        if target.key.kind != FeaSymbolKind::Function {
            return Vec::new();
        }
        let mut imports = Vec::new();
        for (module_path, module) in &self.program.bound.graph.modules {
            let Some(syntax) = &module.syntax else {
                continue;
            };
            for item in &syntax.items {
                let FeaModuleItem::Import(import) = item else {
                    continue;
                };
                if import.kind != FeaImportKind::Function || import.name.text != target.key.name {
                    continue;
                }
                let Some(symbol) =
                    self.program
                        .bound
                        .symbols
                        .get(module_path)
                        .and_then(|symbols| {
                            symbols.get(&FeaSymbolKey {
                                kind: FeaSymbolKind::Function,
                                name: import.name.text.clone(),
                            })
                        })
                else {
                    continue;
                };
                if !self.symbol_resolves_to(symbol, target) {
                    continue;
                }
                let Some(uri) = self.logical_uri(module_path) else {
                    continue;
                };
                imports.push(FeaFunctionImportPath {
                    location: Location {
                        uri,
                        range: span_range(syntax.source(), import.source_span),
                    },
                    source: import.source.clone(),
                });
            }
        }
        imports.sort_by(|left, right| {
            location_key(&left.location).cmp(&location_key(&right.location))
        });
        imports.dedup_by(|left, right| left.location == right.location);
        imports
    }

    pub(crate) fn rename_has_collision(&self, target: &FeaRenameTarget, new_name: &str) -> bool {
        self.program.bound.symbols.values().any(|symbols| {
            if !symbols
                .values()
                .any(|symbol| self.symbol_resolves_to(symbol, target))
            {
                return false;
            }
            let candidate = FeaSymbolKey {
                kind: target.key.kind,
                name: new_name.to_string(),
            };
            symbols
                .get(&candidate)
                .is_some_and(|symbol| !self.symbol_resolves_to(symbol, target))
        })
    }

    pub(crate) fn function_parameter_locations(
        &self,
        definition_uri: &Url,
        function_name: &str,
        parameter_name: &str,
    ) -> Vec<Location> {
        let target = FeaRenameTarget {
            definition_uri: definition_uri.clone(),
            key: FeaSymbolKey {
                kind: FeaSymbolKind::Function,
                name: function_name.to_string(),
            },
            range: Range::default(),
        };
        let mut locations = self
            .program
            .call_sites
            .iter()
            .filter(|site| self.call_site_resolves_to(site, function_name, &target))
            .flat_map(|site| {
                site.named_arguments
                    .iter()
                    .filter(move |(name, _)| name == parameter_name)
                    .filter_map(|(_, span)| self.location(&site.module, *span))
            })
            .collect::<Vec<_>>();
        for (module_path, module) in &self.program.bound.graph.modules {
            let Some(syntax) = &module.syntax else {
                continue;
            };
            for item in &syntax.items {
                let FeaModuleItem::Declaration(declaration) = item else {
                    continue;
                };
                self.generator_parameter_locations_in_block(
                    module_path,
                    &declaration.body,
                    &target,
                    parameter_name,
                    &mut locations,
                );
            }
        }
        dedupe_locations(&mut locations);
        locations
    }

    fn generator_parameter_locations_in_block(
        &self,
        module: &LogicalSourcePath,
        block: &FeaBlock,
        target: &FeaRenameTarget,
        parameter_name: &str,
        locations: &mut Vec<Location>,
    ) {
        for statement in &block.statements {
            let nested = match &statement.kind {
                FeaStatementKind::GeneratedByFunction { function, body } => {
                    let function_name = function.segments.get(1);
                    let resolves_to_target = function_name.is_some_and(|name| {
                        self.program.bound.symbol_uses.iter().any(|usage| {
                            usage.module == *module
                                && usage.span == name.span
                                && self.identity_matches(
                                    &usage.target_module,
                                    &usage.target,
                                    target,
                                )
                        })
                    });
                    if resolves_to_target {
                        for arguments in
                            body.statements
                                .iter()
                                .filter_map(|statement| match &statement.kind {
                                    FeaStatementKind::Block { header, body }
                                        if self.span_text(module, header.span)
                                            == Some("arguments") =>
                                    {
                                        Some(body)
                                    }
                                    _ => None,
                                })
                        {
                            for argument in &arguments.statements {
                                let FeaStatementKind::Assignment { target, .. } = &argument.kind
                                else {
                                    continue;
                                };
                                if let [name] = target.segments.as_slice() {
                                    if name.text == parameter_name {
                                        if let Some(location) = self.location(module, name.span) {
                                            locations.push(location);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Some(body)
                }
                FeaStatementKind::SignalDefinition { body, .. }
                | FeaStatementKind::ModelDefinition { body, .. }
                | FeaStatementKind::Block { body, .. }
                | FeaStatementKind::OutputDefinition { body, .. } => Some(body),
                _ => None,
            };
            if let Some(nested) = nested {
                self.generator_parameter_locations_in_block(
                    module,
                    nested,
                    target,
                    parameter_name,
                    locations,
                );
            }
        }
    }

    fn symbol_resolves_to(&self, symbol: &BoundFeaSymbol, target: &FeaRenameTarget) -> bool {
        match &symbol.origin {
            FeaSymbolOrigin::Local => self.identity_matches(&symbol.module, &symbol.key, target),
            FeaSymbolOrigin::Import {
                target_module,
                target: imported_target,
            } => self.identity_matches(target_module, imported_target, target),
            FeaSymbolOrigin::Function => self
                .program
                .bound
                .functions
                .get(&(symbol.module.clone(), symbol.key.name.clone()))
                .is_some_and(|facts| {
                    let key = FeaSymbolKey {
                        kind: FeaSymbolKind::Function,
                        name: facts.symbol.as_str().to_string(),
                    };
                    self.identity_matches(&facts.module_path, &key, target)
                }),
        }
    }

    fn call_site_resolves_to(
        &self,
        site: &runmat_analysis_language::CheckedFeaCallSite,
        function_name: &str,
        target: &FeaRenameTarget,
    ) -> bool {
        matches!(
            &site.callable,
            runmat_analysis_language::CheckedFeaCallable::Named { path, .. }
                if path.last().is_some_and(|name| name == function_name)
        ) && self.program.bound.symbol_uses.iter().any(|usage| {
            usage.module == site.module
                && site.span.start <= usage.span.start
                && usage.span.end <= site.span.end
                && self.identity_matches(&usage.target_module, &usage.target, target)
        })
    }

    fn identity_matches(
        &self,
        module: &LogicalSourcePath,
        key: &FeaSymbolKey,
        target: &FeaRenameTarget,
    ) -> bool {
        key == &target.key && self.logical_uri(module).as_ref() == Some(&target.definition_uri)
    }

    fn location(&self, module: &LogicalSourcePath, span: Span) -> Option<Location> {
        let source = self
            .program
            .bound
            .graph
            .modules
            .get(module)?
            .syntax
            .as_ref()?
            .source();
        Some(Location {
            uri: self.logical_uri(module)?,
            range: span_range(source, span),
        })
    }

    fn span_text(&self, module: &LogicalSourcePath, span: Span) -> Option<&str> {
        self.program
            .bound
            .graph
            .modules
            .get(module)?
            .syntax
            .as_ref()?
            .source()
            .get(span.start..span.end)
            .map(str::trim)
    }
}

fn dedupe_locations(locations: &mut Vec<Location>) {
    let mut seen = HashSet::new();
    locations.retain(|location| seen.insert(location_key(location)));
    locations.sort_by_key(location_key);
}

fn location_key(location: &Location) -> (String, u32, u32, u32, u32) {
    (
        location.uri.as_str().to_string(),
        location.range.start.line,
        location.range.start.character,
        location.range.end.line,
        location.range.end.character,
    )
}
