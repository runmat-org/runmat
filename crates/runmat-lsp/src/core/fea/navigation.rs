use lsp_types::{Location, Position, Range, SignatureHelp, SignatureInformation, Url};
use runmat_analysis_language::{
    CheckedCallableSignature, CheckedFeaCallable, FeaSymbolKey, FeaSymbolKind, FeaSymbolOrigin,
    LogicalSourcePath,
};
use runmat_lexer::fea::{FeaToken, FeaTokenKind};
use runmat_parser::fea::FeaSyntaxModule;
use runmat_types::Span;

use super::{
    active_parameter, parameter_label, position_to_offset, span_range, symbol_kind_name, token_at,
    FeaDocumentAnalysis,
};

impl FeaDocumentAnalysis {
    pub fn definition(&self, source: &str, position: &Position) -> Option<Location> {
        let (target_module, target) = self.target_at(source, position)?;
        if target.kind == FeaSymbolKind::Function {
            let facts = self.program.bound.functions.values().find(|facts| {
                facts.module_path == target_module && facts.symbol.as_str() == target.name
            })?;
            let target_source = self.function_sources.get(&target_module)?;
            return Some(Location {
                uri: self.logical_uri(&target_module)?,
                range: span_range(target_source, facts.declaration_span),
            });
        }
        let target_symbol = self
            .program
            .bound
            .symbols
            .get(&target_module)?
            .get(&target)?;
        Some(Location {
            uri: self.logical_uri(&target_module)?,
            range: self.logical_span(&target_module, target_symbol.span)?,
        })
    }

    pub fn references(&self, source: &str, position: &Position) -> Vec<Location> {
        let Some((target_module, target)) = self.target_at(source, position) else {
            return Vec::new();
        };
        self.program
            .bound
            .symbol_uses
            .iter()
            .filter(|usage| usage.target_module == target_module && usage.target == target)
            .filter_map(|usage| {
                Some(Location {
                    uri: self.logical_uri(&usage.module)?,
                    range: self.logical_span(&usage.module, usage.span)?,
                })
            })
            .collect()
    }

    pub fn signature_help(&self, source: &str, position: &Position) -> Option<SignatureHelp> {
        let offset = position_to_offset(source, position);
        let site = self
            .program
            .call_sites
            .iter()
            .filter(|site| {
                site.module == self.root && site.span.start <= offset && offset <= site.span.end
            })
            .min_by_key(|site| site.span.end.saturating_sub(site.span.start))?;
        let (name, labels) = match &site.callable {
            CheckedFeaCallable::Operation {
                operation,
                signature,
                ..
            } => (
                operation.as_str().to_string(),
                signature
                    .inputs
                    .iter()
                    .map(parameter_label)
                    .collect::<Vec<_>>(),
            ),
            CheckedFeaCallable::Named { path, signature } => match signature {
                CheckedCallableSignature::Function(signature) => (
                    path.join("."),
                    signature
                        .inputs
                        .iter()
                        .map(parameter_label)
                        .collect::<Vec<_>>(),
                ),
                CheckedCallableSignature::Declaration(signature) => (
                    path.join("."),
                    signature
                        .inputs
                        .iter()
                        .map(|(name, input)| {
                            format!(
                                "{}{}: {:?}",
                                name,
                                if input.required { "" } else { "?" },
                                input.value_type
                            )
                        })
                        .collect(),
                ),
            },
        };
        Some(SignatureHelp {
            signatures: vec![SignatureInformation {
                label: format!("{name}({})", labels.join(", ")),
                documentation: None,
                parameters: Some(
                    labels
                        .into_iter()
                        .map(|label| lsp_types::ParameterInformation {
                            label: lsp_types::ParameterLabel::Simple(label),
                            documentation: None,
                        })
                        .collect(),
                ),
                active_parameter: None,
            }],
            active_signature: Some(0),
            active_parameter: Some(active_parameter(source, site.span, offset) as u32),
        })
    }

    pub(super) fn symbol_at(&self, source: &str, position: &Position) -> Option<FeaSymbolKey> {
        let syntax = self.syntax()?;
        let token = token_at(syntax, source, position)?;
        if token.kind != FeaTokenKind::Identifier {
            return None;
        }
        let name = syntax.tokens.lexeme(token);
        let symbols = self.program.bound.symbols.get(&self.root)?;
        let namespace = preceding_namespace(syntax, token);
        let candidates = symbols
            .keys()
            .filter(|key| {
                key.name == name
                    && namespace
                        .map(|namespace| namespace == symbol_kind_name(key.kind))
                        .unwrap_or(true)
            })
            .cloned()
            .collect::<Vec<_>>();
        match candidates.as_slice() {
            [key] => Some(key.clone()),
            _ => None,
        }
    }

    pub(super) fn target_at(
        &self,
        source: &str,
        position: &Position,
    ) -> Option<(LogicalSourcePath, FeaSymbolKey)> {
        let offset = position_to_offset(source, position);
        if let Some(usage) = self.program.bound.symbol_uses.iter().find(|usage| {
            usage.module == self.root && usage.span.start <= offset && offset < usage.span.end
        }) {
            return Some((usage.target_module.clone(), usage.target.clone()));
        }
        let key = self.symbol_at(source, position)?;
        let symbol = self.program.bound.symbols.get(&self.root)?.get(&key)?;
        match &symbol.origin {
            FeaSymbolOrigin::Local => Some((symbol.module.clone(), symbol.key.clone())),
            FeaSymbolOrigin::Import {
                target_module,
                target,
            } => Some((target_module.clone(), target.clone())),
            FeaSymbolOrigin::Function => self
                .program
                .bound
                .functions
                .get(&(self.root.clone(), key.name.clone()))
                .map(|facts| {
                    (
                        facts.module_path.clone(),
                        FeaSymbolKey {
                            kind: FeaSymbolKind::Function,
                            name: facts.symbol.as_str().to_string(),
                        },
                    )
                }),
        }
    }

    pub(super) fn logical_uri(&self, path: &LogicalSourcePath) -> Option<Url> {
        let physical_path =
            runmat_runtime::analysis::physical_source_path(&self.source_root, path).ok()?;
        #[cfg(not(target_arch = "wasm32"))]
        {
            Url::from_file_path(physical_path).ok()
        }
        #[cfg(target_arch = "wasm32")]
        {
            let mut uri = Url::parse("file:///").ok()?;
            uri.set_path(physical_path.to_str()?);
            Some(uri)
        }
    }

    fn logical_span(&self, path: &LogicalSourcePath, span: Span) -> Option<Range> {
        let source = self
            .program
            .bound
            .graph
            .modules
            .get(path)?
            .syntax
            .as_ref()?
            .source();
        Some(span_range(source, span))
    }
}

fn preceding_namespace(syntax: &FeaSyntaxModule, token: FeaToken) -> Option<&str> {
    let significant = syntax
        .tokens
        .tokens
        .iter()
        .copied()
        .filter(|candidate| !candidate.kind.is_trivia())
        .collect::<Vec<_>>();
    let index = significant
        .iter()
        .position(|candidate| *candidate == token)?;
    if index < 2 || significant[index - 1].kind != FeaTokenKind::Dot {
        return None;
    }
    let namespace = significant[index - 2];
    (namespace.kind == FeaTokenKind::Identifier).then(|| syntax.tokens.lexeme(namespace))
}
