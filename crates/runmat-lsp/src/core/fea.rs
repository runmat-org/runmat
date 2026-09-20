use std::path::{Path, PathBuf};
use std::sync::Arc;

use lsp_types::{
    CompletionItem, CompletionItemKind, Diagnostic, DiagnosticSeverity, DocumentSymbol, Hover,
    HoverContents, Location, MarkupContent, MarkupKind, Position, Range, SemanticToken,
    SemanticTokens, SignatureHelp, SignatureInformation, SymbolInformation, TextEdit, Url,
};
use runmat_analysis_language::{
    CheckedCallableSignature, CheckedFeaCallable, CheckedFeaProgram, FeaDiagnosticSeverity,
    FeaSymbolKey, FeaSymbolKind, FeaSymbolOrigin, LogicalSourcePath,
};
use runmat_lexer::fea::{FeaToken, FeaTokenKind};
use runmat_parser::fea::{format_fea, FeaFormatOptions, FeaModuleItem, FeaSyntaxModule};
use runmat_types::Span;

use super::analysis::make_document_symbol;
use super::position::{offset_to_position, position_to_offset};

mod presentation;
#[cfg(test)]
mod tests;

use presentation::*;

#[derive(Clone)]
pub struct FeaDocumentAnalysis {
    program: Arc<CheckedFeaProgram>,
    root: LogicalSourcePath,
    source_root: PathBuf,
}

impl FeaDocumentAnalysis {
    pub fn check_with_overlays(
        source: &str,
        physical_path: &Path,
        workspace_roots: &[PathBuf],
        project_overlays: &[(PathBuf, Vec<u8>)],
    ) -> Result<Self, String> {
        let source_root = source_root(physical_path, workspace_roots);
        let mut overlays = project_overlays
            .iter()
            .filter(|(path, _)| path.starts_with(&source_root))
            .cloned()
            .collect::<Vec<_>>();
        if let Some((_, overlay)) = overlays.iter_mut().find(|(path, _)| path == physical_path) {
            *overlay = source.as_bytes().to_vec();
        } else {
            overlays.push((physical_path.to_path_buf(), source.as_bytes().to_vec()));
        }
        let program = runmat_runtime::analysis::check_fea_sources_in_root(
            &source_root,
            physical_path,
            &overlays,
        )?;
        let root = program.bound.graph.root.clone();
        Ok(Self {
            program: Arc::new(program),
            root,
            source_root,
        })
    }

    pub fn status_message(&self) -> String {
        let errors = self
            .program
            .diagnostics
            .iter()
            .chain(&self.program.bound.diagnostics)
            .chain(&self.program.bound.graph.diagnostics)
            .filter(|diagnostic| diagnostic.severity == FeaDiagnosticSeverity::Error)
            .count();
        if errors == 0 {
            "FEA study definition is valid".to_string()
        } else {
            format!("FEA study definition has {errors} error(s)")
        }
    }

    pub fn diagnostics(&self, source: &str) -> Vec<Diagnostic> {
        self.program
            .bound
            .graph
            .diagnostics
            .iter()
            .chain(&self.program.bound.diagnostics)
            .chain(&self.program.diagnostics)
            .filter_map(|diagnostic| {
                let location = diagnostic.location.as_ref();
                if location.is_some_and(|location| location.path != self.root) {
                    return None;
                }
                let span = location.map_or(Span::default(), |location| location.span);
                Some(Diagnostic {
                    range: span_range(source, span),
                    severity: Some(match diagnostic.severity {
                        FeaDiagnosticSeverity::Error => DiagnosticSeverity::ERROR,
                        FeaDiagnosticSeverity::Warning => DiagnosticSeverity::WARNING,
                    }),
                    code: Some(lsp_types::NumberOrString::String(diagnostic.code.clone())),
                    source: Some("runmat-fea".to_string()),
                    message: diagnostic.message.clone(),
                    ..Diagnostic::default()
                })
            })
            .collect()
    }

    pub fn syntax(&self) -> Option<&FeaSyntaxModule> {
        self.program
            .bound
            .graph
            .modules
            .get(&self.root)
            .and_then(|module| module.syntax.as_ref())
    }

    pub fn formatting_edits(&self, source: &str) -> Vec<TextEdit> {
        let formatted = format_fea(source, FeaFormatOptions::default()).source;
        if formatted == source {
            return Vec::new();
        }
        vec![TextEdit {
            range: Range::new(
                Position::new(0, 0),
                offset_to_position(source, source.len()),
            ),
            new_text: formatted,
        }]
    }

    pub fn document_symbols(&self, source: &str) -> Vec<DocumentSymbol> {
        let Some(syntax) = self.syntax() else {
            return Vec::new();
        };
        syntax
            .items
            .iter()
            .filter_map(|item| match item {
                FeaModuleItem::Import(import) => Some(make_document_symbol(
                    import.name.text.clone(),
                    Some(format!("imported {}", import_kind_name(import.kind))),
                    symbol_kind(import_kind_symbol(import.kind)),
                    span_range(source, import.span),
                    span_range(source, import.name.span),
                )),
                FeaModuleItem::Declaration(declaration) => Some(make_document_symbol(
                    declaration.name.text.clone(),
                    Some(declaration_kind_name(declaration.kind).to_string()),
                    symbol_kind(declaration_kind_symbol(declaration.kind)),
                    span_range(source, declaration.span),
                    span_range(source, declaration.name.span),
                )),
                FeaModuleItem::Error(_) => None,
            })
            .collect()
    }

    pub fn workspace_symbols(&self, query: Option<&str>) -> Vec<SymbolInformation> {
        let query = query.map(str::to_ascii_lowercase);
        let query = query.as_deref();
        self.program
            .bound
            .graph
            .modules
            .iter()
            .flat_map(|(module_path, module)| {
                let uri = self.logical_uri(module_path);
                module.syntax.iter().flat_map(move |syntax| {
                    let uri = uri.clone();
                    syntax.items.iter().filter_map(move |item| {
                        let FeaModuleItem::Declaration(declaration) = item else {
                            return None;
                        };
                        let kind = declaration_kind_symbol(declaration.kind);
                        let name = format!(
                            "{}.{}",
                            declaration_kind_name(declaration.kind),
                            declaration.name.text
                        );
                        if query.is_some_and(|query| !name.to_ascii_lowercase().contains(query)) {
                            return None;
                        }
                        serde_json::from_value(serde_json::json!({
                            "name": name,
                            "kind": symbol_kind(kind),
                            "location": {
                                "uri": uri.clone()?,
                                "range": span_range(syntax.source(), declaration.name.span),
                            },
                            "containerName": serde_json::Value::Null,
                        }))
                        .ok()
                    })
                })
            })
            .collect()
    }

    pub fn completion_items(&self) -> Vec<CompletionItem> {
        let mut items = KEYWORDS
            .iter()
            .map(|keyword| CompletionItem {
                label: (*keyword).to_string(),
                kind: Some(CompletionItemKind::KEYWORD),
                ..CompletionItem::default()
            })
            .collect::<Vec<_>>();
        if let Some(symbols) = self.program.bound.symbols.get(&self.root) {
            items.extend(symbols.keys().map(|symbol| CompletionItem {
                label: format!("{}.{}", symbol_kind_name(symbol.kind), symbol.name),
                detail: Some(format!("{} {}", symbol_kind_name(symbol.kind), symbol.name)),
                kind: Some(completion_kind(symbol.kind)),
                ..CompletionItem::default()
            }));
        }
        items
    }

    pub fn hover(&self, source: &str, position: &Position) -> Option<Hover> {
        let token = token_at(self.syntax()?, source, position)?;
        let lexeme = self.syntax()?.tokens.lexeme(token);
        let contents = match token.kind {
            FeaTokenKind::Keyword(_) => format!("`{lexeme}` — FEA language keyword"),
            FeaTokenKind::Identifier => {
                let symbol = self.symbol_at(source, position)?;
                format!(
                    "`{}.{}` — {} declaration",
                    symbol_kind_name(symbol.kind),
                    symbol.name,
                    symbol_kind_name(symbol.kind)
                )
            }
            _ => return None,
        };
        Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: contents,
            }),
            range: Some(span_range(source, token.span)),
        })
    }

    pub fn definition(&self, source: &str, position: &Position) -> Option<Location> {
        let (target_module, target) = self.target_at(source, position)?;
        if target.kind == FeaSymbolKind::Function {
            return Some(Location {
                uri: self.logical_uri(&target_module)?,
                range: Range::new(Position::new(0, 0), Position::new(0, 0)),
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

    pub fn semantic_tokens(&self, source: &str) -> Option<SemanticTokens> {
        let syntax = self.syntax()?;
        let mut encoded = Vec::new();
        let mut previous = Position::new(0, 0);
        let mut first = true;
        for token in &syntax.tokens.tokens {
            let Some(token_type) = semantic_token_type(token.kind) else {
                continue;
            };
            let position = offset_to_position(source, token.span.start);
            let delta_line = if first {
                position.line
            } else {
                position.line.saturating_sub(previous.line)
            };
            let delta_start = if first || delta_line > 0 {
                position.character
            } else {
                position.character.saturating_sub(previous.character)
            };
            encoded.push(SemanticToken {
                delta_line,
                delta_start,
                length: token.span.end.saturating_sub(token.span.start) as u32,
                token_type,
                token_modifiers_bitset: 0,
            });
            previous = position;
            first = false;
        }
        Some(SemanticTokens {
            result_id: None,
            data: encoded,
        })
    }

    fn symbol_at(&self, source: &str, position: &Position) -> Option<FeaSymbolKey> {
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

    fn target_at(
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

    fn logical_uri(&self, path: &LogicalSourcePath) -> Option<Url> {
        let relative = path.as_str().strip_prefix("project/")?;
        Url::from_file_path(
            relative
                .split('/')
                .fold(self.source_root.clone(), |path, part| path.join(part)),
        )
        .ok()
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

pub fn is_fea_uri(uri: &Url) -> bool {
    uri.path().to_ascii_lowercase().ends_with(".fea")
}

fn source_root(path: &Path, workspace_roots: &[PathBuf]) -> PathBuf {
    workspace_roots
        .iter()
        .filter(|root| path.starts_with(root))
        .max_by_key(|root| root.components().count())
        .cloned()
        .or_else(|| path.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."))
}

fn token_at(syntax: &FeaSyntaxModule, source: &str, position: &Position) -> Option<FeaToken> {
    let offset = position_to_offset(source, position);
    syntax
        .tokens
        .tokens
        .iter()
        .copied()
        .find(|token| token.span.start <= offset && offset < token.span.end)
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
