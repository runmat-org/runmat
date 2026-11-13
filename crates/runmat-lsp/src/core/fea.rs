use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use lsp_types::{
    CompletionItem, CompletionItemKind, Diagnostic, DiagnosticSeverity, DocumentSymbol, Position,
    Range, SemanticToken, SemanticTokens, SymbolInformation, TextEdit, Url,
};
use runmat_analysis_language::{CheckedFeaProgram, FeaDiagnosticSeverity, LogicalSourcePath};
use runmat_lexer::fea::FeaToken;
#[cfg(target_arch = "wasm32")]
use runmat_parser::fea::parse_fea;
use runmat_parser::fea::{format_fea, FeaFormatOptions, FeaModuleItem, FeaSyntaxModule};
use runmat_types::Span;

use super::analysis::make_document_symbol;
use super::position::{offset_to_position, position_to_offset};

pub(crate) mod highlight;
mod hover;
mod hover_docs;
mod navigation;
mod presentation;
mod project;
#[cfg(not(target_arch = "wasm32"))]
mod rename;
#[cfg(test)]
mod tests;

use presentation::*;
pub(crate) use project::resolve_project_facts;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use rename::{FeaFunctionImportPath, FeaRenameTarget};

#[derive(Clone)]
pub struct FeaDocumentAnalysis {
    program: Arc<CheckedFeaProgram>,
    root: LogicalSourcePath,
    source_root: PathBuf,
    function_sources: BTreeMap<LogicalSourcePath, Arc<str>>,
}

impl FeaDocumentAnalysis {
    pub(crate) fn prepare_with_overlays(
        source: &str,
        physical_path: &Path,
        workspace_roots: &[PathBuf],
        project_overlays: &[(PathBuf, Vec<u8>)],
    ) -> Result<runmat_runtime::analysis::PreparedFeaSource, String> {
        let source_root = source_root(physical_path, workspace_roots);
        let mut overlays = project_overlays
            .iter()
            .filter(|(path, _)| path.starts_with(&source_root))
            .cloned()
            .collect::<Vec<_>>();
        replace_root_overlay(&mut overlays, physical_path, source);
        runmat_runtime::analysis::prepare_fea_sources_in_root(
            &source_root,
            physical_path,
            &overlays,
        )
    }

    pub(crate) fn from_checked(
        checked: runmat_runtime::analysis::CheckedFeaSource,
        function_sources: BTreeMap<LogicalSourcePath, Arc<str>>,
    ) -> Self {
        Self {
            program: Arc::new(checked.program),
            root: checked.root_module,
            source_root: checked.source_root,
            function_sources,
        }
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
        use runmat_analysis_language::{FeaAuthoringWordRole, FeaContextualWord};
        let mut items = runmat_lexer::fea::FeaKeyword::ALL
            .iter()
            .map(|keyword| CompletionItem {
                label: keyword.as_source().to_string(),
                kind: Some(CompletionItemKind::KEYWORD),
                ..CompletionItem::default()
            })
            .collect::<Vec<_>>();
        items.extend(FeaContextualWord::ALL.iter().map(|word| CompletionItem {
            label: word.as_source().to_string(),
            kind: Some(match word.role() {
                FeaAuthoringWordRole::Block => CompletionItemKind::KEYWORD,
                FeaAuthoringWordRole::Property => CompletionItemKind::PROPERTY,
                FeaAuthoringWordRole::EnumValue => CompletionItemKind::ENUM_MEMBER,
            }),
            ..CompletionItem::default()
        }));
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

    pub fn semantic_tokens(&self, source: &str) -> Option<SemanticTokens> {
        let syntax = self.syntax()?;
        Some(semantic_tokens_for_syntax(source, syntax))
    }

    #[cfg(target_arch = "wasm32")]
    pub fn lexical_semantic_tokens(source: &str) -> SemanticTokens {
        semantic_tokens_for_syntax(source, &parse_fea(source.to_string()))
    }
}

fn semantic_tokens_for_syntax(source: &str, syntax: &FeaSyntaxModule) -> SemanticTokens {
    let mut encoded = Vec::new();
    let mut previous = Position::new(0, 0);
    let mut first = true;
    for (token, class) in highlight::classify(source, &syntax.tokens.tokens) {
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
            token_type: class.token_type,
            token_modifiers_bitset: class.modifiers,
        });
        previous = position;
        first = false;
    }
    SemanticTokens {
        result_id: None,
        data: encoded,
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
        .or_else(|| {
            super::project::ProjectContext::discover_from_source_name(path.to_str())
                .and_then(|project| project.manifest_path().parent().map(Path::to_path_buf))
        })
        .or_else(|| path.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."))
}

fn replace_root_overlay(overlays: &mut Vec<(PathBuf, Vec<u8>)>, path: &Path, source: &str) {
    if let Some((_, overlay)) = overlays
        .iter_mut()
        .find(|(overlay_path, _)| overlay_path == path)
    {
        *overlay = source.as_bytes().to_vec();
    } else {
        overlays.push((path.to_path_buf(), source.as_bytes().to_vec()));
    }
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
