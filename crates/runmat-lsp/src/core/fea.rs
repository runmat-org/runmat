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
#[cfg(feature = "native")] // the native backend's rename is the only user
pub(crate) use rename::{FeaFunctionImportPath, FeaRenameTarget};

#[derive(Clone)]
pub struct FeaDocumentAnalysis {
    program: Arc<CheckedFeaProgram>,
    root: LogicalSourcePath,
    source_root: PathBuf,
    function_sources: BTreeMap<LogicalSourcePath, Arc<str>>,
}

impl FeaDocumentAnalysis {
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

    /// Quick fixes for the diagnostics in `range`: units of the expected kind for a wrong unit.
    pub fn code_actions(
        &self,
        uri: &lsp_types::Url,
        source: &str,
        range: Range,
    ) -> Vec<lsp_types::CodeActionOrCommand> {
        super::unit_fixes::unit_code_actions(uri, source, &self.diagnostics(source), range)
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

/// The folder a study's relative paths (`import … from`, `file =`) may reach: the deepest
/// workspace root that holds the study, else the enclosing project's folder, else the root of
/// the study's filesystem. The last case matches the runtime check, whose source root is the
/// provider root; a document never narrows the root to its own folder.
pub(crate) fn fea_source_root(
    path: &Path,
    workspace_roots: &[PathBuf],
    project: Option<&super::project::ProjectContext>,
) -> PathBuf {
    workspace_roots
        .iter()
        .filter(|root| path.starts_with(root))
        .max_by_key(|root| root.components().count())
        .cloned()
        .or_else(|| {
            project.and_then(|project| project.manifest_path().parent().map(Path::to_path_buf))
        })
        .unwrap_or_else(|| filesystem_root(path))
}

/// `/` for `/project/simulation/x.fea`; the drive root on Windows.
fn filesystem_root(path: &Path) -> PathBuf {
    path.ancestors()
        .last()
        .filter(|root| !root.as_os_str().is_empty())
        .map_or_else(|| PathBuf::from("/"), Path::to_path_buf)
}

/// Check the `.fea` study at `physical_path` with `source` as its text.
///
/// Every file the study reads (imported modules, imported `.m` functions, geometry, the project
/// manifest) comes from `source_overlays` (open editor buffers) or else through the installed
/// `runmat_filesystem` provider, so a host that installs a project provider (a cloud project's
/// virtual paths) gets that project's files. Reads use the provider's async API; only a native
/// server's part and assembly regeneration reads synchronously, through the same provider.
/// Relative paths resolve within the folder `fea_source_root` picks.
pub async fn analyze_fea_source(
    source: &str,
    physical_path: &Path,
    workspace_roots: &[PathBuf],
    source_overlays: &BTreeMap<PathBuf, String>,
    compat: super::analysis::CompatMode,
) -> Result<FeaDocumentAnalysis, String> {
    let project = enclosing_project(physical_path).await;
    let source_root = fea_source_root(physical_path, workspace_roots, project.as_ref());
    let mut prepared =
        prepare_fea_snapshot(source, physical_path, &source_root, source_overlays).await?;
    let requests = prepared.function_requests()?;
    let facts = resolve_project_facts(
        &source_root,
        &requests,
        project.as_ref(),
        compat,
        source_overlays,
    )
    .await?;
    let geometry = prepared
        .requests(facts.functions.clone(), facts.profiles.clone())?
        .geometry;
    for request in geometry {
        let path =
            runmat_runtime::analysis::physical_source_path(&source_root, &request.logical_path)
                .map_err(|error| error.to_string())?;
        if runmat_runtime::geometry::design_source::is_design_geometry(
            request.logical_path.as_str(),
        ) {
            // A regeneration failure is reported by the check at the geometry's `file =`.
            match design_geometry(&path).await {
                Ok(bytes) => prepared.add_geometry(request.logical_path, bytes)?,
                Err(error) => prepared.add_geometry_failure(request.logical_path, error),
            }
        } else if let Ok(bytes) = runmat_filesystem::read_async(&path).await {
            prepared.add_geometry(request.logical_path, bytes)?;
        }
    }
    prepared
        .check(facts.functions, facts.profiles)
        .map(|checked| FeaDocumentAnalysis::from_checked(checked, facts.function_sources))
}

/// The frozen project that holds the study. The browser server checks studies without project
/// facts; its host names the project's root with `setWorkspaceRoots`.
#[cfg(not(target_arch = "wasm32"))]
async fn enclosing_project(path: &Path) -> Option<super::project::ProjectContext> {
    super::project::ProjectContext::discover_from_source_name_async(path.to_str()).await
}

#[cfg(target_arch = "wasm32")]
async fn enclosing_project(_path: &Path) -> Option<super::project::ProjectContext> {
    None
}

/// The study's module graph, with each module read from `source_overlays` or else through the
/// provider's async API. A module that can't be read stays out; the check reports it at its
/// import.
pub(crate) async fn prepare_fea_snapshot(
    source: &str,
    physical_path: &Path,
    source_root: &Path,
    source_overlays: &BTreeMap<PathBuf, String>,
) -> Result<runmat_runtime::analysis::PreparedFeaSourceSnapshot, String> {
    use runmat_runtime::analysis::FeaSourceSnapshotBuilder;
    let logical_path = runmat_runtime::analysis::logical_source_path(source_root, physical_path)?;
    let root = FeaSourceSnapshotBuilder::source_module(logical_path, source.as_bytes().to_vec());
    let mut snapshot = FeaSourceSnapshotBuilder::new(source_root, root)?;
    let mut attempted = std::collections::BTreeSet::new();
    loop {
        let missing = snapshot
            .missing_module_paths()?
            .into_iter()
            .filter(|path| attempted.insert(path.clone()))
            .collect::<Vec<_>>();
        if missing.is_empty() {
            break;
        }
        for logical_path in missing {
            let path = runmat_runtime::analysis::physical_source_path(source_root, &logical_path)
                .map_err(|error| error.to_string())?;
            let bytes = match source_overlays.get(&path) {
                Some(text) => text.as_bytes().to_vec(),
                None => match runmat_filesystem::read_async(&path).await {
                    Ok(bytes) => bytes,
                    Err(_) => continue,
                },
            };
            snapshot.add_module(FeaSourceSnapshotBuilder::source_module(logical_path, bytes))?;
        }
    }
    snapshot.prepare()
}

/// The exact geometry of the part or assembly at `path`. A native server regenerates it on the
/// in-process kernel, reading the design's files through the provider; the browser server has no
/// design kernel.
#[cfg(not(target_arch = "wasm32"))]
async fn design_geometry(path: &Path) -> Result<Vec<u8>, String> {
    runmat_runtime::geometry::design_source::read_design_geometry(path)
}

#[cfg(target_arch = "wasm32")]
async fn design_geometry(path: &Path) -> Result<Vec<u8>, String> {
    Err(format!(
        "the browser language server cannot regenerate design geometry `{}`",
        path.display()
    ))
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
