use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::core::analysis::{
    analyze_document_with_compat_and_source, completion_at, definition_locations_at,
    diagnostics_for_document, document_symbols, formatting_edits, function_definitions_in_document,
    function_references_in_document, hover_at, references_locations_at, semantic_tokens_full,
    semantic_tokens_legend, signature_help_at, CompatMode, DocumentAnalysis,
};
use crate::core::position::position_to_offset;
use crate::core::project::ProjectContext;
use crate::core::workspace::{workspace_symbols_from_documents, workspace_symbols_with_project};
use log::{debug, info};
use runmat_config::runtime::{ConfigLoader, LanguageCompatMode};
use serde_json::json;
use tokio::sync::RwLock;
use tower_lsp::jsonrpc::Result as RpcResult;
use tower_lsp::lsp_types::notification::Notification;
use tower_lsp::lsp_types::{
    CompletionOptions, CompletionParams, CompletionResponse, DidChangeConfigurationParams,
    DidChangeTextDocumentParams, DidCloseTextDocumentParams, DidOpenTextDocumentParams,
    DidSaveTextDocumentParams, DocumentFormattingParams, DocumentSymbolParams,
    DocumentSymbolResponse, GotoDefinitionParams, GotoDefinitionResponse, Hover, HoverParams,
    HoverProviderCapability, InitializeParams, InitializeResult, InitializedParams, Location,
    MessageType, OneOf, PositionEncodingKind, ReferenceParams, SemanticTokensOptions,
    SemanticTokensParams, SemanticTokensResult, SemanticTokensServerCapabilities,
    ServerCapabilities, ServerInfo, SignatureHelp, SignatureHelpOptions, SignatureHelpParams,
    TextDocumentContentChangeEvent, TextDocumentSyncCapability, TextDocumentSyncKind, TextEdit,
    Url, WorkspaceFoldersServerCapabilities, WorkspaceServerCapabilities, WorkspaceSymbolParams,
};
use tower_lsp::{async_trait, Client, LanguageServer};
// Cargo substitutes this at compile time so we can surface the precise build version in logs.
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

struct RunmatStatusNotification;

impl Notification for RunmatStatusNotification {
    type Params = serde_json::Value;
    const METHOD: &'static str = "runmat/status";
}

#[derive(Clone, Default)]
struct DocumentState {
    text: String,
    version: Option<i32>,
    analysis: Option<DocumentAnalysis>,
}

struct AnalyzerState {
    documents: HashMap<Url, DocumentState>,
    compat_mode: CompatMode,
    workspace_roots: Vec<PathBuf>,
    project_cache: Option<ProjectCache>,
}

#[derive(Clone)]
struct CachedProjectDoc {
    uri: Url,
    text: String,
    analysis: DocumentAnalysis,
}

#[derive(Clone)]
struct ProjectCache {
    manifest_path: PathBuf,
    compat_mode: CompatMode,
    files: HashMap<PathBuf, CachedProjectDoc>,
}

fn parser_compat(mode: LanguageCompatMode) -> CompatMode {
    match mode {
        LanguageCompatMode::RunMat => CompatMode::RunMat,
        LanguageCompatMode::Matlab => CompatMode::Matlab,
        LanguageCompatMode::Strict => CompatMode::Strict,
    }
}

pub struct RunMatLanguageServer {
    client: Client,
    state: Arc<RwLock<AnalyzerState>>,
}

impl RunMatLanguageServer {
    pub fn new(client: Client) -> Self {
        Self {
            client,
            state: Arc::new(RwLock::new(AnalyzerState {
                documents: HashMap::new(),
                compat_mode: CompatMode::Matlab,
                workspace_roots: Vec::new(),
                project_cache: None,
            })),
        }
    }

    async fn update_document(&self, uri: Url, text: String, version: Option<i32>) {
        {
            let mut state = self.state.write().await;
            let entry = state.documents.entry(uri.clone()).or_default();
            entry.text = text;
            entry.version = version;
        }
        self.reanalyze(&uri).await;
    }

    async fn reanalyze(&self, uri: &Url) {
        let Some((analysis, previous_exports, new_exports)) = self.analyze_and_store(uri).await
        else {
            return;
        };
        self.publish_status_and_diagnostics(uri, &analysis).await;

        let changed_symbols = symmetric_diff_symbols(&previous_exports, &new_exports);
        if changed_symbols.is_empty() {
            return;
        }
        let dependent_uris = self
            .collect_dependent_documents(uri, &changed_symbols)
            .await;
        for dependent in dependent_uris {
            if let Some((analysis, _, _)) = self.analyze_and_store(&dependent).await {
                self.publish_status_and_diagnostics(&dependent, &analysis)
                    .await;
            }
        }
    }

    async fn analyze_and_store(
        &self,
        uri: &Url,
    ) -> Option<(
        DocumentAnalysis,
        std::collections::HashSet<String>,
        std::collections::HashSet<String>,
    )> {
        let previous_exports = {
            let state = self.state.read().await;
            state
                .documents
                .get(uri)
                .and_then(|doc| doc.analysis.as_ref())
                .and_then(|analysis| analysis.semantic.as_ref())
                .map(|semantic| semantic.exported_symbols.clone())
                .unwrap_or_default()
        };
        let compat = {
            let state = self.state.read().await;
            state.compat_mode
        };
        let (text, source_name) = {
            let state = self.state.read().await;
            let doc = state.documents.get(uri)?;
            let source_name = uri
                .to_file_path()
                .ok()
                .and_then(|path| path.to_str().map(str::to_string));
            (doc.text.clone(), source_name)
        };
        let analysis =
            analyze_document_with_compat_and_source(&text, compat, source_name.as_deref());
        let new_exports = analysis
            .semantic
            .as_ref()
            .map(|semantic| semantic.exported_symbols.clone())
            .unwrap_or_default();
        {
            let mut state = self.state.write().await;
            let doc = state.documents.get_mut(uri)?;
            doc.analysis = Some(analysis.clone());
            if let Some(cache) = state.project_cache.as_mut() {
                if let Ok(path) = uri.to_file_path() {
                    if cache.files.contains_key(&path) {
                        cache.files.insert(
                            path,
                            CachedProjectDoc {
                                uri: uri.clone(),
                                text: text.clone(),
                                analysis: analysis.clone(),
                            },
                        );
                    }
                }
            }
        }
        Some((analysis, previous_exports, new_exports))
    }

    async fn publish_status_and_diagnostics(&self, uri: &Url, analysis: &DocumentAnalysis) {
        let status_payload = json!({
            "message": analysis.status_message(),
        });
        let _ = self
            .client
            .send_notification::<RunmatStatusNotification>(status_payload)
            .await;

        self.publish_diagnostics(uri, analysis).await;
    }

    async fn publish_diagnostics(&self, uri: &Url, analysis: &DocumentAnalysis) {
        let text = {
            let state = self.state.read().await;
            state
                .documents
                .get(uri)
                .map(|doc| doc.text.clone())
                .unwrap_or_default()
        };

        let diagnostics = diagnostics_for_document(&text, analysis);

        self.client
            .publish_diagnostics(uri.clone(), diagnostics, None)
            .await;
    }

    async fn remove_document(&self, uri: &Url) {
        {
            let mut state = self.state.write().await;
            state.documents.remove(uri);
            if let Some(cache) = state.project_cache.as_mut() {
                if let Ok(path) = uri.to_file_path() {
                    cache.files.remove(&path);
                }
            }
        }
        self.client
            .publish_diagnostics(uri.clone(), Vec::new(), None)
            .await;
    }

    fn apply_change(text: &mut String, change: TextDocumentContentChangeEvent) {
        if let Some(range) = change.range {
            let start_offset = position_to_offset(text, &range.start);
            let end_offset = position_to_offset(text, &range.end);
            if start_offset <= end_offset && end_offset <= text.len() {
                text.replace_range(start_offset..end_offset, &change.text);
            } else {
                *text = change.text;
            }
        } else {
            *text = change.text;
        }
    }

    async fn collect_dependent_documents(
        &self,
        changed_uri: &Url,
        changed_symbols: &std::collections::HashSet<String>,
    ) -> Vec<Url> {
        let state = self.state.read().await;
        dependent_documents_for_symbols(&state.documents, changed_uri, changed_symbols)
    }

    async fn ensure_project_cache(&self, anchor_uri: Option<&Url>) -> Option<ProjectCache> {
        let (compat_mode, workspace_roots, open_docs) = {
            let state = self.state.read().await;
            let open_docs = state
                .documents
                .iter()
                .filter_map(|(uri, doc)| {
                    doc.analysis
                        .as_ref()
                        .map(|analysis| (uri.clone(), doc.text.clone(), analysis.clone()))
                })
                .collect::<Vec<_>>();
            (state.compat_mode, state.workspace_roots.clone(), open_docs)
        };

        let start_hint = anchor_uri
            .and_then(uri_file_path_string)
            .or_else(|| {
                open_docs
                    .first()
                    .and_then(|(uri, _, _)| uri_file_path_string(uri))
            })
            .or_else(|| {
                workspace_roots
                    .first()
                    .and_then(|path| path.to_str().map(str::to_owned))
            });

        let context = ProjectContext::discover_from_source_name(start_hint.as_deref())?;
        let manifest_path = context.manifest_path().to_path_buf();

        let existing = {
            let state = self.state.read().await;
            state.project_cache.clone()
        };

        if let Some(cache) = existing {
            if cache.manifest_path == manifest_path && cache.compat_mode == compat_mode {
                let mut updated = cache;
                for (uri, text, analysis) in open_docs {
                    if let Ok(path) = uri.to_file_path() {
                        updated.files.insert(
                            path,
                            CachedProjectDoc {
                                uri,
                                text,
                                analysis,
                            },
                        );
                    }
                }
                let mut state = self.state.write().await;
                state.project_cache = Some(updated.clone());
                return Some(updated);
            }
        }

        let mut files = HashMap::new();
        let open_doc_by_path = open_docs
            .iter()
            .filter_map(|(uri, text, analysis)| {
                uri.to_file_path()
                    .ok()
                    .map(|path| (path, (uri.clone(), text.clone(), analysis.clone())))
            })
            .collect::<HashMap<_, _>>();

        for source_file in context.all_source_files() {
            if let Some((uri, text, analysis)) = open_doc_by_path.get(source_file) {
                files.insert(
                    source_file.clone(),
                    CachedProjectDoc {
                        uri: uri.clone(),
                        text: text.clone(),
                        analysis: analysis.clone(),
                    },
                );
                continue;
            }
            let Ok(text) =
                futures::executor::block_on(runmat_filesystem::read_to_string_async(source_file))
            else {
                continue;
            };
            let source_name = source_file.to_str();
            let analysis = analyze_document_with_compat_and_source(&text, compat_mode, source_name);
            let Ok(uri) = Url::from_file_path(source_file) else {
                continue;
            };
            files.insert(
                source_file.clone(),
                CachedProjectDoc {
                    uri,
                    text,
                    analysis,
                },
            );
        }

        let cache = ProjectCache {
            manifest_path,
            compat_mode,
            files,
        };
        let mut state = self.state.write().await;
        state.project_cache = Some(cache.clone());
        Some(cache)
    }
}

fn symmetric_diff_symbols(
    before: &std::collections::HashSet<String>,
    after: &std::collections::HashSet<String>,
) -> std::collections::HashSet<String> {
    before
        .symmetric_difference(after)
        .cloned()
        .collect::<std::collections::HashSet<_>>()
}

fn uri_file_path_string(uri: &Url) -> Option<String> {
    uri.to_file_path()
        .ok()
        .and_then(|path| path.to_str().map(str::to_owned))
}

fn symbol_name_under_cursor(
    text: &str,
    analysis: &DocumentAnalysis,
    position: &tower_lsp::lsp_types::Position,
) -> Option<String> {
    let offset = position_to_offset(text, position);
    analysis
        .tokens
        .iter()
        .find(|token| token.start <= offset && offset < token.end)
        .map(|token| token.lexeme.clone())
}

fn dedupe_location_vec(locations: &mut Vec<Location>) {
    let mut seen = std::collections::HashSet::new();
    locations.retain(|loc| {
        let key = format!(
            "{}:{}:{}:{}:{}",
            loc.uri,
            loc.range.start.line,
            loc.range.start.character,
            loc.range.end.line,
            loc.range.end.character
        );
        seen.insert(key)
    });
}

fn dependent_documents_for_symbols(
    documents: &HashMap<Url, DocumentState>,
    changed_uri: &Url,
    changed_symbols: &std::collections::HashSet<String>,
) -> Vec<Url> {
    documents
        .iter()
        .filter_map(|(uri, doc)| {
            if uri == changed_uri {
                return None;
            }
            let referenced = doc
                .analysis
                .as_ref()
                .and_then(|analysis| analysis.semantic.as_ref())
                .map(|semantic| &semantic.referenced_symbols)?;
            if referenced
                .iter()
                .any(|symbol| changed_symbols.contains(symbol))
            {
                Some(uri.clone())
            } else {
                None
            }
        })
        .collect()
}

#[async_trait]
impl LanguageServer for RunMatLanguageServer {
    async fn initialize(&self, params: InitializeParams) -> RpcResult<InitializeResult> {
        info!("Initializing RunMat language server v{}", SERVER_VERSION);
        let server_info = Some(ServerInfo {
            name: "RunMat Language Server".to_string(),
            version: Some(SERVER_VERSION.to_string()),
        });

        let mut resolved_compat = params
            .initialization_options
            .as_ref()
            .and_then(parse_compat_mode);

        if resolved_compat.is_none() {
            resolved_compat = compat_mode_from_workspace(&params);
        }

        let roots = workspace_roots(&params);
        {
            let mut state = self.state.write().await;
            state.workspace_roots = roots;
            if let Some(mode) = resolved_compat {
                state.compat_mode = parser_compat(mode);
            }
            state.project_cache = None;
        }

        let capabilities = ServerCapabilities {
            text_document_sync: Some(TextDocumentSyncCapability::Kind(
                TextDocumentSyncKind::INCREMENTAL,
            )),
            hover_provider: Some(HoverProviderCapability::Simple(true)),
            definition_provider: Some(OneOf::Left(true)),
            references_provider: Some(OneOf::Left(true)),
            signature_help_provider: Some(SignatureHelpOptions::default()),
            semantic_tokens_provider: Some(
                SemanticTokensServerCapabilities::SemanticTokensOptions(SemanticTokensOptions {
                    legend: semantic_tokens_legend(),
                    range: None,
                    full: Some(tower_lsp::lsp_types::SemanticTokensFullOptions::Bool(true)),
                    work_done_progress_options: Default::default(),
                }),
            ),
            document_formatting_provider: Some(OneOf::Left(true)),
            completion_provider: Some(CompletionOptions {
                resolve_provider: Some(false),
                trigger_characters: None,
                all_commit_characters: None,
                completion_item: None,
                work_done_progress_options: Default::default(),
            }),
            document_symbol_provider: Some(OneOf::Left(true)),
            workspace_symbol_provider: Some(OneOf::Left(true)),
            position_encoding: Some(PositionEncodingKind::UTF8),
            workspace: Some(WorkspaceServerCapabilities {
                workspace_folders: Some(WorkspaceFoldersServerCapabilities {
                    supported: Some(true),
                    change_notifications: Some(OneOf::Left(true)),
                }),
                file_operations: None,
            }),
            ..Default::default()
        };

        Ok(InitializeResult {
            capabilities,
            server_info,
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "RunMat language server ready")
            .await;
    }

    async fn shutdown(&self) -> RpcResult<()> {
        info!("RunMat language server shutting down");
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let uri = params.text_document.uri;
        let version = Some(params.text_document.version);
        self.update_document(uri, params.text_document.text, version)
            .await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri.clone();
        let mut document = {
            let state = self.state.read().await;
            state.documents.get(&uri).cloned().unwrap_or_default()
        };

        for change in params.content_changes {
            Self::apply_change(&mut document.text, change);
        }

        document.version = Some(params.text_document.version);

        {
            let mut state = self.state.write().await;
            state.documents.insert(uri.clone(), document);
        }

        self.reanalyze(&uri).await;
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        self.remove_document(&params.text_document.uri).await;
    }

    async fn did_save(&self, params: DidSaveTextDocumentParams) {
        self.reanalyze(&params.text_document.uri).await;
    }

    async fn hover(&self, params: HoverParams) -> RpcResult<Option<Hover>> {
        let uri = params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        let (text, analysis) = {
            let state = self.state.read().await;
            match state.documents.get(&uri) {
                Some(doc) => (doc.text.clone(), doc.analysis.clone()),
                None => return Ok(None),
            }
        };

        let Some(analysis) = analysis else {
            return Ok(None);
        };

        Ok(hover_at(&text, &analysis, &position))
    }

    async fn completion(&self, params: CompletionParams) -> RpcResult<Option<CompletionResponse>> {
        let uri = params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;

        let (text, analysis) = {
            let state = self.state.read().await;
            match state.documents.get(&uri) {
                Some(doc) => (doc.text.clone(), doc.analysis.clone()),
                None => return Ok(None),
            }
        };

        let Some(analysis) = analysis else {
            return Ok(None);
        };

        Ok(Some(CompletionResponse::Array(completion_at(
            &text, &analysis, &position,
        ))))
    }

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> RpcResult<Option<GotoDefinitionResponse>> {
        let uri = params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        let (text, analysis) = {
            let state = self.state.read().await;
            match state.documents.get(&uri) {
                Some(doc) => (doc.text.clone(), doc.analysis.clone()),
                None => return Ok(None),
            }
        };

        let Some(analysis) = analysis else {
            return Ok(None);
        };

        let mut locations = definition_locations_at(&text, &analysis, &position, &uri);
        if let Some(symbol) = symbol_name_under_cursor(&text, &analysis, &position) {
            if let Some(cache) = self.ensure_project_cache(Some(&uri)).await {
                for doc in cache.files.values() {
                    for range in function_definitions_in_document(&doc.text, &doc.analysis, &symbol)
                    {
                        locations.push(Location {
                            uri: doc.uri.clone(),
                            range,
                        });
                    }
                }
                dedupe_location_vec(&mut locations);
            }
        }
        if locations.is_empty() {
            Ok(None)
        } else {
            Ok(Some(GotoDefinitionResponse::Array(locations)))
        }
    }

    async fn references(&self, params: ReferenceParams) -> RpcResult<Option<Vec<Location>>> {
        let uri = params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;

        let (text, analysis) = {
            let state = self.state.read().await;
            match state.documents.get(&uri) {
                Some(doc) => (doc.text.clone(), doc.analysis.clone()),
                None => return Ok(None),
            }
        };
        let Some(analysis) = analysis else {
            return Ok(None);
        };
        let mut locations = references_locations_at(&text, &analysis, &position, &uri);
        if let Some(symbol) = symbol_name_under_cursor(&text, &analysis, &position) {
            if let Some(cache) = self.ensure_project_cache(Some(&uri)).await {
                for doc in cache.files.values() {
                    for range in function_references_in_document(&doc.text, &doc.analysis, &symbol)
                    {
                        locations.push(Location {
                            uri: doc.uri.clone(),
                            range,
                        });
                    }
                }
                dedupe_location_vec(&mut locations);
            }
        }
        if locations.is_empty() {
            Ok(None)
        } else {
            Ok(Some(locations))
        }
    }

    async fn signature_help(
        &self,
        params: SignatureHelpParams,
    ) -> RpcResult<Option<SignatureHelp>> {
        let uri = params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        let (text, analysis) = {
            let state = self.state.read().await;
            match state.documents.get(&uri) {
                Some(doc) => (doc.text.clone(), doc.analysis.clone()),
                None => return Ok(None),
            }
        };

        let Some(analysis) = analysis else {
            return Ok(None);
        };

        Ok(signature_help_at(&text, &analysis, &position))
    }

    async fn semantic_tokens_full(
        &self,
        params: SemanticTokensParams,
    ) -> RpcResult<Option<SemanticTokensResult>> {
        let uri = params.text_document.uri;

        let analysis = {
            let state = self.state.read().await;
            state.documents.get(&uri).and_then(|d| d.analysis.clone())
        };
        let Some(analysis) = analysis else {
            return Ok(None);
        };

        let text = {
            let state = self.state.read().await;
            state
                .documents
                .get(&uri)
                .map(|doc| doc.text.clone())
                .unwrap_or_default()
        };

        let tokens = semantic_tokens_full(&text, &analysis);
        Ok(tokens.map(SemanticTokensResult::Tokens))
    }

    async fn formatting(
        &self,
        params: DocumentFormattingParams,
    ) -> RpcResult<Option<Vec<TextEdit>>> {
        let uri = params.text_document.uri;
        let (text, analysis) = {
            let state = self.state.read().await;
            match state.documents.get(&uri) {
                Some(doc) => (doc.text.clone(), doc.analysis.clone()),
                None => return Ok(None),
            }
        };

        let Some(analysis) = analysis else {
            return Ok(None);
        };

        let edits = formatting_edits(&text, &analysis);
        Ok(Some(edits))
    }

    async fn document_symbol(
        &self,
        params: DocumentSymbolParams,
    ) -> RpcResult<Option<DocumentSymbolResponse>> {
        let uri = params.text_document.uri;

        let (text, analysis) = {
            let state = self.state.read().await;
            match state.documents.get(&uri) {
                Some(doc) => (doc.text.clone(), doc.analysis.clone()),
                None => return Ok(None),
            }
        };

        let Some(analysis) = analysis else {
            return Ok(None);
        };

        Ok(Some(DocumentSymbolResponse::Nested(document_symbols(
            &text, &analysis,
        ))))
    }

    async fn symbol(
        &self,
        _params: WorkspaceSymbolParams,
    ) -> RpcResult<Option<Vec<lsp_types::SymbolInformation>>> {
        let (docs, compat_mode) = {
            let state = self.state.read().await;
            (
                state
                    .documents
                    .iter()
                    .filter_map(|(uri, doc)| {
                        doc.analysis
                            .as_ref()
                            .map(|analysis| (uri.clone(), doc.text.clone(), analysis.clone()))
                    })
                    .collect::<Vec<_>>(),
                state.compat_mode,
            )
        };
        if let Some(cache) = self.ensure_project_cache(None).await {
            let project_docs = cache
                .files
                .values()
                .map(|doc| (doc.uri.clone(), doc.text.clone(), doc.analysis.clone()))
                .collect::<Vec<_>>();
            return Ok(Some(workspace_symbols_from_documents(
                &project_docs,
                Some(_params.query.as_str()),
            )));
        }
        Ok(Some(workspace_symbols_with_project(
            &docs,
            compat_mode,
            Some(_params.query.as_str()),
        )))
    }

    async fn did_change_configuration(&self, params: DidChangeConfigurationParams) {
        debug!("Configuration updated: {:?}", params.settings);
        self.client
            .log_message(MessageType::INFO, "RunMat configuration updated")
            .await;
    }
}

fn parse_compat_mode(opts: &serde_json::Value) -> Option<LanguageCompatMode> {
    let lang = opts.get("language")?;
    let compat = lang.get("compat")?.as_str()?;
    match compat.trim().to_ascii_lowercase().as_str() {
        "runmat" => Some(LanguageCompatMode::RunMat),
        "matlab" => Some(LanguageCompatMode::Matlab),
        "strict" => Some(LanguageCompatMode::Strict),
        _ => None,
    }
}

fn compat_mode_from_workspace(params: &InitializeParams) -> Option<LanguageCompatMode> {
    for root in workspace_roots(params) {
        if let Some(path) = ConfigLoader::discover_config_path_from(&root) {
            if let Ok(cfg) = ConfigLoader::load_from_file(&path) {
                let compat = cfg.language.compat;
                info!(
                    "Language compatibility set to '{}' via {}",
                    compat_label(parser_compat(compat)),
                    path.display()
                );
                return Some(compat);
            }
        }
    }
    None
}

fn workspace_roots(params: &InitializeParams) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(root_uri) = &params.root_uri {
        if let Ok(path) = root_uri.to_file_path() {
            roots.push(path);
        }
    }
    if let Some(folders) = &params.workspace_folders {
        for folder in folders {
            if let Ok(path) = folder.uri.to_file_path() {
                if !roots.contains(&path) {
                    roots.push(path);
                }
            }
        }
    }
    roots
}

fn compat_label(mode: CompatMode) -> &'static str {
    match mode {
        CompatMode::RunMat => "runmat",
        CompatMode::Matlab => "matlab",
        CompatMode::Strict => "strict",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn dependent_selection_targets_only_docs_referencing_changed_symbols() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("runmat_lsp_invalidation_{suffix}"));
        fs::create_dir_all(root.join("src/+stats")).expect("create package dir");
        fs::write(
            root.join("runmat.toml"),
            r#"
[package]
name = "demo"

[sources]
roots = ["src"]
"#,
        )
        .expect("write manifest");
        fs::write(
            root.join("src/+stats/summarize.m"),
            "function y = summarize(x); y = x + 1; end",
        )
        .expect("write summarize");
        fs::write(root.join("src/main.m"), "x = summarize(41);").expect("write main");
        fs::write(root.join("src/other.m"), "x = 1 + 2;").expect("write other");

        let summarize_uri =
            Url::from_file_path(root.join("src/+stats/summarize.m")).expect("summarize uri");
        let main_uri = Url::from_file_path(root.join("src/main.m")).expect("main uri");
        let other_uri = Url::from_file_path(root.join("src/other.m")).expect("other uri");

        let summarize_text =
            fs::read_to_string(root.join("src/+stats/summarize.m")).expect("read summarize");
        let main_text = fs::read_to_string(root.join("src/main.m")).expect("read main");
        let other_text = fs::read_to_string(root.join("src/other.m")).expect("read other");

        let summarize_analysis = analyze_document_with_compat_and_source(
            &summarize_text,
            CompatMode::RunMat,
            root.join("src/+stats/summarize.m").to_str(),
        );
        let main_analysis = analyze_document_with_compat_and_source(
            &main_text,
            CompatMode::RunMat,
            root.join("src/main.m").to_str(),
        );
        let other_analysis = analyze_document_with_compat_and_source(
            &other_text,
            CompatMode::RunMat,
            root.join("src/other.m").to_str(),
        );

        let mut docs = HashMap::new();
        docs.insert(
            summarize_uri.clone(),
            DocumentState {
                text: summarize_text,
                version: Some(1),
                analysis: Some(summarize_analysis),
            },
        );
        docs.insert(
            main_uri.clone(),
            DocumentState {
                text: main_text,
                version: Some(1),
                analysis: Some(main_analysis),
            },
        );
        docs.insert(
            other_uri.clone(),
            DocumentState {
                text: other_text,
                version: Some(1),
                analysis: Some(other_analysis),
            },
        );

        let changed_symbols = HashSet::from_iter([String::from("summarize")]);
        let dependents = dependent_documents_for_symbols(&docs, &summarize_uri, &changed_symbols);
        assert!(
            dependents.contains(&main_uri),
            "main.m should be invalidated by summarize change"
        );
        assert!(
            !dependents.contains(&other_uri),
            "other.m should not be invalidated"
        );

        let _ = fs::remove_dir_all(&root);
    }
}
