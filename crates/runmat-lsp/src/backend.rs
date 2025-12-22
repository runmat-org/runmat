#![cfg(feature = "native")]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use log::{debug, info};
use crate::core::analysis::{
    analyze_document_with_compat, completion_at, definition_at, diagnostics_for_document,
    document_symbols, formatting_edits, hover_at, semantic_tokens_full, semantic_tokens_legend,
    signature_help_at, CompatMode, DocumentAnalysis,
};
use crate::core::workspace::workspace_symbols;
use crate::core::position::position_to_offset;
use serde::Deserialize;
use tokio::sync::RwLock;
use tower_lsp::jsonrpc::Result as RpcResult;
use tower_lsp::lsp_types::notification::Notification;
use tower_lsp::lsp_types::{
    CompletionOptions, CompletionParams, CompletionResponse, DidChangeConfigurationParams,
    DidChangeTextDocumentParams, DidCloseTextDocumentParams, DidOpenTextDocumentParams,
    DidSaveTextDocumentParams, DocumentFormattingParams, DocumentSymbolParams,
    DocumentSymbolResponse, GotoDefinitionParams, GotoDefinitionResponse, Hover, HoverParams,
    HoverProviderCapability, InitializeParams, InitializeResult, InitializedParams, Location,
    MessageType, OneOf, PositionEncodingKind, SemanticTokensOptions, SemanticTokensParams,
    SemanticTokensResult, SemanticTokensServerCapabilities, ServerCapabilities, ServerInfo,
    SignatureHelp, SignatureHelpOptions, SignatureHelpParams, TextDocumentContentChangeEvent,
    TextDocumentSyncCapability, TextDocumentSyncKind, TextEdit, Url,
    WorkspaceFoldersServerCapabilities, WorkspaceServerCapabilities, WorkspaceSymbolParams,
};
use tower_lsp::{async_trait, Client, LanguageServer};
use serde_json::json;
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
        let compat = {
            let state = self.state.read().await;
            state.compat_mode
        };
        let analysis = {
            let mut state = self.state.write().await;
            if let Some(doc) = state.documents.get_mut(uri) {
                let analysis = analyze_document_with_compat(&doc.text, compat);
                doc.analysis = Some(analysis.clone());
                analysis
            } else {
                return;
            }
        };

        let status_payload = json!({
            "message": analysis.status_message(),
        });
        let _ = self
            .client
            .send_notification::<RunmatStatusNotification>(status_payload)
            .await;

        self.publish_diagnostics(uri, &analysis).await;
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

        if let Some(compat) = resolved_compat {
            let mut state = self.state.write().await;
            state.compat_mode = compat;
        }

        let capabilities = ServerCapabilities {
            text_document_sync: Some(TextDocumentSyncCapability::Kind(
                TextDocumentSyncKind::INCREMENTAL,
            )),
            hover_provider: Some(HoverProviderCapability::Simple(true)),
            definition_provider: Some(OneOf::Left(true)),
            signature_help_provider: Some(SignatureHelpOptions::default()),
            semantic_tokens_provider: Some(SemanticTokensServerCapabilities::SemanticTokensOptions(
                SemanticTokensOptions {
                    legend: semantic_tokens_legend(),
                    range: None,
                    full: Some(tower_lsp::lsp_types::SemanticTokensFullOptions::Bool(true)),
                    work_done_progress_options: Default::default(),
                },
            )),
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
            &text,
            &analysis,
            &position,
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

        let ranges = definition_at(&text, &analysis, &position);
        if ranges.is_empty() {
            Ok(None)
        } else {
            let locs: Vec<Location> = ranges
                .into_iter()
                .map(|range| Location {
                    uri: uri.clone(),
                    range,
                })
                .collect();
            Ok(Some(GotoDefinitionResponse::Array(locs)))
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
        let docs = {
            let state = self.state.read().await;
            state
                .documents
        .iter()
                .filter_map(|(uri, doc)| {
                    doc.analysis
                .as_ref()
                        .map(|analysis| (uri.clone(), doc.text.clone(), analysis.clone()))
        })
        .collect::<Vec<_>>()
        };
        Ok(Some(workspace_symbols(&docs)))
    }

    async fn did_change_configuration(&self, params: DidChangeConfigurationParams) {
        debug!("Configuration updated: {:?}", params.settings);
        self.client
            .log_message(MessageType::INFO, "RunMat configuration updated")
            .await;
    }
}

fn parse_compat_mode(opts: &serde_json::Value) -> Option<CompatMode> {
    let lang = opts.get("language")?;
    let compat = lang.get("compat")?;
    parse_compat_str(compat.as_str()?)
}

fn parse_compat_str(value: &str) -> Option<CompatMode> {
    if value.eq_ignore_ascii_case("matlab") {
        Some(CompatMode::Matlab)
    } else if value.eq_ignore_ascii_case("strict") {
        Some(CompatMode::Strict)
    } else {
        None
    }
}

fn compat_mode_from_workspace(params: &InitializeParams) -> Option<CompatMode> {
    for root in workspace_roots(params) {
        if let Some(mode) = find_configured_compat(&root) {
            return Some(mode);
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

fn find_configured_compat(start: &Path) -> Option<CompatMode> {
    let mut current = start.to_path_buf();
    loop {
        if let Some((mode, path)) = read_compat_in_dir(&current) {
            info!(
                "Language compatibility set to '{}' via {}",
                compat_label(mode),
                path.display()
            );
            return Some(mode);
        }
        if !current.pop() {
            break;
        }
    }
    None
}

fn compat_label(mode: CompatMode) -> &'static str {
    match mode {
        CompatMode::Matlab => "matlab",
        CompatMode::Strict => "strict",
    }
}

const CONFIG_CANDIDATES: &[( &str, ConfigFormat)] = &[
    (".runmat", ConfigFormat::Toml),
    (".runmat.toml", ConfigFormat::Toml),
    (".runmat.yaml", ConfigFormat::Yaml),
    (".runmat.yml", ConfigFormat::Yaml),
    (".runmat.json", ConfigFormat::Json),
    ("runmat.config.toml", ConfigFormat::Toml),
    ("runmat.config.yaml", ConfigFormat::Yaml),
    ("runmat.config.yml", ConfigFormat::Yaml),
    ("runmat.config.json", ConfigFormat::Json),
];

fn read_compat_in_dir(dir: &Path) -> Option<(CompatMode, PathBuf)> {
    for (name, format) in CONFIG_CANDIDATES {
        let path = dir.join(name);
        if let Some(mode) = read_config_file(&path, *format) {
            return Some((mode, path));
        }
    }
    None
}

fn read_config_file(path: &Path, format: ConfigFormat) -> Option<CompatMode> {
    if !path.is_file() {
        return None;
    }
    let contents = fs::read_to_string(path).ok()?;
    parse_config_contents(&contents, format)
}

fn parse_config_contents(contents: &str, format: ConfigFormat) -> Option<CompatMode> {
    let cfg: PartialLanguageConfig = match format {
        ConfigFormat::Toml => toml::from_str(contents).ok()?,
        ConfigFormat::Yaml => serde_yaml::from_str(contents).ok()?,
        ConfigFormat::Json => serde_json::from_str(contents).ok()?,
    };
    let language = cfg.language?;
    language
        .compat
        .as_deref()
        .and_then(parse_compat_str)
}

#[derive(Clone, Copy)]
enum ConfigFormat {
    Toml,
    Yaml,
    Json,
}

#[derive(Deserialize)]
struct PartialLanguageConfig {
    language: Option<LanguageSection>,
}

#[derive(Deserialize)]
struct LanguageSection {
    compat: Option<String>,
}