#![cfg(feature = "native")]

use std::collections::HashMap;
use std::sync::Arc;

use log::{debug, info, warn};
use crate::core::analysis::{
    analyze_document_with_compat, completion_at, definition_at, diagnostics_for_document,
    document_symbols, formatting_edits, hover_at, semantic_tokens_full, semantic_tokens_legend,
    signature_help_at, CompatMode, DocumentAnalysis,
};
use crate::core::workspace::workspace_symbols;
use crate::core::position::position_to_offset;
use crate::core::fusion::fusion_plan_public_from_snapshot;
use runmat_core::{FusionPlanEdge, FusionPlanNode, FusionPlanSnapshot, RunMatSession};
use tokio::sync::RwLock;
use tower_lsp::jsonrpc::Result as RpcResult;
use tower_lsp::lsp_types::notification::Notification;
use tower_lsp::lsp_types::{
    CompletionOptions, CompletionParams, CompletionResponse, DidChangeConfigurationParams,
    DidChangeTextDocumentParams, DidCloseTextDocumentParams, DidOpenTextDocumentParams,
    DidSaveTextDocumentParams, DocumentSymbolParams, DocumentSymbolResponse,
    ExecuteCommandOptions, ExecuteCommandParams, GotoDefinitionParams, GotoDefinitionResponse,
    Hover, HoverParams, HoverProviderCapability, InitializeParams, InitializeResult,
    InitializedParams, Location, MessageType, OneOf, PositionEncodingKind, SemanticTokensOptions,
    SemanticTokensParams, SemanticTokensResult, SemanticTokensServerCapabilities,
    ServerCapabilities, ServerInfo, SignatureHelp, SignatureHelpOptions, SignatureHelpParams,
    TextDocumentContentChangeEvent, TextDocumentSyncCapability, TextDocumentSyncKind, TextEdit,
    WorkspaceSymbolParams,
    Url, WorkspaceFoldersServerCapabilities, WorkspaceServerCapabilities, DocumentFormattingParams,
};
use tower_lsp::{async_trait, Client, LanguageServer};
use serde_json::{json, Value};
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
    fusion_session: Option<runmat_core::RunMatSession>,
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
                fusion_session: None,
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

        let status_payload = serde_json::json!({
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

        if let Some(compat) = params
            .initialization_options
            .as_ref()
            .and_then(parse_compat_mode)
        {
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
            execute_command_provider: Some(ExecuteCommandOptions {
                commands: vec!["runmat/fusionPlan".to_string()],
                work_done_progress_options: Default::default(),
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

    async fn execute_command(&self, params: ExecuteCommandParams) -> RpcResult<Option<Value>> {
        if params.command != "runmat/fusionPlan" {
            return Ok(None);
        }

        let Some(uri) = extract_uri_from_args(&params.arguments) else {
            return Ok(Some(json!({
                "status": "unavailable",
                "reason": "missing uri"
            })));
        };

        let (analysis, fusion_plan_result) = {
            let mut state = self.state.write().await;
            let Some(doc) = state.documents.get(&uri).cloned() else {
                return Ok(Some(json!({
                    "status": "unavailable",
                    "reason": "no document available"
                })));
            };
            if state.fusion_session.is_none() {
                state.fusion_session = RunMatSession::new().ok();
            }
            let session = match state.fusion_session.as_mut() {
                Some(session) => session,
                None => {
                    return Ok(Some(json!({
                        "status": "unavailable",
                        "reason": "fusion session init failed"
                    })));
                }
            };
            let plan = session.compile_fusion_plan(&doc.text);
            (doc.analysis, plan)
        };

        match fusion_plan_result {
            Ok(Some(plan)) => {
                let public = fusion_plan_public_from_snapshot(
                    plan,
                    Some("Fusion snapshot from runtime compile (no execution)".into()),
                );
                return Ok(Some(json!({
                    "status": "ok",
                    "plan": public
                })));
            }
            Ok(None) => {
                // Fall through to static analysis fallback below.
            }
            Err(err) => {
                warn!("Failed to compile fusion plan via runtime pipeline: {err}");
            }
        }

        let Some(analysis) = analysis else {
            return Ok(Some(json!({
                "status": "unavailable",
                "reason": "no analysis available"
            })));
        };

        let plan = fusion_plan_from_analysis(&analysis);
        let public = fusion_plan_public_from_snapshot(
            FusionPlanSnapshot {
                nodes: plan.nodes,
                edges: plan.edges,
                shaders: Vec::new(),
                decisions: Vec::new(),
            },
            Some("Static fusion preview from semantic analysis".into()),
        );

        Ok(Some(json!({
            "status": "ok",
            "plan": public
        })))
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

fn extract_uri_from_args(list: &[Value]) -> Option<Url> {
    for val in list {
        if let Some(s) = val.as_str() {
            if let Ok(uri) = Url::parse(s) {
                return Some(uri);
            }
        }
        if let Some(obj) = val.as_object() {
            if let Some(s) = obj.get("uri").and_then(|v| v.as_str()) {
                if let Ok(uri) = Url::parse(s) {
                    return Some(uri);
                }
            }
            if let Some(s) = obj
                .get("textDocument")
                .and_then(|td| td.get("uri"))
                .and_then(|v| v.as_str())
            {
                if let Ok(uri) = Url::parse(s) {
                    return Some(uri);
                }
            }
        }
    }
    None
}

fn fusion_plan_from_analysis(analysis: &DocumentAnalysis) -> FusionPlanSnapshot {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    if let Some(semantic) = &analysis.semantic {
        for func in &semantic.functions {
            nodes.push(FusionPlanNode {
                id: func.name.clone(),
                kind: "function".to_string(),
                label: func.signature.display(),
                shape: Vec::new(),
                residency: None,
            });
        }

        // Simple heuristic: connect functions in declaration order
        for win in semantic.functions.windows(2) {
            if let [a, b] = win {
                edges.push(FusionPlanEdge {
                    from: a.name.clone(),
                    to: b.name.clone(),
                    reason: Some("static order".to_string()),
                });
            }
        }
    }

    FusionPlanSnapshot {
        nodes,
        edges,
        shaders: Vec::new(),
        decisions: Vec::new(),
    }
}

fn parse_compat_mode(opts: &serde_json::Value) -> Option<CompatMode> {
    let lang = opts.get("language")?;
    let compat = lang.get("compat")?;
    let s = compat.as_str()?;
    match s {
        "matlab" | "MATLAB" => Some(CompatMode::Matlab),
        "strict" | "STRICT" => Some(CompatMode::Strict),
        _ => None,
    }
}