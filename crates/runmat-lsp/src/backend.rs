use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Write as FmtWrite;
use std::sync::Arc;

use log::{debug, info};
use once_cell::sync::Lazy;
use runmat_builtins::{self, AccelTag, BuiltinDoc, BuiltinFunction, Type};
use runmat_hir::{
    self, HirClassMember, HirExpr, HirExprKind, HirLValue, HirStmt, LoweringResult, VarId,
};
use runmat_lexer::{tokenize_detailed, SpannedToken, Token};
use runmat_parser::parse;
use tokio::sync::RwLock;
use tower_lsp::jsonrpc::Result as RpcResult;
use tower_lsp::lsp_types::notification::Notification;
use tower_lsp::lsp_types::{
    CompletionItem, CompletionItemKind, CompletionOptions, CompletionParams, CompletionResponse,
    Diagnostic, DiagnosticSeverity, DidChangeConfigurationParams, DidChangeTextDocumentParams,
    DidCloseTextDocumentParams, DidOpenTextDocumentParams, DidSaveTextDocumentParams,
    DocumentSymbol, DocumentSymbolParams, DocumentSymbolResponse, Documentation, Hover,
    HoverContents, HoverParams, HoverProviderCapability, InitializeParams, InitializeResult,
    InitializedParams, MarkupContent, MarkupKind, MessageType, OneOf, Position,
    PositionEncodingKind, Range, ServerCapabilities, ServerInfo, SymbolKind,
    TextDocumentContentChangeEvent, TextDocumentSyncCapability, TextDocumentSyncKind, Url,
    WorkspaceFoldersServerCapabilities, WorkspaceServerCapabilities,
};
use tower_lsp::{async_trait, Client, LanguageServer};

const RUNMAT_DOC_BASE_URL: &str = "https://runmat.dev/docs/builtins/";
// Cargo substitutes this at compile time so we can surface the precise build version in logs.
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

static BUILTIN_INDEX: Lazy<BuiltinIndex> = Lazy::new(BuiltinIndex::new);

struct RunmatStatusNotification;

impl Notification for RunmatStatusNotification {
    type Params = serde_json::Value;
    const METHOD: &'static str = "runmat/status";
}

#[derive(Clone)]
struct BuiltinEntry {
    function: &'static BuiltinFunction,
    doc: Option<&'static BuiltinDoc>,
}

#[derive(Clone)]
struct ConstantEntry {
    name: &'static str,
    value_type: Type,
}

#[derive(Default)]
struct BuiltinIndex {
    functions: HashMap<String, BuiltinEntry>,
    constants: HashMap<String, ConstantEntry>,
}

impl BuiltinIndex {
    fn new() -> Self {
        let mut functions = HashMap::new();
        let mut constants = HashMap::new();

        let doc_map: HashMap<&'static str, &'static BuiltinDoc> = runmat_builtins::builtin_docs()
            .into_iter()
            .map(|doc| (doc.name, doc))
            .collect();

        for func in runmat_builtins::builtin_functions() {
            let entry = BuiltinEntry {
                function: func,
                doc: doc_map.get(func.name).copied(),
            };
            functions.insert(func.name.to_string(), entry);
        }

        for constant in runmat_builtins::constants() {
            constants.insert(
                constant.name.to_string(),
                ConstantEntry {
                    name: constant.name,
                    value_type: Type::from_value(&constant.value),
                },
            );
        }

        BuiltinIndex {
            functions,
            constants,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct TextRange {
    start: usize,
    end: usize,
}

impl TextRange {
    fn contains(&self, offset: usize) -> bool {
        self.start <= offset && offset < self.end
    }

    fn to_lsp_range(&self, text: &str) -> Range {
        Range {
            start: offset_to_position(text, self.start),
            end: offset_to_position(text, self.end),
        }
    }
}

#[derive(Clone, Debug)]
struct FunctionSignature {
    name: String,
    outputs: Vec<String>,
    inputs: Vec<String>,
    name_range: TextRange,
}

impl FunctionSignature {
    fn display(&self) -> String {
        let mut buf = String::new();
        if !self.outputs.is_empty() {
            if self.outputs.len() == 1 {
                let _ = write!(buf, "{} = ", self.outputs[0]);
            } else {
                let _ = write!(buf, "[{}] = ", self.outputs.join(", "));
            }
        }
        let _ = write!(buf, "{}", self.name);
        let args = self.inputs.join(", ");
        let _ = write!(buf, "({args})");
        buf
    }
}

#[derive(Clone, Debug)]
struct FunctionBlock {
    signature: FunctionSignature,
    span: TextRange,
}

#[derive(Clone)]
struct VariableSymbol {
    name: String,
    ty: Type,
    kind: VariableKind,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum VariableKind {
    Global,
    Parameter,
    Output,
    Local,
}

impl VariableKind {
    fn as_label(&self) -> &'static str {
        match self {
            VariableKind::Global => "global",
            VariableKind::Parameter => "parameter",
            VariableKind::Output => "output",
            VariableKind::Local => "local",
        }
    }
}

#[derive(Clone)]
struct FunctionSemantic {
    name: String,
    signature: FunctionSignature,
    range: TextRange,
    selection: TextRange,
    variables: HashMap<String, VariableSymbol>,
    return_types: Vec<Type>,
}

impl FunctionSemantic {
    fn variable(&self, name: &str) -> Option<&VariableSymbol> {
        self.variables.get(name)
    }
}

#[derive(Clone)]
struct SemanticModel {
    globals: HashMap<String, VariableSymbol>,
    functions: Vec<FunctionSemantic>,
    function_lookup: HashMap<String, Vec<usize>>,
    status_message: String,
}

impl SemanticModel {
    fn function_by_name(&self, name: &str) -> Option<&FunctionSemantic> {
        self.function_lookup
            .get(name)
            .and_then(|indices| indices.first().copied())
            .and_then(|idx| self.functions.get(idx))
    }

    fn function_at_offset(&self, offset: usize) -> Option<&FunctionSemantic> {
        let mut best: Option<&FunctionSemantic> = None;
        for func in &self.functions {
            if func.range.contains(offset) {
                match best {
                    Some(current)
                        if (current.range.end - current.range.start)
                            <= (func.range.end - func.range.start) => {}
                    _ => best = Some(func),
                }
            }
        }
        best
    }
}

#[derive(Clone)]
struct DocumentAnalysis {
    tokens: Vec<SpannedToken>,
    parse_error: Option<ParseErrorInfo>,
    lowering_error: Option<String>,
    semantic: Option<SemanticModel>,
}

impl DocumentAnalysis {
    fn status_message(&self) -> String {
        if let Some(err) = &self.parse_error {
            format!("RunMat: parse error near position {}", err.position)
        } else if let Some(err) = &self.lowering_error {
            format!("RunMat: analysis error ({err})")
        } else if let Some(sem) = &self.semantic {
            sem.status_message.clone()
        } else {
            "RunMat: ready".to_string()
        }
    }
}

#[derive(Clone)]
struct ParseErrorInfo {
    message: String,
    position: usize,
}

#[derive(Clone, Default)]
struct DocumentState {
    text: String,
    version: Option<i32>,
    analysis: Option<DocumentAnalysis>,
}

#[derive(Default)]
struct AnalyzerState {
    documents: HashMap<Url, DocumentState>,
}

pub struct RunMatLanguageServer {
    client: Client,
    state: Arc<RwLock<AnalyzerState>>,
}

impl RunMatLanguageServer {
    pub fn new(client: Client) -> Self {
        Self {
            client,
            state: Arc::new(RwLock::new(AnalyzerState::default())),
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
        let analysis = {
            let mut state = self.state.write().await;
            if let Some(doc) = state.documents.get_mut(uri) {
                let analysis = analyze_document(&doc.text);
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

        let mut diagnostics = Vec::new();

        if let Some(parse_err) = &analysis.parse_error {
            let range = diagnostic_range_for_parse_error(&analysis.tokens, &text, parse_err);
            diagnostics.push(Diagnostic {
                range,
                severity: Some(DiagnosticSeverity::ERROR),
                code: None,
                code_description: None,
                source: Some("runmat-parser".into()),
                message: parse_err.message.clone(),
                related_information: None,
                tags: None,
                data: None,
            });
        }

        if let Some(err) = &analysis.lowering_error {
            diagnostics.push(diagnostic_for_lowering_error(err, &analysis.tokens, &text));
        }

        if let Some(semantic) = &analysis.semantic {
            diagnostics.extend(collect_unknown_type_diagnostics(
                semantic,
                &analysis.tokens,
                &text,
            ));
        }

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
    async fn initialize(&self, _: InitializeParams) -> RpcResult<InitializeResult> {
        info!("Initializing RunMat language server v{}", SERVER_VERSION);
        let server_info = Some(ServerInfo {
            name: "RunMat Language Server".to_string(),
            version: Some(SERVER_VERSION.to_string()),
        });

        let capabilities = ServerCapabilities {
            text_document_sync: Some(TextDocumentSyncCapability::Kind(
                TextDocumentSyncKind::INCREMENTAL,
            )),
            hover_provider: Some(HoverProviderCapability::Simple(true)),
            completion_provider: Some(CompletionOptions {
                resolve_provider: Some(false),
                trigger_characters: None,
                all_commit_characters: None,
                completion_item: None,
                work_done_progress_options: Default::default(),
            }),
            document_symbol_provider: Some(OneOf::Left(true)),
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

        let offset = position_to_offset(&text, &position);
        let token = token_at_offset(&analysis.tokens, offset);
        let Some(token) = token else {
            return Ok(None);
        };

        if !matches!(token.token, Token::Ident) {
            return Ok(None);
        }

        let ident = token.lexeme.clone();

        if let Some(semantic) = &analysis.semantic {
            if let Some(func) = semantic.function_at_offset(offset) {
                if let Some(var) = func.variable(&ident) {
                    let hover = Hover {
                        contents: HoverContents::Markup(MarkupContent {
                            kind: MarkupKind::Markdown,
                            value: format_variable_hover(&ident, var),
                        }),
                        range: Some(
                            TextRange {
                                start: token.start,
                                end: token.end,
                            }
                            .to_lsp_range(&text),
                        ),
                    };
                    return Ok(Some(hover));
                }
            }

            if let Some(global) = semantic.globals.get(&ident) {
                let hover = Hover {
                    contents: HoverContents::Markup(MarkupContent {
                        kind: MarkupKind::Markdown,
                        value: format_variable_hover(&ident, global),
                    }),
                    range: Some(
                        TextRange {
                            start: token.start,
                            end: token.end,
                        }
                        .to_lsp_range(&text),
                    ),
                };
                return Ok(Some(hover));
            }

            if let Some(func) = semantic.function_by_name(&ident) {
                let hover = Hover {
                    contents: HoverContents::Markup(MarkupContent {
                        kind: MarkupKind::Markdown,
                        value: format_function_hover(func),
                    }),
                    range: Some(
                        TextRange {
                            start: token.start,
                            end: token.end,
                        }
                        .to_lsp_range(&text),
                    ),
                };
                return Ok(Some(hover));
            }
        }

        if let Some(entry) = BUILTIN_INDEX.functions.get(&ident) {
            let hover = Hover {
                contents: HoverContents::Markup(MarkupContent {
                    kind: MarkupKind::Markdown,
                    value: format_builtin_hover(entry),
                }),
                range: Some(
                    TextRange {
                        start: token.start,
                        end: token.end,
                    }
                    .to_lsp_range(&text),
                ),
            };
            return Ok(Some(hover));
        }

        if let Some(constant) = BUILTIN_INDEX.constants.get(&ident) {
            let hover = Hover {
                contents: HoverContents::Markup(MarkupContent {
                    kind: MarkupKind::Markdown,
                    value: format_constant_hover(constant),
                }),
                range: Some(
                    TextRange {
                        start: token.start,
                        end: token.end,
                    }
                    .to_lsp_range(&text),
                ),
            };
            return Ok(Some(hover));
        }

        Ok(None)
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

        let offset = position_to_offset(&text, &position);

        let mut items = Vec::new();
        let mut seen = HashSet::new();

        if let Some(semantic) = &analysis.semantic {
            if let Some(func) = semantic.function_at_offset(offset) {
                for symbol in func.variables.values() {
                    if seen.insert(symbol.name.clone()) {
                        items.push(variable_completion(symbol));
                    }
                }
            }

            for symbol in semantic.globals.values() {
                if seen.insert(symbol.name.clone()) {
                    items.push(variable_completion(symbol));
                }
            }

            for func in &semantic.functions {
                if seen.insert(func.name.clone()) {
                    items.push(function_completion(func));
                }
            }
        }

        for (name, entry) in &BUILTIN_INDEX.functions {
            if seen.insert(name.clone()) {
                items.push(builtin_completion(name, entry));
            }
        }

        for (name, constant) in &BUILTIN_INDEX.constants {
            if seen.insert(name.clone()) {
                items.push(constant_completion(name, constant));
            }
        }

        Ok(Some(CompletionResponse::Array(items)))
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

        let mut symbols = Vec::new();
        if let Some(semantic) = &analysis.semantic {
            for func in &semantic.functions {
                let range = func.range.to_lsp_range(&text);
                let selection = func.selection.to_lsp_range(&text);
                let detail = func.signature.display();
                #[allow(deprecated)]
                symbols.push(DocumentSymbol {
                    name: func.name.clone(),
                    detail: Some(detail),
                    kind: SymbolKind::FUNCTION,
                    range,
                    selection_range: selection,
                    children: None,
                    tags: None,
                    deprecated: None,
                });
            }
        }

        Ok(Some(DocumentSymbolResponse::Nested(symbols)))
    }

    async fn did_change_configuration(&self, params: DidChangeConfigurationParams) {
        debug!("Configuration updated: {:?}", params.settings);
        self.client
            .log_message(MessageType::INFO, "RunMat configuration updated")
            .await;
    }
}

fn analyze_document(text: &str) -> DocumentAnalysis {
    let tokens = tokenize_detailed(text);
    match parse(text) {
        Ok(ast) => {
            let lowering_result =
                match runmat_hir::lower_with_full_context(&ast, &HashMap::new(), &HashMap::new()) {
                    Ok(result) => result,
                    Err(err) => {
                        return DocumentAnalysis {
                            tokens,
                            parse_error: None,
                            lowering_error: Some(err),
                            semantic: None,
                        };
                    }
                };

            let lowering_arc = Arc::new(lowering_result);
            let semantic = build_semantic_model(lowering_arc, &tokens, text);

            DocumentAnalysis {
                tokens,
                parse_error: None,
                lowering_error: None,
                semantic: Some(semantic),
            }
        }
        Err(err) => {
            let mut message = err.message.clone();
            if let Some(expected) = &err.expected {
                message = format!("{message} (expected {expected})");
            }
            if let Some(found) = &err.found_token {
                message = format!("{message} (found '{found}')");
            }

            DocumentAnalysis {
                tokens,
                parse_error: Some(ParseErrorInfo {
                    message,
                    position: err.position,
                }),
                lowering_error: None,
                semantic: None,
            }
        }
    }
}

fn build_semantic_model(
    lowering: Arc<LoweringResult>,
    tokens: &[SpannedToken],
    text: &str,
) -> SemanticModel {
    let function_blocks = collect_function_blocks(tokens, text.len());
    let mut block_lookup: HashMap<String, VecDeque<FunctionBlock>> = HashMap::new();
    for block in function_blocks {
        block_lookup
            .entry(block.signature.name.clone())
            .or_default()
            .push_back(block);
    }

    let function_returns = runmat_hir::infer_function_output_types(&lowering.hir);

    let mut functions = Vec::new();
    let mut function_lookup: HashMap<String, Vec<usize>> = HashMap::new();

    let mut to_visit = Vec::new();
    collect_function_statements(&lowering.hir.body, &mut to_visit);

    for func_stmt in to_visit {
        if let HirStmt::Function {
            name,
            params,
            outputs,
            body,
            ..
        } = func_stmt
        {
            let block = block_lookup
                .get_mut(name)
                .and_then(|q| q.pop_front())
                .unwrap_or_else(|| fallback_block(name, tokens, text.len()));

            let mut variables = HashMap::new();
            let mut seen_ids = HashSet::new();

            for &param in params {
                if let Some(symbol) = make_variable_symbol(
                    param,
                    VariableKind::Parameter,
                    &lowering.var_types,
                    &lowering.var_names,
                ) {
                    seen_ids.insert(param);
                    variables.insert(symbol.name.clone(), symbol);
                }
            }

            for &out in outputs {
                if let Some(symbol) = make_variable_symbol(
                    out,
                    VariableKind::Output,
                    &lowering.var_types,
                    &lowering.var_names,
                ) {
                    seen_ids.insert(out);
                    variables.insert(symbol.name.clone(), symbol);
                }
            }

            let mut local_ids = HashSet::new();
            collect_local_var_ids(body, &mut local_ids);
            for var_id in local_ids {
                if !seen_ids.insert(var_id) {
                    continue;
                }
                if let Some(symbol) = make_variable_symbol(
                    var_id,
                    VariableKind::Local,
                    &lowering.var_types,
                    &lowering.var_names,
                ) {
                    variables.insert(symbol.name.clone(), symbol);
                }
            }

            let selection = block.signature.name_range;
            let mut function_semantic = FunctionSemantic {
                name: block.signature.name.clone(),
                signature: block.signature,
                range: block.span,
                selection,
                variables,
                return_types: function_returns.get(name).cloned().unwrap_or_default(),
            };

            // Merge duplicate variable entries by keeping the most specific type.
            let mut dedup = HashMap::new();
            for (name, symbol) in std::mem::take(&mut function_semantic.variables) {
                dedup
                    .entry(name.clone())
                    .and_modify(|existing: &mut VariableSymbol| {
                        existing.ty = existing.ty.unify(&symbol.ty);
                    })
                    .or_insert(symbol);
            }
            function_semantic.variables = dedup;

            let idx = functions.len();
            functions.push(function_semantic);
            function_lookup.entry(name.clone()).or_default().push(idx);
        }
    }

    let mut globals = HashMap::new();
    for (name, &index) in &lowering.variables {
        let var_id = VarId(index);
        if let Some(symbol) = make_variable_symbol(
            var_id,
            VariableKind::Global,
            &lowering.var_types,
            &lowering.var_names,
        ) {
            globals.insert(name.clone(), symbol);
        }
    }

    let mut typed_count = 0usize;
    let mut total_count = 0usize;
    for symbol in globals.values() {
        total_count += 1;
        if !matches!(symbol.ty, Type::Unknown) {
            typed_count += 1;
        }
    }
    for func in &functions {
        for symbol in func.variables.values() {
            total_count += 1;
            if !matches!(symbol.ty, Type::Unknown) {
                typed_count += 1;
            }
        }
    }

    let status_message = if total_count == 0 {
        format!(
            "RunMat: {} functions • {} globals",
            functions.len(),
            globals.len()
        )
    } else {
        format!(
            "RunMat: {} functions • {} globals • {}% typed",
            functions.len(),
            globals.len(),
            (typed_count * 100) / total_count
        )
    };

    SemanticModel {
        globals,
        functions,
        function_lookup,
        status_message,
    }
}

fn collect_unknown_type_diagnostics(
    semantic: &SemanticModel,
    tokens: &[SpannedToken],
    text: &str,
) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();

    for symbol in semantic.globals.values() {
        if matches!(symbol.ty, Type::Unknown) {
            if let Some(range) = find_symbol_range(tokens, &symbol.name, None) {
                diagnostics.push(Diagnostic {
                    range: range.to_lsp_range(text),
                    severity: Some(DiagnosticSeverity::HINT),
                    code: None,
                    code_description: None,
                    source: Some("runmat-typing".into()),
                    message: format!(
                        "Type of global '{}' could not be inferred (falls back to 'unknown').",
                        symbol.name
                    ),
                    related_information: None,
                    tags: None,
                    data: None,
                });
            }
        }
    }

    for func in &semantic.functions {
        let scope = Some(func.range);
        for symbol in func.variables.values() {
            if matches!(symbol.ty, Type::Unknown) {
                if let Some(range) = find_symbol_range(tokens, &symbol.name, scope.as_ref()) {
                    diagnostics.push(Diagnostic {
                        range: range.to_lsp_range(text),
                        severity: Some(DiagnosticSeverity::HINT),
                        code: None,
                        code_description: None,
                        source: Some("runmat-typing".into()),
                        message: format!(
                            "Type of {} '{}' in function '{}' remains unknown.",
                            symbol.kind.as_label(),
                            symbol.name,
                            func.name
                        ),
                        related_information: None,
                        tags: None,
                        data: None,
                    });
                }
            }
        }
    }

    diagnostics
}

fn diagnostic_range_for_parse_error(
    tokens: &[SpannedToken],
    text: &str,
    error: &ParseErrorInfo,
) -> Range {
    if let Some(token) = tokens
        .iter()
        .find(|t| t.start <= error.position && error.position <= t.end)
    {
        TextRange {
            start: token.start,
            end: token.end,
        }
        .to_lsp_range(text)
    } else {
        Range {
            start: offset_to_position(text, error.position),
            end: offset_to_position(text, error.position + 1),
        }
    }
}

fn diagnostic_for_lowering_error(error: &str, tokens: &[SpannedToken], text: &str) -> Diagnostic {
    let message = error.to_string();
    let undefined_var = error
        .split(':')
        .last()
        .map(str::trim)
        .and_then(|s| s.split_whitespace().last())
        .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric() && c != '_'));

    let range = if let Some(name) = undefined_var {
        find_symbol_range(tokens, name, None)
            .unwrap_or(TextRange { start: 0, end: 1 })
            .to_lsp_range(text)
    } else {
        Range {
            start: Position::new(0, 0),
            end: Position::new(0, 0),
        }
    };

    Diagnostic {
        range,
        severity: Some(DiagnosticSeverity::ERROR),
        code: None,
        code_description: None,
        source: Some("runmat-hir".into()),
        message,
        related_information: None,
        tags: None,
        data: None,
    }
}

fn token_at_offset(tokens: &[SpannedToken], offset: usize) -> Option<&SpannedToken> {
    if tokens.is_empty() {
        return None;
    }
    let mut left = 0usize;
    let mut right = tokens.len();
    while left < right {
        let mid = (left + right) / 2;
        let token = &tokens[mid];
        if offset < token.start {
            right = mid;
        } else if offset >= token.end {
            left = mid + 1;
        } else {
            return Some(token);
        }
    }
    None
}

fn find_symbol_range(
    tokens: &[SpannedToken],
    name: &str,
    scope: Option<&TextRange>,
) -> Option<TextRange> {
    tokens
        .iter()
        .filter(|tok| matches!(tok.token, Token::Ident))
        .filter(|tok| tok.lexeme == name)
        .map(|tok| TextRange {
            start: tok.start,
            end: tok.end,
        })
        .find(|range| scope.map_or(true, |scope| scope.contains(range.start)))
}

fn position_to_offset(text: &str, position: &Position) -> usize {
    let mut offset = 0usize;
    let mut current_line = 0u32;
    for line in text.split_inclusive('\n') {
        if current_line == position.line {
            let line_len = line.strip_suffix('\n').unwrap_or(line).len();
            let character = position.character as usize;
            return offset + character.min(line_len);
        }
        offset += line.len();
        current_line += 1;
    }
    text.len()
}

fn offset_to_position(text: &str, mut offset: usize) -> Position {
    if offset > text.len() {
        offset = text.len();
    }

    let mut line = 0u32;
    let mut line_start = 0usize;
    for (idx, byte) in text.bytes().enumerate() {
        if idx == offset {
            break;
        }
        if byte == b'\n' {
            line += 1;
            line_start = idx + 1;
        }
    }

    Position::new(line, (offset - line_start) as u32)
}

fn format_variable_hover(name: &str, symbol: &VariableSymbol) -> String {
    let mut buf = String::new();
    let _ = writeln!(
        buf,
        "```runmat\n{kind} {name}: {ty}\n```",
        kind = symbol.kind.as_label(),
        ty = format_type(&symbol.ty)
    );
    if matches!(symbol.kind, VariableKind::Global) {
        let _ = writeln!(buf, "Global variable available across the workspace.");
    }
    buf
}

fn format_function_hover(func: &FunctionSemantic) -> String {
    let mut buf = String::new();
    let _ = writeln!(buf, "```runmat\nfunction {}\n```", func.signature.display());
    if !func.return_types.is_empty() {
        let types: Vec<String> = func.return_types.iter().map(format_type).collect();
        let _ = writeln!(buf, "Returns: {}", types.join(", "));
    }
    buf
}

fn format_builtin_hover(entry: &BuiltinEntry) -> String {
    let func = entry.function;
    let mut buf = String::new();
    let params: Vec<String> = func.param_types.iter().map(format_type).collect();
    let _ = writeln!(
        buf,
        "```runmat\n{ret} = {name}({args})\n```",
        ret = format_type(&func.return_type),
        name = func.name,
        args = params.join(", ")
    );
    if !func.description.is_empty() {
        let _ = writeln!(buf, "{}", func.description.trim());
    }
    if let Some(doc) = entry.doc {
        if let Some(summary) = doc.summary {
            let _ = writeln!(buf, "\n_{summary}_");
        }
    }
    if !func.accel_tags.is_empty() {
        let _ = writeln!(
            buf,
            "\n**GPU/Fusion:** {}",
            format_accel_tags(func.accel_tags)
        );
    }
    let _ = writeln!(
        buf,
        "\n[Official documentation]({base}{name})",
        base = RUNMAT_DOC_BASE_URL,
        name = func.name
    );
    buf
}

fn format_constant_hover(constant: &ConstantEntry) -> String {
    format!(
        "```runmat\nconstant {name}: {ty}\n```",
        name = constant.name,
        ty = format_type(&constant.value_type)
    )
}

fn variable_completion(symbol: &VariableSymbol) -> CompletionItem {
    CompletionItem {
        label: symbol.name.clone(),
        kind: Some(CompletionItemKind::VARIABLE),
        detail: Some(format!(
            "{} ({})",
            symbol.kind.as_label(),
            format_type(&symbol.ty)
        )),
        ..Default::default()
    }
}

fn function_completion(func: &FunctionSemantic) -> CompletionItem {
    CompletionItem {
        label: func.name.clone(),
        kind: Some(CompletionItemKind::FUNCTION),
        detail: Some(func.signature.display()),
        ..Default::default()
    }
}

fn builtin_completion(name: &str, entry: &BuiltinEntry) -> CompletionItem {
    CompletionItem {
        label: name.to_string(),
        kind: Some(CompletionItemKind::FUNCTION),
        detail: Some(entry.function.description.to_string()),
        documentation: Some(Documentation::MarkupContent(MarkupContent {
            kind: MarkupKind::Markdown,
            value: format_builtin_hover(entry),
        })),
        ..Default::default()
    }
}

fn constant_completion(name: &str, entry: &ConstantEntry) -> CompletionItem {
    CompletionItem {
        label: name.to_string(),
        kind: Some(CompletionItemKind::CONSTANT),
        detail: Some(format_type(&entry.value_type)),
        ..Default::default()
    }
}

fn format_type(ty: &Type) -> String {
    match ty {
        Type::Int => "int".to_string(),
        Type::Num => "double".to_string(),
        Type::Bool => "logical".to_string(),
        Type::Logical => "logical array".to_string(),
        Type::String => "string".to_string(),
        Type::Tensor { shape } => {
            if let Some(shape) = shape {
                let dims: Vec<String> = shape
                    .iter()
                    .map(|d| d.map(|v| v.to_string()).unwrap_or_else(|| "?".into()))
                    .collect();
                format!("tensor<{}>", dims.join("x"))
            } else {
                "tensor".to_string()
            }
        }
        Type::Cell {
            element_type,
            length,
        } => {
            let element = element_type
                .as_ref()
                .map(|t| format_type(t))
                .unwrap_or_else(|| "any".into());
            if let Some(len) = length {
                format!("cell<{element}; {len}>")
            } else {
                format!("cell<{element}>")
            }
        }
        Type::Function { params, returns } => {
            let args: Vec<String> = params.iter().map(format_type).collect();
            format!("fn({}) -> {}", args.join(", "), format_type(returns))
        }
        Type::Void => "void".to_string(),
        Type::Unknown => "unknown".to_string(),
        Type::Union(types) => {
            let parts: Vec<String> = types.iter().map(format_type).collect();
            format!("union<{}>", parts.join(" | "))
        }
        Type::Struct { known_fields } => {
            if let Some(fields) = known_fields {
                format!("struct{{{}}}", fields.join(", "))
            } else {
                "struct".to_string()
            }
        }
    }
}

fn format_accel_tags(tags: &[AccelTag]) -> String {
    tags.iter()
        .map(|tag| match tag {
            AccelTag::Unary => "unary",
            AccelTag::Elementwise => "element-wise",
            AccelTag::Reduction => "reduction",
            AccelTag::MatMul => "matrix multiply",
            AccelTag::Transpose => "transpose",
            AccelTag::ArrayConstruct => "array construction",
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn collect_function_statements<'a>(stmts: &'a [HirStmt], out: &mut Vec<&'a HirStmt>) {
    for stmt in stmts {
        match stmt {
            HirStmt::Function { body, .. } => {
                out.push(stmt);
                collect_function_statements(body, out);
            }
            HirStmt::ClassDef { members, .. } => {
                for member in members {
                    if let HirClassMember::Methods { body, .. } = member {
                        collect_function_statements(body, out);
                    }
                }
            }
            _ => {}
        }
    }
}

fn collect_local_var_ids(stmts: &[HirStmt], out: &mut HashSet<VarId>) {
    for stmt in stmts {
        match stmt {
            HirStmt::Assign(var, expr, _) => {
                out.insert(*var);
                collect_expr_var_ids(expr, out);
            }
            HirStmt::AssignLValue(target, expr, _) => {
                collect_lvalue_var_ids(target, out);
                collect_expr_var_ids(expr, out);
            }
            HirStmt::MultiAssign(vars, expr, _) => {
                for var in vars.iter().flatten() {
                    out.insert(*var);
                }
                collect_expr_var_ids(expr, out);
            }
            HirStmt::If {
                cond,
                then_body,
                elseif_blocks,
                else_body,
            } => {
                collect_expr_var_ids(cond, out);
                collect_local_var_ids(then_body, out);
                for (expr, body) in elseif_blocks {
                    collect_expr_var_ids(expr, out);
                    collect_local_var_ids(body, out);
                }
                if let Some(body) = else_body {
                    collect_local_var_ids(body, out);
                }
            }
            HirStmt::While { cond, body } => {
                collect_expr_var_ids(cond, out);
                collect_local_var_ids(body, out);
            }
            HirStmt::For { var, expr, body } => {
                out.insert(*var);
                collect_expr_var_ids(expr, out);
                collect_local_var_ids(body, out);
            }
            HirStmt::Switch {
                expr,
                cases,
                otherwise,
            } => {
                collect_expr_var_ids(expr, out);
                for (case_expr, body) in cases {
                    collect_expr_var_ids(case_expr, out);
                    collect_local_var_ids(body, out);
                }
                if let Some(body) = otherwise {
                    collect_local_var_ids(body, out);
                }
            }
            HirStmt::TryCatch {
                try_body,
                catch_var,
                catch_body,
            } => {
                collect_local_var_ids(try_body, out);
                if let Some(var) = catch_var {
                    out.insert(*var);
                }
                collect_local_var_ids(catch_body, out);
            }
            HirStmt::ExprStmt(expr, _) => collect_expr_var_ids(expr, out),
            HirStmt::Function { .. } => {}
            HirStmt::ClassDef { .. }
            | HirStmt::Global(_)
            | HirStmt::Persistent(_)
            | HirStmt::Break
            | HirStmt::Continue
            | HirStmt::Return
            | HirStmt::Import { .. } => {}
        }
    }
}

fn collect_expr_var_ids(expr: &HirExpr, out: &mut HashSet<VarId>) {
    match &expr.kind {
        HirExprKind::Var(var_id) => {
            out.insert(*var_id);
        }
        HirExprKind::Unary(_, inner) => collect_expr_var_ids(inner, out),
        HirExprKind::Binary(left, _, right) => {
            collect_expr_var_ids(left, out);
            collect_expr_var_ids(right, out);
        }
        HirExprKind::Tensor(rows) | HirExprKind::Cell(rows) => {
            for row in rows {
                for expr in row {
                    collect_expr_var_ids(expr, out);
                }
            }
        }
        HirExprKind::Index(base, indices) | HirExprKind::IndexCell(base, indices) => {
            collect_expr_var_ids(base, out);
            for expr in indices {
                collect_expr_var_ids(expr, out);
            }
        }
        HirExprKind::Range(start, step, end) => {
            collect_expr_var_ids(start, out);
            if let Some(step) = step {
                collect_expr_var_ids(step, out);
            }
            collect_expr_var_ids(end, out);
        }
        HirExprKind::Member(base, _) => collect_expr_var_ids(base, out),
        HirExprKind::MemberDynamic(base, field) => {
            collect_expr_var_ids(base, out);
            collect_expr_var_ids(field, out);
        }
        HirExprKind::MethodCall(base, _, args) => {
            collect_expr_var_ids(base, out);
            for arg in args {
                collect_expr_var_ids(arg, out);
            }
        }
        HirExprKind::AnonFunc { body, .. } => collect_expr_var_ids(body, out),
        HirExprKind::FuncCall(_, args) => {
            for arg in args {
                collect_expr_var_ids(arg, out);
            }
        }
        HirExprKind::Number(_)
        | HirExprKind::String(_)
        | HirExprKind::Constant(_)
        | HirExprKind::Colon
        | HirExprKind::End
        | HirExprKind::FuncHandle(_)
        | HirExprKind::MetaClass(_) => {}
    }
}

fn collect_lvalue_var_ids(lvalue: &HirLValue, out: &mut HashSet<VarId>) {
    match lvalue {
        HirLValue::Var(var_id) => {
            out.insert(*var_id);
        }
        HirLValue::Member(base, _) => {
            collect_expr_var_ids(base, out);
        }
        HirLValue::Index(base, indices) | HirLValue::IndexCell(base, indices) => {
            collect_expr_var_ids(base, out);
            for expr in indices {
                collect_expr_var_ids(expr, out);
            }
        }
        HirLValue::MemberDynamic(base, field) => {
            collect_expr_var_ids(base, out);
            collect_expr_var_ids(field, out);
        }
    }
}

fn make_variable_symbol(
    var_id: VarId,
    kind: VariableKind,
    types: &[Type],
    names: &HashMap<VarId, String>,
) -> Option<VariableSymbol> {
    let name = names.get(&var_id)?.clone();
    let ty = types.get(var_id.0).cloned().unwrap_or(Type::Unknown);
    Some(VariableSymbol { name, ty, kind })
}

fn collect_function_blocks(tokens: &[SpannedToken], text_len: usize) -> Vec<FunctionBlock> {
    let mut blocks = Vec::new();
    let mut stack: Vec<(BlockKind, usize, FunctionSignature)> = Vec::new();

    for (idx, token) in tokens.iter().enumerate() {
        match token.token {
            Token::Function => {
                let signature = parse_function_signature(tokens, idx + 1).unwrap_or_else(|| {
                    FunctionSignature {
                        name: "anonymous".into(),
                        outputs: Vec::new(),
                        inputs: Vec::new(),
                        name_range: TextRange {
                            start: token.start,
                            end: token.end,
                        },
                    }
                });
                stack.push((BlockKind::Function, token.start, signature));
            }
            Token::If
            | Token::For
            | Token::While
            | Token::Switch
            | Token::Try
            | Token::ClassDef
            | Token::Methods
            | Token::Properties
            | Token::Events
            | Token::Enumeration
            | Token::Arguments => {
                stack.push((BlockKind::Other, token.start, empty_signature()));
            }
            Token::End => {
                if let Some((kind, start, signature)) = stack.pop() {
                    if matches!(kind, BlockKind::Function) {
                        blocks.push(FunctionBlock {
                            signature,
                            span: TextRange {
                                start,
                                end: token.end,
                            },
                        });
                    }
                }
            }
            _ => {}
        }
    }

    while let Some((kind, start, signature)) = stack.pop() {
        if matches!(kind, BlockKind::Function) {
            blocks.push(FunctionBlock {
                signature,
                span: TextRange {
                    start,
                    end: text_len,
                },
            });
        }
    }

    blocks
}

#[derive(Clone, Copy)]
enum BlockKind {
    Function,
    Other,
}

fn empty_signature() -> FunctionSignature {
    FunctionSignature {
        name: String::new(),
        outputs: Vec::new(),
        inputs: Vec::new(),
        name_range: TextRange { start: 0, end: 0 },
    }
}

fn fallback_block(name: &str, tokens: &[SpannedToken], text_len: usize) -> FunctionBlock {
    let range = find_symbol_range(tokens, name, None).unwrap_or(TextRange {
        start: 0,
        end: text_len,
    });
    FunctionBlock {
        signature: FunctionSignature {
            name: name.to_string(),
            outputs: Vec::new(),
            inputs: Vec::new(),
            name_range: range,
        },
        span: TextRange {
            start: range.start,
            end: text_len,
        },
    }
}

fn parse_function_signature(
    tokens: &[SpannedToken],
    mut index: usize,
) -> Option<FunctionSignature> {
    let mut outputs = Vec::new();
    let mut name_parts = Vec::new();
    let mut inputs = Vec::new();
    let mut name_start = None;
    let mut name_end = None;

    while index < tokens.len() && matches!(tokens[index].token, Token::Newline) {
        index += 1;
    }

    if index < tokens.len() && matches!(tokens[index].token, Token::LBracket) {
        index += 1;
        while index < tokens.len() {
            match tokens[index].token {
                Token::Ident => outputs.push(tokens[index].lexeme.clone()),
                Token::Tilde => outputs.push("~".into()),
                Token::RBracket => {
                    index += 1;
                    break;
                }
                _ => {}
            }
            index += 1;
        }
        while index < tokens.len() && matches!(tokens[index].token, Token::Newline) {
            index += 1;
        }
        if index < tokens.len() && matches!(tokens[index].token, Token::Assign) {
            index += 1;
        }
    }

    while index < tokens.len() {
        match tokens[index].token {
            Token::Ident => {
                if name_start.is_none() {
                    name_start = Some(tokens[index].start);
                }
                name_end = Some(tokens[index].end);
                name_parts.push(tokens[index].lexeme.clone());
                index += 1;
            }
            Token::Dot => {
                name_parts.push(".".into());
                name_end = Some(tokens[index].end);
                index += 1;
            }
            Token::LParen => {
                index += 1;
                break;
            }
            Token::Newline | Token::Assign => break,
            _ => {
                index += 1;
                break;
            }
        }
    }

    if name_parts.is_empty() {
        return None;
    }

    while index < tokens.len() {
        match tokens[index].token {
            Token::Ident => inputs.push(tokens[index].lexeme.clone()),
            Token::Tilde => inputs.push("~".into()),
            Token::Comma => {}
            Token::RParen => break,
            _ => {}
        }
        index += 1;
    }

    Some(FunctionSignature {
        name: name_parts.join(""),
        outputs,
        inputs,
        name_range: TextRange {
            start: name_start.unwrap_or(0),
            end: name_end.unwrap_or(name_start.unwrap_or(0)),
        },
    })
}
