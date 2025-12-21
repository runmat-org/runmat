use once_cell::sync::OnceCell;
use runmat_core::RunMatSession;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Once;
use wasm_bindgen::prelude::*;
use serde_wasm_bindgen;
use lsp_types::{CompletionList, Diagnostic, Location, DocumentSymbol, Position, Url, SemanticTokens, TextEdit, SymbolInformation};

use crate::core::analysis::{
    analyze_document_with_compat, completion_at, definition_at, diagnostics_for_document,
    document_symbols as core_document_symbols, formatting_edits, hover_at, semantic_tokens_full,
    signature_help_at, CompatMode, DocumentAnalysis,
};
use crate::core::fusion::fusion_plan_public_from_snapshot;
use crate::core::types::FusionPlanPublic;
use crate::core::workspace::workspace_symbols;

#[derive(Default)]
struct DocStore {
    docs: HashMap<String, DocEntry>,
}

#[derive(Clone)]
struct DocEntry {
    text: String,
    analysis: DocumentAnalysis,
}

thread_local! {
    static COMPAT_MODE: std::cell::Cell<CompatMode> = std::cell::Cell::new(CompatMode::Matlab);
}

#[derive(Serialize, Deserialize)]
pub struct FusionPlanResult {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan: Option<FusionPlanPublic>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

thread_local! {
    static DOCS: RefCell<DocStore> = RefCell::new(DocStore::default());
}

static SESSION: OnceCell<RunMatSession> = OnceCell::new();
static BUILTIN_REGISTRY: Once = Once::new();

fn session() -> Result<&'static RunMatSession, JsValue> {
    ensure_builtins_registered();
    SESSION.get_or_try_init(|| RunMatSession::new().map_err(|e| JsValue::from_str(&format!("{e}"))))
}

fn to_js<T: Serialize>(value: &T) -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(value).map_err(|e| JsValue::from_str(&e.to_string()))
}

fn ensure_builtins_registered() {
    BUILTIN_REGISTRY.call_once(|| {
        #[cfg(target_arch = "wasm32")]
        {
            runmat_runtime::builtins::wasm_registry::register_all();
        }
    });
}

#[wasm_bindgen]
pub fn builtin_inventory_counts() -> JsValue {
    ensure_builtins_registered();
    let funcs = runmat_builtins::builtin_functions().len();
    let docs = runmat_builtins::builtin_docs().len();
    let consts = runmat_builtins::constants().len();
    let registered = runmat_builtins::wasm_registry::is_registered();
    serde_wasm_bindgen::to_value(&(funcs, docs, consts, registered)).unwrap_or(JsValue::NULL)
}

#[wasm_bindgen]
pub fn open_document(uri: String, text: String) {
    ensure_builtins_registered();
    let compat = COMPAT_MODE.with(|c| c.get());
    let analysis = analyze_document_with_compat(&text, compat);
    DOCS.with(|d| {
        d.borrow_mut().docs.insert(uri, DocEntry { text, analysis });
    });
}

#[wasm_bindgen]
pub fn change_document(uri: String, text: String) {
    ensure_builtins_registered();
    let compat = COMPAT_MODE.with(|c| c.get());
    let analysis = analyze_document_with_compat(&text, compat);
    DOCS.with(|d| {
        d.borrow_mut().docs.insert(uri, DocEntry { text, analysis });
    });
}

#[wasm_bindgen]
pub fn close_document(uri: String) {
    DOCS.with(|d| {
        d.borrow_mut().docs.remove(&uri);
    });
}

#[wasm_bindgen]
pub fn fusion_plan(uri: String) -> Result<JsValue, JsValue> {
    ensure_builtins_registered();
    let doc = DOCS.with(|d| d.borrow().docs.get(&uri).cloned());
    let Some(doc) = doc else {
        return to_js(&FusionPlanResult {
            status: "unavailable".into(),
            plan: None,
            reason: Some("no document".into()),
            message: None,
        });
    };

    let sess = session()?;
    match sess.compile_fusion_plan(&doc.text) {
        Ok(Some(plan)) => {
            let public = fusion_plan_public_from_snapshot(
                plan,
                Some("Fusion snapshot (compile-only)".into()),
            );
            to_js(&FusionPlanResult {
                status: "ok".into(),
                plan: Some(public),
                reason: None,
                message: None,
            })
        }
        Ok(None) => to_js(&FusionPlanResult {
            status: "unavailable".into(),
            plan: None,
            reason: Some("no plan".into()),
            message: None,
        }),
        Err(err) => to_js(&FusionPlanResult {
            status: "error".into(),
            plan: None,
            reason: None,
            message: Some(err.to_string()),
        }),
    }
}

#[wasm_bindgen]
pub fn completion(_uri: String, _line: u32, _character: u32) -> Result<JsValue, JsValue> {
    ensure_builtins_registered();
    let entry = DOCS.with(|d| d.borrow().docs.get(&_uri).cloned());
    let Some(doc) = entry else { return Ok(JsValue::NULL); };
    let position = Position::new(_line, _character);
    let items = completion_at(&doc.text, &doc.analysis, &position);
    let list = CompletionList { is_incomplete: false, items };
    to_js(&list)
}

#[wasm_bindgen]
pub fn hover(_uri: String, _line: u32, _character: u32) -> Result<JsValue, JsValue> {
    ensure_builtins_registered();
    let entry = DOCS.with(|d| d.borrow().docs.get(&_uri).cloned());
    let Some(doc) = entry else { return Ok(JsValue::NULL); };
    let position = Position::new(_line, _character);
    let result = hover_at(&doc.text, &doc.analysis, &position);
    match result {
        Some(h) => to_js(&h),
        None => Ok(JsValue::NULL),
    }
}

#[wasm_bindgen]
pub fn definition(_uri: String, _line: u32, _character: u32) -> Result<JsValue, JsValue> {
    ensure_builtins_registered();
    let entry = DOCS.with(|d| d.borrow().docs.get(&_uri).cloned());
    let Some(doc) = entry else { return Ok(JsValue::NULL); };
    let position = Position::new(_line, _character);
    let ranges = definition_at(&doc.text, &doc.analysis, &position);
    let locations: Vec<Location> = ranges
        .into_iter()
        .map(|range| Location {
            uri: Url::parse(&_uri).unwrap_or_else(|_| Url::parse("file:///").unwrap()),
            range,
        })
        .collect();
    to_js(&locations)
}

#[wasm_bindgen]
pub fn references(_uri: String, _line: u32, _character: u32) -> Result<JsValue, JsValue> {
    ensure_builtins_registered();
    // For now, reuse definitions as placeholder references.
    definition(_uri, _line, _character)
}

#[wasm_bindgen]
pub fn signature_help(_uri: String, _line: u32, _character: u32) -> Result<JsValue, JsValue> {
    ensure_builtins_registered();
    let entry = DOCS.with(|d| d.borrow().docs.get(&_uri).cloned());
    let Some(doc) = entry else { return Ok(JsValue::NULL); };
    let position = Position::new(_line, _character);
    let result = signature_help_at(&doc.text, &doc.analysis, &position);
    match result {
        Some(h) => to_js(&h),
        None => Ok(JsValue::NULL),
    }
}

#[wasm_bindgen]
pub fn semantic_tokens(_uri: String) -> Result<JsValue, JsValue> {
    ensure_builtins_registered();
    let entry = DOCS.with(|d| d.borrow().docs.get(&_uri).cloned());
    let Some(doc) = entry else { return Ok(JsValue::NULL); };
    let tokens: Option<SemanticTokens> = semantic_tokens_full(&doc.text, &doc.analysis);
    match tokens {
        Some(t) => to_js(&t),
        None => Ok(JsValue::NULL),
    }
}

#[wasm_bindgen]
pub fn document_symbols(_uri: String) -> Result<JsValue, JsValue> {
    ensure_builtins_registered();
    let entry = DOCS.with(|d| d.borrow().docs.get(&_uri).cloned());
    let Some(doc) = entry else { return Ok(JsValue::NULL); };
    let symbols: Vec<DocumentSymbol> = core_document_symbols(&doc.text, &doc.analysis);
    to_js(&symbols)
}

#[wasm_bindgen]
pub fn workspace_symbols_all() -> Result<JsValue, JsValue> {
    ensure_builtins_registered();
    let docs = DOCS.with(|d| {
        d.borrow()
            .docs
            .iter()
            .map(|(uri, doc)| {
                (
                    Url::parse(uri).unwrap_or_else(|_| Url::parse("file:///").unwrap()),
                    doc.text.clone(),
                    doc.analysis.clone(),
                )
            })
            .collect::<Vec<_>>()
    });
    let syms: Vec<SymbolInformation> = workspace_symbols(&docs);
    to_js(&syms)
}

#[wasm_bindgen]
pub fn formatting(_uri: String) -> Result<JsValue, JsValue> {
    ensure_builtins_registered();
    let entry = DOCS.with(|d| d.borrow().docs.get(&_uri).cloned());
    let Some(doc) = entry else { return Ok(JsValue::NULL); };
    let edits: Vec<TextEdit> = formatting_edits(&doc.text, &doc.analysis);
    to_js(&edits)
}

#[wasm_bindgen]
pub fn diagnostics(_uri: String) -> Result<JsValue, JsValue> {
    ensure_builtins_registered();
    let entry = DOCS.with(|d| d.borrow().docs.get(&_uri).cloned());
    let Some(doc) = entry else { return Ok(JsValue::NULL); };
    let diags: Vec<Diagnostic> = diagnostics_for_document(&doc.text, &doc.analysis);
    to_js(&diags)
}

#[wasm_bindgen(js_name = "setCompatMode")]
pub fn set_compat_mode(mode: String) {
    let parsed = match mode.as_str() {
        "matlab" | "MATLAB" => CompatMode::Matlab,
        "strict" | "STRICT" => CompatMode::Strict,
        _ => CompatMode::Matlab,
    };
    COMPAT_MODE.with(|c| c.set(parsed));
}

