use lsp_types::{
    CompletionList, Diagnostic, DocumentSymbol, Position, SemanticTokens, SymbolInformation,
    TextEdit, Url,
};
use runmat_thread_local::runmat_thread_local;
use serde::Serialize;
use serde_wasm_bindgen;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::Once;
use wasm_bindgen::prelude::*;

use crate::core::analysis::{
    analyze_document_with_compat_and_source_async, completion_at, definition_locations_at_async,
    diagnostics_for_document, document_symbols as core_document_symbols, formatting_edits,
    hover_at, references_locations_at_async, semantic_tokens_full, semantic_tokens_lexical,
    signature_help_at, CompatMode, DocumentAnalysis,
};
use crate::core::workspace::workspace_symbols_with_project_async;

#[derive(Default)]
struct DocStore {
    docs: HashMap<String, DocEntry>,
}

#[derive(Clone)]
struct DocEntry {
    version: u64,
    text: String,
    lexical_tokens: lsp_types::SemanticTokens,
    analysis: Option<DocumentAnalysis>,
    compat: CompatMode,
}

runmat_thread_local! {
    static COMPAT_MODE: Cell<CompatMode> = Cell::new(CompatMode::Matlab);
}

runmat_thread_local! {
    static DOCS: RefCell<DocStore> = RefCell::new(DocStore::default());
}

static BUILTIN_REGISTRY: Once = Once::new();

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

fn source_name_from_uri(uri: &str) -> Option<String> {
    let parsed = Url::parse(uri).ok()?;
    if parsed.scheme() != "file" {
        return None;
    }
    let path = parsed.path();
    if path.is_empty() {
        None
    } else {
        Some(path.to_string())
    }
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
pub async fn open_document(uri: String, text: String) {
    let version = DOCS.with(|d| {
        d.borrow()
            .docs
            .get(&uri)
            .map(|doc| doc.version.saturating_add(1))
            .unwrap_or(1)
    });
    let _ = update_document_lexical(uri.clone(), version, text);
    let _ = analyze_document(uri, version).await;
}

#[wasm_bindgen]
pub async fn change_document(uri: String, text: String) {
    open_document(uri, text).await;
}

/// Commit a complete lexical token snapshot before asynchronous semantic
/// analysis begins. The caller supplies the document revision so a later
/// analysis completion can never overwrite newer text.
#[wasm_bindgen]
pub fn update_document_lexical(
    uri: String,
    version: u64,
    text: String,
) -> Result<JsValue, JsValue> {
    ensure_builtins_registered();
    let compat = COMPAT_MODE.with(|c| c.get());
    let lexical_tokens = semantic_tokens_lexical(&text).unwrap_or_else(|| SemanticTokens {
        result_id: None,
        data: Vec::new(),
    });
    DOCS.with(|d| {
        d.borrow_mut().docs.insert(
            uri,
            DocEntry {
                version,
                text,
                lexical_tokens: lexical_tokens.clone(),
                analysis: None,
                compat,
            },
        );
    });
    to_js(&lexical_tokens)
}

/// Analyze a previously committed lexical snapshot. A result is published only
/// if the requested revision is still current for the URI.
#[wasm_bindgen]
pub async fn analyze_document(uri: String, version: u64) -> Result<JsValue, JsValue> {
    ensure_builtins_registered();
    let entry = DOCS.with(|d| d.borrow().docs.get(&uri).cloned());
    let Some(entry) = entry else {
        return Ok(JsValue::NULL);
    };
    if entry.version != version {
        return Ok(JsValue::NULL);
    }
    let source_name = source_name_from_uri(&uri);
    let analysis = analyze_document_with_compat_and_source_async(
        &entry.text,
        entry.compat,
        source_name.as_deref(),
    )
    .await;
    let tokens =
        semantic_tokens_full(&entry.text, &analysis).unwrap_or(entry.lexical_tokens.clone());
    let committed = DOCS.with(|d| {
        let mut docs = d.borrow_mut();
        let Some(current) = docs.docs.get_mut(&uri) else {
            return false;
        };
        if current.version != version {
            return false;
        }
        current.analysis = Some(analysis);
        true
    });
    if committed {
        to_js(&tokens)
    } else {
        Ok(JsValue::NULL)
    }
}

#[wasm_bindgen]
pub fn close_document(uri: String) {
    DOCS.with(|d| {
        d.borrow_mut().docs.remove(&uri);
    });
}

#[wasm_bindgen]
pub fn completion(_uri: String, _line: u32, _character: u32) -> Result<JsValue, JsValue> {
    ensure_builtins_registered();
    let entry = DOCS.with(|d| d.borrow().docs.get(&_uri).cloned());
    let Some(doc) = entry else {
        return Ok(JsValue::NULL);
    };
    let position = Position::new(_line, _character);
    let Some(analysis) = doc.analysis.as_ref() else {
        return Ok(to_js(&CompletionList {
            is_incomplete: true,
            items: Vec::new(),
        })?);
    };
    let items = completion_at(&doc.text, analysis, &position);
    let list = CompletionList {
        is_incomplete: false,
        items,
    };
    to_js(&list)
}

#[wasm_bindgen]
pub fn hover(_uri: String, _line: u32, _character: u32) -> Result<JsValue, JsValue> {
    ensure_builtins_registered();
    let entry = DOCS.with(|d| d.borrow().docs.get(&_uri).cloned());
    let Some(doc) = entry else {
        return Ok(JsValue::NULL);
    };
    let position = Position::new(_line, _character);
    let Some(analysis) = doc.analysis.as_ref() else {
        return Ok(JsValue::NULL);
    };
    let result = hover_at(&doc.text, analysis, &position);
    match result {
        Some(h) => to_js(&h),
        None => Ok(JsValue::NULL),
    }
}

#[wasm_bindgen]
pub async fn definition(_uri: String, _line: u32, _character: u32) -> Result<JsValue, JsValue> {
    ensure_builtins_registered();
    let entry = DOCS.with(|d| d.borrow().docs.get(&_uri).cloned());
    let Some(doc) = entry else {
        return Ok(JsValue::NULL);
    };
    let position = Position::new(_line, _character);
    let uri = Url::parse(&_uri).unwrap_or_else(|_| Url::parse("file:///").unwrap());
    let Some(analysis) = doc.analysis.as_ref() else {
        return Ok(JsValue::NULL);
    };
    let locations = definition_locations_at_async(&doc.text, analysis, &position, &uri).await;
    to_js(&locations)
}

#[wasm_bindgen]
pub async fn references(_uri: String, _line: u32, _character: u32) -> Result<JsValue, JsValue> {
    ensure_builtins_registered();
    let entry = DOCS.with(|d| d.borrow().docs.get(&_uri).cloned());
    let Some(doc) = entry else {
        return Ok(JsValue::NULL);
    };
    let position = Position::new(_line, _character);
    let uri = Url::parse(&_uri).unwrap_or_else(|_| Url::parse("file:///").unwrap());
    let Some(analysis) = doc.analysis.as_ref() else {
        return Ok(JsValue::NULL);
    };
    let locations = references_locations_at_async(&doc.text, analysis, &position, &uri).await;
    to_js(&locations)
}

#[wasm_bindgen]
pub fn signature_help(_uri: String, _line: u32, _character: u32) -> Result<JsValue, JsValue> {
    ensure_builtins_registered();
    let entry = DOCS.with(|d| d.borrow().docs.get(&_uri).cloned());
    let Some(doc) = entry else {
        return Ok(JsValue::NULL);
    };
    let position = Position::new(_line, _character);
    let Some(analysis) = doc.analysis.as_ref() else {
        return Ok(JsValue::NULL);
    };
    let result = signature_help_at(&doc.text, analysis, &position);
    match result {
        Some(h) => to_js(&h),
        None => Ok(JsValue::NULL),
    }
}

#[wasm_bindgen]
pub fn semantic_tokens(_uri: String) -> Result<JsValue, JsValue> {
    ensure_builtins_registered();
    let entry = DOCS.with(|d| d.borrow().docs.get(&_uri).cloned());
    let Some(doc) = entry else {
        return Ok(JsValue::NULL);
    };
    let tokens = doc
        .analysis
        .as_ref()
        .and_then(|analysis| semantic_tokens_full(&doc.text, analysis))
        .unwrap_or(doc.lexical_tokens);
    to_js(&tokens)
}

#[wasm_bindgen]
pub fn document_symbols(_uri: String) -> Result<JsValue, JsValue> {
    ensure_builtins_registered();
    let entry = DOCS.with(|d| d.borrow().docs.get(&_uri).cloned());
    let Some(doc) = entry else {
        return Ok(JsValue::NULL);
    };
    let Some(analysis) = doc.analysis.as_ref() else {
        return to_js(&Vec::<DocumentSymbol>::new());
    };
    let symbols: Vec<DocumentSymbol> = core_document_symbols(&doc.text, analysis);
    to_js(&symbols)
}

#[wasm_bindgen]
pub async fn workspace_symbols_all() -> Result<JsValue, JsValue> {
    ensure_builtins_registered();
    let compat = COMPAT_MODE.with(|c| c.get());
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
            .filter_map(|(uri, text, analysis)| analysis.map(|analysis| (uri, text, analysis)))
            .collect::<Vec<_>>()
    });
    let syms: Vec<SymbolInformation> =
        workspace_symbols_with_project_async(&docs, compat, None).await;
    to_js(&syms)
}

#[wasm_bindgen]
pub fn formatting(_uri: String) -> Result<JsValue, JsValue> {
    ensure_builtins_registered();
    let entry = DOCS.with(|d| d.borrow().docs.get(&_uri).cloned());
    let Some(doc) = entry else {
        return Ok(JsValue::NULL);
    };
    let Some(analysis) = doc.analysis.as_ref() else {
        return to_js(&Vec::<TextEdit>::new());
    };
    let edits: Vec<TextEdit> = formatting_edits(&doc.text, analysis);
    to_js(&edits)
}

#[wasm_bindgen]
pub fn diagnostics(_uri: String) -> Result<JsValue, JsValue> {
    ensure_builtins_registered();
    let entry = DOCS.with(|d| d.borrow().docs.get(&_uri).cloned());
    let Some(doc) = entry else {
        return Ok(JsValue::NULL);
    };
    let Some(analysis) = doc.analysis.as_ref() else {
        return to_js(&Vec::<Diagnostic>::new());
    };
    let diags: Vec<Diagnostic> = diagnostics_for_document(&doc.text, analysis);
    to_js(&diags)
}

#[wasm_bindgen(js_name = "setCompatMode")]
pub fn set_compat_mode(mode: String) {
    let parsed = match mode.as_str() {
        "runmat" | "RUNMAT" => CompatMode::RunMat,
        "matlab" | "MATLAB" => CompatMode::Matlab,
        "strict" | "STRICT" => CompatMode::Strict,
        _ => CompatMode::Matlab,
    };
    COMPAT_MODE.with(|c| c.set(parsed));
}
