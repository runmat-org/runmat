use lsp_types::{
    CompletionList, Diagnostic, DocumentSymbol, Position, SemanticTokens, SymbolInformation,
    TextEdit, Url,
};
use runmat_thread_local::runmat_thread_local;
use serde::Serialize;
use serde_wasm_bindgen;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Once;
use wasm_bindgen::prelude::*;

use crate::core::analysis::{
    analyze_document_with_compat_and_source_async, completion_at, definition_locations_at_async,
    diagnostics_for_document, document_symbols as core_document_symbols, formatting_edits,
    hover_at, quick_information_at, references_locations_at_async, semantic_document_facts,
    semantic_tokens_full, semantic_tokens_lexical, signature_help_at, CompatMode, DocumentAnalysis,
};
use crate::core::design::{design_dependencies, design_dialect, DesignDocumentAnalysis, DesignFiles};
use crate::core::fea::{is_fea_uri, resolve_project_facts, FeaDocumentAnalysis};
use crate::core::workspace::workspace_symbols_with_project_async;

#[derive(Default)]
struct DocStore {
    docs: HashMap<String, DocEntry>,
}

#[derive(Clone)]
struct DocEntry {
    version: u64,
    revision: u64,
    text: String,
    lexical_tokens: lsp_types::SemanticTokens,
    analysis: Option<DocumentAnalysis>,
    fea_analysis: Option<FeaDocumentAnalysis>,
    fea_error: Option<String>,
    design_analysis: Option<DesignDocumentAnalysis>,
    compat: CompatMode,
}

runmat_thread_local! {
    static COMPAT_MODE: Cell<CompatMode> = Cell::new(CompatMode::RunMat);
}

runmat_thread_local! {
    static DOCS: RefCell<DocStore> = RefCell::new(DocStore::default());
}

runmat_thread_local! {
    static NEXT_DOCUMENT_REVISION: Cell<u64> = Cell::new(0);
}

runmat_thread_local! {
    /// Texts of the files design documents read that aren't open (their circuit, lock, parts),
    /// as the host supplied them with `setDesignFiles`.
    static DESIGN_FILES: RefCell<DesignFiles> = RefCell::new(DesignFiles::new());
}

runmat_thread_local! {
    static WORKSPACE_REVISION: Cell<u64> = Cell::new(0);
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

#[wasm_bindgen(js_name = "projectHandoff")]
pub async fn project_handoff(source_path: String) -> Result<JsValue, JsValue> {
    let frozen = runmat_package::discover_frozen_project_from_async(
        Path::new(&source_path),
        Default::default(),
    )
    .await
    .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let Some(frozen) = frozen else {
        return Ok(JsValue::NULL);
    };
    let handoff = runmat_package::FrozenProjectHandoff::new(frozen);
    handoff
        .validate()
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    to_js(&handoff)
}

#[wasm_bindgen(js_name = "projectRevision")]
pub async fn project_revision(source_path: String) -> Result<JsValue, JsValue> {
    let frozen = runmat_package::discover_frozen_project_from_async(
        Path::new(&source_path),
        Default::default(),
    )
    .await
    .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let Some(frozen) = frozen else {
        return Ok(JsValue::NULL);
    };
    let handoff = runmat_package::FrozenProjectHandoff::new(frozen);
    handoff
        .validate()
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    to_js(&handoff.revision())
}

#[wasm_bindgen(js_name = "validateProjectHandoff")]
pub fn validate_project_handoff(value: JsValue) -> Result<JsValue, JsValue> {
    let handoff: runmat_package::FrozenProjectHandoff = serde_wasm_bindgen::from_value(value)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    handoff
        .validate()
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    to_js(&handoff.revision())
}

#[wasm_bindgen(js_name = "installProjectHandoff")]
pub fn install_project_handoff(value: JsValue) -> Result<JsValue, JsValue> {
    let handoff: runmat_package::FrozenProjectHandoff = serde_wasm_bindgen::from_value(value)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let revision = crate::core::project::ProjectContext::install_handoff(handoff)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    to_js(&revision)
}

#[wasm_bindgen(js_name = "clearProjectHandoff")]
pub fn clear_project_handoff() {
    crate::core::project::ProjectContext::clear_installed_handoff();
}

/// Discover tests from the exact frozen run snapshot supplied by the browser
/// coordinator. This intentionally does not consult the installed project or
/// mutable editor store, so runtime and LSP consumers cannot observe different
/// graph/source revisions.
#[wasm_bindgen(js_name = "discoverTests")]
pub fn discover_tests(value: JsValue) -> Result<JsValue, JsValue> {
    ensure_builtins_registered();
    let snapshot: runmat_test::discovery::FrozenTestRunSnapshot =
        serde_wasm_bindgen::from_value(value)
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let compat = COMPAT_MODE.with(|mode| mode.get());
    let discovery = runmat_static_analysis::testing::discover_frozen_tests(&snapshot, compat);
    to_js(&discovery)
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
    let revision = NEXT_DOCUMENT_REVISION.with(|next| {
        let revision = next.get().saturating_add(1);
        next.set(revision);
        revision
    });
    WORKSPACE_REVISION.with(|workspace| workspace.set(workspace.get().saturating_add(1)));
    let parsed_uri = Url::parse(&uri).ok();
    let is_fea = parsed_uri.as_ref().is_some_and(is_fea_uri);
    let design = parsed_uri.as_ref().and_then(design_dialect);
    let lexical_tokens = if let Some(dialect) = design {
        DesignDocumentAnalysis::lexical_semantic_tokens(dialect, &text)
    } else if is_fea {
        FeaDocumentAnalysis::lexical_semantic_tokens(&text)
    } else {
        semantic_tokens_lexical(&text).unwrap_or_else(|| SemanticTokens {
            result_id: None,
            data: Vec::new(),
        })
    };
    DOCS.with(|d| {
        d.borrow_mut().docs.insert(
            uri,
            DocEntry {
                version,
                revision,
                text,
                lexical_tokens: lexical_tokens.clone(),
                analysis: None,
                fea_analysis: None,
                fea_error: None,
                design_analysis: None,
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
    let workspace_revision = WORKSPACE_REVISION.with(Cell::get);
    let parsed_uri = Url::parse(&uri).ok();
    let is_fea = parsed_uri.as_ref().is_some_and(is_fea_uri);
    if let Some(dialect) = parsed_uri.as_ref().and_then(design_dialect) {
        let files = design_files();
        let analysis = DesignDocumentAnalysis::analyze(
            parsed_uri.as_ref().expect("a parsed design uri"),
            dialect,
            &entry.text,
            &files,
        );
        let tokens = analysis.semantic_tokens();
        let committed = DOCS.with(|d| {
            let mut docs = d.borrow_mut();
            let Some(current) = docs.docs.get_mut(&uri) else {
                return false;
            };
            if current.version != version || current.revision != entry.revision {
                return false;
            }
            current.design_analysis = Some(analysis);
            true
        });
        return if committed {
            to_js(&tokens)
        } else {
            Ok(JsValue::NULL)
        };
    }
    let (analysis, fea_analysis, fea_error, tokens) = if is_fea {
        match analyze_fea_document(&uri, &entry) {
            Ok(analysis) => {
                let tokens = analysis
                    .semantic_tokens(&entry.text)
                    .unwrap_or_else(|| entry.lexical_tokens.clone());
                (None, Some(analysis), None, tokens)
            }
            Err(error) => (None, None, Some(error), entry.lexical_tokens.clone()),
        }
    } else {
        let source_name = source_name_from_uri(&uri);
        let analysis = analyze_document_with_compat_and_source_async(
            &entry.text,
            entry.compat,
            source_name.as_deref(),
        )
        .await;
        let tokens = semantic_tokens_full(&entry.text, &analysis)
            .unwrap_or_else(|| entry.lexical_tokens.clone());
        (Some(analysis), None, None, tokens)
    };
    let committed = DOCS.with(|d| {
        let mut docs = d.borrow_mut();
        let Some(current) = docs.docs.get_mut(&uri) else {
            return false;
        };
        if current.version != version || current.revision != entry.revision {
            return false;
        }
        if is_fea && WORKSPACE_REVISION.with(Cell::get) != workspace_revision {
            return false;
        }
        current.analysis = analysis;
        current.fea_analysis = fea_analysis;
        current.fea_error = fea_error;
        true
    });
    if committed {
        to_js(&tokens)
    } else {
        Ok(JsValue::NULL)
    }
}

/// The files design documents read: open documents' current text over the files the host
/// supplied.
fn design_files() -> DesignFiles {
    let mut files = DESIGN_FILES.with(|f| f.borrow().clone());
    DOCS.with(|documents| {
        for (uri, document) in documents.borrow().docs.iter() {
            if let Some(path) = Url::parse(uri).ok().as_ref().and_then(wasm_file_path) {
                files.insert(path.to_string_lossy().into_owned(), document.text.clone());
            }
        }
    });
    files
}

/// The files a design document reads beyond itself (workspace paths), so the host can supply
/// those that aren't open with `setDesignFiles`.
#[wasm_bindgen(js_name = "designDependencies")]
pub fn design_dependencies_of(uri: String, text: String) -> Result<JsValue, JsValue> {
    let Some(parsed) = Url::parse(&uri).ok() else {
        return to_js(&Vec::<String>::new());
    };
    let Some(dialect) = design_dialect(&parsed) else {
        return to_js(&Vec::<String>::new());
    };
    to_js(&design_dependencies(dialect, parsed.path(), &text))
}

/// Supplies the texts of files design documents read (`{ "/p/io.circuit": "…" }`); a `null`
/// text forgets a file. Open documents always win over these.
#[wasm_bindgen(js_name = "setDesignFiles")]
pub fn set_design_files(files: JsValue) -> Result<JsValue, JsValue> {
    let files: BTreeMap<String, Option<String>> = serde_wasm_bindgen::from_value(files)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    DESIGN_FILES.with(|stored| {
        let mut stored = stored.borrow_mut();
        for (path, text) in files {
            match text {
                Some(text) => {
                    stored.insert(path, text);
                }
                None => {
                    stored.remove(&path);
                }
            }
        }
    });
    WORKSPACE_REVISION.with(|workspace| workspace.set(workspace.get().saturating_add(1)));
    Ok(JsValue::NULL)
}

fn analyze_fea_document(uri: &str, entry: &DocEntry) -> Result<FeaDocumentAnalysis, String> {
    let physical_path = Url::parse(uri)
        .ok()
        .and_then(|url| wasm_file_path(&url))
        .ok_or_else(|| "FEA language tooling requires a file-backed document".to_string())?;
    let (overlays, source_overlays) = DOCS.with(|documents| {
        let documents = documents.borrow();
        let source_overlays: BTreeMap<PathBuf, String> = documents
            .docs
            .iter()
            .filter_map(|(document_uri, document)| {
                let path = wasm_file_path(&Url::parse(document_uri).ok()?)?;
                Some((path, document.text.clone()))
            })
            .collect::<BTreeMap<_, _>>();
        let overlays = source_overlays
            .iter()
            .filter(|(path, _)| {
                path.extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("fea"))
            })
            .map(|(path, source)| (path.clone(), source.as_bytes().to_vec()))
            .collect::<Vec<_>>();
        (overlays, source_overlays)
    });
    let source_root = source_overlays.keys().fold(
        physical_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("/")),
        |root, path| common_ancestor(&root, path),
    );
    let prepared = FeaDocumentAnalysis::prepare_with_overlays(
        &entry.text,
        &physical_path,
        &[source_root],
        &overlays,
    )?;
    let requests = prepared.function_requests()?;
    let facts = resolve_project_facts(
        prepared.source_root(),
        &requests,
        &physical_path,
        entry.compat,
        &source_overlays,
    )?;
    prepared
        .check(facts.functions, facts.profiles)
        .map(|checked| FeaDocumentAnalysis::from_checked(checked, facts.function_sources))
}

fn common_ancestor(left: &Path, right: &Path) -> PathBuf {
    left.ancestors()
        .find(|candidate| right.starts_with(candidate))
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("/"))
}

fn wasm_file_path(uri: &Url) -> Option<PathBuf> {
    if uri.scheme() != "file" {
        return None;
    }
    let encoded = uri.path().as_bytes();
    let mut decoded = Vec::with_capacity(encoded.len());
    let mut index = 0;
    while index < encoded.len() {
        if encoded[index] == b'%' && index + 2 < encoded.len() {
            let high = hex_value(encoded[index + 1])?;
            let low = hex_value(encoded[index + 2])?;
            decoded.push((high << 4) | low);
            index += 3;
        } else {
            decoded.push(encoded[index]);
            index += 1;
        }
    }
    String::from_utf8(decoded).ok().map(PathBuf::from)
}

fn hex_value(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

#[wasm_bindgen]
pub fn close_document(uri: String) {
    DOCS.with(|d| {
        d.borrow_mut().docs.remove(&uri);
    });
    WORKSPACE_REVISION.with(|workspace| workspace.set(workspace.get().saturating_add(1)));
}

#[wasm_bindgen]
pub fn completion(_uri: String, _line: u32, _character: u32) -> Result<JsValue, JsValue> {
    ensure_builtins_registered();
    let entry = DOCS.with(|d| d.borrow().docs.get(&_uri).cloned());
    let Some(doc) = entry else {
        return Ok(JsValue::NULL);
    };
    let position = Position::new(_line, _character);
    if let Some(analysis) = doc.design_analysis.as_ref() {
        return to_js(&CompletionList {
            is_incomplete: false,
            items: analysis.completion_items(&position),
        });
    }
    if let Some(analysis) = doc.fea_analysis.as_ref() {
        return to_js(&CompletionList {
            is_incomplete: false,
            items: analysis.completion_items(),
        });
    }
    let Some(analysis) = doc.analysis.as_ref() else {
        return to_js(&CompletionList {
            is_incomplete: true,
            items: Vec::new(),
        });
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
    if let Some(analysis) = doc.design_analysis.as_ref() {
        return analysis
            .hover(&position)
            .map_or(Ok(JsValue::NULL), |hover| to_js(&hover));
    }
    if let Some(analysis) = doc.fea_analysis.as_ref() {
        return analysis
            .hover(&doc.text, &position)
            .map_or(Ok(JsValue::NULL), |hover| to_js(&hover));
    }
    let Some(analysis) = doc.analysis.as_ref() else {
        return Ok(JsValue::NULL);
    };
    let result = hover_at(&doc.text, analysis, &position);
    match result {
        Some(h) => to_js(&h),
        None => Ok(JsValue::NULL),
    }
}

#[wasm_bindgen(js_name = "quickInformation")]
pub fn quick_information(uri: String, line: u32, character: u32) -> Result<JsValue, JsValue> {
    ensure_builtins_registered();
    let entry = DOCS.with(|documents| documents.borrow().docs.get(&uri).cloned());
    let Some(document) = entry else {
        return Ok(JsValue::NULL);
    };
    if document.fea_analysis.is_some() || document.design_analysis.is_some() {
        return Ok(JsValue::NULL);
    }
    let Some(analysis) = document.analysis.as_ref() else {
        return Ok(JsValue::NULL);
    };
    let position = Position::new(line, character);
    quick_information_at(&document.text, analysis, &position)
        .map_or(Ok(JsValue::NULL), |information| to_js(&information))
}

#[wasm_bindgen(js_name = "semanticFacts")]
pub fn semantic_facts(uri: String) -> Result<JsValue, JsValue> {
    ensure_builtins_registered();
    let entry = DOCS.with(|documents| documents.borrow().docs.get(&uri).cloned());
    let Some(document) = entry else {
        return Ok(JsValue::NULL);
    };
    if document.fea_analysis.is_some() || document.design_analysis.is_some() {
        return Ok(JsValue::NULL);
    }
    let Some(analysis) = document.analysis.as_ref() else {
        return Ok(JsValue::NULL);
    };
    semantic_document_facts(analysis).map_or(Ok(JsValue::NULL), to_js)
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
    if let Some(analysis) = doc.design_analysis.as_ref() {
        return to_js(&analysis.definition(&position).into_iter().collect::<Vec<_>>());
    }
    if let Some(analysis) = doc.fea_analysis.as_ref() {
        return to_js(
            &analysis
                .definition(&doc.text, &position)
                .into_iter()
                .collect::<Vec<_>>(),
        );
    }
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
    if let Some(analysis) = doc.design_analysis.as_ref() {
        return to_js(&analysis.references(&position, true));
    }
    if let Some(analysis) = doc.fea_analysis.as_ref() {
        return to_js(&analysis.references(&doc.text, &position));
    }
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
    if doc.design_analysis.is_some() {
        return Ok(JsValue::NULL);
    }
    if let Some(analysis) = doc.fea_analysis.as_ref() {
        return analysis
            .signature_help(&doc.text, &position)
            .map_or(Ok(JsValue::NULL), |help| to_js(&help));
    }
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
        .design_analysis
        .as_ref()
        .map(DesignDocumentAnalysis::semantic_tokens)
        .or_else(|| {
            doc.fea_analysis
                .as_ref()
                .and_then(|analysis| analysis.semantic_tokens(&doc.text))
        })
        .or_else(|| {
            doc.analysis
                .as_ref()
                .and_then(|analysis| semantic_tokens_full(&doc.text, analysis))
        })
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
    if let Some(analysis) = doc.design_analysis.as_ref() {
        return to_js(&analysis.document_symbols());
    }
    if let Some(analysis) = doc.fea_analysis.as_ref() {
        return to_js(&analysis.document_symbols(&doc.text));
    }
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
    let (docs, fea_analyses) = DOCS.with(|d| {
        let documents = d.borrow();
        let docs = documents
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
            .collect::<Vec<_>>();
        let fea_analyses = documents
            .docs
            .values()
            .filter_map(|document| document.fea_analysis.clone())
            .collect::<Vec<_>>();
        (docs, fea_analyses)
    });
    let mut syms: Vec<SymbolInformation> =
        workspace_symbols_with_project_async(&docs, compat, None).await;
    syms.extend(
        fea_analyses
            .iter()
            .flat_map(|analysis| analysis.workspace_symbols(None)),
    );
    syms.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then_with(|| left.location.uri.as_str().cmp(right.location.uri.as_str()))
            .then_with(|| left.location.range.start.cmp(&right.location.range.start))
    });
    syms.dedup_by(|left, right| {
        left.name == right.name
            && left.location.uri == right.location.uri
            && left.location.range == right.location.range
    });
    to_js(&syms)
}

#[wasm_bindgen]
pub fn formatting(_uri: String) -> Result<JsValue, JsValue> {
    ensure_builtins_registered();
    let entry = DOCS.with(|d| d.borrow().docs.get(&_uri).cloned());
    let Some(doc) = entry else {
        return Ok(JsValue::NULL);
    };
    if let Some(analysis) = doc.design_analysis.as_ref() {
        return to_js(&analysis.formatting_edits());
    }
    if let Some(analysis) = doc.fea_analysis.as_ref() {
        return to_js(&analysis.formatting_edits(&doc.text));
    }
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
    if let Some(analysis) = doc.design_analysis.as_ref() {
        return to_js(&analysis.diagnostics());
    }
    if let Some(analysis) = doc.fea_analysis.as_ref() {
        return to_js(&analysis.diagnostics(&doc.text));
    }
    if let Some(error) = doc.fea_error.as_ref() {
        return to_js(&vec![Diagnostic {
            range: lsp_types::Range::new(Position::new(0, 0), Position::new(0, 0)),
            severity: Some(lsp_types::DiagnosticSeverity::ERROR),
            code: Some(lsp_types::NumberOrString::String(
                "FEA_ANALYSIS_UNAVAILABLE".to_string(),
            )),
            source: Some("runmat-fea".to_string()),
            message: error.clone(),
            ..Diagnostic::default()
        }]);
    }
    let Some(analysis) = doc.analysis.as_ref() else {
        return to_js(&Vec::<Diagnostic>::new());
    };
    let diags: Vec<Diagnostic> = diagnostics_for_document(&doc.text, analysis);
    to_js(&diags)
}

/// The renamable name under the cursor in a design document: `{ range, placeholder }`.
#[wasm_bindgen(js_name = "prepareRename")]
pub fn prepare_rename(uri: String, line: u32, character: u32) -> Result<JsValue, JsValue> {
    let entry = DOCS.with(|d| d.borrow().docs.get(&uri).cloned());
    let Some(analysis) = entry.and_then(|doc| doc.design_analysis) else {
        return Ok(JsValue::NULL);
    };
    match analysis.prepare_rename(&Position::new(line, character)) {
        Some((range, placeholder)) => to_js(&serde_json::json!({
            "range": range,
            "placeholder": placeholder,
        })),
        None => Ok(JsValue::NULL),
    }
}

/// Renames a design document's symbol at its declaration and every use (a workspace edit).
#[wasm_bindgen]
pub fn rename(uri: String, line: u32, character: u32, new_name: String) -> Result<JsValue, JsValue> {
    let entry = DOCS.with(|d| d.borrow().docs.get(&uri).cloned());
    let Some(analysis) = entry.and_then(|doc| doc.design_analysis) else {
        return Ok(JsValue::NULL);
    };
    match analysis.rename(&Position::new(line, character), &new_name) {
        Ok(Some(edit)) => to_js(&edit),
        Ok(None) => Ok(JsValue::NULL),
        Err(message) => Err(JsValue::from_str(&message)),
    }
}

#[wasm_bindgen(js_name = "setCompatMode")]
pub fn set_compat_mode(mode: String) {
    let parsed = match mode.as_str() {
        "runmat" | "RUNMAT" => CompatMode::RunMat,
        "matlab" | "MATLAB" => CompatMode::Matlab,
        "strict" | "STRICT" => CompatMode::Strict,
        _ => CompatMode::RunMat,
    };
    COMPAT_MODE.with(|c| c.set(parsed));
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{ValueFact, ValueKindFact};
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test(async)]
    async fn wave_facts_match_the_native_program_point_product() {
        let uri = "file:///rm1068-wave.m".to_string();
        let source = crate::core::semantic::fixtures::WAVE_SOURCE;
        open_document(uri.clone(), source.to_string()).await;
        let value = semantic_facts(uri).expect("portable semantic facts");
        let facts: runmat_static_analysis::semantic::SemanticDocumentFacts =
            serde_wasm_bindgen::from_value(value).expect("deserialize semantic facts");
        facts
            .validate_current()
            .expect("current browser fact revision");

        for (needle, dimensions) in crate::core::semantic::fixtures::WAVE_EXPECTATIONS {
            let offset = source
                .find(needle)
                .unwrap_or_else(|| panic!("missing {needle}"));
            let position = crate::core::position::offset_to_position(source, offset);
            let value = quick_information(
                "file:///rm1068-wave.m".to_string(),
                position.line,
                position.character,
            )
            .expect("quick information payload");
            let information: runmat_static_analysis::semantic::SemanticQuickInformation =
                serde_wasm_bindgen::from_value(value).expect("deserialize quick information");
            let fact = information
                .observation
                .and_then(|observation| observation.fact)
                .unwrap_or_else(|| panic!("missing fact for {needle}"));
            assert_eq!(
                fact.shape.known_dims(),
                Some(dimensions.iter().copied().map(Some).collect()),
                "wrong browser shape for {needle}: {fact:?}"
            );
        }
    }

    #[wasm_bindgen_test]
    fn complete_value_taxonomy_round_trips_through_the_browser_boundary() {
        use std::collections::BTreeMap;

        let unknown = || ValueFact::unknown(runmat_types::DynamicReason::Unspecified);
        let kinds = vec![
            ValueKindFact::Never,
            ValueKindFact::Unknown,
            ValueKindFact::Void,
            ValueKindFact::Numeric(runmat_types::NumericFact {
                class: runmat_types::NumericClass::Double,
                domain: runmat_types::NumericDomain::Complex,
            }),
            ValueKindFact::Logical,
            ValueKindFact::Character,
            ValueKindFact::String,
            ValueKindFact::Symbolic,
            ValueKindFact::Cell(runmat_types::CellFact {
                element: Box::new(unknown()),
                elements: vec![unknown()],
                elements_complete: true,
            }),
            ValueKindFact::Struct(runmat_types::StructFact::scalar(BTreeMap::new(), true)),
            ValueKindFact::Object(runmat_types::ObjectFact {
                class: None,
                runtime_class: None,
                properties: BTreeMap::new(),
                properties_complete: false,
                handle_semantics: None,
            }),
            ValueKindFact::ClassReference(runmat_types::ClassReferenceFact {
                class: None,
                runtime_class: None,
            }),
            ValueKindFact::Callable(runmat_types::CallableFact {
                identity: None,
                capabilities: Default::default(),
                parameters: vec![unknown()],
                parameters_complete: false,
                outputs: vec![unknown()],
                outputs_complete: false,
                variadic_inputs: true,
                variadic_outputs: true,
                captures: Vec::new(),
                captures_complete: true,
            }),
            ValueKindFact::OutputList(runmat_types::OutputListFact {
                outputs: vec![unknown()],
                variadic: true,
            }),
            ValueKindFact::Exception(runmat_types::ExceptionFact {
                identifier: Some("RunMat:test".to_string()),
            }),
            ValueKindFact::Execution(runmat_types::ExecutionFact::Future {
                output: runmat_types::ValueSequenceFact::single(unknown()),
                state: runmat_types::FutureStateFact::Lazy,
            }),
            ValueKindFact::Execution(runmat_types::ExecutionFact::Task {
                output: runmat_types::ValueSequenceFact::single(unknown()),
                spawn_safety: runmat_types::SpawnSafetyFact::RequiresIsolation,
            }),
            ValueKindFact::Execution(runmat_types::ExecutionFact::Pool),
            ValueKindFact::Execution(runmat_types::ExecutionFact::Job {
                output: runmat_types::ValueSequenceFact::single(unknown()),
            }),
            ValueKindFact::Distributed(runmat_types::DistributedFact {
                id: runmat_types::DistributedValueId {
                    function: runmat_types::ProgramFunctionId(1),
                    ordinal: 2,
                },
                owner: runmat_types::ParallelRegionId(runmat_types::RegionId {
                    function: runmat_types::ProgramFunctionId(1),
                    ordinal: 3,
                }),
                scheme: Some(runmat_types::DistributionScheme::Replicated),
                value: Box::new(unknown()),
                materializable: true,
            }),
            ValueKindFact::Foreign(runmat_types::ForeignFact {
                family: "java".to_string(),
                type_name: Some("java.lang.String".to_string()),
                type_version: None,
                ownership: runmat_types::ForeignOwnershipFact::Owned,
                affinity: runmat_types::ForeignAffinityFact::AnyThread,
                lifetime: runmat_types::ForeignLifetimeFact::Session,
            }),
        ];
        let facts = kinds.into_iter().map(ValueFact::scalar).collect::<Vec<_>>();

        let javascript = serde_wasm_bindgen::to_value(&facts).expect("serialize taxonomy");
        let round_trip: Vec<ValueFact> =
            serde_wasm_bindgen::from_value(javascript).expect("deserialize taxonomy");
        assert_eq!(round_trip, facts);
    }
}
