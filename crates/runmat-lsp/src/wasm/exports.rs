use lsp_types::{
    CompletionList, Diagnostic, DocumentSymbol, Position, SemanticTokens, SymbolInformation,
    TextEdit, Url,
};
use runmat_thread_local::runmat_thread_local;
use serde::Serialize;
use serde_wasm_bindgen;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Once;
use wasm_bindgen::prelude::*;

use crate::core::analysis::{
    analyze_document_with_compat_and_source_async, completion_at, definition_locations_at_async,
    diagnostics_for_document, document_symbols as core_document_symbols, formatting_edits,
    hover_at, quick_information_at, references_locations_at_async, semantic_document_facts,
    semantic_tokens_full, semantic_tokens_lexical, signature_help_at, CompatMode, DocumentAnalysis,
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
            ValueKindFact::Struct(runmat_types::StructFact {
                fields: BTreeMap::new(),
                fields_complete: true,
            }),
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
                output: Box::new(unknown()),
                state: runmat_types::FutureStateFact::Lazy,
            }),
            ValueKindFact::Execution(runmat_types::ExecutionFact::Task {
                output: Box::new(unknown()),
                spawn_safety: runmat_types::SpawnSafetyFact::RequiresIsolation,
            }),
            ValueKindFact::Execution(runmat_types::ExecutionFact::Pool),
            ValueKindFact::Execution(runmat_types::ExecutionFact::Job {
                output: Box::new(unknown()),
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
