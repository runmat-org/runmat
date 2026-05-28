use lsp_types::{SymbolInformation, Url};
use serde_json::json;

use crate::core::analysis::{
    analyze_document_with_compat_and_source_async, document_symbols, CompatMode, DocumentAnalysis,
};
use crate::core::project::ProjectContext;
use std::collections::HashSet;

pub fn workspace_symbols_with_project(
    docs: &[(Url, String, DocumentAnalysis)],
    compat: CompatMode,
    query: Option<&str>,
) -> Vec<SymbolInformation> {
    let mut out = workspace_symbols_from_docs(docs);
    #[cfg(target_arch = "wasm32")]
    let _ = compat;
    #[cfg(not(target_arch = "wasm32"))]
    {
        let source_hint = docs
            .first()
            .and_then(|(uri, _, _)| source_name_from_uri(uri));
        if let Some(project) = ProjectContext::discover_from_source_name(source_hint.as_deref()) {
            out.extend(project_symbols_sync(&project, compat));
        }
    }
    dedupe_symbol_info(&mut out);
    filter_symbols(out, query)
}

pub fn workspace_symbols_from_documents(
    docs: &[(Url, String, DocumentAnalysis)],
    query: Option<&str>,
) -> Vec<SymbolInformation> {
    let mut out = workspace_symbols_from_docs(docs);
    dedupe_symbol_info(&mut out);
    filter_symbols(out, query)
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
pub async fn workspace_symbols_with_project_async(
    docs: &[(Url, String, DocumentAnalysis)],
    compat: CompatMode,
    query: Option<&str>,
) -> Vec<SymbolInformation> {
    let mut out = workspace_symbols_from_docs(docs);
    let source_hint = docs
        .first()
        .and_then(|(uri, _, _)| source_name_from_uri(uri));
    if let Some(project) =
        ProjectContext::discover_from_source_name_async(source_hint.as_deref()).await
    {
        out.extend(project_symbols_async(&project, compat).await);
    }
    dedupe_symbol_info(&mut out);
    filter_symbols(out, query)
}

fn workspace_symbols_from_docs(docs: &[(Url, String, DocumentAnalysis)]) -> Vec<SymbolInformation> {
    let mut out = Vec::new();
    for (uri, text, analysis) in docs {
        append_document_symbols(&mut out, uri, text, analysis);
    }
    out
}

fn append_document_symbols(
    out: &mut Vec<SymbolInformation>,
    uri: &Url,
    text: &str,
    analysis: &DocumentAnalysis,
) {
    for sym in document_symbols(text, analysis) {
        if let Ok(si) = serde_json::from_value::<SymbolInformation>(json!({
            "name": sym.name,
            "kind": sym.kind,
            "tags": sym.tags,
            "location": {
                "uri": uri,
                "range": sym.range,
            },
            "containerName": serde_json::Value::Null,
        })) {
            out.push(si);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn project_symbols_sync(project: &ProjectContext, compat: CompatMode) -> Vec<SymbolInformation> {
    let mut out = Vec::new();
    for source in project.all_source_files() {
        let Some(uri) = file_path_to_url(source) else {
            continue;
        };
        let Some(text) =
            futures::executor::block_on(runmat_filesystem::read_to_string_async(source)).ok()
        else {
            continue;
        };
        let source_name = source.to_str();
        let analysis = crate::core::analysis::analyze_document_with_compat_and_source(
            &text,
            compat,
            source_name,
        );
        append_document_symbols(&mut out, &uri, &text, &analysis);
    }
    out
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
async fn project_symbols_async(
    project: &ProjectContext,
    compat: CompatMode,
) -> Vec<SymbolInformation> {
    let mut out = Vec::new();
    for source in project.all_source_files() {
        let Some(uri) = file_path_to_url(source) else {
            continue;
        };
        let Ok(text) = runmat_filesystem::read_to_string_async(source).await else {
            continue;
        };
        let source_name = source.to_str();
        let analysis =
            analyze_document_with_compat_and_source_async(&text, compat, source_name).await;
        append_document_symbols(&mut out, &uri, &text, &analysis);
    }
    out
}

fn source_name_from_uri(uri: &Url) -> Option<String> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        uri.to_file_path()
            .ok()
            .and_then(|path| path.to_str().map(str::to_owned))
    }
    #[cfg(target_arch = "wasm32")]
    {
        if uri.scheme() != "file" {
            return None;
        }
        let path = uri.path();
        if path.is_empty() {
            None
        } else {
            Some(path.to_string())
        }
    }
}

fn dedupe_symbol_info(symbols: &mut Vec<SymbolInformation>) {
    let mut seen = HashSet::new();
    symbols.retain(|symbol| {
        let key = format!(
            "{}:{}:{}:{}:{}:{}",
            symbol.name,
            symbol.location.uri,
            symbol.location.range.start.line,
            symbol.location.range.start.character,
            symbol.location.range.end.line,
            symbol.location.range.end.character
        );
        seen.insert(key)
    });
}

fn filter_symbols(symbols: Vec<SymbolInformation>, query: Option<&str>) -> Vec<SymbolInformation> {
    let Some(query) = query.map(str::trim).filter(|query| !query.is_empty()) else {
        return symbols;
    };
    let query = query.to_ascii_lowercase();
    symbols
        .into_iter()
        .filter(|symbol| symbol.name.to_ascii_lowercase().contains(&query))
        .collect()
}

fn file_path_to_url(path: &std::path::Path) -> Option<Url> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        Url::from_file_path(path).ok()
    }
    #[cfg(target_arch = "wasm32")]
    {
        let raw = path.to_str()?;
        let normalized = if raw.starts_with('/') {
            raw.to_string()
        } else {
            format!("/{raw}")
        };
        Url::parse(&format!("file://{normalized}")).ok()
    }
}
