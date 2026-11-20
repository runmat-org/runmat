use lsp_types::{SymbolInformation, Url};
use serde_json::json;

use crate::core::analysis::{
    analyze_document_with_compat_and_source_async, document_symbols, CompatMode, DocumentAnalysis,
};
use crate::core::project::SourceContext;
use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};

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
        if let Some(project) = SourceContext::discover_from_source_name(source_hint.as_deref()) {
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

pub fn merge_workspace_symbols(
    mut primary: Vec<SymbolInformation>,
    additional: impl IntoIterator<Item = SymbolInformation>,
    query: Option<&str>,
) -> Vec<SymbolInformation> {
    primary.extend(additional);
    dedupe_symbol_info(&mut primary);
    filter_symbols(primary, query)
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
        SourceContext::discover_from_source_name_async(source_hint.as_deref()).await
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
    // Workspace symbols are the callable and class names, wherever the outline nests them.
    fn named(symbols: Vec<lsp_types::DocumentSymbol>, out: &mut Vec<lsp_types::DocumentSymbol>) {
        for mut symbol in symbols {
            let children = symbol.children.take().unwrap_or_default();
            if matches!(
                symbol.kind,
                lsp_types::SymbolKind::FUNCTION
                    | lsp_types::SymbolKind::CLASS
                    | lsp_types::SymbolKind::METHOD
            ) {
                out.push(symbol);
            }
            named(children, out);
        }
    }
    let mut symbols = Vec::new();
    named(document_symbols(text, analysis), &mut symbols);
    for sym in symbols {
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
fn project_symbols_sync(project: &SourceContext, compat: CompatMode) -> Vec<SymbolInformation> {
    let mut out = Vec::new();
    for source in project.visible_source_files() {
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
    project: &SourceContext,
    compat: CompatMode,
) -> Vec<SymbolInformation> {
    let mut out = Vec::new();
    for source in project.visible_source_files() {
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
    uri_to_path(uri).and_then(|path| path.to_str().map(str::to_owned))
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

fn file_path_to_url(path: &Path) -> Option<Url> {
    path_to_uri(path)
}

/// The file path a `file` URI names.
///
/// A URI the host OS can turn into a local path maps to that path. Otherwise a `file` URI with
/// no host maps to its percent-decoded URI path, such as `/cloud/project/materials.fea`. Those
/// are virtual paths served by the installed `runmat_filesystem` provider; on Windows they have
/// no drive, so `Url::to_file_path` refuses them.
pub fn uri_to_path(uri: &Url) -> Option<PathBuf> {
    #[cfg(not(target_arch = "wasm32"))]
    if let Ok(path) = uri.to_file_path() {
        return Some(path);
    }
    virtual_uri_path(uri).map(PathBuf::from)
}

/// The `file` URI for a path.
///
/// A path the host OS can turn into a URI maps to that URI. Otherwise a rooted path with no
/// drive or prefix (a virtual provider path such as `/cloud/project/materials.fea`) maps to
/// `file://` plus its components joined by `/`, with `.` and `..` resolved. On wasm32 every
/// path is virtual, and a relative path is read from the provider root.
pub fn path_to_uri(path: &Path) -> Option<Url> {
    #[cfg(not(target_arch = "wasm32"))]
    if let Ok(uri) = Url::from_file_path(path) {
        return Some(uri);
    }
    virtual_path_uri(path)
}

/// The percent-decoded path of a host-less `file` URI.
pub(crate) fn virtual_uri_path(uri: &Url) -> Option<String> {
    if uri.scheme() != "file" || uri.host_str().is_some_and(|host| !host.is_empty()) {
        return None;
    }
    let encoded = uri.path().as_bytes();
    if encoded.is_empty() {
        return None;
    }
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
    String::from_utf8(decoded).ok()
}

fn hex_value(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

fn virtual_path_uri(path: &Path) -> Option<Url> {
    let mut rooted = cfg!(target_arch = "wasm32");
    let mut uri_path = String::new();
    for component in path.components() {
        match component {
            Component::Prefix(_) => return None,
            Component::RootDir => rooted = true,
            Component::CurDir => {}
            Component::ParentDir => uri_path.push_str("/.."),
            Component::Normal(part) => {
                uri_path.push('/');
                uri_path.push_str(part.to_str()?);
            }
        }
    }
    if !rooted {
        return None;
    }
    if uri_path.is_empty() {
        uri_path.push('/');
    }
    let mut uri = Url::parse("file:///").ok()?;
    // `set_path` percent-encodes the components and resolves the `..` segments.
    uri.set_path(&uri_path);
    Some(uri)
}

#[cfg(test)]
mod uri_path_tests {
    use super::*;

    #[test]
    fn a_virtual_path_round_trips_through_its_uri() {
        let path = Path::new("/cloud/project/materials.fea");
        let uri = path_to_uri(path).expect("uri");
        assert_eq!(uri.as_str(), "file:///cloud/project/materials.fea");
        assert_eq!(uri_to_path(&uri).as_deref(), Some(path));
    }

    #[test]
    fn a_space_is_percent_encoded_and_decoded() {
        let path = Path::new("/cloud/my project/unsaved file.fea");
        let uri = path_to_uri(path).expect("uri");
        assert_eq!(
            uri.as_str(),
            "file:///cloud/my%20project/unsaved%20file.fea"
        );
        assert_eq!(uri_to_path(&uri).as_deref(), Some(path));
        let parsed = Url::parse("file:///cloud/unsaved%20file.fea").unwrap();
        assert_eq!(
            uri_to_path(&parsed),
            Some(PathBuf::from("/cloud/unsaved file.fea"))
        );
    }

    #[test]
    fn the_virtual_fallback_reads_only_host_less_file_uris() {
        let untitled = Url::parse("untitled:///cloud/project/main.m").unwrap();
        assert_eq!(virtual_uri_path(&untitled), None);
        let remote = Url::parse("file://server/share/main.m").unwrap();
        assert_eq!(virtual_uri_path(&remote), None);
    }

    #[test]
    fn the_virtual_fallback_resolves_dot_segments() {
        let uri =
            virtual_path_uri(Path::new("/cloud/project/simulation/../materials.fea")).expect("uri");
        assert_eq!(uri.as_str(), "file:///cloud/project/materials.fea");
        assert_eq!(
            virtual_path_uri(Path::new("/")).unwrap().as_str(),
            "file:///"
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn a_relative_native_path_has_no_uri() {
        assert_eq!(path_to_uri(Path::new("cloud/project/materials.fea")), None);
    }

    #[cfg(windows)]
    #[test]
    fn a_windows_drive_path_uses_the_native_conversion() {
        let path = Path::new(r"C:\Users\me\project\materials.fea");
        let uri = path_to_uri(path).expect("uri");
        assert_eq!(uri, Url::from_file_path(path).unwrap());
        assert_eq!(uri.as_str(), "file:///C:/Users/me/project/materials.fea");
        assert_eq!(uri_to_path(&uri).as_deref(), Some(path));
    }

    #[cfg(windows)]
    #[test]
    fn a_windows_virtual_path_keeps_forward_slashes() {
        assert!(Url::from_file_path(Path::new("/cloud/project/materials.fea")).is_err());
        // `PathBuf::join` on Windows inserts a backslash; the URI still uses `/`.
        let joined = Path::new("/cloud/project").join("materials.fea");
        let uri = path_to_uri(&joined).expect("uri");
        assert_eq!(uri.as_str(), "file:///cloud/project/materials.fea");
        assert_eq!(
            uri_to_path(&uri),
            Some(PathBuf::from("/cloud/project/materials.fea"))
        );
    }

    #[cfg(windows)]
    #[test]
    fn a_windows_unc_path_uses_the_native_conversion() {
        let path = Path::new(r"\\server\share\materials.fea");
        let uri = path_to_uri(path).expect("uri");
        assert_eq!(uri, Url::from_file_path(path).unwrap());
        assert_eq!(uri_to_path(&uri).as_deref(), Some(path));
    }
}
