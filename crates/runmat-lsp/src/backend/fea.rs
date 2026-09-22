use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use serde_json::json;
use tower_lsp::lsp_types::{Diagnostic, DiagnosticSeverity, NumberOrString, Position, Range, Url};

use crate::core::fea::{is_fea_uri, resolve_project_facts, FeaDocumentAnalysis};

use super::{RunMatLanguageServer, RunmatStatusNotification};

pub(super) async fn reanalyze(server: &RunMatLanguageServer, uri: &Url) -> bool {
    if !is_fea_uri(uri) {
        return false;
    }
    reanalyze_open_documents(server).await;
    true
}

pub(super) async fn reanalyze_open_documents(server: &RunMatLanguageServer) {
    let documents = server
        .state
        .read()
        .await
        .documents
        .keys()
        .filter(|document_uri| is_fea_uri(document_uri))
        .cloned()
        .collect::<Vec<_>>();
    for document_uri in documents {
        reanalyze_document(server, &document_uri).await;
    }
}

pub(super) async fn workspace_analyses(
    server: &RunMatLanguageServer,
) -> Result<Vec<FeaDocumentAnalysis>, String> {
    let (mut workspace_roots, overlays, source_overlays, compat, open_documents) = {
        let state = server.state.read().await;
        let overlays = state
            .documents
            .iter()
            .filter(|(uri, _)| is_fea_uri(uri))
            .filter_map(|(uri, document)| {
                Some((uri.to_file_path().ok()?, document.text.as_bytes().to_vec()))
            })
            .collect::<Vec<_>>();
        let source_overlays = state
            .documents
            .iter()
            .filter_map(|(uri, document)| Some((uri.to_file_path().ok()?, document.text.clone())))
            .collect::<BTreeMap<_, _>>();
        let open_documents = state
            .documents
            .iter()
            .filter(|(uri, _)| is_fea_uri(uri))
            .filter_map(|(uri, document)| {
                Some((uri.to_file_path().ok()?, document.fea_analysis.clone()))
            })
            .collect::<Vec<_>>();
        (
            state.workspace_roots.clone(),
            overlays,
            source_overlays,
            state.compat_mode,
            open_documents,
        )
    };

    if workspace_roots.is_empty() {
        workspace_roots.extend(
            source_overlays
                .keys()
                .find_map(|path| {
                    super::ProjectContext::discover_from_source_name(path.to_str())
                        .and_then(|project| project.manifest_path().parent().map(Path::to_path_buf))
                })
                .into_iter(),
        );
    }
    if workspace_roots.is_empty() {
        workspace_roots.extend(
            source_overlays
                .keys()
                .filter_map(|path| path.parent().map(Path::to_path_buf)),
        );
        workspace_roots.sort();
        workspace_roots.dedup();
    }

    let analyses = open_documents
        .iter()
        .map(|(_, analysis)| analysis.clone())
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            "rename requires every open `.fea` document to have a valid checked graph".to_string()
        })?;
    let open_paths = open_documents
        .into_iter()
        .map(|(path, _)| path)
        .collect::<HashSet<_>>();
    let mut analyses = analyses;
    for path in workspace_fea_paths(&workspace_roots)? {
        if open_paths.contains(&path) {
            continue;
        }
        let source = runmat_filesystem::read_to_string(&path).map_err(|error| {
            format!(
                "rename cannot inspect FEA source {}: {error}",
                path.display()
            )
        })?;
        let analysis = analyze_source(
            &source,
            &path,
            &workspace_roots,
            &overlays,
            &source_overlays,
            compat,
        )
        .map_err(|error| {
            format!(
                "rename cannot verify FEA source {}: {error}",
                path.display()
            )
        })?;
        analyses.push(analysis);
    }
    Ok(analyses)
}

async fn reanalyze_document(server: &RunMatLanguageServer, uri: &Url) {
    let (text, roots, overlays, source_overlays, compat) = {
        let state = server.state.read().await;
        let Some(document) = state.documents.get(uri) else {
            return;
        };
        let overlays = state
            .documents
            .iter()
            .filter(|(document_uri, _)| is_fea_uri(document_uri))
            .filter_map(|(document_uri, document)| {
                Some((
                    document_uri.to_file_path().ok()?,
                    document.text.as_bytes().to_vec(),
                ))
            })
            .collect::<Vec<_>>();
        let source_overlays = state
            .documents
            .iter()
            .filter_map(|(document_uri, document)| {
                Some((document_uri.to_file_path().ok()?, document.text.clone()))
            })
            .collect::<std::collections::BTreeMap<_, _>>();
        (
            document.text.clone(),
            state.workspace_roots.clone(),
            overlays,
            source_overlays,
            state.compat_mode,
        )
    };
    let result = match uri.to_file_path() {
        Ok(path) => analyze_source(&text, &path, &roots, &overlays, &source_overlays, compat),
        Err(()) => Err("FEA language tooling requires a file-backed document".to_string()),
    };
    let (analysis, error) = match result {
        Ok(analysis) => (Some(analysis), None),
        Err(error) => (None, Some(error)),
    };
    {
        let mut state = server.state.write().await;
        let Some(document) = state.documents.get_mut(uri) else {
            return;
        };
        document.analysis = None;
        document.fea_analysis = analysis.clone();
    }
    let message = analysis
        .as_ref()
        .map(FeaDocumentAnalysis::status_message)
        .or_else(|| error.clone())
        .unwrap_or_else(|| "FEA analysis unavailable".to_string());
    let _ = server
        .client
        .send_notification::<RunmatStatusNotification>(json!({ "message": message }))
        .await;
    let diagnostics = match analysis {
        Some(analysis) => analysis.diagnostics(&text),
        None => vec![unavailable_diagnostic(error)],
    };
    server
        .client
        .publish_diagnostics(uri.clone(), diagnostics, None)
        .await;
}

fn analyze_source(
    source: &str,
    path: &Path,
    workspace_roots: &[PathBuf],
    overlays: &[(PathBuf, Vec<u8>)],
    source_overlays: &BTreeMap<PathBuf, String>,
    compat: crate::core::analysis::CompatMode,
) -> Result<FeaDocumentAnalysis, String> {
    let prepared =
        FeaDocumentAnalysis::prepare_with_overlays(source, path, workspace_roots, overlays)?;
    let requests = prepared.function_requests()?;
    let facts = resolve_project_facts(
        prepared.source_root(),
        &requests,
        path,
        compat,
        source_overlays,
    )?;
    prepared
        .check(facts.functions, facts.profiles)
        .map(|checked| FeaDocumentAnalysis::from_checked(checked, facts.function_sources))
}

fn workspace_fea_paths(roots: &[PathBuf]) -> Result<Vec<PathBuf>, String> {
    let mut paths = Vec::new();
    let mut pending = roots.to_vec();
    while let Some(path) = pending.pop() {
        let metadata = runmat_filesystem::symlink_metadata(&path).map_err(|error| {
            format!(
                "rename cannot inspect workspace path {}: {error}",
                path.display()
            )
        })?;
        if metadata.is_symlink() {
            continue;
        }
        if metadata.is_file() {
            if path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("fea"))
            {
                paths.push(path);
            }
            continue;
        }
        if !metadata.is_dir() || ignored_workspace_directory(&path) {
            continue;
        }
        let entries = runmat_filesystem::read_dir(&path).map_err(|error| {
            format!(
                "rename cannot enumerate workspace directory {}: {error}",
                path.display()
            )
        })?;
        pending.extend(entries.into_iter().map(|entry| entry.path().to_path_buf()));
    }
    paths.sort();
    paths.dedup();
    Ok(paths)
}

fn ignored_workspace_directory(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(".git" | ".hg" | ".svn" | "node_modules" | "target")
    )
}

fn unavailable_diagnostic(error: Option<String>) -> Diagnostic {
    Diagnostic {
        range: Range::new(Position::new(0, 0), Position::new(0, 0)),
        severity: Some(DiagnosticSeverity::ERROR),
        code: Some(NumberOrString::String("fea.host.unavailable".to_string())),
        source: Some("runmat-fea".to_string()),
        message: error.unwrap_or_else(|| "FEA analysis unavailable".to_string()),
        ..Diagnostic::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_scan_finds_fea_sources_and_skips_build_trees() {
        let workspace = tempfile::tempdir().unwrap();
        let studies = workspace.path().join("studies");
        let nested = studies.join("nested");
        let target = workspace.path().join("target");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(studies.join("root.fea"), "").unwrap();
        std::fs::write(nested.join("module.FEA"), "").unwrap();
        std::fs::write(nested.join("signal.m"), "").unwrap();
        std::fs::write(target.join("generated.fea"), "").unwrap();

        assert_eq!(
            workspace_fea_paths(&[workspace.path().to_path_buf()]).unwrap(),
            vec![nested.join("module.FEA"), studies.join("root.fea")]
        );
    }
}
