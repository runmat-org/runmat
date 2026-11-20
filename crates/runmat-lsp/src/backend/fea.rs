use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::json;
use tower_lsp::lsp_types::{Diagnostic, DiagnosticSeverity, NumberOrString, Position, Range, Url};

use crate::core::fea::{analyze_fea_source, is_fea_uri, FeaDocumentAnalysis};
use crate::core::workspace::uri_to_path;

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
    let (workspace_roots, source_overlays, compat, open_documents) = {
        let state = server.state.read().await;
        let source_overlays = state
            .documents
            .iter()
            .filter_map(|(uri, document)| Some((uri_to_path(uri)?, document.text.clone())))
            .collect::<BTreeMap<_, _>>();
        let open_documents = state
            .documents
            .iter()
            .filter(|(uri, _)| is_fea_uri(uri))
            .filter_map(|(uri, document)| Some((uri_to_path(uri)?, document.fea_analysis.clone())))
            .collect::<Vec<_>>();
        (
            state.workspace_roots.clone(),
            Arc::new(source_overlays),
            state.compat_mode,
            open_documents,
        )
    };

    // The folders rename searches for studies that aren't open. Each study is still checked
    // against the same source root as an open one.
    let mut scan_roots = workspace_roots.clone();
    if scan_roots.is_empty() {
        for path in source_overlays.keys() {
            if let Some(project) = super::ProjectContext::discover_from_source_name(path.to_str()) {
                scan_roots.extend(project.manifest_path().parent().map(Path::to_path_buf));
                break;
            }
        }
    }
    if scan_roots.is_empty() {
        scan_roots.extend(
            source_overlays
                .keys()
                .filter_map(|path| path.parent().map(Path::to_path_buf)),
        );
        scan_roots.sort();
        scan_roots.dedup();
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
    for path in workspace_fea_paths(&scan_roots)? {
        if open_paths.contains(&path) {
            continue;
        }
        let source = runmat_filesystem::read_to_string(&path).map_err(|error| {
            format!(
                "rename cannot inspect FEA source {}: {error}",
                path.display()
            )
        })?;
        let analysis = check_study(
            source,
            path.clone(),
            workspace_roots.clone(),
            source_overlays.clone(),
            compat,
        )
        .await
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
    let (text, roots, source_overlays, compat) = {
        let state = server.state.read().await;
        let Some(document) = state.documents.get(uri) else {
            return;
        };
        let source_overlays = state
            .documents
            .iter()
            .filter_map(|(document_uri, document)| {
                Some((uri_to_path(document_uri)?, document.text.clone()))
            })
            .collect::<BTreeMap<_, _>>();
        (
            document.text.clone(),
            state.workspace_roots.clone(),
            Arc::new(source_overlays),
            state.compat_mode,
        )
    };
    let result = match uri_to_path(uri) {
        Some(path) => check_study(text.clone(), path, roots, source_overlays, compat).await,
        None => Err("FEA language tooling requires a file-backed document".to_string()),
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

/// Checks the study at `path` with the shared [`analyze_fea_source`] on a blocking thread.
///
/// The check reads through the `runmat_filesystem` provider, whose futures are not `Send` (a
/// browser provider awaits JavaScript promises). The native server's request handlers must stay
/// `Send`, so it drives the check to completion off the async runtime and awaits only the
/// thread's result.
async fn check_study(
    source: String,
    path: PathBuf,
    workspace_roots: Vec<PathBuf>,
    source_overlays: Arc<BTreeMap<PathBuf, String>>,
    compat: crate::core::analysis::CompatMode,
) -> Result<FeaDocumentAnalysis, String> {
    tokio::task::spawn_blocking(move || {
        futures::executor::block_on(analyze_fea_source(
            &source,
            &path,
            &workspace_roots,
            &source_overlays,
            compat,
        ))
    })
    .await
    .map_err(|error| format!("the FEA check stopped: {error}"))?
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

    use tower_lsp::lsp_types::{
        DidChangeWorkspaceFoldersParams, DidOpenTextDocumentParams, InitializeParams,
        TextDocumentItem, WorkspaceFolder, WorkspaceFoldersChangeEvent,
    };
    use tower_lsp::LanguageServer;

    const STEEL: &str = "material steel {\n    mass_density = 7850 kg/m^3\n    elastic {\n        youngs_modulus = 200 GPa\n        poisson_ratio = 0.3\n    }\n}\n";

    /// `project/materials.fea`, `outside.fea` above the project, and `project/simulation/`.
    fn project_layout() -> (tempfile::TempDir, PathBuf, PathBuf) {
        let temporary = tempfile::tempdir().unwrap();
        let project = temporary.path().join("project");
        let simulation = project.join("simulation");
        std::fs::create_dir_all(&simulation).unwrap();
        std::fs::write(project.join("materials.fea"), STEEL).unwrap();
        std::fs::write(temporary.path().join("outside.fea"), STEEL).unwrap();
        (temporary, project, simulation)
    }

    fn folder(path: &Path) -> WorkspaceFolder {
        WorkspaceFolder {
            uri: Url::from_directory_path(path).unwrap(),
            name: path.file_name().unwrap().to_string_lossy().into_owned(),
        }
    }

    async fn open_study(server: &RunMatLanguageServer, path: &Path, text: &str) -> Url {
        let uri = Url::from_file_path(path).unwrap();
        server
            .did_open(DidOpenTextDocumentParams {
                text_document: TextDocumentItem {
                    uri: uri.clone(),
                    language_id: "runmat-fea".into(),
                    version: 1,
                    text: text.into(),
                },
            })
            .await;
        uri
    }

    /// Whether the open study's import was refused as escaping its source root.
    async fn import_refused(server: &RunMatLanguageServer, uri: &Url) -> bool {
        let state = server.state.read().await;
        let document = &state.documents[uri];
        let analysis = document.fea_analysis.as_ref().expect("checked study");
        analysis
            .diagnostics(&document.text)
            .iter()
            .any(|diagnostic| {
                diagnostic.code == Some(NumberOrString::String("fea.module.invalid_path".into()))
                    && diagnostic
                        .message
                        .contains("escapes the declared source root")
            })
    }

    async fn initialize(server: &RunMatLanguageServer, root: &Path) {
        let params = InitializeParams {
            root_uri: Some(Url::from_directory_path(root).unwrap()),
            ..InitializeParams::default()
        };
        server.initialize(params).await.unwrap();
    }

    #[test]
    fn initialize_root_bounds_what_a_study_may_read() {
        let (_temporary, project, simulation) = project_layout();
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(async {
                let (service, _socket) = crate::native::service();
                let server = service.inner();
                initialize(server, &project).await;
                let inside = open_study(
                    server,
                    &simulation.join("inside.fea"),
                    "import material steel from \"../materials.fea\"\n",
                )
                .await;
                let above = open_study(
                    server,
                    &simulation.join("above.fea"),
                    "import material steel from \"../../outside.fea\"\n",
                )
                .await;
                assert!(!import_refused(server, &inside).await);
                assert!(import_refused(server, &above).await);
            });
    }

    #[test]
    fn workspace_folder_changes_recheck_open_studies() {
        let (_temporary, project, simulation) = project_layout();
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(async {
                let (service, _socket) = crate::native::service();
                let server = service.inner();
                initialize(server, &simulation).await;
                let study = open_study(
                    server,
                    &simulation.join("study.fea"),
                    "import material steel from \"../materials.fea\"\n",
                )
                .await;
                assert!(import_refused(server, &study).await);

                server
                    .did_change_workspace_folders(DidChangeWorkspaceFoldersParams {
                        event: WorkspaceFoldersChangeEvent {
                            added: vec![folder(&project)],
                            removed: vec![folder(&simulation)],
                        },
                    })
                    .await;
                assert_eq!(
                    server.state.read().await.workspace_roots,
                    vec![project.clone()]
                );
                assert!(!import_refused(server, &study).await);

                server
                    .did_change_workspace_folders(DidChangeWorkspaceFoldersParams {
                        event: WorkspaceFoldersChangeEvent {
                            added: vec![folder(&simulation)],
                            removed: vec![folder(&project)],
                        },
                    })
                    .await;
                assert!(import_refused(server, &study).await);
            });
    }
}
