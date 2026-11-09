//! Design documents (`.part`, `.assembly`, `.drawing`, `.circuit`, `.circuit.layout`) on the
//! native server: each open one is compiled against open documents and the files on disk it
//! reads, and every open design document recompiles when any of them changes (a layout
//! follows its circuit).

use tower_lsp::lsp_types::Url;

use crate::core::design::{
    design_dependencies, design_dialect, DesignDocumentAnalysis, DesignFiles, DesignValues,
};

use super::{RunMatLanguageServer, RunmatStatusNotification};

/// The most call results the server keeps; past this many, the oldest-keyed go first.
const DESIGN_VALUES_LIMIT: usize = 4096;

/// `runmat/designValues`: `.m` call results the host ran, by call key.
#[derive(Debug, serde::Deserialize)]
pub struct DesignValuesParams {
    pub values: DesignValues,
}

impl RunMatLanguageServer {
    /// Takes `.m` call results the host ran in the RunMat runtime (the editor runs no `.m`
    /// code) and recompiles the open design documents, so the hover shows computed values.
    pub async fn design_values(
        &self,
        params: DesignValuesParams,
    ) -> tower_lsp::jsonrpc::Result<()> {
        {
            let mut state = self.state.write().await;
            state.design_values.extend(params.values);
            while state.design_values.len() > DESIGN_VALUES_LIMIT {
                state.design_values.pop_first();
            }
        }
        reanalyze_open_documents(self).await;
        Ok(())
    }
}

/// Recompiles every open design document when `uri` is one; false otherwise.
pub(super) async fn reanalyze(server: &RunMatLanguageServer, uri: &Url) -> bool {
    if design_dialect(uri).is_none() {
        return false;
    }
    reanalyze_open_documents(server).await;
    true
}

pub(super) async fn reanalyze_open_documents(server: &RunMatLanguageServer) {
    let (documents, open, values) = {
        let state = server.state.read().await;
        let documents: Vec<(Url, String)> = state
            .documents
            .iter()
            .filter(|(uri, _)| design_dialect(uri).is_some())
            .map(|(uri, d)| (uri.clone(), d.text.clone()))
            .collect();
        let open: DesignFiles = state
            .documents
            .iter()
            .filter_map(|(uri, d)| {
                let path = uri.to_file_path().ok()?;
                Some((path.to_string_lossy().into_owned(), d.text.clone()))
            })
            .collect();
        (documents, open, state.design_values.clone())
    };
    for (uri, text) in documents {
        let Some(dialect) = design_dialect(&uri) else {
            continue;
        };
        let path = uri
            .to_file_path()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|_| uri.path().to_string());
        let files = gather(dialect, &path, &text, &open);
        let analysis = DesignDocumentAnalysis::analyze(&uri, dialect, &text, &files, &values);
        let diagnostics = analysis.diagnostics();
        let status = analysis.status_message();
        {
            let mut state = server.state.write().await;
            match state.documents.get_mut(&uri) {
                Some(document) if document.text == text => {
                    document.design_analysis = Some(analysis);
                }
                _ => continue,
            }
        }
        let _ = server
            .client
            .send_notification::<RunmatStatusNotification>(serde_json::json!({ "message": status }))
            .await;
        server
            .client
            .publish_diagnostics(uri.clone(), diagnostics, None)
            .await;
    }
}

/// The files a design source reads, open documents first, then the disk, until nothing new
/// is named.
fn gather(
    dialect: runmat_design_syntax::Dialect,
    path: &str,
    text: &str,
    open: &DesignFiles,
) -> DesignFiles {
    let mut files = DesignFiles::new();
    let mut queue: Vec<(runmat_design_syntax::Dialect, String, String)> =
        vec![(dialect, path.to_string(), text.to_string())];
    let mut seen = std::collections::BTreeSet::from([path.to_string()]);
    while let Some((dialect, at, text)) = queue.pop() {
        for named in design_dependencies(dialect, &at, &text) {
            if !seen.insert(named.clone()) {
                continue;
            }
            for dependency in folder_files(&named, open) {
                if dependency != named && !seen.insert(dependency.clone()) {
                    continue;
                }
                let content = open
                    .get(&dependency)
                    .cloned()
                    .or_else(|| std::fs::read_to_string(&dependency).ok());
                let Some(content) = content else { continue };
                if let Some(next) = runmat_design_syntax::Dialect::for_path(&dependency) {
                    if next != runmat_design_syntax::Dialect::Fea {
                        queue.push((next, dependency.clone(), content.clone()));
                    }
                }
                files.insert(dependency, content);
            }
        }
    }
    files
}

/// The files a dependency stands for: itself, or for a folder (`/p/cabin/`) its design and
/// `.m` files on disk and among the open documents.
fn folder_files(dependency: &str, open: &DesignFiles) -> Vec<String> {
    let Some(folder) = dependency.strip_suffix('/') else {
        return vec![dependency.to_string()];
    };
    let wanted = |p: &str| {
        runmat_cad::modules::kind_of_file(p).is_some()
            || p.ends_with(".m")
            || p.ends_with("/runmat.toml")
    };
    let mut out: std::collections::BTreeSet<String> = open
        .keys()
        .filter(|p| runmat_cad::modules::folder_of(p) == folder && wanted(p))
        .cloned()
        .collect();
    if let Ok(entries) = std::fs::read_dir(if folder.is_empty() { "/" } else { folder }) {
        for entry in entries.flatten() {
            let p = entry.path().to_string_lossy().to_string();
            if entry.path().is_file() && wanted(&p) {
                out.insert(p);
            }
        }
    }
    out.into_iter().collect()
}
