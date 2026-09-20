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
        Ok(path) => match FeaDocumentAnalysis::check_with_overlays(&text, &path, &roots, &overlays)
        {
            Ok(preliminary) => {
                match resolve_project_facts(&preliminary, &path, compat, &source_overlays) {
                    Ok(facts) => FeaDocumentAnalysis::check_with_project_facts(
                        &text,
                        &path,
                        &roots,
                        &overlays,
                        facts.functions,
                        facts.profiles,
                        facts.function_sources,
                    ),
                    Err(error) => Err(error),
                }
            }
            Err(error) => Err(error),
        },
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
