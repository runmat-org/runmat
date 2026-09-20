use serde_json::json;
use tower_lsp::lsp_types::{Diagnostic, DiagnosticSeverity, NumberOrString, Position, Range, Url};

use crate::core::fea::{is_fea_uri, FeaDocumentAnalysis};

use super::{RunMatLanguageServer, RunmatStatusNotification};

pub(super) async fn reanalyze(server: &RunMatLanguageServer, uri: &Url) -> bool {
    if !is_fea_uri(uri) {
        return false;
    }
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
    true
}

async fn reanalyze_document(server: &RunMatLanguageServer, uri: &Url) {
    let (text, roots, overlays) = {
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
        (
            document.text.clone(),
            state.workspace_roots.clone(),
            overlays,
        )
    };
    let result = uri
        .to_file_path()
        .map_err(|()| "FEA language tooling requires a file-backed document".to_string())
        .and_then(|path| FeaDocumentAnalysis::check_with_overlays(&text, &path, &roots, &overlays));
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
