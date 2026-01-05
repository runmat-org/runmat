use lsp_types::{SymbolInformation, Url};
use serde_json::json;

use crate::core::analysis::{document_symbols, DocumentAnalysis};

/// Build workspace symbols by aggregating document symbols across open docs.
/// Uses tags instead of the deprecated `deprecated` field.
pub fn workspace_symbols(docs: &[(Url, String, DocumentAnalysis)]) -> Vec<SymbolInformation> {
    let mut out = Vec::new();
    for (uri, text, analysis) in docs {
        for sym in document_symbols(text, analysis) {
            // Construct via serde to avoid touching the deprecated field.
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
    out
}
