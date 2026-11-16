use super::{wasm_file_path, DOCS, WORKSPACE_REVISION};
use lsp_types::Url;
use runmat_filesystem::CapturedReadScope;
use std::{cell::Cell, future::Future, path::Path};
use wasm_bindgen::JsValue;

/// One host provider and immutable open-file snapshot for a semantic request.
pub(super) struct SourceWorkspace {
    scope: CapturedReadScope,
    revision: u64,
}

impl SourceWorkspace {
    pub(super) fn capture() -> Result<Self, JsValue> {
        let revision = WORKSPACE_REVISION.with(Cell::get);
        let sources = DOCS.with(|documents| {
            let mut sources = std::collections::BTreeMap::new();
            for (uri, document) in &documents.borrow().docs {
                let Some(path) = Url::parse(uri).ok().as_ref().and_then(wasm_file_path) else {
                    continue;
                };
                if sources
                    .insert(path, document.text.clone())
                    .is_some_and(|old| old != document.text)
                {
                    return Err(JsValue::from_str(
                        "Open documents have conflicting text for the same source path",
                    ));
                }
            }
            Ok(sources)
        })?;
        let (scope, _) = CapturedReadScope::capture(&[Path::new(".")], None)
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
        let scope = scope
            .with_virtual_sources(sources)
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
        Ok(Self { scope, revision })
    }

    pub(super) async fn run<F: Future>(&self, future: F) -> Result<Option<F::Output>, JsValue> {
        let output = self
            .scope
            .run(future)
            .await
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
        Ok((WORKSPACE_REVISION.with(Cell::get) == self.revision).then_some(output))
    }
}
