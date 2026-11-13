//! Browser orchestration of filesystem-owned scopes; no provider setter.
use crate::wire::errors::js_error;
use runmat_filesystem::{CapturedExecutionScope, CapturedPinnedScope, CapturedReadScope};
use std::{future::Future, path::Path};
use wasm_bindgen::prelude::*;

pub(crate) struct CapturedFilesystem {
    pub(crate) read: CapturedReadScope,
    pub(crate) pinned: Option<CapturedPinnedScope>,
    original: CapturedExecutionScope,
}

impl CapturedFilesystem {
    pub(crate) fn new(
        original: CapturedReadScope,
        pinned: Option<CapturedPinnedScope>,
    ) -> Result<Self, JsValue> {
        let execution = original
            .execution()
            .map_err(|error| js_error(&error.to_string()))?;
        let read = match &pinned {
            Some(scope) => scope
                .read_scope()
                .map_err(|error| js_error(&error.to_string()))?,
            None => original.clone(),
        };
        Ok(Self {
            read,
            pinned,
            original: execution,
        })
    }

    pub(crate) fn simulation(path: &str, inputs: JsValue) -> Result<(Self, String), JsValue> {
        let inputs: Option<runmat_filesystem::pinned::InputManifest> = if inputs.is_null()
            || inputs.is_undefined()
        {
            None
        } else {
            Some(
                serde_wasm_bindgen::from_value(inputs)
                    .map_err(|error| js_error(&format!("Invalid simulation inputs: {error}")))?,
            )
        };
        let (original, paths) =
            CapturedReadScope::capture(&[Path::new(path), Path::new(".")], None)
                .map_err(|error| js_error(&error.to_string()))?;
        let pinned = inputs
            .map(|inputs| {
                let provider = original.pin_inputs(inputs)?;
                original.pinned(provider)
            })
            .transpose()
            .map_err(|error| js_error(&error.to_string()))?;
        let path = paths[0]
            .to_str()
            .ok_or_else(|| js_error("Simulation path is not UTF-8"))?
            .to_owned();
        Ok((Self::new(original, pinned)?, path))
    }

    pub(crate) async fn publication_supported(&self) -> Result<bool, JsValue> {
        self.run_original(async {
            runmat_filesystem::current_provider().supports_workspace_transactions()
        })
        .await
    }

    pub(crate) async fn run<F: Future>(&self, future: F) -> Result<F::Output, JsValue> {
        let result = match &self.pinned {
            Some(scope) => scope.run(future).await,
            None => self.original.run(future).await,
        };
        result.map_err(|error| js_error(&error.to_string()))
    }

    pub(crate) async fn run_original<F: Future>(&self, future: F) -> Result<F::Output, JsValue> {
        self.original
            .run(future)
            .await
            .map_err(|error| js_error(&error.to_string()))
    }
}
