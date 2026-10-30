use super::provider::JsPackageSourceProvider;
use crate::api::session::RunMatWasm;
use wasm_bindgen::prelude::*;

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CapturedRequest {
    manifest_path: String,
    input_manifest: runmat_filesystem::pinned::InputManifest,
    options: runmat_package::ProjectResolveOptions,
}

#[wasm_bindgen]
impl RunMatWasm {
    #[wasm_bindgen(js_name = prepareCapturedProject)]
    pub async fn prepare_captured_project(
        &self,
        request: JsValue,
        provider: JsValue,
        filesystem: JsValue,
    ) -> Result<JsValue, JsValue> {
        self.ensure_not_disposed()?;
        let request: CapturedRequest = serde_wasm_bindgen::from_value(request)
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
        let provider = JsPackageSourceProvider::new(provider)?;
        // Install one provider owner, including the session's dialog service.
        // The captured project retains this exact owner for later admission.
        self.set_fs_provider(filesystem)?;
        let workspace = runmat_filesystem::current_provider();
        let mut session = self.take_session()?;
        let captured = runmat_package::resolve_captured_project_async(
            std::path::Path::new(&request.manifest_path),
            request.input_manifest,
            request.options,
            &provider,
            workspace.clone(),
            workspace,
        )
        .await
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
        // Acquire the durable lease before installing execution authority. The
        // browser owner renews/releases this lease through the existing resolver.
        let receipt = super::finish_resolution(captured.project.clone(), &provider).await?;
        session
            .install_captured_project(captured)
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
        Ok(receipt)
    }
}
