use runmat_execution_artifact::encryption::{
    EncryptedArtifact, EncryptionContext, PortableExecutionEncryption, PortableExecutionPrivateKey,
};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct BrowserExecutionRecipient {
    provider: PortableExecutionEncryption,
    recipient: runmat_execution_artifact::encryption::ExecutionRecipientKey,
    private_key: PortableExecutionPrivateKey,
}

#[wasm_bindgen]
impl BrowserExecutionRecipient {
    #[wasm_bindgen(constructor)]
    pub fn new(
        fingerprint: String,
        valid_after_unix_millis: u64,
        valid_before_unix_millis: u64,
    ) -> Result<BrowserExecutionRecipient, JsValue> {
        let provider = PortableExecutionEncryption;
        let (recipient, private_key) = provider
            .recipient_from_entropy(
                browser_entropy()?,
                fingerprint,
                valid_after_unix_millis,
                valid_before_unix_millis,
            )
            .map_err(js_error)?;
        Ok(Self {
            provider,
            recipient,
            private_key,
        })
    }

    #[wasm_bindgen(getter)]
    pub fn recipient(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.recipient).map_err(js_error)
    }

    pub fn seal(&self, context: JsValue, plaintext: Vec<u8>) -> Result<JsValue, JsValue> {
        let context: EncryptionContext =
            serde_wasm_bindgen::from_value(context).map_err(js_error)?;
        let artifact = self
            .provider
            .seal_with_entropy(browser_entropy()?, &self.recipient, context, &plaintext)
            .map_err(js_error)?;
        serde_wasm_bindgen::to_value(&artifact).map_err(js_error)
    }

    pub fn open(&self, artifact: JsValue) -> Result<Vec<u8>, JsValue> {
        let artifact: EncryptedArtifact =
            serde_wasm_bindgen::from_value(artifact).map_err(js_error)?;
        self.provider
            .open(&self.private_key, &artifact)
            .map_err(js_error)
    }
}

#[wasm_bindgen(js_name = buildExecutionBundle)]
pub async fn build_execution_bundle(
    project_handoff: JsValue,
    recipe: JsValue,
    executable_bytes: Vec<u8>,
) -> Result<Vec<u8>, JsValue> {
    let handoff: runmat_package::FrozenProjectHandoff =
        serde_wasm_bindgen::from_value(project_handoff).map_err(js_error)?;
    handoff.validate().map_err(js_error)?;
    let recipe: runmat_execution_artifact::ProgramBuildRecipe =
        serde_wasm_bindgen::from_value(recipe).map_err(js_error)?;
    let mut source_bytes = std::collections::BTreeMap::new();
    for path in handoff.project.access_paths.values() {
        source_bytes.insert(
            path.clone(),
            runmat_filesystem::read_async(path)
                .await
                .map_err(js_error)?,
        );
    }
    let revision = recipe.program_revision.clone();
    let reader = move |path: &std::path::Path| {
        source_bytes.get(path).cloned().ok_or_else(|| {
            runmat_execution_artifact::ArtifactError::Invalid(format!(
                "browser source is absent from the frozen project: {}",
                path.display()
            ))
        })
    };
    let bundle =
        runmat_execution_artifact::ExecutionBundleBuilder::new(&handoff.project, revision, reader)
            .map_err(js_error)?
            .with_materialized_program(
                recipe,
                runmat_execution_artifact::ExecutableForm::InterpreterBytecodeV1,
                executable_bytes,
            )
            .build()
            .map_err(js_error)?;
    let mut archive = Vec::new();
    runmat_execution_artifact::archive::write_bundle(
        &bundle,
        &mut archive,
        runmat_execution_artifact::archive::ArchiveLimits::default(),
    )
    .map_err(js_error)?;
    Ok(archive)
}

fn browser_entropy() -> Result<[u8; 32], JsValue> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("window is unavailable"))?;
    let crypto = window.crypto()?;
    let mut entropy = [0_u8; 32];
    crypto.get_random_values_with_u8_array(&mut entropy)?;
    Ok(entropy)
}

fn js_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}
