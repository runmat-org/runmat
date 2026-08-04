use runmat_execution_artifact::encryption::{
    decode_transfer_wire_frame, encode_transfer_wire_frame, open_transfer_frame,
    seal_transfer_frame, EncryptedArtifact, EncryptionContext, PortableExecutionEncryption,
    PortableExecutionPrivateKey, RunKeyEnvelope, RunKeyMaterial, RunObjectEncryption,
    TransferFrameAuthority, TransferWireFrame,
};
use std::collections::HashSet;
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

    #[wasm_bindgen(js_name = openRunKey)]
    pub fn open_run_key(
        &self,
        envelope: JsValue,
        expected_run_identity: String,
        expected_key_epoch: u32,
    ) -> Result<BrowserRunKey, JsValue> {
        let envelope: RunKeyEnvelope =
            serde_wasm_bindgen::from_value(envelope).map_err(js_error)?;
        let key = self
            .provider
            .open_run_key(
                &self.private_key,
                &envelope,
                &self.recipient.fingerprint,
                &expected_run_identity,
                expected_key_epoch,
            )
            .map_err(js_error)?;
        Ok(BrowserRunKey { key })
    }
}

/// Browser-owned content key for one remote run. JavaScript can use it to
/// encrypt/decrypt objects and create recipient envelopes, but cannot extract
/// the raw key bytes.
#[wasm_bindgen]
pub struct BrowserRunKey {
    pub(super) key: RunKeyMaterial,
}

#[wasm_bindgen]
impl BrowserRunKey {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<BrowserRunKey, JsValue> {
        Ok(Self {
            key: RunKeyMaterial::from_entropy(browser_entropy()?).map_err(js_error)?,
        })
    }

    #[wasm_bindgen(js_name = sealForRecipient)]
    pub fn seal_for_recipient(
        &self,
        recipient: JsValue,
        run_identity: String,
        key_epoch: u32,
    ) -> Result<JsValue, JsValue> {
        let recipient = serde_wasm_bindgen::from_value(recipient).map_err(js_error)?;
        let envelope = PortableExecutionEncryption
            .seal_run_key_with_entropy(
                browser_entropy()?,
                &recipient,
                &self.key,
                run_identity,
                key_epoch,
            )
            .map_err(js_error)?;
        serde_wasm_bindgen::to_value(&envelope).map_err(js_error)
    }

    pub fn seal(&self, context: JsValue, plaintext: Vec<u8>) -> Result<JsValue, JsValue> {
        let context: EncryptionContext =
            serde_wasm_bindgen::from_value(context).map_err(js_error)?;
        let object = RunObjectEncryption
            .seal_with_entropy(&self.key, browser_entropy()?, context, &plaintext)
            .map_err(js_error)?;
        serde_wasm_bindgen::to_value(&object).map_err(js_error)
    }

    pub fn open(&self, object: JsValue) -> Result<Vec<u8>, JsValue> {
        let object = serde_wasm_bindgen::from_value(object).map_err(js_error)?;
        RunObjectEncryption
            .open(&self.key, &object)
            .map_err(js_error)
    }

    #[wasm_bindgen(js_name = createFrameSession)]
    pub fn create_frame_session(
        &self,
        run_identity: String,
        session_id: Vec<u8>,
        direction: String,
        key_epoch: u32,
    ) -> Result<BrowserEncryptedFrameSession, JsValue> {
        let session_id = session_id
            .try_into()
            .map_err(|_| JsValue::from_str("execution frame session id must contain 16 bytes"))?;
        BrowserEncryptedFrameSession::new(
            run_identity,
            session_id,
            direction,
            key_epoch,
            self.key.clone(),
        )
    }
}

/// Rust-owned application encryption/replay state for a browser WebSocket or
/// WebTransport host. JavaScript transports the returned opaque bytes without
/// receiving the run key or implementing protocol crypto.
#[wasm_bindgen]
pub struct BrowserEncryptedFrameSession {
    run_identity: String,
    session_id: [u8; 16],
    direction: String,
    key_epoch: u32,
    key: RunKeyMaterial,
    next_sequence: u64,
    highest_received: Option<u64>,
    received_window: u64,
    used_salts: HashSet<[u8; 32]>,
}

impl BrowserEncryptedFrameSession {
    const MAXIMUM_PAYLOAD_BYTES: usize = 8 * 1024 * 1024;
    const MAXIMUM_FRAME_BYTES: usize = Self::MAXIMUM_PAYLOAD_BYTES + 128;

    fn new(
        run_identity: String,
        session_id: [u8; 16],
        direction: String,
        key_epoch: u32,
        key: RunKeyMaterial,
    ) -> Result<Self, JsValue> {
        if run_identity.is_empty()
            || run_identity.len() > 256
            || direction.is_empty()
            || direction.len() > 64
            || !run_identity.is_ascii()
            || !direction.is_ascii()
            || key_epoch == 0
        {
            return Err(JsValue::from_str("execution frame authority is malformed"));
        }
        Ok(Self {
            run_identity,
            session_id,
            direction,
            key_epoch,
            key,
            next_sequence: 0,
            highest_received: None,
            received_window: 0,
            used_salts: HashSet::new(),
        })
    }

    fn authority(&self, frame_kind: u8, sequence: u64) -> TransferFrameAuthority<'_> {
        TransferFrameAuthority {
            run_identity: &self.run_identity,
            session_id: self.session_id,
            direction: &self.direction,
            frame_kind,
            sequence,
            key_epoch: self.key_epoch,
        }
    }

    fn accept_sequence(&mut self, sequence: u64) -> Result<(), JsValue> {
        let Some(highest) = self.highest_received else {
            self.highest_received = Some(sequence);
            self.received_window = 1;
            return Ok(());
        };
        if sequence > highest {
            let shift = sequence - highest;
            self.received_window = if shift >= 64 {
                1
            } else {
                (self.received_window << shift) | 1
            };
            self.highest_received = Some(sequence);
            return Ok(());
        }
        let distance = highest - sequence;
        if distance >= 64 || self.received_window & (1_u64 << distance) != 0 {
            return Err(JsValue::from_str("execution frame was replayed"));
        }
        self.received_window |= 1_u64 << distance;
        Ok(())
    }
}

#[wasm_bindgen]
impl BrowserEncryptedFrameSession {
    pub fn seal(&mut self, frame_kind: u8, plaintext: Vec<u8>) -> Result<Vec<u8>, JsValue> {
        if frame_kind > 3 {
            return Err(JsValue::from_str("execution frame kind is unsupported"));
        }
        let salt = browser_entropy()?;
        if !self.used_salts.insert(salt) {
            return Err(JsValue::from_str(
                "execution frame derivation salt was reused",
            ));
        }
        let sequence = self.next_sequence;
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or_else(|| JsValue::from_str("execution frame sequence overflowed"))?;
        let encrypted_payload = seal_transfer_frame(
            &self.key,
            &self.authority(frame_kind, sequence),
            salt,
            &plaintext,
            Self::MAXIMUM_PAYLOAD_BYTES,
        )
        .map_err(js_error)?;
        encode_transfer_wire_frame(
            &TransferWireFrame {
                session_id: self.session_id,
                sequence,
                frame_kind,
                encrypted_payload,
            },
            Self::MAXIMUM_FRAME_BYTES,
        )
        .map_err(js_error)
    }

    pub fn open(&mut self, encoded: Vec<u8>) -> Result<Vec<u8>, JsValue> {
        let frame =
            decode_transfer_wire_frame(&encoded, Self::MAXIMUM_FRAME_BYTES).map_err(js_error)?;
        if frame.session_id != self.session_id || frame.frame_kind > 3 {
            return Err(JsValue::from_str(
                "execution frame does not match this session",
            ));
        }
        let opened = open_transfer_frame(
            &self.key,
            &self.authority(frame.frame_kind, frame.sequence),
            &frame.encrypted_payload,
            Self::MAXIMUM_PAYLOAD_BYTES,
        )
        .map_err(js_error)?;
        if self.used_salts.contains(&opened.derivation_salt) {
            return Err(JsValue::from_str(
                "execution frame derivation salt was reused",
            ));
        }
        self.accept_sequence(frame.sequence)?;
        self.used_salts.insert(opened.derivation_salt);
        Ok(opened.plaintext)
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

/// Verify and materialize an exact execution bundle into the active browser
/// filesystem provider, returning the existing frozen-project handoff rebased
/// to those bytes. Browser worker hosts should use an ephemeral provider and
/// dispose it with the worker.
#[wasm_bindgen(js_name = materializeExecutionBundle)]
pub async fn materialize_execution_bundle(archive: Vec<u8>) -> Result<JsValue, JsValue> {
    let bundle = runmat_execution_artifact::archive::read_bundle(
        archive.as_slice(),
        runmat_execution_artifact::archive::ArchiveLimits::default(),
    )
    .map_err(js_error)?;
    let identity = bundle.identity().map_err(js_error)?;
    let root = std::path::PathBuf::from(format!(
        ".runmat/execution/{}",
        identity.to_string().replace(':', "_")
    ));
    for object in &bundle.objects {
        if object.descriptor.namespace != runmat_execution_artifact::ObjectNamespace::ProgramSource
        {
            continue;
        }
        let target = root.join(&object.descriptor.logical_name);
        let parent = target
            .parent()
            .ok_or_else(|| JsValue::from_str("bundle source has no materialization parent"))?;
        runmat_filesystem::create_dir_all_async(parent)
            .await
            .map_err(js_error)?;
        runmat_filesystem::write_async(&target, &object.bytes)
            .await
            .map_err(js_error)?;
        let written = runmat_filesystem::read_async(&target)
            .await
            .map_err(js_error)?;
        if runmat_execution::Digest::sha256(&written) != object.descriptor.digest {
            return Err(JsValue::from_str(
                "browser materialization differs from the bundle source digest",
            ));
        }
    }
    let handoff = bundle.project_handoff_at(&root).map_err(js_error)?;
    serde_wasm_bindgen::to_value(&handoff).map_err(js_error)
}

pub(super) fn browser_entropy() -> Result<[u8; 32], JsValue> {
    use wasm_bindgen::JsCast as _;
    let crypto = js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("crypto"))?
        .dyn_into::<web_sys::Crypto>()
        .map_err(|_| JsValue::from_str("Web Crypto is unavailable"))?;
    let mut entropy = [0_u8; 32];
    crypto.get_random_values_with_u8_array(&mut entropy)?;
    Ok(entropy)
}

fn js_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}
