use std::collections::BTreeSet;

use base64::Engine as _;
use runmat_execution::security::{
    DirectQuicEndpoint, EndpointIdentityEvidence, EndpointRecipientKey, EndpointTrustPolicy,
    ExecutionTrustTier,
};
use runmat_execution_artifact::encryption::ExecutionRecipientKey;
use runmat_execution_artifact::encryption::{
    encode_encrypted_run_object, encode_run_key_envelope, EncryptionContext, EncryptionPurpose,
    PortableExecutionEncryption, RunKeyMaterial, RunObjectEncryption,
};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct BrowserTrustPolicy {
    permitted_tiers: BTreeSet<ExecutionTrustTier>,
    #[serde(default)]
    trusted_identity_fingerprints: BTreeSet<String>,
    #[serde(default)]
    allowed_attestation_classes: BTreeSet<String>,
    require_pinned_identity: bool,
    now_unix_millis: u64,
    maximum_clock_skew_millis: u64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct WireRecipient {
    suite: String,
    public_key: String,
    fingerprint: String,
    valid_after_unix_millis: u64,
    valid_before_unix_millis: u64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct WireDirectQuicEndpoint {
    authority: String,
    server_name: String,
    certificate_der: String,
    certificate_sha256: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct WireEvidence {
    schema_version: u16,
    org_id: String,
    cluster_id: String,
    node_id: String,
    allocation_lease_id: String,
    fencing_token: u64,
    run_identity: String,
    identity_public_key: String,
    identity_fingerprint: String,
    recipient: WireRecipient,
    #[serde(default)]
    direct_quic_endpoints: Vec<WireDirectQuicEndpoint>,
    trust_tier: ExecutionTrustTier,
    attestation_class: Option<String>,
    attestation_evidence: Option<String>,
    issued_at_unix_millis: u64,
    expires_at_unix_millis: u64,
    signature: String,
}

/// Verify signed Server admission evidence under browser-local trust policy and
/// return the exact HPKE recipient accepted for the run.
#[wasm_bindgen(js_name = verifyExecutionEndpointIdentity)]
pub fn verify_execution_endpoint_identity(
    evidence: JsValue,
    policy: JsValue,
) -> Result<JsValue, JsValue> {
    let evidence: WireEvidence = serde_wasm_bindgen::from_value(evidence).map_err(js_error)?;
    let policy: BrowserTrustPolicy = serde_wasm_bindgen::from_value(policy).map_err(js_error)?;
    let evidence = evidence.into_domain()?;
    let policy = EndpointTrustPolicy {
        permitted_tiers: policy.permitted_tiers,
        trusted_identity_fingerprints: policy.trusted_identity_fingerprints,
        allowed_attestation_classes: policy.allowed_attestation_classes,
        require_pinned_identity: policy.require_pinned_identity,
        now_unix_millis: policy.now_unix_millis,
        maximum_clock_skew_millis: policy.maximum_clock_skew_millis,
    };
    let recipient =
        ExecutionRecipientKey::from_verified_endpoint(&evidence, &policy).map_err(js_error)?;
    serde_wasm_bindgen::to_value(&VerifiedBrowserEndpoint {
        recipient,
        direct_quic_endpoints: evidence.direct_quic_endpoints,
    })
    .map_err(js_error)
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct VerifiedBrowserEndpoint {
    recipient: ExecutionRecipientKey,
    direct_quic_endpoints: Vec<DirectQuicEndpoint>,
}

/// Build the exact browser WebSocket subprotocol list for an authenticated
/// relay capability returned by the admission endpoint.
#[wasm_bindgen(js_name = executionRelayProtocols)]
pub fn execution_relay_protocols(ticket: String) -> Result<JsValue, JsValue> {
    if ticket.is_empty()
        || ticket.len() > 256
        || !ticket
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(JsValue::from_str("execution relay ticket is malformed"));
    }
    serde_wasm_bindgen::to_value(&vec![
        "runmat-relay-v1".to_string(),
        format!("runmat-ticket.{ticket}"),
    ])
    .map_err(js_error)
}

/// Browser-side authority for the content-blind two-stage submission.
///
/// Network requests remain a small TypeScript host responsibility. Rust owns
/// the run key, exact endpoint recipient, encryption contexts, and canonical
/// ciphertext bytes so browser and native submissions cannot drift.
#[wasm_bindgen]
pub struct BrowserRemoteSubmission {
    run_identity: String,
    endpoint: VerifiedBrowserEndpoint,
    run_key: RunKeyMaterial,
}

#[wasm_bindgen]
impl BrowserRemoteSubmission {
    #[wasm_bindgen(constructor)]
    pub fn new(
        run_identity: String,
        verified_endpoint: JsValue,
    ) -> Result<BrowserRemoteSubmission, JsValue> {
        if run_identity.is_empty() || run_identity.len() > 256 || !run_identity.is_ascii() {
            return Err(JsValue::from_str("remote run identity is malformed"));
        }
        let endpoint: VerifiedBrowserEndpoint =
            serde_wasm_bindgen::from_value(verified_endpoint).map_err(js_error)?;
        Ok(Self {
            run_identity,
            endpoint,
            run_key: RunKeyMaterial::from_entropy(super::execution_artifact::browser_entropy()?)
                .map_err(js_error)?,
        })
    }

    #[wasm_bindgen(getter, js_name = endpointFingerprint)]
    pub fn endpoint_fingerprint(&self) -> String {
        self.endpoint.recipient.fingerprint.clone()
    }

    #[wasm_bindgen(getter, js_name = directQuicEndpoints)]
    pub fn direct_quic_endpoints(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.endpoint.direct_quic_endpoints).map_err(js_error)
    }

    #[wasm_bindgen(js_name = sealRunKeyEnvelope)]
    pub fn seal_run_key_envelope(&self, key_epoch: u32) -> Result<Vec<u8>, JsValue> {
        let envelope = PortableExecutionEncryption
            .seal_run_key_with_entropy(
                super::execution_artifact::browser_entropy()?,
                &self.endpoint.recipient,
                &self.run_key,
                self.run_identity.clone(),
                key_epoch,
            )
            .map_err(js_error)?;
        encode_run_key_envelope(&envelope).map_err(js_error)
    }

    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = encryptObject)]
    pub fn encrypt_object(
        &self,
        purpose: String,
        plaintext: Vec<u8>,
        task_identity: Option<String>,
        attempt_identity: Option<String>,
        chunk_index: u64,
        key_epoch: u32,
    ) -> Result<Vec<u8>, JsValue> {
        let context = EncryptionContext {
            schema_version: 1,
            run_identity: self.run_identity.clone(),
            purpose: parse_purpose(&purpose)?,
            object_digest: runmat_execution::Digest::sha256(&plaintext),
            task_identity,
            attempt_identity,
            chunk_index,
            total_length: plaintext.len() as u64,
            key_epoch,
        };
        let object = RunObjectEncryption
            .seal_with_entropy(
                &self.run_key,
                super::execution_artifact::browser_entropy()?,
                context,
                &plaintext,
            )
            .map_err(js_error)?;
        encode_encrypted_run_object(&object).map_err(js_error)
    }

    #[wasm_bindgen(js_name = decryptObject)]
    pub fn decrypt_object(&self, purpose: String, ciphertext: Vec<u8>) -> Result<Vec<u8>, JsValue> {
        let object = runmat_execution_artifact::encryption::decode_encrypted_run_object(
            &ciphertext,
            64 * 1024 * 1024,
        )
        .map_err(js_error)?;
        if object.context.run_identity != self.run_identity
            || object.context.purpose != parse_purpose(&purpose)?
        {
            return Err(JsValue::from_str(
                "remote artifact scope or encryption purpose is invalid",
            ));
        }
        RunObjectEncryption
            .open(&self.run_key, &object)
            .map_err(js_error)
    }

    #[wasm_bindgen(js_name = createFrameSession)]
    pub fn create_frame_session(
        &self,
        session_id: Vec<u8>,
        direction: String,
        key_epoch: u32,
    ) -> Result<super::execution_artifact::BrowserEncryptedFrameSession, JsValue> {
        super::execution_artifact::BrowserRunKey {
            key: self.run_key.clone(),
        }
        .create_frame_session(self.run_identity.clone(), session_id, direction, key_epoch)
    }
}

fn parse_purpose(value: &str) -> Result<EncryptionPurpose, JsValue> {
    match value {
        "bundle" => Ok(EncryptionPurpose::Bundle),
        "program" => Ok(EncryptionPurpose::Program),
        "input" => Ok(EncryptionPurpose::Input),
        "result" => Ok(EncryptionPurpose::Result),
        "detailed-event" => Ok(EncryptionPurpose::DetailedEvent),
        "log" => Ok(EncryptionPurpose::Log),
        "checkpoint" => Ok(EncryptionPurpose::Checkpoint),
        _ => Err(JsValue::from_str(
            "remote object encryption purpose is unsupported",
        )),
    }
}

impl WireEvidence {
    fn into_domain(self) -> Result<EndpointIdentityEvidence, JsValue> {
        Ok(EndpointIdentityEvidence {
            schema_version: self.schema_version,
            org_id: self.org_id,
            cluster_id: self.cluster_id,
            node_id: self.node_id,
            allocation_lease_id: self.allocation_lease_id,
            fencing_token: self.fencing_token,
            run_identity: self.run_identity,
            identity_public_key: decode(&self.identity_public_key)?,
            identity_fingerprint: self.identity_fingerprint,
            recipient: EndpointRecipientKey {
                suite: self.recipient.suite,
                public_key: decode(&self.recipient.public_key)?,
                fingerprint: self.recipient.fingerprint,
                valid_after_unix_millis: self.recipient.valid_after_unix_millis,
                valid_before_unix_millis: self.recipient.valid_before_unix_millis,
            },
            direct_quic_endpoints: self
                .direct_quic_endpoints
                .into_iter()
                .map(|endpoint| {
                    Ok(DirectQuicEndpoint {
                        authority: endpoint.authority,
                        server_name: endpoint.server_name,
                        certificate_der: decode(&endpoint.certificate_der)?,
                        certificate_sha256: endpoint.certificate_sha256,
                    })
                })
                .collect::<Result<Vec<_>, JsValue>>()?,
            trust_tier: self.trust_tier,
            attestation_class: self.attestation_class,
            attestation_evidence: self
                .attestation_evidence
                .as_deref()
                .map(decode)
                .transpose()?,
            issued_at_unix_millis: self.issued_at_unix_millis,
            expires_at_unix_millis: self.expires_at_unix_millis,
            signature: decode(&self.signature)?,
        })
    }
}

fn decode(value: &str) -> Result<Vec<u8>, JsValue> {
    base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(value)
        .map_err(|_| JsValue::from_str("execution identity contains invalid base64url"))
}

fn js_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

#[cfg(all(test, target_arch = "wasm32"))]
mod tests {
    use super::*;
    use runmat_execution_artifact::encryption::{
        decode_run_key_envelope, PortableExecutionEncryption,
    };
    use wasm_bindgen_test::wasm_bindgen_test;

    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn browser_submission_uses_canonical_envelopes_and_relay_protocols() {
        let (recipient, _) = PortableExecutionEncryption
            .recipient_from_entropy([7; 32], "f".repeat(64), 1, 2_000_000_000_000)
            .unwrap();
        let endpoint = serde_wasm_bindgen::to_value(&VerifiedBrowserEndpoint {
            recipient,
            direct_quic_endpoints: Vec::new(),
        })
        .unwrap();
        let submission = BrowserRemoteSubmission::new("run-browser".into(), endpoint).unwrap();
        let envelope = submission.seal_run_key_envelope(1).unwrap();
        assert_eq!(
            decode_run_key_envelope(&envelope, 64 * 1024)
                .unwrap()
                .encrypted_key
                .context
                .run_identity,
            "run-browser"
        );
        let protocols: Vec<String> =
            serde_wasm_bindgen::from_value(execution_relay_protocols("ticket_1".into()).unwrap())
                .unwrap();
        assert_eq!(protocols, vec!["runmat-relay-v1", "runmat-ticket.ticket_1"]);
    }
}
