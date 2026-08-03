use std::path::Path;

use base64::Engine as _;
use rand::RngCore as _;
use serde::{Deserialize, Serialize};

use runmat_execution::security::{
    EndpointIdentityEvidence, EndpointIdentitySigner, EndpointRecipientKey, ExecutionTrustTier,
};
use runmat_execution_artifact::encryption::PortableExecutionEncryption;
use runmat_execution_transport_native::control::NodeAllocation;

use super::Sandbox;
use crate::enrollment::{decode_secret, NodeCredential};
use crate::{AgentError, AgentResult};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct StoredEndpointIdentity {
    recipient_secret: String,
    evidence: EndpointIdentityEvidence,
}

pub fn prepare_endpoint_identity(
    credential: &NodeCredential,
    allocation: &NodeAllocation,
    sandbox: &Sandbox,
    trust_tier: ExecutionTrustTier,
    now_unix_millis: u64,
) -> AgentResult<EndpointIdentityEvidence> {
    let path = sandbox.root.join("endpoint-identity.json");
    if path.exists() {
        let stored: StoredEndpointIdentity = serde_json::from_slice(&std::fs::read(&path)?)?;
        validate_stored(credential, allocation, &stored, now_unix_millis)?;
        return Ok(stored.evidence);
    }

    let expires_at = u64::try_from(allocation.expires_at_millis).map_err(|_| {
        AgentError::AllocationRejected("allocation expiry is outside the wire range".into())
    })?;
    if expires_at <= now_unix_millis {
        return Err(AgentError::AllocationRejected(
            "allocation expired before endpoint identity creation".into(),
        ));
    }
    let mut recipient_entropy = [0_u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut recipient_entropy);
    let (recipient, _) = PortableExecutionEncryption
        .recipient_from_entropy_with_derived_fingerprint(
            recipient_entropy,
            now_unix_millis.saturating_sub(30_000),
            expires_at,
        )
        .map_err(|error| AgentError::AllocationRejected(error.to_string()))?;
    let signer = EndpointIdentitySigner::from_secret(decode_secret(&credential.identity_secret)?)
        .map_err(|error| AgentError::UnsafeCredential(error.to_string()))?;
    let mut evidence = EndpointIdentityEvidence {
        schema_version: 1,
        org_id: credential.org_id.clone(),
        cluster_id: credential.cluster_id.clone(),
        node_id: credential.node_id.clone(),
        allocation_lease_id: allocation.id.clone(),
        fencing_token: allocation.fencing_token,
        run_identity: allocation.run_id.clone(),
        identity_public_key: credential.identity_public_key.clone(),
        identity_fingerprint: credential.identity_fingerprint.clone(),
        recipient: EndpointRecipientKey {
            suite: "x25519-hkdf-sha256-aes128gcm-v1".into(),
            public_key: recipient.public_key,
            fingerprint: recipient.fingerprint,
            valid_after_unix_millis: recipient.valid_after_unix_millis,
            valid_before_unix_millis: recipient.valid_before_unix_millis,
        },
        direct_quic_endpoints: Vec::new(),
        trust_tier,
        attestation_class: None,
        attestation_evidence: None,
        issued_at_unix_millis: now_unix_millis,
        expires_at_unix_millis: expires_at,
        signature: vec![0; 64],
    };
    signer
        .sign(&mut evidence)
        .map_err(|error| AgentError::AllocationRejected(error.to_string()))?;
    let stored = StoredEndpointIdentity {
        recipient_secret: base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(recipient_entropy),
        evidence: evidence.clone(),
    };
    write_private(&path, &serde_json::to_vec(&stored)?)?;
    Ok(evidence)
}

fn validate_stored(
    credential: &NodeCredential,
    allocation: &NodeAllocation,
    stored: &StoredEndpointIdentity,
    now_unix_millis: u64,
) -> AgentResult<()> {
    let secret = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(&stored.recipient_secret)
        .map_err(|_| AgentError::AllocationRejected("stored recipient key is malformed".into()))?;
    if secret.len() != 32
        || stored.evidence.node_id != credential.node_id
        || stored.evidence.allocation_lease_id != allocation.id
        || stored.evidence.run_identity != allocation.run_id
        || stored.evidence.fencing_token != allocation.fencing_token
        || stored.evidence.expires_at_unix_millis <= now_unix_millis
    {
        return Err(AgentError::AllocationRejected(
            "stored endpoint identity is stale or outside this allocation".into(),
        ));
    }
    let mut policy = runmat_execution::security::EndpointTrustPolicy {
        permitted_tiers: [stored.evidence.trust_tier].into_iter().collect(),
        trusted_identity_fingerprints: [credential.identity_fingerprint.clone()]
            .into_iter()
            .collect(),
        allowed_attestation_classes: Default::default(),
        require_pinned_identity: true,
        now_unix_millis,
        maximum_clock_skew_millis: 30_000,
    };
    if let Some(class) = stored.evidence.attestation_class.clone() {
        policy.allowed_attestation_classes.insert(class);
    }
    policy
        .verify(&stored.evidence)
        .map_err(|error| AgentError::AllocationRejected(error.to_string()))
}

fn write_private(path: &Path, bytes: &[u8]) -> AgentResult<()> {
    let mut options = std::fs::OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    use std::io::Write as _;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}
