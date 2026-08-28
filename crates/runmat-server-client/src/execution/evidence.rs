use anyhow::{Context, Result};
use base64::Engine as _;
use runmat_execution::security::{
    DirectQuicEndpoint, EndpointIdentityEvidence, EndpointRecipientKey, ExecutionTrustTier,
};
use serde::{Deserialize, Serialize};

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
    direct_quic_endpoints: Vec<WireDirectEndpoint>,
    trust_tier: ExecutionTrustTier,
    attestation_class: Option<String>,
    attestation_evidence: Option<String>,
    issued_at_unix_millis: u64,
    expires_at_unix_millis: u64,
    signature: String,
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
struct WireDirectEndpoint {
    authority: String,
    server_name: String,
    certificate_der: String,
    certificate_sha256: String,
}

pub fn endpoint_evidence(value: &impl Serialize) -> Result<EndpointIdentityEvidence> {
    let wire: WireEvidence = serde_json::from_value(serde_json::to_value(value)?)
        .context("invalid endpoint evidence")?;
    Ok(EndpointIdentityEvidence {
        schema_version: wire.schema_version,
        org_id: wire.org_id,
        cluster_id: wire.cluster_id,
        node_id: wire.node_id,
        allocation_lease_id: wire.allocation_lease_id,
        fencing_token: wire.fencing_token,
        run_identity: wire.run_identity,
        identity_public_key: decode(&wire.identity_public_key)?,
        identity_fingerprint: wire.identity_fingerprint,
        recipient: EndpointRecipientKey {
            suite: wire.recipient.suite,
            public_key: decode(&wire.recipient.public_key)?,
            fingerprint: wire.recipient.fingerprint,
            valid_after_unix_millis: wire.recipient.valid_after_unix_millis,
            valid_before_unix_millis: wire.recipient.valid_before_unix_millis,
        },
        direct_quic_endpoints: wire
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
            .collect::<Result<_>>()?,
        trust_tier: wire.trust_tier,
        attestation_class: wire.attestation_class,
        attestation_evidence: wire
            .attestation_evidence
            .as_deref()
            .map(decode)
            .transpose()?,
        issued_at_unix_millis: wire.issued_at_unix_millis,
        expires_at_unix_millis: wire.expires_at_unix_millis,
        signature: decode(&wire.signature)?,
    })
}

fn decode(value: &str) -> Result<Vec<u8>> {
    base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(value)
        .context("endpoint evidence contains invalid base64url")
}
