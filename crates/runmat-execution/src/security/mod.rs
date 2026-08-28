//! Portable, content-free endpoint identity evidence.
//!
//! This module defines canonical signed bytes and bounded wire models. Native
//! and browser hosts supply the platform crypto and local trust policy.

use std::collections::BTreeSet;

use ed25519_dalek::{Signature, Signer as _, SigningKey, Verifier as _, VerifyingKey};
use minicbor::Encoder;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use crate::ContractError;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[repr(u8)]
pub enum ExecutionTrustTier {
    CustomerTrusted,
    HostedOrdinary,
    AttestedConfidential,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EndpointRecipientKey {
    pub suite: String,
    pub public_key: Vec<u8>,
    pub fingerprint: String,
    pub valid_after_unix_millis: u64,
    pub valid_before_unix_millis: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DirectQuicEndpoint {
    pub authority: String,
    pub server_name: String,
    pub certificate_der: Vec<u8>,
    pub certificate_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EndpointIdentityEvidence {
    pub schema_version: u16,
    pub org_id: String,
    pub cluster_id: String,
    pub node_id: String,
    pub allocation_lease_id: String,
    pub fencing_token: u64,
    pub run_identity: String,
    pub identity_public_key: Vec<u8>,
    pub identity_fingerprint: String,
    pub recipient: EndpointRecipientKey,
    pub direct_quic_endpoints: Vec<DirectQuicEndpoint>,
    pub trust_tier: ExecutionTrustTier,
    pub attestation_class: Option<String>,
    pub attestation_evidence: Option<Vec<u8>>,
    pub issued_at_unix_millis: u64,
    pub expires_at_unix_millis: u64,
    pub signature: Vec<u8>,
}

#[derive(Clone)]
pub struct EndpointIdentitySigner(SigningKey);

impl EndpointIdentitySigner {
    pub fn from_secret(secret: [u8; 32]) -> Result<Self, ContractError> {
        if secret.iter().all(|byte| *byte == 0) {
            return Err(invalid("identity signing secret is all zero"));
        }
        Ok(Self(SigningKey::from_bytes(&secret)))
    }

    pub fn public_key(&self) -> [u8; 32] {
        self.0.verifying_key().to_bytes()
    }

    pub fn fingerprint(&self) -> String {
        identity_fingerprint(&self.public_key())
    }

    pub fn sign(&self, evidence: &mut EndpointIdentityEvidence) -> Result<(), ContractError> {
        if evidence.identity_public_key != self.public_key()
            || evidence.identity_fingerprint != self.fingerprint()
        {
            return Err(invalid(
                "signer does not own the declared endpoint identity",
            ));
        }
        evidence.signature = vec![0; 64];
        evidence.validate_shape()?;
        evidence.signature = self.0.sign(&evidence.signing_bytes()?).to_bytes().to_vec();
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EndpointTrustPolicy {
    pub permitted_tiers: BTreeSet<ExecutionTrustTier>,
    pub trusted_identity_fingerprints: BTreeSet<String>,
    pub allowed_attestation_classes: BTreeSet<String>,
    pub require_pinned_identity: bool,
    pub now_unix_millis: u64,
    pub maximum_clock_skew_millis: u64,
}

impl EndpointTrustPolicy {
    pub fn verify(&self, evidence: &EndpointIdentityEvidence) -> Result<(), ContractError> {
        evidence.validate_shape()?;
        if !self.permitted_tiers.contains(&evidence.trust_tier)
            || (self.require_pinned_identity
                && !self
                    .trusted_identity_fingerprints
                    .contains(&evidence.identity_fingerprint))
            || evidence.issued_at_unix_millis
                > self
                    .now_unix_millis
                    .saturating_add(self.maximum_clock_skew_millis)
            || evidence.expires_at_unix_millis
                <= self
                    .now_unix_millis
                    .saturating_sub(self.maximum_clock_skew_millis)
        {
            return Err(invalid(
                "endpoint evidence is not authorized by the local trust policy",
            ));
        }
        if evidence.trust_tier == ExecutionTrustTier::AttestedConfidential
            && !evidence.attestation_class.as_ref().is_some_and(|class| {
                self.allowed_attestation_classes.is_empty()
                    || self.allowed_attestation_classes.contains(class)
            })
        {
            return Err(invalid("attestation class is not locally trusted"));
        }
        let public_key: [u8; 32] = evidence
            .identity_public_key
            .as_slice()
            .try_into()
            .map_err(|_| invalid("endpoint signing key has an invalid length"))?;
        let verifying_key =
            VerifyingKey::from_bytes(&public_key).map_err(|_| invalid("invalid signing key"))?;
        let signature = Signature::from_slice(&evidence.signature)
            .map_err(|_| invalid("invalid endpoint evidence signature"))?;
        verifying_key
            .verify(&evidence.signing_bytes()?, &signature)
            .map_err(|_| invalid("endpoint evidence signature verification failed"))
    }
}

impl EndpointIdentityEvidence {
    pub fn validate_shape(&self) -> Result<(), ContractError> {
        let identifiers = [
            self.org_id.as_str(),
            self.cluster_id.as_str(),
            self.node_id.as_str(),
            self.allocation_lease_id.as_str(),
            self.run_identity.as_str(),
        ];
        if self.schema_version != 1
            || identifiers.iter().any(|value| !bounded_ascii(value, 256))
            || self.fencing_token == 0
            || self.identity_public_key.len() != 32
            || self.identity_fingerprint
                != fingerprint(
                    "runmat-execution-endpoint-identity-v1",
                    &self.identity_public_key,
                )
            || self.recipient.suite != "x25519-hkdf-sha256-aes128gcm-v1"
            || self.recipient.public_key.len() != 32
            || self.recipient.fingerprint
                != fingerprint(
                    "runmat-execution-recipient-key-v1",
                    &self.recipient.public_key,
                )
            || self.recipient.valid_after_unix_millis >= self.recipient.valid_before_unix_millis
            || self.issued_at_unix_millis >= self.expires_at_unix_millis
            || self.recipient.valid_after_unix_millis > self.issued_at_unix_millis
            || self.recipient.valid_before_unix_millis < self.expires_at_unix_millis
            || self.direct_quic_endpoints.len() > 8
            || self
                .direct_quic_endpoints
                .iter()
                .any(|endpoint| !endpoint.validate())
            || self.signature.len() != 64
            || self
                .attestation_class
                .as_deref()
                .is_some_and(|value| !bounded_ascii(value, 128))
            || self
                .attestation_evidence
                .as_ref()
                .is_some_and(|value| value.len() > 64 * 1024)
            || (self.attestation_class.is_some() != self.attestation_evidence.is_some())
            || (self.trust_tier == ExecutionTrustTier::AttestedConfidential
                && self.attestation_class.is_none())
        {
            return Err(ContractError::Invalid {
                field: "endpoint_identity_evidence",
                reason: "evidence is malformed".into(),
            });
        }
        Ok(())
    }

    pub fn signing_bytes(&self) -> Result<Vec<u8>, ContractError> {
        let mut bytes = b"runmat-execution-endpoint-evidence-v1\0".to_vec();
        let mut encoder = Encoder::new(&mut bytes);
        encoder
            .array(20)
            .and_then(|encoder| encoder.u16(self.schema_version))
            .and_then(|encoder| encoder.str(&self.org_id))
            .and_then(|encoder| encoder.str(&self.cluster_id))
            .and_then(|encoder| encoder.str(&self.node_id))
            .and_then(|encoder| encoder.str(&self.allocation_lease_id))
            .and_then(|encoder| encoder.u64(self.fencing_token))
            .and_then(|encoder| encoder.str(&self.run_identity))
            .and_then(|encoder| encoder.bytes(&self.identity_public_key))
            .and_then(|encoder| encoder.str(&self.identity_fingerprint))
            .and_then(|encoder| encoder.str(&self.recipient.suite))
            .and_then(|encoder| encoder.bytes(&self.recipient.public_key))
            .and_then(|encoder| encoder.str(&self.recipient.fingerprint))
            .and_then(|encoder| encoder.u64(self.recipient.valid_after_unix_millis))
            .and_then(|encoder| encoder.u64(self.recipient.valid_before_unix_millis))
            .map_err(encoding)?;
        encoder
            .array(self.direct_quic_endpoints.len() as u64)
            .map_err(encoding)?;
        for endpoint in &self.direct_quic_endpoints {
            encoder
                .array(4)
                .and_then(|encoder| encoder.str(&endpoint.authority))
                .and_then(|encoder| encoder.str(&endpoint.server_name))
                .and_then(|encoder| encoder.bytes(&endpoint.certificate_der))
                .and_then(|encoder| encoder.str(&endpoint.certificate_sha256))
                .map_err(encoding)?;
        }
        encoder.u8(self.trust_tier as u8).map_err(encoding)?;
        encode_optional_text(&mut encoder, self.attestation_class.as_deref())?;
        encode_optional_bytes(&mut encoder, self.attestation_evidence.as_deref())?;
        encoder
            .u64(self.issued_at_unix_millis)
            .and_then(|encoder| encoder.u64(self.expires_at_unix_millis))
            .map_err(encoding)?;
        Ok(bytes)
    }
}

impl DirectQuicEndpoint {
    fn validate(&self) -> bool {
        bounded_ascii(&self.authority, 512)
            && bounded_ascii(&self.server_name, 253)
            && !self.certificate_der.is_empty()
            && self.certificate_der.len() <= 16 * 1024
            && self.certificate_sha256 == format!("{:x}", Sha256::digest(&self.certificate_der))
    }
}

pub fn identity_fingerprint(public_key: &[u8]) -> String {
    fingerprint("runmat-execution-endpoint-identity-v1", public_key)
}

pub fn recipient_fingerprint(public_key: &[u8]) -> String {
    fingerprint("runmat-execution-recipient-key-v1", public_key)
}

fn fingerprint(domain: &str, public_key: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(domain.as_bytes());
    digest.update([0]);
    digest.update(public_key);
    format!("{:x}", digest.finalize())
}

fn bounded_ascii(value: &str, maximum: usize) -> bool {
    !value.is_empty()
        && value.len() <= maximum
        && value.is_ascii()
        && !value.chars().any(char::is_control)
}

fn encode_optional_text(
    encoder: &mut Encoder<&mut Vec<u8>>,
    value: Option<&str>,
) -> Result<(), ContractError> {
    match value {
        Some(value) => encoder.str(value),
        None => encoder.null(),
    }
    .map(|_| ())
    .map_err(encoding)
}

fn encode_optional_bytes(
    encoder: &mut Encoder<&mut Vec<u8>>,
    value: Option<&[u8]>,
) -> Result<(), ContractError> {
    match value {
        Some(value) => encoder.bytes(value),
        None => encoder.null(),
    }
    .map(|_| ())
    .map_err(encoding)
}

fn encoding(error: minicbor::encode::Error<std::convert::Infallible>) -> ContractError {
    ContractError::MalformedProtocol(error.to_string())
}

fn invalid(reason: impl Into<String>) -> ContractError {
    ContractError::Invalid {
        field: "endpoint_identity_evidence",
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence() -> EndpointIdentityEvidence {
        let signer = EndpointIdentitySigner::from_secret([7; 32]).unwrap();
        let recipient_public = [9; 32];
        let mut evidence = EndpointIdentityEvidence {
            schema_version: 1,
            org_id: "org_a".into(),
            cluster_id: "cluster_a".into(),
            node_id: "node_a".into(),
            allocation_lease_id: "lease_a".into(),
            fencing_token: 3,
            run_identity: "run_a".into(),
            identity_public_key: signer.public_key().to_vec(),
            identity_fingerprint: signer.fingerprint(),
            recipient: EndpointRecipientKey {
                suite: "x25519-hkdf-sha256-aes128gcm-v1".into(),
                public_key: recipient_public.to_vec(),
                fingerprint: recipient_fingerprint(&recipient_public),
                valid_after_unix_millis: 900,
                valid_before_unix_millis: 2_000,
            },
            direct_quic_endpoints: vec![DirectQuicEndpoint {
                authority: "127.0.0.1:4433".into(),
                server_name: "runmat.execution".into(),
                certificate_der: vec![1, 2, 3],
                certificate_sha256: format!("{:x}", Sha256::digest([1, 2, 3])),
            }],
            trust_tier: ExecutionTrustTier::CustomerTrusted,
            attestation_class: None,
            attestation_evidence: None,
            issued_at_unix_millis: 1_000,
            expires_at_unix_millis: 1_500,
            signature: vec![0; 64],
        };
        signer.sign(&mut evidence).unwrap();
        evidence
    }

    fn policy(evidence: &EndpointIdentityEvidence) -> EndpointTrustPolicy {
        EndpointTrustPolicy {
            permitted_tiers: [ExecutionTrustTier::CustomerTrusted].into_iter().collect(),
            trusted_identity_fingerprints: [evidence.identity_fingerprint.clone()]
                .into_iter()
                .collect(),
            allowed_attestation_classes: BTreeSet::new(),
            require_pinned_identity: true,
            now_unix_millis: 1_100,
            maximum_clock_skew_millis: 10,
        }
    }

    #[test]
    fn verifies_pinned_evidence_and_rejects_substitution_replay_and_downgrade() {
        let evidence = evidence();
        policy(&evidence).verify(&evidence).unwrap();

        let mut substitution = evidence.clone();
        substitution.run_identity = "run_b".into();
        assert!(policy(&evidence).verify(&substitution).is_err());

        let mut replay = policy(&evidence);
        replay.now_unix_millis = 2_000;
        assert!(replay.verify(&evidence).is_err());

        let mut downgrade = evidence.clone();
        downgrade.trust_tier = ExecutionTrustTier::HostedOrdinary;
        assert!(policy(&evidence).verify(&downgrade).is_err());

        let mut transport_substitution = evidence.clone();
        transport_substitution.direct_quic_endpoints[0].authority = "attacker:4433".into();
        assert!(policy(&evidence).verify(&transport_substitution).is_err());

        let mut certificate_substitution = evidence.clone();
        certificate_substitution.direct_quic_endpoints[0].certificate_der[0] ^= 1;
        assert!(policy(&evidence).verify(&certificate_substitution).is_err());
    }
}
