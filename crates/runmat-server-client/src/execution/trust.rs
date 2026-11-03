use std::collections::BTreeSet;

use anyhow::{bail, Result};
use runmat_execution::security::{EndpointTrustPolicy, ExecutionTrustTier};

/// How a submitting client authorizes the identity returned by its authenticated Server.
/// Encryption and signed, run-scoped endpoint evidence remain mandatory in both modes.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum RemoteEndpointTrust {
    #[default]
    AuthenticatedServer,
    PinnedIdentity(String),
}

impl RemoteEndpointTrust {
    pub fn from_optional_pin(pin: Option<&str>) -> Result<Self> {
        match pin {
            None => Ok(Self::AuthenticatedServer),
            Some(value) => {
                validate_pin(value)?;
                Ok(Self::PinnedIdentity(value.to_owned()))
            }
        }
    }

    pub fn policy_at(&self, now_unix_millis: u64) -> Result<EndpointTrustPolicy> {
        let (trusted_identity_fingerprints, require_pinned_identity) = match self {
            Self::AuthenticatedServer => (BTreeSet::new(), false),
            Self::PinnedIdentity(identity) => {
                validate_pin(identity)?;
                ([identity.clone()].into_iter().collect(), true)
            }
        };
        Ok(EndpointTrustPolicy {
            permitted_tiers: [
                ExecutionTrustTier::CustomerTrusted,
                ExecutionTrustTier::HostedOrdinary,
                ExecutionTrustTier::AttestedConfidential,
            ]
            .into_iter()
            .collect(),
            trusted_identity_fingerprints,
            allowed_attestation_classes: BTreeSet::new(),
            require_pinned_identity,
            now_unix_millis,
            maximum_clock_skew_millis: 30_000,
        })
    }
}

impl From<String> for RemoteEndpointTrust {
    fn from(identity: String) -> Self {
        Self::PinnedIdentity(identity)
    }
}

impl From<&str> for RemoteEndpointTrust {
    fn from(identity: &str) -> Self {
        Self::PinnedIdentity(identity.to_owned())
    }
}

fn validate_pin(value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 256
        || value.chars().any(char::is_whitespace)
        || value.chars().any(char::is_control)
    {
        bail!("endpoint identity pin is empty, malformed, or exceeds its bound");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::security::{
        recipient_fingerprint, EndpointIdentityEvidence, EndpointIdentitySigner,
        EndpointRecipientKey,
    };

    #[test]
    fn server_trust_and_independent_pins_both_require_authentic_fresh_evidence() {
        let signer = EndpointIdentitySigner::from_secret([7; 32]).unwrap();
        let public = [9; 32];
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
                public_key: public.to_vec(),
                fingerprint: recipient_fingerprint(&public),
                valid_after_unix_millis: 900,
                valid_before_unix_millis: 2_000,
            },
            direct_quic_endpoints: Vec::new(),
            trust_tier: ExecutionTrustTier::CustomerTrusted,
            attestation_class: None,
            attestation_evidence: None,
            issued_at_unix_millis: 1_000,
            expires_at_unix_millis: 1_500,
            signature: vec![0; 64],
        };
        signer.sign(&mut evidence).unwrap();
        for mode in [
            RemoteEndpointTrust::AuthenticatedServer,
            signer.fingerprint().into(),
        ] {
            let policy = mode.policy_at(1_000).unwrap();
            policy.verify(&evidence).unwrap();
            let mut changed = evidence.clone();
            changed.run_identity = "run_other".into();
            assert!(policy.verify(&changed).is_err());
            assert!(mode.policy_at(40_000).unwrap().verify(&evidence).is_err());
        }
        assert!(RemoteEndpointTrust::from("different-pin")
            .policy_at(1_000)
            .unwrap()
            .verify(&evidence)
            .is_err());
        assert_eq!(
            RemoteEndpointTrust::from_optional_pin(None).unwrap(),
            RemoteEndpointTrust::AuthenticatedServer
        );
        for pin in ["", " ", "pin\nother"] {
            assert!(RemoteEndpointTrust::from_optional_pin(Some(pin)).is_err());
        }
    }
}
