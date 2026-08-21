use sha2::{Digest as _, Sha256};

use crate::{TransportError, TransportResult};

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct EndpointIdentityMaterial {
    recipient_secret: String,
    pub evidence: runmat_execution::security::EndpointIdentityEvidence,
}

impl EndpointIdentityMaterial {
    pub fn new(
        recipient_entropy: [u8; 32],
        evidence: runmat_execution::security::EndpointIdentityEvidence,
    ) -> Self {
        use base64::Engine as _;
        Self {
            recipient_secret: base64::engine::general_purpose::URL_SAFE_NO_PAD
                .encode(recipient_entropy),
            evidence,
        }
    }

    pub fn recipient_private_key(
        &self,
    ) -> TransportResult<(
        runmat_execution_artifact::encryption::ExecutionRecipientKey,
        runmat_execution_artifact::encryption::PortableExecutionPrivateKey,
    )> {
        use base64::Engine as _;
        let entropy: [u8; 32] = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(&self.recipient_secret)
            .map_err(|_| TransportError::Integrity)?
            .try_into()
            .map_err(|_| TransportError::Integrity)?;
        let (recipient, private) =
            runmat_execution_artifact::encryption::PortableExecutionEncryption
                .recipient_from_entropy_with_derived_fingerprint(
                    entropy,
                    self.evidence.recipient.valid_after_unix_millis,
                    self.evidence.recipient.valid_before_unix_millis,
                )
                .map_err(|error| TransportError::Encryption(error.to_string()))?;
        if recipient.fingerprint != self.evidence.recipient.fingerprint
            || recipient.public_key != self.evidence.recipient.public_key
        {
            return Err(TransportError::Integrity);
        }
        Ok((recipient, private))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EndpointIdentity {
    pub key_fingerprint: String,
    pub credential_epoch: u64,
}

impl EndpointIdentity {
    pub fn new(public_identity: &[u8], credential_epoch: u64) -> TransportResult<Self> {
        if public_identity.len() < 32 || credential_epoch == 0 {
            return Err(TransportError::StaleAuthority);
        }
        Ok(Self {
            key_fingerprint: format!("{:x}", Sha256::digest(public_identity)),
            credential_epoch,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeaseAuthority {
    pub lease_id: String,
    pub fencing_token: u64,
    pub expires_at_millis: i64,
}

impl LeaseAuthority {
    pub fn validate(&self, expected_fence: u64, now_millis: i64) -> TransportResult<()> {
        if self.fencing_token != expected_fence || self.expires_at_millis <= now_millis {
            return Err(TransportError::StaleAuthority);
        }
        Ok(())
    }
}
