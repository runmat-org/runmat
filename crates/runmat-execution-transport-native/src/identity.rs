use sha2::{Digest as _, Sha256};

use crate::{TransportError, TransportResult};

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
