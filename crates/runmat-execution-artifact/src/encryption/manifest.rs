use serde::{Deserialize, Serialize};

use super::{EncryptionContext, ExecutionHpkeSuite};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionRecipientKey {
    pub suite: ExecutionHpkeSuite,
    pub public_key: Vec<u8>,
    pub fingerprint: String,
    pub valid_after_unix_millis: u64,
    pub valid_before_unix_millis: u64,
    pub custodian_uri: Option<String>,
}

impl ExecutionRecipientKey {
    pub fn validate(&self) -> Result<(), crate::ArtifactError> {
        if self.suite != ExecutionHpkeSuite::X25519HkdfSha256Aes128GcmV1
            || self.public_key.len() != 32
            || self.fingerprint.is_empty()
            || self.fingerprint.len() > 256
            || !self.fingerprint.is_ascii()
            || self.fingerprint.chars().any(char::is_control)
            || self.valid_after_unix_millis >= self.valid_before_unix_millis
            || self.custodian_uri.as_deref().is_some_and(|uri| {
                uri.is_empty()
                    || uri.len() > 2048
                    || !uri.is_ascii()
                    || uri.chars().any(char::is_control)
            })
        {
            return Err(crate::ArtifactError::Invalid(
                "execution recipient key is malformed".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EncryptedArtifact {
    pub schema_version: u16,
    pub suite: ExecutionHpkeSuite,
    pub context: EncryptionContext,
    pub encapsulated_key: Vec<u8>,
    pub ciphertext: Vec<u8>,
}
