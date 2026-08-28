use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use serde::{Deserialize, Serialize};

use crate::{ContentDigest, RegistrySourceId};

pub const ENCRYPTED_ARTIFACT_SCHEMA_VERSION: u32 = 1;
pub const PACKAGE_KEY_ENVELOPE_SCHEMA_VERSION: u32 = 1;
pub const P256_PUBLIC_KEY_BYTE_LEN: usize = 65;
pub const AES_256_GCM_NONCE_BYTE_LEN: usize = 12;
pub const AES_256_GCM_WRAPPED_KEY_BYTE_LEN: usize = 48;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ArtifactContentCipher {
    Aes256Gcm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RecipientKeyAlgorithm {
    P256,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum KeyEnvelopeAlgorithm {
    P256HkdfSha256Aes256Gcm,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RecipientEncryptionKey {
    pub id: String,
    pub algorithm: RecipientKeyAlgorithm,
    pub public_key: String,
    pub fingerprint: ContentDigest,
}

impl RecipientEncryptionKey {
    pub fn validate(&self) -> Result<(), String> {
        if self.id.trim().is_empty() || self.id.len() > 128 {
            return Err("recipient encryption key id is invalid".to_string());
        }
        let public_key = decode_exact(
            &self.public_key,
            P256_PUBLIC_KEY_BYTE_LEN,
            "recipient encryption public key",
        )?;
        if p256::PublicKey::from_sec1_bytes(&public_key).is_err()
            || self.fingerprint != ContentDigest::sha256(&public_key)
        {
            return Err("recipient encryption public key is invalid".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct EncryptedArtifactMetadata {
    pub schema_version: u32,
    pub content_cipher: ArtifactContentCipher,
    pub key_version: u64,
    pub plaintext_digest: ContentDigest,
    pub plaintext_byte_len: u64,
    pub nonce: String,
    pub aad_digest: ContentDigest,
}

impl EncryptedArtifactMetadata {
    pub fn new(
        key_version: u64,
        plaintext: &[u8],
        nonce: [u8; AES_256_GCM_NONCE_BYTE_LEN],
        source: &RegistrySourceId,
    ) -> Result<Self, String> {
        let mut value = Self {
            schema_version: ENCRYPTED_ARTIFACT_SCHEMA_VERSION,
            content_cipher: ArtifactContentCipher::Aes256Gcm,
            key_version,
            plaintext_digest: ContentDigest::sha256(plaintext),
            plaintext_byte_len: plaintext.len() as u64,
            nonce: URL_SAFE_NO_PAD.encode(nonce),
            aad_digest: ContentDigest::sha256([]),
        };
        value.aad_digest = ContentDigest::sha256(value.aad_bytes(source)?);
        value.validate(source)?;
        Ok(value)
    }

    pub fn validate(&self, source: &RegistrySourceId) -> Result<(), String> {
        if self.schema_version != ENCRYPTED_ARTIFACT_SCHEMA_VERSION
            || self.key_version == 0
            || self.plaintext_byte_len == 0
        {
            return Err("encrypted artifact metadata is invalid".to_string());
        }
        decode_exact(
            &self.nonce,
            AES_256_GCM_NONCE_BYTE_LEN,
            "encrypted artifact nonce",
        )?;
        if self.aad_digest != ContentDigest::sha256(self.aad_bytes(source)?) {
            return Err("encrypted artifact AAD digest is invalid".to_string());
        }
        Ok(())
    }

    pub fn aad_bytes(&self, source: &RegistrySourceId) -> Result<Vec<u8>, String> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Canonical<'a> {
            format: &'static str,
            registry: &'a str,
            namespace: &'a str,
            name: &'a str,
            version: String,
            tree_digest: String,
            content_cipher: ArtifactContentCipher,
            key_version: u64,
            plaintext_digest: String,
            plaintext_byte_len: u64,
        }
        if self.schema_version != ENCRYPTED_ARTIFACT_SCHEMA_VERSION
            || self.key_version == 0
            || self.plaintext_byte_len == 0
        {
            return Err("encrypted artifact metadata is invalid".to_string());
        }
        serde_json::to_vec(&Canonical {
            format: "runmat-private-artifact-aad-v1",
            registry: source.registry_origin.as_str(),
            namespace: source.package.organization(),
            name: source.package.name(),
            version: source.version.to_string(),
            tree_digest: source.tree_digest.to_string(),
            content_cipher: self.content_cipher,
            key_version: self.key_version,
            plaintext_digest: self.plaintext_digest.to_string(),
            plaintext_byte_len: self.plaintext_byte_len,
        })
        .map_err(|error| error.to_string())
    }

    pub fn decoded_nonce(&self) -> Result<[u8; AES_256_GCM_NONCE_BYTE_LEN], String> {
        decode_exact(
            &self.nonce,
            AES_256_GCM_NONCE_BYTE_LEN,
            "encrypted artifact nonce",
        )?
        .try_into()
        .map_err(|_| "encrypted artifact nonce is invalid".to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PackageKeyEnvelope {
    pub schema_version: u32,
    pub algorithm: KeyEnvelopeAlgorithm,
    pub recipient_key_id: String,
    pub recipient_key_fingerprint: ContentDigest,
    pub ephemeral_public_key: String,
    pub nonce: String,
    pub wrapped_key: String,
    pub context_digest: ContentDigest,
}

impl PackageKeyEnvelope {
    pub fn validate(
        &self,
        recipient: &RecipientEncryptionKey,
        artifact: &EncryptedArtifactMetadata,
    ) -> Result<(), String> {
        if self.schema_version != PACKAGE_KEY_ENVELOPE_SCHEMA_VERSION
            || self.recipient_key_id != recipient.id
            || self.recipient_key_fingerprint != recipient.fingerprint
        {
            return Err("package key envelope recipient is invalid".to_string());
        }
        recipient.validate()?;
        let ephemeral_public_key = decode_exact(
            &self.ephemeral_public_key,
            P256_PUBLIC_KEY_BYTE_LEN,
            "package key envelope ephemeral public key",
        )?;
        if p256::PublicKey::from_sec1_bytes(&ephemeral_public_key).is_err() {
            return Err("package key envelope ephemeral public key is invalid".to_string());
        }
        decode_exact(
            &self.nonce,
            AES_256_GCM_NONCE_BYTE_LEN,
            "package key envelope nonce",
        )?;
        decode_exact(
            &self.wrapped_key,
            AES_256_GCM_WRAPPED_KEY_BYTE_LEN,
            "wrapped package content key",
        )?;
        if self.context_digest != ContentDigest::sha256(self.context_bytes(artifact)?) {
            return Err("package key envelope context digest is invalid".to_string());
        }
        Ok(())
    }

    pub fn context_bytes(&self, artifact: &EncryptedArtifactMetadata) -> Result<Vec<u8>, String> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Canonical<'a> {
            format: &'static str,
            artifact_aad_digest: String,
            key_version: u64,
            recipient_key_id: &'a str,
            recipient_key_fingerprint: String,
            ephemeral_public_key: &'a str,
            algorithm: KeyEnvelopeAlgorithm,
        }
        if self.recipient_key_id.trim().is_empty() || self.recipient_key_id.len() > 128 {
            return Err("package key envelope recipient is invalid".to_string());
        }
        serde_json::to_vec(&Canonical {
            format: "runmat-package-key-envelope-context-v1",
            artifact_aad_digest: artifact.aad_digest.to_string(),
            key_version: artifact.key_version,
            recipient_key_id: &self.recipient_key_id,
            recipient_key_fingerprint: self.recipient_key_fingerprint.to_string(),
            ephemeral_public_key: &self.ephemeral_public_key,
            algorithm: self.algorithm,
        })
        .map_err(|error| error.to_string())
    }

    pub fn decoded_ephemeral_public_key(&self) -> Result<Vec<u8>, String> {
        decode_exact(
            &self.ephemeral_public_key,
            P256_PUBLIC_KEY_BYTE_LEN,
            "package key envelope ephemeral public key",
        )
    }

    pub fn decoded_nonce(&self) -> Result<[u8; AES_256_GCM_NONCE_BYTE_LEN], String> {
        decode_exact(
            &self.nonce,
            AES_256_GCM_NONCE_BYTE_LEN,
            "package key envelope nonce",
        )?
        .try_into()
        .map_err(|_| "package key envelope nonce is invalid".to_string())
    }

    pub fn decoded_wrapped_key(&self) -> Result<Vec<u8>, String> {
        decode_exact(
            &self.wrapped_key,
            AES_256_GCM_WRAPPED_KEY_BYTE_LEN,
            "wrapped package content key",
        )
    }
}

fn decode_exact(value: &str, expected: usize, label: &str) -> Result<Vec<u8>, String> {
    let decoded = URL_SAFE_NO_PAD
        .decode(value)
        .map_err(|_| format!("{label} is not canonical base64url"))?;
    if decoded.len() != expected || URL_SAFE_NO_PAD.encode(&decoded) != value {
        return Err(format!("{label} has an invalid length or encoding"));
    }
    Ok(decoded)
}
