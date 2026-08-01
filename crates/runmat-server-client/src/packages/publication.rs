use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StagePublicationRequest {
    pub version: String,
    pub artifact: PublicationArtifactRequest,
    pub metadata: PublicationMetadataRequest,
    pub idempotency_key: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PublicationArtifactRequest {
    pub digest: String,
    pub tree_digest: String,
    pub byte_len: u64,
    pub media_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encryption: Option<runmat_package::EncryptedArtifactMetadata>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PublicationMetadataRequest {
    pub singleton: bool,
    pub runmat_requirement: Option<String>,
    pub dependencies: Vec<PublicationDependencyRequest>,
    pub features: BTreeMap<String, Vec<String>>,
    pub required_capabilities: Vec<String>,
    pub optional_capabilities: Vec<String>,
    pub readme_digest: Option<String>,
    pub license: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PublicationDependencyRequest {
    pub alias: String,
    pub registry: String,
    pub namespace: String,
    pub name: String,
    pub requirement: String,
    pub group: String,
    pub target: Option<String>,
    pub optional: bool,
    pub default_features: bool,
    pub features: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StagePublicationResponse {
    pub id: String,
    pub status: String,
    pub upload_url: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AttachKeyEnvelopesRequest {
    pub envelopes: Vec<KeyEnvelopeRequest>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct KeyEnvelopeRequest {
    pub recipient_key_id: String,
    pub recipient_key_fingerprint: String,
    pub ephemeral_public_key: String,
    pub nonce: String,
    pub wrapped_key: String,
    pub context_digest: String,
}

impl From<runmat_package::PackageKeyEnvelope> for KeyEnvelopeRequest {
    fn from(value: runmat_package::PackageKeyEnvelope) -> Self {
        Self {
            recipient_key_id: value.recipient_key_id,
            recipient_key_fingerprint: value.recipient_key_fingerprint.to_string(),
            ephemeral_public_key: value.ephemeral_public_key,
            nonce: value.nonce,
            wrapped_key: value.wrapped_key,
            context_digest: value.context_digest.to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PublicationStatusResponse {
    pub id: String,
    pub status: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FinalizePublicationResponse {
    pub publication_id: String,
    pub release_id: String,
    pub release_digest: String,
    pub version: String,
}
