use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RegistryDependency {
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RegistryArtifact {
    pub id: String,
    pub digest: String,
    pub tree_digest: String,
    pub byte_len: u64,
    pub media_type: String,
    pub download_url: String,
    pub expires_at: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RegistryAdvisory {
    pub id: String,
    pub affected_requirement: String,
    pub severity: String,
    pub title: String,
    pub url: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RegistryReleaseMetadata {
    #[serde(flatten)]
    pub release: RegistryReleaseCore,
    pub artifact: RegistryArtifact,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RegistryReleaseCore {
    pub package_id: String,
    pub release_id: String,
    pub registry: String,
    pub namespace: String,
    pub name: String,
    pub version: String,
    pub release_digest: String,
    pub singleton: bool,
    pub runmat_requirement: Option<String>,
    pub features: BTreeMap<String, Vec<String>>,
    pub required_capabilities: Vec<String>,
    pub optional_capabilities: Vec<String>,
    pub readme_digest: Option<String>,
    pub license: Option<String>,
    pub dependencies: Vec<RegistryDependency>,
    pub advisories: Vec<RegistryAdvisory>,
    #[serde(default)]
    pub supply_chain: Option<runmat_package::RegistryReleaseSupplyChain>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RegistryCandidateArtifact {
    pub id: String,
    pub digest: String,
    pub tree_digest: String,
    pub byte_len: u64,
    pub media_type: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RegistryCandidate {
    #[serde(flatten)]
    pub release: RegistryReleaseCore,
    pub artifact: RegistryCandidateArtifact,
    pub yanked: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RegistryCandidateList {
    pub candidates: Vec<RegistryCandidate>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegistryCandidateResponse {
    pub candidates: Vec<RegistryCandidate>,
    pub etag: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryCandidateOutcome {
    Candidates(RegistryCandidateResponse),
    NotModified,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegistryReleaseResponse {
    pub metadata: RegistryReleaseMetadata,
    pub etag: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryReleaseOutcome {
    Release(Box<RegistryReleaseResponse>),
    NotModified,
}

#[derive(Debug, Error)]
pub enum RegistryClientError {
    #[error("registry index URL is invalid")]
    InvalidIndex,
    #[error("registry response contains an unsafe artifact URL")]
    UnsafeArtifactUrl,
    #[error("registry request is unauthorized")]
    Unauthorized,
    #[error("registry request is forbidden")]
    Forbidden,
    #[error("registry package or release was not found")]
    NotFound,
    #[error("registry request was rate limited")]
    RateLimited,
    #[error("registry service is unavailable")]
    Unavailable,
    #[error("registry response exceeds the {limit}-byte transfer limit")]
    TooLarge { limit: u64 },
    #[error("registry response is invalid: {0}")]
    InvalidResponse(String),
    #[error("registry artifact length differs from signed metadata")]
    LengthMismatch,
    #[error("registry artifact digest differs from signed metadata")]
    DigestMismatch,
}
