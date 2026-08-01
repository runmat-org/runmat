use base64::{engine::general_purpose::STANDARD_NO_PAD, Engine as _};
use chrono::{DateTime, Utc};
use ed25519_dalek::{Signature, Verifier as _, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

pub const RELEASE_SUPPLY_CHAIN_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PackageTrustTier {
    Official,
    VerifiedWrapper,
    Community,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SbomFormat {
    CycloneDxJson,
    SpdxJson,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SbomReference {
    pub format: SbomFormat,
    pub digest: String,
    pub media_type: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WrapperProvenance {
    pub upstream_repository: String,
    pub upstream_version: String,
    pub upstream_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BuildProvenance {
    pub source_repository: String,
    pub source_commit: String,
    pub builder_id: String,
    pub workflow_ref: Option<String>,
    pub invocation_id: String,
    pub inventory_digest: String,
    pub release_manifest_digest: String,
    pub sbom: Option<SbomReference>,
    pub license: Option<String>,
    pub wrapper: Option<WrapperProvenance>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RegistryReleaseSupplyChain {
    pub schema_version: u32,
    pub publication_id: String,
    pub publisher_id: String,
    pub publisher_name: String,
    pub public_key: String,
    pub key_fingerprint: String,
    pub payload_digest: String,
    pub sequence: u64,
    pub signed_at: DateTime<Utc>,
    pub trust_tier: PackageTrustTier,
    pub provenance: BuildProvenance,
    pub signature: String,
}

impl RegistryReleaseSupplyChain {
    pub fn verify(&self, package_id: &str) -> Result<(), String> {
        self.validate()?;
        let key_bytes = STANDARD_NO_PAD
            .decode(&self.public_key)
            .map_err(|_| "trusted publisher public key is invalid".to_string())?;
        let key_bytes: [u8; 32] = key_bytes
            .try_into()
            .map_err(|_| "trusted publisher public key is invalid".to_string())?;
        let key = VerifyingKey::from_bytes(&key_bytes)
            .map_err(|_| "trusted publisher public key is invalid".to_string())?;
        let fingerprint = format!("sha256:{:x}", Sha256::digest(key.as_bytes()));
        if fingerprint != self.key_fingerprint {
            return Err("trusted publisher key fingerprint differs from its key".to_string());
        }
        let signature_bytes = STANDARD_NO_PAD
            .decode(&self.signature)
            .map_err(|_| "publication signature is invalid".to_string())?;
        let signature = Signature::from_slice(&signature_bytes)
            .map_err(|_| "publication signature is invalid".to_string())?;
        key.verify(&self.canonical_signing_bytes(package_id)?, &signature)
            .map_err(|_| "publication signature is invalid".to_string())
    }

    pub fn canonical_signing_bytes(&self, package_id: &str) -> Result<Vec<u8>, String> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Canonical<'a> {
            format: &'static str,
            publication_id: &'a str,
            package_id: &'a str,
            publisher_id: &'a str,
            payload_digest: &'a str,
            sequence: u64,
            issued_at_micros: i64,
            provenance: &'a BuildProvenance,
        }
        serde_json::to_vec(&Canonical {
            format: "runmat-publication-attestation-v1",
            publication_id: &self.publication_id,
            package_id,
            publisher_id: &self.publisher_id,
            payload_digest: &self.payload_digest,
            sequence: self.sequence,
            issued_at_micros: self.signed_at.timestamp_micros(),
            provenance: &self.provenance,
        })
        .map_err(|error| error.to_string())
    }

    fn validate(&self) -> Result<(), String> {
        if self.schema_version != RELEASE_SUPPLY_CHAIN_SCHEMA_VERSION
            || self.publication_id.is_empty()
            || self.publisher_id.is_empty()
            || self.publisher_name.trim().is_empty()
            || self.publisher_name.len() > 128
            || self.sequence == 0
            || !valid_sha256(&self.key_fingerprint)
            || !valid_sha256(&self.payload_digest)
            || self.public_key.is_empty()
            || self.public_key.len() > 256
            || self.signature.is_empty()
            || self.signature.len() > 256
            || (matches!(self.trust_tier, PackageTrustTier::VerifiedWrapper)
                && self.provenance.wrapper.is_none())
        {
            return Err("release supply-chain metadata is invalid".to_string());
        }
        self.provenance.validate()
    }
}

impl BuildProvenance {
    fn validate(&self) -> Result<(), String> {
        if !valid_https_url(&self.source_repository)
            || !matches!(self.source_commit.len(), 40 | 64)
            || !self
                .source_commit
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
            || self.builder_id.trim().is_empty()
            || self.builder_id.len() > 512
            || self
                .workflow_ref
                .as_deref()
                .is_some_and(|value| value.trim().is_empty() || value.len() > 512)
            || self.invocation_id.trim().is_empty()
            || self.invocation_id.len() > 512
            || !valid_sha256(&self.inventory_digest)
            || !valid_sha256(&self.release_manifest_digest)
            || self.sbom.as_ref().is_some_and(|value| {
                !valid_sha256(&value.digest)
                    || value.media_type.trim().is_empty()
                    || value.media_type.len() > 128
            })
            || self
                .license
                .as_deref()
                .is_some_and(|value| value.trim().is_empty() || value.len() > 256)
            || self.wrapper.as_ref().is_some_and(|value| {
                !valid_https_url(&value.upstream_repository)
                    || value.upstream_version.trim().is_empty()
                    || value.upstream_version.len() > 256
                    || !valid_sha256(&value.upstream_digest)
            })
        {
            return Err("build provenance is invalid".to_string());
        }
        Ok(())
    }
}

fn valid_https_url(value: &str) -> bool {
    url::Url::parse(value).ok().is_some_and(|url| {
        url.scheme() == "https"
            && url.username().is_empty()
            && url.password().is_none()
            && url.query().is_none()
            && url.fragment().is_none()
            && value.len() <= 2048
    })
}

fn valid_sha256(value: &str) -> bool {
    value.strip_prefix("sha256:").is_some_and(|hex| {
        hex.len() == 64
            && hex
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}
