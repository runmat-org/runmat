use runmat_package::{CanonicalPackageId, ContentDigest, NormalizedRelativePath, PackageVersion};
use serde::{Deserialize, Serialize};

use crate::CacheError;

pub const RELEASE_INVENTORY_SCHEMA_VERSION: u32 = 1;
pub const RELEASE_MANIFEST_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ArtifactEntryRole {
    Source,
    Resource,
    Native,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PublicationEntryContent {
    File(Vec<u8>),
    Directory,
    Symlink(NormalizedRelativePath),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicationEntry {
    pub path: NormalizedRelativePath,
    pub content: PublicationEntryContent,
    pub executable: bool,
    pub role: ArtifactEntryRole,
}

impl PublicationEntry {
    pub fn file(
        path: impl AsRef<str>,
        bytes: Vec<u8>,
        executable: bool,
        role: ArtifactEntryRole,
    ) -> Result<Self, CacheError> {
        Ok(Self {
            path: NormalizedRelativePath::new(path.as_ref())
                .map_err(|error| CacheError::InvalidObject(error.to_string()))?,
            content: PublicationEntryContent::File(bytes),
            executable,
            role,
        })
    }

    pub fn directory(path: impl AsRef<str>) -> Result<Self, CacheError> {
        Ok(Self {
            path: NormalizedRelativePath::new(path.as_ref())
                .map_err(|error| CacheError::InvalidObject(error.to_string()))?,
            content: PublicationEntryContent::Directory,
            executable: false,
            role: ArtifactEntryRole::Resource,
        })
    }

    pub fn symlink(
        path: impl AsRef<str>,
        target: impl AsRef<str>,
        role: ArtifactEntryRole,
    ) -> Result<Self, CacheError> {
        Ok(Self {
            path: NormalizedRelativePath::new(path.as_ref())
                .map_err(|error| CacheError::InvalidObject(error.to_string()))?,
            content: PublicationEntryContent::Symlink(
                NormalizedRelativePath::new(target.as_ref())
                    .map_err(|error| CacheError::InvalidObject(error.to_string()))?,
            ),
            executable: false,
            role,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReleaseInventoryEntry {
    pub path: NormalizedRelativePath,
    pub role: ArtifactEntryRole,
    pub kind: crate::TreeEntryKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub digest: Option<ContentDigest>,
    pub byte_len: u64,
    pub executable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub link_target: Option<NormalizedRelativePath>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReleaseInventory {
    pub schema_version: u32,
    pub entries: Vec<ReleaseInventoryEntry>,
    pub file_count: u64,
    pub total_bytes: u64,
    pub digest: ContentDigest,
}

impl ReleaseInventory {
    pub(crate) fn new(mut entries: Vec<ReleaseInventoryEntry>) -> Result<Self, CacheError> {
        entries.sort();
        if entries.windows(2).any(|pair| pair[0].path == pair[1].path) {
            return Err(CacheError::InvalidObject(
                "release inventory contains duplicate paths".to_string(),
            ));
        }
        let file_count = entries
            .iter()
            .filter(|entry| entry.kind == crate::TreeEntryKind::File)
            .count() as u64;
        let total_bytes = entries.iter().try_fold(0u64, |total, entry| {
            total
                .checked_add(entry.byte_len)
                .ok_or_else(|| CacheError::InvalidObject("release size overflow".to_string()))
        })?;
        let digest = inventory_digest(&entries, file_count, total_bytes)?;
        Ok(Self {
            schema_version: RELEASE_INVENTORY_SCHEMA_VERSION,
            entries,
            file_count,
            total_bytes,
            digest,
        })
    }

    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version != RELEASE_INVENTORY_SCHEMA_VERSION {
            return Err(CacheError::InvalidObject(format!(
                "unsupported release inventory schema {}",
                self.schema_version
            )));
        }
        if Self::new(self.entries.clone())? != *self {
            return Err(CacheError::InvalidObject(
                "release inventory is not canonical".to_string(),
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, CacheError> {
        self.validate()?;
        serde_json::to_vec(self).map_err(|error| CacheError::InvalidObject(error.to_string()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReleaseManifest {
    pub schema_version: u32,
    pub package: CanonicalPackageId,
    pub version: PackageVersion,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runmat_requirement: Option<String>,
    pub inventory_digest: ContentDigest,
    pub tree_digest: ContentDigest,
    pub artifact_digest: ContentDigest,
    pub artifact_byte_len: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub readme: Option<NormalizedRelativePath>,
}

impl ReleaseManifest {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        package: CanonicalPackageId,
        version: PackageVersion,
        runmat_requirement: Option<String>,
        bundle: &ReleaseArtifactBundle,
        license: Option<String>,
        readme: Option<NormalizedRelativePath>,
    ) -> Result<Self, CacheError> {
        if readme.as_ref().is_some_and(|readme| {
            !bundle
                .inventory
                .entries
                .iter()
                .any(|entry| entry.path == *readme && entry.kind == crate::TreeEntryKind::File)
        }) {
            return Err(CacheError::InvalidObject(
                "release README is not a selected publication file".to_string(),
            ));
        }
        let manifest = Self {
            schema_version: RELEASE_MANIFEST_SCHEMA_VERSION,
            package,
            version,
            runmat_requirement,
            inventory_digest: bundle.inventory.digest.clone(),
            tree_digest: bundle.tree_digest.clone(),
            artifact_digest: bundle.artifact_digest.clone(),
            artifact_byte_len: bundle.artifact_bytes.len() as u64,
            license,
            readme,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version != RELEASE_MANIFEST_SCHEMA_VERSION
            || self.artifact_byte_len == 0
            || self
                .runmat_requirement
                .as_deref()
                .map(semver::VersionReq::parse)
                .transpose()
                .is_err()
            || self
                .license
                .as_deref()
                .is_some_and(|license| license.trim().is_empty() || license.len() > 256)
        {
            return Err(CacheError::InvalidObject(
                "release manifest contains invalid fields".to_string(),
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, CacheError> {
        self.validate()?;
        serde_json::to_vec(self).map_err(|error| CacheError::InvalidObject(error.to_string()))
    }

    pub fn digest(&self) -> Result<ContentDigest, CacheError> {
        self.canonical_bytes().map(ContentDigest::sha256)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseArtifactBundle {
    pub artifact_bytes: Vec<u8>,
    pub artifact_digest: ContentDigest,
    pub tree_digest: ContentDigest,
    pub inventory: ReleaseInventory,
}

fn inventory_digest(
    entries: &[ReleaseInventoryEntry],
    file_count: u64,
    total_bytes: u64,
) -> Result<ContentDigest, CacheError> {
    #[derive(Serialize)]
    #[serde(rename_all = "camelCase")]
    struct Input<'a> {
        format: &'static str,
        entries: &'a [ReleaseInventoryEntry],
        file_count: u64,
        total_bytes: u64,
    }
    serde_json::to_vec(&Input {
        format: "runmat-release-inventory-v1",
        entries,
        file_count,
        total_bytes,
    })
    .map(ContentDigest::sha256)
    .map_err(|error| CacheError::InvalidObject(error.to_string()))
}
