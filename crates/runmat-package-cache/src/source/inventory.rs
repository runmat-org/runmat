use super::{GitSnapshot, RegistrySnapshot, ServerProjectSnapshot, SnapshotBlob};
use crate::{
    validate_archive, ArchiveEntryHeader, ArchiveEntryKind, ArchiveLimits, CacheError, TreeEntry,
    TreeManifest,
};
use runmat_package::{
    ContentDigest, GitCommitId, GitSourceId, NormalizedRelativePath, RegistrySourceId,
    ServerProjectSourceId,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TreeInventoryEntryKind {
    File,
    Directory,
    Symlink,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TreeInventoryEntry {
    pub path: String,
    pub kind: TreeInventoryEntryKind,
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        with = "super::base64_bytes"
    )]
    pub bytes: Vec<u8>,
    #[serde(default)]
    pub executable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub link_target: Option<String>,
}

impl TreeInventoryEntry {
    pub fn file(path: impl Into<String>, bytes: Vec<u8>, executable: bool) -> Self {
        Self {
            path: path.into(),
            kind: TreeInventoryEntryKind::File,
            bytes,
            executable,
            link_target: None,
        }
    }

    pub fn directory(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            kind: TreeInventoryEntryKind::Directory,
            bytes: Vec::new(),
            executable: false,
            link_target: None,
        }
    }

    pub fn symlink(path: impl Into<String>, link_target: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            kind: TreeInventoryEntryKind::Symlink,
            bytes: Vec::new(),
            executable: false,
            link_target: Some(link_target.into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GitTreeInventory {
    pub commit: String,
    pub entries: Vec<TreeInventoryEntry>,
}

impl GitTreeInventory {
    pub fn into_snapshot(
        self,
        repository: &str,
        subdir: &str,
        limits: ArchiveLimits,
    ) -> Result<GitSnapshot, CacheError> {
        let commit: GitCommitId =
            self.commit
                .parse()
                .map_err(|error: runmat_package::IdentityError| {
                    CacheError::InvalidObject(error.to_string())
                })?;
        let subdir = NormalizedRelativePath::new(subdir)
            .map_err(|error| CacheError::InvalidObject(error.to_string()))?;
        let (tree, blobs) = inventory_tree(self.entries, limits)?;
        let source = GitSourceId::new(repository, commit, subdir, tree.digest.clone())
            .map_err(|error| CacheError::InvalidObject(error.to_string()))?;
        GitSnapshot::new(source, tree, blobs)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ServerProjectTreeInventory {
    pub project: String,
    pub snapshot: String,
    pub tree_digest: runmat_package::ContentDigest,
    pub entries: Vec<TreeInventoryEntry>,
}

impl ServerProjectTreeInventory {
    pub fn into_snapshot(
        self,
        service: &str,
        limits: ArchiveLimits,
    ) -> Result<ServerProjectSnapshot, CacheError> {
        let (tree, blobs) = inventory_tree(self.entries, limits)?;
        if tree.digest != self.tree_digest {
            return Err(CacheError::InvalidObject(
                "Server snapshot tree digest differs from the canonical inventory".to_string(),
            ));
        }
        let source =
            ServerProjectSourceId::new(service, self.project, self.snapshot, tree.digest.clone())
                .map_err(|error| CacheError::InvalidObject(error.to_string()))?;
        ServerProjectSnapshot::new(source, tree, blobs)
    }
}

pub const REGISTRY_ARTIFACT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RegistryArtifactInventory {
    pub schema_version: u32,
    pub entries: Vec<TreeInventoryEntry>,
}

impl RegistryArtifactInventory {
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, CacheError> {
        if self.schema_version != REGISTRY_ARTIFACT_SCHEMA_VERSION {
            return Err(CacheError::InvalidObject(format!(
                "unsupported registry artifact schema version {}",
                self.schema_version
            )));
        }
        inventory_tree(self.entries.clone(), ArchiveLimits::default())?;
        let mut entries = self.entries.clone();
        for entry in &entries {
            let normalized = NormalizedRelativePath::new(&entry.path)
                .map_err(|error| CacheError::InvalidObject(error.to_string()))?;
            if normalized.as_str() != entry.path {
                return Err(CacheError::InvalidObject(format!(
                    "registry artifact path `{}` is not canonical",
                    entry.path
                )));
            }
        }
        entries.sort_by(|left, right| left.path.cmp(&right.path));
        serde_json::to_vec(&Self {
            schema_version: self.schema_version,
            entries,
        })
        .map_err(|error| CacheError::InvalidObject(error.to_string()))
    }

    pub fn decode_snapshot(
        artifact_bytes: &[u8],
        source: RegistrySourceId,
        limits: ArchiveLimits,
    ) -> Result<RegistrySnapshot, CacheError> {
        let expected = source.artifact_digest.clone();
        Self::decode_snapshot_with_digest(artifact_bytes, source, expected, limits)
    }

    pub fn decode_decrypted_snapshot(
        plaintext_bytes: &[u8],
        source: RegistrySourceId,
        plaintext_digest: ContentDigest,
        limits: ArchiveLimits,
    ) -> Result<RegistrySnapshot, CacheError> {
        Self::decode_snapshot_with_digest(plaintext_bytes, source, plaintext_digest, limits)
    }

    fn decode_snapshot_with_digest(
        artifact_bytes: &[u8],
        source: RegistrySourceId,
        expected_artifact_digest: ContentDigest,
        limits: ArchiveLimits,
    ) -> Result<RegistrySnapshot, CacheError> {
        if ContentDigest::sha256(artifact_bytes) != expected_artifact_digest {
            return Err(CacheError::DigestMismatch(expected_artifact_digest));
        }
        let inventory: Self = serde_json::from_slice(artifact_bytes)
            .map_err(|error| CacheError::InvalidObject(error.to_string()))?;
        if inventory.schema_version != REGISTRY_ARTIFACT_SCHEMA_VERSION {
            return Err(CacheError::InvalidObject(format!(
                "unsupported registry artifact schema version {}",
                inventory.schema_version
            )));
        }
        let (tree, blobs) = inventory_tree(inventory.entries, limits)?;
        if tree.digest != source.tree_digest {
            return Err(CacheError::InvalidObject(
                "registry artifact tree digest differs from locked metadata".to_string(),
            ));
        }
        RegistrySnapshot::new(source, tree, blobs)
    }
}

pub type GitInventoryEntry = TreeInventoryEntry;
pub type GitInventoryEntryKind = TreeInventoryEntryKind;

fn inventory_tree(
    entries: Vec<TreeInventoryEntry>,
    limits: ArchiveLimits,
) -> Result<(TreeManifest, Vec<SnapshotBlob>), CacheError> {
    let headers = entries.iter().map(|entry| ArchiveEntryHeader {
        path: entry.path.clone(),
        kind: match entry.kind {
            TreeInventoryEntryKind::File => ArchiveEntryKind::File,
            TreeInventoryEntryKind::Directory => ArchiveEntryKind::Directory,
            TreeInventoryEntryKind::Symlink => ArchiveEntryKind::Symlink,
        },
        expanded_bytes: entry.bytes.len() as u64,
        compressed_bytes: entry.bytes.len() as u64,
        link_target: entry.link_target.clone(),
        executable: entry.executable,
    });
    let validated = validate_archive(headers, limits)
        .map_err(|error| CacheError::InvalidObject(error.to_string()))?;
    let originals = entries
        .into_iter()
        .map(|entry| {
            NormalizedRelativePath::new(&entry.path)
                .map(|path| (path, entry))
                .map_err(|error| CacheError::InvalidObject(error.to_string()))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let mut tree_entries = Vec::with_capacity(validated.entries.len());
    let mut blobs = BTreeMap::new();
    for entry in validated.entries {
        let original = originals.get(&entry.path).ok_or_else(|| {
            CacheError::InvalidObject(format!(
                "validated tree entry `{}` has no source entry",
                entry.path
            ))
        })?;
        match entry.kind {
            ArchiveEntryKind::File => {
                let blob = SnapshotBlob::new(original.bytes.clone());
                tree_entries.push(TreeEntry::file(
                    entry.path,
                    blob.digest.clone(),
                    blob.bytes.len() as u64,
                    entry.executable,
                ));
                blobs.insert(blob.digest.clone(), blob);
            }
            ArchiveEntryKind::Directory => {
                tree_entries.push(TreeEntry::directory(entry.path));
            }
            ArchiveEntryKind::Symlink => {
                tree_entries.push(TreeEntry::symlink(
                    entry.path,
                    entry
                        .link_target
                        .expect("validated symbolic link has a target"),
                ));
            }
            kind => {
                return Err(CacheError::InvalidObject(format!(
                    "tree inventory cannot contain {kind:?}"
                )));
            }
        }
    }
    let tree = TreeManifest::new(tree_entries)?;
    Ok((tree, blobs.into_values().collect()))
}
