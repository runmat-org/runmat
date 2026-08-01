use crate::{BlobMetadata, CacheError, TreeManifest};
use runmat_package::{
    ContentDigest, GitSourceId, RegistrySourceId, ServerProjectSourceId, SourceId,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SnapshotBlob {
    pub digest: ContentDigest,
    #[serde(with = "super::base64_bytes")]
    pub bytes: Vec<u8>,
}

impl SnapshotBlob {
    pub fn new(bytes: Vec<u8>) -> Self {
        Self {
            digest: ContentDigest::sha256(&bytes),
            bytes,
        }
    }

    pub fn validate(&self) -> Result<(), CacheError> {
        BlobMetadata {
            digest: self.digest.clone(),
            byte_len: self.bytes.len() as u64,
        }
        .verify(&self.bytes)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GitSnapshot {
    pub source: GitSourceId,
    pub tree: TreeManifest,
    pub blobs: Vec<SnapshotBlob>,
}

impl GitSnapshot {
    pub fn new(
        source: GitSourceId,
        tree: TreeManifest,
        mut blobs: Vec<SnapshotBlob>,
    ) -> Result<Self, CacheError> {
        blobs.sort_by(|left, right| left.digest.cmp(&right.digest));
        let snapshot = Self {
            source,
            tree,
            blobs,
        };
        snapshot.validate()?;
        Ok(snapshot)
    }

    pub fn validate(&self) -> Result<(), CacheError> {
        validate_snapshot(
            SourceId::Git(self.source.clone()),
            &self.tree,
            &self.blobs,
            "Git",
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerProjectSnapshot {
    pub source: ServerProjectSourceId,
    pub tree: TreeManifest,
    pub blobs: Vec<SnapshotBlob>,
}

impl ServerProjectSnapshot {
    pub fn new(
        source: ServerProjectSourceId,
        tree: TreeManifest,
        mut blobs: Vec<SnapshotBlob>,
    ) -> Result<Self, CacheError> {
        blobs.sort_by(|left, right| left.digest.cmp(&right.digest));
        let snapshot = Self {
            source,
            tree,
            blobs,
        };
        validate_snapshot(
            SourceId::ServerProject(snapshot.source.clone()),
            &snapshot.tree,
            &snapshot.blobs,
            "Server project",
        )?;
        Ok(snapshot)
    }

    pub fn validate(&self) -> Result<(), CacheError> {
        validate_snapshot(
            SourceId::ServerProject(self.source.clone()),
            &self.tree,
            &self.blobs,
            "Server project",
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegistrySnapshot {
    pub source: RegistrySourceId,
    pub tree: TreeManifest,
    pub blobs: Vec<SnapshotBlob>,
}

impl RegistrySnapshot {
    pub fn new(
        source: RegistrySourceId,
        tree: TreeManifest,
        mut blobs: Vec<SnapshotBlob>,
    ) -> Result<Self, CacheError> {
        blobs.sort_by(|left, right| left.digest.cmp(&right.digest));
        let snapshot = Self {
            source,
            tree,
            blobs,
        };
        snapshot.validate()?;
        Ok(snapshot)
    }

    pub fn validate(&self) -> Result<(), CacheError> {
        validate_snapshot(
            SourceId::Registry(self.source.clone()),
            &self.tree,
            &self.blobs,
            "Registry",
        )
    }
}

fn validate_snapshot(
    source: SourceId,
    tree: &TreeManifest,
    blobs: &[SnapshotBlob],
    kind: &str,
) -> Result<(), CacheError> {
    source
        .validate()
        .map_err(|error| CacheError::InvalidObject(error.to_string()))?;
    tree.validate()?;
    if source.tree_digest() != &tree.digest {
        return Err(CacheError::InvalidObject(format!(
            "{kind} source tree digest does not match its tree manifest"
        )));
    }
    if blobs
        .windows(2)
        .any(|pair| pair[0].digest >= pair[1].digest)
    {
        return Err(CacheError::InvalidObject(format!(
            "{kind} snapshot blobs must be strictly digest-sorted"
        )));
    }
    let mut available = BTreeMap::new();
    for blob in blobs {
        blob.validate()?;
        available.insert(blob.digest.clone(), blob.bytes.len() as u64);
    }
    let referenced: BTreeSet<_> = tree.referenced_blobs();
    let supplied: BTreeSet<_> = available.keys().cloned().collect();
    if supplied != referenced {
        return Err(CacheError::InvalidObject(format!(
            "{kind} snapshot blob closure does not exactly match the tree"
        )));
    }
    for entry in &tree.entries {
        if let Some(digest) = &entry.digest {
            if available.get(digest) != Some(&entry.byte_len) {
                return Err(CacheError::InvalidObject(format!(
                    "{kind} snapshot file `{}` size differs from its blob",
                    entry.path
                )));
            }
        }
    }
    Ok(())
}
