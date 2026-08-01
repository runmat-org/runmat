use crate::{BlobMetadata, CacheError, TreeManifest};
use runmat_package::{ContentDigest, GitSourceId, SourceId};
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
        SourceId::Git(self.source.clone())
            .validate()
            .map_err(|error| CacheError::InvalidObject(error.to_string()))?;
        self.tree.validate()?;
        if self.source.tree_digest != self.tree.digest {
            return Err(CacheError::InvalidObject(
                "Git source tree digest does not match its tree manifest".to_string(),
            ));
        }
        if self
            .blobs
            .windows(2)
            .any(|pair| pair[0].digest >= pair[1].digest)
        {
            return Err(CacheError::InvalidObject(
                "Git snapshot blobs must be strictly digest-sorted".to_string(),
            ));
        }
        let mut available = BTreeMap::new();
        for blob in &self.blobs {
            blob.validate()?;
            available.insert(blob.digest.clone(), blob.bytes.len() as u64);
        }
        let referenced: BTreeSet<_> = self.tree.referenced_blobs();
        let supplied: BTreeSet<_> = available.keys().cloned().collect();
        if supplied != referenced {
            return Err(CacheError::InvalidObject(
                "Git snapshot blob closure does not exactly match the tree".to_string(),
            ));
        }
        for entry in &self.tree.entries {
            if let Some(digest) = &entry.digest {
                if available.get(digest) != Some(&entry.byte_len) {
                    return Err(CacheError::InvalidObject(format!(
                        "Git snapshot file `{}` size differs from its blob",
                        entry.path
                    )));
                }
            }
        }
        Ok(())
    }
}
