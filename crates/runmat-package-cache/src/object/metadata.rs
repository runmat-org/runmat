use super::{BlobMetadata, SourceIndexMetadata, TreeManifest};
use crate::CacheError;
use runmat_package::ContentDigest;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CacheObjectKind {
    Blob,
    Tree,
    SourceIndex,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum CacheObject {
    Blob(BlobMetadata),
    Tree(TreeManifest),
    SourceIndex(SourceIndexMetadata),
}

impl CacheObject {
    pub fn kind(&self) -> CacheObjectKind {
        match self {
            Self::Blob(_) => CacheObjectKind::Blob,
            Self::Tree(_) => CacheObjectKind::Tree,
            Self::SourceIndex(_) => CacheObjectKind::SourceIndex,
        }
    }

    pub fn digest(&self) -> &ContentDigest {
        match self {
            Self::Blob(blob) => &blob.digest,
            Self::Tree(tree) => &tree.digest,
            Self::SourceIndex(index) => &index.digest,
        }
    }

    pub fn logical_byte_len(&self) -> u64 {
        match self {
            Self::Blob(blob) => blob.byte_len,
            Self::Tree(tree) => tree.total_bytes,
            Self::SourceIndex(index) => index.byte_len,
        }
    }

    pub fn stored_payload_bytes(&self) -> u64 {
        match self {
            Self::Blob(blob) => blob.byte_len,
            Self::Tree(_) => 0,
            Self::SourceIndex(index) => index.byte_len,
        }
    }

    pub fn references(&self) -> BTreeSet<ContentDigest> {
        match self {
            Self::Blob(_) => BTreeSet::new(),
            Self::Tree(tree) => tree.referenced_blobs(),
            Self::SourceIndex(index) => [index.tree_digest.clone()].into_iter().collect(),
        }
    }

    pub fn validate(&self) -> Result<(), CacheError> {
        match self {
            Self::Blob(_) => Ok(()),
            Self::Tree(tree) => tree.validate(),
            Self::SourceIndex(index) if index.schema_version > 0 => Ok(()),
            Self::SourceIndex(_) => Err(CacheError::InvalidObject(
                "source index schema version must be greater than zero".to_string(),
            )),
        }
    }
}
