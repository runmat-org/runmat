use runmat_package::ContentDigest;
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CacheNamespace {
    Blob,
    Tree,
    SourceIndex,
    State,
    Staging,
}

impl fmt::Display for CacheNamespace {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Blob => "blob",
            Self::Tree => "tree",
            Self::SourceIndex => "source-index",
            Self::State => "state",
            Self::Staging => "staging",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StorageKey {
    pub namespace: CacheNamespace,
    pub digest: ContentDigest,
}

impl StorageKey {
    pub fn new(namespace: CacheNamespace, digest: ContentDigest) -> Self {
        Self { namespace, digest }
    }

    pub fn portable_key(&self) -> String {
        format!("{}/{}", self.namespace, self.digest)
    }
}
