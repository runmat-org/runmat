use runmat_package::ContentDigest;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceIndexMetadata {
    pub digest: ContentDigest,
    pub tree_digest: ContentDigest,
    pub schema_version: u32,
    pub byte_len: u64,
}
