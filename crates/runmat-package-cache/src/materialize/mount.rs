use runmat_package::{ContentDigest, NormalizedRelativePath};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MountDescriptor {
    pub tree_digest: ContentDigest,
    pub logical_root: NormalizedRelativePath,
    pub read_only: bool,
}

impl MountDescriptor {
    pub fn immutable(tree_digest: ContentDigest, logical_root: NormalizedRelativePath) -> Self {
        Self {
            tree_digest,
            logical_root,
            read_only: true,
        }
    }
}
