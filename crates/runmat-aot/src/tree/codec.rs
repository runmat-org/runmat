//! Metadata-only tree transport. File bytes are raw authenticated components,
//! never embedded in JSON/base64. TreeManifest remains the inventory authority.
use super::{metadata, NativeStorageGrant};
use crate::{AotError, AotResult};
use runmat_execution::Digest;
use runmat_execution_artifact::target_pack::{ComponentId, MAX_COMPONENT_BYTES};
use runmat_package::ContentDigest;
use runmat_package_cache::{TreeEntryKind, TreeManifest};

const TREE_SCHEMA_VERSION: u16 = 2;

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTreeBlobRef {
    pub digest: ContentDigest,
    /// Raw content in the tree root's authenticated dependency closure. Labels
    /// identify bytes only, never an executable/service permission.
    pub component: ComponentId,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeBuildTree {
    pub schema_version: u16,
    pub tree: TreeManifest,
    /// Strict digest order; exactly the nonempty file-content set.
    pub blobs: Vec<NativeTreeBlobRef>,
}
impl NativeBuildTree {
    pub fn new(tree: TreeManifest, blobs: Vec<NativeTreeBlobRef>) -> Self {
        Self {
            schema_version: TREE_SCHEMA_VERSION,
            tree,
            blobs,
        }
    }
    pub fn canonical_bytes(&self) -> AotResult<Vec<u8>> {
        self.validate()?;
        metadata::canonical_json(self, MAX_COMPONENT_BYTES)
    }
    pub(crate) fn decode(
        bytes: &[u8],
        expected: Digest,
        grant: &NativeStorageGrant,
    ) -> AotResult<Self> {
        grant.metadata(bytes.len() as u64)?;
        if bytes.len() as u64 > MAX_COMPONENT_BYTES || Digest::sha256(bytes) != expected {
            return Err(invalid(
                "tree metadata differs from its admitted identity or canonical component bound",
            ));
        }
        #[derive(serde::Deserialize)]
        struct Header {
            schema_version: u16,
        }
        if serde_json::from_slice::<Header>(bytes)
            .map_err(invalid)?
            .schema_version
            != TREE_SCHEMA_VERSION
        {
            return Err(invalid(
                "unsupported native tree schema; expected metadata-only v2",
            ));
        }
        let tree: Self = serde_json::from_slice(bytes).map_err(invalid)?;
        grant.admit(&tree.tree)?;
        tree.validate()?;
        metadata::require_canonical(&tree, bytes)?;
        Ok(tree)
    }
    fn validate(&self) -> AotResult<()> {
        use std::collections::BTreeSet;
        if self.schema_version != TREE_SCHEMA_VERSION
            || self
                .blobs
                .windows(2)
                .any(|pair| pair[0].digest >= pair[1].digest)
        {
            return Err(invalid(
                "native tree schema or blob reference order is invalid",
            ));
        }
        self.tree.validate().map_err(invalid)?;
        let required = self
            .tree
            .entries
            .iter()
            .filter(|entry| entry.kind == TreeEntryKind::File && entry.byte_len != 0)
            .map(|entry| entry.digest.as_ref().expect("validated file"))
            .collect::<BTreeSet<_>>();
        if !required
            .iter()
            .copied()
            .eq(self.blobs.iter().map(|entry| &entry.digest))
            || self.tree.entries.iter().any(|entry| {
                entry.kind == TreeEntryKind::File
                    && (entry.byte_len > MAX_COMPONENT_BYTES
                        || (entry.byte_len == 0
                            && entry.digest.as_ref() != Some(&ContentDigest::sha256([]))))
            })
        {
            return Err(invalid(
                "tree blob references do not exactly cover canonical file content",
            ));
        }
        Ok(())
    }
}

pub(super) fn invalid(error: impl std::fmt::Display) -> AotError {
    AotError::contract("aot.tree.codec", error.to_string())
}
