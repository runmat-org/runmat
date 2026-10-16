//! Explicit operator-selected native build tree bytes. This is a canonical
//! physical inventory, not a toolchain qualification or link permission.

use std::collections::BTreeMap;

use runmat_execution::Digest;
use runmat_execution_artifact::target_pack::ComponentId;
use runmat_package::{ContentDigest, NormalizedRelativePath};
use runmat_package_cache::{TreeEntry, TreeManifest};

use crate::tree::{NativeBuildTree, NativeStorageGrant, NativeTreeBlobRef};
use crate::{AotError, AotResult};

pub enum HostedTreeEntry {
    File {
        path: NormalizedRelativePath,
        bytes: Vec<u8>,
        executable: bool,
    },
    Directory {
        path: NormalizedRelativePath,
    },
    Symlink {
        path: NormalizedRelativePath,
        target: NormalizedRelativePath,
    },
}

pub struct ProducedHostedTree {
    pub metadata: Vec<u8>,
    pub digest: Digest,
    /// One raw blob per distinct nonempty file digest. The labels are content
    /// identities, not source paths or authority to run any of these bytes.
    pub blobs: BTreeMap<ComponentId, Vec<u8>>,
}

pub fn build_hosted_tree(
    entries: Vec<HostedTreeEntry>,
    grant: NativeStorageGrant,
) -> AotResult<ProducedHostedTree> {
    let mut tree_entries = Vec::with_capacity(entries.len());
    let mut content = BTreeMap::<ContentDigest, Vec<u8>>::new();
    for entry in entries {
        match entry {
            HostedTreeEntry::File {
                path,
                bytes,
                executable,
            } => {
                grant.file(bytes.len() as u64)?;
                let digest = ContentDigest::sha256(&bytes);
                tree_entries.push(TreeEntry::file(
                    path,
                    digest.clone(),
                    bytes.len() as u64,
                    executable,
                ));
                if !bytes.is_empty() {
                    content.entry(digest).or_insert(bytes);
                }
            }
            HostedTreeEntry::Directory { path } => {
                tree_entries.push(TreeEntry::directory(path));
            }
            HostedTreeEntry::Symlink { path, target } => {
                tree_entries.push(TreeEntry::symlink(path, target));
            }
        }
    }
    let manifest = TreeManifest::new(tree_entries).map_err(invalid)?;
    grant.admit(&manifest)?;
    let mut blobs = BTreeMap::new();
    let mut references = Vec::with_capacity(content.len());
    for (digest, bytes) in content {
        let mut label = String::from("blob-");
        for byte in digest.bytes() {
            use std::fmt::Write;
            write!(&mut label, "{byte:02x}").expect("write hexadecimal label");
        }
        let component = ComponentId::new(label).map_err(invalid)?;
        references.push(NativeTreeBlobRef {
            digest,
            component: component.clone(),
        });
        blobs.insert(component, bytes);
    }
    let metadata = NativeBuildTree::new(manifest, references).canonical_bytes()?;
    grant.metadata(metadata.len() as u64)?;
    Ok(ProducedHostedTree {
        digest: Digest::sha256(&metadata),
        metadata,
        blobs,
    })
}

fn invalid(error: impl std::fmt::Display) -> AotError {
    AotError::contract("aot.target_pack.hosted_tree", error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_tree_deduplicates_exact_bytes_without_issuing_link_authority() {
        let grant = NativeStorageGrant {
            archive: runmat_package_cache::ArchiveLimits::default(),
            metadata_bytes: 1024 * 1024,
        };
        let produced = build_hosted_tree(
            vec![
                HostedTreeEntry::File {
                    path: NormalizedRelativePath::new("bin/a").unwrap(),
                    bytes: b"exact inventory bytes".to_vec(),
                    executable: true,
                },
                HostedTreeEntry::File {
                    path: NormalizedRelativePath::new("bin/b").unwrap(),
                    bytes: b"exact inventory bytes".to_vec(),
                    executable: true,
                },
            ],
            grant,
        )
        .unwrap();
        assert_eq!(produced.blobs.len(), 1);
        assert_eq!(produced.digest, Digest::sha256(&produced.metadata));
        let decoded = NativeBuildTree::decode(&produced.metadata, produced.digest, &grant).unwrap();
        assert_eq!(decoded.tree.file_count, 2);
        assert_eq!(decoded.blobs.len(), 1);
    }
}
