//! Explicit operator-selected native build tree bytes. This is a canonical
//! physical inventory, not a toolchain qualification or link permission.

use std::collections::BTreeMap;
use std::io::Read;

use runmat_execution::Digest;
use runmat_execution_artifact::target_pack::{ComponentId, MAX_COMPONENT_BYTES};
use runmat_package::{ContentDigest, NormalizedRelativePath};
use runmat_package_cache::{TreeEntry, TreeManifest};

use crate::tree::{NativeBuildTree, NativeStorageGrant, NativeTreeBlobRef};
use crate::{AotError, AotResult};

/// Reads one operator-selected tree component through the native physical-file
/// owner. This rejects symlinks and path replacement around the opened inode.
pub fn read_physical_tree_component(path: &std::path::Path, maximum: u64) -> AotResult<Vec<u8>> {
    crate::tree::physical::read_regular(path, maximum)
}

/// Aggregate physical materialization admitted by one caller storage grant.
/// Metadata has its own bound; these counters cover tree occurrences and
/// explicit native archives that the selected pack retains together.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ContentAddressedMaterialization {
    entries: u64,
    files: u64,
    bytes: u64,
}

impl ContentAddressedMaterialization {
    fn admit(
        &mut self,
        grant: &NativeStorageGrant,
        entries: u64,
        files: u64,
        bytes: u64,
    ) -> AotResult<()> {
        let next = Self {
            entries: self
                .entries
                .checked_add(entries)
                .ok_or_else(|| invalid("aggregate physical entry count overflow"))?,
            files: self
                .files
                .checked_add(files)
                .ok_or_else(|| invalid("aggregate physical file count overflow"))?,
            bytes: self
                .bytes
                .checked_add(bytes)
                .ok_or_else(|| invalid("aggregate physical byte count overflow"))?,
        };
        grant.admit_total(next.entries, next.files, next.bytes)?;
        *self = next;
        Ok(())
    }
}

/// Decode and admit canonical metadata before opening any referenced blob,
/// then retain exactly the digest/extent inventory through the physical owner.
pub fn load_physical_hosted_tree(
    metadata_path: &std::path::Path,
    blob_root: &std::path::Path,
    grant: NativeStorageGrant,
    aggregate: &mut ContentAddressedMaterialization,
) -> AotResult<ProducedHostedTree> {
    let metadata =
        read_physical_tree_component(metadata_path, grant.metadata_bytes.min(MAX_COMPONENT_BYTES))?;
    let digest = Digest::sha256(&metadata);
    let inventory = NativeBuildTree::decode(&metadata, digest, &grant)?;
    let physical_entries = grant.admit(&inventory.tree)?;
    let mut files = 0_u64;
    let mut bytes = 0_u64;
    let mut extents = BTreeMap::<ContentDigest, u64>::new();
    for entry in &inventory.tree.entries {
        if entry.kind != runmat_package_cache::TreeEntryKind::File {
            continue;
        }
        files = files
            .checked_add(1)
            .ok_or_else(|| invalid("tree file count overflow"))?;
        bytes = bytes
            .checked_add(entry.byte_len)
            .ok_or_else(|| invalid("tree byte count overflow"))?;
        if entry.byte_len != 0 {
            let digest = entry.digest.clone().expect("validated file digest");
            if extents
                .insert(digest, entry.byte_len)
                .is_some_and(|prior| prior != entry.byte_len)
            {
                return Err(invalid(
                    "one tree content identity declares inconsistent physical extents",
                ));
            }
        }
    }
    aggregate.admit(&grant, physical_entries, files, bytes)?;

    let mut blobs = BTreeMap::new();
    for reference in &inventory.blobs {
        let expected = *extents
            .get(&reference.digest)
            .ok_or_else(|| invalid("tree blob lacks an admitted physical extent"))?;
        let path = blob_root.join(reference.component.as_str());
        let blob = read_physical_tree_component(&path, expected.min(MAX_COMPONENT_BYTES))?;
        if blob.len() as u64 != expected {
            return Err(invalid(
                "tree blob differs from its admitted physical extent",
            ));
        }
        if blobs.insert(reference.component.clone(), blob).is_some() {
            return Err(invalid("tree repeats a blob component identity"));
        }
    }
    ProducedHostedTree::from_canonical_parts(metadata, blobs, grant)
}

/// Read one explicit native implementation archive only after adding its
/// opened extent to the same aggregate materialization permission.
pub fn read_physical_native_archive(
    path: &std::path::Path,
    grant: &NativeStorageGrant,
    aggregate: &mut ContentAddressedMaterialization,
) -> AotResult<Vec<u8>> {
    let (file, length) = crate::tree::physical::open_bounded_regular(
        path,
        grant.archive.max_file_bytes.min(MAX_COMPONENT_BYTES),
    )?;
    aggregate.admit(grant, 1, 1, length)?;
    let mut bytes = Vec::new();
    file.take(length.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| crate::AotError::io("read admitted native archive", path, error))?;
    if bytes.len() as u64 != length {
        return Err(invalid(
            "native implementation archive changed after admission",
        ));
    }
    Ok(bytes)
}

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
    grant: NativeStorageGrant,
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
        grant,
    })
}

impl ProducedHostedTree {
    /// Reopens an operator-produced tree without turning its metadata into a
    /// toolchain permission. The selected pack producer still decides whether
    /// this exact inventory is admitted as a linker or sysroot component.
    pub fn from_canonical_parts(
        metadata: Vec<u8>,
        blobs: BTreeMap<ComponentId, Vec<u8>>,
        grant: NativeStorageGrant,
    ) -> AotResult<Self> {
        let digest = Digest::sha256(&metadata);
        let decoded = NativeBuildTree::decode(&metadata, digest, &grant)?;
        if decoded.blobs.len() != blobs.len()
            || decoded.blobs.iter().any(|reference| {
                blobs
                    .get(&reference.component)
                    .is_none_or(|bytes| ContentDigest::sha256(bytes) != reference.digest)
            })
        {
            return Err(invalid(
                "tree blobs do not exactly match the canonical metadata inventory",
            ));
        }
        Ok(Self {
            metadata,
            digest,
            blobs,
            grant,
        })
    }

    pub(crate) fn validated_tree(&self) -> AotResult<NativeBuildTree> {
        let decoded = NativeBuildTree::decode(&self.metadata, self.digest, &self.grant)?;
        if decoded.blobs.len() != self.blobs.len()
            || decoded.blobs.iter().any(|reference| {
                self.blobs
                    .get(&reference.component)
                    .is_none_or(|bytes| ContentDigest::sha256(bytes) != reference.digest)
            })
        {
            return Err(invalid(
                "tree blobs changed after their canonical inventory was produced",
            ));
        }
        Ok(decoded)
    }
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

    #[test]
    fn physical_inputs_share_one_grant_before_referenced_blobs_are_opened() {
        let grant = NativeStorageGrant {
            archive: runmat_package_cache::ArchiveLimits {
                max_expanded_bytes: 5,
                ..Default::default()
            },
            metadata_bytes: 1024 * 1024,
        };
        let tree = build_hosted_tree(
            vec![HostedTreeEntry::File {
                path: NormalizedRelativePath::new("bin/tool").unwrap(),
                bytes: b"abc".to_vec(),
                executable: true,
            }],
            grant,
        )
        .unwrap();
        let directory = tempfile::tempdir().unwrap();
        let metadata = directory.path().join("tree.json");
        std::fs::write(&metadata, &tree.metadata).unwrap();
        for (component, bytes) in &tree.blobs {
            std::fs::write(directory.path().join(component.as_str()), bytes).unwrap();
        }
        let mut aggregate = ContentAddressedMaterialization::default();
        let loaded =
            load_physical_hosted_tree(&metadata, directory.path(), grant, &mut aggregate).unwrap();
        assert_eq!(loaded.metadata, tree.metadata);
        assert_eq!(loaded.blobs, tree.blobs);
        let admitted = aggregate;

        // A second individually valid tree must fail the combined grant,
        // before trying to read its deliberately absent physical blobs.
        assert!(matches!(
            load_physical_hosted_tree(
                &metadata,
                &directory.path().join("absent-blobs"),
                grant,
                &mut aggregate,
            ),
            Err(AotError::ResourceUnavailable { .. })
        ));
        assert_eq!(aggregate, admitted);

        let archive = directory.path().join("implementation.a");
        std::fs::write(&archive, b"de").unwrap();
        assert_eq!(
            read_physical_native_archive(&archive, &grant, &mut aggregate).unwrap(),
            b"de"
        );
        let exhausted = aggregate;
        assert!(matches!(
            read_physical_native_archive(&archive, &grant, &mut aggregate),
            Err(AotError::ResourceUnavailable { .. })
        ));
        assert_eq!(aggregate, exhausted);
    }
}
