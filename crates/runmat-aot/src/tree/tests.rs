use super::codec::invalid;
use super::*;
use crate::AotResult;
use runmat_execution::Digest;
use runmat_execution_artifact::target_pack::{ComponentId, MAX_COMPONENT_BYTES};
use runmat_package::ContentDigest;
use runmat_package_cache::{
    validate_archive, ArchiveEntryHeader, ArchiveEntryKind, ArchiveLimits, TreeEntry,
    TreeInventoryEntry, TreeInventoryEntryKind, TreeManifest,
};
use std::collections::BTreeMap;

pub(crate) fn grant() -> NativeStorageGrant {
    NativeStorageGrant {
        archive: ArchiveLimits::default(),
        metadata_bytes: MAX_COMPONENT_BYTES,
    }
}

pub(crate) struct Fixture {
    pub metadata: Vec<u8>,
    pub blobs: BTreeMap<ComponentId, Vec<u8>>,
}
impl Fixture {
    pub fn resolve(&self, storage: &NativeStorageGrant) -> AotResult<ResolvedNativeTree<'_>> {
        super::resolve::resolve_hosted(
            HostedNativeTree {
                metadata: &self.metadata,
                blobs: &self.blobs,
            },
            Digest::sha256(&self.metadata),
            storage,
        )
    }
}
pub(crate) fn fixture(entries: Vec<TreeInventoryEntry>) -> AotResult<Fixture> {
    let safe = validate_archive(
        entries.iter().map(|entry| ArchiveEntryHeader {
            path: entry.path.clone(),
            kind: match entry.kind {
                TreeInventoryEntryKind::File => ArchiveEntryKind::File,
                TreeInventoryEntryKind::Directory => ArchiveEntryKind::Directory,
                TreeInventoryEntryKind::Symlink => ArchiveEntryKind::Symlink,
            },
            expanded_bytes: entry.bytes.len() as u64,
            compressed_bytes: entry.bytes.len() as u64,
            executable: entry.executable,
            link_target: entry.link_target.clone(),
        }),
        ArchiveLimits::default(),
    )
    .map_err(invalid)?;
    let mut tree = Vec::new();
    let mut refs = BTreeMap::new();
    let mut blobs = BTreeMap::new();
    for entry in safe.entries {
        let raw = entries
            .iter()
            .find(|raw| raw.path == entry.path.as_str())
            .expect("normalized fixture");
        tree.push(match entry.kind {
            ArchiveEntryKind::File => {
                let digest = ContentDigest::sha256(&raw.bytes);
                if !raw.bytes.is_empty() && !refs.contains_key(&digest) {
                    let id = ComponentId::new(format!("blob-{}", refs.len())).map_err(invalid)?;
                    blobs.insert(id.clone(), raw.bytes.clone());
                    refs.insert(digest.clone(), id);
                }
                TreeEntry::file(entry.path, digest, raw.bytes.len() as u64, raw.executable)
            }
            ArchiveEntryKind::Directory => TreeEntry::directory(entry.path),
            ArchiveEntryKind::Symlink => {
                TreeEntry::symlink(entry.path, entry.link_target.expect("validated link"))
            }
            _ => unreachable!("fixture kinds are constrained above"),
        });
    }
    let tree = NativeBuildTree::new(
        TreeManifest::new(tree).map_err(invalid)?,
        refs.into_iter()
            .map(|(digest, component)| NativeTreeBlobRef { digest, component })
            .collect(),
    );
    Ok(Fixture {
        metadata: tree.canonical_bytes()?,
        blobs,
    })
}

#[test]
fn metadata_transport_keeps_raw_payload_outside_json_and_borrows_verified_blobs() {
    let data = (0..=255).cycle().take(256 * 1024).collect::<Vec<u8>>();
    let mut source = fixture(vec![TreeInventoryEntry::file(
        "bin/tool",
        data.clone(),
        true,
    )])
    .unwrap();
    assert!(source.metadata.len() < 2048);
    let storage = grant();
    let tree = source.resolve(&storage).unwrap();
    assert_eq!(
        storage.admit(&tree.manifest).unwrap(),
        2,
        "implicit bin directory is a physical entry"
    );
    let mut explicit = tree.manifest.entries.clone();
    explicit.push(TreeEntry::directory(
        runmat_package::NormalizedRelativePath::new("bin").unwrap(),
    ));
    assert_eq!(
        storage
            .admit(&TreeManifest::new(explicit).unwrap())
            .unwrap(),
        2,
        "explicit and implicit directory are counted once"
    );
    let one_entry = NativeStorageGrant {
        archive: ArchiveLimits {
            max_entries: 1,
            ..storage.archive
        },
        ..storage
    };
    assert!(matches!(
        source.resolve(&one_entry),
        Err(crate::AotError::ResourceUnavailable { .. })
    ));
    assert_eq!(
        tree.file("bin/tool").unwrap().as_ptr(),
        source.blobs.values().next().unwrap().as_ptr()
    );
    let lease = tempfile::tempdir().unwrap();
    let root = lease.path().join("tree");
    tree.stage(&root, &storage).unwrap();
    tree.verify_staged(&root).unwrap();
    assert_eq!(std::fs::read(root.join("bin/tool")).unwrap(), data);
    assert!(tree.stage(&root, &storage).is_err());
    drop(tree);
    assert!(matches!(
        source.resolve(&NativeStorageGrant {
            metadata_bytes: 1,
            ..storage
        }),
        Err(crate::AotError::ResourceUnavailable { .. })
    ));
    source.blobs.values_mut().next().unwrap().push(0);
    assert!(
        source.resolve(&storage).is_err(),
        "declared digest/extent cannot admit changed raw blobs"
    );
}

#[test]
fn storage_permission_counts_repeated_occurrences_beyond_old_half_gib_ceiling() {
    let digest = ContentDigest::sha256("declared content only, not verified blob evidence");
    let tree = TreeManifest::new(
        ["a", "b"]
            .into_iter()
            .map(|path| {
                TreeEntry::file(
                    runmat_package::NormalizedRelativePath::new(path).unwrap(),
                    digest.clone(),
                    MAX_COMPONENT_BYTES,
                    false,
                )
            })
            .collect(),
    )
    .unwrap();
    let storage = grant();
    storage.admit(&tree).unwrap();
    assert_eq!(tree.total_bytes, 2 * MAX_COMPONENT_BYTES);
    let small = NativeStorageGrant {
        archive: ArchiveLimits {
            max_expanded_bytes: MAX_COMPONENT_BYTES,
            ..storage.archive
        },
        ..storage
    };
    assert!(matches!(
        small.admit(&tree),
        Err(crate::AotError::ResourceUnavailable { .. })
    ));
    // This tests metadata demand only, not physical admission of missing data.
}

#[test]
fn old_tree_schema_is_rejected_before_changed_shape_decode() {
    let old = br#"{"schema_version":1,"entries":"old shape"}"#;
    assert!(NativeBuildTree::decode(old, Digest::sha256(old), &grant())
        .unwrap_err()
        .to_string()
        .contains("unsupported native tree schema"));
}

#[test]
fn build_tree_reuses_archive_path_collision_and_link_rejection() {
    for entries in [
        vec![TreeInventoryEntry::file("../escape", vec![1], false)],
        vec![
            TreeInventoryEntry::file("A", vec![1], false),
            TreeInventoryEntry::file("a", vec![1], false),
        ],
        vec![
            TreeInventoryEntry::file("a", vec![1], false),
            TreeInventoryEntry::file("a/b", vec![1], false),
        ],
        vec![TreeInventoryEntry::symlink("a", "../../outside")],
    ] {
        assert!(fixture(entries).is_err());
    }
}

#[cfg(unix)]
#[test]
fn complete_framework_links_remain_relative_after_generation_relocation() {
    let source = fixture(vec![
        TreeInventoryEntry::symlink("Foo.framework/Current", "Versions/A"),
        TreeInventoryEntry::directory("Foo.framework/Versions/A"),
        TreeInventoryEntry::file("Foo.framework/Versions/A/Foo", b"image".to_vec(), false),
        TreeInventoryEntry::file(
            "Foo.framework/Versions/A/Resources/data",
            b"resource".to_vec(),
            false,
        ),
    ])
    .unwrap();
    let storage = grant();
    let tree = source.resolve(&storage).unwrap();
    let lease = tempfile::tempdir().unwrap();
    let from = lease.path().join("staged");
    let to = lease.path().join("installed");
    tree.stage(&from, &storage).unwrap();
    std::fs::rename(&from, &to).unwrap();
    tree.verify_staged(&to).unwrap();
    assert_eq!(
        std::fs::read(to.join("Foo.framework/Current/Foo")).unwrap(),
        b"image"
    );
    assert_eq!(
        std::fs::read(to.join("Foo.framework/Current/Resources/data")).unwrap(),
        b"resource"
    );
}

#[cfg(unix)]
#[test]
fn staged_tree_mutated_bytes_fail_revalidation() {
    let source = fixture(vec![
        TreeInventoryEntry::symlink("alias", "data"),
        TreeInventoryEntry::file("data", b"exact".to_vec(), false),
    ])
    .unwrap();
    let storage = grant();
    let tree = source.resolve(&storage).unwrap();
    let lease = tempfile::tempdir().unwrap();
    let root = lease.path().join("tree");
    tree.stage(&root, &storage).unwrap();
    assert_eq!(std::fs::read(root.join("alias")).unwrap(), b"exact");
    use std::os::unix::fs::PermissionsExt;
    let data = root.join("data");
    std::fs::set_permissions(&data, std::fs::Permissions::from_mode(0o600)).unwrap();
    std::fs::write(data, b"changed").unwrap();
    assert!(tree.verify_staged(&root).is_err());
}
