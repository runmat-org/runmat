use crate::{AotError, AotResult};
use runmat_package_cache::{
    relative_link_target, validate_archive, ArchiveEntryHeader, ArchiveEntryKind, ArchiveError,
    ArchiveLimits, TreeEntryKind, TreeManifest,
};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Explicit trusted caller storage permission, never deserialized from a tree
/// or inferred from its claimed lengths. This covers materialization only,
/// not unmeasured linker execution/scratch or runtime allocation authority.
/// Tree entries include unique implicit ancestors; enclosing callers charge
/// their wrapper directories, manifest and bounded publication overhead.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeStorageGrant {
    pub archive: ArchiveLimits,
    /// Maximum bytes of one tree/installation metadata record; this is not a
    /// finite-memory certificate for arbitrary decoding or linker execution.
    pub metadata_bytes: u64,
}
impl NativeStorageGrant {
    pub(crate) fn metadata(&self, bytes: u64) -> AotResult<()> {
        self.archive.validate().map_err(unavailable)?;
        if bytes > self.metadata_bytes {
            return Err(unavailable("metadata storage permission exhausted"));
        }
        Ok(())
    }
    pub(crate) fn admit(&self, tree: &TreeManifest) -> AotResult<u64> {
        self.archive.validate().map_err(unavailable)?;
        if tree.entries.iter().any(|entry| {
            entry.path.as_str().len() > self.archive.max_path_bytes
                || entry
                    .path
                    .as_str()
                    .split('/')
                    .any(|component| component.len() > self.archive.max_component_bytes)
                || entry.link_target.as_ref().is_some_and(|target| {
                    target.as_str().len() > self.archive.max_path_bytes
                        || target
                            .as_str()
                            .split('/')
                            .any(|component| component.len() > self.archive.max_component_bytes)
                })
        }) {
            return Err(unavailable(
                "tree path metadata exceeds caller storage permission",
            ));
        }
        let validated = validate_archive(
            tree.entries.iter().map(|entry| ArchiveEntryHeader {
                path: entry.path.as_str().to_owned(),
                kind: match entry.kind {
                    TreeEntryKind::File => ArchiveEntryKind::File,
                    TreeEntryKind::Directory => ArchiveEntryKind::Directory,
                    TreeEntryKind::Symlink => ArchiveEntryKind::Symlink,
                },
                expanded_bytes: entry.byte_len,
                compressed_bytes: entry.byte_len,
                executable: entry.executable,
                link_target: entry.link_target.as_ref().map(|target| {
                    relative_link_target(&entry.path, target)
                        .to_string_lossy()
                        .into_owned()
                }),
            }),
            self.archive,
        )
        .map_err(|error| match error {
            ArchiveError::Limit { .. } => unavailable(error),
            _ => AotError::contract("aot.tree.storage", error.to_string()),
        })?;
        if !validated
            .entries
            .iter()
            .zip(&tree.entries)
            .all(|(safe, entry)| safe.path == entry.path && safe.link_target == entry.link_target)
        {
            return Err(AotError::contract(
                "aot.tree.storage",
                "tree paths and link targets must already have canonical spelling",
            ));
        }
        tree.validate()
            .map_err(|error| AotError::contract("aot.tree.manifest", error.to_string()))?;
        Ok(physical_inventory(tree, self.archive.max_entries)?.len() as u64)
    }
    pub(crate) fn admit_total(&self, entries: u64, files: u64, bytes: u64) -> AotResult<()> {
        if entries > self.archive.max_entries
            || files > self.archive.max_files
            || bytes > self.archive.max_expanded_bytes
        {
            return Err(unavailable(
                "combined physical occurrence demand exceeds storage permission",
            ));
        }
        Ok(())
    }
    pub(crate) fn file(&self, bytes: u64) -> AotResult<()> {
        if bytes > self.archive.max_file_bytes {
            return Err(unavailable(
                "physical file exceeds caller storage permission",
            ));
        }
        Ok(())
    }
}

/// One bounded inventory algorithm for admission AND physical verification.
/// The containing tree root is accounted by its caller, not a relative member.
pub(super) fn physical_inventory(
    tree: &TreeManifest,
    maximum: u64,
) -> AotResult<BTreeMap<PathBuf, TreeEntryKind>> {
    fn insert(
        entries: &mut BTreeMap<PathBuf, TreeEntryKind>,
        path: PathBuf,
        kind: TreeEntryKind,
        maximum: u64,
    ) -> AotResult<()> {
        if let Some(previous) = entries.get(&path) {
            if *previous != kind {
                return Err(AotError::contract(
                    "aot.tree.storage",
                    "physical entry kinds conflict",
                ));
            }
        } else {
            if entries.len() as u64 >= maximum {
                return Err(unavailable(
                    "implicit/explicit physical entry demand exceeds storage permission",
                ));
            }
            entries.insert(path, kind);
        }
        Ok(())
    }
    let mut entries = BTreeMap::new();
    for entry in &tree.entries {
        let path = PathBuf::from(entry.path.as_str());
        insert(&mut entries, path.clone(), entry.kind, maximum)?;
        let mut parent = path.parent();
        while let Some(path) = parent.filter(|path| !path.as_os_str().is_empty()) {
            insert(
                &mut entries,
                path.to_path_buf(),
                TreeEntryKind::Directory,
                maximum,
            )?;
            parent = path.parent();
        }
    }
    Ok(entries)
}

pub(crate) fn unavailable(error: impl std::fmt::Display) -> AotError {
    AotError::ResourceUnavailable {
        message: error.to_string(),
    }
}
