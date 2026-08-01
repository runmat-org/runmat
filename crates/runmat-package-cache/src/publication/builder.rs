use std::collections::BTreeSet;

use super::{
    PublicationEntry, PublicationEntryContent, PublicationPolicy, ReleaseArtifactBundle,
    ReleaseInventory, ReleaseInventoryEntry,
};
use crate::{
    CacheError, RegistryArtifactInventory, TreeEntry, TreeEntryKind, TreeInventoryEntry,
    TreeManifest, REGISTRY_ARTIFACT_SCHEMA_VERSION,
};
use runmat_package::ContentDigest;

pub struct ReleaseArtifactBuilder;

impl ReleaseArtifactBuilder {
    pub fn build(
        entries: Vec<PublicationEntry>,
        policy: &PublicationPolicy,
    ) -> Result<ReleaseArtifactBundle, CacheError> {
        let mut selected = entries
            .into_iter()
            .filter_map(|entry| match policy.accepts(&entry) {
                Ok(true) => Some(Ok(entry)),
                Ok(false) => None,
                Err(error) => Some(Err(error)),
            })
            .collect::<Result<Vec<_>, _>>()?;
        selected.sort_by(|left, right| left.path.cmp(&right.path));
        if selected.is_empty() {
            return Err(CacheError::InvalidObject(
                "publication contains no selected entries".to_string(),
            ));
        }
        if selected.windows(2).any(|pair| pair[0].path == pair[1].path) {
            return Err(CacheError::InvalidObject(
                "publication contains duplicate paths".to_string(),
            ));
        }

        let selected_paths = selected
            .iter()
            .map(|entry| entry.path.clone())
            .collect::<BTreeSet<_>>();
        let mut wire_entries = Vec::with_capacity(selected.len());
        let mut tree_entries = Vec::with_capacity(selected.len());
        let mut inventory_entries = Vec::with_capacity(selected.len());
        for entry in selected {
            let (wire, tree, inventory) = match entry.content {
                PublicationEntryContent::File(bytes) => {
                    let digest = ContentDigest::sha256(&bytes);
                    let byte_len = bytes.len() as u64;
                    (
                        TreeInventoryEntry::file(entry.path.as_str(), bytes, entry.executable),
                        TreeEntry::file(
                            entry.path.clone(),
                            digest.clone(),
                            byte_len,
                            entry.executable,
                        ),
                        ReleaseInventoryEntry {
                            path: entry.path,
                            role: entry.role,
                            kind: TreeEntryKind::File,
                            digest: Some(digest),
                            byte_len,
                            executable: entry.executable,
                            link_target: None,
                        },
                    )
                }
                PublicationEntryContent::Directory => (
                    TreeInventoryEntry::directory(entry.path.as_str()),
                    TreeEntry::directory(entry.path.clone()),
                    ReleaseInventoryEntry {
                        path: entry.path,
                        role: entry.role,
                        kind: TreeEntryKind::Directory,
                        digest: None,
                        byte_len: 0,
                        executable: false,
                        link_target: None,
                    },
                ),
                PublicationEntryContent::Symlink(target) => {
                    if !selected_paths.contains(&target)
                        && !selected_paths
                            .iter()
                            .any(|path| path.as_str().starts_with(&format!("{target}/")))
                    {
                        return Err(CacheError::InvalidObject(format!(
                            "publication symlink `{}` points outside selected content",
                            entry.path
                        )));
                    }
                    (
                        TreeInventoryEntry::symlink(
                            entry.path.as_str(),
                            archive_link_target(&entry.path, &target),
                        ),
                        TreeEntry::symlink(entry.path.clone(), target.clone()),
                        ReleaseInventoryEntry {
                            path: entry.path,
                            role: entry.role,
                            kind: TreeEntryKind::Symlink,
                            digest: None,
                            byte_len: 0,
                            executable: false,
                            link_target: Some(target),
                        },
                    )
                }
            };
            wire_entries.push(wire);
            tree_entries.push(tree);
            inventory_entries.push(inventory);
        }
        let tree = TreeManifest::new(tree_entries)?;
        let inventory = ReleaseInventory::new(inventory_entries)?;
        let artifact = RegistryArtifactInventory {
            schema_version: REGISTRY_ARTIFACT_SCHEMA_VERSION,
            entries: wire_entries,
        };
        let artifact_bytes = artifact.canonical_bytes()?;
        Ok(ReleaseArtifactBundle {
            artifact_digest: ContentDigest::sha256(&artifact_bytes),
            artifact_bytes,
            tree_digest: tree.digest,
            inventory,
        })
    }
}

fn archive_link_target(
    link: &runmat_package::NormalizedRelativePath,
    target: &runmat_package::NormalizedRelativePath,
) -> String {
    let link_parent = link
        .as_str()
        .rsplit_once('/')
        .map(|(parent, _)| parent.split('/').collect::<Vec<_>>())
        .unwrap_or_default();
    let target_components = target.as_str().split('/').collect::<Vec<_>>();
    let common = link_parent
        .iter()
        .zip(&target_components)
        .take_while(|(left, right)| left == right)
        .count();
    std::iter::repeat_n("..", link_parent.len() - common)
        .chain(target_components[common..].iter().copied())
        .collect::<Vec<_>>()
        .join("/")
}
