//! Build-input transport only. Path, collision, ancestor and link safety are
//! delegated to the existing package archive owner, not reimplemented here.
use std::path::Path;

use runmat_execution::Digest;
use runmat_package_cache::{
    validate_archive, ArchiveEntryHeader, ArchiveEntryKind, ArchiveLimits, TreeInventoryEntry,
    TreeInventoryEntryKind, ValidatedArchive,
};

use crate::{AotError, AotResult};

const BUILD_TREE_SCHEMA_VERSION: u16 = 1;
const MAX_TREE_BYTES: usize = 512 * 1024 * 1024;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeBuildTree {
    pub schema_version: u16,
    pub entries: Vec<TreeInventoryEntry>,
}

impl NativeBuildTree {
    pub fn new(entries: Vec<TreeInventoryEntry>) -> Self {
        Self {
            schema_version: BUILD_TREE_SCHEMA_VERSION,
            entries,
        }
    }

    pub fn canonical_bytes(&self) -> AotResult<Vec<u8>> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).map_err(invalid)?;
        if bytes.len() > MAX_TREE_BYTES {
            return Err(invalid("build tree exceeds its byte bound"));
        }
        Ok(bytes)
    }

    pub(super) fn decode(bytes: &[u8], expected: Digest) -> AotResult<Self> {
        if bytes.len() > MAX_TREE_BYTES || Digest::sha256(bytes) != expected {
            return Err(invalid(
                "build tree differs from its exact admitted identity or bound",
            ));
        }
        #[derive(serde::Deserialize)]
        struct Admission {
            schema_version: u16,
        }
        let header: Admission = serde_json::from_slice(bytes).map_err(invalid)?;
        if header.schema_version != BUILD_TREE_SCHEMA_VERSION {
            return Err(invalid("unsupported build tree schema"));
        }
        let tree: Self = serde_json::from_slice(bytes).map_err(invalid)?;
        if tree.canonical_bytes()? != bytes {
            return Err(invalid("build tree encoding is not canonical"));
        }
        Ok(tree)
    }

    fn validate(&self) -> AotResult<ValidatedArchive> {
        if self.schema_version != BUILD_TREE_SCHEMA_VERSION
            || self
                .entries
                .windows(2)
                .any(|pair| pair[0].path >= pair[1].path)
            || self.entries.iter().any(|entry| match entry.kind {
                TreeInventoryEntryKind::File => entry.link_target.is_some(),
                TreeInventoryEntryKind::Directory => {
                    !entry.bytes.is_empty() || entry.executable || entry.link_target.is_some()
                }
                TreeInventoryEntryKind::Symlink => {
                    !entry.bytes.is_empty() || entry.executable || entry.link_target.is_none()
                }
            })
        {
            return Err(invalid("build tree inventory is not canonical"));
        }
        let headers = self.entries.iter().map(|entry| ArchiveEntryHeader {
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
        });
        let validated = validate_archive(
            headers,
            ArchiveLimits {
                max_entries: 65_536,
                max_files: 65_536,
                max_expanded_bytes: MAX_TREE_BYTES as u64,
                max_file_bytes: MAX_TREE_BYTES as u64,
                ..ArchiveLimits::default()
            },
        )
        .map_err(invalid)?;
        if !validated
            .entries
            .iter()
            .zip(&self.entries)
            .all(|(safe, raw)| safe.path.as_str() == raw.path)
        {
            return Err(invalid("build tree paths must already be normalized"));
        }
        Ok(validated)
    }

    pub(super) fn stage(&self, root: &Path) -> AotResult<()> {
        let validated = self.validate()?;
        std::fs::create_dir(root)
            .map_err(|error| AotError::io("create private build tree", root, error))?;
        for (entry, safe) in self.entries.iter().zip(&validated.entries) {
            let path = root.join(safe.path.as_str());
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|error| AotError::io("create build tree parent", parent, error))?;
            }
            match entry.kind {
                TreeInventoryEntryKind::Directory => std::fs::create_dir_all(&path)
                    .map_err(|error| AotError::io("create build tree directory", &path, error))?,
                TreeInventoryEntryKind::File => {
                    super::inputs::write_new(&path, &entry.bytes)?;
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::PermissionsExt;
                        std::fs::set_permissions(
                            &path,
                            std::fs::Permissions::from_mode(if entry.executable {
                                0o555
                            } else {
                                0o444
                            }),
                        )
                        .map_err(|error| AotError::io("seal build tree file", &path, error))?;
                    }
                }
                TreeInventoryEntryKind::Symlink => {}
            }
        }
        // Links are last; their normalized destinations are checked regular
        // entries by the shared validator and cannot redirect a later write.
        for safe in validated
            .entries
            .iter()
            .filter(|entry| entry.kind == ArchiveEntryKind::Symlink)
        {
            let path = root.join(safe.path.as_str());
            let target = root.join(safe.link_target.as_ref().expect("validated link").as_str());
            #[cfg(unix)]
            std::os::unix::fs::symlink(&target, &path)
                .map_err(|error| AotError::io("stage build tree symlink", &path, error))?;
            #[cfg(windows)]
            {
                let result = if target.is_dir() {
                    std::os::windows::fs::symlink_dir(&target, &path)
                } else {
                    std::os::windows::fs::symlink_file(&target, &path)
                };
                result.map_err(|error| AotError::io("stage build tree symlink", &path, error))?;
            }
        }
        self.verify_staged(root)
    }

    pub(super) fn verify_staged(&self, root: &Path) -> AotResult<()> {
        let validated = self.validate()?;
        for (entry, safe) in self.entries.iter().zip(&validated.entries) {
            let path = root.join(safe.path.as_str());
            let metadata = std::fs::symlink_metadata(&path)
                .map_err(|error| AotError::io("inspect staged build input", &path, error))?;
            #[cfg(unix)]
            if entry.kind == TreeInventoryEntryKind::File {
                use std::os::unix::fs::PermissionsExt;
                if entry.executable != (metadata.permissions().mode() & 0o100 != 0) {
                    return Err(invalid(
                        "staged build-host file mode differs from admitted tree",
                    ));
                }
            }
            let matches = match entry.kind {
                TreeInventoryEntryKind::File => {
                    metadata.is_file()
                        && std::fs::read(&path).map_err(|error| {
                            AotError::io("verify staged build input", &path, error)
                        })? == entry.bytes
                }
                TreeInventoryEntryKind::Directory => metadata.is_dir(),
                TreeInventoryEntryKind::Symlink => {
                    metadata.is_symlink()
                        && std::fs::read_link(&path).map_err(|error| {
                            AotError::io("verify staged build link", &path, error)
                        })? == root
                            .join(safe.link_target.as_ref().expect("validated link").as_str())
                }
            };
            if !matches {
                return Err(invalid(
                    "staged build input differs from immutable admitted tree",
                ));
            }
        }
        Ok(())
    }

    pub(super) fn executable(&self, path: &str) -> Option<&[u8]> {
        self.entries
            .iter()
            .find(|entry| {
                entry.path == path && entry.kind == TreeInventoryEntryKind::File && entry.executable
            })
            .map(|entry| entry.bytes.as_slice())
    }

    pub(super) fn file(&self, path: &str) -> Option<&[u8]> {
        self.entries
            .iter()
            .find(|entry| entry.path == path && entry.kind == TreeInventoryEntryKind::File)
            .map(|entry| entry.bytes.as_slice())
    }
}

fn invalid(error: impl std::fmt::Display) -> AotError {
    AotError::contract("aot.link.build_tree", error.to_string())
}

#[cfg(test)]
#[path = "tests/build_tree.rs"]
mod tests;
