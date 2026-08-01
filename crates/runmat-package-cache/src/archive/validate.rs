use super::collisions::reject_collisions;
use super::paths::{normalize_entry_path, normalize_link_target};
use super::{ArchiveEntryHeader, ArchiveEntryKind, ArchiveLimits};
use runmat_package::NormalizedRelativePath;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ArchiveError {
    #[error("archive limits are invalid: {0}")]
    InvalidLimits(String),
    #[error("archive path `{path}` is invalid: {reason}")]
    InvalidPath { path: String, reason: String },
    #[error("archive entries `{first}` and `{second}` collide on a supported host")]
    Collision { first: String, second: String },
    #[error("archive entry `{path}` uses unsupported type {kind:?}")]
    UnsupportedType {
        path: String,
        kind: ArchiveEntryKind,
    },
    #[error("archive entry `{path}` violates limits: {reason}")]
    Limit { path: String, reason: String },
    #[error("archive entry `{path}` has inconsistent metadata: {reason}")]
    Inconsistent { path: String, reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidatedArchiveEntry {
    pub path: NormalizedRelativePath,
    pub kind: ArchiveEntryKind,
    pub expanded_bytes: u64,
    pub compressed_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub link_target: Option<NormalizedRelativePath>,
    pub executable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidatedArchive {
    pub entries: Vec<ValidatedArchiveEntry>,
    pub file_count: u64,
    pub expanded_bytes: u64,
}

pub fn validate_archive(
    headers: impl IntoIterator<Item = ArchiveEntryHeader>,
    limits: ArchiveLimits,
) -> Result<ValidatedArchive, ArchiveError> {
    limits
        .validate()
        .map_err(|error| ArchiveError::InvalidLimits(error.to_string()))?;
    let mut entries = Vec::new();
    let mut file_count = 0u64;
    let mut expanded_bytes = 0u64;
    for (index, header) in headers.into_iter().enumerate() {
        if index as u64 >= limits.max_entries {
            return Err(ArchiveError::Limit {
                path: header.path,
                reason: "entry-count limit exceeded".to_string(),
            });
        }
        let path = normalize_entry_path(
            &header.path,
            limits.max_path_bytes,
            limits.max_component_bytes,
        )?;
        validate_kind_metadata(&header)?;
        if header.kind == ArchiveEntryKind::File {
            file_count = file_count
                .checked_add(1)
                .ok_or_else(|| ArchiveError::Limit {
                    path: header.path.clone(),
                    reason: "file count overflow".to_string(),
                })?;
            if file_count > limits.max_files {
                return limit(&header.path, "file-count limit exceeded");
            }
            if header.expanded_bytes > limits.max_file_bytes {
                return limit(&header.path, "per-file expanded-size limit exceeded");
            }
            if header.expanded_bytes > 0
                && (header.compressed_bytes == 0
                    || header.expanded_bytes
                        > header
                            .compressed_bytes
                            .saturating_mul(limits.max_compression_ratio))
            {
                return limit(&header.path, "compression-ratio limit exceeded");
            }
        }
        expanded_bytes = expanded_bytes
            .checked_add(header.expanded_bytes)
            .ok_or_else(|| ArchiveError::Limit {
                path: header.path.clone(),
                reason: "expanded-size overflow".to_string(),
            })?;
        if expanded_bytes > limits.max_expanded_bytes {
            return limit(&header.path, "total expanded-size limit exceeded");
        }
        let link_target = header
            .link_target
            .as_deref()
            .map(|target| {
                normalize_link_target(
                    &path,
                    target,
                    limits.max_path_bytes,
                    limits.max_component_bytes,
                )
            })
            .transpose()?;
        entries.push(ValidatedArchiveEntry {
            path,
            kind: header.kind,
            expanded_bytes: header.expanded_bytes,
            compressed_bytes: header.compressed_bytes,
            link_target,
            executable: header.executable,
        });
    }
    entries.sort_by(|left, right| left.path.cmp(&right.path));
    reject_collisions(&entries)?;
    reject_ancestor_conflicts(&entries)?;
    validate_links(&entries)?;
    Ok(ValidatedArchive {
        entries,
        file_count,
        expanded_bytes,
    })
}

pub fn normalize_link_for_entry(
    entry: &NormalizedRelativePath,
    target: &str,
    limits: ArchiveLimits,
) -> Result<NormalizedRelativePath, ArchiveError> {
    limits
        .validate()
        .map_err(|error| ArchiveError::InvalidLimits(error.to_string()))?;
    normalize_link_target(
        entry,
        target,
        limits.max_path_bytes,
        limits.max_component_bytes,
    )
}

fn validate_kind_metadata(header: &ArchiveEntryHeader) -> Result<(), ArchiveError> {
    let valid = match header.kind {
        ArchiveEntryKind::File => header.link_target.is_none(),
        ArchiveEntryKind::Directory => {
            header.link_target.is_none()
                && header.expanded_bytes == 0
                && header.compressed_bytes == 0
                && !header.executable
        }
        ArchiveEntryKind::Symlink | ArchiveEntryKind::Hardlink => {
            header.link_target.is_some()
                && header.expanded_bytes == 0
                && header.compressed_bytes == 0
                && !header.executable
        }
        kind => {
            return Err(ArchiveError::UnsupportedType {
                path: header.path.clone(),
                kind,
            });
        }
    };
    if valid {
        Ok(())
    } else {
        Err(ArchiveError::Inconsistent {
            path: header.path.clone(),
            reason: "entry type, size, executable bit, and link target disagree".to_string(),
        })
    }
}

fn reject_ancestor_conflicts(entries: &[ValidatedArchiveEntry]) -> Result<(), ArchiveError> {
    for (index, entry) in entries.iter().enumerate() {
        if entry.kind == ArchiveEntryKind::Directory {
            continue;
        }
        let prefix = format!("{}/", entry.path.as_str());
        if let Some(descendant) = entries[index + 1..]
            .iter()
            .find(|candidate| candidate.path.as_str().starts_with(&prefix))
        {
            return Err(ArchiveError::Collision {
                first: entry.path.to_string(),
                second: descendant.path.to_string(),
            });
        }
    }
    Ok(())
}

fn validate_links(entries: &[ValidatedArchiveEntry]) -> Result<(), ArchiveError> {
    let kinds: BTreeMap<_, _> = entries
        .iter()
        .map(|entry| (&entry.path, entry.kind))
        .collect();
    for entry in entries.iter().filter(|entry| {
        matches!(
            entry.kind,
            ArchiveEntryKind::Hardlink | ArchiveEntryKind::Symlink
        )
    }) {
        let target = entry
            .link_target
            .as_ref()
            .expect("validated hardlink has target");
        let target_prefix = format!("{}/", target.as_str());
        let target_kind = kinds.get(target).copied();
        let valid = match entry.kind {
            ArchiveEntryKind::Hardlink => target_kind == Some(ArchiveEntryKind::File),
            ArchiveEntryKind::Symlink => {
                target_kind.is_some_and(|kind| {
                    !matches!(kind, ArchiveEntryKind::Symlink | ArchiveEntryKind::Hardlink)
                }) || entries
                    .iter()
                    .any(|candidate| candidate.path.as_str().starts_with(&target_prefix))
            }
            _ => unreachable!(),
        };
        if !valid {
            return Err(ArchiveError::Inconsistent {
                path: entry.path.to_string(),
                reason: format!("link target `{target}` is not a materialized regular entry"),
            });
        }
    }
    Ok(())
}

fn limit<T>(path: &str, reason: &str) -> Result<T, ArchiveError> {
    Err(ArchiveError::Limit {
        path: path.to_string(),
        reason: reason.to_string(),
    })
}
