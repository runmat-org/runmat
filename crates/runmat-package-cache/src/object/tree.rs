use crate::CacheError;
use runmat_package::{ContentDigest, NormalizedRelativePath};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const TREE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TreeEntryKind {
    File,
    Directory,
    Symlink,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TreeEntry {
    pub path: NormalizedRelativePath,
    pub kind: TreeEntryKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub digest: Option<ContentDigest>,
    pub byte_len: u64,
    pub executable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub link_target: Option<NormalizedRelativePath>,
}

impl TreeEntry {
    pub fn file(
        path: NormalizedRelativePath,
        digest: ContentDigest,
        byte_len: u64,
        executable: bool,
    ) -> Self {
        Self {
            path,
            kind: TreeEntryKind::File,
            digest: Some(digest),
            byte_len,
            executable,
            link_target: None,
        }
    }

    pub fn directory(path: NormalizedRelativePath) -> Self {
        Self {
            path,
            kind: TreeEntryKind::Directory,
            digest: None,
            byte_len: 0,
            executable: false,
            link_target: None,
        }
    }

    pub fn symlink(path: NormalizedRelativePath, target: NormalizedRelativePath) -> Self {
        Self {
            path,
            kind: TreeEntryKind::Symlink,
            digest: None,
            byte_len: 0,
            executable: false,
            link_target: Some(target),
        }
    }

    fn validate(&self) -> Result<(), CacheError> {
        match self.kind {
            TreeEntryKind::File if self.digest.is_some() && self.link_target.is_none() => Ok(()),
            TreeEntryKind::Directory
                if self.digest.is_none()
                    && self.link_target.is_none()
                    && self.byte_len == 0
                    && !self.executable =>
            {
                Ok(())
            }
            TreeEntryKind::Symlink
                if self.digest.is_none()
                    && self.link_target.is_some()
                    && self.byte_len == 0
                    && !self.executable =>
            {
                Ok(())
            }
            _ => Err(CacheError::InvalidObject(format!(
                "tree entry `{}` has metadata inconsistent with {:?}",
                self.path, self.kind
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TreeManifest {
    pub schema_version: u32,
    pub digest: ContentDigest,
    pub entries: Vec<TreeEntry>,
    pub file_count: u64,
    pub total_bytes: u64,
}

impl TreeManifest {
    pub fn new(mut entries: Vec<TreeEntry>) -> Result<Self, CacheError> {
        entries.sort();
        for entry in &entries {
            entry.validate()?;
        }
        if entries.windows(2).any(|pair| pair[0].path == pair[1].path) {
            return Err(CacheError::InvalidObject(
                "tree contains duplicate paths".to_string(),
            ));
        }
        for (index, entry) in entries.iter().enumerate() {
            let prefix = format!("{}/", entry.path.as_str());
            if entry.kind != TreeEntryKind::Directory
                && entries[index + 1..]
                    .iter()
                    .any(|candidate| candidate.path.as_str().starts_with(&prefix))
            {
                return Err(CacheError::InvalidObject(format!(
                    "non-directory tree entry `{}` is an ancestor of another entry",
                    entry.path
                )));
            }
        }
        let file_count = entries
            .iter()
            .filter(|entry| entry.kind == TreeEntryKind::File)
            .count() as u64;
        let total_bytes = entries.iter().try_fold(0u64, |total, entry| {
            total
                .checked_add(entry.byte_len)
                .ok_or_else(|| CacheError::InvalidObject("tree byte size overflow".to_string()))
        })?;
        let digest = compute_digest(&entries, file_count, total_bytes)?;
        Ok(Self {
            schema_version: TREE_SCHEMA_VERSION,
            digest,
            entries,
            file_count,
            total_bytes,
        })
    }

    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version != TREE_SCHEMA_VERSION {
            return Err(CacheError::InvalidObject(format!(
                "unsupported tree schema {}",
                self.schema_version
            )));
        }
        let canonical = Self::new(self.entries.clone())?;
        if canonical.entries != self.entries
            || canonical.digest != self.digest
            || canonical.file_count != self.file_count
            || canonical.total_bytes != self.total_bytes
        {
            return Err(CacheError::InvalidObject(
                "tree manifest does not match canonical contents".to_string(),
            ));
        }
        Ok(())
    }

    pub fn referenced_blobs(&self) -> BTreeSet<ContentDigest> {
        self.entries
            .iter()
            .filter_map(|entry| entry.digest.clone())
            .collect()
    }
}

fn compute_digest(
    entries: &[TreeEntry],
    file_count: u64,
    total_bytes: u64,
) -> Result<ContentDigest, CacheError> {
    #[derive(Serialize)]
    struct Input<'a> {
        format: &'static str,
        entries: &'a [TreeEntry],
        file_count: u64,
        total_bytes: u64,
    }
    serde_json::to_vec(&Input {
        format: "runmat-cache-tree-v1",
        entries,
        file_count,
        total_bytes,
    })
    .map(ContentDigest::sha256)
    .map_err(|error| CacheError::InvalidObject(error.to_string()))
}
