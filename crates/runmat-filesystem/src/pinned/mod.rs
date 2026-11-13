//! Execution-scoped filesystem authority. The manifest fixes all input names
//! and versions; subsequent shared edits cannot change a running computation.
mod mounts;
mod output;
mod provider;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub use provider::PinnedFsProvider;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    io,
    path::{Component, Path, PathBuf},
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PinnedEntry {
    pub path: String,
    pub content: PinnedContent,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PinnedContent {
    File {
        #[serde(rename = "versionId")]
        version_id: String,
        size: u64,
    },
    Directory {},
}
impl PinnedEntry {
    fn is_directory(&self) -> bool {
        matches!(self.content, PinnedContent::Directory { .. })
    }
    fn size(&self) -> u64 {
        match self.content {
            PinnedContent::File { size, .. } => size,
            PinnedContent::Directory { .. } => 0,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct InputManifest {
    pub snapshot_id: String,
    pub entries: Vec<PinnedEntry>,
}

impl InputManifest {
    fn index(self) -> io::Result<BTreeMap<PathBuf, PinnedEntry>> {
        if self.snapshot_id.is_empty()
            || self.snapshot_id.len() > 160
            || self.entries.len() > 10_000
        {
            return Err(invalid("invalid input manifest"));
        }
        let mut entries = BTreeMap::new();
        for entry in self.entries {
            let path = normalize(Path::new(&entry.path))?;
            if let PinnedContent::File { version_id, .. } = &entry.content {
                if version_id.is_empty() || version_id.len() > 160 {
                    return Err(invalid("missing immutable input version"));
                }
            }
            if entries.insert(path, entry).is_some() {
                return Err(invalid("duplicate input path"));
            }
        }
        let paths: Vec<_> = entries.keys().cloned().collect();
        for path in paths {
            let mut parent = path.parent();
            while let Some(path) = parent {
                if entries.get(path).is_some_and(|entry| !entry.is_directory()) {
                    return Err(invalid("input file is also a directory"));
                }
                entries
                    .entry(path.to_path_buf())
                    .or_insert_with(|| PinnedEntry {
                        path: path.to_string_lossy().into_owned(),
                        content: PinnedContent::Directory {},
                    });
                parent = path.parent();
            }
        }
        entries
            .entry(PathBuf::from("/"))
            .or_insert_with(|| PinnedEntry {
                path: "/".into(),
                content: PinnedContent::Directory {},
            });
        Ok(entries)
    }
}
pub(super) fn normalize(path: &Path) -> io::Result<PathBuf> {
    let mut result = PathBuf::from("/");
    for part in path.components() {
        match part {
            Component::Normal(part) => result.push(part),
            Component::RootDir | Component::CurDir => {}
            Component::ParentDir => {
                if !result.pop() {
                    return Err(invalid("input path escapes project"));
                }
            }
            Component::Prefix(_) => return Err(invalid("input path must be project relative")),
        }
    }
    Ok(result)
}
fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}
