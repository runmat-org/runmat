use std::path::{Path, PathBuf};

use crate::builtins::common::fs::expand_user_path;
use runmat_filesystem as vfs;

#[derive(Default)]
pub(super) struct Exclusions {
    entries: Vec<Entry>,
}

impl Exclusions {
    pub(super) async fn resolve(text: Option<&str>, root: &super::root::Root) -> Self {
        let mut entries = Vec::new();
        if let Some(text) = text {
            for segment in text.split(crate::builtins::common::path_state::PATH_LIST_SEPARATOR) {
                if let Some(canonical) = resolve_segment(segment.trim(), root).await {
                    entries.push(Entry::new(canonical));
                }
            }
        }
        Self { entries }
    }

    pub(super) fn contains(&self, canonical: &str) -> bool {
        let candidate = super::super::path_list::identity(canonical);
        self.entries.iter().any(|entry| {
            candidate == entry.canonical || candidate.starts_with(&entry.descendant_prefix)
        })
    }
}

struct Entry {
    canonical: String,
    descendant_prefix: String,
}

impl Entry {
    fn new(canonical: String) -> Self {
        let canonical = super::super::path_list::identity(&canonical);
        let mut descendant_prefix = canonical.clone();
        if !descendant_prefix.ends_with(std::path::MAIN_SEPARATOR) {
            descendant_prefix.push(std::path::MAIN_SEPARATOR);
        }
        Self {
            canonical,
            descendant_prefix,
        }
    }
}

async fn resolve_segment(segment: &str, root: &super::root::Root) -> Option<String> {
    if segment.is_empty() {
        return None;
    }
    let expanded = expand_user_path(segment, super::errors::NAME).ok()?;
    let candidate = PathBuf::from(expanded);
    let rooted = if super::super::is_rooted_path(&candidate) {
        candidate
    } else {
        root.path.join(candidate)
    };
    if let Some(canonical) = canonicalize(&rooted).await {
        return Some(canonical);
    }

    let cwd = vfs::current_dir().ok()?;
    let fallback = if super::super::is_rooted_path(Path::new(segment)) {
        PathBuf::from(segment)
    } else {
        cwd.join(segment)
    };
    canonicalize(&fallback).await
}

async fn canonicalize(path: &Path) -> Option<String> {
    vfs::canonicalize_async(path)
        .await
        .ok()
        .map(|path| super::root::canonical_string(&path))
}
