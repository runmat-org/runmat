use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::builtins::common::fs::compare_names;
use runmat_filesystem as vfs;

pub(super) async fn collect(
    root: &super::root::Root,
    exclusions: &super::exclusions::Exclusions,
) -> crate::BuiltinResult<Vec<String>> {
    let mut seen = HashSet::new();
    let mut folders = Vec::new();
    visit(
        &root.path,
        root.canonical.clone(),
        exclusions,
        &mut seen,
        &mut folders,
    )
    .await?;
    Ok(folders)
}

#[async_recursion::async_recursion(?Send)]
async fn visit(
    path: &Path,
    canonical: String,
    exclusions: &super::exclusions::Exclusions,
    seen: &mut HashSet<String>,
    folders: &mut Vec<String>,
) -> crate::BuiltinResult<()> {
    if !seen.insert(super::super::path_list::identity(&canonical))
        || exclusions.contains(&canonical)
    {
        return Ok(());
    }
    folders.push(canonical);

    let Ok(entries) = vfs::read_dir_async(path).await else {
        return Ok(());
    };
    let mut children = Vec::new();
    for entry in entries {
        let source = entry.path().to_path_buf();
        let Ok(metadata) = vfs::metadata_async(&source).await else {
            continue;
        };
        if !metadata.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if is_special_folder(&name) {
            continue;
        }
        let Ok(path) = vfs::canonicalize_async(&source).await else {
            continue;
        };
        children.push(Child {
            canonical: super::root::canonical_string(&path),
            path,
            name,
        });
    }
    children.sort_by(|left, right| compare_names(&left.name, &right.name));
    for child in children {
        visit(&child.path, child.canonical, exclusions, seen, folders).await?;
    }
    Ok(())
}

struct Child {
    path: PathBuf,
    canonical: String,
    name: String,
}

fn is_special_folder(name: &str) -> bool {
    if name.starts_with('@') || name.starts_with('+') {
        return true;
    }
    #[cfg(windows)]
    {
        matches!(name.to_ascii_lowercase().as_str(), "private" | "resources")
    }
    #[cfg(not(windows))]
    {
        matches!(name, "private" | "resources")
    }
}
