use crate::NativeCacheError;
use git2::{ObjectType, Repository, Tree};
use runmat_package::NormalizedRelativePath;
use runmat_package_cache::{GitInventoryEntry, GitTreeInventory};

pub(super) fn snapshot_tree(
    repository: &Repository,
    commit: git2::Oid,
    subdir: &NormalizedRelativePath,
) -> Result<GitTreeInventory, NativeCacheError> {
    let commit = repository
        .find_commit(commit)
        .map_err(super::remote::git_error)?;
    let root = commit.tree().map_err(super::remote::git_error)?;
    let tree = if subdir.as_str() == "." {
        root
    } else {
        let entry = root
            .get_path(std::path::Path::new(subdir.as_str()))
            .map_err(super::remote::git_error)?;
        repository
            .find_tree(entry.id())
            .map_err(super::remote::git_error)?
    };
    let mut entries = Vec::new();
    walk(repository, &tree, "", &mut entries)?;
    Ok(GitTreeInventory {
        commit: commit.id().to_string(),
        entries,
    })
}

fn walk(
    repository: &Repository,
    tree: &Tree<'_>,
    parent: &str,
    entries: &mut Vec<GitInventoryEntry>,
) -> Result<(), NativeCacheError> {
    for item in tree {
        let name = item.name().ok_or_else(|| {
            NativeCacheError::Git("Git tree contains a non-UTF-8 entry name".to_string())
        })?;
        let joined = if parent.is_empty() {
            name.to_string()
        } else {
            format!("{parent}/{name}")
        };
        let path = NormalizedRelativePath::new(&joined)
            .map_err(|error| NativeCacheError::Git(error.to_string()))?;
        match item.kind() {
            Some(ObjectType::Tree) => {
                entries.push(GitInventoryEntry::directory(path.as_str()));
                let child = repository
                    .find_tree(item.id())
                    .map_err(super::remote::git_error)?;
                walk(repository, &child, &joined, entries)?;
            }
            Some(ObjectType::Blob) if item.filemode() == 0o120000 => {
                let blob = repository
                    .find_blob(item.id())
                    .map_err(super::remote::git_error)?;
                let target = std::str::from_utf8(blob.content()).map_err(|_| {
                    NativeCacheError::Git(format!("symlink `{joined}` target is not UTF-8"))
                })?;
                entries.push(GitInventoryEntry::symlink(path.as_str(), target));
            }
            Some(ObjectType::Blob) if matches!(item.filemode(), 0o100644 | 0o100755) => {
                let blob = repository
                    .find_blob(item.id())
                    .map_err(super::remote::git_error)?;
                entries.push(GitInventoryEntry::file(
                    path.as_str(),
                    blob.content().to_vec(),
                    item.filemode() == 0o100755,
                ));
            }
            Some(kind) => {
                return Err(NativeCacheError::Git(format!(
                    "Git tree entry `{joined}` has unsupported object kind {kind:?} or mode {:o}",
                    item.filemode()
                )));
            }
            None => {
                return Err(NativeCacheError::Git(format!(
                    "Git tree entry `{joined}` has an unknown object kind"
                )));
            }
        }
    }
    Ok(())
}
