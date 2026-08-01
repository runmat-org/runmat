use super::staging::{assemble, StagingTree};
use crate::concurrency::ProcessLock;
use crate::filesystem::{make_tree_readonly, CacheLayout};
use crate::NativeCacheError;
use runmat_package::ContentDigest;
use runmat_package_cache::{CacheBackend, TreeEntryKind, TreeManifest};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

pub async fn materialize_tree<B: CacheBackend>(
    backend: &B,
    layout: &CacheLayout,
    tree: &TreeManifest,
) -> Result<PathBuf, NativeCacheError> {
    tree.validate()?;
    layout.create()?;
    let target = layout.tree_path(&tree.digest);
    let _lock = ProcessLock::acquire(&layout.materialization_lock(&tree.digest))?;
    if target.exists() {
        verify_materialized_tree(&target, tree)?;
        make_tree_readonly(&target)?;
        return Ok(target);
    }

    let mut staging = StagingTree::create(&layout.staging)?;
    assemble(backend, staging.path(), tree).await?;
    verify_materialized_tree(staging.path(), tree)?;
    match std::fs::rename(staging.path(), &target) {
        Ok(()) => {
            staging.disarm();
            make_tree_readonly(&target)?;
            Ok(target)
        }
        Err(_error) if target.exists() => {
            verify_materialized_tree(&target, tree)?;
            make_tree_readonly(&target)?;
            Ok(target)
        }
        Err(error) => Err(NativeCacheError::io(target, error)),
    }
}

pub fn verify_materialized_tree(root: &Path, tree: &TreeManifest) -> Result<(), NativeCacheError> {
    tree.validate()?;
    let expected = expected_paths(tree);
    verify_no_extras(root, root, &expected)?;
    for entry in &tree.entries {
        let path = root.join(entry.path.as_str());
        let metadata =
            std::fs::symlink_metadata(&path).map_err(|error| NativeCacheError::io(&path, error))?;
        match entry.kind {
            TreeEntryKind::Directory if metadata.is_dir() && !metadata.file_type().is_symlink() => {
            }
            TreeEntryKind::File if metadata.is_file() && !metadata.file_type().is_symlink() => {
                let bytes =
                    std::fs::read(&path).map_err(|error| NativeCacheError::io(&path, error))?;
                let expected_digest = entry.digest.as_ref().expect("validated file has digest");
                if bytes.len() as u64 != entry.byte_len
                    || ContentDigest::sha256(&bytes) != *expected_digest
                {
                    return corrupt(&path, "file size or digest differs from tree manifest");
                }
            }
            TreeEntryKind::Symlink if metadata.file_type().is_symlink() => {
                let actual = std::fs::read_link(&path)
                    .map_err(|error| NativeCacheError::io(&path, error))?;
                let resolved = lexical_resolve(
                    entry
                        .path
                        .as_str()
                        .rsplit_once('/')
                        .map_or("", |(parent, _)| parent),
                    &actual,
                )?;
                if Some(resolved.as_str()) != entry.link_target.as_ref().map(|path| path.as_str()) {
                    return corrupt(&path, "symlink target differs from tree manifest");
                }
            }
            _ => return corrupt(&path, "filesystem entry type differs from tree manifest"),
        }
    }
    Ok(())
}

fn expected_paths(tree: &TreeManifest) -> BTreeSet<PathBuf> {
    let mut expected = BTreeSet::new();
    for entry in &tree.entries {
        let path = Path::new(entry.path.as_str());
        expected.insert(path.to_path_buf());
        let mut parent = path.parent();
        while let Some(path) = parent {
            if path.as_os_str().is_empty() {
                break;
            }
            expected.insert(path.to_path_buf());
            parent = path.parent();
        }
    }
    expected
}

fn verify_no_extras(
    root: &Path,
    directory: &Path,
    expected: &BTreeSet<PathBuf>,
) -> Result<(), NativeCacheError> {
    for entry in
        std::fs::read_dir(directory).map_err(|error| NativeCacheError::io(directory, error))?
    {
        let path = entry
            .map_err(|error| NativeCacheError::io(directory, error))?
            .path();
        let relative = path
            .strip_prefix(root)
            .expect("walk remains beneath materialization root");
        if !expected.contains(relative) {
            return corrupt(&path, "unexpected filesystem entry");
        }
        let metadata =
            std::fs::symlink_metadata(&path).map_err(|error| NativeCacheError::io(&path, error))?;
        if metadata.is_dir() && !metadata.file_type().is_symlink() {
            verify_no_extras(root, &path, expected)?;
        }
    }
    Ok(())
}

fn lexical_resolve(parent: &str, target: &Path) -> Result<String, NativeCacheError> {
    let mut components: Vec<String> = if parent.is_empty() {
        Vec::new()
    } else {
        parent.split('/').map(str::to_string).collect()
    };
    for component in target.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if components.pop().is_none() {
                    return Err(NativeCacheError::CorruptTree {
                        path: target.to_path_buf(),
                        reason: "symlink escapes materialization root".to_string(),
                    });
                }
            }
            std::path::Component::Normal(value) => {
                components.push(value.to_string_lossy().into_owned());
            }
            _ => {
                return Err(NativeCacheError::CorruptTree {
                    path: target.to_path_buf(),
                    reason: "symlink target is absolute or prefixed".to_string(),
                });
            }
        }
    }
    Ok(components.join("/"))
}

fn corrupt<T>(path: &Path, reason: &str) -> Result<T, NativeCacheError> {
    Err(NativeCacheError::CorruptTree {
        path: path.to_path_buf(),
        reason: reason.to_string(),
    })
}
