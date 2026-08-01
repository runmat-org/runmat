use crate::NativeCacheError;
use std::path::Path;

pub fn make_tree_readonly(root: &Path) -> Result<(), NativeCacheError> {
    make_entry_readonly(root)
}

fn make_entry_readonly(path: &Path) -> Result<(), NativeCacheError> {
    let metadata =
        std::fs::symlink_metadata(path).map_err(|error| NativeCacheError::io(path, error))?;
    if metadata.file_type().is_symlink() {
        return Ok(());
    }
    if metadata.is_dir() {
        let entries = std::fs::read_dir(path).map_err(|error| NativeCacheError::io(path, error))?;
        for entry in entries {
            let child = entry
                .map_err(|error| NativeCacheError::io(path, error))?
                .path();
            make_entry_readonly(&child)?;
        }
    }
    let mut permissions = metadata.permissions();
    permissions.set_readonly(true);
    std::fs::set_permissions(path, permissions).map_err(|error| NativeCacheError::io(path, error))
}
