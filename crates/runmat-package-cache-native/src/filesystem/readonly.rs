use crate::NativeCacheError;
use std::path::Path;

pub fn make_tree_readonly(root: &Path) -> Result<(), NativeCacheError> {
    make_entry_readonly(root)
}

pub(crate) fn make_tree_removable(root: &Path) -> Result<(), NativeCacheError> {
    make_entry_removable(root)
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

fn make_entry_removable(path: &Path) -> Result<(), NativeCacheError> {
    let metadata =
        std::fs::symlink_metadata(path).map_err(|error| NativeCacheError::io(path, error))?;
    if metadata.file_type().is_symlink() {
        return Ok(());
    }
    set_owner_writable(path, &metadata)?;
    if metadata.is_dir() {
        for entry in std::fs::read_dir(path).map_err(|error| NativeCacheError::io(path, error))? {
            let child = entry
                .map_err(|error| NativeCacheError::io(path, error))?
                .path();
            make_entry_removable(&child)?;
        }
    }
    Ok(())
}

#[cfg(unix)]
fn set_owner_writable(path: &Path, metadata: &std::fs::Metadata) -> Result<(), NativeCacheError> {
    use std::os::unix::fs::PermissionsExt as _;
    let mut permissions = metadata.permissions();
    let owner_bits = if metadata.is_dir() { 0o700 } else { 0o600 };
    permissions.set_mode(permissions.mode() | owner_bits);
    std::fs::set_permissions(path, permissions).map_err(|error| NativeCacheError::io(path, error))
}

#[cfg(not(unix))]
#[allow(clippy::permissions_set_readonly_false)] // Clears the Windows read-only attribute; it does not broaden ACLs.
fn set_owner_writable(path: &Path, metadata: &std::fs::Metadata) -> Result<(), NativeCacheError> {
    let mut permissions = metadata.permissions();
    permissions.set_readonly(false);
    std::fs::set_permissions(path, permissions).map_err(|error| NativeCacheError::io(path, error))
}
