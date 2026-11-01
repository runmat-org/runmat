//! Native workspace publication uses a durable commit record and a mount-wide
//! advisory lock. Participating readers recover committed work before reading;
//! external programs reading ordinary paths may observe physical file updates.
mod admission;
mod capture;
mod execution;
mod handle;
mod journal;
mod publish;
#[cfg(test)]
mod tests;
#[cfg(test)]
pub(crate) use tests::failed_commit_for_test;

pub(crate) use capture::capture;
pub(crate) use capture::capture_locked_admitted;
pub(crate) use execution::transition_execution;
pub(crate) use handle::WritableHandle;
pub(crate) use journal::recover_admitted as recover_locked_admitted;
pub(crate) use journal::{guard, guard_admitted, protected, Guard};
pub(crate) use publish::publish_locked;
pub(crate) use publish::{lookup, publish, publish_with_lease};

use std::{
    fs, io,
    path::{Component, Path, PathBuf},
};

pub(crate) const JOURNAL: &str = ".artifacts/workspace-publications";

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

/// Checked batch paths never use sandbox traversal clamping. Journal paths are
/// reserved so ordinary tools cannot rewrite publication receipts or recovery.
fn confined(root: &Path, relative: &Path) -> io::Result<PathBuf> {
    if relative.as_os_str().is_empty()
        || relative.to_str().is_none_or(|path| path.len() > 1024)
        || relative
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(invalid(
            "workspace path must be a confined relative UTF-8 path",
        ));
    }
    let path = root.join(relative);
    ensure_no_symlink(root, &path)?;
    Ok(path)
}

pub(crate) fn ensure_no_symlink(root: &Path, path: &Path) -> io::Result<()> {
    let relative = path
        .strip_prefix(root)
        .map_err(|_| invalid("workspace path escapes its mount"))?;
    let mut current = root.to_path_buf();
    for component in relative.components() {
        if !matches!(component, Component::Normal(_)) {
            return Err(invalid("workspace path escapes its mount"));
        }
        current.push(component);
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "workspace paths cannot traverse symbolic links",
                ));
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

fn sync_directory(path: &Path) -> io::Result<()> {
    crate::native::directory_sync::sync(path)
}

pub(crate) fn sync_ancestors(root: &Path, path: &Path) -> io::Result<()> {
    let mut current = Some(path);
    while let Some(directory) = current {
        if !directory.starts_with(root) {
            break;
        }
        sync_directory(directory)?;
        if directory == root {
            break;
        }
        current = directory.parent();
    }
    Ok(())
}

pub(crate) use journal::uncertain_receipt as journal_receipt;
