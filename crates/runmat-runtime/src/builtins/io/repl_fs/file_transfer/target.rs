use runmat_filesystem as vfs;
use std::io;
use std::path::{Path, PathBuf};

pub(super) async fn remove_existing(path: &Path, is_directory: bool) -> io::Result<()> {
    let result = if is_directory {
        vfs::remove_dir_all_async(path).await
    } else {
        vfs::remove_file_async(path).await
    };
    match result {
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        other => other,
    }
}

pub(super) async fn same_path(left: &Path, right: &Path) -> bool {
    if left == right {
        return true;
    }
    match (
        vfs::canonicalize_async(left).await,
        vfs::canonicalize_async(right).await,
    ) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

pub(super) async fn is_descendant(parent: &Path, candidate: &Path) -> bool {
    if candidate.starts_with(parent) && candidate != parent {
        return true;
    }
    match (
        vfs::canonicalize_async(parent).await,
        vfs::canonicalize_async(candidate).await,
    ) {
        (Ok(parent), Ok(candidate)) => candidate.starts_with(&parent) && candidate != parent,
        _ => false,
    }
}

pub(super) async fn wildcard_matches(pattern: &str) -> io::Result<Vec<PathBuf>> {
    let pattern_path = Path::new(pattern);
    let name = pattern_path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "pattern has no file name"))?;
    let matcher = glob::Pattern::new(&name.to_string_lossy())
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error.msg))?;
    let parent = pattern_path.parent().unwrap_or_else(|| Path::new("."));
    let mut matches = vfs::read_dir_async(parent)
        .await?
        .into_iter()
        .filter(|entry| matcher.matches(&entry.file_name().to_string_lossy()))
        .map(|entry| entry.path().to_path_buf())
        .collect::<Vec<_>>();
    matches.sort();
    Ok(matches)
}

pub(super) fn display(path: &Path) -> String {
    path.display().to_string()
}
