use crate::NativeCacheError;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), NativeCacheError> {
    let parent = path.parent().ok_or_else(|| {
        NativeCacheError::Config(format!(
            "atomic-write target `{}` has no parent",
            path.display()
        ))
    })?;
    std::fs::create_dir_all(parent).map_err(|error| NativeCacheError::io(parent, error))?;
    let temporary = temporary_path(path);
    let result = write_and_promote(&temporary, path, bytes);
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

fn write_and_promote(
    temporary: &Path,
    destination: &Path,
    bytes: &[u8],
) -> Result<(), NativeCacheError> {
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(temporary)
        .map_err(|error| NativeCacheError::io(temporary, error))?;
    file.write_all(bytes)
        .map_err(|error| NativeCacheError::io(temporary, error))?;
    file.sync_all()
        .map_err(|error| NativeCacheError::io(temporary, error))?;
    std::fs::hard_link(temporary, destination)
        .map_err(|error| NativeCacheError::io(destination, error))?;
    std::fs::remove_file(temporary).map_err(|error| NativeCacheError::io(temporary, error))
}

fn temporary_path(path: &Path) -> PathBuf {
    let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("cache");
    path.with_file_name(format!(".{name}.{}.{}.tmp", std::process::id(), sequence))
}
