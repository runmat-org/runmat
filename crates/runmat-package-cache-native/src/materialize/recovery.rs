use crate::filesystem::CacheLayout;
use crate::NativeCacheError;

pub fn remove_interrupted_staging(layout: &CacheLayout) -> Result<Vec<String>, NativeCacheError> {
    layout.create()?;
    let mut removed = Vec::new();
    for entry in std::fs::read_dir(&layout.staging)
        .map_err(|error| NativeCacheError::io(&layout.staging, error))?
    {
        let entry = entry.map_err(|error| NativeCacheError::io(&layout.staging, error))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with("tree-") || !name.ends_with(".partial") {
            continue;
        }
        let path = entry.path();
        let metadata =
            std::fs::symlink_metadata(&path).map_err(|error| NativeCacheError::io(&path, error))?;
        if !metadata.is_dir() || metadata.file_type().is_symlink() {
            continue;
        }
        std::fs::remove_dir_all(&path).map_err(|error| NativeCacheError::io(&path, error))?;
        removed.push(name);
    }
    removed.sort();
    Ok(removed)
}
