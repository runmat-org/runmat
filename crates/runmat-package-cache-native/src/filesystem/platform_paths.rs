use crate::NativeCacheError;
use std::path::PathBuf;

pub fn platform_cache_root() -> Result<PathBuf, NativeCacheError> {
    dirs::cache_dir()
        .map(|root| root.join("runmat").join("packages"))
        .ok_or_else(|| {
            NativeCacheError::Config("platform cache directory is unavailable".to_string())
        })
}
