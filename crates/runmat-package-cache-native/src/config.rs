use crate::filesystem::CacheLayout;
use crate::NativeCacheError;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeCacheConfig {
    pub root: PathBuf,
    pub quota_bytes: Option<u64>,
}

impl NativeCacheConfig {
    pub fn platform_default() -> Result<Self, NativeCacheError> {
        let root = dirs::cache_dir()
            .ok_or_else(|| {
                NativeCacheError::Config("platform cache directory is unavailable".into())
            })?
            .join("runmat")
            .join("packages");
        Ok(Self {
            root,
            quota_bytes: None,
        })
    }

    pub fn layout(&self) -> CacheLayout {
        CacheLayout::new(self.root.clone())
    }

    pub fn validate(&self) -> Result<(), NativeCacheError> {
        if self.root.as_os_str().is_empty() {
            return Err(NativeCacheError::Config(
                "cache root must not be empty".to_string(),
            ));
        }
        if self.quota_bytes == Some(0) {
            return Err(NativeCacheError::Config(
                "quota must be greater than zero".to_string(),
            ));
        }
        Ok(())
    }
}
