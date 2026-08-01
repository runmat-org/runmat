use crate::NativeCacheError;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheLayout {
    pub root: PathBuf,
    pub database: PathBuf,
    pub staging: PathBuf,
    pub trees: PathBuf,
    pub locks: PathBuf,
}

impl CacheLayout {
    pub fn new(root: PathBuf) -> Self {
        Self {
            database: root.join("cache.sqlite3"),
            staging: root.join("staging"),
            trees: root.join("trees"),
            locks: root.join("locks"),
            root,
        }
    }

    pub fn create(&self) -> Result<(), NativeCacheError> {
        for directory in [&self.root, &self.staging, &self.trees, &self.locks] {
            std::fs::create_dir_all(directory)
                .map_err(|error| NativeCacheError::io(directory, error))?;
        }
        Ok(())
    }
}
