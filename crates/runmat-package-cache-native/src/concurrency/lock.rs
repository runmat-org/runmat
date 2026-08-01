use crate::NativeCacheError;
use fs2::FileExt;
use std::fs::File;
use std::path::Path;

#[derive(Debug)]
pub struct ProcessLock {
    file: File,
}

impl ProcessLock {
    pub fn acquire(path: &Path) -> Result<Self, NativeCacheError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| NativeCacheError::io(parent, error))?;
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(path)
            .map_err(|error| NativeCacheError::io(path, error))?;
        file.lock_exclusive()
            .map_err(|error| NativeCacheError::io(path, error))?;
        Ok(Self { file })
    }
}

impl Drop for ProcessLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}
