use std::path::{Path, PathBuf};

use crate::{ProcessHostError, ProcessHostResult};

#[derive(Debug)]
pub struct FileBackedSharedMemory {
    path: PathBuf,
}

impl FileBackedSharedMemory {
    pub fn open_existing(path: impl Into<PathBuf>) -> ProcessHostResult<Self> {
        let path = path.into();
        if !path.is_absolute() {
            return Err(ProcessHostError::Configuration(
                "file-backed shared memory path must be absolute".into(),
            ));
        }
        Ok(Self { path })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}
