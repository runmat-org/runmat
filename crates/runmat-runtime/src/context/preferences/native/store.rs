//! Native application-settings document owner. Never uses the project VFS.

use super::super::document::{PreferenceDocument, PreferenceDocumentError};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use tempfile::NamedTempFile;

#[derive(Debug, thiserror::Error)]
pub(super) enum NativeStoreError {
    #[error("application-settings I/O failed: {0}")]
    Io(#[source] io::Error),
    #[error(transparent)]
    Document(#[from] PreferenceDocumentError),
    #[error("preference replacement committed but durability could not be confirmed: {0}")]
    Indeterminate(#[source] io::Error),
}

#[derive(Clone)]
pub(super) struct NativeStore {
    path: PathBuf,
}

impl NativeStore {
    pub fn new(path: PathBuf) -> Result<Self, io::Error> {
        if !path.is_absolute() || path.file_name().is_none() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "preference settings file must be an absolute file path",
            ));
        }
        Ok(Self { path })
    }

    pub fn read(&self) -> Result<PreferenceDocument, NativeStoreError> {
        match fs::read(&self.path) {
            Ok(bytes) => PreferenceDocument::decode(&bytes).map_err(NativeStoreError::Document),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                Ok(PreferenceDocument::empty())
            }
            Err(error) => Err(NativeStoreError::Io(error)),
        }
    }

    pub fn begin_write(&self) -> Result<NativeWrite, NativeStoreError> {
        let parent = self.path.parent().expect("validated absolute file path");
        fs::create_dir_all(parent).map_err(NativeStoreError::Io)?;
        let mut lock_name = self
            .path
            .file_name()
            .expect("validated file name")
            .to_os_string();
        lock_name.push(".lock");
        let lock_path = self.path.with_file_name(lock_name);
        let lock = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(lock_path)
            .map_err(NativeStoreError::Io)?;
        lock.lock().map_err(NativeStoreError::Io)?;
        let document = self.read()?;
        Ok(NativeWrite {
            _lock: lock,
            path: self.path.clone(),
            document,
        })
    }
}

pub(super) struct NativeWrite {
    _lock: File,
    path: PathBuf,
    pub document: PreferenceDocument,
}

impl NativeWrite {
    pub fn stage(self) -> Result<NativeStage, NativeStoreError> {
        let parent = self.path.parent().expect("validated absolute file path");
        let bytes = self.document.encode()?;
        let mut temporary = NamedTempFile::new_in(parent).map_err(NativeStoreError::Io)?;
        temporary.write_all(&bytes).map_err(NativeStoreError::Io)?;
        temporary.flush().map_err(NativeStoreError::Io)?;
        temporary
            .as_file()
            .sync_all()
            .map_err(NativeStoreError::Io)?;
        Ok(NativeStage {
            temporary,
            _lock: self._lock,
            path: self.path,
        })
    }
}

/// Dropping an uncommitted stage removes only its private temporary file.
pub(super) struct NativeStage {
    temporary: NamedTempFile,
    _lock: File,
    path: PathBuf,
}

impl NativeStage {
    pub fn commit(self) -> Result<(), NativeStoreError> {
        let parent = self.path.parent().expect("validated absolute file path");
        self.temporary
            .persist(&self.path)
            .map_err(|error| NativeStoreError::Io(error.error))?;
        sync_parent(parent).map_err(NativeStoreError::Indeterminate)
    }
}

fn sync_parent(parent: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        File::open(parent)?.sync_all()
    }
    #[cfg(not(unix))]
    {
        let _ = parent;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "parent-directory durability confirmation is unavailable on this host",
        ))
    }
}
