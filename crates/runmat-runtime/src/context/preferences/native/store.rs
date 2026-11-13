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
    #[cfg(test)]
    commit_gate: Option<std::sync::Arc<CommitGate>>,
}

#[cfg(test)]
pub(super) struct CommitGate {
    pub reached: std::sync::mpsc::Sender<()>,
    pub wake_test: std::sync::mpsc::Sender<()>,
    pub resume: std::sync::Mutex<std::sync::mpsc::Receiver<()>>,
}

impl NativeStore {
    pub fn new(path: PathBuf) -> Result<Self, io::Error> {
        if !path.is_absolute() || path.file_name().is_none() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "preference settings file must be an absolute file path",
            ));
        }
        Ok(Self {
            path,
            #[cfg(test)]
            commit_gate: None,
        })
    }

    #[cfg(test)]
    pub fn with_commit_gate(mut self, gate: std::sync::Arc<CommitGate>) -> Self {
        self.commit_gate = Some(gate);
        self
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

    pub fn try_begin_write(&self) -> Result<Option<NativeWrite>, NativeStoreError> {
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
            .truncate(false)
            .read(true)
            .write(true)
            .open(lock_path)
            .map_err(NativeStoreError::Io)?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(fs::TryLockError::WouldBlock) => return Ok(None),
            Err(fs::TryLockError::Error(source)) => return Err(NativeStoreError::Io(source)),
        }
        let document = self.read()?;
        Ok(Some(NativeWrite {
            _lock: lock,
            path: self.path.clone(),
            document,
            #[cfg(test)]
            commit_gate: self.commit_gate.clone(),
        }))
    }
}

pub(super) struct NativeWrite {
    _lock: File,
    path: PathBuf,
    pub document: PreferenceDocument,
    #[cfg(test)]
    commit_gate: Option<std::sync::Arc<CommitGate>>,
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
            #[cfg(test)]
            commit_gate: self.commit_gate,
        })
    }
}

/// Dropping an uncommitted stage removes only its private temporary file.
pub(super) struct NativeStage {
    temporary: NamedTempFile,
    _lock: File,
    path: PathBuf,
    #[cfg(test)]
    commit_gate: Option<std::sync::Arc<CommitGate>>,
}

impl NativeStage {
    pub fn commit(self) -> Result<(), NativeStoreError> {
        #[cfg(test)]
        if let Some(gate) = &self.commit_gate {
            gate.reached.send(()).map_err(|source| {
                NativeStoreError::Io(io::Error::new(io::ErrorKind::BrokenPipe, source))
            })?;
            let _ = gate.wake_test.send(());
            gate.resume
                .lock()
                .expect("test gate mutex")
                .recv_timeout(std::time::Duration::from_secs(3))
                .map_err(|source| {
                    NativeStoreError::Io(io::Error::new(io::ErrorKind::TimedOut, source))
                })?;
        }
        #[cfg(windows)]
        return commit_windows(self);

        #[cfg(not(windows))]
        {
            let parent = self.path.parent().expect("validated absolute file path");
            self.temporary
                .persist(&self.path)
                .map_err(|error| NativeStoreError::Io(error.error))?;
            sync_parent(parent).map_err(NativeStoreError::Indeterminate)
        }
    }
}

#[cfg(windows)]
fn commit_windows(stage: NativeStage) -> Result<(), NativeStoreError> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MoveFileExW, SetFileAttributesW, FILE_ATTRIBUTE_NORMAL, MOVEFILE_REPLACE_EXISTING,
        MOVEFILE_WRITE_THROUGH,
    };

    fn wide(path: &Path) -> Vec<u16> {
        path.as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect()
    }

    // Closing the source handle before replacement matches Windows rename
    // rules. TempPath remains the sole cleanup owner until the move succeeds.
    let mut temporary = stage.temporary.into_temp_path();
    let source = wide(temporary.as_ref());
    let destination = wide(&stage.path);
    let normalized = unsafe { SetFileAttributesW(source.as_ptr(), FILE_ATTRIBUTE_NORMAL) };
    if normalized == 0 {
        return Err(NativeStoreError::Io(io::Error::last_os_error()));
    }
    let moved = unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if moved == 0 {
        return Err(NativeStoreError::Indeterminate(io::Error::last_os_error()));
    }
    temporary.disable_cleanup(true);
    Ok(())
}

#[cfg(unix)]
fn sync_parent(parent: &Path) -> io::Result<()> {
    File::open(parent)?.sync_all()
}

#[cfg(not(any(unix, windows)))]
fn sync_parent(_parent: &Path) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "parent-directory durability confirmation is unavailable on this host",
    ))
}
