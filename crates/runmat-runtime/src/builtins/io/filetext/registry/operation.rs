use super::*;
use std::io::{Seek, SeekFrom};

#[derive(Debug)]
struct SharedRestoreCause(Arc<io::Error>);

impl std::fmt::Display for SharedRestoreCause {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "file cursor restoration failed: {}", self.0)
    }
}

impl std::error::Error for SharedRestoreCause {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.0.as_ref())
    }
}

#[derive(Clone, Debug)]
pub(crate) enum FileOperationError {
    InvalidIdentifier,
    Busy,
    Poisoned,
    RestoreFailed(Arc<io::Error>),
}

impl FileOperationError {
    pub(crate) fn into_io_error(self) -> io::Error {
        match self {
            Self::InvalidIdentifier => {
                io::Error::new(io::ErrorKind::NotFound, "file identifier is not registered")
            }
            Self::Busy => io::Error::new(
                io::ErrorKind::WouldBlock,
                "file identifier is in another operation",
            ),
            Self::Poisoned => io::Error::other("registered file handle is poisoned"),
            Self::RestoreFailed(source) => {
                io::Error::new(io::ErrorKind::Other, SharedRestoreCause(source))
            }
        }
    }
}

/// Exact registered file checked out for one operation. No registry or handle
/// mutex is held during filesystem callbacks, suspension, or CPU parsing.
pub(crate) struct FileOperation {
    fid: i32,
    handle: SharedFileHandle,
    file: Option<File>,
    rollback_cursor_to: Option<u64>,
    pub(crate) info: FileInfo,
}

impl FileOperation {
    pub(crate) fn file_mut(&mut self) -> &mut File {
        self.file
            .as_mut()
            .expect("a live operation retains its file")
    }

    pub(crate) fn restore_cursor_on_drop(&mut self, position: u64) {
        self.rollback_cursor_to = Some(position);
    }

    pub(crate) fn commit_cursor(&mut self) {
        self.rollback_cursor_to = None;
    }
}

impl Drop for FileOperation {
    fn drop(&mut self) {
        let restore_failure = match (self.rollback_cursor_to, self.file.as_mut()) {
            (Some(position), Some(file)) => file.seek(SeekFrom::Start(position)).err(),
            _ => None,
        };
        let mut registry = REGISTRY.lock().unwrap_or_else(|poison| poison.into_inner());
        let current = registry.entries.get_mut(&self.fid).filter(|entry| {
            entry
                .file_handle()
                .is_some_and(|handle| Arc::ptr_eq(&handle, &self.handle))
        });
        if let Some(entry) = current {
            if let Some(error) = restore_failure {
                entry.restore_failure = Some(Arc::new(error));
            }
            let mut slot = self
                .handle
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if slot.is_none() {
                *slot = self.file.take();
            }
        }
    }
}

pub(crate) fn acquire_file(fid: i32) -> Result<FileOperation, FileOperationError> {
    let registry = REGISTRY.lock().map_err(|_| FileOperationError::Poisoned)?;
    let entry = registry
        .entries
        .get(&fid)
        .ok_or(FileOperationError::InvalidIdentifier)?;
    if let Some(source) = entry.restore_failure.as_ref() {
        return Err(FileOperationError::RestoreFailed(Arc::clone(source)));
    }
    let handle = entry
        .file_handle()
        .ok_or(FileOperationError::InvalidIdentifier)?;
    let mut slot = handle.try_lock().map_err(|error| match error {
        std::sync::TryLockError::WouldBlock => FileOperationError::Busy,
        std::sync::TryLockError::Poisoned(_) => FileOperationError::Poisoned,
    })?;
    let file = slot.take().ok_or(FileOperationError::Busy)?;
    let info = entry.info();
    drop(slot);
    drop(registry);
    Ok(FileOperation {
        fid,
        handle,
        file: Some(file),
        rollback_cursor_to: None,
        info,
    })
}

#[cfg(test)]
#[path = "operation/tests.rs"]
mod tests;
