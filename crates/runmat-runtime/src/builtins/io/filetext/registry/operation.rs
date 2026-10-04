use super::*;
use std::io::{Read, Seek, SeekFrom, Write};

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

#[derive(Debug)]
struct SharedResourceCause(Arc<io::Error>);

impl std::fmt::Display for SharedResourceCause {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "filesystem resource authority invalidated: {}",
            self.0
        )
    }
}

impl std::error::Error for SharedResourceCause {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.0.as_ref())
    }
}

#[derive(Clone, Debug)]
pub(crate) enum FileOperationError {
    InvalidIdentifier,
    ForeignSession,
    ResourceRevoked(Arc<io::Error>),
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
            Self::ForeignSession => io::Error::new(
                io::ErrorKind::PermissionDenied,
                "file identifier belongs to another or expired session",
            ),
            Self::ResourceRevoked(source) => {
                io::Error::new(io::ErrorKind::PermissionDenied, SharedResourceCause(source))
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
    issuer: FileIssuer,
    filesystem: Option<Arc<FileFilesystem>>,
}

impl FileOperation {
    pub(crate) fn file_mut(&mut self) -> &mut Self {
        self
    }

    pub(crate) fn restore_cursor_on_drop(&mut self, position: u64) {
        self.rollback_cursor_to = Some(position);
    }

    pub(crate) fn commit_cursor(&mut self) {
        self.rollback_cursor_to = None;
    }

    pub(crate) fn check_authority(&self, runtime: Option<&RuntimeContext>) -> io::Result<()> {
        if !self.issuer.admits(runtime) {
            return Err(FileOperationError::ForeignSession.into_io_error());
        }
        if let Some(filesystem) = &self.filesystem {
            filesystem.check()?;
        }
        Ok(())
    }

    fn check_live(&self) -> io::Result<()> {
        if !self.issuer.live() {
            return Err(FileOperationError::ForeignSession.into_io_error());
        }
        if let Some(filesystem) = &self.filesystem {
            filesystem.check()?;
        }
        Ok(())
    }
}

impl Read for FileOperation {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.check_live()?;
        self.file
            .as_mut()
            .expect("operation retains file")
            .read(buffer)
    }
}

impl Write for FileOperation {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.check_live()?;
        self.file
            .as_mut()
            .expect("operation retains file")
            .write(buffer)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.check_live()?;
        self.file.as_mut().expect("operation retains file").flush()
    }
}

impl Seek for FileOperation {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.check_live()?;
        self.file
            .as_mut()
            .expect("operation retains file")
            .seek(position)
    }
}

impl Drop for FileOperation {
    fn drop(&mut self) {
        let restore_failure = if !self.issuer.live() {
            None
        } else if let Some(position) = self.rollback_cursor_to {
            match self
                .filesystem
                .as_ref()
                .map(|filesystem| filesystem.check())
            {
                Some(Err(error)) => Some(error),
                _ => self
                    .file
                    .as_mut()
                    .and_then(|file| file.seek(SeekFrom::Start(position)).err()),
            }
        } else {
            None
        };
        let mut registry = REGISTRY.lock().unwrap_or_else(|poison| poison.into_inner());
        let retired = registry
            .entries
            .get(&self.fid)
            .is_some_and(|entry| entry.retired);
        if retired {
            let removed = registry.entries.remove(&self.fid);
            drop(registry);
            drop(removed);
            return;
        }
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

#[cfg(test)]
pub(crate) fn acquire_file(fid: i32) -> Result<FileOperation, FileOperationError> {
    acquire_file_in(fid, None)
}

pub(crate) fn acquire_file_in(
    fid: i32,
    runtime: Option<&RuntimeContext>,
) -> Result<FileOperation, FileOperationError> {
    let registry = REGISTRY.lock().map_err(|_| FileOperationError::Poisoned)?;
    let entry = registry
        .entries
        .get(&fid)
        .ok_or(FileOperationError::InvalidIdentifier)?;
    if !entry_visible(entry, runtime) {
        return Err(FileOperationError::ForeignSession);
    }
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
    let issuer = entry.issuer.clone();
    let filesystem = entry.filesystem.clone();
    drop(slot);
    drop(registry);
    let operation = FileOperation {
        fid,
        handle,
        file: Some(file),
        rollback_cursor_to: None,
        info,
        issuer,
        filesystem,
    };
    if !operation.issuer.admits(runtime) {
        return Err(FileOperationError::ForeignSession);
    }
    if let Some(filesystem) = &operation.filesystem {
        filesystem
            .check()
            .map_err(|source| FileOperationError::ResourceRevoked(Arc::new(source)))?;
    }
    Ok(operation)
}

#[cfg(test)]
#[path = "operation/tests.rs"]
mod tests;
