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
    text_input: Option<super::super::encoding::TextInputBox>,
    rollback_cursor_to: Option<u64>,
    pub(crate) info: FileInfo,
    issuer: FileIssuer,
    filesystem: Option<Arc<FileFilesystem>>,
}

impl FileOperation {
    pub(crate) fn prepare_encoded_input(
        &mut self,
        runtime: Option<&RuntimeContext>,
    ) -> crate::BuiltinResult<()> {
        self.check_authority(runtime).map_err(|source| {
            crate::build_runtime_error("fread: file authority rejected encoded input")
                .with_builtin("fread")
                .with_source(source)
                .build()
        })?;
        if self.text_input.is_none() {
            let encoding = super::super::encoding::TextEncoding::parse(&self.info.encoding)
                .map_err(|detail| {
                    crate::build_runtime_error(format!("fread: {detail}"))
                        .with_builtin("fread")
                        .build()
                })?;
            let at_start = self
                .file
                .as_mut()
                .expect("operation retains file")
                .stream_position()
                .map_err(|source| {
                    crate::build_runtime_error("fread: cannot inspect encoded source position")
                        .with_builtin("fread")
                        .with_source(source)
                        .build()
                })?
                == 0;
            self.text_input = Some(
                super::super::encoding::EncodedInput::allocate(runtime, encoding, at_start)
                    .map_err(|source| {
                        crate::build_runtime_error("fread: encoded input control admission failed")
                            .with_builtin("fread")
                            .with_source(source)
                            .build()
                    })?,
            );
        }
        Ok(())
    }

    pub(crate) fn encoded_input_step(
        &mut self,
    ) -> io::Result<super::super::encoding::EncodedInputStep> {
        self.check_live()?;
        let mut input = self.text_input.take().expect("encoded input was prepared");
        let result = input.step(|buffer| {
            self.file
                .as_mut()
                .expect("operation retains file")
                .read(buffer)
        });
        self.text_input = Some(input);
        result
    }

    pub(crate) fn skip_encoded_bytes(&mut self, bytes: i64) -> io::Result<u64> {
        self.check_live()?;
        if bytes == 0 {
            return self.stream_position();
        }
        let result = self
            .file
            .as_mut()
            .expect("operation retains file")
            .seek(SeekFrom::Current(bytes));
        match &result {
            Ok(position) => {
                if let Some(input) = &mut self.text_input {
                    input.reset(*position == 0);
                }
            }
            Err(_) => self.text_input = None,
        }
        result
    }

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
        if !buffer.is_empty() {
            self.text_input = None;
        }
        self.file
            .as_mut()
            .expect("operation retains file")
            .read(buffer)
    }
}

impl Write for FileOperation {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.check_live()?;
        if !buffer.is_empty() {
            self.text_input = None;
        }
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
    fn stream_position(&mut self) -> io::Result<u64> {
        self.check_live()?;
        self.file
            .as_mut()
            .expect("operation retains file")
            .stream_position()
    }

    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.check_live()?;
        self.text_input = None;
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
            entry.text_input = if self.rollback_cursor_to.is_none() {
                self.text_input.take()
            } else {
                None
            };
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
    let mut registry = REGISTRY.lock().map_err(|_| FileOperationError::Poisoned)?;
    let entry = registry
        .entries
        .get_mut(&fid)
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
    let text_input = entry.text_input.take();
    drop(slot);
    drop(registry);
    let operation = FileOperation {
        fid,
        handle,
        file: Some(file),
        text_input,
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
