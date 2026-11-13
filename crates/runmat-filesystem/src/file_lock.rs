//! A native single-writer lease held by the kernel, independent of PID metadata.
use std::fs::{File, OpenOptions, TryLockError};
use std::io;
use std::path::Path;

/// Keep the same lock-file inode across owners. Never unlink or atomically replace
/// that path: a contender could otherwise lock a different inode concurrently.
/// The caller owns creation/security of its containing directory.
/// This lease must live as long as any state mutation or provider dispatch it guards.
pub struct ExclusiveFileLock {
    file: File,
}

impl ExclusiveFileLock {
    /// A contended lock returns `None`; unsupported locking or I/O returns an error.
    /// Existing file contents do not establish ownership and are not modified.
    pub fn try_acquire(path: &Path) -> io::Result<Option<Self>> {
        if std::fs::symlink_metadata(path).is_ok_and(|metadata| !metadata.file_type().is_file()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "authority lock must be a regular file",
            ));
        }
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true).truncate(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options.mode(0o600);
        }
        let file = options.open(path)?;
        if !file.metadata()?.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "authority lock must be a regular file",
            ));
        }
        match file.try_lock() {
            Ok(()) => Ok(Some(Self { file })),
            Err(TryLockError::WouldBlock) => Ok(None),
            Err(TryLockError::Error(error)) => Err(error),
        }
    }
}

impl Drop for ExclusiveFileLock {
    fn drop(&mut self) {
        // Closing the private file also releases the lock if explicit unlock fails.
        let _ = self.file.unlock();
    }
}

#[cfg(test)]
mod tests;
