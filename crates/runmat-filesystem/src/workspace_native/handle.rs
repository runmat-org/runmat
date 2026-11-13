use super::*;
use std::io::{Read, Seek, SeekFrom, Write};

/// File handles retain normal inode semantics. Writes into an inode superseded
/// by a committed publication are rejected rather than acknowledged and lost.
pub(crate) struct WritableHandle {
    pub file: fs::File,
    pub root: PathBuf,
    pub path: PathBuf,
}

impl WritableHandle {
    fn validate(&self) -> io::Result<Option<Guard>> {
        let guard = guard(&self.root, true)?;
        protected(&self.root, &self.path)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let current = fs::metadata(&self.path)?;
            let opened = self.file.metadata()?;
            if current.dev() != opened.dev() || current.ino() != opened.ino() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "file was replaced after this handle opened; reopen it before writing",
                ));
            }
        }
        #[cfg(windows)]
        {
            use crate::native::windows_identity::{self, Kind};
            let current = windows_identity::open(&self.path, Kind::File)?;
            if !windows_identity::same(&self.file, &current)? {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "file was replaced after this handle opened; reopen it before writing",
                ));
            }
        }
        Ok(guard)
    }
}

impl Read for WritableHandle {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        self.file.read(bytes)
    }
}
impl Seek for WritableHandle {
    fn seek(&mut self, offset: SeekFrom) -> io::Result<u64> {
        self.file.seek(offset)
    }
}
impl Write for WritableHandle {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let _guard = self.validate()?;
        self.file.write(bytes)
    }
    fn flush(&mut self) -> io::Result<()> {
        let _guard = self.validate()?;
        self.file.flush()
    }
}
#[async_trait::async_trait(?Send)]
impl crate::FileHandle for WritableHandle {
    async fn metadata_async(&self) -> io::Result<crate::FsMetadata> {
        Ok(self.file.metadata()?.into())
    }
    async fn sync_all_async(&mut self) -> io::Result<()> {
        let _guard = self.validate()?;
        self.file.sync_all()
    }
}
